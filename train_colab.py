# MACOS GRPO Training - Google Colab Notebook
#
# This script is designed to run in Google Colab with a T4 GPU.
# Upload your project files to Colab or clone from your repo.
#
# Quick start:
#   1. Open Google Colab (colab.research.google.com)
#   2. Set runtime to T4 GPU
#   3. Upload this file or paste into cells
#   4. Run all cells

# ============================================================
# Cell 1: Install dependencies
# ============================================================
# !pip install unsloth trl peft datasets accelerate
# !pip install fastapi uvicorn pydantic==1.10.13 gymnasium numpy

# If using unsloth nightly (recommended for latest fixes):
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ============================================================
# Cell 2: Upload project files
# ============================================================
# Option A: Clone from git
# !git clone <your-repo-url> macos && cd macos

# Option B: Upload zip
# from google.colab import files
# uploaded = files.upload()  # upload your project zip
# !unzip adaptive-os-openenv.zip -d macos && cd macos

# Option C: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/adaptive-os-openenv

# ============================================================
# Cell 3: Verify environment works
# ============================================================
import sys
sys.path.insert(0, '.')

from env.text_wrapper import observation_to_text, SYSTEM_PROMPT, generate_training_dataset
from env.core import AdaptiveOSEnv
from train_grpo import format_reward, valid_action_reward, contextual_reward, reasoning_reward

# Quick test
env = AdaptiveOSEnv(task='hard')
obs = env.reset()
print(observation_to_text(obs))
print("\nEnvironment OK!")

# ============================================================
# Cell 4: Generate training dataset
# ============================================================
print("Generating training dataset...")
dataset = generate_training_dataset(
    n_episodes_per_task=15,   # 15 episodes per difficulty
    tasks=["easy", "medium", "hard"],
    max_samples=1000,         # Cap at 1000 samples
)
print(f"Dataset: {len(dataset)} prompts")
print(f"Sample:\n{dataset[0]['prompt'][1]['content'][:300]}")

# ============================================================
# Cell 5: Load model with Unsloth
# ============================================================
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen2.5-1.5B-Instruct"  # Small but capable
# Alternatives:
# model_name = "unsloth/Qwen2.5-3B-Instruct"  # Stronger, needs more VRAM
# model_name = "unsloth/Llama-3.2-1B-Instruct" # Meta's small model
# model_name = "unsloth/Llama-3.2-3B-Instruct" # Meta's medium model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model_name}")
print(f"Trainable params: {model.print_trainable_parameters()}")

# ============================================================
# Cell 6: Configure and run GRPO training
# ============================================================
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="macos-llm-clean",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_generations=4,               # 4 completions per prompt for GRPO
    max_completion_length=512,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    report_to="none",
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward,           # XML formatting
        valid_action_reward,     # Valid action + PID
        contextual_reward,       # Right action for the situation
        reasoning_reward,        # Quality of thinking
    ],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting GRPO training...")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Generations/prompt: {training_args.num_generations}")
print(f"  Reward functions: format, valid_action, contextual, reasoning")

trainer.train()

# ============================================================
# Cell 7: Save model
# ============================================================
# Save LoRA adapter
trainer.save_model("macos-llm-clean")
tokenizer.save_pretrained("macos-llm-clean")

# Save merged model (for easy inference)
model.save_pretrained_merged(
    "macos-llm-clean/merged",
    tokenizer,
    save_method="merged_16bit",
)

import json
metadata = {
    "model": model_name,
    "epochs": 3,
    "dataset_size": len(dataset),
    "reward_functions": ["format", "valid_action", "contextual", "reasoning"],
    "training_type": "GRPO",
    "stack": "TRL + Unsloth",
}
with open("macos-llm-clean/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model saved to macos-llm-clean/")

# ============================================================
# Cell 8: Quick test - before vs after
# ============================================================
FastLanguageModel.for_inference(model)

from env.text_wrapper import observation_to_text, parse_llm_response
import torch

env = AdaptiveOSEnv(task='hard')
obs = env.reset()

# Step a few times to get interesting state
from env.models import Action
for _ in range(5):
    obs, _, _, _ = env.step(Action(action_type="SCHEDULE"))

obs_text = observation_to_text(obs)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": obs_text},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.3, do_sample=True)

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("=== SYSTEM STATE ===")
print(obs_text)
print("\n=== LLM RESPONSE ===")
print(response)
print("\n=== PARSED ACTION ===")
action = parse_llm_response(response)
print(f"Action: {action.action_type}, Target PID: {action.target_pid}")

# ============================================================
# Cell 9: Full benchmark
# ============================================================
# Uncomment to run full benchmark after training
# from llm_inference import benchmark_mode, load_llm
# model, tokenizer = load_llm("macos-llm-clean")
# results = benchmark_mode(model, tokenizer)

# ============================================================
# Cell 10: Download trained model
# ============================================================
# !zip -r macos-llm-clean.zip macos-llm-clean/
# from google.colab import files
# files.download('macos-llm-clean.zip')

# Or push to Hugging Face Hub:
# model.push_to_hub_merged("your-username/macos-llm-clean", tokenizer, save_method="merged_16bit")
