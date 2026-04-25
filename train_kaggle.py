# MACOS GRPO Training - Kaggle Notebook Script
#
# This script is designed to run in Kaggle Notebooks with T4 GPU.
# 
# Setup Instructions:
#   1. Create a new Kaggle Notebook
#   2. Set runtime to GPU (T4)
#   3. Add your repo as a Data Source (or git clone it)
#   4. Copy-paste this entire script into a cell
#   5. Run the cell
#
# Quick start:
#   python train_kaggle.py --epochs 2 --max-samples 500
#
# For interactive Kaggle Notebook, use the code sections below.

import os
import re
import json
import argparse
import random
import sys
import numpy as np

# ============================================================
# KAGGLE SETUP
# ============================================================
KAGGLE_OUTPUT_DIR = "/kaggle/working/macos-llm-clean"
os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)

# Add repo to path
sys.path.insert(0, '.')

print(f"[Kaggle] Output directory: {KAGGLE_OUTPUT_DIR}")
print(f"[Kaggle] Available VRAM: Checking...")

import torch
print(f"[Kaggle] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Kaggle] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Kaggle] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ============================================================
# IMPORTS
# ============================================================
from env.text_wrapper import (
    SYSTEM_PROMPT,
    observation_to_text,
    parse_llm_response,
    score_action_in_context,
    generate_training_dataset,
)

# ============================================================
# REWARD FUNCTIONS (Verifiers for GRPO)
# Each takes (completions, **kwargs) -> list[float]
# ============================================================

VALID_ACTIONS = {"SCHEDULE", "KILL", "PRIORITIZE", "THROTTLE", "DELAY", "REALLOCATE"}


def format_reward(completions, **kwargs) -> list[float]:
    """
    Reward for correct output formatting.
    Checks for required XML tags: <action>, <target_pid>, <reason>.
    Bonus for <think> tags (reasoning trace).
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0

        # Required: <action> tag
        if re.search(r'<action>\s*\w+\s*</action>', text, re.IGNORECASE):
            score += 0.4

        # Required: <target_pid> tag
        if re.search(r'<target_pid>\s*\d+\s*</target_pid>', text, re.IGNORECASE):
            score += 0.3

        # Required: <reason> tag
        if re.search(r'<reason>.*?</reason>', text, re.IGNORECASE | re.DOTALL):
            score += 0.15

        # Bonus: <think> reasoning block
        if re.search(r'<think>.*?</think>', text, re.IGNORECASE | re.DOTALL):
            score += 0.15

        rewards.append(score)
    return rewards


def valid_action_reward(completions, **kwargs) -> list[float]:
    """
    Reward for choosing a valid action type and a valid target PID.
    """
    env_states = kwargs.get("env_state_json", [])
    rewards = []

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0

        # Parse action type
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE)
        if action_match:
            action_type = action_match.group(1).upper().strip()
            if action_type in VALID_ACTIONS:
                score += 0.5

                # Parse target PID
                target_match = re.search(r'<target_pid>\s*(\d+)\s*</target_pid>', text, re.IGNORECASE)
                if action_type == "SCHEDULE":
                    # SCHEDULE doesn't need a target, but having pid=0 is fine
                    score += 0.3
                elif target_match:
                    target_pid = int(target_match.group(1))
                    score += 0.3

        rewards.append(score)
    return rewards


def contextual_reward(completions, **kwargs) -> list[float]:
    """
    Core reward: Is the action appropriate for the current system state?
    This is the most important signal for the policy.
    """
    env_states = kwargs.get("env_state_json", [])
    tasks = kwargs.get("task", [])
    rewards = []

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0

        if i < len(env_states) and i < len(tasks):
            try:
                state = json.loads(env_states[i])
                task = tasks[i]
                
                # Use score_action_in_context from text_wrapper
                action_score = score_action_in_context(state, text, task)
                score = action_score  # Already in [0, 1]
            except:
                score = 0.3  # Partial credit for parsing attempt

        rewards.append(score)
    return rewards


def reasoning_reward(completions, **kwargs) -> list[float]:
    """
    Reward for quality of reasoning in <think> block.
    Checks for:
    - Presence and length of reasoning
    - Mention of system state factors
    - Explicit decision logic
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0

        # Extract <think> block
        think_match = re.search(r'<think>(.*?)</think>', text, re.IGNORECASE | re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
            score += 0.25  # Bonus for having thoughts at all

            # Check for state analysis (mentions of system metrics)
            state_keywords = ["cpu", "memory", "processes", "load", "queue", "utilization"]
            if any(kw in think_text.lower() for kw in state_keywords):
                score += 0.25

            # Check for decision logic
            logic_keywords = ["because", "therefore", "since", "if", "then", "should"]
            if any(kw in think_text.lower() for kw in logic_keywords):
                score += 0.25

            # Length bonus (minimum 50 chars for "real" thinking)
            if len(think_text.strip()) >= 50:
                score += 0.25

        rewards.append(min(score, 1.0))
    return rewards


# ============================================================
# TRAINING
# ============================================================

def train_kaggle(
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct",
    epochs: int = 2,
    max_samples: int = 500,
    batch_size: int = 8,
    lr: float = 1e-4,
    resume_from: str = None,
):
    """
    Main training function optimized for Kaggle.
    
    Args:
        model_name: HuggingFace model ID (Unsloth format)
        epochs: Number of training epochs
        max_samples: Maximum dataset size (Kaggle constraint)
        batch_size: Batch size for training
        lr: Learning rate
        resume_from: Optional checkpoint to resume from
    """
    
    print("\n" + "="*60)
    print("MACOS GRPO Training - Kaggle Edition")
    print("="*60)
    
    # ============================================================
    # Step 1: Load model with Unsloth (4-bit quantization)
    # ============================================================
    print("\n[1/5] Loading model with Unsloth...")
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    
    # Add LoRA adapters for efficient training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"✓ Loaded {model_name}")
    print(f"✓ LoRA rank: 16, Alpha: 32")
    
    # ============================================================
    # Step 2: Generate training dataset
    # ============================================================
    print("\n[2/5] Generating training dataset...")
    
    dataset_dict = generate_training_dataset(
        n_episodes_per_task=5,  # Reduced for Kaggle
        tasks=["easy", "medium", "hard"],
        max_samples=max_samples,
    )
    
    # Convert to Hugging Face Dataset format
    from datasets import Dataset
    
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in dataset_dict],
        "env_state_json": [item.get("env_state_json", "{}") for item in dataset_dict],
        "task": [item.get("task", "medium") for item in dataset_dict],
        "ideal_reward": [item.get("ideal_reward", 0.7) for item in dataset_dict],
    })
    
    print(f"✓ Generated {len(dataset)} training samples")
    print(f"✓ Sample keys: {dataset.column_names}")
    
    # ============================================================
    # Step 3: Set up GRPO trainer
    # ============================================================
    print("\n[3/5] Setting up GRPO trainer...")
    from trl import GRPOConfig, GRPOTrainer
    
    training_args = GRPOConfig(
        output_dir=KAGGLE_OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,  # Kaggle supports bfloat16
        gradient_checkpointing=True,
        report_to=[],  # Disable wandb/tensorboard for Kaggle
        seed=42,
    )
    
    # Create reward function that combines all verifiers
    def combined_reward(completions, **kwargs) -> list[float]:
        f_reward = np.array(format_reward(completions, **kwargs))
        v_reward = np.array(valid_action_reward(completions, **kwargs))
        c_reward = np.array(contextual_reward(completions, **kwargs))
        r_reward = np.array(reasoning_reward(completions, **kwargs))
        
        # Weighted combination (contextual is most important)
        combined = 0.2 * f_reward + 0.2 * v_reward + 0.4 * c_reward + 0.2 * r_reward
        return combined.tolist()
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[combined_reward],
        tokenizer=tokenizer,
    )
    
    print("✓ Trainer configured")
    print(f"✓ Output: {KAGGLE_OUTPUT_DIR}")
    
    # ============================================================
    # Step 4: Train
    # ============================================================
    print("\n[4/5] Starting training...")
    print(f"✓ Epochs: {epochs}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Learning rate: {lr}")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Final loss: {train_result.training_loss:.4f}")
    
    # ============================================================
    # Step 5: Save model
    # ============================================================
    print("\n[5/5] Saving model...")
    
    model.save_pretrained(KAGGLE_OUTPUT_DIR)
    tokenizer.save_pretrained(KAGGLE_OUTPUT_DIR)
    
    # Save metadata
    metadata = {
        "model": model_name,
        "epochs": epochs,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_loss": float(train_result.training_loss),
        "training_samples": len(dataset),
    }
    
    with open(os.path.join(KAGGLE_OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved to {KAGGLE_OUTPUT_DIR}")
    print(f"\nTo download: Use Kaggle Notebook 'Output' section")
    
    return trainer, model, tokenizer


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MACOS LLM on Kaggle")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct",
                        help="Model name (Unsloth format)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum dataset size")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    trainer, model, tokenizer = train_kaggle(
        model_name=args.model,
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
    )
    
    print("\n✓ Ready to use model for inference!")
    print(f"✓ Location: {KAGGLE_OUTPUT_DIR}")
