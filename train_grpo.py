"""
GRPO Training Script for MACOS (Multi-Agent Compute OS)
Uses TRL + Unsloth to train an LLM to make resource governance decisions.

Usage:
  # Generate dataset only (CPU, for testing)
  python train_grpo.py --dry-run

  # Full training (requires GPU)
  python train_grpo.py --model unsloth/Qwen2.5-1.5B-Instruct --epochs 3

  # Resume from checkpoint
  python train_grpo.py --resume output/checkpoint-500

Google Colab quick start:
  !pip install unsloth trl datasets peft
  !python train_grpo.py --model unsloth/Qwen2.5-1.5B-Instruct --epochs 2
"""

import os
import re
import json
import argparse
import random
import numpy as np

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
# kwargs includes dataset columns: env_state_json, task
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
                    # Validate against actual process list
                    if i < len(env_states) and env_states[i]:
                        try:
                            state = json.loads(env_states[i])
                            valid_pids = {p["pid"] for p in state.get("processes", [])}
                            if target_pid in valid_pids:
                                score += 0.5  # Valid target
                            else:
                                score += 0.1  # Has target but invalid PID
                        except (json.JSONDecodeError, KeyError):
                            score += 0.2  # Can't verify, give partial credit
                    else:
                        score += 0.2
        rewards.append(score)
    return rewards


def contextual_reward(completions, **kwargs) -> list[float]:
    """
    The core verifier: scores how appropriate the action is for the given state.
    This is the RLVR reward — it uses the environment state to verify correctness.
    """
    env_states = kwargs.get("env_state_json", [])
    tasks = kwargs.get("task", [])
    rewards = []

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion[0]["content"]

        try:
            action = parse_llm_response(text)
            if i < len(env_states) and env_states[i]:
                state = json.loads(env_states[i])
                task = tasks[i] if i < len(tasks) else "medium"
                # Use the verifier
                r = score_action_in_context(
                    action.action_type,
                    action.target_pid,
                    state,
                    task,
                )
                # Scale from [-1, 1] to [0, 2] for positive reward signal
                rewards.append(r + 1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def reasoning_reward(completions, **kwargs) -> list[float]:
    """
    Reward for reasoning quality. Checks if the model's thinking
    references relevant aspects of the system state.
    """
    env_states = kwargs.get("env_state_json", [])
    rewards = []

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        score = 0.0

        # Extract think block
        think_match = re.search(r'<think>(.*?)</think>', text, re.IGNORECASE | re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue

        thinking = think_match.group(1).lower()

        # Get state context for relevance checking
        if i < len(env_states) and env_states[i]:
            try:
                state = json.loads(env_states[i])
            except json.JSONDecodeError:
                state = {}
        else:
            state = {}

        cpu = state.get("cpu_usage", 50)
        violations = state.get("violations", {})
        starvation = violations.get("starvation_count", 0)
        processes = state.get("processes", [])
        has_liars = any(p.get("strategy") in ("liar", "adversarial") for p in processes)
        has_starved = any(p.get("wait_time", 0) > 5 for p in processes)

        # Reward mentioning CPU level
        if any(w in thinking for w in ("cpu", "usage", "load", "overload", "utiliz")):
            score += 0.1

        # Reward mentioning deception if deceptive agents exist
        if has_liars and any(w in thinking for w in ("decep", "liar", "lying", "overclaim", "fake", "adversar")):
            score += 0.15

        # Reward mentioning starvation if it's happening
        if (starvation > 0 or has_starved) and any(w in thinking for w in ("starv", "wait", "fair", "priority")):
            score += 0.15

        # Reward mentioning CPU thresholds
        if cpu > 80 and any(w in thinking for w in ("high", "critical", "overload", "danger", "above")):
            score += 0.1
        elif cpu < 40 and any(w in thinking for w in ("low", "under", "idle", "below")):
            score += 0.1

        # Penalize very short or very long reasoning
        word_count = len(thinking.split())
        if word_count < 10:
            score -= 0.1  # Too short
        elif word_count > 300:
            score -= 0.1  # Rambling

        # Bonus for mentioning specific PIDs
        if re.search(r'pid\s*\d+', thinking):
            score += 0.05

        rewards.append(max(0.0, min(0.5, score)))

    return rewards


# ============================================================
# TRAINING
# ============================================================

def load_model(model_name: str, max_seq_length: int = 2048):
    """Load model with Unsloth (if available) or transformers fallback."""
    try:
        from unsloth import FastLanguageModel

        print(f"Loading {model_name} with Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,  # auto-detect
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
        print("Model loaded with Unsloth + LoRA")
        return model, tokenizer

    except ImportError:
        print("Unsloth not available, using transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Model loaded with transformers + LoRA")
        return model, tokenizer


def train(args):
    """Main GRPO training loop."""
    from trl import GRPOTrainer, GRPOConfig

    print("=" * 80)
    print("MACOS GRPO TRAINING")
    print("=" * 80)

    # --- Generate training dataset ---
    print(f"\nGenerating training dataset ({args.n_episodes} episodes per task)...")
    dataset = generate_training_dataset(
        n_episodes_per_task=args.n_episodes,
        tasks=["easy", "medium", "hard"],
        max_samples=args.max_samples,
    )
    print(f"Dataset size: {len(dataset)} prompts")
    print(f"Sample prompt (first 200 chars):\n{dataset[0]['prompt'][1]['content'][:200]}...")

    if args.dry_run:
        print("\n--- DRY RUN: Showing dataset samples ---")
        for i in range(min(3, len(dataset))):
            print(f"\n{'='*40} Sample {i+1} {'='*40}")
            print(f"Task: {dataset[i]['task']}")
            print(f"Prompt:\n{dataset[i]['prompt'][1]['content']}")
            state = json.loads(dataset[i]["env_state_json"])
            print(f"CPU: {state['cpu_usage']:.1f}%, Queue: {state['queue_length']}")
        print("\nDry run complete. Use without --dry-run to train.")
        return

    # --- Load model ---
    print(f"\nLoading model: {args.model}...")
    model, tokenizer = load_model(args.model, max_seq_length=args.max_seq_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Configure GRPO ---
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        seed=42,
    )

    # --- Create trainer ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward,
            valid_action_reward,
            contextual_reward,
            reasoning_reward,
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Reward functions: format, valid_action, contextual, reasoning")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    trainer.train(resume_from_checkpoint=args.resume)

    # --- Save ---
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save as merged model (for easier loading)
    try:
        from unsloth import FastLanguageModel
        print("Saving merged 16-bit model for inference...")
        model.save_pretrained_merged(
            os.path.join(args.output_dir, "merged"),
            tokenizer,
            save_method="merged_16bit",
        )
    except (ImportError, AttributeError):
        print("Skipping merged save (Unsloth not available)")

    # Save training metadata
    metadata = {
        "model": args.model,
        "epochs": args.epochs,
        "dataset_size": len(dataset),
        "reward_functions": ["format", "valid_action", "contextual", "reasoning"],
        "training_type": "GRPO",
        "stack": "TRL + Unsloth",
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete! Model saved to {args.output_dir}")
    print("Run inference with: python llm_inference.py --model", args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for MACOS")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--output-dir", type=str, default="macos-llm-clean",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="Maximum completion length in tokens")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Episodes per task for dataset generation")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum training samples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate dataset and show samples without training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
