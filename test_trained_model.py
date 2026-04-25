#!/usr/bin/env python3
"""
Quick test script for your trained GRPO model.
Usage: python test_trained_model.py
"""

import os
import sys
import torch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.core import AdaptiveOSEnv
from env.text_wrapper import observation_to_text, parse_llm_response, SYSTEM_PROMPT

TRAINED_MODEL_PATH = "./macos-llm-clean"  # Your trained model directory
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_trained_model():
    """Load base model + trained LoRA adapter"""
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Loading adapter from: {TRAINED_MODEL_PATH}")
    
    try:
        # Try with transformers + PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
        model.eval()
        
        print("✅ Model loaded successfully (transformers + PEFT)")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have extracted the zip file:")
        print(f"  unzip ~/Downloads/macos-trained.zip -d ./trained_model")
        sys.exit(1)


def test_model(model, tokenizer, task="medium", num_steps=5):
    """Run a quick test episode"""
    print(f"\n{'='*60}")
    print(f"Testing trained model on {task.upper()} difficulty")
    print(f"{'='*60}\n")
    
    env = AdaptiveOSEnv(task=task)
    obs = env.reset()
    
    total_reward = 0
    
    for step in range(num_steps):
        # Convert observation to text
        obs_text = observation_to_text(obs)
        
        # Create messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]
        
        # Generate response
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Parse action
        action = parse_llm_response(response)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}:")
        print(f"  Action: {action.get('action', '?'):12s} PID: {action.get('target_pid', 'N/A'):>4s}")
        print(f"  Reward: {reward:+.3f}")
        print(f"  Reasoning: {response[:80]}...")
        print()
        
        if done or truncated:
            break
    
    print(f"{'='*60}")
    print(f"Episode Complete!")
    print(f"  Total Steps: {step+1}")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Final Grade: {info.get('grade', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n🚀 Testing GRPO-trained MACOS model\n")
    
    # Load model
    model, tokenizer = load_trained_model()
    
    # Run test
    test_model(model, tokenizer, task="medium", num_steps=5) 
    
    print("✅ Test complete! Your trained model is working.\n")
    print("Next steps:")
    print("  - Run full evaluation: python llm_inference.py --model ./trained_model --mode demo")
    print("  - Compare with heuristic: python llm_inference.py --model ./trained_model --mode benchmark")
    print("  - Adversarial test: python llm_inference.py --model ./trained_model --mode whatif")
