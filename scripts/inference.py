#!/usr/bin/env python3
"""
Inference Script for Fine-Tuned LLaMA Model
Cameron Pattison - Empirical Study on P4 Claims
December 2025

This script loads the fine-tuned model and provides an interactive chat interface.

Usage:
    python inference.py --model outputs/cameron-llama-3.2-3b/instruction
    python inference.py --model outputs/cameron-llama-3.2-3b/instruction --interactive
    python inference.py --model outputs/cameron-llama-3.2-3b/instruction --prompt "What is your view on AI consciousness?"
"""

import argparse
import os
from pathlib import Path

import torch
import yaml

# Try Unsloth first for faster inference
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, config: dict, use_unsloth: bool = True):
    """
    Load the fine-tuned model for inference.
    
    Can load either:
    1. LoRA adapters + base model
    2. Merged model
    """
    model_path = Path(model_path)
    merged_path = model_path / "merged"
    
    # Check if merged model exists
    if merged_path.exists():
        print(f"Loading merged model from: {merged_path}")
        model_name = str(merged_path)
        is_merged = True
    else:
        print(f"Loading LoRA adapters from: {model_path}")
        is_merged = False
    
    if UNSLOTH_AVAILABLE and use_unsloth:
        print("Using Unsloth for inference...")
        if is_merged:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(merged_path),
                max_seq_length=config["training"]["max_seq_length"],
                dtype=None,
                load_in_4bit=True,
            )
        else:
            # Load base model then adapters
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config["model"]["name"],
                max_seq_length=config["training"]["max_seq_length"],
                dtype=None,
                load_in_4bit=True,
                token=os.environ.get("HF_TOKEN"),
            )
            model = PeftModel.from_pretrained(model, str(model_path))
        
        # Enable native inference mode
        FastLanguageModel.for_inference(model)
    else:
        print("Using standard transformers for inference...")
        
        if is_merged:
            # Load merged model directly
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(merged_path),
                quantization_config=bnb_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
        else:
            # Load base model + LoRA adapters
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                config["model"]["name"],
                quantization_config=bnb_config,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
            )
            model = PeftModel.from_pretrained(base_model, str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """
    Generate a response from the model.
    """
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Manual formatting for LLaMA
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def interactive_mode(model, tokenizer, system_prompt: str):
    """
    Run an interactive chat session.
    """
    print("\n" + "=" * 60)
    print("Interactive Mode - Cameron's Fine-Tuned LLaMA")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to reset the conversation.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            print("[Conversation cleared]\n")
            continue
        
        print("Cameron: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, user_input, system_prompt
        )
        print(response)
        print()


def run_test_prompts(model, tokenizer, system_prompt: str):
    """
    Run a set of test prompts to evaluate the model.
    """
    test_prompts = [
        "What is your view on AI consciousness?",
        "How do you approach questions of moral status?",
        "Tell me about your background in philosophy.",
        "What is your argument about the isomorphism between human and AI cognition?",
        "What do you think about the P4 proposal for patient preference prediction?",
        "How does your work in Islamic philosophy inform your current research?",
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Prompts")
    print("=" * 60 + "\n")
    
    for prompt in test_prompts:
        print(f"PROMPT: {prompt}")
        print("-" * 40)
        response = generate_response(model, tokenizer, prompt, system_prompt)
        print(f"RESPONSE: {response}")
        print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLaMA")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run (non-interactive)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test prompts"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth even if available"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    system_prompt = config.get("system_prompt", "").strip()
    
    # Load model
    model, tokenizer = load_model(
        args.model, config,
        use_unsloth=not args.no_unsloth
    )
    
    # Run based on mode
    if args.test:
        run_test_prompts(model, tokenizer, system_prompt)
    elif args.interactive:
        interactive_mode(model, tokenizer, system_prompt)
    elif args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt, system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
    else:
        # Default to interactive mode
        interactive_mode(model, tokenizer, system_prompt)


if __name__ == "__main__":
    main()
