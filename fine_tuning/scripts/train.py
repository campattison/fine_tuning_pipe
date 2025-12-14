#!/usr/bin/env python3
"""
Training Script for LLaMA Fine-Tuning
Cameron Pattison - Empirical Study on P4 Claims
December 2025

This script fine-tunes LLaMA 3.2-3B-Instruct on personal writings using QLoRA.
Supports both continued pre-training and instruction tuning.

Usage:
    # With Unsloth (recommended - 2x faster)
    python train.py --config config.yaml --stage pretraining
    python train.py --config config.yaml --stage instruction
    
    # Without Unsloth
    python train.py --config config.yaml --stage pretraining --no-unsloth

Environment Variables:
    HF_TOKEN: Your Hugging Face token for model access
    WANDB_API_KEY: (Optional) For logging to Weights & Biases
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Try to import Unsloth for faster training
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available. Using standard transformers training.")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_standard(config: dict):
    """
    Set up model and tokenizer using standard Hugging Face + PEFT.
    
    Uses 4-bit quantization with QLoRA for memory efficiency.
    """
    model_name = config["model"]["name"]
    
    print(f"Loading model: {model_name}")
    
    # Quantization config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["model"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["model"]["use_nested_quant"],
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
        target_modules=config["lora"]["target_modules"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def setup_model_unsloth(config: dict):
    """
    Set up model and tokenizer using Unsloth for faster training.
    
    Provides 2x speedup and 60% memory reduction.
    """
    model_name = config["model"]["name"]
    max_seq_length = config["training"]["max_seq_length"]
    
    print(f"Loading model with Unsloth: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config["model"]["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing="unsloth",  # Optimized checkpointing
        random_state=42,
    )
    
    return model, tokenizer


def load_training_data(config: dict, stage: str, tokenizer):
    """
    Load and format training data based on the training stage.
    """
    data_dir = Path(config["data"]["processed_dir"])
    
    if stage == "pretraining":
        train_file = data_dir / "pretraining_train.jsonl"
        val_file = data_dir / "pretraining_val.jsonl"
        text_field = "text"
    else:  # instruction
        train_file = data_dir / "instruction_train.jsonl"
        val_file = data_dir / "instruction_val.jsonl"
        text_field = None  # Will use messages format
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Run 'python scripts/prepare_data.py --mode {stage}' first."
        )
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    val_dataset = load_dataset("json", data_files=str(val_file), split="train") if val_file.exists() else None
    
    print(f"Loaded {len(train_dataset)} training samples")
    if val_dataset:
        print(f"Loaded {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset, text_field


def format_instruction_sample(sample, tokenizer):
    """
    Format instruction sample using the model's chat template.
    """
    messages = sample["messages"]
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback: manual formatting for LLaMA
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    return {"text": formatted}


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: dict,
    stage: str,
    text_field: str = None,
):
    """
    Create the SFTTrainer with appropriate configuration.
    """
    training_config = config["training"]
    output_dir = Path(config["output"]["checkpoint_dir"]) / stage
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        num_train_epochs=training_config["num_train_epochs"],
        max_steps=training_config["max_steps"],
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        logging_steps=training_config["logging_steps"],
        report_to=training_config["report_to"],
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=training_config["save_steps"] if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        gradient_checkpointing=True,
        group_by_length=True,
    )
    
    # Format function for instruction tuning
    formatting_func = None
    if stage == "instruction":
        formatting_func = lambda sample: format_instruction_sample(sample, tokenizer)
        text_field = "text"
        # Apply formatting to dataset
        train_dataset = train_dataset.map(
            lambda x: format_instruction_sample(x, tokenizer),
            remove_columns=["messages", "metadata"] if "metadata" in train_dataset.column_names else ["messages"]
        )
        if val_dataset:
            val_dataset = val_dataset.map(
                lambda x: format_instruction_sample(x, tokenizer),
                remove_columns=["messages", "metadata"] if "metadata" in val_dataset.column_names else ["messages"]
            )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=training_config["max_seq_length"],
        dataset_text_field=text_field or "text",
        packing=True,  # Pack multiple samples into one sequence for efficiency
    )
    
    return trainer


def save_model(model, tokenizer, config: dict, stage: str):
    """
    Save the fine-tuned model.
    """
    output_dir = Path(config["output"]["final_model_dir"]) / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"Model saved to: {output_dir}")
    
    # Optionally merge and save full model
    if config["output"].get("merge_and_save", False):
        print("Merging LoRA weights with base model...")
        if UNSLOTH_AVAILABLE:
            merged_dir = output_dir / "merged"
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"Merged model saved to: {merged_dir}")
        else:
            # Standard PEFT merge
            from peft import PeftModel
            merged_model = model.merge_and_unload()
            merged_dir = output_dir / "merged"
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"Merged model saved to: {merged_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on personal writings")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--stage",
        choices=["pretraining", "instruction"],
        required=True,
        help="Training stage"
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth even if available"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    args = parser.parse_args()
    
    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. You may need to set it for model access.")
        print("Run: export HF_TOKEN='your_token_here'")
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 60)
    print(f"LLaMA Fine-Tuning - Stage: {args.stage.upper()}")
    print("=" * 60)
    
    # Set up model
    use_unsloth = UNSLOTH_AVAILABLE and not args.no_unsloth
    if use_unsloth:
        print("Using Unsloth for optimized training")
        model, tokenizer = setup_model_unsloth(config)
    else:
        print("Using standard Hugging Face training")
        model, tokenizer = setup_model_standard(config)
    
    # Load data
    train_dataset, val_dataset, text_field = load_training_data(config, args.stage, tokenizer)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        stage=args.stage,
        text_field=text_field,
    )
    
    # Train
    print("\nStarting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save model
    save_model(model, tokenizer, config, args.stage)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Print next steps
    if args.stage == "pretraining":
        print("\nNext step: Run instruction tuning")
        print(f"  python scripts/train.py --config {args.config} --stage instruction")
    else:
        print("\nNext step: Test the model")
        print(f"  python scripts/inference.py --model outputs/cameron-llama-3.2-3b/instruction")


if __name__ == "__main__":
    main()
