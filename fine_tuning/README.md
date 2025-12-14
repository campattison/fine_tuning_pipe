# Fine-Tuning LLaMA 3.2-3B-Instruct on Personal Writings

**Cameron Pattison | Empirical Study on P4 Claims | December 2025**

This project fine-tunes Meta's LLaMA 3.2-3B-Instruct model on a corpus of philosophical writings to create a personalized language model. This serves as an empirical component of the research outlined in `p4_experiment.tex`, testing claims about fine-tuning approaches for patient preference prediction.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Option A: Google Colab (Recommended)](#option-a-google-colab-recommended)
  - [Option B: Local GPU](#option-b-local-gpu)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Inference](#3-inference)
- [Configuration](#configuration)
- [Empirical Study Notes](#empirical-study-notes)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Goals

1. **Empirical Testing**: Generate data to evaluate claims made in the P4 paper about fine-tuning approaches
2. **Style Capture**: Create a model that captures the author's writing style and reasoning patterns
3. **Methodological Documentation**: Maintain clear records of the fine-tuning process

### Training Approach

We use a **two-stage training pipeline**:

1. **Continued Pre-training**: Train the model to predict next tokens on raw corpus text, capturing writing style and domain knowledge
2. **Instruction Tuning**: Fine-tune on question-answer pairs for controllable interaction

This hybrid approach allows observation of:
- "Replica" effects from next-token prediction (cf. Section 3.2.2 of P4 paper)
- Controlled interaction patterns from instruction tuning
- Potential tension between captured patterns and steerability

### Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Base Model | LLaMA 3.2-3B-Instruct | Pre-trained foundation |
| Quantization | QLoRA (4-bit NF4) | Memory efficiency |
| PEFT | LoRA adapters | Parameter-efficient fine-tuning |
| Training | SFTTrainer (TRL) | Supervised fine-tuning |
| Optimization | Unsloth | 2x faster, 60% less memory |

---

## Project Structure

```
fine_tuning/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Training configuration
├── scripts/
│   ├── prepare_data.py      # Data preprocessing
│   ├── train.py             # Training script
│   └── inference.py         # Model inference
├── data/
│   └── processed/           # Formatted training data
│       ├── pretraining_train.jsonl
│       ├── pretraining_val.jsonl
│       ├── instruction_train.jsonl
│       └── instruction_val.jsonl
└── outputs/
    ├── checkpoints/         # Training checkpoints
    │   ├── pretraining/
    │   └── instruction/
    └── cameron-llama-3.2-3b/  # Final model
        ├── pretraining/
        └── instruction/
```

---

## Setup

### Prerequisites

- Python 3.10+
- Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`
- GPU with 16GB+ VRAM (or Google Colab)

### Option A: Google Colab (Recommended)

The easiest way to run this project is via Google Colab, which provides free GPU access.

1. **Create a new Colab notebook**

2. **Install dependencies**:
```python
# Install Unsloth for optimal performance
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# Or standard installation (slower but more compatible)
# !pip install torch transformers datasets peft trl bitsandbytes accelerate
```

3. **Mount Google Drive and upload files**:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy project files to Colab
!cp -r "/content/drive/MyDrive/fine_tuning" /content/
```

4. **Set Hugging Face token**:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"

# Or login interactively
from huggingface_hub import login
login()
```

### Option B: Local GPU

For local development with a CUDA-capable GPU:

1. **Create virtual environment**:
```bash
cd fine_tuning
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

2. **Install dependencies**:
```bash
# Standard installation
pip install -r requirements.txt

# Optional: Install Unsloth for faster training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

3. **Set environment variables**:
```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # Optional, for logging
```

---

## Usage

### 1. Data Preparation

Process the corpus into training-ready formats:

```bash
cd fine_tuning

# Prepare both pre-training and instruction data
python scripts/prepare_data.py --mode both

# Or prepare individually
python scripts/prepare_data.py --mode pretraining
python scripts/prepare_data.py --mode instruction
```

This will:
- Parse `complete_corpus.txt` into individual documents
- Clean and normalize text
- Chunk documents for training
- Create train/validation splits
- Save to `data/processed/`

**Output**:
```
Parsed 25 documents
  - [Medical Ethics] p4_experiment.txt: 45234 chars
  - [Philosophy Papers] Synthese.txt: 52341 chars
  ...
Pre-training: 127 train, 14 val samples
Instruction: 198 train, 22 val samples
```

### 2. Training

#### Stage 1: Continued Pre-training

Train the model to predict next tokens on your raw writings:

```bash
python scripts/train.py --config config.yaml --stage pretraining
```

**Expected duration**: ~30-60 minutes on a T4 GPU, ~10-20 minutes on an A100

#### Stage 2: Instruction Tuning

Fine-tune on question-answer pairs:

```bash
python scripts/train.py --config config.yaml --stage instruction
```

**Expected duration**: ~20-40 minutes on a T4 GPU

#### Resume from Checkpoint

If training is interrupted:

```bash
python scripts/train.py --config config.yaml --stage pretraining \
    --resume outputs/checkpoints/pretraining/checkpoint-100
```

### 3. Inference

Test the fine-tuned model:

```bash
# Interactive chat
python scripts/inference.py --model outputs/cameron-llama-3.2-3b/instruction --interactive

# Single prompt
python scripts/inference.py --model outputs/cameron-llama-3.2-3b/instruction \
    --prompt "What is your view on AI consciousness?"

# Run test prompts
python scripts/inference.py --model outputs/cameron-llama-3.2-3b/instruction --test
```

**Example output**:
```
PROMPT: What is your view on AI consciousness?
----------------------------------------
RESPONSE: In my work, I argue that consciousness may be the more salient 
determinant of moral standing than rational capacity alone. While I've 
examined how advanced transformer architectures mirror aspects of human 
inferential reasoning, the question of whether these systems possess 
genuine phenomenal experience remains open. My position draws on...
```

---

## Configuration

Key settings in `config.yaml`:

### Model Settings

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  load_in_4bit: true  # QLoRA quantization
```

### LoRA Settings

```yaml
lora:
  r: 16           # Rank - higher = more capacity, more memory
  lora_alpha: 32  # Scaling factor
  lora_dropout: 0.05
```

### Training Settings

```yaml
training:
  max_seq_length: 2048
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
  learning_rate: 2.0e-4
  num_train_epochs: 3
```

### Memory Optimization

If running out of memory:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 16
- Reduce `max_seq_length` to 1024
- Ensure `load_in_4bit: true`

---

## Empirical Study Notes

This project is designed to generate empirical data for testing claims in `p4_experiment.tex`. Key observations to document:

### 1. Inextricability (Section 3.1.1)

- [ ] Do model outputs blend personal patterns with population-level regularities?
- [ ] Can you distinguish "Cameron-like" responses from generic philosophical reasoning?
- [ ] Evidence of catastrophic forgetting with aggressive fine-tuning?

### 2. Steerability (Section 3.1.2)

- [ ] Does the model incorporate new context provided at inference time?
- [ ] Can you override fine-tuned patterns with explicit instructions?
- [ ] Comparison with base model's context-following ability?

### 3. Replica Characteristics (Section 3.2.2)

- [ ] Does sentence-completion training produce speech "in your voice"?
- [ ] Phenomenological observations about interacting with the model
- [ ] Differences between pre-training-only vs. instruction-tuned models

### 4. Domination Concerns (Section 3.3)

- [ ] How persuasive/authoritative do the model's outputs feel?
- [ ] Would a surrogate find the outputs too compelling to override?
- [ ] Comparison with simple retrieval of original documents

---

## Troubleshooting

### "CUDA out of memory"

1. Reduce batch size: `per_device_train_batch_size: 1`
2. Increase gradient accumulation: `gradient_accumulation_steps: 16`
3. Reduce sequence length: `max_seq_length: 1024`
4. Ensure 4-bit quantization is enabled

### "Model not found" on Hugging Face

1. Ensure you've accepted the LLaMA license at https://huggingface.co/meta-llama
2. Verify your token: `huggingface-cli whoami`
3. Check token has read access to gated models

### Unsloth Installation Issues

If Unsloth fails to install, use standard training:
```bash
python scripts/train.py --config config.yaml --stage pretraining --no-unsloth
```

### Slow Training

1. Install Unsloth for 2x speedup
2. Use Colab Pro for A100 access
3. Enable `packing: true` in SFTTrainer (already enabled)

---

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- P4 Experiment Paper: `../p4_experiment.tex`

---

## License

This project is for research purposes. Model usage subject to Meta's LLaMA license.

**Author**: Cameron Pattison  
**Date**: December 2025  
**Version**: 1.0.0
