#!/usr/bin/env python3
"""
Data Preparation Script for LLaMA Fine-Tuning
Cameron Pattison - Empirical Study on P4 Claims
December 2025

This script processes the complete_corpus.txt and prepares it for fine-tuning.
It supports two training approaches:
1. Continued Pre-training: Next-token prediction on raw text
2. Instruction Tuning: Synthetic Q&A pairs from the corpus

Usage:
    python prepare_data.py --mode pretraining
    python prepare_data.py --mode instruction
    python prepare_data.py --mode both
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """
    Parse the complete corpus file into individual documents.
    
    Returns a list of dicts with 'title', 'section', and 'content' keys.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    documents = []
    
    # Split by document markers
    doc_pattern = r'-{80}\nDocument \d+: (.+?)\n-{80}\n'
    parts = re.split(doc_pattern, content)
    
    # First part is header, then alternating: title, content, title, content...
    current_section = "Unknown"
    
    # Identify sections
    section_pattern = r'={80}\n(.+?)\n={80}'
    sections = re.findall(section_pattern, content)
    
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            title = parts[i].strip()
            doc_content = parts[i + 1].strip()
            
            # Determine section based on position in file
            if "MEDICAL ETHICS" in content[:content.find(title)].split("=")[-2]:
                section = "Medical Ethics"
            elif "PHILOSOPHY PAPERS" in content[:content.find(title)].split("=")[-2]:
                section = "Philosophy Papers"
            elif "APPLICATION" in content[:content.find(title)].split("=")[-2]:
                section = "Application Materials"
            else:
                section = "Unknown"
            
            # Clean up content
            doc_content = clean_text(doc_content)
            
            if doc_content:  # Only add non-empty documents
                documents.append({
                    "title": title,
                    "section": section,
                    "content": doc_content
                })
    
    return documents


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Removes:
    - Page numbers (standalone numbers)
    - Excessive whitespace
    - Section markers
    """
    # Remove standalone page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int, tokenizer=None) -> List[str]:
    """
    Split text into chunks with overlap.
    
    If tokenizer is provided, chunks by tokens. Otherwise, chunks by characters.
    """
    if tokenizer is None:
        # Character-based chunking (approximate)
        # Assume ~4 characters per token as rough estimate
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + char_chunk_size
            chunk = text[start:end]
            
            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                last_para = chunk.rfind('\n\n')
                if last_para > char_chunk_size * 0.5:
                    chunk = chunk[:last_para]
                    end = start + last_para
                else:
                    # Look for sentence break
                    last_period = max(chunk.rfind('. '), chunk.rfind('.\n'))
                    if last_period > char_chunk_size * 0.5:
                        chunk = chunk[:last_period + 1]
                        end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - char_overlap
        
        return [c for c in chunks if c]
    else:
        # Token-based chunking
        tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        return chunks


def prepare_pretraining_data(documents: List[Dict], config: dict) -> List[Dict]:
    """
    Prepare data for continued pre-training (next-token prediction).
    
    Format: {"text": "<|begin_of_text|>...<|end_of_text|>"}
    """
    chunk_size = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]
    
    training_samples = []
    
    for doc in documents:
        # Create document header
        header = f"# {doc['title']}\n\n"
        full_text = header + doc["content"]
        
        # Chunk the document
        chunks = chunk_text(full_text, chunk_size, chunk_overlap)
        
        for chunk in chunks:
            # Format for LLaMA
            sample = {
                "text": f"<|begin_of_text|>{chunk}<|end_of_text|>",
                "metadata": {
                    "source": doc["title"],
                    "section": doc["section"]
                }
            }
            training_samples.append(sample)
    
    return training_samples


def generate_instruction_pairs(documents: List[Dict], config: dict) -> List[Dict]:
    """
    Generate instruction-response pairs from documents.
    
    This creates synthetic training data by:
    1. Extracting key passages
    2. Creating prompts that would elicit those passages
    3. Formatting as chat conversations
    
    Note: For a production system, you'd want to use an LLM to generate
    more diverse and natural instruction pairs.
    """
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    instruction_samples = []
    
    # Instruction templates for different types of content
    templates = {
        "Philosophy Papers": [
            ("What is your argument about {topic}?", "In my work on {topic}, I argue that "),
            ("Can you explain your view on {topic}?", "My view on {topic} is based on "),
            ("What are the key points in your analysis of {topic}?", "The key points regarding {topic} are: "),
        ],
        "Medical Ethics": [
            ("What is your position on {topic}?", "Regarding {topic}, I maintain that "),
            ("How do you analyze the ethical issues around {topic}?", "The ethical analysis of {topic} requires "),
        ],
        "Application Materials": [
            ("Tell me about your background.", ""),
            ("What are your research interests?", ""),
            ("Describe your academic experience.", ""),
        ],
    }
    
    for doc in documents:
        section = doc["section"]
        content = doc["content"]
        
        # Extract paragraphs as potential responses
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
        
        for para in paragraphs[:10]:  # Limit per document
            # Extract a potential topic from the paragraph
            # Simple heuristic: use first few words or key terms
            words = para.split()[:20]
            topic = ' '.join(words[:5]) + "..."
            
            # Create instruction pair
            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": f"What are your thoughts on: {topic}"},
                    {"role": "assistant", "content": para}
                ],
                "metadata": {
                    "source": doc["title"],
                    "section": doc["section"]
                }
            }
            instruction_samples.append(sample)
    
    return instruction_samples


def save_dataset(samples: List[Dict], output_path: str, format: str = "jsonl"):
    """Save processed samples to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {output_path}")


def split_dataset(samples: List[Dict], train_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train and validation sets."""
    import random
    random.seed(42)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLaMA fine-tuning")
    parser.add_argument(
        "--mode", 
        choices=["pretraining", "instruction", "both"],
        default="both",
        help="Data preparation mode"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to corpus file (overrides config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine corpus path
    corpus_path = args.corpus or config["data"]["corpus_path"]
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(os.path.dirname(args.config), corpus_path)
    
    print(f"Loading corpus from: {corpus_path}")
    
    # Parse corpus
    documents = parse_corpus(corpus_path)
    print(f"Parsed {len(documents)} documents")
    
    for doc in documents:
        print(f"  - [{doc['section']}] {doc['title']}: {len(doc['content'])} chars")
    
    output_dir = Path(config["data"]["processed_dir"])
    train_ratio = config["data"]["train_split"]
    
    # Prepare pre-training data
    if args.mode in ["pretraining", "both"]:
        print("\nPreparing pre-training data...")
        pretraining_samples = prepare_pretraining_data(documents, config)
        
        train, val = split_dataset(pretraining_samples, train_ratio)
        save_dataset(train, output_dir / "pretraining_train.jsonl")
        save_dataset(val, output_dir / "pretraining_val.jsonl")
        
        print(f"Pre-training: {len(train)} train, {len(val)} val samples")
    
    # Prepare instruction data
    if args.mode in ["instruction", "both"]:
        print("\nPreparing instruction tuning data...")
        instruction_samples = generate_instruction_pairs(documents, config)
        
        train, val = split_dataset(instruction_samples, train_ratio)
        save_dataset(train, output_dir / "instruction_train.jsonl")
        save_dataset(val, output_dir / "instruction_val.jsonl")
        
        print(f"Instruction: {len(train)} train, {len(val)} val samples")
    
    print("\nData preparation complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
