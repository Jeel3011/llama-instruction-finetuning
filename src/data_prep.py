# src/data_prep.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer
from configs.config import get_config


def format_instruction(example):
    """Format an Alpaca example into the instruction-following prompt template."""
    if example.get("input", ""):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n"
        )
    example["text"] = prompt + example["output"]
    return example


def prepare_dataset(config):
    """Load, format, and tokenize the Alpaca dataset.

    Returns:
        tokenized: DatasetDict with 'train' and 'test' splits.
        tokenizer: The tokenizer for the configured model.
    """
    # Load raw dataset
    dataset = load_dataset(config.data.name)

    # Optionally limit dataset size (useful for quick tests)
    if config.data.max_samples:
        dataset = dataset["train"].select(
            range(min(config.data.max_samples, len(dataset["train"])))
        )
    else:
        dataset = dataset["train"]

    # Train / test split
    dataset = dataset.train_test_split(test_size=config.data.test_size, seed=42)

    # Apply instruction formatting
    dataset = dataset.map(format_instruction, desc="Formatting instructions")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.model.max_length,
        )
        # Set labels = input_ids for causal LM training
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    return tokenized, tokenizer