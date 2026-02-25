import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer
from configs.config import get_config
config = get_config()

def format_instruction(example):
    if example["input"]:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
"""

    example["text"] = prompt + example["output"]
    return example


def prepare_dataset(config):
    dataset = load_dataset(config.data.name)

    # Limit dataset to max_samples if specified
    if config.data.max_samples:
        dataset = dataset["train"].select(range(min(config.data.max_samples, len(dataset["train"]))))
    else:
        dataset = dataset["train"]

    dataset = dataset.train_test_split(
        test_size=config.data.test_size,
    )

    dataset = dataset.map(format_instruction)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        use_fast=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.model.max_length
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    tokenized.set_format("torch")

    return tokenized, tokenizer