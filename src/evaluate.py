# src/evaluate.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data_prep import prepare_dataset, format_instruction
from configs.config import get_config


def load_model(config, adapter_path: str = None):
    """Load base model + optional LoRA adapter for evaluation."""
    adapter_path = adapter_path or config.training.output_dir

    print(f"Loading base model: {config.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=torch.float16 if config.training.fp16 else torch.float32,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, max_samples: int = 100):
    """Compute perplexity on an evaluation dataset split."""
    total_loss = 0.0
    count = 0
    device = next(model.parameters()).device

    samples = dataset.select(range(min(max_samples, len(dataset))))

    with torch.no_grad():
        for sample in samples:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            if not torch.isnan(outputs.loss):
                total_loss += outputs.loss.item()
                count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def generate_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 256):
    """Generate a model response for a given instruction."""
    example = {"instruction": instruction, "input": input_text, "output": ""}
    formatted = format_instruction(example)
    prompt = formatted["text"]

    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (strip the prompt)
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def main():
    config = get_config()

    model, tokenizer = load_model(config)

    print("\nPreparing test dataset...")
    dataset, _ = prepare_dataset(config)
    test_dataset = dataset["test"]

    # ── Perplexity ──────────────────────────────────────────────────────
    print("\nComputing perplexity on 100 test samples...")
    ppl, avg_loss = compute_perplexity(model, tokenizer, test_dataset, max_samples=100)
    print(f"  Average loss : {avg_loss:.4f}")
    print(f"  Perplexity   : {ppl:.2f}")

    # ── Sample generations ───────────────────────────────────────────────
    SAMPLE_INSTRUCTIONS = [
        ("Give three tips for staying healthy.", ""),
        ("Write a haiku about the ocean.", ""),
        ("Summarize the following text.", "The Great Wall of China is one of the greatest wonders of the ancient world."),
    ]

    print("\n" + "=" * 60)
    print("Sample Generations")
    print("=" * 60)
    for instruction, inp in SAMPLE_INSTRUCTIONS:
        response = generate_response(model, tokenizer, instruction, inp)
        print(f"\n📝 Instruction: {instruction}")
        if inp:
            print(f"   Input:       {inp}")
        print(f"🤖 Response:  {response}")
        print("-" * 60)


if __name__ == "__main__":
    main()
