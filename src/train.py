# src/train.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from src.data_prep import prepare_dataset
from configs.config import get_config


def main():
    config = get_config()

    print("=" * 60)
    print("Llama 3.2 Instruction Fine-Tuning")
    print("=" * 60)
    print(f"Model:      {config.model.name}")
    print(f"Dataset:    {config.data.name}")
    print(f"Max samples: {config.data.max_samples or 'Full dataset'}")
    print(f"Epochs:     {config.training.num_epochs}")
    print(f"LoRA rank:  {config.lora.r}")
    print("=" * 60)

    # ── Prepare dataset ─────────────────────────────────────────────────
    dataset, tokenizer = prepare_dataset(config=config)
    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Test  samples: {len(dataset['test'])}")

    # ── Load base model ─────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=torch.float16 if config.training.fp16 else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False          # Required for gradient checkpointing

    # ── Apply LoRA ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        fp16=config.training.fp16,
        push_to_hub=config.hub.push_to_hub,
        hub_model_id=config.hub.repo_name if config.hub.push_to_hub else None,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving model to {config.training.output_dir}")
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    print("Done! ✓")


if __name__ == "__main__":
    main()