import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForLanguageModeling
from peft import LoraConfig,get_peft_model
from data_prep import prepare_dataset


from configs.config import get_config


def main():
    config = get_config()

    dataset, tokenizer = prepare_dataset(config=config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.float16 if config.training.fp16 else torch.float32,
        device_map="auto",
    )
    
    loraconfig = LoraConfig(
        r = config.lora.r,
        lora_alpha = config.lora.lora_alpha,
        lora_dropout = config.lora.lora_dropout,
        target_modules = config.lora.target_modules,
        bias= config.lora.bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model,loraconfig)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        fp16=config.training.fp16,
        push_to_hub=config.hub.push_to_hub,
        hub_model_id=config.hub.repo_name,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(config.training.output_dir)

if __name__ == "__main__":
    main()    
    