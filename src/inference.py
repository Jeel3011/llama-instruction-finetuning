# src/inference.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data_prep import format_instruction
from configs.config import get_config


class LlamaInference:
    """Simple inference wrapper for the fine-tuned Llama model."""

    def __init__(self, adapter_path: str = None, config=None):
        config = config or get_config()
        adapter_path = adapter_path or config.training.output_dir

        print(f"Loading model: {config.model.name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            dtype=torch.float16 if config.training.fp16 else torch.float32,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        print("Model ready ✓")

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response for a given instruction.

        Args:
            instruction: The task instruction.
            input_text:  Optional context/input for the instruction.
            max_new_tokens: Maximum tokens to generate.
            temperature:  Sampling temperature (higher = more creative).
            top_p:         Nucleus sampling probability.
            do_sample:     Use sampling (True) or greedy decoding (False).

        Returns:
            The model's response as a string.
        """
        example = {"instruction": instruction, "input": input_text, "output": ""}
        prompt = format_instruction(example)["text"]

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt tokens; return only the new generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self):
        """Interactive command-line chat loop."""
        print("\nLlama Fine-Tuned Model — Interactive Chat")
        print("Type 'quit' to exit, 'reset' for a new conversation.\n")

        while True:
            instruction = input("Instruction: ").strip()
            if instruction.lower() == "quit":
                print("Goodbye!")
                break
            if instruction.lower() == "reset":
                print("Cleared.\n")
                continue
            if not instruction:
                continue

            inp = input("Input (press Enter to skip): ").strip()
            print("\nGenerating...", flush=True)
            response = self.generate(instruction, inp)
            print(f"\n🤖 {response}\n")
            print("-" * 50)


def main():
    """Entry point for single-shot inference or interactive chat."""
    import argparse

    parser = argparse.ArgumentParser(description="Llama fine-tuned model inference")
    parser.add_argument(
        "--instruction", type=str, default=None,
        help="Instruction to run (single-shot mode). If omitted, starts interactive chat."
    )
    parser.add_argument(
        "--input", type=str, default="",
        help="Optional input context for the instruction."
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to the LoRA adapter directory. Defaults to config output_dir."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate."
    )
    args = parser.parse_args()

    model = LlamaInference(adapter_path=args.adapter_path)

    if args.instruction:
        # Single-shot mode
        response = model.generate(args.instruction, args.input, args.max_new_tokens)
        print(f"\n🤖 Response:\n{response}")
    else:
        # Interactive chat
        model.chat()


if __name__ == "__main__":
    main()
