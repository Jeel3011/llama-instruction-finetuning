# 🦙 Llama 3.2 Instruction Fine-Tuning

Fine-tunes **Llama 3.2-1B** on the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using **LoRA** (Low-Rank Adaptation). Training runs on Google Colab (T4 GPU, free tier).

## 📁 Project Structure

```
llama-instruction-finetuning/
├── configs/
│   └── config.py          # All training/LoRA/data hyperparameters
├── src/
│   ├── data_prep.py       # Dataset loading, formatting, tokenization
│   ├── train.py           # Training loop with HuggingFace Trainer
│   ├── evaluate.py        # Perplexity + sample generation evaluation
│   └── inference.py       # Interactive chat + single-shot inference
├── notebook/
│   └── finetune_llama.ipynb  # 📓 Google Colab training notebook
├── results/               # Saved LoRA adapter after training
└── requirements.txt
```

## ☁️ Train on Google Colab (Recommended)

1. Open `notebook/finetune_llama.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Follow the cells step-by-step

The notebook handles: dependency install → repo clone → HF login → data prep → training → save to Drive/Hub → evaluation → inference demo.

## 💻 Local Setup (for development)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Test the data pipeline locally:
```bash
python -c "
from configs.config import get_config
from src.data_prep import prepare_dataset
c = get_config()
c.data.max_samples = 10
d, t = prepare_dataset(c)
print(d)
"
```

## 🔧 Configuration

Edit `configs/config.py` to change any hyperparameter:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `meta-llama/Llama-3.2-1B` | HF model ID |
| `lora.r` | `16` | LoRA rank (higher = more params) |
| `lora.lora_alpha` | `32` | LoRA scaling |
| `lora.target_modules` | `["q_proj", "v_proj"]` | Layers to fine-tune |
| `training.num_epochs` | `3` | Training epochs |
| `training.batch_size` | `4` | Per-device batch size |
| `training.learning_rate` | `2e-4` | Learning rate |
| `data.max_samples` | `None` | Limit dataset (None = full 52K) |
| `hub.push_to_hub` | `False` | Push adapter to HF Hub after training |

## 🤖 Inference (after training)

Interactive chat:
```bash
python src/inference.py
```

Single instruction:
```bash
python src/inference.py --instruction "Give three tips for staying healthy."
```

With context input:
```bash
python src/inference.py \
  --instruction "Summarize the following text." \
  --input "The Eiffel Tower was built in 1889..."
```

## 📈 Training Results

| Metric | Value |
|--------|-------|
| Base model | Llama 3.2-1B |
| Dataset | Alpaca (52K instructions) |
| Trainable params | ~1.7M (0.14% of 1.24B) |
| Adapter size | ~800KB |

*(Update after your training run)*

## Requirements

- HuggingFace account with Llama 3.2 access approved at [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Google Colab free account (for T4 GPU)