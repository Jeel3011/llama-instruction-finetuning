# configs/config.py

from dataclasses import dataclass, field
from typing import List, Optional


# -------------------------
# Model Config
# -------------------------

@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 512
    use_fp16: bool = True


# -------------------------
# LoRA Config
# -------------------------

@dataclass
class LoraConfigCustom:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"


# -------------------------
# Training Config
# -------------------------

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2       # Keep only the 2 most recent checkpoints
    fp16: bool = True


# -------------------------
# Data Config
# -------------------------

@dataclass
class DataConfig:
    name: str = "tatsu-lab/alpaca"
    train_split: str = "train"
    test_size: float = 0.1
    max_samples: Optional[int] = 5000   # Set None for full 52K dataset


# -------------------------
# HuggingFace Hub Config
# -------------------------

@dataclass
class HubConfig:
    repo_name: str = "Jeel3011/llama-alpaca-finetuned"
    push_to_hub: bool = False           # Set True only if HF token is configured


# -------------------------
# Master Config
# -------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigCustom = field(default_factory=LoraConfigCustom)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def get_config():
    return Config()