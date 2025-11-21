"""
LoRA Configuration for Legal Domain Fine-Tuning

Defines LoRA hyperparameters and target modules.
"""

from dataclasses import dataclass
from typing import List, Optional
import yaml


@dataclass
class LoRAConfig:
    """LoRA configuration for legal domain."""
    
    # LoRA hyperparameters
    r: int = 16  # Rank (low-rank dimension)
    alpha: int = 32  # Scaling parameter
    dropout: float = 0.05  # Dropout rate
    bias: str = "none"  # Bias type: "none", "all", "lora_only"
    
    # Target modules (which layers to apply LoRA)
    target_modules: List[str] = None
    
    # Training parameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Output
    output_dir: str = "models/lora_legal"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'LoRAConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict.get('lora', {}))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "output_dir": self.output_dir,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps
        }
    
    def get_peft_config(self):
        """Get PEFT LoRA config for training."""
        from peft import LoraConfig
        
        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type="CAUSAL_LM"
        )


def get_legal_lora_config() -> LoRAConfig:
    """Get optimized LoRA config for legal domain."""
    return LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation_steps=8,
        num_epochs=3
    )


if __name__ == "__main__":
    # Example usage
    config = get_legal_lora_config()
    print("LoRA Configuration:")
    print(f"  Rank (r): {config.r}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")

