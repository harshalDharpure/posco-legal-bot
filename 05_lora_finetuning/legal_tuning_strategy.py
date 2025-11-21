"""
Legal-Domain Parameter-Efficient Tuning Strategy

Optimized LoRA strategy for legal domain adaptation.
"""

from lora_config import LoRAConfig
from typing import Dict, List


class LegalTuningStrategy:
    """Strategy for legal domain fine-tuning."""
    
    STRATEGIES = {
        "conservative": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "description": "Conservative: Fewer parameters, lower risk of overfitting"
        },
        "balanced": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "description": "Balanced: Good trade-off between capacity and efficiency"
        },
        "aggressive": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.01,
            "description": "Aggressive: More parameters, higher capacity"
        }
    }
    
    # Legal-specific target modules
    LEGAL_TARGET_MODULES = {
        "full": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "attention_only": [
            "q_proj", "v_proj", "k_proj", "o_proj"
        ],
        "ffn_only": [
            "gate_proj", "up_proj", "down_proj"
        ]
    }
    
    def __init__(self, strategy: str = "balanced"):
        """
        Initialize tuning strategy.
        
        Args:
            strategy: Strategy name ("conservative", "balanced", "aggressive")
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy_name = strategy
        self.strategy_config = self.STRATEGIES[strategy]
    
    def get_lora_config(
        self,
        target_modules: str = "full",
        learning_rate: float = 2e-4
    ) -> LoRAConfig:
        """
        Get LoRA config for strategy.
        
        Args:
            target_modules: Which modules to target ("full", "attention_only", "ffn_only")
            learning_rate: Learning rate
            
        Returns:
            LoRAConfig
        """
        if target_modules not in self.LEGAL_TARGET_MODULES:
            raise ValueError(f"Unknown target modules: {target_modules}")
        
        config = LoRAConfig(
            r=self.strategy_config["r"],
            alpha=self.strategy_config["alpha"],
            dropout=self.strategy_config["dropout"],
            target_modules=self.LEGAL_TARGET_MODULES[target_modules],
            learning_rate=learning_rate
        )
        
        return config
    
    def get_training_recommendations(self) -> Dict:
        """Get training recommendations for legal domain."""
        return {
            "strategy": self.strategy_name,
            "description": self.strategy_config["description"],
            "recommended_learning_rate": 2e-4,
            "recommended_batch_size": 4,
            "recommended_epochs": 3,
            "recommended_warmup_steps": 100,
            "data_requirements": {
                "minimum_samples": 1000,
                "recommended_samples": 10000,
                "legal_sections": ["IPC", "CrPC", "Constitution"]
            },
            "evaluation_metrics": [
                "legal_accuracy",
                "citation_accuracy",
                "language_fluency",
                "hallucination_rate"
            ]
        }


def get_optimal_strategy(
    dataset_size: int,
    available_gpu_memory: str = "medium"
) -> LegalTuningStrategy:
    """
    Get optimal strategy based on dataset size and resources.
    
    Args:
        dataset_size: Number of training examples
        available_gpu_memory: GPU memory ("low", "medium", "high")
        
    Returns:
        LegalTuningStrategy
    """
    if dataset_size < 1000:
        return LegalTuningStrategy("conservative")
    elif dataset_size < 10000:
        return LegalTuningStrategy("balanced")
    else:
        if available_gpu_memory == "high":
            return LegalTuningStrategy("aggressive")
        else:
            return LegalTuningStrategy("balanced")


if __name__ == "__main__":
    # Example usage
    strategy = LegalTuningStrategy("balanced")
    config = strategy.get_lora_config()
    
    print("Legal Tuning Strategy:")
    print(f"  Strategy: {strategy.strategy_name}")
    print(f"  Description: {strategy.strategy_config['description']}")
    print(f"  Rank (r): {config.r}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Target modules: {config.target_modules}")
    
    recommendations = strategy.get_training_recommendations()
    print(f"\nRecommendations:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")

