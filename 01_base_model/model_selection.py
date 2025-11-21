"""
Base Model Selection for Multilingual Legal Conversational Bot

Supports:
- IndicLegal-LLaMA-7B (Recommended for Indian legal domain)
- LLaMA-3-8B-Instruct (General multilingual)
- IndicBERT (Hindi-English bilingual)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BitsAndBytesConfig
)
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple


class LegalModelSelector:
    """Selects and loads appropriate base model for legal NLP tasks."""
    
    MODELS = {
        "indiclegal-llama": {
            "name": "ai4bharat/IndicLegal-LLaMA-7B",
            "type": "causal_lm",
            "description": "Pre-trained on Indian legal corpus, supports Hindi + English",
            "max_length": 4096,
            "recommended": True
        },
        "llama-3-instruct": {
            "name": "meta-llama/Llama-3-8B-Instruct",
            "type": "causal_lm",
            "description": "General multilingual instruction-tuned model",
            "max_length": 8192,
            "recommended": False
        },
        "indicbert": {
            "name": "ai4bharat/IndicBERT",
            "type": "masked_lm",
            "description": "Hindi-English bilingual BERT",
            "max_length": 512,
            "recommended": False
        }
    }
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize model selector with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['base_model']['device']
        self.quantization_config = None
        
        if self.device == "cuda" and torch.cuda.is_available():
            # 4-bit quantization for memory efficiency
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model."""
        return self.MODELS.get(model_key, {})
    
    def list_models(self) -> Dict:
        """List all available models."""
        return self.MODELS
    
    def load_model(
        self,
        model_key: str = "indiclegal-llama",
        use_quantization: bool = True
    ) -> Tuple[any, any]:
        """
        Load model and tokenizer.
        
        Args:
            model_key: Key of model to load
            use_quantization: Whether to use 4-bit quantization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")
        
        model_info = self.MODELS[model_key]
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        print(f"Loading model: {model_name}")
        print(f"Type: {model_type}")
        print(f"Description: {model_info['description']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        if model_type == "causal_lm":
            if use_quantization and self.quantization_config:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
        elif model_type == "masked_lm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        return model, tokenizer
    
    def get_recommended_model(self) -> str:
        """Get the recommended model for legal tasks."""
        for key, info in self.MODELS.items():
            if info.get("recommended", False):
                return key
        return "indiclegal-llama"  # Default fallback


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select and load base model")
    parser.add_argument(
        "--model",
        type=str,
        default="indiclegal-llama",
        choices=["indiclegal-llama", "llama-3-instruct", "indicbert"],
        help="Model to load"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    selector = LegalModelSelector()
    
    if args.list:
        print("\nAvailable Models:")
        print("=" * 80)
        for key, info in selector.list_models().items():
            print(f"\n{key.upper()}")
            print(f"  Name: {info['name']}")
            print(f"  Type: {info['type']}")
            print(f"  Description: {info['description']}")
            print(f"  Max Length: {info['max_length']}")
            print(f"  Recommended: {info['recommended']}")
        return
    
    # Load model
    model, tokenizer = selector.load_model(
        model_key=args.model,
        use_quantization=not args.no_quantization
    )
    
    # Test tokenization
    test_text = "IPC Section 302 deals with punishment for murder."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"\nTest tokenization:")
    print(f"Input: {test_text}")
    print(f"Token IDs: {tokens['input_ids']}")
    print(f"Decoded: {tokenizer.decode(tokens['input_ids'][0])}")


if __name__ == "__main__":
    main()

