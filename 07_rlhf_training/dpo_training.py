"""
DPO Training for Legal Domain

Direct Preference Optimization using preference pairs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOTrainingArguments
from peft import PeftModel
from typing import List, Dict
import json
import argparse
from pathlib import Path


class LegalDPOTrainer:
    """DPO trainer for legal domain."""
    
    def __init__(
        self,
        model_path: str,
        learning_rate: float = 1e-5,
        beta: float = 0.1
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model_path: Path to base model
            learning_rate: Learning rate
            beta: DPO beta parameter
        """
        base_model_name = "ai4bharat/IndicLegal-LLaMA-7B"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA if available
        if Path(model_path).exists():
            self.model = PeftModel.from_pretrained(self.model, model_path)
        
        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.learning_rate = learning_rate
        self.beta = beta
    
    def prepare_preference_dataset(
        self,
        dataset_path: str
    ) -> List[Dict]:
        """
        Prepare preference dataset for DPO.
        
        Args:
            dataset_path: Path to preference pairs JSON
            
        Returns:
            List of preference examples
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = data.get('entries', [])
        
        # Format for DPO
        dpo_examples = []
        for entry in entries:
            dpo_example = {
                "prompt": entry['query'],
                "chosen": entry['chosen_answer'],
                "rejected": entry['rejected_answer']
            }
            dpo_examples.append(dpo_example)
        
        return dpo_examples
    
    def train(
        self,
        dataset: List[Dict],
        output_dir: str = "models/dpo_legal",
        num_epochs: int = 3
    ):
        """
        Train with DPO.
        
        Args:
            dataset: Preference dataset
            output_dir: Output directory
            num_epochs: Number of epochs
        """
        from datasets import Dataset
        
        # Convert to HuggingFace dataset
        hf_dataset = Dataset.from_list(dataset)
        
        # Training arguments
        training_args = DPOTrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            beta=self.beta,
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="no",
            fp16=True
        )
        
        # DPO trainer
        dpo_trainer = DPOTrainer(
            self.model,
            self.ref_model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=self.tokenizer,
            beta=self.beta
        )
        
        # Train
        print("Starting DPO training...")
        dpo_trainer.train()
        
        # Save model
        dpo_trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")


def create_preference_pairs(
    dataset_path: str,
    output_path: str
):
    """
    Create preference pairs from dataset.
    
    Args:
        dataset_path: Input dataset
        output_path: Output preference pairs file
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data.get('entries', [])
    
    # Create preference pairs
    # In practice, these would come from human annotators
    preference_pairs = []
    
    for entry in entries:
        # Simulate preference: better answer has citations
        answer = entry.get('answer', '')
        has_citations = 'Section' in answer or 'Article' in answer
        
        if has_citations:
            chosen = answer
            # Create a worse version without citations
            rejected = answer.replace('Section', '').replace('Article', '')
        else:
            rejected = answer
            # Create a better version with citations
            chosen = answer + " (See relevant legal sections for details.)"
        
        preference_pairs.append({
            "query": entry.get('question', ''),
            "chosen_answer": chosen,
            "rejected_answer": rejected,
            "context": entry.get('context', '')
        })
    
    # Save
    output_data = {
        "dataset_name": "Legal Preference Pairs",
        "total_pairs": len(preference_pairs),
        "entries": preference_pairs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(preference_pairs)} preference pairs")
    print(f"Saved to {output_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Preference pairs dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/dpo_legal",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter"
    )
    parser.add_argument(
        "--create-pairs",
        action="store_true",
        help="Create preference pairs from dataset"
    )
    
    args = parser.parse_args()
    
    if args.create_pairs:
        # Create preference pairs
        output_pairs = args.dataset.replace('.json', '_preferences.json')
        create_preference_pairs(args.dataset, output_pairs)
        args.dataset = output_pairs
    
    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data.get('entries', [])
    
    # Initialize trainer
    trainer = LegalDPOTrainer(args.model, beta=args.beta)
    
    # Prepare dataset
    dpo_dataset = trainer.prepare_preference_dataset(args.dataset)
    
    # Train
    trainer.train(dpo_dataset, args.output, args.epochs)
    
    print("DPO training complete!")


if __name__ == "__main__":
    main()

