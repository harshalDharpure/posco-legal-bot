"""
LoRA Fine-Tuning Training Script

Full training code template for legal domain fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import argparse
from pathlib import Path
from lora_config import LoRAConfig, get_legal_lora_config
from typing import Dict, List


class LegalLoRATrainer:
    """LoRA trainer for legal domain."""
    
    def __init__(
        self,
        model_name: str,
        lora_config: LoRAConfig,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: Base model name
            lora_config: LoRA configuration
            device: Device to use
        """
        self.model_name = model_name
        self.lora_config = lora_config
        self.device = device
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with quantization
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        peft_config = self.lora_config.get_peft_config()
        self.model = get_peft_model(self.model, peft_config)
        
        self.model.print_trainable_parameters()
    
    def prepare_dataset(
        self,
        dataset_path: str,
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset JSON file
            max_length: Maximum sequence length
            
        Returns:
            HuggingFace Dataset
        """
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = data.get('entries', [])
        print(f"Loaded {len(entries)} training examples")
        
        # Format as prompts
        def format_prompt(entry: Dict) -> str:
            """Format entry as training prompt."""
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            context = entry.get('context', '')
            
            # Create prompt
            prompt = f"""You are a legal assistant. Answer the following question based on the provided context.

Context: {context}

Question: {question}

Answer: {answer}"""
            
            return prompt
        
        # Create prompts
        prompts = [format_prompt(entry) for entry in entries]
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": prompts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        output_dir: str = None
    ):
        """
        Train LoRA model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.lora_config.output_dir
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.lora_config.num_epochs,
            per_device_train_batch_size=self.lora_config.batch_size,
            gradient_accumulation_steps=self.lora_config.gradient_accumulation_steps,
            learning_rate=self.lora_config.learning_rate,
            warmup_steps=self.lora_config.warmup_steps,
            max_grad_norm=self.lora_config.max_grad_norm,
            logging_steps=self.lora_config.logging_steps,
            save_steps=self.lora_config.save_steps,
            eval_steps=self.lora_config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True,
            report_to="tensorboard"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def save_model(self, output_dir: str):
        """Save LoRA model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")
    parser.add_argument(
        "--model",
        type=str,
        default="ai4bharat/IndicLegal-LLaMA-7B",
        help="Base model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training dataset JSON file"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Evaluation dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/lora_legal",
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="LoRA config YAML file (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        lora_config = LoRAConfig.from_yaml(args.config)
    else:
        lora_config = get_legal_lora_config()
        # Override with command-line args
        lora_config.num_epochs = args.epochs
        lora_config.batch_size = args.batch_size
        lora_config.learning_rate = args.learning_rate
        lora_config.output_dir = args.output
    
    # Initialize trainer
    trainer = LegalLoRATrainer(args.model, lora_config)
    
    # Prepare datasets
    train_dataset = trainer.prepare_dataset(args.dataset)
    eval_dataset = None
    if args.eval_dataset:
        eval_dataset = trainer.prepare_dataset(args.eval_dataset)
    
    # Train
    trainer.train(train_dataset, eval_dataset, args.output)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

