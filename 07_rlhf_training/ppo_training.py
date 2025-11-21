"""
PPO Training for Legal Correctness

Proximal Policy Optimization loop for legal domain.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from peft import PeftModel
from reward_model import LegalRewardModel
from typing import List, Dict
import argparse


class LegalPPOTrainer:
    """PPO trainer for legal domain."""
    
    def __init__(
        self,
        model_path: str,
        reward_model_path: str,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        mini_batch_size: int = 2,
        ppo_epochs: int = 4
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model_path: Path to base model
            reward_model_path: Path to reward model
            learning_rate: Learning rate
            batch_size: Batch size
            mini_batch_size: Mini batch size
            ppo_epochs: PPO epochs
        """
        # Load model
        base_model_name = "ai4bharat/IndicLegal-LLaMA-7B"
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
        
        # Load reward model
        self.reward_model = LegalRewardModel()
        if Path(reward_model_path).exists():
            self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.reward_model.eval()
        
        # PPO config
        self.ppo_config = PPOConfig(
            model_name=base_model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
            cliprange=0.2,
            cliprange_value=0.2,
            gamma=1.0,
            lam=0.95
        )
        
        # PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def train_step(
        self,
        queries: List[str],
        contexts: List[str] = None
    ) -> Dict:
        """
        Perform one PPO training step.
        
        Args:
            queries: List of queries
            contexts: Optional contexts
            
        Returns:
            Training metrics
        """
        # Generate responses
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Compute rewards
        rewards = []
        for i, response in enumerate(responses):
            context = contexts[i] if contexts else None
            reward_dict = self.reward_model.forward(response, context)
            rewards.append(reward_dict['weighted_reward'])
        
        rewards = torch.tensor(rewards)
        
        # PPO step
        stats = self.ppo_trainer.step(queries, responses, rewards)
        
        return stats
    
    def train(
        self,
        dataset: List[Dict],
        num_steps: int = 1000,
        output_dir: str = "models/ppo_legal"
    ):
        """
        Train with PPO.
        
        Args:
            dataset: List of training examples with queries and contexts
            num_steps: Number of training steps
            output_dir: Output directory
        """
        print(f"Starting PPO training for {num_steps} steps")
        
        for step in range(num_steps):
            # Sample batch
            batch = dataset[step % len(dataset):step % len(dataset) + self.ppo_config.batch_size]
            
            queries = [ex['query'] for ex in batch]
            contexts = [ex.get('context') for ex in batch]
            
            # Training step
            stats = self.train_step(queries, contexts)
            
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{num_steps}")
                print(f"  Mean reward: {stats['ppo/mean_scores']:.4f}")
                print(f"  Policy loss: {stats['ppo/policy/entropy']:.4f}")
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        required=True,
        help="Path to reward model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ppo_legal",
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    import json
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    entries = dataset.get('entries', [])
    
    # Initialize trainer
    trainer = LegalPPOTrainer(args.model, args.reward_model)
    
    # Train
    trainer.train(entries, args.steps, args.output)
    
    print("PPO training complete!")


if __name__ == "__main__":
    main()

