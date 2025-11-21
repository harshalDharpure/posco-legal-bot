"""
Legal-Q-Bot: Generates Legal Answers

Primary bot for generating legal answers to user queries.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Optional
from pathlib import Path


class LegalQBot:
    """Legal question-answering bot."""
    
    def __init__(
        self,
        model_path: str = "models/lora_legal",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize Legal-Q-Bot.
        
        Args:
            model_path: Path to fine-tuned model
            temperature: Sampling temperature
            max_tokens: Maximum generation length
        """
        base_model_name = "ai4bharat/IndicLegal-LLaMA-7B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load LoRA if available
        if Path(model_path).exists():
            self.model = PeftModel.from_pretrained(self.base_model, model_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_answer(
        self,
        query: str,
        context: str = None,
        language: str = "en"
    ) -> Dict:
        """
        Generate legal answer.
        
        Args:
            query: User query
            context: Optional RAG context
            language: Language ("en" or "hi")
            
        Returns:
            Dictionary with answer and metadata
        """
        # Create prompt
        if context:
            prompt = f"""Context: {context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Question: {query}

Answer:"""
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip()
        
        return {
            "answer": answer,
            "query": query,
            "context": context,
            "language": language,
            "bot": "Legal-Q-Bot"
        }


if __name__ == "__main__":
    # Example usage
    bot = LegalQBot()
    
    query = "What is the punishment for murder under IPC?"
    result = bot.generate_answer(query)
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")

