"""
Reward Model for Legal Domain

Combines multiple reward signals:
- Legal factuality
- Citation accuracy
- Language fluency
- Safety
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np


class LegalRewardModel(nn.Module):
    """Reward model for legal domain RLHF."""
    
    def __init__(
        self,
        base_model_name: str = "ai4bharat/IndicLegal-LLaMA-7B",
        legal_factuality_weight: float = 0.4,
        citation_accuracy_weight: float = 0.3,
        language_fluency_weight: float = 0.2,
        safety_weight: float = 0.1
    ):
        """
        Initialize reward model.
        
        Args:
            base_model_name: Base model for encoding
            legal_factuality_weight: Weight for legal factuality reward
            citation_accuracy_weight: Weight for citation accuracy reward
            language_fluency_weight: Weight for language fluency reward
            safety_weight: Weight for safety reward
        """
        super().__init__()
        
        self.legal_factuality_weight = legal_factuality_weight
        self.citation_accuracy_weight = citation_accuracy_weight
        self.language_fluency_weight = language_fluency_weight
        self.safety_weight = safety_weight
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Reward heads
        hidden_size = self.encoder.config.hidden_size
        
        self.legal_factuality_head = nn.Linear(hidden_size, 1)
        self.citation_accuracy_head = nn.Linear(hidden_size, 1)
        self.language_fluency_head = nn.Linear(hidden_size, 1)
        self.safety_head = nn.Linear(hidden_size, 1)
        
        # Final reward head
        self.reward_head = nn.Linear(4, 1)
    
    def forward(
        self,
        text: str,
        context: str = None
    ) -> Dict[str, float]:
        """
        Compute reward for text.
        
        Args:
            text: Text to evaluate
            context: Optional context for factuality check
            
        Returns:
            Dictionary with reward components and total reward
        """
        # Encode text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Compute individual rewards
        legal_factuality = torch.sigmoid(self.legal_factuality_head(pooled)).item()
        citation_accuracy = torch.sigmoid(self.citation_accuracy_head(pooled)).item()
        language_fluency = torch.sigmoid(self.language_fluency_head(pooled)).item()
        safety = torch.sigmoid(self.safety_head(pooled)).item()
        
        # Combine rewards
        rewards = torch.tensor([
            legal_factuality,
            citation_accuracy,
            language_fluency,
            safety
        ]).unsqueeze(0)
        
        total_reward = self.reward_head(rewards).item()
        
        # Apply weights
        weighted_reward = (
            legal_factuality * self.legal_factuality_weight +
            citation_accuracy * self.citation_accuracy_weight +
            language_fluency * self.language_fluency_weight +
            safety * self.safety_weight
        )
        
        return {
            "legal_factuality": legal_factuality,
            "citation_accuracy": citation_accuracy,
            "language_fluency": language_fluency,
            "safety": safety,
            "total_reward": total_reward,
            "weighted_reward": weighted_reward
        }
    
    def compute_reward_batch(
        self,
        texts: List[str],
        contexts: List[str] = None
    ) -> List[Dict[str, float]]:
        """
        Compute rewards for batch of texts.
        
        Args:
            texts: List of texts
            contexts: Optional list of contexts
            
        Returns:
            List of reward dictionaries
        """
        rewards = []
        for i, text in enumerate(texts):
            context = contexts[i] if contexts else None
            reward = self.forward(text, context)
            rewards.append(reward)
        
        return rewards


class LegalFactualityReward:
    """Legal factuality reward component."""
    
    def __init__(self):
        """Initialize factuality reward."""
        pass
    
    def compute(
        self,
        answer: str,
        context: str,
        question: str
    ) -> float:
        """
        Compute legal factuality reward.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Factuality score (0-1)
        """
        # Check if answer is consistent with context
        # Simple implementation - can be enhanced with NLI models
        
        # Check for contradictions
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Simple overlap check
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        
        overlap = len(answer_words.intersection(context_words))
        total = len(answer_words)
        
        if total == 0:
            return 0.0
        
        overlap_ratio = overlap / total
        
        # Check for legal citations
        import re
        citations = re.findall(r'(IPC|CrPC|Article|Section)\s+\d+', answer, re.IGNORECASE)
        citation_bonus = min(len(citations) * 0.1, 0.3)
        
        return min(overlap_ratio + citation_bonus, 1.0)


class CitationAccuracyReward:
    """Citation accuracy reward component."""
    
    def __init__(self):
        """Initialize citation accuracy reward."""
        pass
    
    def compute(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Compute citation accuracy reward.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Citation accuracy score (0-1)
        """
        import re
        from citation_extractor import LegalCitationExtractor
        
        extractor = LegalCitationExtractor()
        
        # Extract citations from answer and context
        answer_citations = extractor.extract_citations(answer)
        context_citations = extractor.extract_citations(context)
        
        if not answer_citations:
            return 0.0  # No citations = low reward
        
        # Check if citations match context
        answer_cite_texts = {
            c.get('section', c.get('article', '')) for c in answer_citations
        }
        context_cite_texts = {
            c.get('section', c.get('article', '')) for c in context_citations
        }
        
        if not answer_cite_texts:
            return 0.0
        
        # Compute accuracy
        matches = answer_cite_texts.intersection(context_cite_texts)
        accuracy = len(matches) / len(answer_cite_texts) if answer_cite_texts else 0.0
        
        return accuracy


class LanguageFluencyReward:
    """Language fluency reward component."""
    
    def __init__(self):
        """Initialize fluency reward."""
        pass
    
    def compute(self, text: str) -> float:
        """
        Compute language fluency reward.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Fluency score (0-1)
        """
        # Simple fluency check based on:
        # - Sentence length
        # - Word repetition
        # - Grammar indicators
        
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        # Average sentence length (reasonable range: 10-30 words)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if 10 <= avg_length <= 30:
            length_score = 1.0
        elif avg_length < 10:
            length_score = avg_length / 10
        else:
            length_score = max(0, 1 - (avg_length - 30) / 30)
        
        # Check for repetition
        words = text.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        diversity_score = unique_words / total_words
        
        # Combined fluency score
        fluency = (length_score + diversity_score) / 2
        
        return min(fluency, 1.0)


class SafetyReward:
    """Safety reward component to avoid harmful legal advice."""
    
    DANGEROUS_PATTERNS = [
        r'guarantee.*result',
        r'definitely.*win',
        r'100%.*success',
        r'no.*risk',
        r'always.*legal'
    ]
    
    def __init__(self):
        """Initialize safety reward."""
        import re
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
    
    def compute(self, text: str) -> float:
        """
        Compute safety reward.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Safety score (0-1), higher is safer
        """
        # Check for dangerous patterns
        danger_count = sum(1 for pattern in self.patterns if pattern.search(text))
        
        # Check for disclaimers
        has_disclaimer = any(
            word in text.lower() for word in ['consult', 'expert', 'advice', 'legal counsel']
        )
        
        # Safety score
        if danger_count == 0 and has_disclaimer:
            return 1.0
        elif danger_count == 0:
            return 0.8
        elif danger_count <= 1:
            return 0.5
        else:
            return 0.2


if __name__ == "__main__":
    # Example usage
    reward_model = LegalRewardModel()
    
    answer = "IPC Section 302 provides that whoever commits murder shall be punished with death or imprisonment for life."
    context = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    
    reward = reward_model.forward(answer, context)
    
    print("Reward Components:")
    print(f"  Legal Factuality: {reward['legal_factuality']:.3f}")
    print(f"  Citation Accuracy: {reward['citation_accuracy']:.3f}")
    print(f"  Language Fluency: {reward['language_fluency']:.3f}")
    print(f"  Safety: {reward['safety']:.3f}")
    print(f"  Weighted Reward: {reward['weighted_reward']:.3f}")

