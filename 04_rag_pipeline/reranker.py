"""
Re-ranking Model for Retrieved Documents

Uses cross-encoder for better relevance scoring.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np


class LegalReranker:
    """Re-ranker for legal document retrieval."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize re-ranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        print(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Re-ranker loaded")
    
    def rerank(
        self,
        query: str,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Re-rank retrieved results.
        
        Args:
            query: User query
            results: List of retrieved documents
            
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result['document']] for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
            result['score'] = float(score)  # Update main score
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results


if __name__ == "__main__":
    # Example usage
    reranker = LegalReranker()
    
    # Mock results
    results = [
        {"document": "IPC Section 302 deals with murder.", "score": 0.8},
        {"document": "CrPC Section 438 is about bail.", "score": 0.7},
        {"document": "Article 21 guarantees right to life.", "score": 0.6}
    ]
    
    query = "What is the punishment for murder?"
    reranked = reranker.rerank(query, results)
    
    print("Re-ranked results:")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Score: {result['rerank_score']:.4f} - {result['document']}")

