"""
RAG Relevance Score Metric (Recall@k)

Evaluates quality of retrieved documents.
"""

from typing import Dict, List
import numpy as np


class RAGRelevanceMetric:
    """RAG relevance evaluation metric."""
    
    def compute_recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Compute Recall@k.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            Recall@k score
        """
        if not relevant_docs:
            return 0.0
        
        # Take top k retrieved
        top_k_retrieved = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_retrieved = sum(1 for doc in top_k_retrieved if doc in relevant_docs)
        
        recall = relevant_retrieved / len(relevant_docs)
        
        return recall
    
    def compute_precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Compute Precision@k.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            Precision@k score
        """
        if not retrieved_docs:
            return 0.0
        
        # Take top k retrieved
        top_k_retrieved = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_retrieved = sum(1 for doc in top_k_retrieved if doc in relevant_docs)
        
        precision = relevant_retrieved / len(top_k_retrieved)
        
        return precision
    
    def compute_ndcg_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Compute NDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            NDCG@k score
        """
        # Take top k
        top_k = retrieved_docs[:k]
        
        # Compute DCG
        dcg = 0.0
        for i, doc in enumerate(top_k):
            relevance = 1.0 if doc in relevant_docs else 0.0
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute IDCG (ideal DCG)
        ideal_relevant = relevant_docs[:k]
        idcg = 0.0
        for i in range(len(ideal_relevant)):
            idcg += 1.0 / np.log2(i + 2)
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Comprehensive retrieval evaluation.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        for k in k_values:
            recall = self.compute_recall_at_k(retrieved_docs, relevant_docs, k)
            precision = self.compute_precision_at_k(retrieved_docs, relevant_docs, k)
            ndcg = self.compute_ndcg_at_k(retrieved_docs, relevant_docs, k)
            
            results[f"recall@{k}"] = recall
            results[f"precision@{k}"] = precision
            results[f"ndcg@{k}"] = ndcg
        
        # Mean metrics
        results["mean_recall"] = np.mean([results[f"recall@{k}"] for k in k_values])
        results["mean_precision"] = np.mean([results[f"precision@{k}"] for k in k_values])
        results["mean_ndcg"] = np.mean([results[f"ndcg@{k}"] for k in k_values])
        
        return results


if __name__ == "__main__":
    # Example usage
    metric = RAGRelevanceMetric()
    
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc6"]
    
    results = metric.evaluate_retrieval(retrieved, relevant)
    
    print("RAG Relevance Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")

