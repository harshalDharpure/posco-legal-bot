"""
Legal Accuracy Score Metric

Evaluates factual correctness of legal information in answers.
"""

from typing import Dict, List
import re
from citation_extractor import LegalCitationExtractor


class LegalAccuracyMetric:
    """Legal accuracy evaluation metric."""
    
    def __init__(self):
        """Initialize metric."""
        self.citation_extractor = LegalCitationExtractor()
    
    def compute_accuracy(
        self,
        answer: str,
        ground_truth: str,
        context: str = None
    ) -> Dict:
        """
        Compute legal accuracy score.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            context: Retrieved context
            
        Returns:
            Accuracy metrics
        """
        # Extract citations from both
        answer_citations = self.citation_extractor.extract_citations(answer)
        gt_citations = self.citation_extractor.extract_citations(ground_truth)
        
        # Citation accuracy
        answer_cite_texts = {
            c.get('section', c.get('article', '')) for c in answer_citations
        }
        gt_cite_texts = {
            c.get('section', c.get('article', '')) for c in gt_citations
        }
        
        if gt_cite_texts:
            citation_precision = len(answer_cite_texts.intersection(gt_cite_texts)) / len(answer_cite_texts) if answer_cite_texts else 0
            citation_recall = len(answer_cite_texts.intersection(gt_cite_texts)) / len(gt_cite_texts)
            citation_f1 = 2 * citation_precision * citation_recall / (citation_precision + citation_recall) if (citation_precision + citation_recall) > 0 else 0
        else:
            citation_precision = 0
            citation_recall = 0
            citation_f1 = 0
        
        # Semantic similarity (simple word overlap)
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        overlap = len(answer_words.intersection(gt_words))
        semantic_similarity = overlap / len(gt_words) if gt_words else 0
        
        # Factual consistency with context
        context_consistency = 0.0
        if context:
            context_words = set(context.lower().split())
            answer_context_overlap = len(answer_words.intersection(context_words))
            context_consistency = answer_context_overlap / len(answer_words) if answer_words else 0
        
        # Overall accuracy (weighted combination)
        accuracy = (
            citation_f1 * 0.4 +
            semantic_similarity * 0.4 +
            context_consistency * 0.2
        )
        
        return {
            "accuracy": accuracy,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_f1": citation_f1,
            "semantic_similarity": semantic_similarity,
            "context_consistency": context_consistency
        }
    
    def evaluate_batch(
        self,
        answers: List[str],
        ground_truths: List[str],
        contexts: List[str] = None
    ) -> Dict:
        """
        Evaluate batch of answers.
        
        Args:
            answers: List of generated answers
            ground_truths: List of ground truth answers
            contexts: Optional list of contexts
            
        Returns:
            Aggregate metrics
        """
        if contexts is None:
            contexts = [None] * len(answers)
        
        all_metrics = []
        for answer, gt, context in zip(answers, ground_truths, contexts):
            metrics = self.compute_accuracy(answer, gt, context)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_accuracy = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
        avg_citation_f1 = sum(m["citation_f1"] for m in all_metrics) / len(all_metrics)
        avg_semantic = sum(m["semantic_similarity"] for m in all_metrics) / len(all_metrics)
        
        return {
            "average_accuracy": avg_accuracy,
            "average_citation_f1": avg_citation_f1,
            "average_semantic_similarity": avg_semantic,
            "num_samples": len(answers),
            "individual_metrics": all_metrics
        }


if __name__ == "__main__":
    # Example usage
    metric = LegalAccuracyMetric()
    
    answer = "IPC Section 302 provides punishment for murder with death or life imprisonment."
    ground_truth = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    context = "IPC Section 302 deals with murder punishment."
    
    result = metric.compute_accuracy(answer, ground_truth, context)
    
    print("Legal Accuracy Metrics:")
    print(f"  Overall Accuracy: {result['accuracy']:.3f}")
    print(f"  Citation F1: {result['citation_f1']:.3f}")
    print(f"  Semantic Similarity: {result['semantic_similarity']:.3f}")
    print(f"  Context Consistency: {result['context_consistency']:.3f}")

