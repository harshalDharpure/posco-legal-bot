"""
Hallucination Penalty Metric

Penalizes answers for fabricated or unsupported information.
"""

from typing import Dict, List
from anti_hallucination import AntiHallucinationGuard
from citation_extractor import LegalCitationExtractor


class HallucinationPenaltyMetric:
    """Hallucination penalty evaluation metric."""
    
    def __init__(self):
        """Initialize metric."""
        self.anti_hallucination = AntiHallucinationGuard()
        self.citation_extractor = LegalCitationExtractor()
    
    def compute_penalty(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict:
        """
        Compute hallucination penalty.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Penalty metrics
        """
        # Check for hallucinations
        hallucination_checks = self.anti_hallucination.check_hallucination(
            answer, context, question
        )
        
        # Base penalty from hallucination score
        base_penalty = hallucination_checks["hallucination_score"]
        
        # Additional penalties
        penalties = {
            "no_citations": 0.0,
            "citations_not_in_context": 0.0,
            "factual_inconsistency": 0.0
        }
        
        # Penalty for no citations
        if not hallucination_checks["has_citations"]:
            penalties["no_citations"] = 0.3
        
        # Penalty for citations not in context
        if not hallucination_checks["citations_in_context"]:
            penalties["citations_not_in_context"] = 0.4
        
        # Penalty for factual inconsistency
        if not hallucination_checks["factual_consistency"]:
            penalties["factual_inconsistency"] = 0.3
        
        # Total penalty
        total_penalty = base_penalty + sum(penalties.values())
        total_penalty = min(total_penalty, 1.0)  # Cap at 1.0
        
        # Score (inverse of penalty)
        score = 1.0 - total_penalty
        
        return {
            "penalty": total_penalty,
            "score": score,
            "base_penalty": base_penalty,
            "component_penalties": penalties,
            "hallucination_checks": hallucination_checks
        }
    
    def evaluate_batch(
        self,
        answers: List[str],
        contexts: List[str],
        questions: List[str]
    ) -> Dict:
        """
        Evaluate batch of answers.
        
        Args:
            answers: List of generated answers
            contexts: List of contexts
            questions: List of questions
            
        Returns:
            Aggregate metrics
        """
        all_metrics = []
        for answer, context, question in zip(answers, contexts, questions):
            metrics = self.compute_penalty(answer, context, question)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_penalty = sum(m["penalty"] for m in all_metrics) / len(all_metrics)
        avg_score = sum(m["score"] for m in all_metrics) / len(all_metrics)
        
        # Count high-penalty answers
        high_penalty_count = sum(1 for m in all_metrics if m["penalty"] > 0.5)
        
        return {
            "average_penalty": avg_penalty,
            "average_score": avg_score,
            "high_penalty_count": high_penalty_count,
            "high_penalty_ratio": high_penalty_count / len(all_metrics),
            "num_samples": len(answers),
            "individual_metrics": all_metrics
        }


if __name__ == "__main__":
    # Example usage
    metric = HallucinationPenaltyMetric()
    
    answer = "IPC Section 302 provides punishment for murder."
    context = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life."
    question = "What is the punishment for murder?"
    
    result = metric.compute_penalty(answer, context, question)
    
    print("Hallucination Penalty Metrics:")
    print(f"  Penalty: {result['penalty']:.3f}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Base Penalty: {result['base_penalty']:.3f}")
    print(f"  Component Penalties: {result['component_penalties']}")

