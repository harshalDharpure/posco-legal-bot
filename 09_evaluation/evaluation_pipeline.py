"""
Complete Evaluation Pipeline

Combines all evaluation metrics for comprehensive assessment.
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse

from legal_accuracy import LegalAccuracyMetric
from rag_relevance import RAGRelevanceMetric
from multilingual_consistency import MultilingualConsistencyMetric
from hallucination_penalty import HallucinationPenaltyMetric


class EvaluationPipeline:
    """Complete evaluation pipeline."""
    
    def __init__(self):
        """Initialize evaluation pipeline."""
        self.legal_accuracy = LegalAccuracyMetric()
        self.rag_relevance = RAGRelevanceMetric()
        self.multilingual_consistency = MultilingualConsistencyMetric()
        self.hallucination_penalty = HallucinationPenaltyMetric()
    
    def evaluate(
        self,
        test_dataset: List[Dict],
        retrieved_docs: Dict[str, List[str]] = None
    ) -> Dict:
        """
        Comprehensive evaluation.
        
        Args:
            test_dataset: List of test examples
            retrieved_docs: Optional dictionary mapping query IDs to retrieved documents
            
        Returns:
            Complete evaluation results
        """
        results = {
            "legal_accuracy": {},
            "rag_relevance": {},
            "multilingual_consistency": {},
            "hallucination_penalty": {},
            "overall_scores": {}
        }
        
        # Extract data
        answers = [ex.get("answer", "") for ex in test_dataset]
        ground_truths = [ex.get("ground_truth", ex.get("answer", "")) for ex in test_dataset]
        contexts = [ex.get("context", "") for ex in test_dataset]
        questions = [ex.get("question", "") for ex in test_dataset]
        
        # Legal accuracy
        accuracy_results = self.legal_accuracy.evaluate_batch(answers, ground_truths, contexts)
        results["legal_accuracy"] = accuracy_results
        
        # Hallucination penalty
        penalty_results = self.hallucination_penalty.evaluate_batch(answers, contexts, questions)
        results["hallucination_penalty"] = penalty_results
        
        # Multilingual consistency (if available)
        answers_en = [ex.get("answer", "") for ex in test_dataset if ex.get("language") == "en"]
        answers_hi = [ex.get("hindi_answer", "") for ex in test_dataset if ex.get("language") == "en"]
        
        if answers_en and answers_hi and len(answers_en) == len(answers_hi):
            consistency_results = self.multilingual_consistency.evaluate_batch(answers_en, answers_hi)
            results["multilingual_consistency"] = consistency_results
        
        # RAG relevance (if retrieved docs provided)
        if retrieved_docs:
            # This would require ground truth relevant documents
            # For now, placeholder
            results["rag_relevance"] = {"note": "Requires ground truth relevant documents"}
        
        # Overall scores
        results["overall_scores"] = {
            "legal_accuracy": accuracy_results["average_accuracy"],
            "hallucination_score": penalty_results["average_score"],
            "overall_score": (
                accuracy_results["average_accuracy"] * 0.6 +
                penalty_results["average_score"] * 0.4
            )
        }
        
        return results
    
    def evaluate_from_file(
        self,
        test_file: str,
        output_file: str = None
    ) -> Dict:
        """
        Evaluate from test dataset file.
        
        Args:
            test_file: Path to test dataset JSON
            output_file: Optional output file for results
            
        Returns:
            Evaluation results
        """
        # Load test dataset
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = data.get('entries', [])
        
        # Evaluate
        results = self.evaluate(entries)
        
        # Save results
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Evaluation results saved to {output_file}")
        
        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Test dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EvaluationPipeline()
    
    # Evaluate
    results = pipeline.evaluate_from_file(args.test_dataset, args.output)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 60)
    print(f"Legal Accuracy: {results['overall_scores']['legal_accuracy']:.3f}")
    print(f"Hallucination Score: {results['overall_scores']['hallucination_score']:.3f}")
    print(f"Overall Score: {results['overall_scores']['overall_score']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

