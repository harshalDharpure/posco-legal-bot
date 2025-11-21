"""
Multilingual Consistency Score

Evaluates consistency between Hindi and English answers.
"""

from typing import Dict, List
import re


class MultilingualConsistencyMetric:
    """Multilingual consistency evaluation metric."""
    
    def __init__(self):
        """Initialize metric."""
        pass
    
    def compute_consistency(
        self,
        answer_en: str,
        answer_hi: str
    ) -> Dict:
        """
        Compute consistency between English and Hindi answers.
        
        Args:
            answer_en: English answer
            answer_hi: Hindi answer
            
        Returns:
            Consistency metrics
        """
        # Extract citations from both
        citations_en = re.findall(r'(IPC|CrPC|Article|Section)\s+\d+', answer_en, re.IGNORECASE)
        citations_hi = re.findall(r'(IPC|CrPC|Article|Section)\s+\d+', answer_hi, re.IGNORECASE)
        citations_hi_devanagari = re.findall(r'धारा\s+\d+|अनुच्छेद\s+\d+', answer_hi)
        
        # Citation consistency
        citation_consistency = 0.0
        if citations_en or citations_hi or citations_hi_devanagari:
            # Check if same number of citations
            total_citations_en = len(citations_en)
            total_citations_hi = len(citations_hi) + len(citations_hi_devanagari)
            
            if total_citations_en > 0 and total_citations_hi > 0:
                citation_consistency = min(total_citations_en, total_citations_hi) / max(total_citations_en, total_citations_hi)
        
        # Extract numbers (section numbers, years, etc.)
        numbers_en = set(re.findall(r'\b\d{3,4}\b', answer_en))
        numbers_hi = set(re.findall(r'\b\d{3,4}\b', answer_hi))
        
        number_overlap = len(numbers_en.intersection(numbers_hi))
        number_consistency = number_overlap / len(numbers_en.union(numbers_hi)) if numbers_en.union(numbers_hi) else 0.0
        
        # Length consistency (answers should be similar length)
        length_ratio = min(len(answer_en), len(answer_hi)) / max(len(answer_en), len(answer_hi)) if max(len(answer_en), len(answer_hi)) > 0 else 0.0
        
        # Overall consistency
        consistency = (
            citation_consistency * 0.4 +
            number_consistency * 0.4 +
            length_ratio * 0.2
        )
        
        return {
            "consistency": consistency,
            "citation_consistency": citation_consistency,
            "number_consistency": number_consistency,
            "length_consistency": length_ratio
        }
    
    def evaluate_batch(
        self,
        answers_en: List[str],
        answers_hi: List[str]
    ) -> Dict:
        """
        Evaluate batch of answer pairs.
        
        Args:
            answers_en: List of English answers
            answers_hi: List of Hindi answers
            
        Returns:
            Aggregate metrics
        """
        all_metrics = []
        for ans_en, ans_hi in zip(answers_en, answers_hi):
            metrics = self.compute_consistency(ans_en, ans_hi)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_consistency = sum(m["consistency"] for m in all_metrics) / len(all_metrics)
        avg_citation = sum(m["citation_consistency"] for m in all_metrics) / len(all_metrics)
        avg_number = sum(m["number_consistency"] for m in all_metrics) / len(all_metrics)
        
        return {
            "average_consistency": avg_consistency,
            "average_citation_consistency": avg_citation,
            "average_number_consistency": avg_number,
            "num_samples": len(answers_en),
            "individual_metrics": all_metrics
        }


if __name__ == "__main__":
    # Example usage
    metric = MultilingualConsistencyMetric()
    
    answer_en = "IPC Section 302 provides punishment for murder with death or life imprisonment."
    answer_hi = "IPC धारा 302 हत्या के लिए मृत्युदंड या आजीवन कारावास की सजा प्रदान करती है।"
    
    result = metric.compute_consistency(answer_en, answer_hi)
    
    print("Multilingual Consistency Metrics:")
    print(f"  Overall Consistency: {result['consistency']:.3f}")
    print(f"  Citation Consistency: {result['citation_consistency']:.3f}")
    print(f"  Number Consistency: {result['number_consistency']:.3f}")
    print(f"  Length Consistency: {result['length_consistency']:.3f}")

