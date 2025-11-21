"""
Validator-Bot: RLHF-Trained Hallucination Detector

Validates answers for correctness, citations, and safety.
"""

from anti_hallucination import AntiHallucinationGuard
from safety_layer import LegalSafetyLayer
from typing import Dict


class ValidatorBot:
    """Validator bot for answer quality and safety."""
    
    def __init__(self):
        """Initialize Validator-Bot."""
        self.anti_hallucination = AntiHallucinationGuard()
        self.safety_layer = LegalSafetyLayer()
    
    def validate(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict:
        """
        Comprehensive answer validation.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Validation results
        """
        # Hallucination check
        hallucination_checks = self.anti_hallucination.check_hallucination(
            answer, context, question
        )
        
        # Safety check
        safety_checks = self.safety_layer.check_safety(answer, question)
        
        # Combined validation
        is_valid = (
            hallucination_checks["is_safe"] and
            safety_checks["is_safe"] and
            hallucination_checks["has_citations"] and
            hallucination_checks["citations_in_context"]
        )
        
        validation_score = (
            (1 - hallucination_checks["hallucination_score"]) * 0.5 +
            (1 if safety_checks["is_safe"] else 0) * 0.3 +
            (1 if hallucination_checks["has_citations"] else 0) * 0.2
        )
        
        return {
            "is_valid": is_valid,
            "validation_score": validation_score,
            "hallucination_checks": hallucination_checks,
            "safety_checks": safety_checks,
            "bot": "Validator-Bot",
            "recommendations": self._get_recommendations(
                hallucination_checks,
                safety_checks
            )
        }
    
    def _get_recommendations(
        self,
        hallucination_checks: Dict,
        safety_checks: Dict
    ) -> List[str]:
        """Get recommendations for improvement."""
        recommendations = []
        
        if not hallucination_checks["has_citations"]:
            recommendations.append("Add legal citations to support the answer")
        
        if not hallucination_checks["citations_in_context"]:
            recommendations.append("Ensure citations match the retrieved context")
        
        if not safety_checks["is_safe"]:
            recommendations.append("Add safety disclaimers")
        
        if hallucination_checks["hallucination_score"] > 0.5:
            recommendations.append("Review answer for factual accuracy")
        
        return recommendations
    
    def filter_and_enhance(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict:
        """
        Filter unsafe content and enhance answer.
        
        Args:
            answer: Original answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Enhanced answer and validation
        """
        # Validate
        validation = self.validate(answer, context, question)
        
        # Filter if needed
        if not validation["is_valid"]:
            # Filter hallucinations
            filtered_answer, _ = self.anti_hallucination.filter_unsafe_answer(
                answer, context, question
            )
            
            # Filter safety issues
            filtered_answer, _ = self.safety_layer.filter_unsafe_content(
                filtered_answer, question
            )
            
            answer = filtered_answer
        
        return {
            "answer": answer,
            "validation": validation,
            "bot": "Validator-Bot"
        }


if __name__ == "__main__":
    # Example usage
    bot = ValidatorBot()
    
    answer = "IPC Section 302 provides punishment for murder."
    context = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life."
    question = "What is the punishment for murder?"
    
    validation = bot.validate(answer, context, question)
    
    print("Validation Results:")
    print(f"  Is Valid: {validation['is_valid']}")
    print(f"  Validation Score: {validation['validation_score']:.2f}")
    print(f"  Recommendations: {validation['recommendations']}")

