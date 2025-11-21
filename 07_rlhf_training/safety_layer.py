"""
Safety Layer Design

Prevents harmful legal advice and ensures responsible AI behavior.
"""

import re
from typing import Dict, List, Tuple
from citation_extractor import LegalCitationExtractor


class LegalSafetyLayer:
    """Safety layer for legal AI."""
    
    # Dangerous patterns that should trigger warnings
    DANGEROUS_PATTERNS = [
        r'guarantee.*result',
        r'definitely.*win',
        r'100%.*success',
        r'no.*risk',
        r'always.*legal',
        r'never.*illegal',
        r'guaranteed.*outcome'
    ]
    
    # Required disclaimers
    REQUIRED_DISCLAIMERS = [
        "This is not legal advice",
        "Consult a qualified lawyer",
        "For informational purposes only"
    ]
    
    def __init__(self):
        """Initialize safety layer."""
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
        self.citation_extractor = LegalCitationExtractor()
    
    def check_safety(
        self,
        answer: str,
        question: str = None
    ) -> Dict:
        """
        Check answer for safety issues.
        
        Args:
            answer: Generated answer
            question: Original question (optional)
            
        Returns:
            Safety check results
        """
        checks = {
            "is_safe": True,
            "warnings": [],
            "risk_level": "low",
            "requires_disclaimer": False
        }
        
        # Check for dangerous patterns
        danger_count = sum(1 for pattern in self.patterns if pattern.search(answer))
        
        if danger_count > 0:
            checks["is_safe"] = False
            checks["risk_level"] = "high" if danger_count >= 2 else "medium"
            checks["warnings"].append(f"Found {danger_count} potentially dangerous claims")
            checks["requires_disclaimer"] = True
        
        # Check for disclaimers
        has_disclaimer = any(
            disclaimer.lower() in answer.lower() 
            for disclaimer in self.REQUIRED_DISCLAIMERS
        )
        
        if not has_disclaimer and checks["risk_level"] != "low":
            checks["requires_disclaimer"] = True
            checks["warnings"].append("Missing required disclaimer")
        
        # Check for absolute statements
        absolute_statements = re.findall(
            r'\b(always|never|all|none|every|guaranteed|definitely)\b',
            answer,
            re.IGNORECASE
        )
        
        if len(absolute_statements) > 3:
            checks["warnings"].append("Too many absolute statements")
            if checks["risk_level"] == "low":
                checks["risk_level"] = "medium"
        
        return checks
    
    def add_safety_disclaimer(
        self,
        answer: str,
        risk_level: str = "medium"
    ) -> str:
        """
        Add safety disclaimer to answer.
        
        Args:
            answer: Original answer
            risk_level: Risk level ("low", "medium", "high")
            
        Returns:
            Answer with disclaimer
        """
        if risk_level == "high":
            disclaimer = "\n\n⚠️ IMPORTANT: This information is for educational purposes only and does not constitute legal advice. Please consult a qualified legal professional for advice specific to your situation."
        elif risk_level == "medium":
            disclaimer = "\n\nNote: This is general information and not legal advice. Consult a lawyer for your specific case."
        else:
            disclaimer = "\n\nDisclaimer: This information is provided for informational purposes only."
        
        # Check if disclaimer already exists
        if any(d.lower() in answer.lower() for d in self.REQUIRED_DISCLAIMERS):
            return answer
        
        return answer + disclaimer
    
    def filter_unsafe_content(
        self,
        answer: str,
        question: str = None
    ) -> Tuple[str, Dict]:
        """
        Filter unsafe content from answer.
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Tuple of (filtered_answer, safety_checks)
        """
        safety_checks = self.check_safety(answer, question)
        
        filtered_answer = answer
        
        # Remove or modify dangerous claims
        if safety_checks["risk_level"] == "high":
            # Replace absolute statements
            filtered_answer = re.sub(
                r'\b(always|never|all|none|guaranteed|definitely)\b',
                'often',
                filtered_answer,
                flags=re.IGNORECASE
            )
        
        # Add disclaimer if needed
        if safety_checks["requires_disclaimer"]:
            filtered_answer = self.add_safety_disclaimer(
                filtered_answer,
                safety_checks["risk_level"]
            )
        
        return filtered_answer, safety_checks
    
    def validate_answer(
        self,
        answer: str,
        question: str,
        context: str
    ) -> Dict:
        """
        Comprehensive answer validation.
        
        Args:
            answer: Generated answer
            question: Original question
            context: Retrieved context
            
        Returns:
            Validation results
        """
        # Safety check
        safety_checks = self.check_safety(answer, question)
        
        # Citation check
        citations = self.citation_extractor.extract_citations(answer)
        has_citations = len(citations) > 0
        
        # Context consistency
        answer_lower = answer.lower()
        context_lower = context.lower()
        overlap = len(set(answer_lower.split()).intersection(set(context_lower.split())))
        consistency_score = overlap / len(answer_lower.split()) if answer_lower.split() else 0
        
        validation = {
            "safety": safety_checks,
            "has_citations": has_citations,
            "consistency_score": consistency_score,
            "is_valid": (
                safety_checks["is_safe"] and
                has_citations and
                consistency_score > 0.3
            )
        }
        
        return validation


if __name__ == "__main__":
    # Example usage
    safety = LegalSafetyLayer()
    
    # Unsafe answer
    unsafe_answer = "This will definitely win your case. Guaranteed success!"
    checks = safety.check_safety(unsafe_answer)
    filtered, _ = safety.filter_unsafe_content(unsafe_answer)
    
    print("Safety Check:")
    print(f"  Is Safe: {checks['is_safe']}")
    print(f"  Risk Level: {checks['risk_level']}")
    print(f"  Warnings: {checks['warnings']}")
    print(f"\nFiltered Answer:\n{filtered}")

