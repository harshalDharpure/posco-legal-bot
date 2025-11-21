"""
Anti-Hallucination Guardrails

Detects and prevents hallucinations in legal answers.
"""

import re
from typing import Dict, List, Tuple, Optional
from citation_extractor import LegalCitationExtractor


class AntiHallucinationGuard:
    """Guardrails to prevent hallucinations."""
    
    def __init__(self):
        """Initialize anti-hallucination guard."""
        self.citation_extractor = LegalCitationExtractor()
    
    def check_hallucination(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict:
        """
        Check if answer contains hallucinations.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Dictionary with hallucination check results
        """
        checks = {
            "has_citations": False,
            "citations_in_context": False,
            "factual_consistency": True,
            "hallucination_score": 0.0,
            "warnings": [],
            "is_safe": True
        }
        
        # Check 1: Does answer have citations?
        citations = self.citation_extractor.extract_citations(answer)
        checks["has_citations"] = len(citations) > 0
        
        if not checks["has_citations"]:
            checks["warnings"].append("Answer does not contain legal citations")
            checks["hallucination_score"] += 0.3
        
        # Check 2: Are citations present in context?
        if citations:
            context_citations = self.citation_extractor.extract_citations(context)
            context_citation_texts = {c.get('section', c.get('article', '')) for c in context_citations}
            
            answer_citation_texts = {c.get('section', c.get('article', '')) for c in citations}
            
            if answer_citation_texts:
                overlap = answer_citation_texts.intersection(context_citation_texts)
                if overlap:
                    checks["citations_in_context"] = True
                else:
                    checks["warnings"].append("Citations in answer not found in context")
                    checks["hallucination_score"] += 0.4
            else:
                checks["citations_in_context"] = True  # Case citations, etc.
        else:
            checks["citations_in_context"] = False
        
        # Check 3: Factual consistency
        consistency = self._check_factual_consistency(answer, context)
        checks["factual_consistency"] = consistency["consistent"]
        if not consistency["consistent"]:
            checks["warnings"].extend(consistency["warnings"])
            checks["hallucination_score"] += 0.3
        
        # Determine if safe
        checks["is_safe"] = (
            checks["has_citations"] and
            checks["citations_in_context"] and
            checks["factual_consistency"] and
            checks["hallucination_score"] < 0.5
        )
        
        return checks
    
    def _check_factual_consistency(
        self,
        answer: str,
        context: str
    ) -> Dict:
        """
        Check factual consistency between answer and context.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Consistency check results
        """
        # Extract key facts from answer
        answer_facts = self._extract_facts(answer)
        context_facts = self._extract_facts(context)
        
        # Check overlap
        consistent = True
        warnings = []
        
        # Check for contradictory information
        for fact in answer_facts:
            if fact not in context_facts:
                # Check if it contradicts context
                if self._contradicts(fact, context_facts):
                    consistent = False
                    warnings.append(f"Fact contradicts context: {fact}")
        
        return {
            "consistent": consistent,
            "warnings": warnings,
            "answer_facts": answer_facts,
            "context_facts": context_facts
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text."""
        facts = []
        
        # Extract numbers (sections, articles, years)
        numbers = re.findall(r'\d+', text)
        facts.extend([f"number_{n}" for n in numbers[:10]])  # Limit to first 10
        
        # Extract legal terms
        legal_terms = re.findall(
            r'(murder|theft|bail|arrest|punishment|imprisonment|fine|life|death)',
            text,
            re.IGNORECASE
        )
        facts.extend(legal_terms)
        
        return facts
    
    def _contradicts(self, fact: str, context_facts: List[str]) -> bool:
        """Check if fact contradicts context."""
        # Simple contradiction detection
        # In a real implementation, this would be more sophisticated
        return False
    
    def filter_unsafe_answer(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Tuple[str, Dict]:
        """
        Filter unsafe/hallucinated content from answer.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Tuple of (filtered_answer, check_results)
        """
        checks = self.check_hallucination(answer, context, question)
        
        if checks["is_safe"]:
            return answer, checks
        
        # Filter unsafe content
        filtered_answer = answer
        
        # Remove citations not in context
        if not checks["citations_in_context"]:
            citations = self.citation_extractor.extract_citations(answer)
            context_citations = self.citation_extractor.extract_citations(context)
            context_citation_texts = {c.get('section', c.get('article', '')) for c in context_citations}
            
            for cite in citations:
                cite_text = cite.get('section', cite.get('article', ''))
                if cite_text and cite_text not in context_citation_texts:
                    # Remove citation from answer
                    filtered_answer = filtered_answer.replace(cite['full_text'], '')
        
        # Add disclaimer if needed
        if checks["hallucination_score"] > 0.5:
            disclaimer = "\n\n[Note: This answer may contain unverified information. Please consult a legal expert for accurate advice.]"
            filtered_answer += disclaimer
        
        return filtered_answer, checks


if __name__ == "__main__":
    # Example usage
    guard = AntiHallucinationGuard()
    
    answer = "IPC Section 302 provides punishment for murder with death or life imprisonment."
    context = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    question = "What is the punishment for murder?"
    
    checks = guard.check_hallucination(answer, context, question)
    
    print("Hallucination Check Results:")
    print(f"  Has citations: {checks['has_citations']}")
    print(f"  Citations in context: {checks['citations_in_context']}")
    print(f"  Factual consistency: {checks['factual_consistency']}")
    print(f"  Hallucination score: {checks['hallucination_score']:.2f}")
    print(f"  Is safe: {checks['is_safe']}")
    if checks['warnings']:
        print(f"  Warnings: {checks['warnings']}")

