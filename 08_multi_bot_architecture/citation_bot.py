"""
Citation-Bot: Adds Legal Citations

Extracts and adds IPC/CrPC/Constitution sections to answers.
"""

from citation_extractor import LegalCitationExtractor
from typing import Dict, List
import re


class CitationBot:
    """Citation extraction and addition bot."""
    
    def __init__(self):
        """Initialize Citation-Bot."""
        self.citation_extractor = LegalCitationExtractor()
    
    def add_citations(
        self,
        answer: str,
        context: str = None
    ) -> Dict:
        """
        Add citations to answer.
        
        Args:
            answer: Original answer
            context: Optional context for citation validation
            
        Returns:
            Dictionary with enhanced answer and citations
        """
        # Extract existing citations
        citations, formatted = self.citation_extractor.extract_and_format(answer)
        
        # If no citations found, try to infer from context
        if not citations and context:
            context_citations, _ = self.citation_extractor.extract_and_format(context)
            if context_citations:
                # Add relevant citations to answer
                relevant_cites = context_citations[:3]  # Top 3
                cite_text = self.citation_extractor.format_citations(relevant_cites)
                answer = f"{answer}\n\nRelevant Legal Sections: {cite_text}"
                citations = relevant_cites
                formatted = cite_text
        
        # Enhance answer with citations if missing
        if not citations:
            # Try to find section numbers mentioned without proper citation
            section_numbers = re.findall(r'\b(\d{3,4})\b', answer)
            if section_numbers:
                # Assume IPC sections (can be enhanced with context)
                citations = [
                    {"type": "IPC", "section": num, "full_text": f"IPC Section {num}"}
                    for num in section_numbers[:3]
                ]
                formatted = ", ".join([f"IPC Section {c['section']}" for c in citations])
                answer = f"{answer}\n\nCitations: {formatted}"
        
        return {
            "answer": answer,
            "citations": formatted,
            "citation_details": citations,
            "bot": "Citation-Bot"
        }
    
    def validate_citations(
        self,
        answer: str,
        context: str
    ) -> Dict:
        """
        Validate citations against context.
        
        Args:
            answer: Answer with citations
            context: Retrieved context
            
        Returns:
            Validation results
        """
        answer_citations = self.citation_extractor.extract_citations(answer)
        context_citations = self.citation_extractor.extract_citations(context)
        
        answer_cite_texts = {
            c.get('section', c.get('article', '')) for c in answer_citations
        }
        context_cite_texts = {
            c.get('section', c.get('article', '')) for c in context_citations
        }
        
        matches = answer_cite_texts.intersection(context_cite_texts)
        accuracy = len(matches) / len(answer_cite_texts) if answer_cite_texts else 0.0
        
        return {
            "accuracy": accuracy,
            "matches": len(matches),
            "total_citations": len(answer_citations),
            "is_valid": accuracy >= 0.5
        }


if __name__ == "__main__":
    # Example usage
    bot = CitationBot()
    
    answer = "The punishment for murder is death or life imprisonment."
    context = "IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life."
    
    result = bot.add_citations(answer, context)
    print(f"Enhanced Answer:\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")

