"""
Legal Citation Extraction Logic

Extracts legal citations (IPC, CrPC, Constitution, Cases) from text.
"""

import re
from typing import List, Dict, Tuple


class LegalCitationExtractor:
    """Extracts legal citations from text."""
    
    # Citation patterns
    PATTERNS = {
        'ipc_section': re.compile(r'IPC\s+Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'crpc_section': re.compile(r'CrPC\s+Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'constitution_article': re.compile(r'Article\s+(\d+[A-Z]?)\s+of\s+the\s+Constitution', re.IGNORECASE),
        'article': re.compile(r'Article\s+(\d+[A-Z]?)', re.IGNORECASE),
        'section': re.compile(r'Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'hindi_section': re.compile(r'धारा\s+(\d+[A-Z]?)', re.IGNORECASE),
        'hindi_article': re.compile(r'अनुच्छेद\s+(\d+[A-Z]?)', re.IGNORECASE),
        
        # Case citations
        'case_citation': re.compile(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            re.IGNORECASE
        ),
        'citation_format': re.compile(
            r'(\d{4})\s+(AIR|SCC|SCR|SCALE|JT|BLJR)\s+(\d+)',
            re.IGNORECASE
        ),
    }
    
    def __init__(self):
        """Initialize citation extractor."""
        pass
    
    def extract_citations(self, text: str) -> List[Dict]:
        """
        Extract all citations from text.
        
        Args:
            text: Input text
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Extract IPC sections
        for match in self.PATTERNS['ipc_section'].finditer(text):
            citations.append({
                'type': 'IPC',
                'section': match.group(1),
                'full_text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract CrPC sections
        for match in self.PATTERNS['crpc_section'].finditer(text):
            citations.append({
                'type': 'CrPC',
                'section': match.group(1),
                'full_text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract Constitution articles
        for match in self.PATTERNS['constitution_article'].finditer(text):
            citations.append({
                'type': 'Constitution',
                'article': match.group(1),
                'full_text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract generic articles (if not already captured)
        for match in self.PATTERNS['article'].finditer(text):
            # Check if already captured as Constitution article
            if not any(c['start'] <= match.start() <= c['end'] 
                      for c in citations if c['type'] == 'Constitution'):
                citations.append({
                    'type': 'Article',
                    'article': match.group(1),
                    'full_text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Extract case citations
        for match in self.PATTERNS['case_citation'].finditer(text):
            citations.append({
                'type': 'Case',
                'case_name': f"{match.group(1)} v. {match.group(2)}",
                'full_text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract citation formats (e.g., "2023 AIR 123")
        for match in self.PATTERNS['citation_format'].finditer(text):
            citations.append({
                'type': 'Citation',
                'year': match.group(1),
                'reporter': match.group(2),
                'page': match.group(3),
                'full_text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Sort by position
        citations.sort(key=lambda x: x['start'])
        
        return citations
    
    def format_citations(self, citations: List[Dict]) -> str:
        """
        Format citations as string.
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Formatted citation string
        """
        formatted = []
        
        for cite in citations:
            if cite['type'] == 'IPC':
                formatted.append(f"IPC Section {cite['section']}")
            elif cite['type'] == 'CrPC':
                formatted.append(f"CrPC Section {cite['section']}")
            elif cite['type'] == 'Constitution':
                formatted.append(f"Constitution Article {cite['article']}")
            elif cite['type'] == 'Case':
                formatted.append(f"Case: {cite['case_name']}")
            elif cite['type'] == 'Citation':
                formatted.append(f"{cite['year']} {cite['reporter']} {cite['page']}")
        
        return ", ".join(formatted)
    
    def extract_and_format(self, text: str) -> Tuple[List[Dict], str]:
        """
        Extract citations and format them.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (citations list, formatted string)
        """
        citations = self.extract_citations(text)
        formatted = self.format_citations(citations)
        
        return citations, formatted


if __name__ == "__main__":
    # Example usage
    extractor = LegalCitationExtractor()
    
    text = """
    IPC Section 302 deals with murder. According to CrPC Section 438, 
    anticipatory bail can be granted. Article 21 of the Constitution 
    guarantees right to life. The case of Kesavananda Bharati v. State of Kerala 
    is a landmark judgment.
    """
    
    citations, formatted = extractor.extract_and_format(text)
    
    print("Extracted Citations:")
    for cite in citations:
        print(f"  {cite['type']}: {cite.get('section', cite.get('article', cite.get('case_name', cite.get('full_text'))))}")
    
    print(f"\nFormatted: {formatted}")

