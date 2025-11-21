"""
Clause-Level Legal Extraction

Extracts legal clauses from segmented text:
- IPC sections
- CrPC sections
- Constitution articles
- Legal provisions
- Case citations
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import json


class LegalClauseExtractor:
    """Extracts legal clauses from text."""
    
    # Legal section patterns
    PATTERNS = {
        'ipc_section': re.compile(r'IPC\s+Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'crpc_section': re.compile(r'CrPC\s+Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'constitution_article': re.compile(r'Article\s+(\d+[A-Z]?)', re.IGNORECASE),
        'section': re.compile(r'Section\s+(\d+[A-Z]?)', re.IGNORECASE),
        'hindi_section': re.compile(r'धारा\s+(\d+[A-Z]?)', re.IGNORECASE),
        'hindi_article': re.compile(r'अनुच्छेद\s+(\d+[A-Z]?)', re.IGNORECASE),
        
        # Case citations
        'case_citation': re.compile(r'([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
        'citation_format': re.compile(r'(\d{4})\s+(AIR|SCC|SCR|SCALE|JT)\s+(\d+)', re.IGNORECASE),
    }
    
    def __init__(self):
        """Initialize clause extractor."""
        pass
    
    def extract_clauses(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract legal clauses from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted clauses by type
        """
        clauses = {
            'ipc_sections': [],
            'crpc_sections': [],
            'constitution_articles': [],
            'sections': [],
            'case_citations': [],
            'all_clauses': []
        }
        
        # Extract IPC sections
        for match in self.PATTERNS['ipc_section'].finditer(text):
            clauses['ipc_sections'].append({
                'type': 'IPC',
                'section': match.group(1),
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'context': self._get_context(text, match.start(), match.end())
            })
        
        # Extract CrPC sections
        for match in self.PATTERNS['crpc_section'].finditer(text):
            clauses['crpc_sections'].append({
                'type': 'CrPC',
                'section': match.group(1),
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'context': self._get_context(text, match.start(), match.end())
            })
        
        # Extract Constitution articles
        for match in self.PATTERNS['constitution_article'].finditer(text):
            clauses['constitution_articles'].append({
                'type': 'Constitution',
                'article': match.group(1),
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'context': self._get_context(text, match.start(), match.end())
            })
        
        # Extract generic sections
        for match in self.PATTERNS['section'].finditer(text):
            # Skip if already captured as IPC/CrPC
            if not any(match.start() >= c['start'] and match.end() <= c['end'] 
                      for c in clauses['ipc_sections'] + clauses['crpc_sections']):
                clauses['sections'].append({
                    'type': 'Section',
                    'section': match.group(1),
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'context': self._get_context(text, match.start(), match.end())
                })
        
        # Extract case citations
        for match in self.PATTERNS['case_citation'].finditer(text):
            clauses['case_citations'].append({
                'type': 'Case',
                'case_name': match.group(1),
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'context': self._get_context(text, match.start(), match.end())
            })
        
        # Combine all clauses
        clauses['all_clauses'] = (
            clauses['ipc_sections'] +
            clauses['crpc_sections'] +
            clauses['constitution_articles'] +
            clauses['sections'] +
            clauses['case_citations']
        )
        
        # Sort by position
        clauses['all_clauses'].sort(key=lambda x: x['start'])
        
        return clauses
    
    def _get_context(self, text: str, start: int, end: int, context_window: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end].strip()
    
    def extract_from_file(
        self,
        input_file: str,
        output_file: str
    ):
        """
        Extract clauses from file.
        
        Args:
            input_file: Input file path
            output_file: Output JSON file path
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        clauses = self.extract_clauses(text)
        
        # Add file metadata
        clauses['file_name'] = Path(input_file).stem
        clauses['total_clauses'] = len(clauses['all_clauses'])
        
        # Save results
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clauses, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {clauses['total_clauses']} clauses")
        print(f"  IPC sections: {len(clauses['ipc_sections'])}")
        print(f"  CrPC sections: {len(clauses['crpc_sections'])}")
        print(f"  Constitution articles: {len(clauses['constitution_articles'])}")
        print(f"  Case citations: {len(clauses['case_citations'])}")
    
    def extract_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.txt"
    ):
        """
        Extract clauses from all files in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        print(f"Found {len(files)} files to process")
        
        all_clauses = []
        for file in tqdm(files, desc="Extracting clauses"):
            output_file = Path(output_dir) / f"{file.stem}_clauses.json"
            self.extract_from_file(str(file), str(output_file))
            
            # Load and aggregate
            with open(output_file, 'r', encoding='utf-8') as f:
                clauses = json.load(f)
                all_clauses.append(clauses)
        
        # Save summary
        summary = {
            'total_files': len(all_clauses),
            'total_clauses': sum(c['total_clauses'] for c in all_clauses),
            'ipc_sections': sum(len(c['ipc_sections']) for c in all_clauses),
            'crpc_sections': sum(len(c['crpc_sections']) for c in all_clauses),
            'constitution_articles': sum(len(c['constitution_articles']) for c in all_clauses),
            'case_citations': sum(len(c['case_citations']) for c in all_clauses),
            'files': [c['file_name'] for c in all_clauses]
        }
        
        summary_path = Path(output_dir) / "clauses_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nExtraction complete!")
        print(f"Total clauses extracted: {summary['total_clauses']}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Clause Extraction")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file or directory"
    )
    
    args = parser.parse_args()
    
    extractor = LegalClauseExtractor()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        extractor.extract_from_file(str(input_path), args.output)
    elif input_path.is_dir():
        # Directory
        extractor.extract_from_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

