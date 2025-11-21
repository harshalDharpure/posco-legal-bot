"""
OCR Noise Correction and Cleaning

Cleans OCR output by:
- Removing common OCR errors
- Fixing character recognition mistakes
- Normalizing whitespace
- Correcting Hindi-specific OCR errors
"""

import re
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm


class OCRCleaner:
    """Cleans OCR output for legal documents."""
    
    # Common OCR error patterns (Hindi + English)
    OCR_ERRORS = {
        # Common character confusions
        r'0': 'O',  # Zero to O (context-dependent)
        r'1': 'I',  # One to I (context-dependent)
        r'5': 'S',  # Five to S (context-dependent)
        r'8': 'B',  # Eight to B (context-dependent)
        
        # Hindi-specific common errors
        r'अा': 'आ',  # अ + ा → आ
        r'इि': 'ई',  # इ + ि → ई
        r'उु': 'ऊ',  # उ + ु → ऊ
        r'एे': 'ऐ',  # ए + े → ऐ
        r'ओो': 'औ',  # ओ + ो → औ
        
        # Spacing issues
        r'([क-ह])([ा-ौ])': r'\1\2',  # Fix matra spacing
        r'([ा-ौ])\s+([क-ह])': r'\1\2',  # Remove space between matra and consonant
    }
    
    # Legal section patterns to preserve
    LEGAL_PATTERNS = [
        r'IPC\s+Section\s+\d+',
        r'CrPC\s+Section\s+\d+',
        r'Article\s+\d+',
        r'Section\s+\d+',
        r'धारा\s+\d+',
        r'अनुच्छेद\s+\d+'
    ]
    
    def __init__(self):
        """Initialize OCR cleaner."""
        self.error_patterns = [re.compile(pattern) for pattern in self.OCR_ERRORS.keys()]
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove line breaks in middle of sentences
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
        
        # Fix common OCR errors
        for pattern, replacement in self.OCR_ERRORS.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix Hindi matra spacing
        text = self._fix_hindi_spacing(text)
        
        # Remove special characters that are likely OCR errors
        text = re.sub(r'[^\w\s।,;:!?\-\(\)\[\]{}"\']', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        # Fix multiple periods
        text = re.sub(r'\.{3,}', '...', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _fix_hindi_spacing(self, text: str) -> str:
        """Fix spacing issues in Hindi text."""
        # Fix matra spacing (vowel marks)
        text = re.sub(r'([क-ह])\s+([ा-ौ])', r'\1\2', text)
        text = re.sub(r'([ा-ौ])\s+([क-ह])', r'\1\2', text)
        
        # Fix halant (्) spacing
        text = re.sub(r'([क-ह])\s+्\s+([क-ह])', r'\1्\2', text)
        
        return text
    
    def clean_file(
        self,
        input_file: str,
        output_file: str,
        preserve_structure: bool = True
    ):
        """
        Clean OCR output file.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            preserve_structure: Whether to preserve document structure
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if preserve_structure:
            # Split by paragraphs
            paragraphs = text.split('\n\n')
            cleaned_paragraphs = [self.clean_text(p) for p in paragraphs]
            cleaned_text = '\n\n'.join(cleaned_paragraphs)
        else:
            cleaned_text = self.clean_text(text)
        
        # Save cleaned text
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
    
    def clean_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.txt"
    ):
        """
        Clean all files in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        print(f"Found {len(files)} files to clean")
        
        for file in tqdm(files, desc="Cleaning files"):
            output_file = Path(output_dir) / file.name
            self.clean_file(str(file), str(output_file))
        
        print(f"Cleaned {len(files)} files")
        print(f"Output saved to {output_dir}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="OCR Cleaning Pipeline")
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
    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="Don't preserve document structure"
    )
    
    args = parser.parse_args()
    
    cleaner = OCRCleaner()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        cleaner.clean_file(
            str(input_path),
            args.output,
            preserve_structure=not args.no_preserve_structure
        )
        print(f"Cleaned file saved to {args.output}")
    elif input_path.is_dir():
        # Directory
        cleaner.clean_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

