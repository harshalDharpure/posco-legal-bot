"""
Sentence Segmentation for Legal Text

Segments legal text into sentences with special handling for:
- Legal citations
- Section references
- Case names
- Hindi + English mixed text
"""

import re
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm


class LegalSentenceSegmenter:
    """Segments legal text into sentences."""
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = [
        r'[।!?]',  # Hindi danda, exclamation, question
        r'\.(?=\s+[A-Zअ-ह])',  # Period followed by capital letter
        r'\.(?=\s+\d)',  # Period followed by number (section reference)
    ]
    
    # Patterns that should NOT break sentences
    NO_BREAK_PATTERNS = [
        r'IPC\s+Section\s+\d+\.',  # IPC Section 302.
        r'CrPC\s+Section\s+\d+\.',  # CrPC Section 438.
        r'Article\s+\d+\.',  # Article 21.
        r'Section\s+\d+\.',  # Section 302.
        r'v\.\s+',  # v. (versus in case names)
        r'vs\.\s+',  # vs. (versus)
        r'etc\.',  # etc.
        r'e\.g\.',  # e.g.
        r'i\.e\.',  # i.e.
        r'Dr\.',  # Dr.
        r'Mr\.',  # Mr.
        r'Mrs\.',  # Mrs.
        r'Prof\.',  # Prof.
    ]
    
    def __init__(self):
        """Initialize sentence segmenter."""
        self.sentence_end_re = re.compile('|'.join(self.SENTENCE_ENDINGS))
        self.no_break_re = re.compile('|'.join(self.NO_BREAK_PATTERNS))
    
    def segment(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not text:
            return []
        
        # Split by sentence endings
        sentences = []
        current_sentence = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            # Check if this is a sentence ending
            if self.sentence_end_re.match(char):
                # Check if this should NOT break (e.g., "IPC Section 302.")
                lookahead = text[i:i+20]
                if not self.no_break_re.search(lookahead):
                    # This is a real sentence break
                    sentence = current_sentence.strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = ""
            
            i += 1
        
        # Add remaining text as last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Filter out very short sentences (likely errors)
        sentences = [s for s in sentences if len(s) > 10]
        
        return sentences
    
    def segment_file(
        self,
        input_file: str,
        output_file: str,
        min_length: int = 10
    ):
        """
        Segment file into sentences.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            min_length: Minimum sentence length
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sentences = self.segment(text)
        
        # Filter by minimum length
        sentences = [s for s in sentences if len(s) >= min_length]
        
        # Save sentences (one per line)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        print(f"Segmented into {len(sentences)} sentences")
    
    def segment_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.txt"
    ):
        """
        Segment all files in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        print(f"Found {len(files)} files to segment")
        
        total_sentences = 0
        for file in tqdm(files, desc="Segmenting files"):
            output_file = Path(output_dir) / file.name
            self.segment_file(str(file), str(output_file))
            
            # Count sentences
            with open(output_file, 'r', encoding='utf-8') as f:
                total_sentences += len(f.readlines())
        
        print(f"\nSegmented {len(files)} files")
        print(f"Total sentences: {total_sentences}")
        print(f"Average sentences per file: {total_sentences / len(files) if files else 0:.1f}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Sentence Segmentation")
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
        "--min-length",
        type=int,
        default=10,
        help="Minimum sentence length (default: 10)"
    )
    
    args = parser.parse_args()
    
    segmenter = LegalSentenceSegmenter()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        segmenter.segment_file(str(input_path), args.output, args.min_length)
    elif input_path.is_dir():
        # Directory
        segmenter.segment_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

