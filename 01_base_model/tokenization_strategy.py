"""
Multilingual Legal Tokenization Strategy

Handles tokenization for Hindi + English legal text with:
- Legal domain-specific tokens
- Multilingual support (Hindi + English)
- Special legal section markers
"""

from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import re
from pathlib import Path


class MultilingualLegalTokenizer:
    """Tokenization strategy for Hindi + English legal text."""
    
    # Legal section patterns
    IPC_PATTERN = r'IPC\s+Section\s+(\d+)'
    CrPC_PATTERN = r'CrPC\s+Section\s+(\d+)'
    CONSTITUTION_PATTERN = r'Article\s+(\d+)'
    SECTION_PATTERN = r'Section\s+(\d+)'
    
    # Special legal tokens
    LEGAL_TOKENS = [
        "<IPC>", "<CrPC>", "<Constitution>",
        "<Section>", "<Article>", "<Case>",
        "<Citation>", "<Act>", "<Rule>",
        "<Order>", "<Judgment>", "<Petition>"
    ]
    
    def __init__(
        self,
        model_name: str = "ai4bharat/IndicLegal-LLaMA-7B",
        add_legal_tokens: bool = True
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Base model name
            add_legal_tokens: Whether to add legal domain tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if add_legal_tokens:
            self._add_legal_tokens()
    
    def _add_legal_tokens(self):
        """Add legal domain-specific tokens to tokenizer."""
        # Get current vocabulary size
        current_vocab_size = len(self.tokenizer)
        
        # Add special tokens
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.LEGAL_TOKENS
        })
        
        print(f"Added {len(self.LEGAL_TOKENS)} legal tokens")
        print(f"Vocabulary size: {current_vocab_size} -> {len(self.tokenizer)}")
    
    def tokenize_legal_text(
        self,
        text: str,
        max_length: int = 4096,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict:
        """
        Tokenize legal text with special handling.
        
        Args:
            text: Input legal text
            max_length: Maximum sequence length
            padding: Whether to pad
            truncation: Whether to truncate
            return_tensors: Return format
            
        Returns:
            Tokenized output
        """
        # Pre-process: Mark legal sections
        text = self._mark_legal_sections(text)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=True
        )
        
        return encoded
    
    def _mark_legal_sections(self, text: str) -> str:
        """
        Mark legal sections with special tokens.
        
        Args:
            text: Input text
            
        Returns:
            Text with marked sections
        """
        # Mark IPC sections
        text = re.sub(
            self.IPC_PATTERN,
            r'<IPC> Section \1',
            text,
            flags=re.IGNORECASE
        )
        
        # Mark CrPC sections
        text = re.sub(
            self.CrPC_PATTERN,
            r'<CrPC> Section \1',
            text,
            flags=re.IGNORECASE
        )
        
        # Mark Constitution articles
        text = re.sub(
            self.CONSTITUTION_PATTERN,
            r'<Constitution> Article \1',
            text,
            flags=re.IGNORECASE
        )
        
        return text
    
    def decode_legal_text(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_legal_token_ids(self) -> Dict[str, int]:
        """Get mapping of legal tokens to IDs."""
        return {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in self.LEGAL_TOKENS
        }
    
    def analyze_tokenization(
        self,
        text: str,
        show_details: bool = True
    ) -> Dict:
        """
        Analyze tokenization of text.
        
        Args:
            text: Input text
            show_details: Whether to show detailed analysis
            
        Returns:
            Analysis dictionary
        """
        tokens = self.tokenize_legal_text(text, return_tensors=None)
        token_ids = tokens['input_ids']
        
        analysis = {
            "text_length": len(text),
            "num_tokens": len(token_ids),
            "tokens_per_char": len(token_ids) / len(text) if len(text) > 0 else 0,
            "token_ids": token_ids[:20] if show_details else None,  # First 20
            "decoded": self.decode_legal_text(token_ids)
        }
        
        if show_details:
            print(f"\nTokenization Analysis:")
            print(f"  Text length: {analysis['text_length']} characters")
            print(f"  Number of tokens: {analysis['num_tokens']}")
            print(f"  Tokens per character: {analysis['tokens_per_char']:.2f}")
            print(f"  First 20 token IDs: {analysis['token_ids']}")
        
        return analysis


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Legal Tokenization")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text file or text string"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ai4bharat/IndicLegal-LLaMA-7B",
        help="Model name"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed tokenization analysis"
    )
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = MultilingualLegalTokenizer(model_name=args.model)
    
    # Read input
    if Path(args.input).exists():
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.input
    
    # Tokenize
    encoded = tokenizer.tokenize_legal_text(text)
    
    print(f"\nInput text: {text[:100]}...")
    print(f"\nTokenized:")
    print(f"  Input IDs shape: {encoded['input_ids'].shape}")
    print(f"  Attention mask shape: {encoded['attention_mask'].shape}")
    
    if args.analyze:
        tokenizer.analyze_tokenization(text)


if __name__ == "__main__":
    main()

