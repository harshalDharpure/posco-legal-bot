"""
Legal Vocabulary Adaptation

Adapts base model vocabulary for legal domain by:
- Adding legal terminology
- Expanding IPC/CrPC/Constitution sections
- Adding case law citations
- Multilingual legal terms (Hindi + English)
"""

import json
from pathlib import Path
from typing import List, Dict, Set
from transformers import AutoTokenizer
import yaml


class LegalVocabularyAdapter:
    """Adapts vocabulary for legal domain."""
    
    def __init__(
        self,
        base_tokenizer: AutoTokenizer,
        legal_terms_file: str = "data/legal_terms.txt",
        ipc_sections: bool = True,
        crpc_sections: bool = True,
        constitution_articles: bool = True
    ):
        """
        Initialize vocabulary adapter.
        
        Args:
            base_tokenizer: Base model tokenizer
            legal_terms_file: Path to legal terms file
            ipc_sections: Whether to add IPC sections
            crpc_sections: Whether to add CrPC sections
            constitution_articles: Whether to add Constitution articles
        """
        self.tokenizer = base_tokenizer
        self.legal_terms_file = legal_terms_file
        self.ipc_sections = ipc_sections
        self.crpc_sections = crpc_sections
        self.constitution_articles = constitution_articles
        
        # Legal vocabulary components
        self.legal_terms: Set[str] = set()
        self.ipc_vocab: Set[str] = set()
        self.crpc_vocab: Set[str] = set()
        self.constitution_vocab: Set[str] = set()
        self.case_law_vocab: Set[str] = set()
    
    def load_legal_terms(self) -> Set[str]:
        """Load legal terms from file."""
        legal_terms = set()
        
        if Path(self.legal_terms_file).exists():
            with open(self.legal_terms_file, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip()
                    if term:
                        legal_terms.add(term)
        else:
            # Generate default legal terms
            legal_terms = self._generate_default_legal_terms()
        
        self.legal_terms = legal_terms
        return legal_terms
    
    def _generate_default_legal_terms(self) -> Set[str]:
        """Generate default legal terms if file doesn't exist."""
        terms = {
            # IPC Terms
            "murder", "culpable homicide", "theft", "robbery", "dacoity",
            "assault", "rape", "kidnapping", "extortion", "cheating",
            "forgery", "criminal breach of trust", "misappropriation",
            
            # CrPC Terms
            "bail", "anticipatory bail", "arrest", "remand", "custody",
            "charge sheet", "FIR", "investigation", "trial", "judgment",
            "appeal", "revision", "writ petition",
            
            # Constitution Terms
            "fundamental rights", "directive principles", "constitutional remedy",
            "writ of habeas corpus", "writ of mandamus", "writ of certiorari",
            "writ of prohibition", "writ of quo warranto",
            
            # Hindi Legal Terms (Romanized)
            "homicide", "qatl", "chori", "dakaiti", "balatkar",
            "apaharan", "zabardasti", "dhoka", "jali", "bewafaai",
            
            # General Legal Terms
            "plaintiff", "defendant", "petitioner", "respondent",
            "appellant", "respondent", "complainant", "accused",
            "witness", "evidence", "testimony", "cross-examination",
            "prosecution", "defense", "counsel", "advocate", "judge",
            "magistrate", "court", "tribunal", "high court", "supreme court"
        }
        return terms
    
    def generate_ipc_vocabulary(self) -> Set[str]:
        """Generate IPC section vocabulary."""
        ipc_vocab = set()
        
        # IPC sections 1-511
        for section in range(1, 512):
            ipc_vocab.add(f"IPC Section {section}")
            ipc_vocab.add(f"Section {section} IPC")
            ipc_vocab.add(f"IPC {section}")
        
        # Common IPC sections with descriptions
        common_sections = {
            302: "punishment for murder",
            304: "punishment for culpable homicide not amounting to murder",
            307: "attempt to murder",
            376: "punishment for rape",
            420: "cheating and dishonestly inducing delivery of property",
            498A: "husband or relative of husband of a woman subjecting her to cruelty"
        }
        
        for section, desc in common_sections.items():
            ipc_vocab.add(f"IPC Section {section} {desc}")
        
        self.ipc_vocab = ipc_vocab
        return ipc_vocab
    
    def generate_crpc_vocabulary(self) -> Set[str]:
        """Generate CrPC section vocabulary."""
        crpc_vocab = set()
        
        # CrPC sections 1-484
        for section in range(1, 485):
            crpc_vocab.add(f"CrPC Section {section}")
            crpc_vocab.add(f"Section {section} CrPC")
            crpc_vocab.add(f"CrPC {section}")
        
        # Common CrPC sections
        common_sections = {
            41: "when police may arrest without warrant",
            438: "direction for grant of bail to person apprehending arrest",
            439: "special powers of high court or court of session regarding bail",
            482: "saving of inherent powers of high court"
        }
        
        for section, desc in common_sections.items():
            crpc_vocab.add(f"CrPC Section {section} {desc}")
        
        self.crpc_vocab = crpc_vocab
        return crpc_vocab
    
    def generate_constitution_vocabulary(self) -> Set[str]:
        """Generate Constitution article vocabulary."""
        constitution_vocab = set()
        
        # Articles 1-395
        for article in range(1, 396):
            constitution_vocab.add(f"Article {article}")
            constitution_vocab.add(f"Constitution Article {article}")
            constitution_vocab.add(f"Art. {article}")
        
        # Fundamental Rights (Part III)
        fundamental_rights = {
            14: "equality before law",
            15: "prohibition of discrimination",
            19: "protection of certain rights regarding freedom of speech",
            21: "protection of life and personal liberty",
            32: "remedies for enforcement of rights"
        }
        
        for article, desc in fundamental_rights.items():
            constitution_vocab.add(f"Article {article} {desc}")
        
        self.constitution_vocab = constitution_vocab
        return constitution_vocab
    
    def generate_case_law_vocabulary(self) -> Set[str]:
        """Generate case law citation vocabulary."""
        case_vocab = set()
        
        # Common case citation patterns
        patterns = [
            "AIR", "SCC", "SCR", "SCALE", "JT", "BLJR",
            "All India Reporter", "Supreme Court Cases", "Supreme Court Reports"
        ]
        
        # Example landmark cases
        landmark_cases = [
            "Kesavananda Bharati v. State of Kerala",
            "Maneka Gandhi v. Union of India",
            "Vishaka v. State of Rajasthan",
            "Shayara Bano v. Union of India"
        ]
        
        for pattern in patterns:
            case_vocab.add(pattern)
        
        for case in landmark_cases:
            case_vocab.add(case)
            # Add year variations
            for year in range(1950, 2024):
                case_vocab.add(f"{case} {year}")
        
        self.case_law_vocab = case_vocab
        return case_vocab
    
    def build_legal_vocabulary(self) -> Set[str]:
        """Build complete legal vocabulary."""
        vocab = set()
        
        # Load legal terms
        vocab.update(self.load_legal_terms())
        
        # Add IPC sections
        if self.ipc_sections:
            vocab.update(self.generate_ipc_vocabulary())
        
        # Add CrPC sections
        if self.crpc_sections:
            vocab.update(self.generate_crpc_vocabulary())
        
        # Add Constitution articles
        if self.constitution_articles:
            vocab.update(self.generate_constitution_vocabulary())
        
        # Add case law
        vocab.update(self.generate_case_law_vocabulary())
        
        return vocab
    
    def adapt_tokenizer(self, vocab_size: int = 50000) -> AutoTokenizer:
        """
        Adapt tokenizer with legal vocabulary.
        
        Args:
            vocab_size: Target vocabulary size
            
        Returns:
            Adapted tokenizer
        """
        # Build legal vocabulary
        legal_vocab = self.build_legal_vocabulary()
        
        print(f"Built legal vocabulary: {len(legal_vocab)} terms")
        print(f"Current vocab size: {len(self.tokenizer)}")
        
        # Add special tokens for legal domain
        special_tokens = [
            "<IPC>", "<CrPC>", "<Constitution>",
            "<Section>", "<Article>", "<Case>",
            "<Citation>", "<Act>", "<Rule>"
        ]
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        
        print(f"Added {len(special_tokens)} special tokens")
        print(f"New vocab size: {len(self.tokenizer)}")
        
        return self.tokenizer
    
    def save_vocabulary(self, output_file: str = "data/legal_vocabulary.json"):
        """Save legal vocabulary to file."""
        vocab_dict = {
            "legal_terms": list(self.legal_terms),
            "ipc_vocab": list(self.ipc_vocab),
            "crpc_vocab": list(self.crpc_vocab),
            "constitution_vocab": list(self.constitution_vocab),
            "case_law_vocab": list(self.case_law_vocab)
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Saved vocabulary to {output_file}")


def main():
    """Example usage."""
    import argparse
    from model_selection import LegalModelSelector
    
    parser = argparse.ArgumentParser(description="Legal Vocabulary Adaptation")
    parser.add_argument(
        "--model",
        type=str,
        default="indiclegal-llama",
        help="Base model key"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/legal_vocabulary.json",
        help="Output vocabulary file"
    )
    
    args = parser.parse_args()
    
    # Load base model and tokenizer
    selector = LegalModelSelector()
    _, tokenizer = selector.load_model(model_key=args.model, use_quantization=False)
    
    # Adapt vocabulary
    adapter = LegalVocabularyAdapter(tokenizer)
    adapted_tokenizer = adapter.adapt_tokenizer(vocab_size=args.vocab_size)
    
    # Save vocabulary
    adapter.save_vocabulary(args.output)
    
    print("\nVocabulary adaptation complete!")


if __name__ == "__main__":
    main()

