"""
Data Augmentation for Multilingual Consistency

Augments dataset with:
- Paraphrasing
- Back-translation (Hindi ↔ English)
- Synonym replacement
- Question rephrasing
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse
import random
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


class LegalDataAugmenter:
    """Augments legal dataset for multilingual consistency."""
    
    # Paraphrasing templates
    PARAPHRASE_TEMPLATES = {
        'what': ['What is', 'Can you explain', 'Tell me about', 'Describe'],
        'explain': ['Explain', 'Elaborate on', 'Clarify', 'Describe'],
        'punishment': ['What is the punishment', 'What penalty', 'What sentence']
    }
    
    # Synonym replacements for legal terms
    LEGAL_SYNONYMS = {
        'punishment': ['penalty', 'sentence', 'penal consequence'],
        'section': ['provision', 'clause', 'article'],
        'law': ['statute', 'act', 'legislation'],
        'court': ['tribunal', 'judiciary', 'bench']
    }
    
    def __init__(self, augmentation_factor: float = 0.5):
        """
        Initialize augmenter.
        
        Args:
            augmentation_factor: Fraction of data to augment (0.0 to 1.0)
        """
        self.augmentation_factor = augmentation_factor
        random.seed(42)
    
    def augment_entry(self, entry: Dict) -> List[Dict]:
        """
        Augment a single entry.
        
        Args:
            entry: Original entry
            
        Returns:
            List of augmented entries
        """
        augmented = [entry]  # Include original
        
        # Paraphrase question
        paraphrased = self._paraphrase_question(entry)
        if paraphrased:
            augmented.append(paraphrased)
        
        # Add back-translation if available
        if 'hindi_question' in entry and 'question' in entry:
            back_translated = self._back_translate(entry)
            if back_translated:
                augmented.append(back_translated)
        
        # Synonym replacement
        synonym_replaced = self._replace_synonyms(entry)
        if synonym_replaced:
            augmented.append(synonym_replaced)
        
        return augmented
    
    def _paraphrase_question(self, entry: Dict) -> Dict:
        """Paraphrase the question."""
        question = entry.get('question', '')
        if not question:
            return None
        
        # Simple paraphrasing based on question start
        paraphrased = entry.copy()
        
        if question.startswith('What is'):
            paraphrased['question'] = question.replace('What is', random.choice(['Can you explain', 'Tell me about']), 1)
        elif question.startswith('What does'):
            paraphrased['question'] = question.replace('What does', random.choice(['Can you explain what', 'Tell me what']), 1)
        elif question.startswith('Explain'):
            paraphrased['question'] = question.replace('Explain', random.choice(['Elaborate on', 'Describe']), 1)
        else:
            return None
        
        paraphrased['augmented'] = True
        paraphrased['augmentation_type'] = 'paraphrase'
        
        return paraphrased
    
    def _back_translate(self, entry: Dict) -> Dict:
        """Create back-translated entry."""
        if 'hindi_question' not in entry or 'question' not in entry:
            return None
        
        # Create entry with Hindi as primary
        back_translated = entry.copy()
        back_translated['question'] = entry['hindi_question']
        back_translated['language'] = 'hi'
        if 'hindi_answer' in entry:
            back_translated['answer'] = entry.get('hindi_answer', entry['answer'])
        back_translated['augmented'] = True
        back_translated['augmentation_type'] = 'back_translation'
        
        return back_translated
    
    def _replace_synonyms(self, entry: Dict) -> Dict:
        """Replace synonyms in question/answer."""
        question = entry.get('question', '')
        answer = entry.get('answer', '')
        
        if not question and not answer:
            return None
        
        synonym_replaced = entry.copy()
        
        # Replace in question
        for word, synonyms in self.LEGAL_SYNONYMS.items():
            if word.lower() in question.lower():
                synonym = random.choice(synonyms)
                synonym_replaced['question'] = question.replace(word, synonym, 1)
                break
        
        # Replace in answer
        for word, synonyms in self.LEGAL_SYNONYMS.items():
            if word.lower() in answer.lower():
                synonym = random.choice(synonyms)
                synonym_replaced['answer'] = answer.replace(word, synonym, 1)
                break
        
        synonym_replaced['augmented'] = True
        synonym_replaced['augmentation_type'] = 'synonym_replacement'
        
        return synonym_replaced
    
    def augment_file(
        self,
        input_file: str,
        output_file: str
    ):
        """
        Augment dataset file.
        
        Args:
            input_file: Input dataset JSON file
            output_file: Output augmented dataset file
        """
        # Load dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        entries = dataset.get('entries', [])
        print(f"Original entries: {len(entries)}")
        
        # Augment
        augmented_entries = []
        num_to_augment = int(len(entries) * self.augmentation_factor)
        entries_to_augment = random.sample(entries, num_to_augment)
        
        for entry in entries:
            augmented_entries.append(entry)  # Always include original
            
            if entry in entries_to_augment:
                augmented = self.augment_entry(entry)
                # Add only new augmented entries (skip original)
                for aug_entry in augmented[1:]:
                    augmented_entries.append(aug_entry)
        
        print(f"Augmented entries: {len(augmented_entries)}")
        print(f"Augmentation factor: {len(augmented_entries) / len(entries):.2f}x")
        
        # Save augmented dataset
        augmented_dataset = {
            "dataset_name": dataset.get('dataset_name', 'Legal QA'),
            "split": dataset.get('split', 'train'),
            "original_entries": len(entries),
            "total_entries": len(augmented_entries),
            "entries": augmented_entries
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Saved augmented dataset to {output_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Data Augmentation")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output augmented dataset file"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Augmentation factor (0.0 to 1.0, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    augmenter = LegalDataAugmenter(augmentation_factor=args.factor)
    augmenter.augment_file(args.input, args.output)


if __name__ == "__main__":
    main()

