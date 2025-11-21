"""
Dataset Builder: Convert Legal Text to QA Format

Converts extracted legal clauses into:
Question → Answer → Context → Legal Section format
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import re


class LegalDatasetBuilder:
    """Builds legal QA dataset from extracted clauses."""
    
    # Question templates
    QUESTION_TEMPLATES = {
        'ipc': [
            "What does IPC Section {section} say?",
            "What is IPC Section {section}?",
            "Explain IPC Section {section}.",
            "What is the punishment under IPC Section {section}?",
            "IPC Section {section} के बारे में बताएं।",
            "IPC धारा {section} क्या कहती है?",
        ],
        'crpc': [
            "What does CrPC Section {section} say?",
            "What is CrPC Section {section}?",
            "Explain CrPC Section {section}.",
            "CrPC Section {section} के बारे में बताएं।",
            "CrPC धारा {section} क्या कहती है?",
        ],
        'constitution': [
            "What does Article {article} of the Constitution say?",
            "What is Article {article}?",
            "Explain Article {article} of the Constitution.",
            "संविधान के अनुच्छेद {article} के बारे में बताएं।",
            "संविधान का अनुच्छेद {article} क्या कहता है?",
        ],
        'general': [
            "What does Section {section} say?",
            "Explain Section {section}.",
            "धारा {section} के बारे में बताएं।",
        ]
    }
    
    def __init__(self):
        """Initialize dataset builder."""
        pass
    
    def build_qa_from_clause(
        self,
        clause: Dict,
        context: str,
        template_type: str = 'ipc'
    ) -> List[Dict]:
        """
        Build QA pairs from a legal clause.
        
        Args:
            clause: Extracted clause dictionary
            context: Context text around the clause
            template_type: Type of question template to use
            
        Returns:
            List of QA entries
        """
        entries = []
        
        # Get section/article number
        section_num = clause.get('section') or clause.get('article', '')
        if not section_num:
            return entries
        
        # Get templates
        templates = self.QUESTION_TEMPLATES.get(template_type, self.QUESTION_TEMPLATES['general'])
        
        # Build QA pairs
        for template in templates:
            # Determine language
            is_hindi = any(char in template for char in 'कखगघ')
            
            question = template.format(section=section_num, article=section_num)
            answer = clause.get('context', context)
            
            entry = {
                "question": question,
                "answer": answer,
                "context": context,
                "legal_section": f"{clause['type']} {section_num}",
                "language": "hi" if is_hindi else "en",
                "section_type": clause['type'],
                "section_number": section_num
            }
            
            # Add Hindi translation if English question
            if not is_hindi:
                entry["hindi_question"] = self._translate_to_hindi_template(template, section_num)
            
            # Add English translation if Hindi question
            if is_hindi:
                entry["english_question"] = self._translate_to_english_template(template, section_num)
            
            entries.append(entry)
        
        return entries
    
    def _translate_to_hindi_template(self, template: str, section_num: str) -> str:
        """Translate English template to Hindi."""
        translations = {
            "What does IPC Section {section} say?": "IPC धारा {section} क्या कहती है?",
            "What is IPC Section {section}?": "IPC धारा {section} क्या है?",
            "Explain IPC Section {section}.": "IPC धारा {section} समझाएं।",
        }
        return translations.get(template, template).format(section=section_num)
    
    def _translate_to_english_template(self, template: str, section_num: str) -> str:
        """Translate Hindi template to English."""
        translations = {
            "IPC धारा {section} क्या कहती है?": "What does IPC Section {section} say?",
            "IPC धारा {section} क्या है?": "What is IPC Section {section}?",
        }
        return translations.get(template, template).format(section=section_num)
    
    def build_from_clauses_file(
        self,
        clauses_file: str,
        output_file: str
    ):
        """
        Build dataset from clauses JSON file.
        
        Args:
            clauses_file: Input clauses JSON file
            output_file: Output dataset JSON file
        """
        with open(clauses_file, 'r', encoding='utf-8') as f:
            clauses_data = json.load(f)
        
        all_entries = []
        
        # Process IPC sections
        for clause in clauses_data.get('ipc_sections', []):
            context = clause.get('context', '')
            entries = self.build_qa_from_clause(clause, context, 'ipc')
            all_entries.extend(entries)
        
        # Process CrPC sections
        for clause in clauses_data.get('crpc_sections', []):
            context = clause.get('context', '')
            entries = self.build_qa_from_clause(clause, context, 'crpc')
            all_entries.extend(entries)
        
        # Process Constitution articles
        for clause in clauses_data.get('constitution_articles', []):
            context = clause.get('context', '')
            entries = self.build_qa_from_clause(clause, context, 'constitution')
            all_entries.extend(entries)
        
        # Save dataset
        dataset = {
            "dataset_name": "Multilingual Legal QA",
            "total_entries": len(all_entries),
            "entries": all_entries
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Built {len(all_entries)} QA entries")
        print(f"Saved to {output_file}")
    
    def build_from_directory(
        self,
        input_dir: str,
        output_file: str,
        pattern: str = "*_clauses.json"
    ):
        """
        Build dataset from directory of clause files.
        
        Args:
            input_dir: Input directory
            output_file: Output dataset file
            pattern: File pattern to match
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        print(f"Found {len(files)} clause files")
        
        all_entries = []
        for file in tqdm(files, desc="Building dataset"):
            with open(file, 'r', encoding='utf-8') as f:
                clauses_data = json.load(f)
            
            # Process all clause types
            for clause_type in ['ipc_sections', 'crpc_sections', 'constitution_articles']:
                for clause in clauses_data.get(clause_type, []):
                    context = clause.get('context', '')
                    template_type = clause_type.replace('_sections', '').replace('_articles', '')
                    entries = self.build_qa_from_clause(clause, context, template_type)
                    all_entries.extend(entries)
        
        # Save combined dataset
        dataset = {
            "dataset_name": "Multilingual Legal QA",
            "total_entries": len(all_entries),
            "entries": all_entries
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\nBuilt {len(all_entries)} QA entries from {len(files)} files")
        print(f"Saved to {output_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Dataset Builder")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input clauses file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset JSON file"
    )
    
    args = parser.parse_args()
    
    builder = LegalDatasetBuilder()
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        builder.build_from_clauses_file(str(input_path), args.output)
    elif input_path.is_dir():
        # Directory
        builder.build_from_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

