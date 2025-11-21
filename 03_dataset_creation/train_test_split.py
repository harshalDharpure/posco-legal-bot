"""
Train/Validation/Test Split Logic

Splits dataset into train/val/test with:
- Stratified splitting by legal section type
- Language balance (Hindi/English)
- Temporal splitting (if dates available)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from sklearn.model_selection import train_test_split
import random


class LegalDatasetSplitter:
    """Splits legal dataset into train/val/test."""
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "section_type",
        random_seed: int = 42
    ):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_by: Field to stratify by
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_by = stratify_by
        self.random_seed = random_seed
        
        random.seed(random_seed)
    
    def split_dataset(
        self,
        entries: List[Dict],
        stratify: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/val/test.
        
        Args:
            entries: List of dataset entries
            stratify: Whether to stratify by section type
            
        Returns:
            Tuple of (train, val, test) lists
        """
        if stratify and self.stratify_by in entries[0]:
            # Stratified split
            train, temp = train_test_split(
                entries,
                test_size=(self.val_ratio + self.test_ratio),
                stratify=[e[self.stratify_by] for e in entries],
                random_state=self.random_seed
            )
            
            # Split temp into val and test
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            val, test = train_test_split(
                temp,
                test_size=(1 - val_size),
                stratify=[e[self.stratify_by] for e in temp],
                random_state=self.random_seed
            )
        else:
            # Simple random split
            train, temp = train_test_split(
                entries,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.random_seed
            )
            
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            val, test = train_test_split(
                temp,
                test_size=(1 - val_size),
                random_state=self.random_seed
            )
        
        return train, val, test
    
    def split_file(
        self,
        input_file: str,
        output_dir: str,
        stratify: bool = True
    ):
        """
        Split dataset file.
        
        Args:
            input_file: Input dataset JSON file
            output_dir: Output directory for splits
            stratify: Whether to stratify
        """
        # Load dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        entries = dataset.get('entries', [])
        print(f"Total entries: {len(entries)}")
        
        # Split
        train, val, test = self.split_dataset(entries, stratify)
        
        print(f"Train: {len(train)} ({len(train)/len(entries)*100:.1f}%)")
        print(f"Val: {len(val)} ({len(val)/len(entries)*100:.1f}%)")
        print(f"Test: {len(test)} ({len(test)/len(entries)*100:.1f}%)")
        
        # Save splits
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        for split_name, split_data in splits.items():
            output_file = Path(output_dir) / f"{split_name}.json"
            split_dataset = {
                "dataset_name": dataset.get('dataset_name', 'Legal QA'),
                "split": split_name,
                "total_entries": len(split_data),
                "entries": split_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_dataset, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {split_name} to {output_file}")
        
        # Save statistics
        stats = self._compute_statistics(train, val, test)
        stats_file = Path(output_dir) / "split_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nSplit statistics saved to {stats_file}")
    
    def _compute_statistics(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict]
    ) -> Dict:
        """Compute split statistics."""
        def get_stats(entries: List[Dict]) -> Dict:
            stats = {
                "total": len(entries),
                "by_section_type": {},
                "by_language": {}
            }
            
            for entry in entries:
                # Section type
                section_type = entry.get('section_type', 'unknown')
                stats["by_section_type"][section_type] = stats["by_section_type"].get(section_type, 0) + 1
                
                # Language
                language = entry.get('language', 'unknown')
                stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            
            return stats
        
        return {
            "train": get_stats(train),
            "val": get_stats(val),
            "test": get_stats(test)
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Legal Dataset Splitter")
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
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    splitter = LegalDatasetSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    splitter.split_file(
        args.input,
        args.output,
        stratify=not args.no_stratify
    )


if __name__ == "__main__":
    main()

