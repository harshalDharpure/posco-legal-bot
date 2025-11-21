# 3. Dataset Creation for Model Training

This module creates training datasets from extracted legal text in Question → Answer → Context → Legal Section format.

## Components

- `dataset_builder.py`: Convert extracted law text into QA format
- `train_test_split.py`: Train/validation/test split logic
- `data_augmentation.py`: Data augmentation for multilingual consistency
- `sample_data.json`: 50-100 sample training entries (IPC, CrPC, Constitution)

## Dataset Format

```json
{
  "question": "What is the punishment for murder under IPC?",
  "answer": "IPC Section 302 provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
  "context": "Indian Penal Code, Section 302...",
  "legal_section": "IPC Section 302",
  "language": "en",
  "hindi_question": "IPC के तहत हत्या की सजा क्या है?",
  "hindi_answer": "IPC धारा 302 प्रदान करती है कि जो कोई हत्या करता है, उसे मृत्युदंड या आजीवन कारावास से दंडित किया जाएगा, और जुर्माना भी लगाया जा सकता है।"
}
```

## Usage

```bash
# Build dataset from extracted clauses
python dataset_builder.py --input data/clauses/ --output data/dataset/

# Split into train/val/test
python train_test_split.py --input data/dataset/ --output data/splits/

# Augment data
python data_augmentation.py --input data/splits/train.json --output data/splits/train_augmented.json
```

