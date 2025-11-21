# 1. Base Model Selection & Architecture

This module handles base model selection, tokenization strategy, and legal vocabulary adaptation for multilingual (Hindi + English) legal text processing.

## Components

- `model_selection.py`: Base model selection logic
- `tokenization_strategy.py`: Multilingual legal tokenization
- `legal_vocab_adaptation.py`: Legal vocabulary adaptation
- `architecture_diagram.txt`: Model architecture breakdown

## Supported Models

1. **IndicLegal-LLaMA-7B**: Pre-trained on Indian legal corpus
2. **LLaMA-3-8B-Instruct**: General multilingual instruction-tuned model
3. **IndicBERT**: Hindi-English bilingual BERT

## Usage

```bash
python model_selection.py --model indiclegal-llama
python tokenization_strategy.py --input data/legal_text.txt
python legal_vocab_adaptation.py --vocab_size 50000
```

