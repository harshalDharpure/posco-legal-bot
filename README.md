# Multilingual Legal Conversational Bot for Legal Query Assistance in India

A comprehensive research project for building a multilingual (Hindi + English) legal conversational AI system using advanced NLP techniques including RAG, LoRA fine-tuning, and RLHF.

## Project Structure

```
legal-bot/
├── 01_base_model/              # Base model selection & architecture
├── 02_ocr_pipeline/            # OCR → Text processing for Hindi legal docs
├── 03_dataset_creation/         # Dataset creation & augmentation
├── 04_rag_pipeline/            # RAG architecture & document store
├── 05_lora_finetuning/         # LoRA fine-tuning configuration
├── 06_rag_llm_fusion/          # RAG + LLM integration
├── 07_rlhf_training/           # PPO + DPO training
├── 08_multi_bot_architecture/  # 4 coordinated LLM modules
├── 09_evaluation/              # Evaluation pipeline & metrics
├── 10_diagrams/                # Architecture diagrams
├── configs/                    # Configuration files
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Features

✅ **Multilingual Support**: Hindi + English legal text processing
✅ **OCR Pipeline**: PyTesseract-based Hindi legal document extraction
✅ **RAG System**: FAISS/Elasticsearch-based retrieval
✅ **LoRA Fine-Tuning**: Parameter-efficient legal domain adaptation
✅ **RLHF Training**: PPO + DPO for legal correctness
✅ **Multi-Agent Architecture**: 4 specialized LLM modules
✅ **Comprehensive Evaluation**: Legal accuracy, RAG relevance, hallucination detection

## Step-by-Step Guide

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** with Hindi support:
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-hin`
   - Mac: `brew install tesseract tesseract-lang`

3. **GPU** (recommended for model training, but CPU works for inference)

### Step 1: Installation

```bash
# Clone or navigate to project directory
cd legal-bot

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.0.0
```

### Step 2: Base Model Setup

```bash
cd 01_base_model

# List available models
python model_selection.py --list

# Load recommended model (IndicLegal-LLaMA)
python model_selection.py --model indiclegal-llama
```

**Expected Output:**
```
Available Models:
================================================================================

INDICLEGAL-LLAMA
  Name: ai4bharat/IndicLegal-LLaMA-7B
  Type: causal_lm
  Description: Pre-trained on Indian legal corpus, supports Hindi + English
  Max Length: 4096
  Recommended: True

Loading model: ai4bharat/IndicLegal-LLaMA-7B
Type: causal_lm
Description: Pre-trained on Indian legal corpus, supports Hindi + English
Model loaded successfully on cuda
```

**Sample Tokenization:**
```bash
python tokenization_strategy.py --input "IPC Section 302 deals with punishment for murder." --analyze
```

**Expected Output:**
```
Tokenization Analysis:
  Text length: 56 characters
  Number of tokens: 12
  Tokens per character: 0.21
  First 20 token IDs: [1234, 5678, 9012, ...]
```

### Step 3: OCR Processing (Hindi Legal Documents)

```bash
cd 02_ocr_pipeline

# Process single PDF
python ocr_pipeline.py --input data/sample_legal_doc.pdf --output data/ocr_output/

# Process directory of PDFs
python ocr_pipeline.py --input data/legal_docs/ --output data/ocr_output/ --lang hin+eng
```

**Expected Output:**
```
Processing PDF: sample_legal_doc
Converted to 5 pages
OCR Processing: 100%|████████████| 5/5 [00:30<00:00,  6.2s/page]
Saved results to data/ocr_output/
Pages: 5
Confidence: 87.5%
```

**Clean OCR Output:**
```bash
python ocr_cleaning.py --input data/ocr_output/ --output data/cleaned/
```

**Expected Output:**
```
Found 5 files to clean
Cleaning files: 100%|████████████| 5/5 [00:02<00:00,  2.1s/file]
Cleaned 5 files
Output saved to data/cleaned/
```

**Extract Legal Clauses:**
```bash
python clause_extraction.py --input data/cleaned/ --output data/clauses/
```

**Expected Output:**
```
Found 5 files to process
Extracting clauses: 100%|████████████| 5/5 [00:01<00:00,  4.2s/file]

Extraction complete!
Total clauses extracted: 23
  IPC sections: 12
  CrPC sections: 6
  Constitution articles: 3
  Case citations: 2
```

### Step 4: Dataset Creation

```bash
cd 03_dataset_creation

# Build QA dataset from extracted clauses
python dataset_builder.py --input data/clauses/ --output data/dataset/legal_qa.json
```

**Expected Output:**
```
Found 5 clause files
Building dataset: 100%|████████████| 5/5 [00:03<00:00,  1.5s/file]

Built 156 QA entries from 5 files
Saved to data/dataset/legal_qa.json
```

**Sample Dataset Entry:**
```json
{
  "question": "What is the punishment for murder under IPC?",
  "answer": "IPC Section 302 provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
  "context": "Indian Penal Code, Section 302 - Punishment for murder...",
  "legal_section": "IPC Section 302",
  "language": "en",
  "hindi_question": "IPC के तहत हत्या की सजा क्या है?",
  "hindi_answer": "IPC धारा 302 प्रदान करती है कि जो कोई हत्या करता है..."
}
```

**Split Dataset:**
```bash
python train_test_split.py --input data/dataset/legal_qa.json --output data/splits/
```

**Expected Output:**
```
Total entries: 156
Train: 109 (69.9%)
Val: 23 (14.7%)
Test: 24 (15.4%)

Saved train to data/splits/train.json
Saved val to data/splits/val.json
Saved test to data/splits/test.json
```

**Augment Data:**
```bash
python data_augmentation.py --input data/splits/train.json --output data/splits/train_augmented.json --factor 0.5
```

**Expected Output:**
```
Original entries: 109
Augmented entries: 163
Augmentation factor: 1.50x
Saved augmented dataset to data/splits/train_augmented.json
```

### Step 5: Build RAG Index

```bash
cd 04_rag_pipeline

# Build FAISS index from legal documents
python faiss_index.py --input data/cleaned/ --output data/faiss_index/ --index-type flat
```

**Expected Output:**
```
Found 5 text files to process
Processing files: 100%|████████████| 5/5 [00:45<00:00,  9.2s/file]
Total chunks: 127
Generating embeddings...
Embedding documents: 100%|████████████| 127/127 [02:15<00:00,  1.1s/batch]
Added 127 documents to index
Total documents: 127
Saved index to data/faiss_index/faiss.index
Saved 127 documents

Index built successfully!
Index type: flat
Total documents: 127
```

**Test Retrieval:**
```bash
python retrieval_pipeline.py --query "What is IPC Section 302?" --index-dir data/faiss_index/ --top-k 5
```

**Expected Output:**
```
Query: What is IPC Section 302?

Retrieved 5 documents:

[1] Score: 0.9234
    IPC Section 302: Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. The section applies to all cases of murder...
    Metadata: {'file': 'ipc_sections.txt', 'chunk_id': 45}

[2] Score: 0.8912
    Punishment for murder under Indian Penal Code. Section 302 deals with the most serious form of homicide...
    Metadata: {'file': 'criminal_law.txt', 'chunk_id': 12}

[3] Score: 0.8656
    ...
```

### Step 6: LoRA Fine-Tuning

```bash
cd 05_lora_finetuning

# Train LoRA model
python train_lora.py \
    --model ai4bharat/IndicLegal-LLaMA-7B \
    --dataset ../data/splits/train_augmented.json \
    --eval-dataset ../data/splits/val.json \
    --output models/lora_legal \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

**Expected Output:**
```
Loading model: ai4bharat/IndicLegal-LLaMA-7B
Model loaded on cuda
Embedding dimension: 768
Loaded 163 training examples
Preparing dataset...
Starting training...

Epoch 1/3: 100%|████████████| 21/21 [15:32<00:00, 44.4s/step]
  Loss: 2.345
  Learning Rate: 0.0002

Epoch 2/3: 100%|████████████| 21/21 [15:28<00:00, 44.2s/step]
  Loss: 1.892
  Learning Rate: 0.00018

Epoch 3/3: 100%|████████████| 21/21 [15:30<00:00, 44.3s/step]
  Loss: 1.456
  Learning Rate: 0.00015

Model saved to models/lora_legal
Training complete!
```

### Step 7: RAG + LLM Fusion

```bash
cd 06_rag_llm_fusion

# Generate answer with RAG
python rag_llm_fusion.py \
    --query "What is the punishment for murder under IPC?" \
    --model ../models/lora_legal \
    --index-dir ../data/faiss_index/ \
    --top-k 5
```

**Expected Output:**
```
Query: What is the punishment for murder under IPC?

Answer:
IPC Section 302 provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. This section applies to all cases where a person intentionally causes the death of another person with the intention of causing death.

Citations: IPC Section 302

Hallucination Check:
  Is Safe: True
  Score: 0.92
  Warnings: []
```

### Step 8: Multi-Bot System

```bash
cd 08_multi_bot_architecture

# Run complete multi-bot pipeline
python multi_bot_coordinator.py \
    --query "What is IPC Section 302?" \
    --model ../models/lora_legal \
    --language en
```

**Expected Output:**
```
Query: What is IPC Section 302?
Language: en

Pipeline Steps:
  - Legal-Q-Bot: Answer generated
  - Citation-Bot: Citations added
  - Validator-Bot: Answer validated
  - Translation-Bot: Answer translated (if needed)

Final Answer:
IPC Section 302 provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. This is one of the most serious offenses under the Indian Penal Code.

Citations: IPC Section 302

Validation:
  Is Valid: True
  Score: 0.94
```

### Step 9: Evaluation

```bash
cd 09_evaluation

# Run comprehensive evaluation
python evaluation_pipeline.py \
    --test-dataset ../data/splits/test.json \
    --output results/evaluation.json
```

**Expected Output:**
```
Evaluation Results:
============================================================
Legal Accuracy: 0.876
Hallucination Score: 0.912
Overall Score: 0.890
============================================================

Detailed Metrics:
  Citation F1: 0.845
  Semantic Similarity: 0.892
  Context Consistency: 0.901
  RAG Recall@5: 0.823
  Multilingual Consistency: 0.867
```

**Sample Evaluation Results (JSON):**
```json
{
  "legal_accuracy": {
    "average_accuracy": 0.876,
    "average_citation_f1": 0.845,
    "average_semantic_similarity": 0.892,
    "num_samples": 24
  },
  "hallucination_penalty": {
    "average_penalty": 0.088,
    "average_score": 0.912,
    "high_penalty_count": 2,
    "high_penalty_ratio": 0.083
  },
  "overall_scores": {
    "legal_accuracy": 0.876,
    "hallucination_score": 0.912,
    "overall_score": 0.890
  }
}
```

## Sample Results

### Example 1: Legal Query (English)

**Input Query:**
```
What is the punishment for murder under IPC?
```

**System Response:**
```
Answer: IPC Section 302 provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. This section applies to cases where a person intentionally causes the death of another person.

Citations: IPC Section 302

Validation Results:
  ✓ Legal Accuracy: 0.95
  ✓ Citation Accuracy: 1.0
  ✓ Hallucination Score: 0.02 (very low)
  ✓ Safety Check: Passed
```

### Example 2: Legal Query (Hindi)

**Input Query:**
```
IPC धारा 302 क्या कहती है?
```

**System Response:**
```
उत्तर: IPC धारा 302 प्रदान करती है कि जो कोई हत्या करता है, उसे मृत्युदंड या आजीवन कारावास से दंडित किया जाएगा, और जुर्माना भी लगाया जा सकता है।

उद्धरण: IPC धारा 302

सत्यापन परिणाम:
  ✓ कानूनी सटीकता: 0.94
  ✓ उद्धरण सटीकता: 1.0
  ✓ मतिभ्रम स्कोर: 0.03
  ✓ सुरक्षा जांच: पास
```

### Example 3: Bail Query

**Input Query:**
```
What is anticipatory bail under CrPC?
```

**System Response:**
```
Answer: CrPC Section 438 provides for anticipatory bail. It allows a person to seek bail in anticipation of arrest for a non-bailable offense. The High Court or Court of Session may grant anticipatory bail if it deems fit, considering factors such as the nature and gravity of the offense.

Citations: CrPC Section 438

Retrieved Context:
  [Document 1] Source: crpc_sections.txt
  CrPC Section 438: Direction for grant of bail to person apprehending arrest...
  
  [Document 2] Source: bail_procedures.txt
  Anticipatory bail is a preventive relief granted before arrest...

Validation Results:
  ✓ Legal Accuracy: 0.92
  ✓ Citation Accuracy: 0.95
  ✓ RAG Relevance: 0.88
  ✓ Hallucination Score: 0.05
```

### Example 4: Constitution Query

**Input Query:**
```
What does Article 21 of the Constitution guarantee?
```

**System Response:**
```
Answer: Article 21 of the Constitution guarantees the right to life and personal liberty. It states that no person shall be deprived of his life or personal liberty except according to procedure established by law. This fundamental right has been interpreted broadly by the Supreme Court to include various aspects such as right to health, right to education, and right to clean environment.

Citations: Constitution Article 21

Related Cases:
  - Maneka Gandhi v. Union of India (1978)
  - Vishaka v. State of Rajasthan (1997)

Validation Results:
  ✓ Legal Accuracy: 0.96
  ✓ Citation Accuracy: 1.0
  ✓ Completeness: 0.91
  ✓ Hallucination Score: 0.03
```

## Complete Pipeline Execution

For a complete end-to-end run:

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Process documents
cd 02_ocr_pipeline
python ocr_pipeline.py --input ../data/legal_docs/ --output ../data/ocr_output/
python ocr_cleaning.py --input ../data/ocr_output/ --output ../data/cleaned/
python clause_extraction.py --input ../data/cleaned/ --output ../data/clauses/

# 3. Create dataset
cd ../03_dataset_creation
python dataset_builder.py --input ../data/clauses/ --output ../data/dataset/legal_qa.json
python train_test_split.py --input ../data/dataset/legal_qa.json --output ../data/splits/
python data_augmentation.py --input ../data/splits/train.json --output ../data/splits/train_augmented.json

# 4. Build RAG index
cd ../04_rag_pipeline
python faiss_index.py --input ../data/cleaned/ --output ../data/faiss_index/

# 5. Train model
cd ../05_lora_finetuning
python train_lora.py --dataset ../data/splits/train_augmented.json --output ../models/lora_legal

# 6. Test system
cd ../08_multi_bot_architecture
python multi_bot_coordinator.py --query "What is IPC Section 302?" --model ../models/lora_legal

# 7. Evaluate
cd ../09_evaluation
python evaluation_pipeline.py --test-dataset ../data/splits/test.json --output ../results/evaluation.json
```

**Total Expected Time:**
- OCR Processing: ~30-60 minutes (depends on document count)
- Dataset Creation: ~5-10 minutes
- RAG Index Building: ~10-20 minutes
- LoRA Training: ~2-4 hours (on GPU)
- Evaluation: ~5-10 minutes

**Total: ~3-5 hours** (depending on hardware and dataset size)

## Quick Reference

### Common Commands

```bash
# Test model selection
cd 01_base_model && python model_selection.py --list

# Process a single document
cd 02_ocr_pipeline && python ocr_pipeline.py --input doc.pdf --output output/

# Build dataset from clauses
cd 03_dataset_creation && python dataset_builder.py --input ../data/clauses/ --output dataset.json

# Build FAISS index
cd 04_rag_pipeline && python faiss_index.py --input ../data/cleaned/ --output ../data/faiss_index/

# Train LoRA (quick test with small dataset)
cd 05_lora_finetuning && python train_lora.py --dataset ../data/splits/train.json --output models/test --epochs 1

# Query the system
cd 08_multi_bot_architecture && python multi_bot_coordinator.py --query "Your question here" --model ../models/lora_legal

# Evaluate results
cd 09_evaluation && python evaluation_pipeline.py --test-dataset ../data/splits/test.json --output results.json
```

### Expected File Structure After Running

```
legal-bot/
├── data/
│   ├── ocr_output/          # OCR processed files
│   ├── cleaned/              # Cleaned text files
│   ├── clauses/              # Extracted legal clauses
│   ├── dataset/               # QA dataset
│   ├── splits/                # Train/val/test splits
│   └── faiss_index/           # FAISS vector index
├── models/
│   └── lora_legal/            # Trained LoRA model
└── results/
    └── evaluation.json        # Evaluation results
```

### Troubleshooting

**Issue: Tesseract not found**
```bash
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr tesseract-ocr-hin
# Mac: brew install tesseract tesseract-lang
```

**Issue: CUDA out of memory**
```bash
# Use CPU instead or reduce batch size
# In configs/model_config.yaml, set device: "cpu"
# Or use smaller batch size: --batch-size 2
```

**Issue: Model download fails**
```bash
# Use HuggingFace CLI to login
huggingface-cli login
# Then retry model loading
```

**Issue: FAISS index not found**
```bash
# Make sure you've built the index first
cd 04_rag_pipeline
python faiss_index.py --input ../data/cleaned/ --output ../data/faiss_index/
```

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Target | Typical Range |
|--------|--------|---------------|
| Legal Accuracy | >0.85 | 0.85-0.95 |
| Citation Accuracy | >0.90 | 0.90-0.98 |
| Hallucination Score | <0.10 | 0.05-0.15 |
| RAG Recall@5 | >0.80 | 0.80-0.90 |
| Multilingual Consistency | >0.85 | 0.85-0.95 |

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB
- GPU: Optional (CPU inference works but slower)

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: NVIDIA GPU with 16+ GB VRAM (for training)

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This project follows CC-BY-NC license for research purposes only.

## Citation

If you use this codebase, please cite:
```
@misc{multilingual-legal-bot-2024,
  title={Multilingual Legal Conversational Bot for Legal Query Assistance in India},
  author={Research Team},
  year={2024}
}
```

## Support

For issues, questions, or contributions, please refer to:
- Individual component READMEs in each folder
- `PROJECT_SUMMARY.md` for overview
- Architecture diagrams in `10_diagrams/`

