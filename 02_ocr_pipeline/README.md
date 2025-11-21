# 2. OCR → Text Processing Pipeline

This module handles OCR processing for Hindi legal PDFs, including noise correction, sentence segmentation, and clause-level legal extraction.

## Components

- `ocr_pipeline.py`: PyTesseract OCR pipeline for Hindi legal PDFs
- `ocr_cleaning.py`: Cleaning script for OCR noise correction
- `sentence_segmentation.py`: Sentence segmentation for legal text
- `clause_extraction.py`: Clause-level legal extraction

## Dependencies

- PyTesseract
- Pillow
- pdf2image
- Hindi language data for Tesseract

## Usage

```bash
# Install Tesseract with Hindi support
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr tesseract-ocr-hin

# Process PDFs
python ocr_pipeline.py --input data/legal_docs/ --output data/ocr_output/

# Clean OCR output
python ocr_cleaning.py --input data/ocr_output/ --output data/cleaned/

# Segment sentences
python sentence_segmentation.py --input data/cleaned/ --output data/segmented/

# Extract clauses
python clause_extraction.py --input data/segmented/ --output data/clauses/
```

