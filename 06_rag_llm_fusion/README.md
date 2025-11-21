# 6. RAG + LLM Fusion

This module integrates RAG retrieval with LLM generation, including citation extraction and anti-hallucination guardrails.

## Components

- `rag_llm_fusion.py`: Main RAG + LLM integration
- `citation_extractor.py`: Legal citation extraction logic
- `anti_hallucination.py`: Anti-hallucination guardrails
- `prompt_templates.py`: Input prompt structure for legal QA

## Features

- Query → RAG Retrieval → LLM Generation
- Automatic citation extraction
- Answer verification using retrieved context
- Hallucination detection and prevention

## Usage

```bash
# Run RAG + LLM pipeline
python rag_llm_fusion.py \
    --query "What is IPC Section 302?" \
    --model models/lora_legal \
    --index-dir data/faiss_index/
```

