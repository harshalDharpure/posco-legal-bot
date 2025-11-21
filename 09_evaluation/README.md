# 9. Evaluation Pipeline

This module provides comprehensive evaluation metrics and expert evaluation forms for the legal conversational bot.

## Components

- `legal_accuracy.py`: Legal accuracy score metric
- `rag_relevance.py`: RAG relevance score metric (Recall@k)
- `multilingual_consistency.py`: Multilingual consistency score
- `hallucination_penalty.py`: Hallucination penalty metric
- `expert_evaluation_form.md`: Human legal expert evaluation form
- `evaluation_pipeline.py`: Complete evaluation pipeline

## Metrics

1. **Legal Accuracy**: Factual correctness of legal information
2. **RAG Relevance**: Quality of retrieved documents (Recall@k)
3. **Multilingual Consistency**: Consistency between Hindi/English answers
4. **Hallucination Penalty**: Penalty for fabricated information
5. **Citation Accuracy**: Accuracy of legal citations

## Usage

```bash
# Run evaluation
python evaluation_pipeline.py \
    --test-dataset data/splits/test.json \
    --output results/evaluation.json
```

