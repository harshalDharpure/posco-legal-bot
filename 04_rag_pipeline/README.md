# 4. RAG Pipeline Model Architecture

This module implements the Retrieval-Augmented Generation pipeline with FAISS/Elasticsearch document store.

## Components

- `embedding_model.py`: Embedding model selection and setup for Hindi + English
- `faiss_index.py`: Code to build FAISS index
- `elasticsearch_setup.py`: Alternative Elasticsearch document store
- `retrieval_pipeline.py`: Query → Retrieval → Re-ranking → LLM pipeline logic
- `reranker.py`: Re-ranking model for retrieved documents

## Architecture

1. **Document Processing**: Convert legal documents to embeddings
2. **Index Building**: Build FAISS/Elasticsearch index
3. **Query Processing**: Embed user query
4. **Retrieval**: Retrieve top-k relevant documents
5. **Re-ranking**: Re-rank retrieved documents
6. **Context Assembly**: Prepare context for LLM

## Usage

```bash
# Build FAISS index
python faiss_index.py --input data/legal_docs/ --output data/faiss_index/

# Build Elasticsearch index (alternative)
python elasticsearch_setup.py --input data/legal_docs/

# Test retrieval
python retrieval_pipeline.py --query "What is IPC Section 302?" --top-k 5
```

