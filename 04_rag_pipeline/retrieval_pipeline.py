"""
RAG Retrieval Pipeline

Implements: Query → Retrieval → Re-ranking → Context Assembly
"""

from typing import List, Dict, Optional
import numpy as np
from embedding_model import MultilingualLegalEmbedder
from faiss_index import FAISSLegalIndex
from reranker import LegalReranker
import argparse


class RAGRetrievalPipeline:
    """Complete RAG retrieval pipeline."""
    
    def __init__(
        self,
        index: FAISSLegalIndex,
        embedder: MultilingualLegalEmbedder,
        reranker: Optional[LegalReranker] = None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            index: FAISS index
            embedder: Embedding model
            reranker: Optional re-ranker model
        """
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            rerank: Whether to re-rank results
            
        Returns:
            List of retrieved documents
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: Retrieve from index (retrieve more if reranking)
        retrieve_k = top_k * 3 if rerank and self.reranker else top_k
        results = self.index.search(query_embedding, top_k=retrieve_k)
        
        # Step 3: Re-rank if enabled
        if rerank and self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query, results)
            results = results[:top_k]  # Take top-k after reranking
        
        return results
    
    def get_context(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 2000
    ) -> str:
        """
        Get formatted context for LLM.
        
        Args:
            query: User query
            top_k: Number of documents to include
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        # Retrieve documents
        results = self.retrieve(query, top_k=top_k)
        
        # Format context
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            doc_text = result['document']
            metadata = result.get('metadata', {})
            
            # Format document
            doc_format = f"[Document {i}]\n"
            if 'file' in metadata:
                doc_format += f"Source: {metadata['file']}\n"
            if 'legal_section' in metadata:
                doc_format += f"Legal Section: {metadata['legal_section']}\n"
            doc_format += f"Content: {doc_text}\n\n"
            
            # Check length
            if current_length + len(doc_format) > max_context_length:
                break
            
            context_parts.append(doc_format)
            current_length += len(doc_format)
        
        context = "".join(context_parts)
        return context
    
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Retrieve with full metadata.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Dictionary with query, results, and context
        """
        results = self.retrieve(query, top_k=top_k)
        context = self.get_context(query, top_k=top_k)
        
        return {
            "query": query,
            "results": results,
            "context": context,
            "num_results": len(results)
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="RAG Retrieval Pipeline")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to search"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory with FAISS index"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual",
        help="Embedding model"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking"
    )
    
    args = parser.parse_args()
    
    from pathlib import Path
    
    # Load index
    index = FAISSLegalIndex(dimension=768)
    index_path = Path(args.index_dir)
    index.load(
        str(index_path / "faiss.index"),
        str(index_path / "documents.pkl"),
        str(index_path / "metadata.json")
    )
    
    # Initialize embedder
    embedder = MultilingualLegalEmbedder(model_name=args.embedding_model)
    
    # Initialize reranker (optional)
    reranker = None if args.no_rerank else LegalReranker()
    
    # Create pipeline
    pipeline = RAGRetrievalPipeline(index, embedder, reranker)
    
    # Retrieve
    result = pipeline.retrieve_with_metadata(args.query, top_k=args.top_k)
    
    print(f"\nQuery: {result['query']}")
    print(f"\nRetrieved {result['num_results']} documents:\n")
    
    for i, doc_result in enumerate(result['results'], 1):
        print(f"[{i}] Score: {doc_result['score']:.4f}")
        print(f"    {doc_result['document'][:200]}...")
        if doc_result.get('metadata'):
            print(f"    Metadata: {doc_result['metadata']}")
        print()


if __name__ == "__main__":
    main()

