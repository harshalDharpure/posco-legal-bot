"""
Embedding Model Selection for Hindi + English Legal Text

Supports multilingual embedding models for legal domain.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
import numpy as np


class MultilingualLegalEmbedder:
    """Multilingual embedding model for legal text."""
    
    MODELS = {
        "paraphrase-multilingual": {
            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "dimension": 768,
            "languages": ["hi", "en", "mr", "gu", "ta", "te", "kn", "ml", "bn", "or", "pa", "as"],
            "description": "Multilingual MPNet model, supports Hindi + English"
        },
        "indicsentence": {
            "name": "ai4bharat/indicsentence-bert-base",
            "dimension": 768,
            "languages": ["hi", "en", "mr", "gu", "ta", "te", "kn", "ml", "bn", "or", "pa", "as"],
            "description": "IndicSentence BERT, optimized for Indian languages"
        },
        "legal-bert": {
            "name": "nlpaueb/legal-bert-base-uncased",
            "dimension": 768,
            "languages": ["en"],
            "description": "Legal domain BERT (English only)"
        }
    }
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual",
        device: str = None
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Model key or full model name
            device: Device to use (cuda/cpu)
        """
        if model_name in self.MODELS:
            model_info = self.MODELS[model_name]
            full_model_name = model_info["name"]
            self.dimension = model_info["dimension"]
        else:
            full_model_name = model_name
            self.dimension = 768  # Default
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading embedding model: {full_model_name}")
        self.model = SentenceTransformer(full_model_name, device=device)
        self.device = device
        
        print(f"Model loaded on {device}")
        print(f"Embedding dimension: {self.dimension}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed(query, show_progress=False)
    
    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            batch_size: Batch size
            
        Returns:
            Document embeddings
        """
        return self.embed(documents, batch_size=batch_size)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding (1D array)
            document_embeddings: Document embeddings (2D array)
            
        Returns:
            Similarity scores
        """
        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(document_embeddings, query_embedding)
        return similarities


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Legal Embedder")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual",
        choices=list(MultilingualLegalEmbedder.MODELS.keys()) + ["custom"],
        help="Embedding model to use"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="IPC Section 302 deals with punishment for murder.",
        help="Text to embed"
    )
    
    args = parser.parse_args()
    
    embedder = MultilingualLegalEmbedder(model_name=args.model)
    
    # Embed text
    embedding = embedder.embed_query(args.text)
    
    print(f"\nText: {args.text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")


if __name__ == "__main__":
    main()

