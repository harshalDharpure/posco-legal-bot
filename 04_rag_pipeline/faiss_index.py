"""
FAISS Index Building for Legal Documents

Builds and manages FAISS index for efficient similarity search.
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import json
import argparse
from tqdm import tqdm
from embedding_model import MultilingualLegalEmbedder


class FAISSLegalIndex:
    """FAISS index for legal document retrieval."""
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat"
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Create index based on type
        if index_type == "flat":
            # Exact search (L2 distance)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # Inverted file index (approximate, faster)
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World (approximate, fast)
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        print(f"Created {index_type} FAISS index with dimension {dimension}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to index.
        
        Args:
            embeddings: Document embeddings (N x dimension)
            documents: Document texts
            metadata: Optional metadata for each document
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        print(f"Added {len(documents)} documents to index")
        print(f"Total documents: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding (1D array)
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    "rank": i + 1,
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distance),
                    "score": float(1 / (1 + distance))  # Convert distance to similarity score
                }
                results.append(result)
        
        return results
    
    def save(self, index_path: str, documents_path: str, metadata_path: str):
        """
        Save index to disk.
        
        Args:
            index_path: Path to save FAISS index
            documents_path: Path to save documents
            metadata_path: Path to save metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved index to {index_path}")
        print(f"Saved {len(self.documents)} documents")
    
    def load(
        self,
        index_path: str,
        documents_path: str,
        metadata_path: str
    ):
        """
        Load index from disk.
        
        Args:
            index_path: Path to FAISS index
            documents_path: Path to documents
            metadata_path: Path to metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Load documents
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded index with {len(self.documents)} documents")
    
    def build_from_directory(
        self,
        input_dir: str,
        embedder: MultilingualLegalEmbedder,
        output_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Build index from directory of text files.
        
        Args:
            input_dir: Input directory with text files
            embedder: Embedding model
            output_dir: Output directory for index
            chunk_size: Chunk size for documents
            chunk_overlap: Overlap between chunks
        """
        input_path = Path(input_dir)
        text_files = list(input_path.glob("*.txt"))
        
        print(f"Found {len(text_files)} text files")
        
        all_documents = []
        all_metadata = []
        
        # Process files
        for file in tqdm(text_files, desc="Processing files"):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Chunk document
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                all_documents.append(chunk)
                all_metadata.append({
                    "file": file.name,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
        
        print(f"Total chunks: {len(all_documents)}")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = embedder.embed_documents(all_documents)
        
        # Add to index
        self.add_documents(embeddings, all_documents, all_metadata)
        
        # Save index
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.save(
            str(Path(output_dir) / "faiss.index"),
            str(Path(output_dir) / "documents.pkl"),
            str(Path(output_dir) / "metadata.json")
        )
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text into chunks."""
        chunks = []
        words = text.split()
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            i += chunk_size - chunk_overlap
        
        return chunks


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with text files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for index"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="Index type (default: flat)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for documents (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = MultilingualLegalEmbedder(model_name=args.embedding_model)
    
    # Build index
    index = FAISSLegalIndex(
        dimension=embedder.get_dimension(),
        index_type=args.index_type
    )
    
    index.build_from_directory(
        args.input,
        embedder,
        args.output,
        chunk_size=args.chunk_size
    )
    
    print(f"\nIndex built successfully!")
    print(f"Index type: {args.index_type}")
    print(f"Total documents: {index.index.ntotal}")


if __name__ == "__main__":
    main()

