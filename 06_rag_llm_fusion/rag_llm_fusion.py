"""
RAG + LLM Fusion

Integrates RAG retrieval with LLM generation for legal QA.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional
import argparse
from pathlib import Path

# Import RAG components
import sys
sys.path.append(str(Path(__file__).parent.parent / "04_rag_pipeline"))
from retrieval_pipeline import RAGRetrievalPipeline
from faiss_index import FAISSLegalIndex
from embedding_model import MultilingualLegalEmbedder

# Import fusion components
from prompt_templates import LegalPromptTemplates
from citation_extractor import LegalCitationExtractor
from anti_hallucination import AntiHallucinationGuard


class RAGLLMFusion:
    """RAG + LLM fusion for legal QA."""
    
    def __init__(
        self,
        model_path: str,
        index_dir: str,
        embedding_model: str = "paraphrase-multilingual",
        device: str = "cuda"
    ):
        """
        Initialize RAG + LLM fusion.
        
        Args:
            model_path: Path to LoRA fine-tuned model
            index_dir: Directory with FAISS index
            embedding_model: Embedding model name
            device: Device to use
        """
        self.device = device
        
        # Load LLM
        print(f"Loading LLM from {model_path}")
        base_model_name = "ai4bharat/IndicLegal-LLaMA-7B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load LoRA weights if available
        if Path(model_path).exists():
            print(f"Loading LoRA weights from {model_path}")
            self.model = PeftModel.from_pretrained(self.base_model, model_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        
        # Load RAG components
        print(f"Loading RAG index from {index_dir}")
        index_path = Path(index_dir)
        
        embedder = MultilingualLegalEmbedder(model_name=embedding_model)
        index = FAISSLegalIndex(dimension=embedder.get_dimension())
        index.load(
            str(index_path / "faiss.index"),
            str(index_path / "documents.pkl"),
            str(index_path / "metadata.json")
        )
        
        self.retrieval_pipeline = RAGRetrievalPipeline(index, embedder)
        
        # Initialize components
        self.citation_extractor = LegalCitationExtractor()
        self.anti_hallucination = AntiHallucinationGuard()
        self.prompt_templates = LegalPromptTemplates()
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        max_length: int = 512,
        temperature: float = 0.7,
        language: str = "en"
    ) -> Dict:
        """
        Generate answer using RAG + LLM.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_length: Maximum generation length
            temperature: Sampling temperature
            language: Language ("en" or "hi")
            
        Returns:
            Dictionary with answer and metadata
        """
        # Step 1: Retrieve relevant context
        context = self.retrieval_pipeline.get_context(query, top_k=top_k)
        
        # Step 2: Extract citations from context
        context_citations, _ = self.citation_extractor.extract_and_format(context)
        
        # Step 3: Create prompt
        prompt = self.prompt_templates.legal_qa_prompt(query, context, language)
        
        # Step 4: Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode answer
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip()
        
        # Step 5: Extract citations from answer
        answer_citations, formatted_citations = self.citation_extractor.extract_and_format(answer)
        
        # Step 6: Check for hallucinations
        hallucination_checks = self.anti_hallucination.check_hallucination(
            answer, context, query
        )
        
        # Step 7: Filter unsafe content if needed
        if not hallucination_checks["is_safe"]:
            answer, _ = self.anti_hallucination.filter_unsafe_answer(
                answer, context, query
            )
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "citations": formatted_citations,
            "citation_details": answer_citations,
            "hallucination_checks": hallucination_checks,
            "retrieved_docs": len(context.split("[Document")) - 1
        }
    
    def answer_with_verification(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Generate answer with verification step.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with verified answer
        """
        # Generate initial answer
        result = self.generate_answer(query, top_k=top_k)
        
        # Verify answer
        verification_prompt = self.prompt_templates.verification_prompt(
            result["answer"],
            result["context"],
            query
        )
        
        inputs = self.tokenizer(verification_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                temperature=0.1,
                do_sample=False
            )
        
        verification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result["verification"] = verification
        result["verified"] = "Yes" in verification or "yes" in verification
        
        return result


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="RAG + LLM Fusion")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to answer"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to LoRA model"
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
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "hi"],
        help="Language"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable answer verification"
    )
    
    args = parser.parse_args()
    
    # Initialize fusion
    fusion = RAGLLMFusion(args.model, args.index_dir)
    
    # Generate answer
    if args.verify:
        result = fusion.answer_with_verification(args.query, args.top_k)
    else:
        result = fusion.generate_answer(args.query, args.top_k, language=args.language)
    
    # Print results
    print(f"\nQuery: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")
    print(f"\nHallucination Check:")
    print(f"  Is Safe: {result['hallucination_checks']['is_safe']}")
    print(f"  Score: {result['hallucination_checks']['hallucination_score']:.2f}")
    if result['hallucination_checks']['warnings']:
        print(f"  Warnings: {result['hallucination_checks']['warnings']}")
    
    if args.verify:
        print(f"\nVerification: {result['verified']}")
        print(f"Verification Response: {result['verification']}")


if __name__ == "__main__":
    main()

