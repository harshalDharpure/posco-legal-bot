"""
Prompt Templates for Legal QA

Structured prompts for legal question answering with RAG context.
"""

from typing import Dict, List, Optional


class LegalPromptTemplates:
    """Prompt templates for legal QA."""
    
    @staticmethod
    def legal_qa_prompt(
        question: str,
        context: str,
        language: str = "en"
    ) -> str:
        """
        Create legal QA prompt with context.
        
        Args:
            question: User question
            context: Retrieved context from RAG
            language: Language ("en" or "hi")
            
        Returns:
            Formatted prompt
        """
        if language == "hi":
            prompt = f"""आप एक कानूनी सहायक हैं। निम्नलिखित प्रश्न का उत्तर दें जो प्रदान किए गए संदर्भ पर आधारित है।

संदर्भ:
{context}

प्रश्न: {question}

उत्तर: आपको अपने उत्तर में प्रासंगिक कानूनी धाराओं (IPC, CrPC, संविधान) का उल्लेख करना चाहिए।"""
        else:
            prompt = f"""You are a legal assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer: You should cite relevant legal sections (IPC, CrPC, Constitution) in your answer."""
        
        return prompt
    
    @staticmethod
    def legal_qa_with_citations(
        question: str,
        context: str,
        citations: List[str],
        language: str = "en"
    ) -> str:
        """
        Create prompt with explicit citations.
        
        Args:
            question: User question
            context: Retrieved context
            citations: List of citations
            language: Language
            
        Returns:
            Formatted prompt
        """
        citations_text = "\n".join([f"- {cite}" for cite in citations])
        
        if language == "hi":
            prompt = f"""आप एक कानूनी सहायक हैं। निम्नलिखित प्रश्न का उत्तर दें।

संदर्भ:
{context}

प्रासंगिक कानूनी धाराएं:
{citations_text}

प्रश्न: {question}

उत्तर:"""
        else:
            prompt = f"""You are a legal assistant. Answer the following question.

Context:
{context}

Relevant Legal Sections:
{citations_text}

Question: {question}

Answer:"""
        
        return prompt
    
    @staticmethod
    def verification_prompt(
        answer: str,
        context: str,
        question: str
    ) -> str:
        """
        Create prompt for answer verification.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            question: Original question
            
        Returns:
            Verification prompt
        """
        prompt = f"""Verify if the following answer is factually correct based on the provided context.

Question: {question}

Context:
{context}

Answer to verify:
{answer}

Is this answer factually correct based on the context? Answer with "Yes" or "No" and explain why."""
        
        return prompt
    
    @staticmethod
    def citation_extraction_prompt(
        answer: str
    ) -> str:
        """
        Create prompt for citation extraction.
        
        Args:
            answer: Answer text
            
        Returns:
            Citation extraction prompt
        """
        prompt = f"""Extract all legal citations from the following text. Citations include:
- IPC sections (e.g., "IPC Section 302")
- CrPC sections (e.g., "CrPC Section 438")
- Constitution articles (e.g., "Article 21")
- Case citations (e.g., "Kesavananda Bharati v. State of Kerala")

Text:
{answer}

Extract all citations in the format: [Type] [Number/Name]"""
        
        return prompt


def get_prompt_template(template_name: str) -> callable:
    """Get prompt template function by name."""
    templates = {
        "legal_qa": LegalPromptTemplates.legal_qa_prompt,
        "legal_qa_with_citations": LegalPromptTemplates.legal_qa_with_citations,
        "verification": LegalPromptTemplates.verification_prompt,
        "citation_extraction": LegalPromptTemplates.citation_extraction_prompt
    }
    
    return templates.get(template_name)

