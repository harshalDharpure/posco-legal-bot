"""
Multi-Bot Coordinator

Coordinates 4 LLM modules for legal query assistance.
"""

from typing import Dict, Optional
from legal_q_bot import LegalQBot
from citation_bot import CitationBot
from translation_bot import TranslationBot
from validator_bot import ValidatorBot
import argparse


class MultiBotCoordinator:
    """Coordinates multiple specialized bots."""
    
    def __init__(
        self,
        model_path: str = "models/lora_legal",
        enable_translation: bool = True,
        enable_validation: bool = True
    ):
        """
        Initialize coordinator.
        
        Args:
            model_path: Path to fine-tuned model
            enable_translation: Enable translation bot
            enable_validation: Enable validation bot
        """
        # Initialize bots
        self.legal_q_bot = LegalQBot(model_path)
        self.citation_bot = CitationBot()
        self.translation_bot = TranslationBot() if enable_translation else None
        self.validator_bot = ValidatorBot() if enable_validation else None
    
    def process_query(
        self,
        query: str,
        context: str = None,
        language: str = "en",
        target_language: str = None
    ) -> Dict:
        """
        Process query through multi-bot pipeline.
        
        Args:
            query: User query
            context: Optional RAG context
            language: Query language
            target_language: Target language for answer (if different)
            
        Returns:
            Complete response with all bot outputs
        """
        results = {
            "query": query,
            "language": language,
            "pipeline": []
        }
        
        # Step 1: Translate query if needed
        if self.translation_bot and target_language and target_language != language:
            translation_result = self.translation_bot.translate_query(query, target_language)
            query = translation_result["translated"]
            results["translated_query"] = translation_result
            results["pipeline"].append("Translation-Bot: Query translated")
        
        # Step 2: Generate answer (Legal-Q-Bot)
        legal_result = self.legal_q_bot.generate_answer(query, context, language)
        answer = legal_result["answer"]
        results["legal_q_bot"] = legal_result
        results["pipeline"].append("Legal-Q-Bot: Answer generated")
        
        # Step 3: Add citations (Citation-Bot)
        citation_result = self.citation_bot.add_citations(answer, context)
        answer = citation_result["answer"]
        results["citation_bot"] = citation_result
        results["pipeline"].append("Citation-Bot: Citations added")
        
        # Step 4: Validate answer (Validator-Bot)
        if self.validator_bot and context:
            validation_result = self.validator_bot.filter_and_enhance(
                answer, context, query
            )
            answer = validation_result["answer"]
            results["validator_bot"] = validation_result
            results["pipeline"].append("Validator-Bot: Answer validated")
        
        # Step 5: Translate answer if needed
        if self.translation_bot and target_language and target_language != language:
            answer_translation = self.translation_bot.translate_answer(answer, target_language)
            results["translated_answer"] = answer_translation
            results["pipeline"].append("Translation-Bot: Answer translated")
            answer = answer_translation["translated"]
        
        # Final answer
        results["final_answer"] = answer
        results["citations"] = citation_result.get("citations", "")
        
        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Multi-Bot Coordinator")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lora_legal",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "hi"],
        help="Query language"
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default=None,
        choices=["en", "hi"],
        help="Target language for answer"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional RAG context"
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="Disable translation bot"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation bot"
    )
    
    args = parser.parse_args()
    
    # Initialize coordinator
    coordinator = MultiBotCoordinator(
        model_path=args.model,
        enable_translation=not args.no_translation,
        enable_validation=not args.no_validation
    )
    
    # Process query
    result = coordinator.process_query(
        args.query,
        args.context,
        args.language,
        args.target_language
    )
    
    # Print results
    print(f"\nQuery: {result['query']}")
    print(f"Language: {result['language']}")
    print(f"\nPipeline Steps:")
    for step in result['pipeline']:
        print(f"  - {step}")
    
    print(f"\nFinal Answer:\n{result['final_answer']}")
    
    if result.get('citations'):
        print(f"\nCitations: {result['citations']}")
    
    if result.get('validator_bot'):
        validation = result['validator_bot']['validation']
        print(f"\nValidation:")
        print(f"  Is Valid: {validation['is_valid']}")
        print(f"  Score: {validation['validation_score']:.2f}")


if __name__ == "__main__":
    main()

