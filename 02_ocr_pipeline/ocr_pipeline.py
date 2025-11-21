"""
PyTesseract OCR Pipeline for Hindi Legal PDFs

Processes Hindi legal documents with:
- PDF to image conversion
- Hindi OCR (Devanagari script)
- Multi-page document handling
- Quality optimization
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import json


class HindiLegalOCR:
    """OCR pipeline for Hindi legal documents."""
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        lang: str = "hin+eng",
        dpi: int = 300,
        ocr_config: str = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ।,"
    ):
        """
        Initialize OCR pipeline.
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
            lang: Language code (hin+eng for Hindi + English)
            dpi: DPI for PDF to image conversion
            ocr_config: Tesseract OCR configuration
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.lang = lang
        self.dpi = dpi
        self.ocr_config = ocr_config
        
        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception as e:
            raise RuntimeError(f"Tesseract not found. Please install Tesseract OCR. Error: {e}")
    
    def pdf_to_images(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Convert PDF to images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images
            
        Returns:
            List of PIL Images
        """
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='png'
            )
            
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                pdf_name = Path(pdf_path).stem
                for i, img in enumerate(images):
                    img_path = Path(output_dir) / f"{pdf_name}_page_{i+1}.png"
                    img.save(img_path)
            
            return images
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return []
    
    def ocr_image(
        self,
        image: Image.Image,
        page_num: int = 1
    ) -> Dict[str, any]:
        """
        Perform OCR on a single image.
        
        Args:
            image: PIL Image
            page_num: Page number for tracking
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.ocr_config
            )
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "page_num": page_num,
                "word_count": len([w for w in data['text'] if w.strip()]),
                "raw_data": data
            }
        except Exception as e:
            print(f"Error in OCR for page {page_num}: {e}")
            return {
                "text": "",
                "confidence": 0,
                "page_num": page_num,
                "word_count": 0,
                "raw_data": {}
            }
    
    def process_pdf(
        self,
        pdf_path: str,
        save_intermediate: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process entire PDF document.
        
        Args:
            pdf_path: Path to PDF file
            save_intermediate: Whether to save intermediate images
            output_dir: Output directory for results
            
        Returns:
            Dictionary with full OCR results
        """
        pdf_name = Path(pdf_path).stem
        print(f"\nProcessing PDF: {pdf_name}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        print(f"Converted to {len(images)} pages")
        
        if not images:
            return {
                "pdf_name": pdf_name,
                "pages": [],
                "full_text": "",
                "total_pages": 0,
                "avg_confidence": 0
            }
        
        # Process each page
        pages = []
        full_text = []
        
        for i, image in enumerate(tqdm(images, desc="OCR Processing")):
            result = self.ocr_image(image, page_num=i+1)
            pages.append(result)
            full_text.append(result["text"])
        
        # Combine all pages
        combined_text = "\n\n".join(full_text)
        
        # Calculate overall statistics
        confidences = [p["confidence"] for p in pages if p["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        result = {
            "pdf_name": pdf_name,
            "pages": pages,
            "full_text": combined_text,
            "total_pages": len(pages),
            "avg_confidence": avg_confidence,
            "total_words": sum(p["word_count"] for p in pages)
        }
        
        # Save results
        if output_dir:
            self._save_results(result, output_dir)
        
        return result
    
    def _save_results(
        self,
        result: Dict,
        output_dir: str
    ):
        """Save OCR results to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        pdf_name = result["pdf_name"]
        
        # Save full text
        txt_path = Path(output_dir) / f"{pdf_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result["full_text"])
        
        # Save metadata
        metadata_path = Path(output_dir) / f"{pdf_name}_metadata.json"
        metadata = {
            "pdf_name": result["pdf_name"],
            "total_pages": result["total_pages"],
            "avg_confidence": result["avg_confidence"],
            "total_words": result["total_words"]
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {output_dir}")
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.pdf"
    ):
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Input directory with PDFs
            output_dir: Output directory for results
            pattern: File pattern to match
        """
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob(pattern))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.process_pdf(
                str(pdf_file),
                output_dir=output_dir
            )
            results.append(result)
        
        # Save summary
        summary_path = Path(output_dir) / "ocr_summary.json"
        summary = {
            "total_files": len(results),
            "total_pages": sum(r["total_pages"] for r in results),
            "avg_confidence": sum(r["avg_confidence"] for r in results) / len(results) if results else 0,
            "files": [r["pdf_name"] for r in results]
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed {len(results)} files")
        print(f"Total pages: {summary['total_pages']}")
        print(f"Average confidence: {summary['avg_confidence']:.2f}%")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Hindi Legal OCR Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input PDF file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Path to tesseract executable"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hin+eng",
        help="Language code (default: hin+eng)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF conversion (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = HindiLegalOCR(
        tesseract_cmd=args.tesseract_cmd,
        lang=args.lang,
        dpi=args.dpi
    )
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        result = ocr.process_pdf(str(input_path), output_dir=args.output)
        print(f"\nOCR Complete!")
        print(f"Pages: {result['total_pages']}")
        print(f"Confidence: {result['avg_confidence']:.2f}%")
    elif input_path.is_dir():
        # Directory
        ocr.process_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

