"""OCR Processing for RAG System - SOTA Implementation.

This module provides:
- OCR for scanned PDFs and images
- Multiple OCR engine support (EasyOCR, Tesseract, PaddleOCR)
- Confidence-based text filtering
"""

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger

from lasagna.core.config import rag_config

# Check available OCR engines
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.debug("easyocr not installed")

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.debug("pytesseract not installed")

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.debug("paddleocr not installed")

# Image processing
try:
    from PIL import Image
    import io

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image processing disabled.")

try:
    import pdf2image

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.debug("pdf2image not installed. PDF to image conversion disabled.")


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float
    page_number: Optional[int] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None


class OCREngine:
    """Base class for OCR engines."""

    def __init__(self, languages: Optional[List[str]] = None):
        """Initialize OCR engine.

        Args:
            languages: Languages to detect
        """
        self.languages = languages if languages is not None else rag_config.ocr_languages

    def process_image(self, image: "Image.Image") -> List[OCRResult]:
        """Process an image and extract text.

        Args:
            image: PIL Image

        Returns:
            List of OCR results
        """
        raise NotImplementedError


class EasyOCREngine(OCREngine):
    """EasyOCR-based OCR engine."""

    def __init__(self, languages: Optional[List[str]] = None):
        super().__init__(languages)
        self._reader = None

    @property
    def reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            if not EASYOCR_AVAILABLE:
                raise ImportError("easyocr not installed. Install with: pip install easyocr")
            self._reader = easyocr.Reader(self.languages, gpu=rag_config.use_gpu)
            logger.info(f"EasyOCR reader initialized for languages: {self.languages}")
        return self._reader

    def process_image(self, image: "Image.Image") -> List[OCRResult]:
        """Process image with EasyOCR."""
        import numpy as np

        # Convert PIL to numpy array
        image_array = np.array(image)

        results = []
        try:
            ocr_results = self.reader.readtext(image_array)

            for bbox, text, confidence in ocr_results:
                if confidence >= rag_config.ocr_confidence_threshold:
                    # Convert bbox to simple tuple
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    bounding_box = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords)),
                    )

                    results.append(
                        OCRResult(text=text, confidence=confidence, bounding_box=bounding_box)
                    )

        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")

        return results


class TesseractEngine(OCREngine):
    """Tesseract-based OCR engine."""

    def __init__(self, languages: Optional[List[str]] = None):
        super().__init__(languages)
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")

    def process_image(self, image: "Image.Image") -> List[OCRResult]:
        """Process image with Tesseract."""
        try:
            # Get detailed results with confidence
            lang = "+".join(self.languages)
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

            results = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if text and conf >= rag_config.ocr_confidence_threshold * 100:
                    results.append(
                        OCRResult(
                            text=text,
                            confidence=conf / 100,
                            bounding_box=(
                                data["left"][i],
                                data["top"][i],
                                data["left"][i] + data["width"][i],
                                data["top"][i] + data["height"][i],
                            ),
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            return []


class PaddleOCREngine(OCREngine):
    """PaddleOCR-based OCR engine."""

    def __init__(self, languages: Optional[List[str]] = None):
        super().__init__(languages)
        self._ocr = None

    @property
    def ocr(self):
        """Lazy load PaddleOCR."""
        if self._ocr is None:
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("paddleocr not installed. Install with: pip install paddleocr")

            # Map language codes
            lang = "en" if "en" in self.languages else self.languages[0]
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=rag_config.use_gpu)
            logger.info(f"PaddleOCR initialized for language: {lang}")
        return self._ocr

    def process_image(self, image: "Image.Image") -> List[OCRResult]:
        """Process image with PaddleOCR."""
        import numpy as np

        image_array = np.array(image)

        results = []
        try:
            ocr_results = self.ocr.ocr(image_array, cls=True)

            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox, (text, confidence) = line

                    if confidence >= rag_config.ocr_confidence_threshold:
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]

                        results.append(
                            OCRResult(
                                text=text,
                                confidence=confidence,
                                bounding_box=(
                                    int(min(x_coords)),
                                    int(min(y_coords)),
                                    int(max(x_coords)),
                                    int(max(y_coords)),
                                ),
                            )
                        )

        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")

        return results


class OCRProcessor:
    """Main OCR processor with engine selection and fallback."""

    def __init__(self, engine: Optional[str] = None):
        """Initialize OCR processor.

        Args:
            engine: OCR engine to use (easyocr, tesseract, paddleocr)
        """
        self.engine_name = engine or rag_config.ocr_engine
        self._engine: Optional[OCREngine] = None

    def _get_engine(self) -> OCREngine:
        """Get or create OCR engine with fallback."""
        if self._engine is not None:
            return self._engine

        engine_order = [self.engine_name]
        # Add fallbacks
        if self.engine_name != "easyocr":
            engine_order.append("easyocr")
        if self.engine_name != "tesseract":
            engine_order.append("tesseract")
        if self.engine_name != "paddleocr":
            engine_order.append("paddleocr")

        for engine in engine_order:
            try:
                if engine == "easyocr" and EASYOCR_AVAILABLE:
                    self._engine = EasyOCREngine()
                    logger.info("Using EasyOCR engine")
                    return self._engine
                elif engine == "tesseract" and TESSERACT_AVAILABLE:
                    self._engine = TesseractEngine()
                    logger.info("Using Tesseract engine")
                    return self._engine
                elif engine == "paddleocr" and PADDLEOCR_AVAILABLE:
                    self._engine = PaddleOCREngine()
                    logger.info("Using PaddleOCR engine")
                    return self._engine
            except Exception as e:
                logger.warning(f"Failed to initialize {engine}: {e}")
                continue

        raise RuntimeError("No OCR engine available. Install easyocr, pytesseract, or paddleocr.")

    def process_pdf(self, pdf_path: str) -> str:
        """Process a PDF file with OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        if not rag_config.enable_ocr:
            return ""

        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available for PDF OCR")
            return ""

        if not PIL_AVAILABLE:
            logger.warning("Pillow not available for image processing")
            return ""

        logger.info(f"Processing PDF with OCR: {pdf_path}")

        try:
            # Get page count first to log progress without loading all pages
            import pypdf as _pypdf_count

            with open(pdf_path, "rb") as _f:
                _total_pages = len(_pypdf_count.PdfReader(_f).pages)

            engine = self._get_engine()
            all_text = []

            # Process pages one at a time to avoid OOM on large PDFs
            for page_num in range(1, _total_pages + 1):
                logger.debug(f"Processing page {page_num}/{_total_pages}")
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    first_page=page_num,
                    last_page=page_num,
                )
                if not images:
                    continue

                image = images[0]
                results = engine.process_image(image)
                # Release PIL image immediately
                image.close()
                del images

                # Sort by position (top to bottom, left to right)
                results.sort(
                    key=lambda r: (
                        r.bounding_box[1] if r.bounding_box else 0,
                        r.bounding_box[0] if r.bounding_box else 0,
                    )
                )

                page_text = " ".join(r.text for r in results)
                if page_text.strip():
                    all_text.append(f"[Page {page_num}]\n{page_text}")

            full_text = "\n\n".join(all_text)
            logger.info(f"OCR extracted {len(full_text)} characters from {_total_pages} pages")
            return full_text

        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return ""

    def process_image_file(self, image_path: str) -> str:
        """Process an image file with OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        if not rag_config.enable_ocr:
            return ""

        if not PIL_AVAILABLE:
            logger.warning("Pillow not available for image processing")
            return ""

        logger.info(f"Processing image with OCR: {image_path}")

        try:
            image = Image.open(image_path)
            engine = self._get_engine()
            results = engine.process_image(image)

            # Sort by position
            results.sort(
                key=lambda r: (
                    r.bounding_box[1] if r.bounding_box else 0,
                    r.bounding_box[0] if r.bounding_box else 0,
                )
            )

            text = " ".join(r.text for r in results)
            logger.info(f"OCR extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return ""

    def process_image_bytes(self, image_bytes: bytes) -> str:
        """Process image bytes with OCR.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Extracted text
        """
        if not rag_config.enable_ocr or not PIL_AVAILABLE:
            return ""

        try:
            image = Image.open(io.BytesIO(image_bytes))
            engine = self._get_engine()
            results = engine.process_image(image)

            text = " ".join(r.text for r in results)
            return text

        except Exception as e:
            logger.error(f"Image bytes OCR failed: {e}")
            return ""


def is_scanned_pdf(pdf_path: str) -> bool:
    """Check if a PDF is likely a scanned document.

    Args:
        pdf_path: Path to PDF

    Returns:
        True if PDF appears to be scanned/image-based
    """
    try:
        import pypdf

        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)

            # Check first few pages for text content
            text_found = False
            pages_to_check = min(3, len(reader.pages))

            for i in range(pages_to_check):
                text = reader.pages[i].extract_text()
                if text and len(text.strip()) > 100:
                    text_found = True
                    break

            return not text_found

    except Exception as e:
        logger.warning(f"Could not determine if PDF is scanned: {e}")
        return False


# Module-level singleton with thread-safe initialization
_ocr_processor: Optional[OCRProcessor] = None
_ocr_processor_lock = threading.Lock()


def get_ocr_processor() -> OCRProcessor:
    """Get or create the OCR processor singleton (thread-safe).

    Returns:
        OCRProcessor instance
    """
    global _ocr_processor
    if _ocr_processor is None:
        with _ocr_processor_lock:
            if _ocr_processor is None:
                _ocr_processor = OCRProcessor()
    return _ocr_processor


def reset_ocr_processor():
    """Reset the OCR processor singleton."""
    global _ocr_processor
    with _ocr_processor_lock:
        _ocr_processor = None
