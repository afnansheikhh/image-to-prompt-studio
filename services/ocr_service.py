"""
Modular OCR extraction service.
Supports pytesseract, system tesseract CLI, or graceful heuristic fallback.
"""

import subprocess
import shutil
import re
from typing import Dict, Any, List
from PIL import Image


class OCRService:
    """Service to extract text, numeric values, and structural layout from images."""

    _tesseract_cmd = None

    @classmethod
    def find_tesseract(cls) -> bool:
        """Check if tesseract CLI or pytesseract is available."""
        if cls._tesseract_cmd is not None:
            return cls._tesseract_cmd != ""

        # Check in common system paths
        paths_to_check = [
            shutil.which("tesseract"),
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/usr/bin/tesseract",
        ]
        for p in paths_to_check:
            if p and shutil.which(p):
                cls._tesseract_cmd = p
                return True

        cls._tesseract_cmd = ""
        return False

    @classmethod
    def extract_text(cls, image: Image.Image) -> Dict[str, Any]:
        """
        Extract text lines, numbers, and layout from image.
        Returns dictionary with text list, full string, and structural stats.
        """
        rgb_img = image.convert("RGB")

        extracted_lines: List[str] = []
        raw_text = ""

        # Attempt 1: System Tesseract CLI if available
        if cls.find_tesseract():
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    rgb_img.save(tmp.name)
                    cmd = [cls._tesseract_cmd, tmp.name, "stdout", "--oem", "1", "-l", "eng"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if proc.returncode == 0:
                        raw_text = proc.stdout
                        extracted_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            except Exception:
                pass

        # Attempt 2: pytesseract package if installed
        if not extracted_lines:
            try:
                import pytesseract
                raw_text = pytesseract.image_to_string(rgb_img)
                extracted_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            except Exception:
                pass

        # Filter and structure findings
        clean_lines = [re.sub(r"[^\w\s\$\%\.\,\:\-\+\/\(\)]", "", l).strip() for l in extracted_lines if len(l.strip()) > 1]
        
        # Categorize numbers, labels, potential titles
        numeric_values = []
        potential_titles = []
        labels = []

        for line in clean_lines:
            digits = re.findall(r"[-+]?\d*\.?\d+(?:%|\$|k|m|b)?", line, re.IGNORECASE)
            if digits:
                numeric_values.extend(digits)
            if len(line.split()) <= 4 and not digits:
                labels.append(line)
            elif len(line.split()) > 4:
                potential_titles.append(line)

        # Estimate text density
        text_char_count = sum(len(l) for l in clean_lines)
        has_dense_text = text_char_count > 150
        has_moderate_text = 20 <= text_char_count <= 150

        return {
            "has_text": len(clean_lines) > 0,
            "raw_text": "\n".join(clean_lines),
            "lines": clean_lines,
            "line_count": len(clean_lines),
            "char_count": text_char_count,
            "numeric_values": list(set(numeric_values))[:15],
            "labels": labels[:10],
            "potential_titles": potential_titles[:5],
            "is_dense": has_dense_text,
            "is_moderate": has_moderate_text,
        }
