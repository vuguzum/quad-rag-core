# qdrant_rag_core/file_processor.py
import os
import mimetypes
from pathlib import Path
from typing import List
from .utils import is_text_file
# qdrant_rag_core/file_processor.py
import re
from typing import List
from typing import List, Optional
from .config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_RATIO
import fitz  # pymupdf
import pdfplumber
import PyPDF2

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: float = CHUNK_OVERLAP_RATIO,
    only_indices: Optional[List[int]] = None
) -> List[str]:
    """
    Splits text into chunks. If only_indices is specified — returns only specified chunks.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, int(chunk_size * (1 - overlap)))
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        if only_indices is not None and chunk_id not in only_indices:
            # Skip unnecessary chunks — only move the pointer
            i += step
            chunk_id += 1
            continue

        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        i += step
        chunk_id += 1

    return chunks

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    
def process_file(file_path: str, content_types: List[str]) -> str|None:
    mime, _ = mimetypes.guess_type(file_path)
    if not mime:
        return None

    if "text" in content_types and (mime.startswith("text/") or is_text_file(file_path)):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            return normalize_text(text)
    elif "pdf" in content_types and mime == "application/pdf":
        text = _extract_text_from_pdf(file_path)
        return normalize_text(text)
    # Other types — later
    return None

def _extract_text_from_pdf(file_path: str) -> str:
    if not file_path.endswith(".pdf"):
        return ""
    try:
            # Attempt 1: PyPDF2
            text = _extract_pdf_PyPDF2(file_path)
            if text.strip():
                return text
    except Exception as e:
        print(f"[DEBUG] PyPDF2 failed for {file_path}: {e}")
    try:
            # Attempt 2: fitz
            text = _extract_pdf_fitz(file_path)
            if text.strip():
                return text
    except Exception as e:
        print(f"[DEBUG] fitz failed for {file_path}: {e}")
    try:
            # Attempt 3: pdfplumber
            text = _extract_pdf_pdfplumber(file_path)
            if text.strip():
                return text
    except Exception as e:
        print(f"[DEBUG] pdfplumber failed for {file_path}: {e}")
    return ""


def _extract_pdf_PyPDF2(path: str) -> str:
    try:
        text = ""

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages)
    except Exception:
        return text
    
def _extract_pdf_fitz(file_path: str) -> str:
    """Extracts text from PDF using fitz."""
    try:
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n" # type: ignore
        doc.close()
        if text.strip():
            return text
    except Exception as e:
        print(f"[DEBUG] fitz failed for {file_path}: {e}")

    return text

def _extract_pdf_pdfplumber(file_path: str) -> str:
    """Extracts text from PDF using pdfplumber."""
    text = ""
    # Attempt 1: pdfplumber (better with tables and columns)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"[DEBUG] pdfplumber failed for {file_path}: {e}")

    return text    



    

    