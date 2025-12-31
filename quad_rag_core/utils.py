# qdrant_rag_core/utils.py
import hashlib
import os
import mimetypes
from pathlib import Path
from .config import TEXT_FILE_EXTENSIONS

def file_hash(file_path: str) -> str:
    """Calculates SHA-256 hash of file for change tracking."""
    if not os.path.exists(file_path):
        return ""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (OSError, IOError):
        return ""

def get_file_mime_type(file_path: str) -> str:
    """Determines MIME type of file."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"

def is_text_file(file_path: str) -> bool:
    """Checks if file is text (by extension and MIME)."""
    mime = get_file_mime_type(file_path)
    if mime.startswith("text/"):
        return True
    _, ext = os.path.splitext(file_path)
    return ext.lower() in TEXT_FILE_EXTENSIONS

def normalize_path(path: str) -> str:
    """Converts path to canonical form (absolute, normalized)."""
    return os.path.normpath(os.path.abspath(path))

def get_file_size_mb(file_path: str) -> float:
    """Returns file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

