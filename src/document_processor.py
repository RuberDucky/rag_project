"""Document processing and text extraction."""
import os
from pathlib import Path
from typing import List, Optional
import logging
from pypdf import PdfReader
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document types and extract text."""
    
    SUPPORTED_TYPES = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain'
    }
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            raise
    
    @classmethod
    def extract_text(cls, file_path: str, file_type: str) -> str:
        """Extract text from document based on file type."""
        ext = file_type.lower().replace('.', '')
        
        if ext == 'pdf':
            return cls.extract_text_from_pdf(file_path)
        elif ext in ['docx', 'doc']:
            return cls.extract_text_from_docx(file_path)
        elif ext == 'txt':
            return cls.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """Check if file type is supported."""
        ext = Path(filename).suffix.lower().replace('.', '')
        return ext in DocumentProcessor.SUPPORTED_TYPES.keys()
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename."""
        return Path(filename).suffix.lower()
