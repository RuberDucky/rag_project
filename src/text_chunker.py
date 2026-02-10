"""Text chunking and splitting utilities."""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_settings
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Handle text chunking with LangChain text splitters."""
    
    def __init__(self):
        self.settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to split
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunks with content and metadata
        """
        try:
            chunks = self.splitter.split_text(text)
            
            chunk_list = []
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    "content": chunk,
                    "chunk_index": idx,
                    "metadata": metadata or {}
                }
                chunk_list.append(chunk_data)
            
            logger.info(f"Created {len(chunk_list)} chunks from text")
            return chunk_list
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def chunk_document(
        self,
        text: str,
        document_id: int,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document with specific metadata.
        
        Args:
            text: Document text
            document_id: Document database ID
            filename: Original filename
            
        Returns:
            List of chunks ready for database insertion
        """
        metadata = {
            "document_id": document_id,
            "filename": filename
        }
        
        return self.chunk_text(text, metadata)
