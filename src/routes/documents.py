"""API routes for document management."""
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query
from typing import List
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime
from uuid import UUID

from src.models import Document
from src.schemas import DocumentUploadResponse, DocumentListItem
from src.document_processor import DocumentProcessor
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

settings = get_settings()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: UUID | None = Query(default=None)
):
    """
    Upload and process a document.
    
    Supports: PDF, DOCX, DOC, TXT files.
    """
    # Validate file type
    if not DocumentProcessor.is_supported_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported: PDF, DOCX, DOC, TXT"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    try:
        resolved_user_id = user_id or uuid.uuid4()

        # Generate unique filename
        file_ext = DocumentProcessor.get_file_extension(file.filename)
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create database record
        doc = await Document.create(
            user_id=resolved_user_id,
            filename=unique_filename,
            original_filename=file.filename,
            file_type=file_ext.replace('.', ''),
            file_size=file_size
        )
        
        # Process document asynchronously
        try:
            # Extract text
            text = DocumentProcessor.extract_text(str(file_path), file_ext)
            
            # Chunk text
            chunker = TextChunker()
            chunks = chunker.chunk_document(text, doc.id, file.filename)
            
            # Generate embeddings and store in vector database
            vector_store = VectorStore()
            chunk_count = await vector_store.add_chunks_with_embeddings(doc.id, chunks)
            
            # Update document status
            doc.processed = True
            doc.chunk_count = chunk_count
            await doc.save()
            
            logger.info(f"Successfully processed document {doc.id}: {file.filename}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc.id}: {e}")
            # Keep document record but mark as not processed
            doc.processed = False
            await doc.save()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )
        
        return DocumentUploadResponse(
            id=doc.id,
            user_id=doc.user_id,
            filename=doc.original_filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            upload_date=doc.upload_date,
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/", response_model=List[DocumentListItem])
async def list_documents(user_id: UUID | None = Query(default=None)):
    """Get list of uploaded documents, optionally filtered by user_id."""
    query = Document.all()
    if user_id:
        query = query.filter(user_id=user_id)
    documents = await query.order_by("-upload_date")
    return [DocumentListItem.model_validate(doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentListItem)
async def get_document(document_id: int, user_id: UUID | None = Query(default=None)):
    """Get details of a specific document."""
    query = Document.filter(id=document_id)
    if user_id:
        query = query.filter(user_id=user_id)
    doc = await query.first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return DocumentListItem.model_validate(doc)


@router.delete("/{document_id}")
async def delete_document(document_id: int, user_id: UUID | None = Query(default=None)):
    """Delete a document and its embeddings."""
    query = Document.filter(id=document_id)
    if user_id:
        query = query.filter(user_id=user_id)
    doc = await query.first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Delete file from disk
        file_path = UPLOAD_DIR / doc.filename
        if file_path.exists():
            os.remove(file_path)
        
        # Delete embeddings
        vector_store = VectorStore()
        await vector_store.delete_document_embeddings(doc.id)
        
        # Delete document (chunks will be cascade deleted)
        await doc.delete()
        
        logger.info(f"Deleted document {document_id}")
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
