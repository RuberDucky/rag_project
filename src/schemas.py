"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    id: int
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    message: str


class DocumentListItem(BaseModel):
    """Schema for document list item."""
    id: int
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    processed: bool
    chunk_count: int
    
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[UUID] = None


class ContextItem(BaseModel):
    """Schema for context item from retrieval."""
    content: str
    source: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    session_id: UUID
    message: str
    response: str
    context: List[ContextItem]
    timestamp: datetime


class ConversationHistory(BaseModel):
    """Schema for conversation history."""
    session_id: UUID
    title: Optional[str]
    created_at: datetime
    message_count: int


class MessageItem(BaseModel):
    """Schema for individual message."""
    role: str
    content: str
    timestamp: datetime
    
    class Config:
        from_attributes = True


class ConversationDetail(BaseModel):
    """Schema for detailed conversation with messages."""
    session_id: UUID
    title: Optional[str]
    created_at: datetime
    messages: List[MessageItem]


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    database: str
    ollama: str
    timestamp: datetime
