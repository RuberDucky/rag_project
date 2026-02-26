"""API routes for chat functionality."""
from fastapi import APIRouter, HTTPException, status, Query
from typing import List
from uuid import UUID
from datetime import datetime

from src.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationDetail,
    MessageItem
)
from src.chat_engine import RAGChatEngine
from src.models import Conversation, Message
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

chat_engine = RAGChatEngine()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get AI response with RAG.
    
    If session_id is provided, continues existing conversation.
    Otherwise, creates a new conversation.
    """
    try:
        result = await chat_engine.chat(
            user_message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return ChatResponse(
            user_id=result["user_id"],
            session_id=result["session_id"],
            message=request.message,
            response=result["response"],
            context=result["context"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/conversations", response_model=List[ConversationHistory])
async def list_conversations(user_id: UUID | None = Query(default=None)):
    """Get list of conversations, optionally filtered by user_id."""
    query = Conversation.all()
    if user_id:
        query = query.filter(user_id=user_id)
    conversations = await query.order_by("-updated_at")
    
    result = []
    for conv in conversations:
        message_count = await Message.filter(conversation_id=conv.id).count()
        result.append(
            ConversationHistory(
                user_id=conv.user_id,
                session_id=conv.session_id,
                title=conv.title,
                created_at=conv.created_at,
                message_count=message_count
            )
        )
    
    return result


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(session_id: UUID, user_id: UUID | None = Query(default=None)):
    """Get conversation details with all messages."""
    query = Conversation.filter(session_id=session_id)
    if user_id:
        query = query.filter(user_id=user_id)
    conversation = await query.first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = await Message.filter(
        conversation_id=conversation.id
    ).order_by("timestamp")
    
    return ConversationDetail(
        user_id=conversation.user_id,
        session_id=conversation.session_id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[MessageItem.model_validate(msg) for msg in messages]
    )


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: UUID, user_id: UUID | None = Query(default=None)):
    """Delete a conversation and all its messages."""
    query = Conversation.filter(session_id=session_id)
    if user_id:
        query = query.filter(user_id=user_id)
    conversation = await query.first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    await conversation.delete()
    logger.info(f"Deleted conversation {session_id}")
    
    return {"message": "Conversation deleted successfully"}
