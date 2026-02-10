"""API routes for chat functionality."""
from fastapi import APIRouter, HTTPException, status
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
            session_id=request.session_id
        )
        
        return ChatResponse(
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
async def list_conversations():
    """Get list of all conversations."""
    conversations = await Conversation.all().order_by("-updated_at")
    
    result = []
    for conv in conversations:
        message_count = await Message.filter(conversation_id=conv.id).count()
        result.append(
            ConversationHistory(
                session_id=conv.session_id,
                title=conv.title,
                created_at=conv.created_at,
                message_count=message_count
            )
        )
    
    return result


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(session_id: UUID):
    """Get conversation details with all messages."""
    conversation = await Conversation.get_or_none(session_id=session_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = await Message.filter(
        conversation_id=conversation.id
    ).order_by("timestamp")
    
    return ConversationDetail(
        session_id=conversation.session_id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[MessageItem.model_validate(msg) for msg in messages]
    )


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: UUID):
    """Delete a conversation and all its messages."""
    conversation = await Conversation.get_or_none(session_id=session_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    await conversation.delete()
    logger.info(f"Deleted conversation {session_id}")
    
    return {"message": "Conversation deleted successfully"}
