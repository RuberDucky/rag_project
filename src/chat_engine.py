"""Conversation memory management."""
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from src.models import Conversation, Message
from src.ollama_client import OllamaClient
from src.vector_store import VectorStore
from src.schemas import ContextItem
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manage conversation history and context."""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.vector_store = VectorStore()
        self.max_history = 10  # Keep last 10 messages for context
    
    async def get_or_create_conversation(
        self,
        user_id: UUID,
        session_id: Optional[UUID] = None
    ) -> Conversation:
        """Get existing conversation or create new one."""
        if session_id:
            conversation = await Conversation.get_or_none(session_id=session_id, user_id=user_id)
            if conversation:
                return conversation
        
        # Create new conversation
        conversation = await Conversation.create(
            user_id=user_id,
            session_id=session_id or uuid4()
        )
        logger.info(f"Created new conversation: {conversation.session_id}")
        return conversation
    
    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        context: Optional[List[Dict]] = None
    ) -> Message:
        """Add message to conversation."""
        message = await Message.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            context_used=context
        )
        return message
    
    async def get_conversation_history(
        self,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation message history."""
        query = Message.filter(conversation_id=conversation_id).order_by("-timestamp")
        
        if limit:
            query = query.limit(limit)
        
        messages = await query
        return list(reversed(messages))  # Return in chronological order
    
    async def format_history_for_llm(
        self,
        conversation_id: int
    ) -> List[Dict[str, str]]:
        """Format conversation history for LLM."""
        messages = await self.get_conversation_history(
            conversation_id,
            limit=self.max_history
        )
        
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return formatted
    
    async def generate_conversation_title(
        self,
        conversation_id: int
    ) -> str:
        """Generate a title for the conversation based on first messages."""
        messages = await self.get_conversation_history(conversation_id, limit=2)
        
        if not messages:
            return "New Conversation"
        
        first_message = messages[0].content
        # Simple title generation - take first 50 chars
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."
        
        return title


class RAGChatEngine:
    """Main RAG chat engine combining retrieval and generation."""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
    
    async def chat(
        self,
        user_message: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None
    ) -> Dict:
        """
        Process chat message with RAG.
        
        Args:
            user_message: User's message
            user_id: User ID for ownership scoping
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary with response and metadata
        """
        resolved_user_id = user_id or uuid4()

        # Get or create conversation for this user
        conversation = await self.memory.get_or_create_conversation(
            user_id=resolved_user_id,
            session_id=session_id
        )
        
        # Retrieve relevant context from vector store
        relevant_chunks = await self.vector_store.similarity_search(
            query=user_message,
            user_id=resolved_user_id
        )
        
        # Format context for prompt
        context_text = self._format_context(relevant_chunks)
        
        # Get conversation history
        history = await self.memory.format_history_for_llm(conversation.id)
        
        # Build prompt with context and history
        system_prompt = self._build_system_prompt(context_text)
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await self.ollama.chat(messages)
        
        # Save messages to database
        await self.memory.add_message(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
            context=[chunk["content"] for chunk in relevant_chunks]
        )
        
        await self.memory.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
            context=None
        )
        
        # Update conversation title if first exchange
        if await Message.filter(conversation_id=conversation.id).count() <= 2:
            title = await self.memory.generate_conversation_title(conversation.id)
            conversation.title = title
            await conversation.save()
        
        # Format context for response
        context_items = [
            ContextItem(
                content=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                source=chunk["source"],
                score=chunk["score"]
            )
            for chunk in relevant_chunks[:3]  # Return top 3 for display
        ]
        
        return {
            "user_id": conversation.user_id,
            "session_id": conversation.session_id,
            "response": response,
            "context": context_items
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {idx}: {chunk['source']}]\n{chunk['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context."""
        return f"""You are a helpful customer support assistant. Use the following context from our knowledge base to answer user questions accurately and helpfully.

If the context contains relevant information, use it to provide detailed and accurate answers. If the context doesn't contain relevant information, you can still try to help based on your general knowledge, but mention that the information might not be from the official knowledge base.

Always be polite, professional, and aim to resolve the user's query effectively.

CONTEXT:
{context}

Remember to:
1. Cite sources when using information from the context
2. Be concise but thorough
3. Ask clarifying questions if needed
4. Admit if you don't know something rather than making up information
"""
