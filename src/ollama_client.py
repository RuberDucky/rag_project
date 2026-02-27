"""Ollama client for embeddings and chat."""
import httpx
from typing import List, Dict, Any, AsyncIterator
import json
from src.config import get_settings
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL
        self.embedding_model = self.settings.EMBEDDING_MODEL
        self.chat_model = self.settings.CHAT_MODEL
        self.chat_options = {
            "temperature": self.settings.OLLAMA_TEMPERATURE,
            "top_p": self.settings.OLLAMA_TOP_P,
            "repeat_penalty": self.settings.OLLAMA_REPEAT_PENALTY,
        }
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using Ollama."""
        url = f"{self.base_url}/api/embeddings"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    url,
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", [])
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """Send chat request to Ollama."""
        url = f"{self.base_url}/api/chat"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url,
                    json={
                        "model": self.chat_model,
                        "messages": messages,
                        "stream": stream,
                        "options": self.chat_options,
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                raise

    async def chat_stream_tokens(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens from Ollama chat API."""
        url = f"{self.base_url}/api/chat"

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json={
                        "model": self.chat_model,
                        "messages": messages,
                        "stream": True,
                        "options": self.chat_options,
                    },
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        payload = json.loads(line)

                        if payload.get("done"):
                            break

                        chunk = payload.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
            except Exception as e:
                logger.error(f"Error in streaming chat: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
