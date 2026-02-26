"""Vector store operations with pgvector."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from tortoise import Tortoise
from src.models import DocumentChunk
from src.ollama_client import OllamaClient
from src.config import get_settings
import logging
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector embeddings and similarity search with pgvector."""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.settings = get_settings()
    
    async def add_chunks_with_embeddings(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Add document chunks and their embeddings to the database.
        
        Args:
            document_id: Document ID
            chunks: List of chunk dictionaries with content and metadata
            
        Returns:
            Number of chunks added
        """
        conn = Tortoise.get_connection("default")
        added_count = 0
        
        try:
            for chunk_data in chunks:
                # Create document chunk
                chunk = await DocumentChunk.create(
                    document_id=document_id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    metadata=chunk_data["metadata"]
                )
                
                # Generate embedding
                embedding = await self.ollama.get_embedding(chunk_data["content"])
                
                # Store embedding in pgvector table
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                await conn.execute_query(
                    """
                    INSERT INTO document_embeddings (chunk_id, embedding)
                    VALUES ($1, $2::vector)
                    """,
                    [chunk.id, embedding_str]
                )
                
                added_count += 1
                
                if added_count % 10 == 0:
                    logger.info(f"Processed {added_count}/{len(chunks)} chunks")
            
            logger.info(f"Successfully added {added_count} chunks with embeddings")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding chunks with embeddings: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        user_id: UUID,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for relevant chunks.
        
        Args:
            query: Search query
            user_id: User ID to scope retrieval
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if top_k is None:
            top_k = self.settings.TOP_K_RESULTS
        
        conn = Tortoise.get_connection("default")
        
        try:
            # Generate query embedding
            query_embedding = await self.ollama.get_embedding(query)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Perform similarity search using cosine distance
            results = await conn.execute_query_dict(
                """
                SELECT 
                    dc.id as chunk_id,
                    dc.content,
                    dc.metadata,
                    d.original_filename,
                    1 - (de.embedding <=> $1::vector) as similarity
                FROM document_embeddings de
                JOIN document_chunks dc ON de.chunk_id = dc.id
                JOIN documents d ON dc.document_id = d.id
                WHERE d.processed = true AND d.user_id = $2
                ORDER BY de.embedding <=> $1::vector
                LIMIT $3
                """,
                [embedding_str, str(user_id), top_k]
            )
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "source": row["original_filename"],
                    "score": float(row["similarity"])
                })
            
            logger.info(f"Found {len(formatted_results)} relevant chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def delete_document_embeddings(self, document_id: int):
        """Delete all embeddings for a document."""
        conn = Tortoise.get_connection("default")
        
        try:
            await conn.execute_query(
                """
                DELETE FROM document_embeddings
                WHERE chunk_id IN (
                    SELECT id FROM document_chunks WHERE document_id = $1
                )
                """,
                [document_id]
            )
            logger.info(f"Deleted embeddings for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise
