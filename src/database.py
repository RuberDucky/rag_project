"""Database initialization and management."""
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise
from src.config import get_settings
import logging

logger = logging.getLogger(__name__)


async def init_db():
    """Initialize database connection and create tables."""
    settings = get_settings()
    
    try:
        # Initialize Tortoise ORM with explicit configuration
        await Tortoise.init(
            db_url=settings.database_url,
            modules={"models": ["src.models"]},
            _create_db=True
        )
        await Tortoise.generate_schemas()
        
        logger.info("Tortoise ORM initialized successfully")
        
        # Create pgvector extension and vector table if not exists
        conn = Tortoise.get_connection("default")
        
        # Enable pgvector extension
        await conn.execute_script("""
            CREATE EXTENSION IF NOT EXISTS vector;
        """)
        
        # Create vector embeddings table
        await conn.execute_script("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                chunk_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
                embedding vector(1024),
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS document_embeddings_chunk_id_idx 
            ON document_embeddings(chunk_id);
        """)
        
        # Create HNSW index for fast similarity search
        try:
            await conn.execute_script("""
                CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
                ON document_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
        except Exception as e:
            logger.warning(f"HNSW index already exists or error: {e}")
        
        logger.info("Database initialized successfully with pgvector support")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db():
    """Close database connections."""
    await Tortoise.close_connections()
    logger.info("Database connections closed")
