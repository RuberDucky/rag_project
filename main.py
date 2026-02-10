"""Main FastAPI application for RAG Chatbot."""
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from src.routes import documents, chat
from src.schemas import HealthCheck
from src.ollama_client import OllamaClient
from src.config import get_settings
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


# Function to setup pgvector tables
async def setup_pgvector():
    """Setup pgvector extension and tables."""
    try:
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
                embedding vector(4096),
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS document_embeddings_chunk_id_idx 
            ON document_embeddings(chunk_id);
        """)
        
        # Note: pgvector indexes (HNSW and IVFFlat) only support up to 2000 dimensions
        # For 4096 dimensions, we use sequential scan (fine for small-medium datasets)
        # For large datasets, consider using dimensionality reduction or a different vector DB
        logger.info("Using sequential scan for similarity search (4096 dimensions > 2000 limit)")
        
        logger.info("pgvector setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up pgvector: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Chatbot application...")
    # pgvector setup will be done after Tortoise init via register_tortoise
    await setup_pgvector()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown handled by register_tortoise
    logger.info("Shutting down RAG Chatbot application...")


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Customer support chatbot with RAG using LangChain, Ollama, and pgvector",
    version="1.0.0",
    lifespan=lifespan
)

# Register Tortoise ORM with FastAPI
register_tortoise(
    app,
    db_url=settings.database_url,
    modules={"models": ["src.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    # Check database
    db_status = "healthy"
    try:
        conn = Tortoise.get_connection("default")
        await conn.execute_query("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")
    
    # Check Ollama
    ollama_status = "healthy"
    try:
        ollama = OllamaClient()
        is_healthy = await ollama.health_check()
        if not is_healthy:
            ollama_status = "unhealthy: unreachable"
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"
        logger.error(f"Ollama health check failed: {e}")
    
    overall_status = "healthy" if db_status == "healthy" and ollama_status == "healthy" else "degraded"
    
    return HealthCheck(
        status=overall_status,
        database=db_status,
        ollama=ollama_status,
        timestamp=datetime.now()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )   

