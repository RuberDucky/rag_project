"""Configuration management for the RAG application."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Config
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Database Config
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    
    # Ollama Config
    OLLAMA_BASE_URL: str = "http://192.168.18.7:11434"
    EMBEDDING_MODEL: str = "qwen3-embedding:latest"
    CHAT_MODEL: str = "lfm2.5-thinking"
    
    # RAG Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIMENSION: int = 4096
    
    # Upload Config
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    @property
    def database_url(self) -> str:
        """Get the full database URL."""
        return f"postgres://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def tortoise_orm_config(self) -> dict:
        """Get Tortoise ORM configuration."""
        return {
            "connections": {
                "default": self.database_url
            },
            "apps": {
                "models": {
                    "models": ["src.models"],
                    "default_connection": "default",
                }
            },
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
