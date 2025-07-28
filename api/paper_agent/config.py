"""Configuration settings for the Paper Agent system.

This module contains all configuration settings used throughout the application.
It uses environment variables with sensible defaults for development.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "Paper Agent"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # LLM settings
    LLM_PROVIDER: str = "local"
    LLM_MODEL: str = "llama3:8b-instruct-q4_K_M"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Knowledge base settings
    CHROMA_PERSIST_DIR: str = "data/chroma"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Grobid settings
    GROBID_URL: str = "http://localhost:8070/api"
    
    # Neo4j settings for Docker environment
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # Model configuration
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Create settings instance
settings = Settings()

# Export settings
__all__ = ['settings']
