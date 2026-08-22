"""
Configuration Settings

Environment-based configuration for OpenEvolve API service
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "OpenEvolve API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # BubbleLab Service URLs
    BUBBLELAB_API_URL: str = "http://localhost:3001"
    JUDGE_API_URL: str = "http://localhost:3001/api/evolution-judge"
    MUTATE_API_URL: str = "http://localhost:3001/api/evolution-mutate"
    LEANAIDE_API_URL: str = "http://localhost:3001/api/leanaide"
    Z3_API_URL: str = "http://localhost:7655"

    # Timeouts (in seconds)
    JUDGE_TIMEOUT: float = 60.0
    MUTATE_TIMEOUT: float = 60.0
    LEANAIDE_TIMEOUT: float = 120.0
    Z3_TIMEOUT: float = 60.0

    # Execution
    MAX_WORKERS: int = 5
    TASK_TIMEOUT: float = 600.0  # 10 minutes
    HEARTBEAT_INTERVAL: float = 5.0

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8001",
    ]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()
