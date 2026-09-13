"""Base configuration module for Production RAG Knowledge System.

This module provides initial configuration settings loaded from environment variables.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Base application settings."""

    app_name: str = "Production RAG Knowledge System"
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
