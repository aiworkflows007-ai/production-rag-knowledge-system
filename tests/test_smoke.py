"""Basic smoke test to ensure project environment and package structure are sound."""

from production_rag import __version__
from production_rag.config import settings


def test_package_metadata():
    assert __version__ == "0.1.0"


def test_default_settings():
    assert settings.app_name == "Production RAG Knowledge System"
    assert settings.app_env in ["development", "testing", "production"]
