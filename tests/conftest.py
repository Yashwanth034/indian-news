"""Shared pytest fixtures for IndiaNews."""
import pytest

from src.config_loader import get_config


@pytest.fixture(scope="session")
def config():
    return get_config()
