"""
RAG System - A simple but powerful Retrieval-Augmented Generation system.
"""

from .retriever import Retriever
from .generator import Generator
from .config import get_config

__version__ = "0.1.0"
__all__ = ["Retriever", "Generator", "get_config"] 