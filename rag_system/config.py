import os
from typing import Dict, Any

# Model Configuration
USE_OPENAI = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI Models
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_GENERATION_MODEL = "gpt-3.5-turbo"

# Ollama Models
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "deepseek-coder-v2"

# Retriever Configuration
INDEX_PATH = "index"
DOCUMENTS_PATH = "documents"
TOP_K = 3

# Generator Configuration
DEFAULT_TEMPLATE = "text"

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "use_openai": USE_OPENAI,
        "openai_api_key": OPENAI_API_KEY,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "openai_generation_model": OPENAI_GENERATION_MODEL,
        "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL,
        "ollama_generation_model": OLLAMA_GENERATION_MODEL,
        "index_path": INDEX_PATH,
        "documents_path": DOCUMENTS_PATH,
        "top_k": TOP_K,
        "default_template": DEFAULT_TEMPLATE
    }
