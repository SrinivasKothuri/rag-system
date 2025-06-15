from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI

class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        pass
    
    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion for the given prompt."""
        pass

class OllamaWrapper(ModelWrapper):
    """Wrapper for Ollama models."""
    
    def __init__(self, embedding_model: str, generation_model: str):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.base_url = "http://localhost:11434/api"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama's embedding model."""
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion using Ollama's generation model."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.generation_model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

class OpenAIWrapper(ModelWrapper):
    """Wrapper for OpenAI models."""
    
    def __init__(self, embedding_model: str, generation_model: str, api_key: str):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.client = OpenAI(api_key=api_key)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's embedding model."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion using OpenAI's generation model."""
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

def create_model_wrapper(config: Dict[str, Any]) -> ModelWrapper:
    """Factory function to create the appropriate model wrapper based on configuration."""
    if config.get("use_openai", False):
        if not config.get("openai_api_key"):
            raise ValueError("OpenAI API key is required when use_openai is True")
        return OpenAIWrapper(
            embedding_model=config["openai_embedding_model"],
            generation_model=config["openai_generation_model"],
            api_key=config["openai_api_key"]
        )
    else:
        return OllamaWrapper(
            embedding_model=config["ollama_embedding_model"],
            generation_model=config["ollama_generation_model"]
        ) 