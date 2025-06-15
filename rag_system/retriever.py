import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from .model_wrapper import create_model_wrapper
from .config import get_config

class Retriever:
    def __init__(self):
        self.config = get_config()
        self.model = create_model_wrapper(self.config)
        self.index = None
        self.documents = []
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if os.path.exists(self.config["index_path"]):
            self.index = faiss.read_index(self.config["index_path"])
            self._load_documents()
        # Don't create index here - we'll create it when we know the embedding dimension
    
    def _load_documents(self):
        """Load cached documents."""
        if os.path.exists(self.config["documents_path"]):
            with open(self.config["documents_path"], 'r') as f:
                self.documents = json.load(f)
    
    def _save_documents(self):
        """Save documents to cache."""
        with open(self.config["documents_path"], 'w') as f:
            json.dump(self.documents, f)

    def _initialize_index(self, embedding_dim: int):
        """Initialize the FAISS index with the correct dimension."""
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif self.index.d != embedding_dim:
            raise ValueError(f"Index dimension mismatch. Expected {self.index.d}, got {embedding_dim}")

    def load_documents(self, data_dir: str):
        """Load and process documents from the specified directory."""
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding="utf-8") as f:
                    content = f.read()
                    embedding = self.model.generate_embedding(content)
                    self.documents.append({
                        'content': content,
                        'metadata': {'source': filename}
                    })
                    # Create a contiguous array with the correct shape and type
                    embedding_array = np.array(embedding, dtype=np.float32)
                    if embedding_array.ndim == 1:
                        embedding_array = embedding_array.reshape(1, -1)
                    
                    # Initialize index if needed and validate dimension
                    self._initialize_index(embedding_array.shape[1])
                    self.index.add(embedding_array)

        # Save the updated index and documents
        faiss.write_index(self.index, self.config["index_path"])
        self._save_documents()
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for the most relevant documents."""
        if top_k is None:
            top_k = self.config["top_k"]
            
        query_embedding = self.model.generate_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            top_k
        )
        
        return [self.documents[i] for i in indices[0]]