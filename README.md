# RAG Starter System

A simple but powerful Retrieval-Augmented Generation (RAG) system that combines document retrieval with text generation to provide accurate and context-aware responses.

## Components

### Retriever
The retriever component is responsible for:
- Loading and processing documents
- Creating embeddings for documents and queries
- Building and maintaining a FAISS index for efficient similarity search
- Retrieving the most relevant documents for a given query

### Generator
The generator component:
- Takes a query and retrieved documents as input
- Constructs a prompt that combines the context and query
- Generates a response using a language model
- Currently supports both Ollama and OpenAI models

### Prompt Management
The system includes a flexible prompt management system:
- Prompt templates are stored in YAML files in the `prompts` directory
- Supports different prompt types (text, JSON, code) for various use cases
- Easy to modify and extend prompt templates without changing code
- Templates can include variables and formatting instructions

## Usage

### Basic Usage
```python
from rag_system.retriever import Retriever
from rag_system.generator import Generator
from rag_system.prompt_manager import PromptManager

# Initialize components
retriever = Retriever()
generator = Generator()
prompt_manager = PromptManager()

# Load and index documents
retriever.load_documents("data")
retriever.build_index()

# Process a query
query = "What is the capital of France?"
top_docs = retriever.search(query)
answer = generator.generate(query, top_docs)
print(answer)
```

### Switching Between Ollama and OpenAI

#### Using Ollama (Default)
```python
# Retriever with Ollama embeddings
retriever = Retriever(model_name='nomic-embed-text')  # Uses Ollama's embedding model

# Generator with Ollama
generator = Generator(model="deepseek-coder-v2")  # Uses Ollama's language model
```

#### Using OpenAI
```python
# Retriever with OpenAI embeddings
retriever = Retriever(
    model_name='text-embedding-ada-002',
    use_openai=True,
    openai_api_key='your-api-key'
)

# Generator with OpenAI
generator = Generator(
    model="gpt-3.5-turbo",
    use_openai=True,
    openai_api_key='your-api-key'
)
```

## Configuration

### Retriever Configuration
- `model_name`: The embedding model to use
- `use_openai`: Boolean to switch between Ollama and OpenAI
- `openai_api_key`: Required when using OpenAI
- `index_path`: Path to save/load the FAISS index
- `documents_path`: Path to save/load the document cache

### Generator Configuration
- `model`: The language model to use
- `use_openai`: Boolean to switch between Ollama and OpenAI
- `openai_api_key`: Required when using OpenAI

### Prompt Templates
The system uses YAML files for prompt templates, located in the `prompts` directory:
- `text.yaml`: General text generation prompts
- `json.yaml`: JSON-specific generation prompts
- `code.yaml`: Code generation prompts

Example prompt template structure:
```yaml
name: "text_generation"
description: "Template for general text generation"
template: |
  Context: {context}
  Question: {query}
  Please provide a detailed answer based on the context above.
variables:
  - context
  - query
```

## Requirements
- Python 3.8+
- FAISS
- NumPy
- Requests
- OpenAI (optional, for OpenAI integration)
- PyYAML (for prompt template management)

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Ollama:
   - Install Ollama from https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull nomic-embed-text
     ollama pull deepseek-coder-v2
     ```

3. For OpenAI:
   - Get an API key from https://platform.openai.com
   - Set your API key in the code or environment variables

## Data Format
Place your documents in the `data` directory. The system will process all text files in this directory.

## Notes
- The system automatically caches the FAISS index and document embeddings for faster subsequent runs
- For best results, ensure your documents are well-structured and relevant to your use case
- The number of retrieved documents can be adjusted using the `top_k` parameter in the `search` method
- Prompt templates can be easily modified by editing the YAML files in the `prompts` directory