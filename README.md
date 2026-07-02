# RAG Starter System - Comprehensive Guide

A simple but powerful Retrieval-Augmented Generation (RAG) system that combines document retrieval with text generation to provide accurate and context-aware responses. This guide provides detailed instructions for setup, configuration, and usage.

## Table of Contents
1. [System Components](#system-components)
2. [Detailed Setup Instructions](#detailed-setup-instructions)
3. [Configuration Guide](#configuration-guide)
4. [Usage Examples](#usage-examples)
5. [Switching Between Models](#switching-between-models)
6. [Data Management](#data-management)
7. [Troubleshooting](#troubleshooting)

---

## System Components

### 1. The Retriever Component
The retriever is the backbone of document search and context retrieval. It handles:

**Document Processing:**
- Loads documents from specified directories
- Processes various text file formats
- Chunks documents into manageable segments for better retrieval
- Removes duplicates and normalizes text formatting

**Embedding Creation:**
- Converts both documents and user queries into numerical embeddings (vectors)
- Uses either Ollama's local embedding models or OpenAI's cloud-based embeddings
- Stores embeddings efficiently for quick lookups

**Index Building & Maintenance:**
- Creates a FAISS (Facebook AI Similarity Search) index for fast similarity matching
- Supports incremental index updates as new documents are added
- Caches indices locally to avoid rebuilding on every run
- Handles index versioning and compatibility

**Document Retrieval:**
- Performs semantic similarity search using the built embeddings
- Returns the top-K most relevant documents based on query similarity
- Provides relevance scores for each retrieved document

### 2. Generator Component
The generator produces responses based on retrieved context:

**Prompt Construction:**
- Combines retrieved documents into a cohesive context window
- Integrates user queries with context-aware prompts
- Manages token limits to prevent overflow

**Response Generation:**
- Sends constructed prompts to language models (Ollama or OpenAI)
- Handles streaming or complete response generation
- Implements error handling and retry logic
- Supports multiple output formats (text, JSON, code)

**Model Support:**
- **Ollama:** Local, privacy-focused models (deepseek-coder-v2, mistral, etc.)
- **OpenAI:** Cloud-based models (gpt-3.5-turbo, gpt-4, etc.)

### 3. Prompt Management System
Flexible, template-based prompt engineering:

**Template Storage:**
- Organized in YAML format within the `prompts` directory
- Separated by type: text generation, JSON output, code generation
- Version-controlled for consistency across runs

**Dynamic Variables:**
- Templates support variable substitution
- Common variables: `{context}`, `{query}`, `{user_input}`
- Custom variables can be defined per use case

**Easy Modification:**
- Update prompts without touching code
- Test different prompt strategies quickly
- Maintain multiple templates for A/B testing

---

## Detailed Setup Instructions

### Prerequisites
Before starting, ensure you have:
- **Python 3.8 or higher** installed on your system
- **pip** (Python package manager) available in your PATH
- **Git** (optional, for version control)
- **At least 2GB of free disk space** (for models and indices)

### Step 1: Create a Project Directory

```bash
# Create a new directory for your RAG project
mkdir my-rag-system
cd my-rag-system

# Initialize a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

Create a `requirements.txt` file in your project root with the following content:

```
faiss-cpu==1.7.4
numpy==1.24.3
requests==2.31.0
pyyaml==6.0
openai==1.0.0  # Optional: only needed for OpenAI integration
ollama==0.0.1  # Optional: for Ollama integration
python-dotenv==1.0.0  # Optional: for environment variable management
```

Then install all dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If you're using a GPU, consider installing `faiss-gpu` instead of `faiss-cpu` for better performance.

### Step 3: Set Up Ollama (For Local Model Support)

If you want to use Ollama for local embeddings and language models:

#### Installation
1. Visit https://ollama.ai
2. Download and install the appropriate version for your OS (Windows, macOS, Linux)
3. Follow the installation prompts
4. Verify installation by opening a terminal and running:
   ```bash
   ollama --version
   ```

#### Pulling Required Models

```bash
# Pull the embedding model (lightweight, ~100MB)
ollama pull nomic-embed-text

# Pull the language model (larger, 5-40GB depending on model)
ollama pull deepseek-coder-v2
# Or choose from alternatives:
# ollama pull mistral
# ollama pull llama2
# ollama pull neural-chat
```

#### Starting Ollama

Ollama typically runs as a background service. To start it:

```bash
# On macOS/Linux:
ollama serve

# On Windows:
# Ollama runs automatically after installation. 
# To verify it's running, check if port 11434 is active:
curl http://localhost:11434/api/tags
```

### Step 4: Set Up OpenAI (Optional, For Cloud Models)

If you prefer using OpenAI's models:

1. **Create an OpenAI Account:**
   - Visit https://platform.openai.com
   - Sign up or log in to your account
   - Navigate to "API keys" section

2. **Generate an API Key:**
   - Click "Create new secret key"
   - Copy the key immediately (you won't see it again)
   - Store it securely

3. **Set Environment Variable:**
   ```bash
   # Create a .env file in your project root
   echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
   
   # Or set it directly in your shell
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

4. **Verify the Connection:**
   ```python
   import openai
   openai.api_key = "your-api-key"
   response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}])
   print(response)
   ```

### Step 5: Create Project Structure

```bash
# Create necessary directories
mkdir data
mkdir prompts
mkdir indices
mkdir outputs
mkdir logs

# Create the main package directory
mkdir rag_system
touch rag_system/__init__.py
touch rag_system/retriever.py
touch rag_system/generator.py
touch rag_system/prompt_manager.py
```

Your project structure should now look like:
```
my-rag-system/
├── venv/
├── data/                    # Your documents go here
├── prompts/                 # Prompt templates (YAML files)
├── indices/                 # Cached FAISS indices
├── outputs/                 # Generated responses and logs
├── logs/                    # System logs
├── rag_system/
│   ├── __init__.py
│   ├── retriever.py
│   ├── generator.py
│   └── prompt_manager.py
├── requirements.txt
├── .env                     # API keys (add to .gitignore)
└── main.py                  # Your entry point
```

### Step 6: Add Sample Data

Create sample documents in the `data` directory:

```bash
# Create a sample document
cat > data/sample.txt << 'EOF'
France is a country located in Western Europe. The capital of France is Paris, 
which is also the largest city. Paris is known for its iconic landmarks including 
the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The country has 
a rich history spanning over 2,000 years and is a major global center of art, 
fashion, gastronomy, and culture.
EOF
```

### Step 7: Initialize Prompt Templates

Create YAML prompt templates:

```bash
# Create text.yaml for general text generation
cat > prompts/text.yaml << 'EOF'
name: "text_generation"
description: "Template for general text generation"
template: |
  Context: {context}
  
  Question: {query}
  
  Instructions:
  - Provide a detailed, accurate answer based on the context above
  - If the context doesn't contain relevant information, state that clearly
  - Keep the response concise but comprehensive
  
  Answer:
variables:
  - context
  - query
EOF

# Create json.yaml for JSON output
cat > prompts/json.yaml << 'EOF'
name: "json_generation"
description: "Template for JSON structured output"
template: |
  Based on the following context, generate a JSON response with the structure shown:
  
  Context: {context}
  
  Query: {query}
  
  Return a valid JSON object with the following fields:
  - answer: The main answer
  - confidence: Confidence level (0-1)
  - sources: List of source documents used
  
  JSON Response:
variables:
  - context
  - query
EOF

# Create code.yaml for code generation
cat > prompts/code.yaml << 'EOF'
name: "code_generation"
description: "Template for code generation tasks"
template: |
  Context: {context}
  
  Code Request: {query}
  
  Instructions:
  - Generate clean, well-commented code
  - Include error handling
  - Follow best practices for the language
  - Add usage examples if applicable
  
  Generated Code:
variables:
  - context
  - query
EOF
```

---

## Configuration Guide

### Detailed Retriever Configuration

```python
from rag_system.retriever import Retriever

# Create retriever with custom configuration
retriever = Retriever(
    # Embedding model name
    model_name='nomic-embed-text',  # For Ollama
    # model_name='text-embedding-ada-002',  # For OpenAI
    
    # Model provider
    use_openai=False,  # Set to True for OpenAI
    openai_api_key='your-api-key-here',  # Required if use_openai=True
    
    # Index management
    index_path='indices/faiss_index.bin',  # Where to save/load the index
    documents_path='indices/documents.pkl',  # Where to cache documents
    
    # Search parameters
    top_k=5,  # Number of documents to retrieve
    similarity_threshold=0.3,  # Minimum similarity score
    
    # Chunk settings (for document splitting)
    chunk_size=512,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks
    
    # Performance
    batch_size=32,  # Number of embeddings to process at once
    device='cpu'  # Use 'cuda' for GPU acceleration
)

# Load and index documents
retriever.load_documents('data')  # Load all text files from data/ directory
retriever.build_index()  # Create FAISS index

# Save the index for future use
retriever.save_index()
```

### Detailed Generator Configuration

```python
from rag_system.generator import Generator

# Configuration for Ollama (local, privacy-focused)
generator_ollama = Generator(
    model="deepseek-coder-v2",  # Model name
    base_url="http://localhost:11434",  # Ollama server URL
    temperature=0.7,  # Creativity (0.0-1.0, higher = more creative)
    max_tokens=2048,  # Maximum response length
    top_p=0.9,  # Nucleus sampling parameter
    top_k=40,  # Top-K sampling parameter
    use_openai=False
)

# Configuration for OpenAI (cloud-based, more powerful)
generator_openai = Generator(
    model="gpt-3.5-turbo",  # Or gpt-4, gpt-4-turbo-preview
    openai_api_key="sk-your-api-key",
    temperature=0.7,
    max_tokens=2048,
    top_p=1.0,
    use_openai=True,
    request_timeout=30  # Timeout in seconds
)
```

### Prompt Manager Configuration

```python
from rag_system.prompt_manager import PromptManager

# Initialize prompt manager
prompt_manager = PromptManager(
    prompts_dir='prompts',  # Directory containing YAML templates
    default_template='text.yaml'  # Default template to use
)

# Load a specific template
text_prompt = prompt_manager.load_template('text.yaml')

# Load all templates
all_prompts = prompt_manager.load_all_templates()

# Get a template by name
template = prompt_manager.get_template('text_generation')
```

---

## Usage Examples

### Basic Usage Example

```python
from rag_system.retriever import Retriever
from rag_system.generator import Generator
from rag_system.prompt_manager import PromptManager

def main():
    # Step 1: Initialize components
    print("Initializing RAG system...")
    retriever = Retriever(model_name='nomic-embed-text')
    generator = Generator(model="deepseek-coder-v2")
    prompt_manager = PromptManager()
    
    # Step 2: Load and index documents
    print("Loading documents...")
    retriever.load_documents("data")
    
    print("Building index (this may take a while on first run)...")
    retriever.build_index()
    
    # Step 3: Process queries
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")
    
    # Step 4: Retrieve relevant documents
    print("Searching for relevant documents...")
    top_docs = retriever.search(query, top_k=3)
    
    # Step 5: Extract context from retrieved documents
    context = "\n\n".join([doc['content'] for doc in top_docs])
    
    # Step 6: Load prompt template
    template = prompt_manager.load_template('text.yaml')
    
    # Step 7: Generate answer
    print("Generating response...")
    answer = generator.generate(
        query=query,
        context=context,
        template=template
    )
    
    # Step 8: Display results
    print(f"\nAnswer:\n{answer}")
    
    # Optional: Display source documents
    print("\n--- Sources ---")
    for i, doc in enumerate(top_docs, 1):
        print(f"{i}. {doc['filename']} (Similarity: {doc['score']:.2%})")

if __name__ == "__main__":
    main()
```

### Advanced Multi-Query Example

```python
from rag_system.retriever import Retriever
from rag_system.generator import Generator
from rag_system.prompt_manager import PromptManager
import json

def advanced_rag_workflow():
    # Initialize with specific configurations
    retriever = Retriever(
        model_name='nomic-embed-text',
        top_k=5,
        chunk_size=512,
        chunk_overlap=50
    )
    
    generator = Generator(
        model="deepseek-coder-v2",
        temperature=0.5,
        max_tokens=1024
    )
    
    prompt_manager = PromptManager()
    
    # Load and prepare documents
    retriever.load_documents("data")
    retriever.build_index()
    
    # Define multiple queries
    queries = [
        "What is the capital of France?",
        "Tell me about Paris landmarks",
        "What is French history?"
    ]
    
    results = []
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Processing: {query}")
        print('='*50)
        
        # Retrieve documents
        top_docs = retriever.search(query, top_k=3)
        context = "\n\n".join([doc['content'] for doc in top_docs])
        
        # Generate response
        template = prompt_manager.load_template('text.yaml')
        answer = generator.generate(query, context, template)
        
        # Store results
        result = {
            'query': query,
            'answer': answer,
            'sources': [doc['filename'] for doc in top_docs]
        }
        results.append(result)
        
        print(f"Answer: {answer[:200]}...")
    
    # Save results to file
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} queries")
    print("Results saved to outputs/results.json")

if __name__ == "__main__":
    advanced_rag_workflow()
```

### JSON Output Example

```python
def generate_json_output():
    retriever = Retriever(model_name='nomic-embed-text')
    generator = Generator(model="deepseek-coder-v2")
    prompt_manager = PromptManager()
    
    # Load documents and build index
    retriever.load_documents("data")
    retriever.build_index()
    
    # Query
    query = "What is France known for?"
    top_docs = retriever.search(query, top_k=3)
    context = "\n\n".join([doc['content'] for doc in top_docs])
    
    # Use JSON template
    template = prompt_manager.load_template('json.yaml')
    json_response = generator.generate(query, context, template)
    
    # Parse and pretty-print
    try:
        parsed = json.loads(json_response)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("Response is not valid JSON")
        print(json_response)

if __name__ == "__main__":
    generate_json_output()
```

---

## Switching Between Models

### Using Ollama (Recommended for Privacy & Cost)

```python
from rag_system.retriever import Retriever
from rag_system.generator import Generator

# Retriever with Ollama embeddings
retriever = Retriever(
    model_name='nomic-embed-text',  # Lightweight embedding model
    use_openai=False
)

# Generator with different Ollama models
# Option 1: Deepseek Coder (good for technical tasks)
generator = Generator(
    model="deepseek-coder-v2",
    use_openai=False
)

# Option 2: Mistral (good balance of speed and quality)
# generator = Generator(model="mistral", use_openai=False)

# Option 3: Llama2 (larger, more capable)
# generator = Generator(model="llama2", use_openai=False)

# Option 4: Neural Chat (optimized for conversations)
# generator = Generator(model="neural-chat", use_openai=False)
```

### Using OpenAI (More Powerful, Requires API Key)

```python
from rag_system.retriever import Retriever
from rag_system.generator import Generator
import os

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')

# Retriever with OpenAI embeddings
retriever = Retriever(
    model_name='text-embedding-ada-002',
    use_openai=True,
    openai_api_key=api_key
)

# Generator with different OpenAI models
# Option 1: GPT-3.5 Turbo (fastest, most economical)
generator = Generator(
    model="gpt-3.5-turbo",
    use_openai=True,
    openai_api_key=api_key,
    temperature=0.7
)

# Option 2: GPT-4 (most capable, slower, more expensive)
# generator = Generator(
#     model="gpt-4",
#     use_openai=True,
#     openai_api_key=api_key
# )

# Option 3: GPT-4 Turbo (faster than GPT-4, more capable than 3.5)
# generator = Generator(
#     model="gpt-4-turbo-preview",
#     use_openai=True,
#     openai_api_key=api_key
# )
```

### Hybrid Approach (Ollama Retriever + OpenAI Generator)

```python
# Use local embeddings (Ollama) for privacy + cloud generation (OpenAI) for quality
retriever = Retriever(
    model_name='nomic-embed-text',
    use_openai=False  # Local embeddings
)

generator = Generator(
    model="gpt-3.5-turbo",
    use_openai=True,  # Cloud generation
    openai_api_key='your-api-key'
)
```

---

## Data Management

### Loading Documents

```python
# Load all documents from a directory
retriever.load_documents('data')

# Load documents with filtering
retriever.load_documents(
    'data',
    file_extensions=['.txt', '.md'],  # Only specific types
    recursive=True  # Include subdirectories
)

# Specify maximum file size to load
retriever.load_documents(
    'data',
    max_file_size=10*1024*1024  # 10MB limit
)
```

### Managing Indices

```python
# Build index (creates embeddings for all documents)
retriever.build_index()

# Save index to disk (for quick reloading)
retriever.save_index()

# Load previously saved index
retriever.load_index('indices/faiss_index.bin')

# Update index with new documents
retriever.add_documents('data/new_documents')
retriever.update_index()

# Clear and rebuild index
retriever.clear_index()
retriever.build_index()
```

### Document Organization Best Practices

```
data/
├── general/
│   ├── overview.txt
│   └── background.txt
├── technical/
│   ├── architecture.md
│   └── installation.md
├── reference/
│   ├── faq.txt
│   └── glossary.txt
└── examples/
    ├── example1.txt
    └── example2.txt
```

**Tips:**
- Keep documents well-organized by category
- Use descriptive filenames
- Include metadata in document headers (title, author, date)
- Maintain consistent formatting
- Remove duplicates regularly

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **Ollama Connection Error**
```
Error: Cannot connect to Ollama at localhost:11434
```
**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve

# Verify the model is downloaded
ollama list

# If missing, pull the model
ollama pull nomic-embed-text
```

#### 2. **OpenAI API Key Error**
```
Error: Invalid API key
```
**Solution:**
```bash
# Verify API key format (should start with sk-)
echo $OPENAI_API_KEY

# Try setting it directly in code
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-actual-key'
```

#### 3. **FAISS Index Memory Error**
```
Error: Memory allocation failed for index
```
**Solution:**
```python
# Use smaller chunk sizes
retriever = Retriever(
    chunk_size=256,  # Reduce from 512
    batch_size=16    # Reduce batch size
)

# Or use GPU-enabled FAISS
# pip install faiss-gpu
```

#### 4. **Slow Index Building**
**Solution:**
```python
# Use batch processing
retriever.build_index(batch_size=64)

# On subsequent runs, load cached index
retriever.load_index('indices/faiss_index.bin')

# Verify documents were loaded
print(f"Loaded {len(retriever.documents)} documents")
```

#### 5. **Poor Retrieval Quality**
**Solution:**
```python
# Adjust retrieval parameters
top_docs = retriever.search(
    query,
    top_k=10,  # Get more candidates
    similarity_threshold=0.2  # Lower threshold
)

# Check document quality
for doc in retriever.documents:
    print(f"{doc['filename']}: {len(doc['content'])} characters")
```

#### 6. **Python Import Errors**
```
Error: No module named 'rag_system'
```
**Solution:**
```bash
# Ensure you're in the correct directory
pwd

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports in your code
import sys
sys.path.insert(0, '/full/path/to/my-rag-system')
```

### Performance Optimization

```python
# For large document collections:
retriever = Retriever(
    chunk_size=768,      # Larger chunks = fewer embeddings
    chunk_overlap=100,   # More overlap = better context
    batch_size=128,      # Larger batches = faster processing
    device='cuda'        # Use GPU if available
)

# For faster responses:
generator = Generator(
    model="deepseek-coder-v2",  # Smaller models are faster
    temperature=0.3,            # Lower temperature = faster
    max_tokens=512              # Smaller output = faster
)

# Cache settings:
retriever.save_index()  # Save to disk
# Next run: retriever.load_index()  # Skip embedding step
```

---

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Ollama installed and models pulled (or OpenAI API key configured)
- [ ] Project directories created (data, prompts, indices)
- [ ] Sample documents added to `data/`
- [ ] Prompt templates created in `prompts/`
- [ ] Initial test run successful

You're now ready to use the RAG system! Start with the basic usage example and explore advanced features as needed.