# RAG Starter System

A simple Retrieval-Augmented Generation (RAG) system that combines document retrieval with text generation.

## What I updated (modernization pass)

This repository started as a good minimal example, but it had a few gaps for practical use. It now includes:

- Safer retrieval behavior when the index is empty.
- A reusable `add_documents()` API for dynamic ingestion.
- Optional URL ingestion into the local FAISS index.
- Optional live web-search augmentation per query.
- CLI flags for data directory, URL ingestion, and web search.

## Components

### Retriever
- Loads and embeds local `.txt` files.
- Builds and persists a FAISS index.
- Supports incremental ingestion via `add_documents()`.

### Generator
- Builds prompts from retrieved context.
- Generates answers using either Ollama or OpenAI.

### Web ingestion/search (new)
- Fetch readable text from public URLs and ingest into index.
- Use DuckDuckGo HTML search results as temporary context at query time.

## Usage

### Basic usage

```bash
python run_rag.py
```

### Local docs directory

```bash
python run_rag.py --data-dir data
```

### Ingest public documentation URLs into the local index

```bash
python run_rag.py \
  --load-url https://docs.python.org/3/library/asyncio.html \
  --load-url https://fastapi.tiangolo.com/tutorial/
```

### Enable live web search augmentation

```bash
python run_rag.py --web-search --web-results 3
```

You can combine both:

```bash
python run_rag.py \
  --load-url https://docs.python.org/3/library/asyncio.html \
  --web-search
```

## Recommended next upgrades

To further improve quality/reliability, consider:

1. **Chunking + overlap** before embedding (instead of whole-document embeddings).
2. **Metadata-aware filtering** (`source`, `type`, tags) in retrieval.
3. **Hybrid retrieval** (BM25 + vector) and reranking.
4. **Answer citations** aligned to chunk IDs.
5. **Config via env/CLI** for model names and retrieval parameters.
6. **Tests** for retriever/search edge cases and web-ingestion parsing.
7. **Observability**: structured logs and latency/token metrics.

## Requirements

- Python 3.8+
- FAISS
- NumPy
- Requests
- OpenAI (optional)
- PyYAML

Install:

```bash
pip install -r requirements.txt
```

## Notes

- URL/web content quality varies by website structure.
- Some sites block scraping; those pages are skipped in web augmentation.
- Web search context is added in-memory per query unless explicitly ingested with `--load-url`.
