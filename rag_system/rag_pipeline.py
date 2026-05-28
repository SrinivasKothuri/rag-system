import argparse
import sys
import os
from .retriever import Retriever
from .generator import Generator
from .config import get_config
from .web_ingestion import url_to_document, web_search_to_documents


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG system")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing local .txt documents"
    )
    parser.add_argument(
        "--load-url",
        action="append",
        default=[],
        help="Public URL to ingest into the local index (repeatable)"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Augment each query with live web search results"
    )
    parser.add_argument(
        "--web-results",
        type=int,
        default=3,
        help="Max number of URLs to fetch from web search per query"
    )
    return parser


def main():
    try:
        _ = get_config()
        args = _build_parser().parse_args()

        print("Initializing RAG system...")
        retriever = Retriever()
        generator = Generator()

        if os.path.exists(args.data_dir):
            print(f"Loading local documents from '{args.data_dir}'...")
            retriever.load_documents(args.data_dir)
            print("Local documents loaded successfully.")
        else:
            print(f"Warning: data directory '{args.data_dir}' not found. Continuing without local files.")

        if args.load_url:
            print("Ingesting URL documents into index...")
            url_docs = []
            for url in args.load_url:
                try:
                    doc = url_to_document(url)
                    if doc["content"].strip():
                        url_docs.append(doc)
                        print(f"  - loaded {url}")
                    else:
                        print(f"  - skipped {url} (no readable text)")
                except Exception as err:
                    print(f"  - failed {url}: {err}")
            if url_docs:
                retriever.add_documents(url_docs)

        print("\nRAG System Ready! (Type 'exit' to quit)")
        while True:
            try:
                query = input("\nEnter your query: ").strip()
            except Exception as err:
                print(f"Input error: {err}")
                break

            if query.lower() == 'exit':
                break

            if not query:
                print("Please enter a valid query.")
                continue

            try:
                print("\nRetrieving relevant documents...")
                top_docs = retriever.search(query)

                if args.web_search:
                    print("Fetching web search context...")
                    top_docs.extend(web_search_to_documents(query, max_results=args.web_results))

                print("Generating response...")
                answer = generator.generate(query, top_docs)

                print("\nAnswer:")
                print("-" * 50)
                print(answer)
                print("-" * 50)

                print("\nSources:")
                if not top_docs:
                    print("No sources available.")
                for i, doc in enumerate(top_docs, 1):
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    print(f"{i}. {source}")

            except Exception as err:
                print(f"Error processing query: {err}")

    except Exception as err:
        print(f"Error initializing RAG system: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
