from .retriever import Retriever
from .generator import Generator
from .config import get_config
import sys
import os

def main():
    try:
        # Get configuration
        config = get_config()
        
        # Initialize components
        print("Initializing RAG system...")
        retriever = Retriever()
        generator = Generator()
        
        # Check if data directory exists
        if not os.path.exists("data"):
            print("Error: 'data' directory not found. Please create it and add your documents.")
            sys.exit(1)
            
        # Load documents
        print("Loading documents...")
        try:
            retriever.load_documents("data")
            print("Documents loaded successfully.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            sys.exit(1)
        
        # Interactive query loop
        print("\nRAG System Ready! (Type 'exit' to quit)")
        while True:
            try:
                query = input("\nEnter your query: ").strip()
            except Exception as e:
                print(f"Input error: {e}")
                break
            
            if query.lower() == 'exit':
                break
                
            if not query:
                print("Please enter a valid query.")
                continue
                
            try:
                # Retrieve relevant documents
                print("\nRetrieving relevant documents...")
                top_docs = retriever.search(query)
                
                # Generate response
                print("Generating response...")
                answer = generator.generate(query, top_docs)
                
                # Print results
                print("\nAnswer:")
                print("-" * 50)
                print(answer)
                print("-" * 50)
                
                # Print sources
                print("\nSources:")
                for i, doc in enumerate(top_docs, 1):
                    source = doc.get('metadata', {}).get('source', 'Unknown')
                    print(f"{i}. {source}")
                    
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()