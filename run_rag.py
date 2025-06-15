#!/usr/bin/env python3
"""
Command-line interface for the RAG system.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.rag_pipeline import main

if __name__ == "__main__":
    main() 