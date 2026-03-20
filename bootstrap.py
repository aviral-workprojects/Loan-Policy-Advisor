#!/usr/bin/env python3
"""
Bootstrap Script
=================
Run this ONCE before starting the API to:
  1. Load all documents from data/ directory
  2. Build and save the FAISS index

Usage:
  python bootstrap.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from services.retrieval import RetrievalService

def main():
    print("=" * 60)
    print("Loan Policy Advisor — Bootstrap")
    print("=" * 60)

    retrieval = RetrievalService()

    print("\n[1] Loading documents from data/ …")
    count = retrieval.load_documents()
    print(f"    Loaded {count} chunks from {len(list(__import__('config').DATA_DIR.iterdir()))} source folders")

    if count == 0:
        print("\n⚠️  No documents found. Make sure data/ folders contain .txt or .pdf files.")
        return

    print("\n[2] Building FAISS index (this may take a minute)…")
    retrieval.build_index(save=True)

    print("\n✅ Bootstrap complete!")
    print(f"   Index saved to models/index.faiss")
    print(f"   Chunks: {count}")
    print("\nYou can now start the API:")
    print("   uvicorn api.main:app --reload --port 8000")

if __name__ == "__main__":
    main()
