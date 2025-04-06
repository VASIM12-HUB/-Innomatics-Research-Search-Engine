import os
import chromadb
from chromadb.config import Settings
from typing import Optional
from app.clean_subtitles import clean_subtitle_text
from app.chunk_text import chunk_text 
from app.generate_embeddings import generate_embeddings
import numpy as np

# Initialize ChromaDB with new persistent client API
client = chromadb.PersistentClient(path="vectorstores/chromadb_store")
collection = client.get_or_create_collection(
    name="subtitles",
    metadata={"hnsw:space": "cosine"}  # Using cosine similarity
)

def ingest_subtitle_file(filepath: str) -> bool:
    """Process and ingest subtitle file with improved error handling"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        clean_text = clean_subtitle_text(raw_text)
        if not clean_text:
            return False

        chunks = chunk_text(clean_text)
        if not chunks:
            return False

        embeddings = generate_embeddings(chunks)
        if embeddings is None:
            return False

        # Batch processing with metadata
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            collection.add(
                documents=batch_chunks,
                embeddings=[emb.tolist() for emb in batch_embeddings],
                ids=[f"{os.path.basename(filepath)}_chunk_{i+j}" for j in range(len(batch_chunks))],
                metadatas=[{
                    "source": os.path.basename(filepath),
                    "chunk_num": i+j,
                    "length": len(chunk.split())
                } for j, chunk in enumerate(batch_chunks)]
            )
        
        return True
        
    except Exception as e:
        print(f"Failed to ingest {filepath}: {str(e)}")
        return False
