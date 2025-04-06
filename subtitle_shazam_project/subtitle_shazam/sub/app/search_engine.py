import chromadb
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional
from app.chunk_text import chunk_text
import numpy as np

# Initialize ChromaDB with new client API
client = chromadb.PersistentClient(path="vectorstores/chromadb_store")
collection = client.get_or_create_collection(
    name="subtitles",
    metadata={"hnsw:space": "cosine"}
)

# Initialize model with device handling
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def semantic_search(query: str, top_k: int = 5) -> Optional[Dict]:
    """
    Enhanced semantic search with query expansion and result filtering
    """
    if not query.strip():
        return None

    try:
        # Generate query embedding
        query_embedding = model.encode(
            query,
            device=device,
            convert_to_tensor=False
        ).tolist()

        # Query ChromaDB with additional metadata filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert distances to similarity scores (1 - distance)
        for i in range(len(results["distances"][0])):
            results["distances"][0][i] = 1 - results["distances"][0][i]
        
        return {
            "documents": results["documents"][0],
            "sources": [m["source"] for m in results["metadatas"][0]],
            "scores": results["distances"][0]
        }
        
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return None
