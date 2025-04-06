from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def generate_embeddings(chunks: List[str]) -> Optional[np.ndarray]:
    """Generate embeddings with error handling and batching"""
    if not chunks:
        return None
        
    try:
        return model.encode(
            chunks,
            device=device,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return None
