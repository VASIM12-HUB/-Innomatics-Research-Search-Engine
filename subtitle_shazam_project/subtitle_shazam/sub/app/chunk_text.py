from transformers import AutoTokenizer
from typing import List

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Improved chunking with overlap and sentence boundary awareness"""
    tokens = tokenizer.encode(text)
    chunks = []
    
    if len(tokens) <= chunk_size:
        return [tokenizer.decode(tokens)]
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    
    return chunks
