import re
from typing import Optional

def clean_subtitle_text(text: str) -> Optional[str]:
    """Enhanced cleaning for subtitle text with validation"""
    if not text.strip():
        return None
        
    # Remove timestamps and sequence numbers
    cleaned = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', text)
    
    # Remove HTML tags and special formatting
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = re.sub(r'\{[^}]+\}', '', cleaned)
    
    # Normalize whitespace and punctuation
    cleaned = re.sub(r'[^\w\s.,!?\'"-]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None
