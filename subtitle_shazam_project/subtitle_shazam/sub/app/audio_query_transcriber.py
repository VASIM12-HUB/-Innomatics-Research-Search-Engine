import whisper
import torch
from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_whisper_model(model_size: str = "base"):
    """Load Whisper model with error handling"""
    try:
        return whisper.load_model(model_size).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

model = load_whisper_model()

def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio with error handling and chunk processing"""
    try:
        result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None
