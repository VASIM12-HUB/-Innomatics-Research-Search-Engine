import tempfile
import os
import glob
from models.whisper_loader import whisper_model
from models.embedding_loader import embedding_model
from utils.text_cleaner import clean_text
from sentence_transformers import util

def transcribe_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    result = whisper_model.transcribe(temp_audio_path)
    return clean_text(result["text"])

def match_subtitles(query_text):
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)
    subtitle_files = glob.glob("data/*.srt")
    results = []

    for file_path in subtitle_files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        chunk = ""
        chunk_id = 0
        for line in lines:
            if "-->" in line or line.strip().isdigit():
                continue
            line = line.strip()
            if not line:
                continue
            chunk += line + " "
            if len(chunk.split()) > 30:
                cleaned = clean_text(chunk)
                doc_embedding = embedding_model.encode(cleaned, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
                print(f"{os.path.basename(file_path)} chunk {chunk_id} similarity: {similarity}")
                if similarity > 0.4:
                    results.append((similarity, os.path.basename(file_path), cleaned))
                chunk = ""
                chunk_id += 1

    results.sort(reverse=True)
    return results