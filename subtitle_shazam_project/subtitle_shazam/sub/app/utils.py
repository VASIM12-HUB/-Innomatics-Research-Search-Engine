import streamlit as st
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional
from app.search_engine import semantic_search
from app.audio_query_transcriber import transcribe_audio

# Constants
SUBTITLE_DIR = "data"
VECTOR_STORE_DIR = "vectorstores/chromadb_store"
TEMP_AUDIO_DIR = "temp_audio"
MIN_SIMILARITY = 0.4  # Lower threshold for semantic search

def advanced_clean_text(text: str) -> str:
    """Enhanced cleaning for both audio transcripts and subtitle files"""
    if not text:
        return ""
        
    # Remove all timestamp formats
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}[,.]\d{3}', '', text)
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'-->', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    
    # Normalize whitespace and case
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def display_semantic_results(results: Dict) -> None:
    """Display semantic search results in Streamlit"""
    if not results or not results.get("documents"):
        st.info("No matches found.")
        return
    
    st.subheader(f"Top Semantic Matches:")
    
    for i, (doc, source, score) in enumerate(zip(
        results["documents"],
        results["sources"],
        results["scores"]
    ), 1):
        with st.expander(f"Match {i} (Score: {score:.1%})"):
            st.markdown(f"**Source:** `{source}`")
            st.markdown("**Relevant Content:**")
            st.text(doc)

# Initialize directories
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Streamlit App
st.set_page_config(page_title="Subtitle Shazam", layout="wide")
st.title("🎙️ Subtitle Shazam - Semantic Video Subtitle Search")

# Input mode selection
option = st.radio("Choose Query Input", ["Text Query", "Audio Query"], horizontal=True)

if option == "Text Query":
    query = st.text_area("Enter your search query:", height=100)
    if st.button("Search") and query.strip():
        with st.spinner("Finding matches..."):
            cleaned_query = advanced_clean_text(query)
            results = semantic_search(cleaned_query)
            display_semantic_results(results)
            
else:  # Audio Query
    uploaded_file = st.file_uploader(
        "Upload Audio (MP3/WAV, max 2min)", 
        type=["mp3", "wav"]
    )
    
    if uploaded_file:
        # Save audio temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"temp_audio_{timestamp}.wav")
        
        try:
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path)
                if transcript:
                    cleaned_transcript = advanced_clean_text(transcript)
                    
                    st.success("Transcription completed!")
                    with st.expander("View Transcription"):
                        st.text_area("Cleaned Transcript", cleaned_transcript, height=150)
                    
                    # Find matches
                    with st.spinner("Searching subtitles..."):
                        results = semantic_search(cleaned_transcript)
                        display_semantic_results(results)
                else:
                    st.error("Failed to transcribe audio")
                    
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
