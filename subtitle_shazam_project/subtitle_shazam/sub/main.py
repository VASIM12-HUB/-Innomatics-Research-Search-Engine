import streamlit as st
import os
import whisper
import tempfile
from sentence_transformers import SentenceTransformer, util
from chromadb import PersistentClient
import numpy as np
import glob
import re

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Updated ChromaDB client setup (new API)
client = PersistentClient(path="db")
collection = client.get_or_create_collection(name="subtitles")

# Function to clean text
def clean_text(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)  # remove timestamps
    text = re.sub(r"\n", " ", text)  # replace newlines
    text = re.sub(r"[^\w\s.,!?]", "", text, flags=re.UNICODE)  # keep Unicode
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# Transcribe audio and clean the text
def transcribe_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    result = whisper_model.transcribe(temp_audio_path)
    return clean_text(result["text"])

# Match transcription with subtitle chunks
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

# Streamlit UI
st.title("🎙️ Subtitle Shazam - Semantic Video Subtitle Search")
st.subheader("Choose Query Input")

option = st.radio("Select Input Type", ["Text Query", "Audio Query"], label_visibility="collapsed")

if option == "Text Query":
    text_input = st.text_area("Enter your query text")
    if st.button("Search") and text_input:
        results = match_subtitles(clean_text(text_input))
        if results:
            st.markdown("## __________________________")
            st.markdown("### Top Matches:")
            for idx, (score, filename, content) in enumerate(results, 1):
                st.markdown(f"**Result {idx}:**")
                st.markdown(f"**Subtitle name:** {filename}")
                st.markdown(f"**Content:** {content}")
        else:
            st.warning("No matches found.")

elif option == "Audio Query":
    uploaded_audio = st.file_uploader("Upload Audio (MP3/WAV, max 2min)", type=["mp3", "wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        if st.button("Transcribe and Search"):
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(uploaded_audio)
                st.success("Transcription completed!")
                st.markdown("### Transcription Details")
                st.write(transcription)

                results = match_subtitles(transcription)
                if results:
                    st.markdown("## __________________________")
                    st.markdown("### Top Matches:")
                    for idx, (score, filename, content) in enumerate(results, 1):
                        st.markdown(f"**Result {idx}:**")
                        st.markdown(f"**Subtitle name:** {filename}")
                        st.markdown(f"**Content:** {content}")
                else:
                    st.warning("No matches found.")

st.info("ℹ️ Results show exact matches from subtitle files")
st.warning("⚠️ Store subtitle files in the 'data' directory")
