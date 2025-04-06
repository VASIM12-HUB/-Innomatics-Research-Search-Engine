import streamlit as st
from src.interface import handle_text_query, handle_audio_query

st.title("🎙️ Subtitle Shazam - Semantic Video Subtitle Search")
st.subheader("Choose Query Input")

option = st.radio("Select Input Type", ["Text Query", "Audio Query"], label_visibility="collapsed")

if option == "Text Query":
    handle_text_query()
else:
    handle_audio_query()

st.info("ℹ️ Results show exact matches from subtitle files")
st.warning("⚠️ Store subtitle files in the 'data' directory")