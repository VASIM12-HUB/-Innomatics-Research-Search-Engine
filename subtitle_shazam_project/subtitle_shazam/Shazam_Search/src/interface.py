import streamlit as st
from src.processor import match_subtitles, transcribe_audio
from utils.text_cleaner import clean_text

def handle_text_query():
    text_input = st.text_area("Enter your query text")
    if st.button("Search") and text_input:
        results = match_subtitles(clean_text(text_input))
        show_results(results)

def handle_audio_query():
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
                show_results(results)

def show_results(results):
    if results:
        st.markdown("## __________________________")
        st.markdown("### Top Matches:")
        for idx, (score, filename, content) in enumerate(results, 1):
            st.markdown(f"**Result {idx}:**")
            st.markdown(f"**Subtitle name:** {filename}")
            st.markdown(f"**Content:** {content}")
    else:
        st.warning("No matches found.")