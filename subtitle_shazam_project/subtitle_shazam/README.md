
Shazam Clone - Semantic Subtitle Search Engine
A smart search engine that finds matching subtitle content from `.srt` files using semantic similarity with audio or text input — inspired by how Shazam matches music.

Overview

This project allows users to:

- Upload a 2-minute audio clip (e.g., from a TV series or movie)
- Or enter a text query
- Then find the most relevant subtitle chunks from stored `.srt` files using semantic search

The system uses:
- Whisper for speech-to-text transcription
- SentenceTransformers for semantic embeddings
- ChromaDB for vector storage
- Streamlit for an interactive web interface

Project Structure

Shazam_Clone_Search_Engine/
│
├── app.py                  Main Streamlit frontend
├── src/
│   ├── cleaner.py          Text cleaning utilities
│   ├── embedder.py         Sentence embedding logic
│   ├── matcher.py          Subtitle chunk matching logic
│   ├── transcriber.py      Audio transcription using Whisper
│   └── utils.py            Model loading, helper functions
│
├── data/                   Place your .srt subtitle files here
│
├── db/                     ChromaDB vector storage
│
├── requirements.txt        Python dependencies
└── README.md               Project documentation

Setup Instructions

1. Clone the repository:
git clone https://github.com/vjabhi000985/Shazam_Clone_Search_Engine.git
cd Shazam_Clone_Search_Engine

2. Install dependencies:
pip install -r requirements.txt

3. Add subtitle files:
- Place all `.srt` subtitle files in the `data/` directory.

4. Run the Streamlit app:
streamlit run app.py

How It Works

- Whisper transcribes audio input into text.
- The transcribed or text input is embedded using a SentenceTransformer model.
- Subtitle files are processed in chunks for better contextual understanding.
- Cosine similarity is computed between query embeddings and subtitle embeddings.
- Chunks with high similarity scores are returned as top matches.

Features

- Supports both text and audio-based queries
- Subtitle chunking avoids information loss
- Intelligent cleaning of subtitle text
- Modular design for easy extension or reuse
- Results include subtitle file name and matched content

Technologies Used

- Whisper (OpenAI) – for audio transcription
- SentenceTransformers – for generating semantic embeddings
- ChromaDB – for storing and querying vector embeddings
- Streamlit – for building the user interface

Future Improvements

- Add TF-IDF keyword-based search support
- Show timestamps with matched content
- Highlight matched phrases in context
- Support subtitles in multiple languages

Requirements

The following packages are needed to run the project:

streamlit
openai-whisper
sentence-transformers
torch
numpy
chromadb
ffmpeg-python

If you encounter issues with `ffmpeg`, install it system-wide using:

sudo apt install ffmpeg
