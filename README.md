### Document Assistant – RAG-based PDF Q&A
A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content using local embeddings and Groq-hosted LLMs.

## Features
Upload any PDF document

Ask natural language questions

Uses FAISS for vector similarity search

Uses HuggingFace sentence-transformers for embeddings (free & local)

Uses Groq (LLaMA 3.1) for fast inference

Built with LangChain + Streamlit

## Architecture (High-level)

PDF → Text Splitter → Embeddings → FAISS
                        ↓
                  Similarity Search
                        ↓
              Retrieved Context + Question
                        ↓
                   Groq LLM Answer
## Tech Stack
Python

Streamlit

LangChain

FAISS

HuggingFace Sentence Transformers

Groq LLaMA 3.1

## Setup Instructions
1. Clone the repo
git clone https://github.com/SahiraAli/document-assistant.git
cd document-assistant
2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Add environment variables
Create a .env file:
GROQ_API_KEY=your_actual_key_here
5. Run the app
streamlit run app.py
