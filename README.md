# RAG Document Q&A App

A minimal Retrieval-Augmented Generation app for an AI engineer internship portfolio.

The app lets a user upload PDF or TXT files, splits the documents into chunks, creates embeddings, retrieves the most relevant chunks for a question, and asks an LLM to answer using only those chunks.

## Tech stack

- Python
- Streamlit
- OpenAI API
- NumPy
- pypdf

This version intentionally uses an in-memory NumPy vector store to keep the code easy to understand. A production version could replace it with Chroma, FAISS, Pinecone, Weaviate, or pgvector.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## How to use

1. Upload one or more PDF or TXT files.
2. Click **Build document index**.
3. Ask a question about the documents.
4. Read the answer and inspect the retrieved source chunks.

A sample text file is included in `sample_docs/ai_engineer_interview_notes.txt`.
