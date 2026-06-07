# RAG Document Q&A App

A minimal Retrieval-Augmented Generation app for an AI engineer internship portfolio.

The app lets a user upload PDF or TXT files, splits the documents into chunks, creates embeddings, retrieves the most relevant chunks for a question, and asks an LLM to answer using only those chunks.

## Why this project matters

This project demonstrates the practical AI engineering workflow behind many modern LLM applications:

1. Document ingestion
2. Text extraction
3. Chunking
4. Embedding generation
5. Vector similarity search
6. Prompt construction
7. Grounded answer generation
8. Source inspection

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

## Interview explanation

A good 60-second explanation:

> I built a small RAG document Q&A app. Users upload a PDF or text file, the app extracts text, splits it into overlapping chunks, creates embeddings for each chunk, and stores those vectors in memory. When the user asks a question, the app embeds the question, retrieves the most similar chunks using cosine similarity, and sends those chunks to the model with instructions to answer only from the provided context. The app also shows the retrieved source chunks so the user can verify the answer.

## Design choices

- **Streamlit** keeps the UI simple and demo-friendly.
- **Overlapping chunks** help preserve context across chunk boundaries.
- **Cosine similarity** retrieves semantically similar chunks.
- **Source labels** make answers easier to verify.
- **In-memory storage** keeps the first version simple.

## Limitations

- The vector store is not persistent.
- PDF extraction quality depends on the PDF.
- No reranking step is included.
- No authentication or deployment setup is included.
- Evaluation is manual and small.

## Possible improvements

- Add persistent vector storage with Chroma, FAISS, or pgvector.
- Add reranking after initial retrieval.
- Add document deletion and re-indexing.
- Add automated evaluation with a test set.
- Add Docker deployment.
- Add cost and latency tracking.
- Add support for DOCX, CSV, and web pages.
