import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_id: int


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to a .env file or your shell environment.")
        st.stop()
    return OpenAI(api_key=api_key)


def extract_text_from_upload(uploaded_file) -> List[Tuple[str, str, int | None]]:
    """Return a list of (text, source_filename, page_number)."""
    filename = uploaded_file.name
    suffix = filename.lower().split(".")[-1]

    if suffix == "txt":
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        return [(text, filename, None)]

    if suffix == "pdf":
        reader = PdfReader(uploaded_file)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, filename, i))
        return pages

    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, source: str, page: int | None, chunk_size: int, overlap: int) -> List[Chunk]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(Chunk(text=chunk, source=source, page=page, chunk_id=chunk_id))
        chunk_id += 1

        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    if not texts:
        return np.array([])

    response = client.embeddings.create(model=model, input=texts)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def cosine_top_k(query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if doc_embeddings.size == 0:
        return []

    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)

    # Avoid division by zero.
    similarities = doc_embeddings @ query_embedding / (doc_norms * query_norm + 1e-10)
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_indices]


def format_context(chunks: List[Chunk]) -> str:
    sections = []
    for idx, chunk in enumerate(chunks, start=1):
        location = f"{chunk.source}"
        if chunk.page is not None:
            location += f", page {chunk.page}"
        location += f", chunk {chunk.chunk_id}"
        sections.append(f"[Source {idx}: {location}]\n{chunk.text}")
    return "\n\n".join(sections)


def answer_question(client: OpenAI, question: str, retrieved_chunks: List[Chunk], model: str) -> str:
    context = format_context(retrieved_chunks)
    instructions = (
        "You are a careful document question-answering assistant. "
        "Answer using only the provided context. "
        "If the answer is not in the context, say you could not find it in the uploaded documents. "
        "Cite sources inline using labels like [Source 1] or [Source 2]. "
        "Keep the answer concise and useful."
    )
    user_input = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        store=False,
    )
    return response.output_text


def build_index(uploaded_files, chunk_size: int, overlap: int, embedding_model: str):
    client = get_client()
    all_chunks: List[Chunk] = []

    for uploaded_file in uploaded_files:
        extracted_items = extract_text_from_upload(uploaded_file)
        for text, source, page in extracted_items:
            all_chunks.extend(chunk_text(text, source, page, chunk_size, overlap))

    if not all_chunks:
        st.warning("No text could be extracted from the uploaded files.")
        return [], np.array([])

    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_texts(client, texts, embedding_model)
    return all_chunks, embeddings


def main():
    st.set_page_config(page_title="RAG Document Q&A", page_icon="🔎", layout="wide")
    st.title("🔎 RAG Document Q&A")
    st.caption("Upload documents, retrieve relevant chunks, and ask grounded questions with citations.")

    with st.sidebar:
        st.header("Settings")
        chat_model = st.text_input("Answer model", value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
        embedding_model = st.text_input("Embedding model", value=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        chunk_size = st.slider("Chunk size", min_value=300, max_value=2000, value=900, step=100)
        overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=150, step=50)
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = np.array([])

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Build document index", type="primary", disabled=not uploaded_files):
            with st.spinner("Extracting text, chunking, and creating embeddings..."):
                chunks, embeddings = build_index(uploaded_files, chunk_size, overlap, embedding_model)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
            st.success(f"Indexed {len(st.session_state.chunks)} chunks.")

        if st.session_state.chunks:
            st.subheader("Indexed chunks")
            st.write(f"Total chunks: {len(st.session_state.chunks)}")
            with st.expander("Preview first 3 chunks"):
                for chunk in st.session_state.chunks[:3]:
                    page_label = f"page {chunk.page}" if chunk.page is not None else "text file"
                    st.markdown(f"**{chunk.source} — {page_label}, chunk {chunk.chunk_id}**")
                    st.write(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))

    with col2:
        question = st.text_input("Ask a question about your documents")
        ask_disabled = not question or len(st.session_state.chunks) == 0

        if st.button("Ask", disabled=ask_disabled):
            client = get_client()
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                query_embedding = embed_texts(client, [question], embedding_model)[0]
                results = cosine_top_k(query_embedding, st.session_state.embeddings, top_k)
                retrieved_chunks = [st.session_state.chunks[i] for i, _ in results]
                answer = answer_question(client, question, retrieved_chunks, chat_model)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved sources")
            for rank, (chunk_index, score) in enumerate(results, start=1):
                chunk = st.session_state.chunks[chunk_index]
                page_label = f"page {chunk.page}" if chunk.page is not None else "text file"
                with st.expander(f"Source {rank}: {chunk.source} — {page_label}, chunk {chunk.chunk_id} | score {score:.3f}"):
                    st.write(chunk.text)

    st.divider()
    st.markdown(
        """
        **What this project demonstrates:** document ingestion, chunking, embeddings, vector similarity search, prompt construction, source-grounded answering, and basic evaluation readiness.
        """
    )


if __name__ == "__main__":
    main()
