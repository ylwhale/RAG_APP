# Evaluation Questions

Use these questions with `sample_docs/ai_engineer_interview_notes.txt` to demo and test the app.

| Question | Expected behavior |
|---|---|
| What are the main steps in a RAG app? | Should list ingestion, extraction, chunking, embeddings, retrieval, prompt construction, answer generation, and evaluation. |
| What is an embedding? | Should explain that embeddings are numerical vector representations of text. |
| How does the app answer a user's question? | Should describe embedding the question, finding similar chunks, and passing context to the model. |
| How can we reduce hallucinations? | Should mention context grounding, citations, missing-information behavior, evaluation, and logging failures. |
| What future improvements could be added? | Should mention persistent vector storage, better PDF parsing, reranking, auth, cost/latency monitoring, or evaluation. |
