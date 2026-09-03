## Features

- Upload PDF documents through the Streamlit UI
- Validate and extract text from PDF documents
- Detect PDFs with no extractable text
- Split documents into configurable text chunks
- Generate vector embeddings using HuggingFace
- Store embeddings in ChromaDB with session-based document isolation
- Retrieve relevant document chunks using semantic similarity search
- Apply similarity thresholds and lightweight keyword-based reranking
- Rewrite conversational follow-up questions into standalone search queries
- Generate context-based answers using OpenRouter
- Handle PDF, embedding, vector database, retrieval, and LLM failures gracefully
- Display source page references for generated answers
- Support conversational chat history