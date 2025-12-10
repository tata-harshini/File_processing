File Processing Microservices (RAG + Google Gemini)

A microservices system for file upload → text extraction → chunking → embedding → FAISS indexing → RAG-based summarization & analysis.
Services used:

File Service — upload, store files, trigger processing

Text Extractor — extract text from files and store chunks + embeddings

AI Service — summarization and analysis (uses Google Gemini when configured, otherwise local fallback)

MongoDB — metadata and chunk storage

FAISS — vector index for semantic retrieval

Features

Upload files and persist metadata

Extract text from PDF / DOCX / TXT

Chunking (configurable chunk size)

Embeddings using Google Gemini (preferred) or SentenceTransformers (fallback)

FAISS-based fast nearest-neighbor retrieval

RAG pipeline for context-aware summarization

AI-powered analysis (topics, sentiment, insights) via Gemini (if configured)

Local fallback: extractive summarizer + local embeddings if Gemini unavailable
