# 🔍 DeepRAG — Advanced Retrieval-Augmented Generation System

DeepRAG is a **production-inspired Retrieval-Augmented Generation (RAG) system** built completely from scratch.
The goal of this project is to demonstrate a **deep engineering-level understanding** of modern RAG pipelines — not a demo or a wrapper around existing frameworks.

This project covers the full lifecycle:
**Ingestion → Chunking → Embeddings → Vector Search → Re-Ranking → Prompt Grounding → Answer Generation**.

---

## 🚀 Key Features

- 📄 **Document Ingestion**
  - Supports TXT and PDF files
  - Extracts text with rich metadata (source, page, language)

- ✂️ **Recursive Chunking Engine**
  - Token-based chunking
  - Overlap to preserve context
  - Deterministic chunk IDs (hash-based)

- 🧠 **Embeddings**
  - Model: `BAAI/bge-m3`
  - Multilingual (Arabic & English)
  - Normalized embeddings for cosine similarity
  - Embedding cache to avoid recomputation

- ⚡ **Vector Store**
  - FAISS `IndexFlatIP`
  - External mapping between FAISS IDs and chunk IDs
  - Persistent save/load support

- 🔎 **Semantic Retrieval**
  - Top-K semantic search
  - Metadata-based filtering (language, source)
  - Recall-safe retrieval strategy

- 🎯 **Cross-Encoder Re-Ranking**
  - Model: `BAAI/bge-reranker-base`
  - Re-ranks Top-50 candidates → Top-5
  - High-precision relevance scoring

- 🛡 **Anti-Hallucination Prompting**
  - Context-only answering
  - Explicit fallback: **"لا أعلم"**
  - Mandatory grounding in retrieved chunks
  - Source-aware responses

---

## 🧠 System Architecture

Documents (TXT / PDF)
↓
Chunking Engine
↓
Embedding Model (BGE-M3)
↓
FAISS Vector Index + Metadata Store
↓
Top-K Semantic Retrieval
↓
Cross-Encoder Re-Ranking
↓
Prompt Builder (Context Grounding)
↓
LLM (LLaMA 3)
↓
Answer + Sources


---

## 📂 Project Structure

deep_rag/
│
├── ingestion/
│ ├── loader.py # TXT / PDF document loaders
│ └── chunker.py # Recursive token-based chunking
│
├── embeddings/
│ └── embedder.py # Embedding engine + caching layer
│
├── vector_store/
│ └── faiss_index.py # FAISS index + vector-to-chunk mapping
│
├── retrieval/
│ ├── retriever.py # Top-K retrieval + metadata filtering
│ └── reranker.py # Cross-Encoder re-ranking
│
├── llm/
│ ├── prompt.py # Strict anti-hallucination prompts
│ └── generator.py # LLM answer generation
│
└── DeepRAG.ipynb # End-to-end pipeline notebook


---

## 🧪 Example Use Cases

- 📚 University course material Q&A
- 🏢 Company internal knowledge base
- ⚖️ Legal document analysis
- 🧬 Medical or research paper question answering

---

## 🛡 Anti-Hallucination Strategy

The system is explicitly designed to **prevent hallucination**:

- LLM is restricted to retrieved context only
- No external or pretrained knowledge allowed
- Explicit instruction to answer **"لا أعلم"** when information is missing
- Only Top-5 re-ranked chunks are passed to the LLM
- Answers are always returned with sources

---

## 📈 Why This Project Is Different

✔ Built without LangChain / LlamaIndex abstractions  
✔ Full control over every RAG component  
✔ Clear separation of concerns  
✔ Production-inspired design decisions  
✔ Focus on correctness, recall, and grounding  

This project reflects **real RAG engineering**, not prompt-only experimentation.

---

## 🧑‍💻 Author

**Omar Yaser**  
Computer Science Student — AI & Machine Learning  
Faculty of Computers & Information, Mansoura University  

---

## 📌 Notes

- The notebook is used to demonstrate and validate the full pipeline.
- The architecture is intentionally modular to support future extensions:
  - FastAPI deployment
  - Hybrid search
  - Evaluation metrics
  - Large-scale indexing

