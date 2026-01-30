# DeepRAG System – Production-Grade RAG with Drift Monitoring & Benchmarking

A **production-grade Retrieval-Augmented Generation (RAG) system** designed with real-world AI engineering practices in mind. The project goes beyond basic RAG implementations by introducing **systematic benchmarking, drift detection, MLflow tracking, and robustness controls against hallucination**.

This repository represents an end-to-end **LLMOps-ready RAG architecture**, suitable for research, applied AI, and enterprise use cases.

---

## 🚀 Key Highlights

* End-to-end modular RAG pipeline (Embedding → Retrieval → Reranking → Generation)
* FAISS IVF indexing for scalable, low-latency semantic search
* Cross-Encoder reranking for high-precision retrieval
* Comprehensive evaluation & benchmarking framework
* Continuous **Drift Detection** (Query, Retrieval, Hallucination)
* MLflow-based experiment tracking and monitoring
* Strong hallucination mitigation via prompt constraints and numeric grounding
* Production-oriented error handling and serialization safety

---

## 🧠 System Architecture (High-Level)

**Pipeline Flow:**

1. Document ingestion & preprocessing
2. Semantic chunking
3. Dense embedding (BAAI/bge-m3)
4. FAISS IVF retrieval
5. BM25 + metadata filtering
6. Multi-retriever fusion
7. Cross-Encoder reranking
8. LLM answer generation
9. Evaluation, logging, and drift monitoring

---

## 📁 Project Structure

```
Buliding Rag System/
├── benchmark.py                    # Full offline benchmark runner
├── drift_monitoring_example.py     # Integrated benchmark + drift monitoring
├── drift_detection.py              # Drift detection logic
├── drift_dashboard.py              # Monitoring dashboard utilities
├── evaluation_dataset.json         # Evaluation questions & references
├── evaluation_results/             # Per-question metric artifacts
├── benchmark_reports/              # Aggregated benchmark outputs
│   ├── full_results.json
│   ├── full_results.csv
│   ├── all_metrics.json
│   ├── drift_report.txt
│   └── metrics_pickles/
├── pipelines/
│   └── pipeline.py                 # Main RAG orchestration layer
├── steps/
│   ├── data_ingestion.py
│   ├── chunking_engine.py
│   ├── embedder.py
│   ├── faiss_index.py
│   ├── bm25_index.py
│   ├── retrievalfiltering.py
│   ├── fusion.py
│   ├── Cross_Encoder.py
│   ├── query_expansion.py
│   ├── prompt_engineering.py
├── metrics/
│   └── metrics.py                  # Evaluation metric implementations
└── data/
    └── processed/
        ├── faiss.index
        ├── faiss_mapping.json
        └── embeddings/
```

---

## 📊 Evaluation & Benchmarking

The system includes a **research-grade evaluation framework**.

### Retrieval Metrics

* **MRR (Mean Reciprocal Rank)**
* **Precision@K**
* **Recall@K**

### Answer Quality Metrics

* **Answer Relevance** (binary)
* **Hallucination Score** (0–1)
* **Jaccard Similarity**

### Performance Metrics

* FAISS search latency
* End-to-end pipeline time

All metrics are:

* Logged per question
* Aggregated across datasets
* Persisted as JSON / CSV / Pickle
* Tracked in **MLflow**

---

## 📈 Drift Detection System

A core differentiator of this project.

### Drift Types Monitored

* **Query Drift** – semantic change in incoming queries
* **Retrieval Drift** – degradation in retrieval quality
* **Hallucination Drift** – increased unsupported generation

### Features

* Batch-based drift analysis
* Daily scheduled checks
* Automatic drift reports
* MLflow logging of drift signals
* Stability trend tracking

This enables **long-term reliability monitoring**, not just one-off evaluation.

---

## 🧪 MLflow Integration

MLflow is used as a first-class component:

* Experiment-level tracking
* Per-run metrics & parameters
* Artifact storage (reports, pickles)
* Safe handling of nested runs
* Robust serialization of pipeline outputs

The system avoids common MLflow pitfalls such as:

* Active run conflicts
* Non-serializable object logging

---

## 🛡️ Hallucination Mitigation

During evaluation, it was observed that **surface-level abstraction** (e.g., translating numeric values into natural language) could trigger hallucination flags.

### Solution Implemented

* Enforced **literal numeric & unit preservation** in generation prompts
* Grounded answer generation strictly to retrieved context

This significantly reduced false hallucination detection while preserving correctness.

---

## ⚙️ Configuration

### Embedding & Retrieval

```python
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5
```

### FAISS IVF

```python
nlist = 100
nprobe = 10
use_ivf = True
```

### Drift Monitoring

```python
BATCH_SIZE = 50
DAILY_CHECK_HOUR = 2
```

---

## ▶️ Usage

### Run Full Benchmark

```bash
python benchmark.py
```

### Run Benchmark + Drift Monitoring

```bash
python drift_monitoring_example.py
```

### View MLflow Dashboard

```bash
mlflow ui
```

Access at: `http://localhost:5000`

---

## 📦 Outputs

### Benchmark Reports

* `full_results.json`
* `full_results.csv`
* `all_metrics.json`
* `drift_report.txt`

### Per-Question Artifacts

* Retrieval metrics
* Hallucination scores
* Answer relevance
* Jaccard similarity

---

## 🧩 Design Decisions (Rationale)

* **BGE-M3**: Strong multilingual + retrieval performance
* **FAISS IVF**: Scales efficiently for large corpora
* **Cross-Encoder**: Improves ranking precision beyond dense retrieval
* **MLflow**: Industry-standard experiment tracking
* **Drift Monitoring**: Required for real production LLM systems

---

## 🔮 Future Improvements

* Online (real-time) drift detection
* GPU-accelerated FAISS
* Cost & latency monitoring
* Human feedback integration
* Adaptive retriever selection

---

## 👨‍💻 Author

**Omar Yaser**
AI Engineer – RAG, LLMs, and MLOps

---

## ✅ Project Status

**Production-Ready**

* Stable pipeline
* Verified benchmarks
* Robust drift monitoring
* Clean MLflow integration
* Extensive documentation

This project reflects real-world AI engineering practices rather than academic prototypes.
