from pathlib import Path
from typing import Dict, List, Optional

from zenml import pipeline, step

from steps.bm25_index import bm25_search_step
from steps.chunking_engine import chunk_documents
from steps.data_ingestion import load
from steps.embedder import chunks_embedding
from steps.faiss_index import faiss_index_step
from steps.retrievalfiltering import retrieval_step
from steps.Cross_Encoder import CrossEncoderStep
from steps.prompt_engineering import llama_llm_generation_step
from steps.fusion import fusion_step
from metrics.metrics import evaluate_pipeline

import mlflow
import time

@step
def build_metadata_store(embedded_chunks: List[Dict]) -> Dict[str, Dict]:
    """Build metadata store from embedded chunks."""
    return {
        chunk["chunk_id"]: {
            **chunk["metadata"],
            "text": chunk.get("text", "")
        }
        for chunk in embedded_chunks
    }

@pipeline
def rag_pipeline(
    query: str,
    query_vector: List[float],
    filters: Optional[Dict] = None,
    relevant_ids: Optional[List[str]] = [],
    doc_path: str = r"E:\\pyDS\\Buliding Rag System\\data",
) -> str:
    """
    RAG (Retrieval-Augmented Generation) pipeline.
    
    Orchestrates document loading, chunking, embedding, retrieval, and LLM-based answer generation.
    
    Args:
        query: The user's question
        query_vector: The embedding vector of the query
        filters: Optional metadata filters for retrieval
        doc_path: Path to the documents directory
        
    Returns:
        The LLM-generated answer as a string
    """
    start_total_time = time.time()

    start_load_time = time.time()
    docs = load(Path(doc_path))
    chunks = chunk_documents(docs)
    end_load_time = time.time()

    mlflow.log_metric("data_loading_time_seconds", (end_load_time - start_load_time) * 1000)
    
    start_embedding_time = time.time()
    embedded_chunks = chunks_embedding(chunks)
    end_embedding_time = time.time()
    mlflow.log_metric("embedding_time_seconds", (end_embedding_time - start_embedding_time) * 1000)

    start_faiss_time = time.time()
    fi = faiss_index_step(embedded_chunks, vector_dim=1024)  # Returns FaissIndex instance
    end_faiss_time = time.time()
    mlflow.log_metric("faiss_index_time_seconds", (end_faiss_time - start_faiss_time) * 1000)


    chunk_metadata_store = build_metadata_store(embedded_chunks)

    start_faiss_reytrieval_time = time.time()
    # FAISS retrieval
    faiss_results = retrieval_step(
        faiss_index=fi,
        chunks_metadata=chunk_metadata_store,
        query_vector=query_vector,  # To be provided during pipeline run
        top_k=50,
        filters=filters,  # To be provided during pipeline run
    )
    end_faiss_reytrieval_time = time.time()
    mlflow.log_metric("faiss_retrieval_time_seconds", (end_faiss_reytrieval_time - start_faiss_reytrieval_time) * 1000)
    
    start_bm25_time = time.time()
    # BM25 retrieval
    bm25_results = bm25_search_step(
        chunks=chunks,
        query=query,
        top_k=50
    )
    end_bm25_time = time.time()
    mlflow.log_metric("bm25_retrieval_time_seconds", (end_bm25_time - start_bm25_time) * 1000)

    # Fusion of FAISS and BM25 results
    start_fusion_time = time.time() 
    fused_results = fusion_step(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
        top_k=20,
    )
    end_fusion_time = time.time()
    mlflow.log_metric("fusion_time_seconds", (end_fusion_time - start_fusion_time) * 1000)

    # Re-ranking with Cross-Encoder
    start_cross_encoder_time = time.time()
    reranked_results = CrossEncoderStep(
        query=query,  # To be provided during pipeline run
        results_from_faiss=fused_results,
        chunks=chunks,
        top_k=5,
    )
    end_cross_encoder_time = time.time()
    mlflow.log_metric("cross_encoder_time_seconds", (end_cross_encoder_time - start_cross_encoder_time) * 1000)

    # LLM Generation
    start_llm_time = time.time()
    final_answer = llama_llm_generation_step(
        question=query,  # To be provided during pipeline run
        chunks=reranked_results,
    )
    end_llm_time = time.time()
    mlflow.log_metric("llm_generation_time_seconds", (end_llm_time - start_llm_time) * 1000)

    end_total_time = time.time()
    mlflow.log_metric("total_pipeline_time_seconds", (end_total_time - start_total_time) * 1000)

    all_metrics = evaluate_pipeline(
        question=query,
        retrieved_chunks=reranked_results,
        answer=final_answer,
        relevant_ids=relevant_ids,
        context_chunks=reranked_results,
        retriever=None,
        paraphrases=[]
    )
    mlflow.log_metrics(all_metrics)

    return final_answer