from zenml import pipeline
from steps.data_ingestion import load
from steps.embedder import chunks_embedding
from steps.chunking_engine import chunk_documents
from steps.faiss_index import faiss_index_step
from steps.retrievalfiltering import retrieval_step
from steps.Cross_Encoder import CrossEncoderStep
from steps.prompt_engineering import llama_llm_generation_step
from pathlib import Path


import pickle
from zenml import step
import numpy as np
from typing import List, Dict, Optional

@step
def build_metadata_store(embedeed_chunks: List[Dict]) -> Dict[str, Dict]:
    """Build metadata store from embedded chunks."""
    return {
        chunk["chunk_id"]: {
            **chunk["metadata"],
            "text": chunk.get("text", "")
        }
        for chunk in embedeed_chunks
    }

@pipeline
def rag_pipline(query: str,
                query_vector: List[float],
                filters: Optional[Dict] = None,
                doc_path: str = r"E:\\pyDS\\Buliding Rag System\\data"):

    docs = load(Path(doc_path))
    chunks = chunk_documents(docs)
    embedeed_chunks = chunks_embedding(chunks)
    fi =  faiss_index_step(embedeed_chunks, vector_dim=1024) # will retrun FaissIndex instance

    chunk_metadata_store = build_metadata_store(embedeed_chunks)

    results = retrieval_step(
        faiss_index=fi,
        chunks_metadata=chunk_metadata_store,
        query_vector=query_vector,  # to be provided during pipeline run
        top_k=50,
        filters=filters  # to be provided during pipeline run
    )

    reranked_results = CrossEncoderStep(
        query=query,  # to be provided during pipeline run
        results_from_faiss=results,
        chunks=chunks,
        top_k=5
    )

    final_answer = llama_llm_generation_step(
        question=query,  # to be provided during pipeline run
        chunks=reranked_results
    )

    return final_answer