from zenml import pipeline
from steps.data_ingestion import load
from steps.embedder import chunks_embedding
from steps.chunking_engine import chunk_documents
from steps.faiss_index import faiss_index_step
from steps.retrievalfiltering import retrieval_step
from steps.Cross_Encoder import CrossEncoderStep
from steps.prompt_engineering import qwen_llm_generation_step


import pickle
from zenml import step

@pipeline
def rag_pipline( query : None ,
                 query_vector: None ,
                 filters: None ,
                 doc_path: str = "E:\pyDS\Buliding Rag System\data"):

    docs = load(doc_path)
    chunks = chunk_documents(docs)
    embedeed_chunks = chunks_embedding(chunks)
    fi =  faiss_index_step(embedeed_chunks, vector_dim=1024) # will retrun FaissIndex instance

    chunk_metadata_store = {
    chunk["chunk_id"]: chunk["metadata"]
    for chunk in embedeed_chunks
    }

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

    final_answer = qwen_llm_generation_step(
        question=query,  # to be provided during pipeline run
        chunks=reranked_results
    )

    return final_answer