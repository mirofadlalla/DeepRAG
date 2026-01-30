# # الهدف

# # ناخد Top-50 chunks من Retriever
# # نعيد ترتيبهم باستخدام Cross-Encoder
# # نطلع Top-5 فقط بأعلى دقة ممكنة
# import numpy as np
# from typing import List, Dict

from zenml import step

# from sentence_transformers import CrossEncoder


# class CrossEncoderReRanker:
#     def __init__(self, model_name: str = "BAAI/bge-reranker-base" ,
#                  device = "cuda" ,
#                  top_k: int = 5) -> List[Dict]:

#         self.model = CrossEncoder(model_name , device=device)

#     def CE(self , query , rsults_from_faiss  , chunks, top_k = 5):
#         chunks_ids_from_faiss = [r['chunk_id'] for r in rsults_from_faiss]
#         # Create mapping of chunk_id to text and result
#         chunk_id_to_result = {r['chunk_id']: r for r in rsults_from_faiss}
#         chunk_id_to_text = {}
        
#         for chunk in chunks:
#             if chunk['id'] in chunks_ids_from_faiss:
#                 chunk_id_to_text[chunk['id']] = chunk['text']
        
#         # Build pairs in same order as chunk IDs
#         ordered_chunk_ids = []
#         texts = []
#         for cid in chunks_ids_from_faiss:
#             if cid in chunk_id_to_text:
#                 ordered_chunk_ids.append(cid)
#                 texts.append(chunk_id_to_text[cid])
        
#         pairs = [(query, t) for t in texts]
#         scores = self.model.predict(pairs)
        
#         scored = []
#         for cid, score in zip(ordered_chunk_ids, scores):
#             result = chunk_id_to_result[cid]
#             scored.append({
#                 **result,
#                 "score": float(score)
#             })
        
#         scored.sort(key=lambda x: x["score"], reverse=True)
#         return scored[:top_k]


# @step()
# def CrossEncoderStep(
#     query: str,
#     results_from_faiss: List[Dict],
#     chunks: List[Dict],
#     top_k: int = 5
# ) -> List[Dict]:
#     reranker = CrossEncoderReRanker(top_k=top_k)
#     reranked_results = reranker.CE(
#         query,
#         results_from_faiss,
#         chunks,
#         top_k
#     )
#     return reranked_results

import os
from zenml import step
from typing import List, Dict
from huggingface_hub import InferenceClient
import logging
import mlflow
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path="E:\pyDS\Buliding Rag System\.env")

class CrossEncoderReRankerHF:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
    ):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )
        self.model_name = model_name
        self.top_k = top_k

        mlflow.log_param("cross_encoder_model", model_name)
        mlflow.log_param("cross_encoder_top_k", top_k)

    def _score_pair(self, query: str, chunk_text: str) -> float:
        """
        Scores a single (query, chunk) pair using HF Inference API
        """
        response = self.client.text_classification(
            text=f"{query} [SEP] {chunk_text}",
            model=self.model_name,
        )

        # HF بيرجع list
        mlflow.log_metric("cross_encoder_inference_score", float(response[0]["score"]))
        return float(response[0]["score"])

    def rerank(
        self,
        query: str,
        results_from_faiss: List[Dict],
        chunks: List[Dict],
    ) -> List[Dict]:

        # Map chunk_id → text (support legacy 'id' key)
        chunk_id_to_text = {
            (c.get("chunk_id") or c.get("id")): c.get("text", "")
            for c in chunks
            if (c.get("chunk_id") or c.get("id"))
        }

        reranked = []

        try:
            for result in results_from_faiss:
                chunk_id = result["chunk_id"]

                if chunk_id not in chunk_id_to_text:
                    continue

                score = self._score_pair(
                    query=query,
                    chunk_text=chunk_id_to_text[chunk_id],
                )

                reranked.append({
                    **result,
                    "score": score,
                })
        except Exception as e:
            logging.warning("Cross-encoder failed (%s). Returning input results unchanged.", e)
            # Fall back to returning original results (up to top_k) if scoring fails
            return results_from_faiss[: self.top_k]

        # Sort by relevance score
        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked[: self.top_k]


@step(enable_cache=False)
def CrossEncoderStep(
    query: str,
    results_from_faiss: List[Dict],
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:

    reranker = CrossEncoderReRankerHF(top_k=top_k)
    reranked_results = reranker.rerank(
        query=query,
        results_from_faiss=results_from_faiss,
        chunks=chunks,
    )
    mlflow.log_param("cross_encoder_top_k", top_k)

    return reranked_results
