# الهدف

# ناخد Top-50 chunks من Retriever
# نعيد ترتيبهم باستخدام Cross-Encoder
# نطلع Top-5 فقط بأعلى دقة ممكنة
import numpy as np
from typing import List, Dict

from zenml import step

from sentence_transformers import CrossEncoder

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base" ,
                 device = "cuda" ,
                 top_k: int = 5) -> List[Dict]:

        self.model = CrossEncoder(model_name , device=device)

    def CE(self , query , rsults_from_faiss  , chunks, top_k = 5):
        chunks_ids_from_faiss = [r['chunk_id'] for r in rsults_from_faiss]
        texts = [chunks[i]['text'] for i in range(len(chunks)) if chunks[i]['id'] in chunks_ids_from_faiss]

        pairs = [
            (query, t) for t in texts
        ]

        scores = self.model.predict(pairs)

        scored = []
        for c, score in zip(rsults_from_faiss, scores):
            scored.append({
                **c,
                "score": float(score)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:top_k]


@step()
def cross_encoder_rerank_step(
    query: str,
    results_from_faiss: List[Dict],
    chunks: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    reranker = CrossEncoderReRanker(top_k=top_k)
    reranked_results = reranker.CE(
        query,
        results_from_faiss,
        chunks,
        top_k
    )
    return reranked_results