from collections import defaultdict
from zenml import step
import mlflow
class RRFFusion:
    def __init__(self, k=30):
        self.k = k

    def fuse(self, dense_results, sparse_results, top_k=10):
        scores = defaultdict(float)

        # dense_results and sparse_results are lists of dicts with 'chunk_id' and 'score'
        for rank, item in enumerate(dense_results, start=1):
            # item expected to be a dict
            doc_id = item.get("chunk_id") if isinstance(item, dict) else None
            if not doc_id:
                continue
            scores[doc_id] += 1 / (self.k + rank)

        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item.get("chunk_id") if isinstance(item, dict) else None
            if not doc_id:
                continue
            scores[doc_id] += 1 / (self.k + rank)

        # Build fused list preserving score and best metadata where available
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_chunk_ids = [cid for cid, _ in fused[:top_k]]

        # Build a lookup from available results to full item (prefer dense then sparse)
        lookup = {}
        for item in dense_results + sparse_results:
            if isinstance(item, dict) and item.get("chunk_id"):
                lookup[item["chunk_id"]] = item

        fused_items = []
        for cid in fused_chunk_ids:
            base = lookup.get(cid, {"chunk_id": cid})
            merged = {**base, "fused_score": scores[cid]}
            fused_items.append(merged)

        mlflow.log_param("fusion_top_k", top_k)

        return fused_items


@step(enable_cache=False)
def fusion_step(
    faiss_results: list,
    bm25_results: list,
    top_k: int = 10
) -> list:
    fusioner = RRFFusion(k=60)
    fused_results = fusioner.fuse(faiss_results, bm25_results, top_k)
    return fused_results