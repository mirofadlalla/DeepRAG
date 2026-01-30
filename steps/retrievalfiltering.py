from typing import List, Dict, Optional
import numpy as np
from zenml import step
import mlflow
import time


class Retriever:
    def __init__(self, faiss_index, chunks_metadata: Dict[str, Dict], nprobe: int = 3):
        """
        faiss_index: instance of FaissIndex with IVF support
        chunks_metadata: dict chunk_id -> metadata
        nprobe: عدد المجموعات (clusters) للبحث فيها - قيمة أعلى = دقة أعلى لكن بطء أكثر
        """
        self.faiss_index = faiss_index
        self.chunks_metadata = chunks_metadata
        self.nprobe = nprobe
        
        # Set nprobe if index is of type IVF
        if hasattr(faiss_index.index, 'nprobe'):
            faiss_index.index.nprobe = nprobe
            mlflow.log_param("retriever_nprobe", nprobe)

    def query_retreviel(self,
                        query_vector: np.ndarray,
                        top_k: int = 50,
                        filters: Optional[Dict] = None
                    ) -> List[Dict]:
      """
      البحث عن أفضل النتائج باستخدام FAISS IVF
      """
      # Request more than top_k because some may be filtered out
      candidate_results = self.faiss_index.search(query_vector, top_k * 2)

      results = []

      for cid, score in candidate_results:
        meta = self.chunks_metadata.get(cid)

        if not meta:
          continue

        if filters and "language" in filters and filters['language'] != meta['language']:
          continue

        results.append({
                "chunk_id": cid,
                "text": meta.get("text", ""),
                "metadata":{
                    "source": meta.get("source", ""),
                    "page": meta.get("page", -1),
                    "language": meta.get("language", ""),
                    "doc_type": meta.get("doc_type", "")
                },
                "score": score
            })

      return results[:top_k]

@step(enable_cache=False)
def retrieval_step(
    faiss_index,
    chunks_metadata: Dict[str, Dict],
    query_vector: List[float],
    top_k: int = 50,
    filters: Optional[Dict] = None,
    nprobe: int = 10
) -> List[Dict]:
    """
    خطوة البحث عن الـ chunks المرتبطة باستخدام FAISS IVF
    nprobe: تحكم بـ trade-off بين السرعة والدقة
    """
    retriever = Retriever(faiss_index, chunks_metadata, nprobe=nprobe)
    # Convert list back to numpy array
    query_vector_np = np.array(query_vector, dtype=np.float32)
    
    retrieval_start = time.time()
    results = retriever.query_retreviel(
      query_vector_np,
      top_k,
      filters
    )
    retrieval_end = time.time()
    
    retrieval_latency_ms = (retrieval_end - retrieval_start) * 1000
    mlflow.log_metric("retrieval_execution_latency_ms", retrieval_latency_ms)
    mlflow.log_param("retrieval_top_k", top_k)
    
    print(f"⚡ Retrieval completed in {retrieval_latency_ms:.2f}ms with nprobe={nprobe}")
    
    return results