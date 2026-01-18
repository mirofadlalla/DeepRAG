from typing import List, Dict
import numpy as np
from zenml import step


class Retriever:
    def __init__(self, faiss_index, chunks_metadata: Dict[str, Dict]):
        """
        faiss_index: instance of FaissIndex
        chunks_metadata: dict chunk_id -> metadata
        """
        self.faiss_index = faiss_index
        self.chunks_metadata = chunks_metadata

    def query_retreviel(self,
                        query_vector: np.ndarray,
                        top_k: int = 50,
                        filters: Dict = None
                    ) -> List[Dict]:
      canditae_ids = self.faiss_index.search(query_vector , top_k * 2)

      results = []

      for cid in canditae_ids :
        meta = self.chunks_metadata.get(cid)

        if not meta :
          continue

        if "language" in filters and filters['language'] != meta['language'] :
          continue

        results.append({
                "chunk_id": cid,
                # "text": meta.get("text" , ""),
                "metadata": meta
            })

      return results[:top_k]

@step()
def retrieval_step(
    faiss_index,
    chunks_metadata: Dict[str, Dict],
    query_vector: np.ndarray,
    top_k: int = 50,
    filters: Dict = None
) -> List[Dict]:
    retriever = Retriever(faiss_index, chunks_metadata)
    results = retriever.query_retreviel(
        query_vector,
        top_k,
        filters
    )
    return results