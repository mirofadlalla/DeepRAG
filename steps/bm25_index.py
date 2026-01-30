import nltk
from rank_bm25 import BM25Okapi
from zenml import step
import mlflow
# nltk.download("punkt")

class BM25Indexer:
    def __init__(self, chunks):
        """Initialize BM25 indexer with chunks.
        
        Args:
            chunks: List of dicts with 'chunk_id' and 'text' keys
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.tokenized = [nltk.word_tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)
        mlflow.log_param("num_indexed_chunks", len(chunks))

    def search(self, query, top_k=5):
        """Search for top_k chunks matching the query.
        
        Returns:
            List of dicts with structure: {chunk_id, text, metadata, score}
        """
        q_tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(q_tokens)

        ranked = sorted(
            list(zip(range(len(self.chunks)), scores)),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            chunk = self.chunks[idx]
            # support both current 'chunk_id' and legacy 'id' keys
            cid = chunk.get("chunk_id") or chunk.get("id")
            results.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "score": float(score)
            })
        return results


@step(enable_cache=False)
def bm25_search_step(
    chunks: list,
    query: str,
    top_k: int = 50
) -> list:
    """Search chunks using BM25 algorithm.
    
    Args:
        chunks: List of chunk dicts with chunk_id and text
        query: Search query
        top_k: Number of top results to return
        
    Returns:
        List of result dicts with {chunk_id, text, metadata, score}
    """
    indexer = BM25Indexer(chunks)
    results = indexer.search(query, top_k)
    return results