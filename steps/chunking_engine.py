from typing import List, Dict
import hashlib
from transformers import AutoTokenizer

from zenml import step

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 80
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Dict]) -> List[Dict]:
        chunks = []

        for doc in documents:
            tokens = tokenizer.encode(doc["text"])
            start = 0

            while start < len(tokens):
                end = start + self.chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)

                chunk_id = self._make_chunk_id(
                    chunk_text,
                    doc["metadata"]
                )

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": doc["metadata"]
                })

                start += self.chunk_size - self.overlap

        return chunks

    def _make_chunk_id(self, text: str, metadata: Dict) -> str:
        base = (
            text
            + metadata["source"]
            + str(metadata["page"])
        )
        return hashlib.sha256(base.encode()).hexdigest()


@step(enable_cache=True)
def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 300,
    overlap: int = 80
) -> List[Dict]:
    chunker = RecursiveChunker()
    return chunker.chunk(documents)
