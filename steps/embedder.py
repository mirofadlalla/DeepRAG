'''
الهدف الحقيقي

تحويل كل chunk إلى vector:

normalized
reusable
cached
deterministic

بحيث:

❌ مفيش chunk يتعمله embedding مرتين
❌ مفيش drift
✅ أي إعادة ingestion تبقى cheap

المشكلة

لو عندك:
100k chunks

كل مرة تشغل السيستم بتعمل embedding من الأول

يبقى:
cost - latency - waste


✅ الحل

نعمل Embedding Cache Layer
يعتمد على:

🔑 Key => hash(chunk_text)


ليه مش chunk_id؟
chunk_id فيه metadata

embedding لازم يعتمد على النص فقط
'''
import os
import hashlib
from typing import List, Dict
import logging

import pickle
from zenml import step
import numpy as np
from sentence_transformers import SentenceTransformer

# إعداد logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class EmbeddingCache:
    def __init__(self, cache_dir: str = "data/processed/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, text_hash: str) -> str:
        '''
        Get the file path for a given text hash
    '''
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def exists(self, text_hash: str) -> bool:
        '''
        Check if embedding for the given text hash exists in cache
        '''
        return os.path.exists(self._path(text_hash))

    def load(self, text_hash: str) -> np.ndarray:
        '''
        Load the embedding for the given text hash from cache
        '''
        return np.load(self._path(text_hash))

    def save(self, text_hash: str, vector: np.ndarray):
        '''
        Save the embedding for the given text hash to cache
        '''
        np.save(self._path(text_hash), vector)


class Embedder:
    def __init__(self, model_name="BAAI/bge-m3", device="cuda", batch_size=32):
        '''
        model_name: name of the sentence transformer model
        device: "cuda" or "cpu"
        batch_size: number of texts to encode in a single batch
        '''
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.cache = EmbeddingCache()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def chunk_embeded(self, chunks):
        '''
        Chunks: List of dicts with keys 'id', 'text', 'metadata
        Returns: List of dicts with keys 'chunk_id', 'vector', 'metadata'
        '''
        embedded = []
        texts_to_encode = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            text_hash = self._hash(text)
            if self.cache.exists(text_hash):
                vector = self.cache.load(text_hash)
                embedded.append({
                    "chunk_id": chunk["id"],
                    # "text" : chunk["text"],
                    "vector": vector,
                    "metadata": chunk["metadata"]
                })
            else:
                texts_to_encode.append(text)
                chunk_indices.append(i)
                embedded.append(None)  # placeholder

        if texts_to_encode:
            logging.info(f"Encoding {len(texts_to_encode)} chunks in batches of {self.batch_size}...")
            vectors = self.model.encode(texts_to_encode, batch_size=self.batch_size, normalize_embeddings=True)
            for idx, vector in zip(chunk_indices, vectors):
                chunk = chunks[idx]
                text_hash = self._hash(chunk["text"])
                self.cache.save(text_hash, vector)
                embedded[idx] = {
                    "chunk_id": chunk["id"],
                    # "text" : chunk["text"],
                    "vector": vector,
                    "metadata": chunk["metadata"]
                }

        logging.info("All chunks embedded ✅")

        return embedded
        # output_path = r"E:\pyDS\Buliding Rag System\embedded_chunks.pkl"

        # print("Saving embedded chunks...")
        # with open(output_path, "wb") as f:
        #     pickle.dump(embedded, f)

        # print("Saved successfully at:", output_path)


@step(enable_cache=True)
def chunks_embedding(chunks) :
    em = Embedder()
    return em.chunk_embeded(chunks)
