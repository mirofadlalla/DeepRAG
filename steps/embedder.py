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
from typing import Dict, List, Any , Optional
import logging

import pickle
from zenml import step
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer

# Configure logging
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
    def __init__(self, model_name="BAAI/bge-m3", device="cuda", batch_size=7):
        '''
        model_name: name of the sentence transformer model
        device: "cuda" or "cpu"
        batch_size: number of texts to encode in a single batch
        '''
        self.batch_size = batch_size
        mlflow.log_param("embedding_model", model_name)
        mlflow.log_param("embedding_device", device)
        self.cache = EmbeddingCache()

        # Try loading the requested model; if it fails (commonly due to memory),
        # fall back to a lightweight model with a clear log message.
        try:
            logging.info(f"Loading SentenceTransformer: {model_name} (device={device})")
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}': {e}")
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            logging.info(f"Falling back to smaller model '{fallback}' (device=cpu). This reduces memory requirements.")
            try:
                self.model = SentenceTransformer(fallback, device="cpu")
            except Exception as e2:
                logging.error(f"Failed to load fallback model '{fallback}': {e2}")
                raise

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
                # Ensure vector is numpy array
                vector_array = np.array(vector, dtype=np.float32)
                embedded.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "vector": vector_array,
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
                # Ensure vector is numpy array
                vector_array = np.array(vector, dtype=np.float32)
                self.cache.save(text_hash, vector_array)
                embedded[idx] = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "vector": vector_array,
                    "metadata": chunk["metadata"]
                }

        logging.info("All chunks embedded ✅")
        
        # Convert numpy arrays to lists for ZenML serialization
        for chunk in embedded:
            if isinstance(chunk.get("vector"), np.ndarray):
                chunk["vector"] = chunk["vector"].tolist()
        
        return embedded


@step(enable_cache=True )
def chunks_embedding(chunks, x_optiona:Optional[bool] = False) :
    em = Embedder()
    return em.chunk_embeded(chunks)
