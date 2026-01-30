'''
Docstring for steps.faiss_index
الهدف

نخزن embeddings في FAISS

نقدر نعمل Top-K retrieval بسرعة
نحافظ على ربط vector ↔ metadata / chunk_id
نقدر save / load index لأي وقت
'''
import faiss
import numpy as np
import os
import json
from typing import List, Dict
import mlflow
from zenml import step

class FaissIndex:
    def __init__(self, vector_dim: int,
                  index_path: str = "data/processed/faiss.index",
                  mapping_path: str = "data/processed/faiss_mapping.json"): # fiass_id => chunk_id
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.mapping_path = mapping_path



        self.index = faiss.IndexFlatIP(vector_dim)
        self.mapping = {}  # faiss_id -> chunk_id
        self.faiss_id = 0
        mlflow.log_param("faiss_vector_dim", vector_dim)

    def add(self , embeddings: List[Dict]):
        '''
        embeddings: List of dicts with keys 'chunk_id' and 'vector'
        Adds embeddings to the FAISS index and updates the mapping.

        '''
        for emb in embeddings :
            vector = emb['vector']
            # Convert to numpy array if it's a list
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            else:
                vector = np.array(vector, dtype=np.float32)
            # Ensure vector is 2D (1, dim)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            self.index.add(vector)

            self.mapping[str(self.faiss_id)] = emb['chunk_id']
            self.faiss_id +=1

    def search(self , query_vector: np.ndarray, top_k: int = 50) -> List[tuple]:
        '''
        Searches the FAISS index for the top_k nearest neighbors of the query_vector.
        Returns a list of tuples (chunk_id, score) for the nearest neighbors.
        '''

        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        d, I = self.index.search(query_vector, top_k)
        # FAISS returns -1 for missing results; filter those out
        results = []
        for i, idx in enumerate(I[0]):
            idx = int(idx)
            if idx != -1:
                chunk_id = self.mapping[str(idx)]
                score = float(d[0][i])
                results.append((chunk_id, score))
        return results  # return list of (chunk_id, score) tuples
    
    def save(self): 
        '''
        Save the faiss index and the mapping to disk
        '''
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)

    def load(self):
        '''
        Load the faiss index and the mapping from disk
        
        '''
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
            # Update faiss_id to continue from where we left off
            if self.mapping:
                self.faiss_id = max(map(int, self.mapping.keys())) + 1


@step(enable_cache=False)
def faiss_index_step(embeddings: List[Dict], vector_dim: int):
    faiss_index = FaissIndex(vector_dim)
    faiss_index.add(embeddings)
    faiss_index.save()
    mlflow.log_param("faiss_index_size", faiss_index.index.ntotal)
    return faiss_index