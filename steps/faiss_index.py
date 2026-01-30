'''
Docstring for steps.faiss_index
الهدف

نخزن embeddings في FAISS باستخدام IVF (Inverted File Index)

نقدر نعمل Top-K retrieval بسرعة عالية جداً
نحافظ على ربط vector ↔ metadata / chunk_id
نقدر save / load index لأي وقت
IVF يقسم المتجهات لمجموعات (clusters) للبحث السريع
'''
import faiss
import numpy as np
import os
import json
from typing import List, Dict , Optional
import mlflow
from zenml import step
import time

class FaissIndex:
    def __init__(self, vector_dim: int,
                  index_path: str = "data/processed/faiss.index",
                  mapping_path: str = "data/processed/faiss_mapping.json",
                  use_ivf: bool = True,
                  nlist: int = 20,
                  nprobe: int = 3): # fiass_id => chunk_id
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.use_ivf = use_ivf
        self.nlist = nlist  # Number of clusters in IVF
        self.nprobe = nprobe  # Number of clusters to search in
        self.is_trained = False  # Whether IVF is trained

        # Create index using IVF if use_ivf = True
        if self.use_ivf:
            quantizer = faiss.IndexFlatIP(vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist)
        else:
            self.index = faiss.IndexFlatIP(vector_dim)
        
        self.mapping = {}  # faiss_id -> chunk_id
        self.faiss_id = 0
        
        mlflow.log_param("faiss_vector_dim", vector_dim)
        mlflow.log_param("faiss_use_ivf", use_ivf)
        if use_ivf:
            mlflow.log_param("faiss_nlist", nlist)
            mlflow.log_param("faiss_nprobe", nprobe)

    def add(self, embeddings: List[Dict]):
        '''
        embeddings: List of dicts with keys 'chunk_id' and 'vector'
        Adds embeddings to the FAISS index and updates the mapping.
        
        تجميع البيانات أولاً ثم إضافتها دفعة واحدة لتحسين الأداء
        '''
        vectors_to_add = []
        
        # Aggregate all vectors
        for emb in embeddings:
            vector = emb['vector']
            # Convert to numpy array if it's a list
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            else:
                vector = np.array(vector, dtype=np.float32)
            # Ensure vector is 2D (1, dim)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            vectors_to_add.append(vector)
            self.mapping[str(self.faiss_id)] = emb['chunk_id']
            self.faiss_id += 1
        
        # Add all vectors at once
        if vectors_to_add:
            vectors_array = np.vstack(vectors_to_add)
            
            # Train IVF if not trained yet
            if self.use_ivf and not self.is_trained:
                if len(vectors_array) >= self.nlist:
                    print(f"Training IVF index with {len(vectors_array)} vectors...")
                    self.index.train(vectors_array)
                    self.is_trained = True
                    print(f"IVF index trained successfully!")
                else:
                    print(f"Not enough vectors ({len(vectors_array)}) to train IVF (need >= {self.nlist})")
            
            # Add vectors to the index
            self.index.add(vectors_array)

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> List[tuple]:
        '''
        Searches the FAISS index for the top_k nearest neighbors of the query_vector.
        Returns a list of tuples (chunk_id, score) for the nearest neighbors.
        
        استخدام IVF يقلل وقت البحث بشكل كبير
        '''
        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Set nprobe if index is of type IVF
        if self.use_ivf and hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Measure search time (optional)
        start_time = time.time()
        d, I = self.index.search(query_vector, top_k)
        search_time = time.time() - start_time
        
        # FAISS returns -1 for missing results; filter those out
        results = []
        for i, idx in enumerate(I[0]):
            idx = int(idx)
            if idx != -1:
                chunk_id = self.mapping[str(idx)]
                score = float(d[0][i])
                results.append((chunk_id, score))
        
        # Log search time to MLflow
        mlflow.log_metric("faiss_search_time", search_time)
        
        return results  # return list of (chunk_id, score) tuples
    
    def save(self): 
        '''
        Save the faiss index and the mapping to disk
        '''
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)

    def load(self):
        '''
        Load the faiss index and the mapping from disk
        
        إعادة تحميل الـ index المحفوظ مع المعاملات الخاصة به
        '''
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
            # If index is of type IVF, set nprobe
            if self.use_ivf and hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
                self.is_trained = True  # Loaded index is already trained
        
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
            # Update faiss_id to continue from where we left off
            if self.mapping:
                self.faiss_id = max(map(int, self.mapping.keys())) + 1


@step(enable_cache=True)
def faiss_index_step(embeddings: List[Dict], vector_dim: int , x_optiona:Optional[bool] = False):
    '''
    خطوة بناء FAISS Index باستخدام IVF للبحث السريع جداً
    '''
    # Using IVF with nlist=100 and nprobe=10 for good balance between speed and accuracy
    faiss_index = FaissIndex(
        vector_dim=vector_dim,
        use_ivf=True,
        nlist=100,
        nprobe=10
    )
    
    print(f"📊 Building FAISS IVF index with {len(embeddings)} embeddings...")
    faiss_index.add(embeddings)
    faiss_index.save()
    
    mlflow.log_param("faiss_index_size", faiss_index.index.ntotal)
    mlflow.log_param("faiss_is_trained", faiss_index.is_trained)
    
    print(f"✅ FAISS Index built successfully with {faiss_index.index.ntotal} vectors")
    
    return faiss_index