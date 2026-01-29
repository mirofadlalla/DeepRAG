from pipelines.pipeline import rag_pipline
import pickle
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    questions = "What was the outcome, as of the document's date, of the Epic Games v. Google trial in California in December 2023?"

    doc_path = r"E:\pyDS\Buliding Rag System\data"
    query_vec = model.encode(questions, normalize_embeddings=True)
    # Convert numpy array to list for Pydantic serialization
    query_vec_list = query_vec.tolist() if isinstance(query_vec, __import__('numpy').ndarray) else query_vec
    run = rag_pipline(query=questions, query_vector=query_vec_list, filters=None, doc_path=doc_path)
