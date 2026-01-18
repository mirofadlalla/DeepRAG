from pipelines.pipeline import rag_pipline
import pickle
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    questions = "What was the direct financial impact of this change on the Income Statement for the year ended December 31, 2023? Be specific about the amounts for depreciation expense and net income."

    doc_path = "E:\pyDS\Buliding Rag System\data"
    query_vec = model.encode(questions, normalize_embeddings=True)
    ans = rag_pipline(query=questions, query_vector=query_vec, filters=None, doc_path=doc_path)

    print(ans)