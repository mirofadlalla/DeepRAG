import os
import logging
import pickle
from typing import List, Dict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from zenml import step
import mlflow

load_dotenv()

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ================= Retrieval Metrics =================
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k

def mean_reciprocal_rank(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def retrieval_metrics(retrieved: List[str], relevant: List[str], k_values: List[int] = [1,3,5,10]) -> Dict:
    metrics = {"mrr": mean_reciprocal_rank(retrieved, relevant)}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
    return metrics

# ================= Answer Relevance =================
def answer_relevance(question: str, answer: str) -> bool:
    client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Is the answer relevant? Reply ONLY with Yes or No.
    """
    try:
        res = client.text_generation(
            prompt=prompt,
            model=HF_MODEL,
            max_new_tokens=5,
            temperature=0.0
        )
        return "yes" in res.lower()
    except Exception as e:
        logging.error(f"Answer relevance check failed: {e}")
        return False

# ================= Hallucination Detection =================
def hallucination_rate(question: str, answer: str, context_chunks: List[Dict]) -> Dict:
    client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
    context = "\n".join(c.get("text","") for c in context_chunks)
    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    {answer}

    Rate hallucination from 0 to 10. Reply ONLY with a number.
    """
    try:
        res = client.text_generation(
            prompt=prompt,
            model=HF_MODEL,
            max_new_tokens=5,
            temperature=0.0
        )
        score = int("".join(filter(str.isdigit, res)))
        score = min(max(score,0),10)/10.0
    except Exception as e:
        logging.error(f"Hallucination detection failed: {e}")
        score = 0.0
    return {"hallucination_score": score, "is_hallucinating": score>0.5}

# ================= Stability =================
def jaccard_similarity(a: List[str], b: List[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def retrieval_stability_test(retriever, question: str, runs: int = 3, k: int = 5):
    all_runs = []
    for _ in range(runs):
        chunks = retriever(question, top_k=k)
        ids = [c["chunk_id"] for c in chunks]
        all_runs.append(ids)
    base = all_runs[0]
    scores = [jaccard_similarity(base, run) for run in all_runs[1:]]
    return {"avg_jaccard": sum(scores)/len(scores), "runs": all_runs}

def rephrase_stability_test(retriever, question: str, paraphrases: List[str], k: int = 5):
    base_ids = [c["chunk_id"] for c in retriever(question, top_k=k)]
    scores = []
    for p in paraphrases:
        ids = [c["chunk_id"] for c in retriever(p, top_k=k)]
        scores.append(jaccard_similarity(base_ids, ids))
    return sum(scores)/len(scores)

# ================= ZenML Step =================
@step(enable_cache=False)
def evaluate_pipeline(
    question: str,
    retrieved_chunks: List[Dict],
    answer: str,
    relevant_ids: List[str],
    context_chunks: List[Dict],
    retriever=None,
    paraphrases: List[str] = []
) -> Dict:

    # Extract retrieved IDs
    retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

    # Compute metrics
    retrieval_met = retrieval_metrics(retrieved_ids, relevant_ids)
    relevance = answer_relevance(question, answer)
    hallucination_met = hallucination_rate(question, answer, context_chunks) if context_chunks else {}
    stability_retrieval = retrieval_stability_test(retriever, question) if retriever else {}
    stability_rephrase = rephrase_stability_test(retriever, question, paraphrases) if retriever and paraphrases else {}

    all_metrics = {
        "question": question,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        "retrieval_metrics": retrieval_met,
        "answer": answer,
        "answer_relevance": relevance,
        "hallucination": hallucination_met,
        "stability_retrieval": stability_retrieval,
        "stability_rephrase": stability_rephrase
    }

    # Save in MLflow
    mlflow.set_experiment("DeepRAG-Evaluation")
    with mlflow.start_run(run_name=f"evaluation_{hash(question)}"):
        # Log retrieval metrics
        mlflow.log_metrics(retrieval_met)
        mlflow.log_metric("answer_relevance", int(relevance))
        if hallucination_met:
            mlflow.log_metric("hallucination_score", hallucination_met.get("hallucination_score",0))
        if stability_retrieval:
            mlflow.log_metric("stability_avg_jaccard", stability_retrieval.get("avg_jaccard",0))
        if stability_rephrase:
            mlflow.log_metric("stability_rephrase_avg_jaccard", stability_rephrase)

    # Save all results to file
    os.makedirs("evaluation_results", exist_ok=True)
    file_path = os.path.join("evaluation_results", f"{hash(question)}_metrics.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(all_metrics, f)

    print(f"✅ Evaluation metrics saved: {file_path}")
    return all_metrics
