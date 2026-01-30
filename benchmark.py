import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import mlflow
from typing import Dict

from pipelines.pipeline import rag_pipeline
from metrics.metrics import (
    retrieval_metrics,
    answer_relevance,
    hallucination_rate,
    jaccard_similarity
)

# ----------------- CONFIG -----------------
MLFLOW_EXPERIMENT = "DeepRAG-Benchmark"
MODEL_NAME = "BAAI/bge-m3"
DOC_PATH = r"E:\pyDS\Buliding Rag System\data"
TOP_K = 5
PARAPHRASES_KEY = "paraphrases"
EVAL_DATASET_PATH = r"evaluation_dataset.json" 

# ----------------- LOAD DATASET -----------------
with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

# ----------------- LOAD MODEL -----------------
model = SentenceTransformer(MODEL_NAME, device="cpu")

# ----------------- SETUP MLflow -----------------
mlflow.set_experiment(MLFLOW_EXPERIMENT)

if __name__ == "__main__":
    with mlflow.start_run(run_name="benchmark_full_dataset"):
        # Log hyperparameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("doc_path", DOC_PATH)
        mlflow.log_param("dataset_size", len(eval_dataset))

        all_results = []
        all_metrics_list = []
        pickle_files_list = []
        
        for idx, item in enumerate(eval_dataset):
            question = item["question"]
            relevant_ids = item["relevant_ids"]
            paraphrases = item.get(PARAPHRASES_KEY, [])

            # Encode question
            query_vec = model.encode(question, normalize_embeddings=True)
            query_vec_list = query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec

            # Run RAG Pipeline - saves metrics pickle in evaluation_results
            final_answer = rag_pipeline(
                query=question,
                query_vector=query_vec_list,
                filters=None,
                relevant_ids=relevant_ids,
                doc_path=DOC_PATH
            )

            # Add pickle file path to list
            metrics_file = os.path.join("evaluation_results", f"{hash(question)}_metrics.pkl")
            pickle_files_list.append(metrics_file)
            
            # Load metrics from pickle file
            question_metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, "rb") as f:
                    question_metrics = pickle.load(f)
            
            # Convert final_answer to string to avoid JSON serialization issues
            answer_str = str(final_answer)
            
            # Combine results with metrics
            result_item = {
                "id": item["id"],
                "question": question,
                "answer": answer_str,
                "metrics": question_metrics
            }
            
            # Save results and metrics
            all_results.append(result_item)
            all_metrics_list.append(question_metrics)
            
            # Log metrics to MLflow
            if question_metrics:
                # Log retrieval metrics
                if "retrieval_metrics" in question_metrics:
                    ret_met = question_metrics["retrieval_metrics"]
                    for metric_name, metric_value in ret_met.items():
                        try:
                            mlflow.log_metric(f"q{idx+1}_{metric_name}", metric_value)
                        except:
                            pass
                
                # Log answer relevance
                if "answer_relevance" in question_metrics:
                    mlflow.log_metric(f"q{idx+1}_answer_relevance", int(question_metrics["answer_relevance"]))
                
                # Log hallucination metrics
                if "hallucination" in question_metrics and question_metrics["hallucination"]:
                    hall = question_metrics["hallucination"]
                    if "hallucination_score" in hall:
                        mlflow.log_metric(f"q{idx+1}_hallucination_score", hall["hallucination_score"])
            
            # Log progress
            mlflow.log_metric("processed_questions", idx + 1)
            print(f"✅ Processed {idx + 1}/{len(eval_dataset)}: {item['id']}")

        # Save all results once
        os.makedirs("benchmark_reports", exist_ok=True)
        
        with open("benchmark_reports/full_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        df = pd.DataFrame(all_results)
        df.to_csv("benchmark_reports/full_results.csv", index=False)
        
        # Save all metrics as JSON once
        with open("benchmark_reports/all_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics_list, f, indent=4, ensure_ascii=False)
        
        # Save pickle file paths list
        with open("benchmark_reports/pickle_files_list.json", "w", encoding="utf-8") as f:
            json.dump(pickle_files_list, f, indent=4, ensure_ascii=False)
        
        # Save all pickle files in a separate backup folder
        metrics_backup_dir = os.path.join("benchmark_reports", "metrics_pickles")
        os.makedirs(metrics_backup_dir, exist_ok=True)
        
        for idx, pkl_file in enumerate(pickle_files_list):
            if os.path.exists(pkl_file):
                # copy to backup dir
                dest_file = os.path.join(metrics_backup_dir, f"metrics_{idx+1}.pkl")
                with open(pkl_file, "rb") as src:
                    metrics_data = pickle.load(src)
                with open(dest_file, "wb") as dst:
                    pickle.dump(metrics_data, dst)
        
        # load artifacts to mlflow
        mlflow.log_artifact("benchmark_reports/full_results.json")
        mlflow.log_artifact("benchmark_reports/full_results.csv")
        mlflow.log_artifact("benchmark_reports/all_metrics.json")
        mlflow.log_artifact("benchmark_reports/pickle_files_list.json")
        mlflow.log_artifact(metrics_backup_dir)
        
        # Log summary metrics
        mlflow.log_metric("total_questions_processed", len(all_results))
        
        # Calculate and log aggregate metrics
        if all_metrics_list:
            # Average retrieval metrics
            all_retrieval_metrics = [m.get("retrieval_metrics", {}) for m in all_metrics_list]
            if all_retrieval_metrics:
                for metric_name in ["mrr", "recall@1", "recall@3", "recall@5", "recall@10", 
                                   "precision@1", "precision@3", "precision@5", "precision@10"]:
                    values = [m.get(metric_name, 0) for m in all_retrieval_metrics if metric_name in m]
                    if values:
                        avg_value = sum(values) / len(values)
                        mlflow.log_metric(f"avg_{metric_name}", avg_value)
            
            # Average hallucination score
            hall_scores = [m.get("hallucination", {}).get("hallucination_score", 0) 
                          for m in all_metrics_list]
            if hall_scores:
                mlflow.log_metric("avg_hallucination_score", sum(hall_scores) / len(hall_scores))
            
            # Average answer relevance
            relevance_scores = [int(m.get("answer_relevance", 0)) for m in all_metrics_list]
            if relevance_scores:
                mlflow.log_metric("avg_answer_relevance", sum(relevance_scores) / len(relevance_scores))
        
        print("✅ Benchmark finished. Results saved in benchmark_reports/")
        print(f"📊 MLflow Run ID: {mlflow.active_run().info.run_id}")
