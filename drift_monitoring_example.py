"""Example of using Drift Monitoring with Benchmark"""

from drift_detection import (
    DriftDetectionMonitor,
    DriftScheduler,
    generate_drift_report,
    detect_drift_from_benchmarks,
    log_drift_to_mlflow
)
from benchmark import eval_dataset
import json
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import mlflow
from pipelines.pipeline import rag_pipeline

# ===============================
# Configuration
# ===============================

MODEL_NAME = "BAAI/bge-m3"
DOC_PATH = r"E:\pyDS\Buliding Rag System\data"
BATCH_SIZE = 50  # Check every 50 questions
DAILY_CHECK_HOUR = 2  # 2 AM in the morning

# ===============================
# Setup
# ===============================

model = SentenceTransformer(MODEL_NAME, device="cpu")

# Create drift monitor
monitor = DriftDetectionMonitor(
    batch_size=BATCH_SIZE,
    check_daily=True,
    daily_check_hour=DAILY_CHECK_HOUR,
    alert_threshold="WARNING",  # Alert on WARNING or CRITICAL
    metrics_dir="benchmark_reports"
)

# Create scheduler (optional)
try:
    scheduler = DriftScheduler()
    scheduler.setup(monitor, daily_check_hour=DAILY_CHECK_HOUR)
    scheduler.start()
    print("✅ Scheduler started for background monitoring")
except ImportError:
    print("⚠️ APScheduler not available. Using batch monitoring only.")
    scheduler = None

# ===============================
# Benchmark with Drift Monitoring
# ===============================

mlflow.set_experiment("DeepRAG-Benchmark-With-Drift")

with mlflow.start_run(run_name="benchmark_with_monitoring"):
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("daily_check_hour", DAILY_CHECK_HOUR)
    mlflow.log_param("dataset_size", len(eval_dataset))
    
    all_results = []
    all_metrics_list = []
    
    for idx, item in enumerate(eval_dataset):
        question = item["question"]
        relevant_ids = item["relevant_ids"]
        
        # Encode question
        query_vec = model.encode(question, normalize_embeddings=True)
        query_vec_list = query_vec.tolist()
        
        # Run RAG Pipeline
        print(f"\n[{idx+1}/{len(eval_dataset)}] Processing: {item['id']}")
        final_answer = rag_pipeline(
            query=question,
            query_vector=query_vec_list,
            filters=None,
            relevant_ids=relevant_ids,
            doc_path=DOC_PATH
        )
        
        # Load metrics from pickle
        metrics_file = os.path.join("evaluation_results", f"{hash(question)}_metrics.pkl")
        question_metrics = {}
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "rb") as f:
                question_metrics = pickle.load(f)
        
        # Convert final_answer to string or dict for JSON serialization
        # Handle both string and PipelineRunResponse objects
        if hasattr(final_answer, '__dict__'):
            # It's an object like PipelineRunResponse
            answer_str = str(final_answer)
        else:
            # It's already a string or primitive
            answer_str = final_answer
        
        all_results.append({
            "id": item["id"],
            "question": question,
            "answer": answer_str,
            "metrics": question_metrics
        })
        all_metrics_list.append(question_metrics)
        
        # Add metrics to monitor (automatically triggers check if needed)
        is_alert = monitor.add_metrics(question_metrics)
        
        if is_alert:
            print(f"⚠️ DRIFT DETECTED! Check monitor.alerts for details")
        
        # Log progress
        mlflow.log_metric("processed_questions", idx + 1)
    
    # ===============================
    # Save Results
    # ===============================
    
    os.makedirs("benchmark_reports", exist_ok=True)
    
    with open("benchmark_reports/full_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    with open("benchmark_reports/all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics_list, f, indent=4, ensure_ascii=False)
    
    # ===============================
    # Final Drift Check
    # ===============================
    
    print("\n" + "="*70)
    print("🔍 FINAL DRIFT DETECTION CHECK")
    print("="*70)
    
    final_drift_results = detect_drift_from_benchmarks(all_metrics_list)
    
    # Log drift results to the current MLflow run instead of creating a new one
    try:
        mlflow.log_param("final_drift_status", final_drift_results.get("status", "unknown"))
        if "metrics" in final_drift_results:
            for metric_name, metric_value in final_drift_results["metrics"].items():
                try:
                    mlflow.log_metric(f"final_drift_{metric_name}", float(metric_value))
                except:
                    mlflow.log_param(f"final_drift_{metric_name}", str(metric_value))
    except Exception as e:
        print(f"⚠️ Warning: Could not log drift results to MLflow: {e}")
    
    # ===============================
    # Generate Report
    # ===============================
    
    report = generate_drift_report(
        monitor,
        output_file="benchmark_reports/drift_report.txt"
    )
    print(report)
    
    # Log report to MLflow
    mlflow.log_artifact("benchmark_reports/drift_report.txt")
    
    # ===============================
    # Summary
    # ===============================
    
    summary = monitor.get_summary()
    
    print("\n" + "="*70)
    print("📊 MONITORING SUMMARY")
    print("="*70)
    print(f"Total Drift Checks: {summary['total_checks']}")
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Current Status: {summary['current_status']}")
    print(f"Questions Processed: {idx + 1}")
    
    # Log summary to MLflow
    mlflow.log_metric("total_drift_checks", summary['total_checks'])
    mlflow.log_metric("total_drift_alerts", summary['total_alerts'])
    mlflow.log_param("final_drift_status", summary['current_status'])
    
    print("\n✅ Benchmark with Drift Monitoring completed!")
    print(f"📁 Results saved in: benchmark_reports/")
    print(f"📄 Report: benchmark_reports/drift_report.txt")

# ===============================
# Cleanup
# ===============================

if scheduler:
    # Let the scheduler run in the background
    # Call scheduler.stop() when done
    print("\n✅ Scheduler running in background. Call scheduler.stop() to stop it.")
