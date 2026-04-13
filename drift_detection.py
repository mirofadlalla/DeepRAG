import numpy as np
import os
import json
import pickle
from typing import List, Dict, Optional, Callable
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import threading
from collections import deque

# ===============================
# Baseline Management
# ===============================

def save_baseline_embeddings(embeddings: np.ndarray, baseline_dir: str = "baselines") -> str:
    """
    Save baseline embeddings for future comparison
    
    Args:
        embeddings: numpy array of embeddings
        baseline_dir: folder to save baseline
    
    Returns:
        Path to saved file
    """
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_path = os.path.join(baseline_dir, "baseline_embeddings.npy")
    np.save(baseline_path, embeddings)
    print(f"✅ Baseline embeddings saved: {baseline_path}")
    return baseline_path


def load_baseline_embeddings(baseline_dir: str = "baselines") -> Optional[np.ndarray]:
    """
    Load baseline embeddings
    """
    baseline_path = os.path.join(baseline_dir, "baseline_embeddings.npy")
    if os.path.exists(baseline_path):
        embeddings = np.load(baseline_path)
        print(f"✅ Baseline embeddings loaded: {baseline_path}")
        return embeddings
    print(f"⚠️ No baseline embeddings found at {baseline_path}")
    return None


# ===============================
# Query Drift
# ===============================

def query_drift_score(
    baseline_embeddings: np.ndarray,
    live_embeddings: np.ndarray
) -> float:
    """
    Calculate drift score in question embeddings
    
    0.0  → no drift
    >0.3 → Severe drift
    
    Args:
        baseline_embeddings: embeddings from baseline dataset
        live_embeddings: embeddings from live/current questions
        
    Returns:
        drift score (0-1)
    """
    try:
        sims = cosine_similarity(live_embeddings, baseline_embeddings)
        max_sims = sims.max(axis=1)
        drift = 1 - max_sims.mean()
        return float(drift)
    except Exception as e:
        print(f"❌ Error calculating query drift: {e}")
        return 0.0


# ===============================
# Retrieval Drift
# ===============================

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """
    Calculate Jaccard similarity between two lists
    
    Args:
        a: first list of items
        b: second list of items
    
    Returns:
        Jaccard similarity score (0-1)
    """
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def retrieval_drift_test(
    retrieved_results_run1: List[Dict],
    retrieved_results_run2: List[Dict],
    top_k: int = 5
) -> float:
    """
    Compare retrieval results between two runs
    (same question - compare results)
    
    Args:
        retrieved_results_run1: retrieval results from first run
        retrieved_results_run2: retrieval results from second run
        top_k: number of results to take
        
    Returns:
        Jaccard similarity score
    """
    ids1 = [c["chunk_id"] for c in retrieved_results_run1[:top_k]]
    ids2 = [c["chunk_id"] for c in retrieved_results_run2[:top_k]]
    
    return jaccard_similarity(ids1, ids2)


def retrieval_stability_from_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Calculate stability metrics from saved metrics list
    
    Args:
        metrics_list: list of evaluation metrics from each run
        
    Returns:
        stability metrics dict with average, min, max
        
    Example:
        {"avg_stability": 0.95, "min_stability": 0.85, "max_stability": 1.0, "stability_std": 0.05}
    """
    if not metrics_list:
        return {}
    
    stability_scores = [
        m.get("stability_retrieval", {}).get("avg_jaccard", 1.0)
        for m in metrics_list
    ]
    
    if stability_scores:
        return {
            "avg_stability": sum(stability_scores) / len(stability_scores),
            "min_stability": min(stability_scores),
            "max_stability": max(stability_scores),
            "stability_std": np.std(stability_scores)
        }
    return {}


# ===============================
# System-Level Decision
# ===============================

def drift_decision(
    query_drift: float,
    retrieval_stability: float,
    hallucination_score: float = 0.0
) -> str:
    """
    Make decision based on drift metrics
    
    Args:
        query_drift: drift score in questions (0-1)
        retrieval_stability: Jaccard similarity for retrieval (0-1)
        hallucination_score: hallucination score (0-1)
        
    Returns:
        "CRITICAL" | "WARNING" | "STABLE"
    """
    # Critical thresholds
    if query_drift > 0.3 or retrieval_stability < 0.4 or hallucination_score > 0.7:
        return "CRITICAL"
    
    # Warning thresholds
    if query_drift > 0.15 or retrieval_stability < 0.7 or hallucination_score > 0.5:
        return "WARNING"
    
    return "STABLE"


# ===============================
# Integration with Benchmark
# ===============================

def detect_drift_from_benchmarks(
    current_metrics: List[Dict],
    baseline_metrics: Optional[List[Dict]] = None,
    model_name: str = "BAAI/bge-m3"
) -> Dict:
    """
    Detect drift by comparing benchmark metrics
    
    Args:
        current_metrics: metrics from current run
        baseline_metrics: baseline metrics (optional)
        model_name: embedding model name
        
    Returns:
        Dict with keys:
            - query_drift (float): shift score in questions
            - retrieval_stability (float): retrieval results stability
            - avg_hallucination (float): average hallucination
            - drift_status (str): "STABLE" or "WARNING" or "CRITICAL"
            - recommendations (List[str]): improvement instructions
    """
    results = {
        "query_drift": 0.0,
        "retrieval_stability": 1.0,
        "avg_hallucination": 0.0,
        "drift_status": "STABLE",
        "recommendations": []
    }
    
    if not current_metrics:
        return results
    
    # all_metrics = { all metric values
    #     "question": question,
    #     "retrieved_ids": retrieved_ids,
    #     "relevant_ids": relevant_ids,
    #     "retrieval_metrics": retrieval_met,
    #     "answer": answer_text,
    #     "answer_relevance": relevance,
    #     "hallucination": hallucination_met,
    #     "stability_retrieval": stability_retrieval,
    #     "stability_rephrase": stability_rephrase,
    #     "retriever_type": retriever  # Log which retriever was used
    # }

    # Calculate average retrieval stability
    stability_scores = [
        m.get("stability_retrieval", {}).get("avg_jaccard", 1.0)
        for m in current_metrics
    ]
    avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 1.0
    results["retrieval_stability"] = avg_stability
    
    # Calculate average hallucination
    hall_scores = [
        m.get("hallucination", {}).get("hallucination_score", 0.0)
        for m in current_metrics
    ]
    avg_hallucination = sum(hall_scores) / len(hall_scores) if hall_scores else 0.0
    results["avg_hallucination"] = avg_hallucination
    
    # Calculate query drift if baseline exists
    if baseline_metrics:
        baseline_stability = sum([
            m.get("stability_retrieval", {}).get("avg_jaccard", 1.0)
            for m in baseline_metrics
        ]) / len(baseline_metrics) if baseline_metrics else 1.0
        
        results["query_drift"] = abs(avg_stability - baseline_stability)
    
    # Make decision
    results["drift_status"] = drift_decision(
        results["query_drift"],
        results["retrieval_stability"],
        results["avg_hallucination"]
    )
    
    # Recommendations
    if results["drift_status"] == "CRITICAL":
        results["recommendations"] = [
            "🚨 Baseline embeddings update required",
            "🔧 May need to retrain embedding model",
            "⚙️ Check quality of new documents"
        ]
    elif results["drift_status"] == "WARNING":
        results["recommendations"] = [
            "⚠️ Monitor drift closely",
            "📊 Compare with old baseline",
            "🔄 May need re-indexing"
        ]
    else:
        results["recommendations"] = [
            "✅ System is stable",
            "📈 Metrics within expected range"
        ]
    
    return results


# ===============================
# Logging
# ===============================

def log_drift_to_mlflow(drift_results: Dict, run_name: str = "drift_detection") -> bool:
    """
    Log drift results to MLflow
    
    Args:
        drift_results: drift detection results
        run_name: run name in MLflow
    
    Returns:
        True if logging succeeded, False otherwise
    """
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("query_drift", drift_results.get("query_drift", 0))
            mlflow.log_metric("retrieval_stability", drift_results.get("retrieval_stability", 1))
            mlflow.log_metric("avg_hallucination", drift_results.get("avg_hallucination", 0))
            mlflow.log_param("drift_status", drift_results.get("drift_status", "UNKNOWN"))
            
            for idx, rec in enumerate(drift_results.get("recommendations", [])):
                mlflow.log_param(f"recommendation_{idx+1}", rec)
            
            print(f"✅ Drift detection results logged to MLflow")
            return True
    except Exception as e:
        print(f"❌ Failed to log drift results: {e}")
        return False


# ===============================
# Batch & Periodic Monitoring
# ===============================

class DriftDetectionMonitor:
    """
    Periodic drift monitor
    - Every N questions (batch monitoring)
    - Daily (scheduled monitoring)
    - Instant alerts when drift is detected
    """
    
    def __init__(
        self,
        batch_size: int = 50,
        check_daily: bool = True,
        daily_check_hour: int = 2,  # 2 AM
        alert_threshold: str = "WARNING",
        metrics_dir: str = "benchmark_reports"
    ):
        """
        Args:
            batch_size: number of questions before each check
            check_daily: enable daily check
            daily_check_hour: hour for daily check
            alert_threshold: alert level (WARNING or CRITICAL)
            metrics_dir: folder to save metrics
        """
        self.batch_size = batch_size
        self.check_daily = check_daily
        self.daily_check_hour = daily_check_hour
        self.alert_threshold = alert_threshold
        self.metrics_dir = metrics_dir
        
        # Trackers
        self.question_counter = 0
        self.last_check_time = datetime.now()
        self.metrics_buffer = deque(maxlen=batch_size)  # Last N metrics
        self.drift_history = []
        self.alerts = []
        
        # Tracking file
        self.monitor_log_file = os.path.join(metrics_dir, "drift_monitor.json")
        os.makedirs(metrics_dir, exist_ok=True)
        self._load_history()
        
        print(f"✅ DriftDetectionMonitor initialized")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Daily check: {check_daily} at {daily_check_hour}:00")
        print(f"   - Alert threshold: {alert_threshold}")
    
    
    def _load_history(self):
        """Load previous drift history"""
        if os.path.exists(self.monitor_log_file):
            try:
                with open(self.monitor_log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.drift_history = data.get("drift_history", [])
                    self.alerts = data.get("alerts", [])
                    print(f"✅ Loaded drift history: {len(self.drift_history)} checks")
            except Exception as e:
                print(f"⚠️ Could not load history: {e}")
    
    
    def _save_history(self):
        """Save drift history"""
        try:
            data = {
                "drift_history": self.drift_history,
                "alerts": self.alerts,
                "last_update": datetime.now().isoformat()
            }
            with open(self.monitor_log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save history: {e}")
    
    
    def add_metrics(self, metric: Dict) -> bool:
        """
        Add new metric and check if checking is needed
        
        Returns:
            True if check was performed
        """
        self.metrics_buffer.append(metric)
        self.question_counter += 1
        
        # Check batch condition
        if self.question_counter % self.batch_size == 0:
            print(f"\n🔍 Batch check triggered after {self.question_counter} questions")
            return self._run_drift_check(f"batch_{self.question_counter}")
        
        # Check daily condition
        if self.check_daily and self._should_run_daily_check():
            print(f"\n🔍 Daily drift check triggered")
            return self._run_drift_check("daily_check")
        
        return False
    
    
    def _should_run_daily_check(self) -> bool:
        """Check if daily check is needed"""
        now = datetime.now()
        next_check_time = self.last_check_time.replace(
            hour=self.daily_check_hour,
            minute=0,
            second=0,
            microsecond=0
        )
        
        if next_check_time < self.last_check_time:
            next_check_time += timedelta(days=1)
        
        return now >= next_check_time
    
    
    def _run_drift_check(self, check_type: str) -> bool:
        """
        Run drift check
        
        Args:
            check_type: type of check (batch/daily/scheduled)
        
        Returns:
            True if drift was detected, False otherwise
        """
        if len(self.metrics_buffer) == 0:
            print("⚠️ No metrics to check")
            return False
        
        # Compute metrics from buffer
        current_metrics = list(self.metrics_buffer)
        drift_results = detect_drift_from_benchmarks(current_metrics)
        
        # Log results
        check_record = {
            "timestamp": datetime.now().isoformat(),
            "check_type": check_type,
            "question_count": self.question_counter,
            "metrics_checked": len(current_metrics),
            "drift_results": drift_results
        }
        
        self.drift_history.append(check_record)
        self._save_history()
        
        # Log to MLflow
        try:
            log_drift_to_mlflow(drift_results, run_name=f"drift_{check_type}_{len(self.drift_history)}")
        except:
            pass
        
        # Check alerts
        drift_status = drift_results.get("drift_status", "STABLE")
        is_alert = (drift_status == "CRITICAL") or \
                  (drift_status == "WARNING" and self.alert_threshold == "WARNING")
        
        if is_alert:
            alert = self._create_alert(drift_results, check_type)
            self.alerts.append(alert)
            self._send_alert(alert)
            self._save_history()
            return True
        
        print(f"✅ Drift check completed: {drift_status}")
        self.last_check_time = datetime.now()
        
        return False
    
    
    def _create_alert(self, drift_results: Dict, check_type: str) -> Dict:
        """Create alert"""
        return {
            "timestamp": datetime.now().isoformat(),
            "check_type": check_type,
            "drift_status": drift_results.get("drift_status"),
            "severity_metrics": {
                "query_drift": drift_results.get("query_drift", 0),
                "retrieval_stability": drift_results.get("retrieval_stability", 1),
                "avg_hallucination": drift_results.get("avg_hallucination", 0)
            },
            "recommendations": drift_results.get("recommendations", [])
        }
    
    
    def _send_alert(self, alert: Dict):
        """Send alert"""
        status = alert.get("drift_status")
        print(f"\n{'='*60}")
        print(f"🚨 DRIFT ALERT - {status}")
        print(f"{'='*60}")
        print(f"⏰ Time: {alert['timestamp']}")
        print(f"📊 Check Type: {alert['check_type']}")
        print(f"\nSeverity Metrics:")
        for key, val in alert['severity_metrics'].items():
            print(f"  - {key}: {val:.4f}")
        print(f"\nRecommendations:")
        for rec in alert['recommendations']:
            print(f"  {rec}")
        print(f"{'='*60}\n")
    
    
    def get_summary(self) -> Dict:
        """Get current status summary"""
        if not self.drift_history:
            return {
                "total_checks": 0,
                "total_alerts": len(self.alerts),
                "current_status": "NO_DATA"
            }
        
        last_check = self.drift_history[-1]
        latest_results = last_check.get("drift_results", {})
        
        # Calculate trends
        recent_checks = self.drift_history[-5:]  # Last 5 checks
        stability_trend = [
            c.get("drift_results", {}).get("retrieval_stability", 1)
            for c in recent_checks
        ]
        
        return {
            "total_checks": len(self.drift_history),
            "total_alerts": len(self.alerts),
            "last_check": last_check.get("timestamp"),
            "current_status": latest_results.get("drift_status", "UNKNOWN"),
            "current_metrics": {
                "query_drift": latest_results.get("query_drift", 0),
                "retrieval_stability": latest_results.get("retrieval_stability", 1),
                "avg_hallucination": latest_results.get("avg_hallucination", 0)
            },
            "stability_trend": stability_trend,
            "recent_recommendations": latest_results.get("recommendations", [])
        }
    
    
    def get_alerts_since(self, hours_ago: int = 24) -> List[Dict]:
        """Get alerts from last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_ago)
        cutoff_iso = cutoff_time.isoformat()
        
        recent_alerts = [
            alert for alert in self.alerts
            if alert.get("timestamp", "") >= cutoff_iso
        ]
        
        return recent_alerts


# ===============================
# Scheduled Monitoring (APScheduler)
# ===============================

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    print("⚠️ APScheduler not installed. For scheduled monitoring, run: pip install apscheduler")


class DriftScheduler:
    """
    Periodic drift detection scheduler
    """
    
    def __init__(self):
        if not HAS_APSCHEDULER:
            raise ImportError("APScheduler is required. Install with: pip install apscheduler")
        
        self.scheduler = BackgroundScheduler()
        self.monitor = None
        print("✅ DriftScheduler initialized")
    
    
    def setup(
        self,
        monitor: DriftDetectionMonitor,
        daily_check_hour: int = 2,
        interval_minutes: int = 60
    ):
        """
        Scheduler settings
        
        Args:
            monitor: DriftDetectionMonitor instance
            daily_check_hour: hour for daily check
            interval_minutes: periodic check every N minutes
        """
        self.monitor = monitor
        
        # Daily check
        self.scheduler.add_job(
            self._daily_check_job,
            'cron',
            hour=daily_check_hour,
            minute=0,
            id='daily_drift_check'
        )
        
        # Periodic check
        self.scheduler.add_job(
            self._periodic_check_job,
            'interval',
            minutes=interval_minutes,
            id='periodic_drift_check'
        )
        
        print(f"✅ Scheduler jobs added:")
        print(f"   - Daily check at {daily_check_hour}:00")
        print(f"   - Periodic check every {interval_minutes} minutes")
    
    
    def _daily_check_job(self):
        """Daily check function"""
        print("\n📅 Running scheduled daily drift check...")
        if self.monitor:
            metrics = list(self.monitor.metrics_buffer)
            if metrics:
                self.monitor._run_drift_check("scheduled_daily")
    
    
    def _periodic_check_job(self):
        """Periodic check function"""
        print("\n⏰ Running periodic drift check...")
        if self.monitor:
            metrics = list(self.monitor.metrics_buffer)
            if metrics:
                self.monitor._run_drift_check("scheduled_periodic")
    
    
    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            print("✅ Drift scheduler started")
    
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            print("✅ Drift scheduler stopped")


# ===============================
# Dashboard Report
# ===============================

def generate_drift_report(monitor: DriftDetectionMonitor, output_file: str = None) -> str:
    """
    Generate comprehensive drift report
    """
    summary = monitor.get_summary()
    recent_alerts = monitor.get_alerts_since(hours_ago=24)
    
    report = f"""
{'='*70}
                    DRIFT DETECTION REPORT
{'='*70}

📊 SUMMARY
---------
Total Checks: {summary.get('total_checks', 0)}
Total Alerts: {summary.get('total_alerts', 0)}
Last Check: {summary.get('last_check', 'N/A')}
Current Status: {summary.get('current_status', 'N/A')}

📈 CURRENT METRICS
------------------
Query Drift: {summary.get('current_metrics', {}).get('query_drift', 0):.4f}
Retrieval Stability: {summary.get('current_metrics', {}).get('retrieval_stability', 1):.4f}
Avg Hallucination: {summary.get('current_metrics', {}).get('avg_hallucination', 0):.4f}

📉 STABILITY TREND (Last 5 checks)
-----------------------------------
{' → '.join([f'{s:.3f}' for s in summary.get('stability_trend', [])])}

⚠️ RECENT ALERTS (Last 24h)
---------------------------
"""
    
    if recent_alerts:
        for alert in recent_alerts:
            report += f"\n⏰ {alert.get('timestamp')}\n"
            report += f"   Status: {alert.get('drift_status')}\n"
            report += f"   Type: {alert.get('check_type')}\n"
    else:
        report += "\n✅ No alerts in the last 24 hours\n"
    
    report += f"\n📋 RECOMMENDATIONS\n"
    report += f"{'='*70}\n"
    for rec in summary.get('recent_recommendations', []):
        report += f"{rec}\n"
    
    report += f"\n{'='*70}\n"
    
    # Save report if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Report saved: {output_file}")
    
    return report