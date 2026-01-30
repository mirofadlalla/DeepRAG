"""
Drift Detection Dashboard
Display current status and alerts
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional
from drift_detection import DriftDetectionMonitor


def display_dashboard(monitor: DriftDetectionMonitor, refresh_interval: int = 60):
    """
    Display live dashboard for drift detection
    
    Args:
        monitor: DriftDetectionMonitor instance
        refresh_interval: Refresh every N seconds
    """
    try:
        import time
        import os
        
        while True:
            # Clear the screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Get current data
            summary = monitor.get_summary()
            recent_alerts = monitor.get_alerts_since(hours_ago=24)
            
            # Display the dashboard
            print("\n" + "="*80)
            print(" "*20 + "🔍 DRIFT DETECTION DASHBOARD")
            print("="*80)
            
            # Status
            status = summary.get('current_status', 'UNKNOWN')
            status_emoji = {
                'STABLE': '✅',
                'WARNING': '⚠️',
                'CRITICAL': '🚨',
                'NO_DATA': '❓'
            }.get(status, '❓')
            
            print(f"\n{status_emoji} Current Status: {status}")
            print(f"⏰ Last Update: {summary.get('last_check', 'N/A')}")
            
            # Metrics
            print("\n📊 Current Metrics:")
            metrics = summary.get('current_metrics', {})
            print(f"   Query Drift: {metrics.get('query_drift', 0):.4f}")
            print(f"   Retrieval Stability: {metrics.get('retrieval_stability', 1):.4f}")
            print(f"   Avg Hallucination: {metrics.get('avg_hallucination', 0):.4f}")
            
            # Trend
            trend = summary.get('stability_trend', [])
            if trend:
                print(f"\n📈 Stability Trend (Last 5 checks):")
                print(f"   {' → '.join([f'{s:.3f}' for s in trend])}")
            
            # Statistics
            print(f"\n📈 Statistics:")
            print(f"   Total Checks: {summary.get('total_checks', 0)}")
            print(f"   Total Alerts: {summary.get('total_alerts', 0)}")
            
            # Recent Alerts
            print(f"\n⚠️ Recent Alerts (Last 24h): {len(recent_alerts)}")
            if recent_alerts:
                for i, alert in enumerate(recent_alerts[-5:], 1):  # Last 5 alerts
                    print(f"   {i}. {alert.get('timestamp')} - {alert.get('drift_status')}")
            else:
                print("   ✅ No alerts")
            
            # Recommendations
            print(f"\n💡 Recommendations:")
            recommendations = summary.get('recent_recommendations', [])
            if recommendations:
                for rec in recommendations:
                    print(f"   {rec}")
            else:
                print("   ✅ No specific recommendations")
            
            print("\n" + "="*80)
            print(f"🔄 Refreshing in {refresh_interval}s... (Press Ctrl+C to stop)")
            print("="*80 + "\n")
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")


def export_dashboard_to_html(
    monitor: DriftDetectionMonitor,
    output_file: str = "benchmark_reports/drift_dashboard.html"
) -> str:
    """
    صدّر dashboard كـ HTML
    """
    summary = monitor.get_summary()
    recent_alerts = monitor.get_alerts_since(hours_ago=24)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Drift Detection Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            }}
            
            .card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            
            .metric {{
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            
            .metric-label {{
                font-weight: 500;
            }}
            
            .metric-value {{
                color: #667eea;
                font-weight: 700;
            }}
            
            .status {{
                text-align: center;
                font-size: 1.5em;
                margin: 20px 0;
            }}
            
            .status.stable {{
                color: #28a745;
            }}
            
            .status.warning {{
                color: #ffc107;
            }}
            
            .status.critical {{
                color: #dc3545;
            }}
            
            .alerts {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            
            .alert-item {{
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 10px 0;
                background: #fff5f5;
                border-radius: 5px;
            }}
            
            .alert-item.warning {{
                border-left-color: #ffc107;
                background: #fffbf0;
            }}
            
            .recommendation {{
                margin: 10px 0;
                padding: 10px;
                background: #e7f3ff;
                border-left: 4px solid #667eea;
                border-radius: 5px;
            }}
            
            .footer {{
                text-align: center;
                color: white;
                margin-top: 30px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Drift Detection Dashboard</h1>
                <p>Real-time monitoring of RAG system drift</p>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 Current Status</h3>
                    <div class="status {summary.get('current_status', 'UNKNOWN').lower()}">
                        {summary.get('current_status', 'UNKNOWN')}
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Check:</span>
                        <span class="metric-value">{summary.get('last_check', 'N/A')}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📈 Key Metrics</h3>
                    <div class="metric">
                        <span class="metric-label">Query Drift:</span>
                        <span class="metric-value">{summary.get('current_metrics', {}).get('query_drift', 0):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Retrieval Stability:</span>
                        <span class="metric-value">{summary.get('current_metrics', {}).get('retrieval_stability', 1):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Hallucination:</span>
                        <span class="metric-value">{summary.get('current_metrics', {}).get('avg_hallucination', 0):.4f}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 Statistics</h3>
                    <div class="metric">
                        <span class="metric-label">Total Checks:</span>
                        <span class="metric-value">{summary.get('total_checks', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Alerts:</span>
                        <span class="metric-value">{summary.get('total_alerts', 0)}</span>
                    </div>
                </div>
            </div>
            
            <div class="alerts">
                <h3>⚠️ Recent Alerts (Last 24h)</h3>
    """
    
    if recent_alerts:
        for alert in recent_alerts:
            alert_class = "warning" if alert.get('drift_status') == 'WARNING' else ""
            html += f"""
                <div class="alert-item {alert_class}">
                    <strong>{alert.get('drift_status')}</strong> - {alert.get('timestamp')}
                    <br><small>{alert.get('check_type')}</small>
                </div>
            """
    else:
        html += "<p>✅ No alerts in the last 24 hours</p>"
    
    html += """
            </div>
            
            <div style="margin-top: 30px;">
                <div class="card">
                    <h3>💡 Recommendations</h3>
    """
    
    for rec in summary.get('recent_recommendations', []):
        html += f'<div class="recommendation">{rec}</div>'
    
    html += """
                </div>
            </div>
            
            <div class="footer">
                <p>🔄 Refreshing every 60 seconds | Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the file
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Dashboard exported to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Usage example
    print("⏳ Loading monitor data...")
    monitor = DriftDetectionMonitor()
    
    print("\n📊 Starting live dashboard...")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        display_dashboard(monitor, refresh_interval=10)
    except KeyboardInterrupt:
        print("\n✅ Exporting HTML dashboard...")
        export_dashboard_to_html(monitor)
        print("✅ Done!")
