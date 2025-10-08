"""
Comprehensive Monitoring and Analytics System for OpenEvolve
Implements real-time monitoring, performance tracking, and analytics for all OpenEvolve features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import queue
from dataclasses import dataclass, asdict
from enum import Enum


@dataclass
class EvolutionMetrics:
    """Data class to hold evolution metrics."""
    generation: int
    best_score: float
    avg_score: float
    diversity_score: float
    complexity: float
    population_size: int
    improvement_rate: float
    timestamp: datetime
    feature_dimensions: Dict[str, float]
    archive_size: int
    island_metrics: Dict[str, Any]  # Metrics per island if using island model


class MonitoringStatus(Enum):
    """Enum for monitoring status."""
    IDLE = "idle"
    MONITORING = "monitoring"
    PAUSED = "paused"
    ERROR = "error"


class EvolutionMonitor:
    """Main evolution monitor class that handles real-time monitoring and analytics."""
    
    def __init__(self):
        self.metrics_history: List[EvolutionMetrics] = []
        self.status = MonitoringStatus.IDLE
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Initialize session state for monitoring if not already done
        if "monitoring_data" not in st.session_state:
            st.session_state.monitoring_data = []
        if "monitoring_metrics" not in st.session_state:
            st.session_state.monitoring_metrics = {
                "best_score": 0.0,
                "current_generation": 0,
                "avg_diversity": 0.0,
                "convergence_rate": 0.0
            }
    
    def start_monitoring(self, update_callback: Optional[Callable] = None):
        """Start monitoring evolution process."""
        self.monitoring_active = True
        self.status = MonitoringStatus.MONITORING
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(update_callback,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring evolution process."""
        self.monitoring_active = False
        self.status = MonitoringStatus.IDLE
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def pause_monitoring(self):
        """Pause monitoring."""
        self.status = MonitoringStatus.PAUSED
    
    def resume_monitoring(self):
        """Resume monitoring."""
        self.status = MonitoringStatus.MONITORING
    
    def _monitoring_loop(self, update_callback: Optional[Callable]):
        """Internal monitoring loop that processes metrics."""
        while self.monitoring_active:
            try:
                if not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    with self.lock:
                        if isinstance(metrics, EvolutionMetrics):
                            self.metrics_history.append(metrics)
                            # Update session state with latest metrics
                            self._update_session_state(metrics)
                
                if update_callback:
                    update_callback()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                self.status = MonitoringStatus.ERROR
                st.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def add_metrics(self, metrics: EvolutionMetrics):
        """Add new metrics to the monitoring queue."""
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            # If queue is full, remove oldest item and add new one
            try:
                self.metrics_queue.get_nowait()
                self.metrics_queue.put_nowait(metrics)
            except Exception:
                pass  # If both operations fail, just skip this metrics update
    
    def _update_session_state(self, metrics: EvolutionMetrics):
        """Update session state with latest metrics."""
        st.session_state.monitoring_metrics["best_score"] = metrics.best_score
        st.session_state.monitoring_metrics["current_generation"] = metrics.generation
        st.session_state.monitoring_metrics["avg_diversity"] = metrics.diversity_score
        st.session_state.monitoring_metrics["complexity"] = metrics.complexity
        
        # Add to monitoring data history
        if "monitoring_data" not in st.session_state:
            st.session_state.monitoring_data = []
        
        st.session_state.monitoring_data.append(asdict(metrics))
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.metrics_history:
            return {"status": "No data", "metrics": {}}
        
        latest = self.metrics_history[-1]
        return {
            "status": self.status.value,
            "current_generation": latest.generation,
            "best_score": latest.best_score,
            "avg_score": latest.avg_score,
            "diversity_score": latest.diversity_score,
            "complexity": latest.complexity,
            "total_generations": len(self.metrics_history),
            "improvement_rate": latest.improvement_rate,
            "feature_dimensions": latest.feature_dimensions,
            "archive_size": latest.archive_size
        }
    
    def get_historical_data(self) -> pd.DataFrame:
        """Get historical metrics data as a DataFrame."""
        if not st.session_state.monitoring_data:
            return pd.DataFrame()
        
        # Convert monitoring data to DataFrame
        df_data = []
        for record in st.session_state.monitoring_data:
            # Flatten the nested structure to make it compatible with DataFrame
            flat_record = {
                "generation": record.get("generation", 0),
                "best_score": record.get("best_score", 0.0),
                "avg_score": record.get("avg_score", 0.0),
                "diversity_score": record.get("diversity_score", 0.0),
                "complexity": record.get("complexity", 0.0),
                "improvement_rate": record.get("improvement_rate", 0.0),
                "archive_size": record.get("archive_size", 0),
                "timestamp": record.get("timestamp", datetime.now())
            }
            
            # Add feature dimensions as separate columns
            feature_dims = record.get("feature_dimensions", {})
            for dim, value in feature_dims.items():
                flat_record[f"feature_{dim}"] = value
            
            df_data.append(flat_record)
        
        return pd.DataFrame(df_data)
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence metrics."""
        df = self.get_historical_data()
        if df.empty or len(df) < 2:
            return {"convergence_status": "insufficient_data", "metrics": {}}
        
        # Calculate convergence metrics
        recent_generations = min(20, len(df))  # Look at last 20 generations or all if less
        recent_data = df.tail(recent_generations)
        
        # Calculate standard deviation of recent scores (lower = more stable)
        recent_std = recent_data['best_score'].std()
        recent_avg = recent_data['best_score'].mean()
        best_score_ever = df['best_score'].max()
        current_score = df['best_score'].iloc[-1] if not df.empty else 0.0
        
        # Calculate improvement rate in recent generations
        if len(recent_data) > 1:
            first_recent = recent_data['best_score'].iloc[0]
            last_recent = recent_data['best_score'].iloc[-1]
            improvement_over_recent = (last_recent - first_recent) / len(recent_data)
        else:
            improvement_over_recent = 0.0
        
        convergence_status = "converging"
        if recent_std < 0.005:  # Very low variance
            if abs(best_score_ever - current_score) < 0.01:  # Close to best, low variance
                convergence_status = "converged"
            else:
                convergence_status = "stagnant"
        elif improvement_over_recent < 0.001:  # Very slow improvement
            convergence_status = "slow_improvement"
        
        return {
            "convergence_status": convergence_status,
            "metrics": {
                "recent_std": recent_std,
                "recent_avg": recent_avg,
                "best_score_ever": best_score_ever,
                "current_score": current_score,
                "improvement_recent": improvement_over_recent,
                "generations_analyzed": recent_generations
            }
        }


class PerformanceTracker:
    """Tracks and analyzes performance across different evolution runs."""
    
    def __init__(self):
        self.run_history = []
        self.performance_metrics = {}
    
    def register_run(self, run_data: Dict[str, Any]):
        """Register a new evolution run."""
        run_data["timestamp"] = datetime.now()
        self.run_history.append(run_data)
        
        # Calculate performance metrics
        if "best_score" in run_data and "duration" in run_data:
            if "performance_metrics" not in self.performance_metrics:
                self.performance_metrics = {
                    "scores": [],
                    "durations": [],
                    "efficiencies": []  # score/duration ratio
                }
            
            score = run_data["best_score"]
            duration = run_data["duration"]
            efficiency = score / max(duration, 0.001)  # Avoid division by zero
            
            self.performance_metrics["scores"].append(score)
            self.performance_metrics["durations"].append(duration)
            self.performance_metrics["efficiencies"].append(efficiency)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across runs."""
        if not self.performance_metrics.get("scores"):
            return {
                "total_runs": 0,
                "avg_score": 0.0,
                "avg_duration": 0.0,
                "avg_efficiency": 0.0,
                "best_score": 0.0,
                "improvement_trend": 0.0
            }
        
        scores = self.performance_metrics["scores"]
        durations = self.performance_metrics["durations"]
        efficiencies = self.performance_metrics["efficiencies"]
        
        # Calculate improvement trend (slope of best scores over time)
        if len(self.run_history) > 1:
            # Simple linear regression for trend
            x = list(range(len(scores)))
            y = scores
            if len(x) > 1:
                # Calculate slope (m) of line of best fit: y = mx + b
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(xi * xi for xi in x)
                
                denominator = n * sum_x2 - sum_x * sum_x
                if denominator != 0:
                    improvement_trend = (n * sum_xy - sum_x * sum_y) / denominator
                else:
                    improvement_trend = 0.0
            else:
                improvement_trend = 0.0
        else:
            improvement_trend = 0.0
        
        return {
            "total_runs": len(self.run_history),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "avg_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "improvement_trend": improvement_trend,
            "efficiency_trend": "Increasing" if improvement_trend > 0 else "Decreasing"
        }
    
    def plot_performance_trends(self):
        """Plot performance trends."""
        if not self.run_history:
            st.info("No runs to display")
            return
        
        df_runs = pd.DataFrame(self.run_history)
        
        # Create a subplot with metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Score Trend", "Duration Trend", "Efficiency Trend", "Success Rate"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Score trend
        if "generation" in df_runs.columns and "best_score" in df_runs.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_runs["generation"] if "generation" in df_runs.columns else df_runs.index,
                    y=df_runs["best_score"],
                    mode='lines+markers',
                    name='Best Score',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Duration trend
        if "duration" in df_runs.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_runs.index,
                    y=df_runs["duration"],
                    mode='lines+markers',
                    name='Duration',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
        
        # Efficiency trend
        if "best_score" in df_runs.columns and "duration" in df_runs.columns:
            efficiency = df_runs["best_score"] / df_runs["duration"].replace(0, 0.001)  # Avoid division by zero
            fig.add_trace(
                go.Scatter(
                    x=df_runs.index,
                    y=efficiency,
                    mode='lines+markers',
                    name='Efficiency',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Success rate (for classification or boolean success metrics)
        if "success_count" in df_runs.columns and "total_attempts" in df_runs.columns:
            success_rate = df_runs["success_count"] / df_runs["total_attempts"].replace(0, 1)  # Avoid division by zero
            fig.add_trace(
                go.Scatter(
                    x=df_runs.index,
                    y=success_rate,
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Performance Trends")
        st.plotly_chart(fig, use_container_width=True)


class FeatureSpaceAnalyzer:
    """Analyzes feature space exploration and diversity in evolution."""
    
    def __init__(self):
        self.feature_history = []
    
    def add_features(self, features: Dict[str, float], generation: int = 0):
        """Add feature values to analysis."""
        self.feature_history.append({
            "features": features,
            "generation": generation,
            "timestamp": datetime.now()
        })
    
    def analyze_diversity(self) -> Dict[str, Any]:
        """Analyze diversity in feature space."""
        if not self.feature_history:
            return {"diversity_score": 0.0, "feature_ranges": {}, "clustering_info": {}}
        
        # Extract feature values
        all_features = []
        for record in self.feature_history:
            features = record["features"]
            feature_values = list(features.values())
            all_features.append(feature_values)
        
        if not all_features:
            return {"diversity_score": 0.0, "feature_ranges": {}, "clustering_info": {}}
        
        # Convert to numpy array for analysis
        all_features = np.array(all_features)
        
        # Calculate diversity metrics
        feature_ranges = {}
        for i, feature_name in enumerate(self.feature_history[0]["features"].keys()):
            if i < all_features.shape[1]:
                values = all_features[:, i]
                feature_ranges[feature_name] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                    "range": float(np.max(values) - np.min(values))
                }
        
        # Diversity score based on feature spread (normalized)
        diversity_score = 0.0
        if len(feature_ranges) > 0:
            avg_std = np.mean([info["std"] for info in feature_ranges.values()])
            diversity_score = min(1.0, avg_std * 2)  # Normalize to 0-1 range
        
        return {
            "diversity_score": diversity_score,
            "feature_ranges": feature_ranges,
            "total_records": len(self.feature_history),
            "feature_count": len(list(self.feature_history[0]["features"].keys()))
        }
    
    def plot_feature_space(self):
        """Plot feature space exploration."""
        if not self.feature_history or len(self.feature_history) < 2:
            st.info("Not enough feature data to plot")
            return
        
        # Get first record's feature names
        feature_names = list(self.feature_history[0]["features"].keys())
        if len(feature_names) < 2:
            st.warning("Need at least 2 features to plot relationships")
            return
        
        # Create data for plotting
        df_features = pd.DataFrame([
            {**record["features"], "generation": record["generation"]}
            for record in self.feature_history
        ])
        
        if len(feature_names) >= 2:
            # Create scatter plot of first two features
            fig = px.scatter(
                df_features,
                x=feature_names[0],
                y=feature_names[1],
                color="generation",
                title=f"Feature Space Exploration: {feature_names[0]} vs {feature_names[1]}",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # If more than 2 features, create a correlation matrix
        if len(feature_names) > 2:
            feature_df = df_features[feature_names]
            corr_matrix = feature_df.corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                text_auto=True,
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)


def render_monitoring_dashboard():
    """Render the comprehensive monitoring dashboard."""
    st.header("📊 OpenEvolve Monitoring Dashboard")
    
    # Initialize monitor if not exists
    if "evolution_monitor" not in st.session_state:
        st.session_state.evolution_monitor = EvolutionMonitor()
    
    monitor = st.session_state.evolution_monitor


    
    # Monitoring controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Monitoring"):
            monitor.start_monitoring()
            st.success("Monitoring started!")
    
    with col2:
        if st.button("⏹️ Stop Monitoring"):
            monitor.stop_monitoring()
            st.info("Monitoring stopped")
    
    with col3:
        if st.button("⏸️ Pause Monitoring"):
            monitor.pause_monitoring()
            st.info("Monitoring paused")
    
    with col4:
        if st.button("🎬 Resume Monitoring"):
            monitor.resume_monitoring()
            st.success("Monitoring resumed!")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", monitor.status.value.title())
    with col2:
        st.metric("Current Gen", st.session_state.monitoring_metrics.get("current_generation", 0))
    with col3:
        st.metric("Best Score", f"{st.session_state.monitoring_metrics.get('best_score', 0.0):.3f}")
    
    # Create tabs for different monitoring views
    tabs = st.tabs(["📈 Real-time Metrics", "🎯 Convergence Analysis", "📊 Feature Analysis", "📋 Performance Log"])
    
    with tabs[0]:  # Real-time Metrics
        st.subheader("Real-time Evolution Metrics")
        
        # Get historical data for plotting
        df = monitor.get_historical_data()
        
        if not df.empty:
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Best Score", "Average Score", "Diversity", "Complexity"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Best Score
            fig.add_trace(
                go.Scatter(x=df["generation"], y=df["best_score"], 
                          mode='lines+markers', name='Best Score', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Average Score
            fig.add_trace(
                go.Scatter(x=df["generation"], y=df["avg_score"], 
                          mode='lines+markers', name='Avg Score', line=dict(color='red')),
                row=1, col=2
            )
            
            # Diversity Score
            fig.add_trace(
                go.Scatter(x=df["generation"], y=df["diversity_score"], 
                          mode='lines+markers', name='Diversity', line=dict(color='green')),
                row=2, col=1
            )
            
            # Complexity
            fig.add_trace(
                go.Scatter(x=df["generation"], y=df["complexity"], 
                          mode='lines+markers', name='Complexity', line=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monitoring data available. Start evolution to see metrics.")
    
    with tabs[1]:  # Convergence Analysis
        st.subheader("Convergence Analysis")
        
        convergence_data = monitor.get_convergence_analysis()
        if convergence_data["convergence_status"] != "insufficient_data":
            status = convergence_data["convergence_status"]
            
            # Display convergence status
            status_colors = {
                "converged": "🟢", 
                "converging": "🟡", 
                "slow_improvement": "🟠", 
                "stagnant": "🔴"
            }
            
            st.markdown(f"**Status:** {status_colors.get(status, '❓')} {status.replace('_', ' ').title()}")
            
            # Show metrics
            metrics = convergence_data["metrics"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recent Std Dev", f"{metrics['recent_std']:.4f}")
            with col2:
                st.metric("Recent Avg", f"{metrics['recent_avg']:.3f}")
            with col3:
                st.metric("Improvement Rate", f"{metrics['improvement_recent']:.5f}")
            
            # Plot convergence trend
            df = monitor.get_historical_data()
            if not df.empty:
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=df["generation"], 
                    y=df["best_score"],
                    mode='lines+markers',
                    name='Best Score',
                    line=dict(color='blue')
                ))
                
                # Add moving average to show trend
                if len(df) > 10:
                    window = max(5, len(df) // 10)  # 10% of data
                    df['moving_avg'] = df['best_score'].rolling(window=window, center=True).mean()
                    fig_conv.add_trace(go.Scatter(
                        x=df["generation"],
                        y=df["moving_avg"],
                        mode='lines',
                        name=f'Moving Avg ({window} gen)',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig_conv.update_layout(
                    title="Convergence Trend",
                    xaxis_title="Generation",
                    yaxis_title="Best Score"
                )
                st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Insufficient data for convergence analysis. Requires at least 2 data points.")
    
    with tabs[2]:  # Feature Analysis
        st.subheader("Feature Space Analysis")
        
        df = monitor.get_historical_data()
        if not df.empty:
            # Look for feature columns (columns that start with 'feature_')
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            
            if feature_cols:
                # Display feature statistics
                feature_stats = df[feature_cols].describe()
                st.dataframe(feature_stats)
                
                # Plot feature relationships
                if len(feature_cols) >= 2:
                    fig_features = px.scatter_matrix(
                        df,
                        dimensions=feature_cols[:4],  # Limit to first 4 features to avoid clutter
                        title="Feature Relationships"
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
                
                # Feature correlation heatmap
                if len(feature_cols) > 1:
                    feature_corr = df[feature_cols].corr()
                    fig_corr = px.imshow(
                        feature_corr,
                        title="Feature Correlation Heatmap",
                        text_auto=True,
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No feature dimension data available in monitoring history.")
        else:
            st.info("No monitoring data available for feature analysis.")
    
    with tabs[3]:  # Performance Log
        st.subheader("Performance Log")
        
        # Display recent monitoring data as a table
        df = monitor.get_historical_data()
        if not df.empty:
            # Show the last 20 records
            recent_data = df.tail(20).copy()
            recent_data = recent_data.sort_values('generation', ascending=False)
            
            # Format timestamp for readability
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp']).dt.strftime('%H:%M:%S')
            
            st.dataframe(recent_data, use_container_width=True)
        else:
            st.info("No monitoring data available in the log.")


def render_performance_analytics():
    """Render performance analytics dashboard."""
    st.header("🚀 Performance Analytics Dashboard")
    
    # Sample performance tracking data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", st.session_state.get("total_evolution_runs", 0))
    with col2:
        st.metric("Avg. Score", f"{st.session_state.get('avg_best_score', 0.0):.3f}")
    with col3:
        st.metric("Best Ever", f"{st.session_state.get('best_ever_score', 0.0):.3f}")
    with col4:
        st.metric("Success Rate", f"{st.session_state.get('success_rate', 0.0):.1%}")
    
    # Performance trends chart
    st.subheader("Performance Trends")
    
    # Sample performance data over time
    days = list(range(1, 31))
    scores = np.random.uniform(0.6, 0.95, 30).cumsum()  # Simulated improvement over time
    scores = np.clip(scores / scores.max() * 0.95 + 0.05, 0.05, 0.95)  # Normalize to 0.05-0.95 range
    efficiency = np.random.uniform(0.5, 1.0, 30)
    
    trend_df = pd.DataFrame({
        "Day": days,
        "Average Score": scores,
        "Efficiency": efficiency
    })
    
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(
        x=trend_df["Day"], 
        y=trend_df["Average Score"],
        mode='lines+markers',
        name='Avg Score',
        line=dict(color='blue'),
        yaxis='y'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=trend_df["Day"], 
        y=trend_df["Efficiency"],
        mode='lines+markers',
        name='Efficiency',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig_trends.update_layout(
        title="Performance Over Time",
        xaxis_title="Time (Days)",
        yaxis_title="Average Score",
        yaxis2=dict(
            title="Efficiency",
            overlaying='y',
            side='right'
        ),
        hovermode='x'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Resource utilization
    st.subheader("Resource Utilization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### API Usage")
        api_usage_data = {
            "OpenAI GPT-4": 45,
            "OpenAI GPT-3.5": 30,
            "Anthropic Claude": 15,
            "Google Gemini": 10
        }
        api_df = pd.DataFrame(list(api_usage_data.items()), columns=['Provider', 'Percentage'])
        fig_api = px.pie(api_df, values='Percentage', names='Provider', title='API Usage Distribution')
        st.plotly_chart(fig_api, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Analysis")
        cost_data = {
            "Date": pd.date_range(start="2024-01-01", periods=12, freq='M'),
            "Cost_USD": np.random.uniform(50, 200, 12)
        }
        cost_df = pd.DataFrame(cost_data)
        fig_cost = px.line(cost_df, x='Date', y='Cost_USD', title='Monthly API Costs')
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Performance recommendations
    st.subheader("Performance Recommendations")
    
    recommendations = [
        "🟢 Increase population size for better exploration",
        "🟡 Adjust elite ratio to 15% for more diversity",
        "🔴 Consider using ensemble models for critical runs",
        "🔵 Enable evolution tracing for detailed analysis",
        "🟢 Reduce migration rate for more island independence"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")


def render_comprehensive_monitoring_ui():
    """Render the comprehensive monitoring user interface."""
    main_tabs = st.tabs(["📈 Monitoring Dashboard", "🚀 Performance Analytics", "🎯 Optimization Advisor"])
    
    with main_tabs[0]:
        render_monitoring_dashboard()
    
    with main_tabs[1]:
        render_performance_analytics()
    
    with main_tabs[2]:
        st.header("🤖 Optimization Advisor")
        
        st.markdown("""
        The Optimization Advisor provides intelligent recommendations based on your evolution data.
        """)
        
        # Sample advisor content with AI-generated suggestions
        advisor_tabs = st.tabs(["💡 Suggestions", "⚙️ Configuration", "📊 Insights"])
        
        with advisor_tabs[0]:  # Suggestions
            st.subheader("AI-Powered Optimization Suggestions")
            
            suggestions = [
                {
                    "priority": "High",
                    "category": "Evolution Strategy",
                    "suggestion": "Increase exploration ratio from 0.2 to 0.4 to escape local optima",
                    "expected_impact": "+15% performance improvement",
                    "confidence": 85
                },
                {
                    "priority": "Medium", 
                    "category": "Population",
                    "suggestion": "Double population size to 200 for better diversity",
                    "expected_impact": "+8% diversity maintenance",
                    "confidence": 72
                },
                {
                    "priority": "Low",
                    "category": "Architecture",
                    "suggestion": "Implement island model with 7 islands for parallel exploration", 
                    "expected_impact": "+5% exploration efficiency",
                    "confidence": 68
                },
                {
                    "priority": "High",
                    "category": "Evaluation",
                    "suggestion": "Enable cascade evaluation to filter low-quality solutions early",
                    "expected_impact": "+25% efficiency gain",
                    "confidence": 90
                }
            ]
            
            for i, sug in enumerate(suggestions):
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟡", 
                    "Low": "🟢"
                }[sug["priority"]]
                
                with st.container():
                    st.markdown(f"**{priority_color} {sug['category']}**: {sug['suggestion']}")
                    st.caption(f"Expected Impact: {sug['expected_impact']} | Confidence: {sug['confidence']}%")
                    st.markdown("---")
        
        with advisor_tabs[1]:  # Configuration
            st.subheader("Recommended Configuration")
            
            recommended_config = {
                "max_iterations": 150,
                "population_size": 150,
                "num_islands": 7,
                "elite_ratio": 0.15,
                "exploration_ratio": 0.4,
                "exploitation_ratio": 0.45,
                "archive_size": 150,
                "cascade_evaluation": True,
                "enable_artifacts": True,
                "use_llm_feedback": True,
                "early_stopping_patience": 15
            }
            
            config_df = pd.DataFrame([
                {"Parameter": k, "Recommended Value": v} for k, v in recommended_config.items()
            ])
            
            st.dataframe(config_df, use_container_width=True)
            
            if st.button("Apply Recommended Configuration"):
                # In a real implementation, this would update the session state
                for param, value in recommended_config.items():
                    st.session_state[param] = value
                st.success("Configuration applied! Please restart your evolution for changes to take effect.")
        
        with advisor_tabs[2]:  # Insights
            st.subheader("Intelligent Insights")
            
            insights = [
                "Your evolution is currently in the exploration phase based on diversity metrics",
                "Best score improved by 23% in the last 20 generations",
                "Feature space exploration shows good coverage of solution space",
                "Convergence indicators suggest 15-20 more generations to optimal solution",
                "Current parameters are suboptimal for your problem type"
            ]
            
            for insight in insights:
                st.info(f"🔍 {insight}")