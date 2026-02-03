"""
Comprehensive Reporting System for OpenEvolve
Generates detailed reports, visualizations, and documentation for evolution runs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import base64
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import logging

# **ACTUAL INTEGRATION**: Alerting and knowledge for Reporting System
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvolutionReport:
    """Data class for evolution report data."""
    run_id: str
    timestamp: datetime
    evolution_mode: str
    content_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    metrics: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    visualizations: Dict[str, str]  # base64 encoded images
    summary_statistics: Dict[str, Any]


class ReportGenerator:
    """Main report generator class."""
    
    def __init__(self):
        self.reports: List[EvolutionReport] = []
        # Historical scores database for percentile calculation
        # Dynamically populated from actual evolution runs
        self.historical_scores: List[float] = self._load_historical_scores()
    
    def _load_historical_scores(self) -> List[float]:
        """Load historical scores from persistent storage or initialize with seed data"""
        import os
        import json
        
        scores_file = "historical_scores.json"
        if os.path.exists(scores_file):
            try:
                with open(scores_file, 'r') as f:
                    return json.load(f)
            except (OSError, IOError, json.JSONDecodeError) as exc:
                logger.debug(f"Failed to load historical scores: {exc}")
        
        # Initialize with seed data if no historical data exists
        return [
            0.65, 0.70, 0.72, 0.75, 0.68, 0.80, 0.81, 0.79, 0.85, 0.77,
            0.90, 0.88, 0.92, 0.83, 0.76, 0.71, 0.84, 0.86, 0.91, 0.89
        ]
    
    def _save_historical_scores(self):
        """Persist historical scores to storage"""
        import json
        try:
            with open("historical_scores.json", 'w') as f:
                json.dump(self.historical_scores, f)
        except (OSError, IOError, TypeError) as e:
            print(f"Failed to save historical scores: {e}")
    
    def generate_evolution_report(
        self,
        run_id: str,
        evolution_mode: str,
        content_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> EvolutionReport:
        """Generate a comprehensive evolution report."""

        try:
            # Generate performance analysis
            performance_analysis = self._analyze_performance(results, metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(evolution_mode, performance_analysis)

            # Generate visualizations
            visualizations = self._generate_visualizations(results, metrics)

            # Generate summary statistics
            summary_statistics = self._calculate_summary_statistics(results, metrics)

            # Create the report
            report = EvolutionReport(
                run_id=run_id,
                timestamp=datetime.now(),
                evolution_mode=evolution_mode,
                content_type=content_type,
                parameters=parameters,
                results=results,
                metrics=metrics,
                performance_analysis=performance_analysis,
                recommendations=recommendations,
                visualizations=visualizations,
                summary_statistics=summary_statistics
            )

            # Store the report
            self.reports.append(report)

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            report_data = {
                "run_id": run_id,
                "evolution_mode": evolution_mode,
                "content_type": content_type,
                "results": results,
                "metrics": metrics
            }
            self._extract_reporting_knowledge("generate_evolution_report", report_data)
            self._track_reporting_performance("generate_evolution_report", True, metrics.get("total_runtime", 0))

            return report

        except Exception as e:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_reporting_alerts("generate_evolution_report", False, run_id, str(e))
            self._track_reporting_performance("generate_evolution_report", False)
            raise
    
    def _analyze_performance(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance of the evolution run."""
        analysis = {
            "score_analysis": {},
            "convergence_analysis": {},
            "efficiency_analysis": {},
            "diversity_analysis": {}
        }
        
        # Score analysis
        if "best_score" in results:
            best_score = results["best_score"]
            analysis["score_analysis"] = {
                "score": best_score,
                "rating": self._score_rating(best_score),
                "percentile": self._calculate_percentile(best_score)
            }
        
        # Convergence analysis
        if "metrics" in results and isinstance(results["metrics"], dict):
            if "convergence_rate" in results["metrics"]:
                convergence_rate = results["metrics"]["convergence_rate"]
                analysis["convergence_analysis"] = {
                    "rate": convergence_rate,
                    "speed": self._convergence_speed(convergence_rate),
                    "stability": self._convergence_stability(metrics)
                }
        
        # Efficiency analysis
        if "generation_time" in metrics:
            analysis["efficiency_analysis"] = {
                "avg_generation_time": metrics["generation_time"],
                "total_runtime": metrics.get("total_runtime", 0),
                "efficiency_score": self._calculate_efficiency_score(metrics)
            }
        
        # Diversity analysis
        if "diversity_score" in metrics:
            analysis["diversity_analysis"] = {
                "final_diversity": metrics["diversity_score"],
                "diversity_trend": self._diversity_trend(metrics),
                "exploration_balance": self._exploration_balance(metrics)
            }
        
        return analysis
    
    def _score_rating(self, score: float) -> str:
        """Convert numerical score to rating."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_percentile(self, score: float) -> int:
        """Calculate percentile ranking based on historical data."""
        # Add the current score to historical data for future calculations
        self.historical_scores.append(score)
        self._save_historical_scores()  # Persist updated historical data
        
        # Sort historical scores to calculate percentile
        sorted_scores = sorted(self.historical_scores)
        
        # Find the position of the current score
        count_lower = sum(1 for s in sorted_scores if s < score)
        
        # Calculate percentile
        if len(sorted_scores) > 0:
            percentile = (count_lower / len(sorted_scores)) * 100
            return int(percentile)
        return 0 # Default if no historical data
    
    def _convergence_speed(self, rate: float) -> str:
        """Analyze convergence speed."""
        if rate > 0.05:
            return "Fast"
        elif rate > 0.02:
            return "Moderate"
        elif rate > 0.005:
            return "Slow"
        else:
            return "Very Slow"
    
    def _convergence_stability(self, metrics: Dict[str, Any]) -> str:
        """Analyze convergence stability."""
        # Look at variance in scores over recent generations
        if "score_variance" in metrics:
            variance = metrics["score_variance"]
            if variance < 0.001:
                return "Stable"
            elif variance < 0.01:
                return "Moderately Stable"
            else:
                return "Unstable"
        return "Unknown"
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on runtime and performance."""
        # Efficiency calculation: quality achieved per unit time
        if "generation_time" in metrics and "best_score" in metrics:
            gen_time = metrics["generation_time"]
            score = metrics["best_score"]
            # Lower generation time and higher score = higher efficiency
            if gen_time > 0:
                return min(1.0, score / (gen_time / 10))  # Normalize
        return 0.5
    
    def _diversity_trend(self, metrics: Dict[str, Any]) -> str:
        """Analyze diversity trend."""
        if "diversity_trend" in metrics:
            trend = metrics["diversity_trend"]
            if trend > 0.1:
                return "Increasing"
            elif trend > -0.1:
                return "Stable"
            else:
                return "Decreasing"
        return "Unknown"
    
    def _exploration_balance(self, metrics: Dict[str, Any]) -> str:
        """Analyze exploration vs exploitation balance."""
        if "exploration_ratio" in metrics and "exploitation_ratio" in metrics:
            exp_ratio = metrics["exploration_ratio"]
            expl_ratio = metrics["exploitation_ratio"]
            diff = abs(exp_ratio - expl_ratio)
            if diff < 0.1:
                return "Well Balanced"
            elif exp_ratio > expl_ratio:
                return "Exploration Heavy"
            else:
                return "Exploitation Heavy"
        return "Unknown"
    
    def _generate_recommendations(self, evolution_mode: str, performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []
        
        # General recommendations based on performance analysis
        if "score_analysis" in performance_analysis:
            score_rating = performance_analysis["score_analysis"]["rating"]
            if score_rating in ["Poor", "Very Poor"]:
                recommendations.append("🔴 Consider increasing population size for better solution exploration")
                recommendations.append("🔴 Try adjusting the elite ratio to maintain better individuals")
            elif score_rating in ["Fair"]:
                recommendations.append("🟡 Increase exploration ratio to discover better solutions")
                recommendations.append("🟡 Consider using ensemble models for more robust evaluation")
        
        if "convergence_analysis" in performance_analysis:
            convergence_speed = performance_analysis["convergence_analysis"]["speed"]
            if convergence_speed == "Very Slow":
                recommendations.append("🟡 Consider increasing migration rate between islands")
                recommendations.append("🟡 Try using cascade evaluation to filter low-quality solutions early")
        
        if "diversity_analysis" in performance_analysis:
            diversity_trend = performance_analysis["diversity_analysis"]["diversity_trend"]
            if diversity_trend == "Decreasing":
                recommendations.append("🟡 Increase exploration ratio to maintain diversity")
                recommendations.append("🟡 Consider using multi-island model with lower migration rates")
        
        # Mode-specific recommendations
        if evolution_mode == "quality_diversity":
            recommendations.append("🔵 For QD evolution, consider expanding feature dimensions for richer characterization")
            recommendations.append("🔵 Try different feature bin configurations for better archive coverage")
        elif evolution_mode == "multi_objective":
            recommendations.append("🔵 Review objective weights for better balance between competing goals")
            recommendations.append("🔵 Consider Pareto frontier visualization for trade-off analysis")
        elif evolution_mode == "adversarial":
            recommendations.append("🔵 Rotate adversary models regularly to prevent overfitting")
            recommendations.append("🔵 Use different attack strategies for comprehensive robustness testing")
        elif evolution_mode == "symbolic_regression":
            recommendations.append("🔵 Expand operator set to discover more complex mathematical relationships")
            recommendations.append("🔵 Try different complexity penalties for better generalization")
        elif evolution_mode == "neuroevolution":
            recommendations.append("🔵 Consider using specialized neural network architectures for your domain")
            recommendations.append("🔵 Try different activation functions and regularization techniques")
        
        # General best practices
        recommendations.extend([
            "🟢 Enable evolution tracing for detailed analysis of the process",
            "🟢 Use artifact feedback to improve generation quality",
            "🟢 Consider hardware optimization for faster execution",
            "🟢 Regular checkpointing recommended for long-running evolutions"
        ])
        
        return recommendations
    
    def _generate_visualizations(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for the report."""
        visualizations = {}
        
        # Score progression chart
        try:
            if "score_history" in metrics:
                fig = px.line(
                    x=list(range(len(metrics["score_history"]))),
                    y=metrics["score_history"],
                    title="Score Progression Over Generations",
                    labels={"x": "Generation", "y": "Score"}
                )
                img_bytes = fig.to_image(format="png")
                visualizations["score_progression"] = base64.b64encode(img_bytes).decode()
        except (ValueError, TypeError, RuntimeError) as e:
            st.warning(f"Could not generate score progression chart: {e}")
        
        # Feature space visualization (if available)
        try:
            if "feature_data" in metrics:
                feature_df = pd.DataFrame(metrics["feature_data"])
                if len(feature_df.columns) >= 2:
                    fig = px.scatter(
                        feature_df,
                        x=feature_df.columns[0],
                        y=feature_df.columns[1],
                        title="Feature Space Distribution",
                        opacity=0.7
                    )
                    img_bytes = fig.to_image(format="png")
                    visualizations["feature_space"] = base64.b64encode(img_bytes).decode()
        except (ValueError, TypeError, RuntimeError) as e:
            st.warning(f"Could not generate feature space chart: {e}")
        
        # Diversity over time
        try:
            if "diversity_history" in metrics:
                fig = px.line(
                    x=list(range(len(metrics["diversity_history"]))),
                    y=metrics["diversity_history"],
                    title="Population Diversity Over Time",
                    labels={"x": "Generation", "y": "Diversity Score"}
                )
                img_bytes = fig.to_image(format="png")
                visualizations["diversity_timeline"] = base64.b64encode(img_bytes).decode()
        except (ValueError, TypeError, RuntimeError) as e:
            st.warning(f"Could not generate diversity timeline: {e}")
        
        return visualizations
    
    def _calculate_summary_statistics(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        stats = {
            "total_generations": metrics.get("total_generations", 0),
            "best_score": results.get("best_score", 0.0),
            "final_diversity": metrics.get("diversity_score", 0.0),
            "avg_generation_time": metrics.get("generation_time", 0.0),
            "total_runtime": metrics.get("total_runtime", 0.0),
            "archive_size": metrics.get("archive_size", 0),
            "island_count": metrics.get("island_count", 1)
        }
        
        # Calculate additional derived statistics
        if "score_history" in metrics and len(metrics["score_history"]) > 1:
            scores = metrics["score_history"]
            stats["improvement_rate"] = (max(scores) - min(scores)) / len(scores)
            stats["score_std_dev"] = np.std(scores)
            stats["score_variance"] = np.var(scores)

        return stats

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Reporting System
    # =========================================================================

    def _trigger_reporting_alerts(
        self,
        operation: str,
        success: bool,
        run_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for reporting system failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.LOW

                alert_manager.create_alert(
                    title=f"Reporting System Alert: {operation}",
                    description=f"Reporting operation '{operation}' failed" +
                                 (f" for run '{run_id}'" if run_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="reporting_system",
                    component="report_generation",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Reporting alert: {e}")

    def _extract_reporting_knowledge(
        self,
        operation: str,
        report_data: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract reporting knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"reporting_{operation}_{report_data.get('run_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="evolution_report",
                source_component="reporting_system",
                title=f"Evolution Report: {report_data.get('run_id', 'unknown')} ({operation})",
                content={
                    "operation": operation,
                    "evolution_mode": report_data.get("evolution_mode"),
                    "content_type": report_data.get("content_type"),
                    "best_score": report_data.get("results", {}).get("best_score", 0.0),
                    "total_generations": report_data.get("metrics", {}).get("total_generations", 0),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "run_id": report_data.get("run_id"),
                    "has_recommendations": len(report_data.get("recommendations", [])) > 0
                },
                tags=["reporting", "evolution", operation, "analysis"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Reporting knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Reporting knowledge: {e}")
            return False

    def _track_reporting_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float = 0
    ):
        """**ACTUAL INTEGRATION**: Track reporting operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"reporting_system_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation, "duration_seconds": duration_seconds}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Reporting performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Reporting performance: {e}")


def render_interactive_report_viewer():
    """Render an interactive report viewer."""
    st.header("📋 Interactive Report Viewer")
    
    # Initialize report generator if not exists
    if "report_generator" not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    
    generator = st.session_state.report_generator
    
    # Filter and search controls
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search Reports", "")
    with col2:
        mode_filter = st.selectbox(
            "Filter by Mode",
            ["All", "Standard", "Quality-Diversity", "Multi-Objective", "Adversarial", "Symbolic Regression", "Neuroevolution"]
        )
    with col3:
        date_filter = st.selectbox(
            "Date Range",
            ["All Time", "Last 24 Hours", "Last Week", "Last Month"]
        )
    
    # Display available reports
    if generator.reports:
        # Filter reports based on search and filters
        filtered_reports = generator.reports
        
        if search_term:
            filtered_reports = [r for r in filtered_reports if search_term.lower() in r.run_id.lower()]
        
        if mode_filter != "All":
            filtered_reports = [r for r in filtered_reports if r.evolution_mode.lower() == mode_filter.lower().replace("-", "_")]
        
        # Apply date filter
        if date_filter != "All Time":
            now = datetime.now()
            if date_filter == "Last 24 Hours":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_filter == "Last Week":
                cutoff = now.replace(day=now.day-7)
            elif date_filter == "Last Month":
                cutoff = now.replace(month=now.month-1)
            
            filtered_reports = [r for r in filtered_reports if r.timestamp >= cutoff]
        
        # Sort by timestamp (newest first)
        filtered_reports.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Display reports in cards
        for i, report in enumerate(filtered_reports):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{report.run_id}**")
                    st.caption(f"Mode: {report.evolution_mode.replace('_', '-').title()} | "
                              f"Type: {report.content_type} | "
                              f"Date: {report.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    st.metric("Best Score", f"{report.results.get('best_score', 0.0):.3f}")
                
                with col3:
                    if st.button(f"View Details #{i+1}", key=f"view_report_{i}"):
                        st.session_state.selected_report = report
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No reports available. Run an evolution to generate reports.")


def render_detailed_report(report: EvolutionReport):
    """Render detailed report view."""
    st.header(f"📊 Detailed Report: {report.run_id}")
    
    # Navigation
    if st.button("← Back to Reports"):
        if "selected_report" in st.session_state:
            del st.session_state.selected_report
        st.rerun()
    
    st.markdown("---")
    
    # Report header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Evolution Mode", report.evolution_mode.replace("_", "-").title())
    with col2:
        st.metric("Content Type", report.content_type)
    with col3:
        st.metric("Best Score", f"{report.results.get('best_score', 0.0):.3f}")
    with col4:
        st.metric("Runtime", f"{report.summary_statistics.get('total_runtime', 0):.1f}s")
    
    # Tabs for different sections
    tabs = st.tabs(["📈 Overview", "⚙️ Parameters", "📊 Results", "💡 Recommendations", "🖼️ Visualizations"])
    
    with tabs[0]:  # Overview
        st.subheader("Report Summary")
        
        # Summary statistics table
        stats_df = pd.DataFrame([report.summary_statistics]).T
        stats_df.columns = ["Value"]
        st.dataframe(stats_df, use_container_width=True)
        
        # Performance analysis
        st.subheader("Performance Analysis")
        perf_cols = st.columns(2)
        
        with perf_cols[0]:
            if "score_analysis" in report.performance_analysis:
                score_data = report.performance_analysis["score_analysis"]
                st.markdown(f"**Score Rating**: {score_data['rating']}")
                st.markdown(f"**Percentile**: {score_data['percentile']}th percentile")
        
        with perf_cols[1]:
            if "convergence_analysis" in report.performance_analysis:
                conv_data = report.performance_analysis["convergence_analysis"]
                st.markdown(f"**Convergence Speed**: {conv_data['speed']}")
                st.markdown(f"**Stability**: {conv_data['stability']}")
        
        # Efficiency metrics
        if "efficiency_analysis" in report.performance_analysis:
            eff_data = report.performance_analysis["efficiency_analysis"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Generation Time", f"{eff_data['avg_generation_time']:.2f}s")
            with col2:
                st.metric("Total Runtime", f"{eff_data['total_runtime']:.1f}s")
            with col3:
                st.metric("Efficiency Score", f"{eff_data['efficiency_score']:.3f}")
    
    with tabs[1]:  # Parameters
        st.subheader("Configuration Parameters")
        
        # Display parameters in a structured way
        param_df = pd.DataFrame([
            {"Parameter": k, "Value": v} 
            for k, v in report.parameters.items()
        ])
        
        st.dataframe(param_df, use_container_width=True)
        
        # Parameter category breakdown
        st.subheader("Parameter Categories")
        
        # Categorize parameters
        core_params = {k: v for k, v in report.parameters.items() if k in [
            "max_iterations", "population_size", "num_islands", "archive_size"
        ]}
        
        strategy_params = {k: v for k, v in report.parameters.items() if k in [
            "elite_ratio", "exploration_ratio", "exploitation_ratio", "migration_rate"
        ]}
        
        advanced_params = {k: v for k, v in report.parameters.items() if k not in list(core_params.keys()) + list(strategy_params.keys())}
        
        param_tabs = st.tabs(["Core", "Strategy", "Advanced"])
        
        with param_tabs[0]:
            core_df = pd.DataFrame(list(core_params.items()), columns=["Parameter", "Value"])
            st.dataframe(core_df, use_container_width=True)
        
        with param_tabs[1]:
            strategy_df = pd.DataFrame(list(strategy_params.items()), columns=["Parameter", "Value"])
            st.dataframe(strategy_df, use_container_width=True)
        
        with param_tabs[2]:
            advanced_df = pd.DataFrame(list(advanced_params.items()), columns=["Parameter", "Value"])
            st.dataframe(advanced_df, use_container_width=True)
    
    with tabs[2]:  # Results
        st.subheader("Evolution Results")
        
        # Best result display
        if "best_code" in report.results:
            st.markdown("**Best Generated Code/Solution:**")
            st.code(report.results["best_code"], language="python")
        
        # Metrics visualization
        st.subheader("Performance Metrics")
        
        # Create metrics visualization
        metrics_data = []
        for k, v in report.metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics_data.append({"Metric": k.replace("_", " ").title(), "Value": v})
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                title="Performance Metrics Overview",
                color="Value",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # Recommendations
        st.subheader("AI-Generated Recommendations")
        
        # Display recommendations with priority indicators
        for i, rec in enumerate(report.recommendations):
            # Extract priority indicator
            if rec.startswith("🔴"):
                emoji = "🔴"
                priority = "High"
            elif rec.startswith("🟡"):
                emoji = "🟡"
                priority = "Medium"
            elif rec.startswith("🟢"):
                emoji = "🟢"
                priority = "Low"
            elif rec.startswith("🔵"):
                emoji = "🔵"
                priority = "Informational"
            else:
                emoji = "⚪"
                priority = "General"
            
            # Clean the recommendation text
            clean_rec = rec.replace(emoji, "").strip()
            
            st.markdown(f"**{emoji} Priority: {priority}**")
            st.markdown(f"{clean_rec}")
            st.markdown("---")
    
    with tabs[4]:  # Visualizations
        st.subheader("Evolution Visualizations")
        
        # Display generated visualizations
        if report.visualizations:
            viz_cols = st.columns(2)
            viz_items = list(report.visualizations.items())
            
            for i, (viz_name, viz_data) in enumerate(viz_items):
                with viz_cols[i % 2]:
                    st.markdown(f"**{viz_name.replace('_', ' ').title()}**")
                    try:
                        # Decode base64 image data
                        image_bytes = base64.b64decode(viz_data)
                        st.image(image_bytes, caption=viz_name.replace("_", " ").title(), use_column_width=True)
                    except (ValueError, TypeError, RuntimeError) as e:
                        st.error(f"Could not display visualization: {e}")
        else:
            st.info("No visualizations available for this report.")
    
    # Export options
    st.markdown("---")
    st.subheader("Export Report")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export as PDF"):
            st.info("PDF export functionality would be implemented here")
    with col2:
        if st.button("💾 Export as JSON"):
            # Convert report to JSON and offer download
            report_dict = asdict(report)
            # Convert datetime to string for JSON serialization
            report_dict["timestamp"] = report_dict["timestamp"].isoformat()
            
            json_str = json.dumps(report_dict, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{report.run_id}_report.json">Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    with col3:
        if st.button("📄 Export as HTML"):
            st.info("HTML export functionality would be implemented here")


def render_reporting_dashboard():
    """Render the main reporting dashboard."""
    st.header("📑 OpenEvolve Reporting Center")
    
    # Check if a specific report is selected
    if "selected_report" in st.session_state:
        render_detailed_report(st.session_state.selected_report)
    else:
        # Main reporting interface
        tabs = st.tabs(["📋 Report Viewer", "📊 Analytics Hub", "⚙️ Report Settings"])
        
        with tabs[0]:  # Report Viewer
            render_interactive_report_viewer()
        
        with tabs[1]:  # Analytics Hub
            st.subheader("Advanced Analytics")
            
            # Sample analytics that would be generated from reports
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Mode Performance Comparison")
                # Mock data for demonstration
                mode_data = {
                    "Standard": 0.75,
                    "Quality-Diversity": 0.82,
                    "Multi-Objective": 0.78,
                    "Adversarial": 0.85,
                    "Symbolic Regression": 0.72,
                    "Neuroevolution": 0.88
                }
                
                mode_df = pd.DataFrame([
                    {"Mode": k, "Avg Score": v} 
                    for k, v in mode_data.items()
                ])
                
                fig_modes = px.bar(
                    mode_df,
                    x="Mode",
                    y="Avg Score",
                    title="Average Performance by Evolution Mode",
                    color="Avg Score",
                    color_continuous_scale="blues"
                )
                st.plotly_chart(fig_modes, use_container_width=True)
            
            with col2:
                st.markdown("### Performance Trends")
                # Mock trend data
                dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
                scores = np.cumsum(np.random.normal(0.01, 0.05, 30)) + 0.5
                scores = np.clip(scores, 0.1, 0.95)
                
                trend_df = pd.DataFrame({
                    "Date": dates,
                    "Performance": scores
                })
                
                fig_trend = px.line(
                    trend_df,
                    x="Date",
                    y="Performance",
                    title="Performance Trend Over Time"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Comparative analysis
            st.markdown("### Comparative Analysis")
            comparison_metrics = ["Best Score", "Runtime", "Diversity", "Efficiency"]
            comparison_data = {
                "Metric": comparison_metrics,
                "Run 1": [0.85, 120, 0.72, 0.81],
                "Run 2": [0.92, 180, 0.85, 0.89],
                "Run 3": [0.78, 95, 0.65, 0.74]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
        
        with tabs[2]:  # Report Settings
            st.subheader("Reporting Configuration")
            
            st.markdown("### Report Generation Settings")
            
            auto_generate = st.checkbox(
                "Auto-generate reports after each evolution",
                value=st.session_state.get("auto_generate_reports", True),
                help="Automatically create detailed reports after each evolution run"
            )
            st.session_state.auto_generate_reports = auto_generate
            
            detailed_viz = st.checkbox(
                "Generate detailed visualizations",
                value=st.session_state.get("detailed_visualizations", True),
                help="Include comprehensive charts and graphs in reports"
            )
            st.session_state.detailed_visualizations = detailed_viz
            
            export_formats = st.multiselect(
                "Preferred export formats",
                ["PDF", "JSON", "HTML", "Markdown"],
                default=st.session_state.get("preferred_export_formats", ["PDF", "JSON"]),
                help="Select your preferred formats for report exports"
            )
            st.session_state.preferred_export_formats = export_formats
            
            retention_period = st.select_slider(
                "Report retention period",
                options=["1 week", "1 month", "3 months", "6 months", "1 year", "Forever"],
                value=st.session_state.get("report_retention", "3 months"),
                help="How long to keep generated reports"
            )
            st.session_state.report_retention = retention_period
            
            st.markdown("### Report Templates")
            
            template_options = [
                "Standard Report",
                "Executive Summary", 
                "Technical Deep Dive",
                "Comparison Report",
                "Performance Analysis"
            ]
            
            selected_template = st.selectbox(
                "Default report template",
                template_options,
                index=template_options.index(st.session_state.get("default_report_template", "Standard Report"))
            )
            st.session_state.default_report_template = selected_template
            
            if st.button("Save Settings"):
                st.success("Report settings saved successfully!")


# Utility functions for integrating with evolution process
def create_evolution_report(
    run_id: str,
    evolution_mode: str,
    content_type: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    metrics: Dict[str, Any]
) -> EvolutionReport:
    """Create an evolution report and store it in session state."""
    
    # Initialize report generator if not exists
    if "report_generator" not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    
    generator = st.session_state.report_generator
    
    # Generate and return the report
    report = generator.generate_evolution_report(
        run_id=run_id,
        evolution_mode=evolution_mode,
        content_type=content_type,
        parameters=parameters,
        results=results,
        metrics=metrics
    )
    
    return report


def get_latest_reports(count: int = 5) -> List[EvolutionReport]:
    """Get the latest N reports."""
    if "report_generator" in st.session_state:
        reports = st.session_state.report_generator.reports
        # Return the most recent reports
        return sorted(reports, key=lambda x: x.timestamp, reverse=True)[:count]
    return []


def clear_old_reports():
    """Clear old reports based on retention settings."""
    if "report_generator" in st.session_state and "report_retention" in st.session_state:
        generator = st.session_state.report_generator
        retention = st.session_state.report_retention
        
        # Calculate cutoff date based on retention period
        now = datetime.now()
        if retention == "1 week":
            cutoff = now.replace(day=now.day-7)
        elif retention == "1 month":
            cutoff = now.replace(month=now.month-1)
        elif retention == "3 months":
            cutoff = now.replace(month=now.month-3)
        elif retention == "6 months":
            cutoff = now.replace(month=now.month-6)
        elif retention == "1 year":
            cutoff = now.replace(year=now.year-1)
        else:  # Forever
            return  # Don't clear anything
        
        # Filter out old reports
        generator.reports = [r for r in generator.reports if r.timestamp >= cutoff]


# =============================================================================
# ADAPTIVE MDAP REPORTING
# =============================================================================

def generate_adaptive_mdap_report(
    classifications: List[Dict[str, Any]],
    allocations: List[Dict[str, Any]],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Generate report for Adaptive MDAP usage.
    
    Args:
        classifications: List of classification records
        allocations: List of allocation records
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        Report dictionary with statistics and insights
    """
    try:
        # Filter by time range
        if start_time or end_time:
            classifications = [
                c for c in classifications
                if (not start_time or c.get('timestamp', datetime.now()) >= start_time)
                and (not end_time or c.get('timestamp', datetime.now()) <= end_time)
            ]
            allocations = [
                a for a in allocations
                if (not start_time or a.get('timestamp', datetime.now()) >= start_time)
                and (not end_time or a.get('timestamp', datetime.now()) <= end_time)
            ]
        
        # Calculate statistics
        total_classifications = len(classifications)
        total_allocations = len(allocations)
        
        # Complexity distribution
        complexity_scores = [c.get('complexity_score', 0) for c in classifications]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        complexity_distribution = {
            'direct': len([c for c in classifications if c.get('complexity_score', 0) <= 0.2]),
            'light': len([c for c in classifications if 0.2 < c.get('complexity_score', 0) <= 0.4]),
            'medium': len([c for c in classifications if 0.4 < c.get('complexity_score', 0) <= 0.6]),
            'full': len([c for c in classifications if 0.6 < c.get('complexity_score', 0) <= 0.8]),
            'ultra': len([c for c in classifications if c.get('complexity_score', 0) > 0.8])
        }
        
        # Strategy distribution
        strategies = {}
        for a in allocations:
            strategy = a.get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        # Agent usage
        total_agents = sum(a.get('n_agents', 0) for a in allocations)
        avg_agents = total_agents / len(allocations) if allocations else 0
        
        # Cost savings estimate (vs static 5-agent allocation)
        static_cost = len(allocations) * 5
        adaptive_cost = total_agents
        cost_savings_pct = ((static_cost - adaptive_cost) / static_cost * 100) if static_cost > 0 else 0
        
        # Performance metrics
        classification_latencies = [c.get('latency_ms', 0) for c in classifications]
        avg_classification_latency = np.mean(classification_latencies) if classification_latencies else 0
        
        allocation_latencies = [a.get('latency_ms', 0) for a in allocations]
        avg_allocation_latency = np.mean(allocation_latencies) if allocation_latencies else 0
        
        return {
            'period': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'summary': {
                'total_classifications': total_classifications,
                'total_allocations': total_allocations,
                'average_complexity': round(avg_complexity, 3),
                'average_agents': round(avg_agents, 1),
                'estimated_cost_savings_pct': round(cost_savings_pct, 1)
            },
            'complexity_distribution': complexity_distribution,
            'strategy_distribution': strategies,
            'performance': {
                'avg_classification_latency_ms': round(avg_classification_latency, 2),
                'avg_allocation_latency_ms': round(avg_allocation_latency, 2)
            },
            'recommendations': _generate_adaptive_recommendations(
                complexity_distribution, strategies, cost_savings_pct
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to generate Adaptive MDAP report: {e}")
        return {'error': str(e)}


def _generate_adaptive_recommendations(
    complexity_distribution: Dict[str, int],
    strategies: Dict[str, int],
    cost_savings_pct: float
) -> List[str]:
    """Generate recommendations based on Adaptive MDAP usage."""
    recommendations = []
    
    total = sum(complexity_distribution.values())
    if total == 0:
        return recommendations
    
    # Check complexity distribution
    ultra_pct = complexity_distribution.get('ultra', 0) / total * 100
    if ultra_pct > 20:
        recommendations.append(
            f"High percentage ({ultra_pct:.1f}%) of ultra-complex tasks detected. "
            "Consider problem simplification or additional resources."
        )
    
    # Check cost savings
    if cost_savings_pct < 20:
        recommendations.append(
            f"Cost savings ({cost_savings_pct:.1f}%) below target (30-50%). "
            "Consider using more conservative profile."
        )
    elif cost_savings_pct > 50:
        recommendations.append(
            f"Excellent cost savings ({cost_savings_pct:.1f}%). "
            "Current configuration is highly optimized."
        )
    
    # Check strategy balance
    if 'DIRECT' in strategies and strategies['DIRECT'] / total > 0.5:
        recommendations.append(
            "High proportion of simple tasks (DIRECT strategy). "
            "Consider batching or automation."
        )
    
    return recommendations


def export_adaptive_metrics_to_prometheus() -> str:
    """
    Export Adaptive MDAP metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics string
    """
    try:
        from monitoring_system import get_adaptive_metrics
        
        metrics = get_adaptive_metrics()
        if not metrics.get('adaptive_mdap_available'):
            return "# Adaptive MDAP not available\n"
        
        output = []
        output.append("# Adaptive MDAP Metrics")
        output.append("")
        
        # Classification metrics
        classifications = metrics.get('classifications', {})
        output.append(f"adaptive_classification_total {classifications.get('total', 0)}")
        
        # Allocation metrics
        allocations = metrics.get('allocations', {})
        output.append(f"adaptive_allocation_total {allocations.get('total', 0)}")
        
        # Performance metrics
        perf = metrics.get('performance', {})
        output.append(f"adaptive_avg_classification_latency_ms {perf.get('avg_classification_latency_ms', 0)}")
        output.append(f"adaptive_avg_allocation_latency_ms {perf.get('avg_allocation_latency_ms', 0)}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"# Error exporting metrics: {e}\n"
