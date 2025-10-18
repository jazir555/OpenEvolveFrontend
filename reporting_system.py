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
        # Simulated historical data for percentile calculation
        self.historical_scores: List[float] = [
            0.65, 0.70, 0.72, 0.75, 0.68, 0.80, 0.81, 0.79, 0.85, 0.77,
            0.90, 0.88, 0.92, 0.83, 0.76, 0.71, 0.84, 0.86, 0.91, 0.89
        ]
    
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
        
        return report
    
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
        # In a real implementation, this would compare against historical data
        # For now, we use a simulated historical data set
        
        # Add the current score to historical data for future calculations (optional, but makes it dynamic)
        self.historical_scores.append(score)
        
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
        # Simplified efficiency calculation
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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
                    except Exception as e:
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