"""
Analytics Dashboard Module

This module provides comprehensive analytics and visualization for the Decomposition Workflow,
including workflow performance, team effectiveness, gauntlet performance, and solution quality metrics.
"""
from __future__ import annotations


from ui_shim import ui as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from workflow_structures import WorkflowState, PerformanceMetrics, KnowledgeArtifact
from knowledge_manager import KnowledgeManager


class AnalyticsDashboard:
    """Manages analytics dashboard for the Decomposition Workflow."""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        """
        Initialize the Analytics Dashboard.
        
        Args:
            knowledge_manager: KnowledgeManager instance for accessing metrics and artifacts
        """
        self.knowledge_manager = knowledge_manager
    
    def render_analytics_dashboard(self):
        """Render the main analytics dashboard UI."""
        st.title("📊 Decomposition Workflow Analytics")
        
        # Create tabs for different analytics views
        tabs = st.tabs([
            "Overview",
            "Workflow Performance",
            "Team Analytics",
            "Gauntlet Analytics",
            "Solution Quality",
            "Knowledge Base Stats"
        ])
        
        with tabs[0]:
            self._render_overview()
        
        with tabs[1]:
            self._render_workflow_performance()
        
        with tabs[2]:
            self._render_team_analytics()
        
        with tabs[3]:
            self._render_gauntlet_analytics()
        
        with tabs[4]:
            self._render_solution_quality()
        
        with tabs[5]:
            self._render_knowledge_base_stats()
    
    def _render_overview(self):
        """Render overview dashboard with key metrics."""
        st.header("Overview")
        
        # Get all metrics
        all_metrics = self.knowledge_manager.get_performance_metrics(limit=1000)
        all_artifacts = self.knowledge_manager.get_all_artifacts()
        
        # Calculate summary statistics
        workflow_metrics = [m for m in all_metrics if m.entity_type == "workflow"]
        team_metrics = [m for m in all_metrics if m.entity_type == "team"]
        gauntlet_metrics = [m for m in all_metrics if m.entity_type == "gauntlet"]
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Workflows",
                len(workflow_metrics),
                help="Total number of workflows executed"
            )
        
        with col2:
            success_rate = self._calculate_success_rate(workflow_metrics)
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                help="Percentage of workflows that completed successfully"
            )
        
        with col3:
            st.metric(
                "Knowledge Artifacts",
                len(all_artifacts),
                help="Total number of knowledge artifacts extracted"
            )
        
        with col4:
            avg_duration = self._calculate_average_duration(workflow_metrics)
            st.metric(
                "Avg Duration",
                f"{avg_duration:.1f}m",
                help="Average workflow execution time in minutes"
            )
        
        # Recent activity chart
        st.subheader("Recent Activity")
        if workflow_metrics:
            activity_df = self._prepare_activity_data(workflow_metrics)
            fig = px.line(
                activity_df,
                x="date",
                y="count",
                title="Workflows Over Time",
                labels={"date": "Date", "count": "Number of Workflows"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No workflow data available yet.")
    
    def _render_workflow_performance(self):
        """Render workflow performance analytics."""
        st.header("Workflow Performance")
        
        # Get workflow metrics
        workflow_metrics = self.knowledge_manager.get_performance_metrics(
            entity_type="workflow",
            limit=100
        )
        
        if not workflow_metrics:
            st.info("No workflow performance data available yet.")
            return
        
        # Performance metrics over time
        st.subheader("Performance Trends")
        
        perf_data = []
        for metric in workflow_metrics:
            perf_data.append({
                "workflow_id": metric.entity_id,
                "timestamp": datetime.fromtimestamp(metric.timestamp),
                "success": metric.metrics.get("success", False),
                "duration": metric.metrics.get("duration_minutes", 0),
                "sub_problems_solved": metric.metrics.get("sub_problems_solved", 0),
                "refinement_loops": metric.metrics.get("refinement_loops", 0)
            })
        
        df = pd.DataFrame(perf_data)
        
        # Success rate over time
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x="timestamp",
                y="duration",
                color="success",
                title="Workflow Duration Over Time",
                labels={"timestamp": "Date", "duration": "Duration (minutes)"},
                color_discrete_map={True: "green", False: "red"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df,
                x="timestamp",
                y="sub_problems_solved",
                title="Sub-Problems Solved Per Workflow",
                labels={"timestamp": "Date", "sub_problems_solved": "Sub-Problems Solved"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Workflow Metrics")
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True
        )
    
    def _render_team_analytics(self):
        """Render team performance analytics."""
        st.header("Team Analytics")
        
        # Get team metrics
        team_metrics = self.knowledge_manager.get_performance_metrics(
            entity_type="team",
            limit=100
        )
        
        if not team_metrics:
            st.info("No team performance data available yet.")
            return
        
        # Aggregate team performance
        team_performance = {}
        for metric in team_metrics:
            team_name = metric.entity_id
            if team_name not in team_performance:
                team_performance[team_name] = {
                    "total_tasks": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_score": [],
                    "domains": set()
                }
            
            team_performance[team_name]["total_tasks"] += 1
            if metric.metrics.get("success", False):
                team_performance[team_name]["successes"] += 1
            else:
                team_performance[team_name]["failures"] += 1
            
            if "score" in metric.metrics:
                team_performance[team_name]["avg_score"].append(metric.metrics["score"])
            
            if metric.domain:
                team_performance[team_name]["domains"].add(metric.domain)
        
        # Calculate success rates and average scores
        team_data = []
        for team_name, perf in team_performance.items():
            success_rate = (perf["successes"] / perf["total_tasks"] * 100) if perf["total_tasks"] > 0 else 0
            avg_score = sum(perf["avg_score"]) / len(perf["avg_score"]) if perf["avg_score"] else 0
            
            team_data.append({
                "Team": team_name,
                "Total Tasks": perf["total_tasks"],
                "Successes": perf["successes"],
                "Failures": perf["failures"],
                "Success Rate": f"{success_rate:.1f}%",
                "Avg Score": f"{avg_score:.2f}",
                "Domains": ", ".join(perf["domains"]) if perf["domains"] else "N/A"
            })
        
        df = pd.DataFrame(team_data)
        
        # Team comparison chart
        st.subheader("Team Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df,
                x="Team",
                y="Total Tasks",
                color="Success Rate",
                title="Team Activity and Success Rate",
                labels={"Total Tasks": "Number of Tasks"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success vs Failure pie chart for selected team
            selected_team = st.selectbox("Select Team for Details", df["Team"].tolist())
            team_row = df[df["Team"] == selected_team].iloc[0]
            
            fig = go.Figure(data=[go.Pie(
                labels=["Successes", "Failures"],
                values=[team_row["Successes"], team_row["Failures"]],
                marker_colors=["green", "red"]
            )])
            fig.update_layout(title=f"{selected_team} Success/Failure Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed team metrics table
        st.subheader("Team Performance Summary")
        st.dataframe(df, use_container_width=True)
    
    def _render_gauntlet_analytics(self):
        """Render gauntlet performance analytics."""
        st.header("Gauntlet Analytics")
        
        # Get gauntlet metrics
        gauntlet_metrics = self.knowledge_manager.get_performance_metrics(
            entity_type="gauntlet",
            limit=100
        )
        
        if not gauntlet_metrics:
            st.info("No gauntlet performance data available yet.")
            return
        
        # Aggregate gauntlet performance
        gauntlet_performance = {}
        for metric in gauntlet_metrics:
            gauntlet_name = metric.entity_id
            if gauntlet_name not in gauntlet_performance:
                gauntlet_performance[gauntlet_name] = {
                    "total_runs": 0,
                    "approved": 0,
                    "rejected": 0,
                    "avg_scores": [],
                    "avg_duration": []
                }
            
            gauntlet_performance[gauntlet_name]["total_runs"] += 1
            if metric.metrics.get("approved", False):
                gauntlet_performance[gauntlet_name]["approved"] += 1
            else:
                gauntlet_performance[gauntlet_name]["rejected"] += 1
            
            if "average_score" in metric.metrics:
                gauntlet_performance[gauntlet_name]["avg_scores"].append(metric.metrics["average_score"])
            
            if "duration_seconds" in metric.metrics:
                gauntlet_performance[gauntlet_name]["avg_duration"].append(metric.metrics["duration_seconds"])
        
        # Calculate statistics
        gauntlet_data = []
        for gauntlet_name, perf in gauntlet_performance.items():
            approval_rate = (perf["approved"] / perf["total_runs"] * 100) if perf["total_runs"] > 0 else 0
            avg_score = sum(perf["avg_scores"]) / len(perf["avg_scores"]) if perf["avg_scores"] else 0
            avg_duration = sum(perf["avg_duration"]) / len(perf["avg_duration"]) if perf["avg_duration"] else 0
            
            gauntlet_data.append({
                "Gauntlet": gauntlet_name,
                "Total Runs": perf["total_runs"],
                "Approved": perf["approved"],
                "Rejected": perf["rejected"],
                "Approval Rate": f"{approval_rate:.1f}%",
                "Avg Score": f"{avg_score:.2f}",
                "Avg Duration": f"{avg_duration:.1f}s"
            })
        
        df = pd.DataFrame(gauntlet_data)
        
        # Gauntlet comparison chart
        st.subheader("Gauntlet Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df,
                x="Gauntlet",
                y="Total Runs",
                color="Approval Rate",
                title="Gauntlet Activity and Approval Rate",
                labels={"Total Runs": "Number of Runs"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Approval vs Rejection for selected gauntlet
            selected_gauntlet = st.selectbox("Select Gauntlet for Details", df["Gauntlet"].tolist())
            gauntlet_row = df[df["Gauntlet"] == selected_gauntlet].iloc[0]
            
            fig = go.Figure(data=[go.Pie(
                labels=["Approved", "Rejected"],
                values=[gauntlet_row["Approved"], gauntlet_row["Rejected"]],
                marker_colors=["green", "red"]
            )])
            fig.update_layout(title=f"{selected_gauntlet} Approval/Rejection Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed gauntlet metrics table
        st.subheader("Gauntlet Performance Summary")
        st.dataframe(df, use_container_width=True)
    
    def _render_solution_quality(self):
        """Render solution quality analytics."""
        st.header("Solution Quality Analytics")
        
        # Get solution pattern artifacts
        solution_artifacts = self.knowledge_manager.retrieve_relevant_knowledge(
            problem_statement="",  # Get all
            artifact_types=["solution_pattern"],
            limit=100
        )
        
        if not solution_artifacts:
            st.info("No solution quality data available yet.")
            return
        
        # Analyze solution quality
        quality_data = []
        for artifact in solution_artifacts:
            content = artifact.content
            quality_metrics = content.get("quality_metrics", {})
            
            quality_data.append({
                "Solution ID": artifact.id[:8],
                "Approach": content.get("solution_approach", "N/A"),
                "Model": content.get("generated_by_model", "N/A"),
                "Effectiveness": artifact.effectiveness_score,
                "Usage Count": artifact.usage_count,
                "Domain": artifact.domain or "N/A"
            })
        
        df = pd.DataFrame(quality_data)
        
        # Quality distribution
        st.subheader("Solution Quality Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x="Effectiveness",
                nbins=20,
                title="Solution Effectiveness Distribution",
                labels={"Effectiveness": "Effectiveness Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df,
                x="Usage Count",
                y="Effectiveness",
                color="Domain",
                title="Usage vs Effectiveness",
                labels={"Usage Count": "Times Used", "Effectiveness": "Effectiveness Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performing solutions
        st.subheader("Top Performing Solutions")
        top_solutions = df.nlargest(10, "Effectiveness")
        st.dataframe(top_solutions, use_container_width=True)
    
    def _render_knowledge_base_stats(self):
        """Render knowledge base statistics."""
        st.header("Knowledge Base Statistics")
        
        all_artifacts = self.knowledge_manager.get_all_artifacts()
        
        if not all_artifacts:
            st.info("Knowledge base is empty.")
            return
        
        # Artifact type distribution
        artifact_types = {}
        domains = {}
        total_usage = 0
        
        for artifact in all_artifacts:
            # Count by type
            artifact_types[artifact.artifact_type] = artifact_types.get(artifact.artifact_type, 0) + 1
            
            # Count by domain
            if artifact.domain:
                domains[artifact.domain] = domains.get(artifact.domain, 0) + 1
            
            # Total usage
            total_usage += artifact.usage_count
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Artifacts", len(all_artifacts))
        
        with col2:
            st.metric("Total Usage", total_usage)
        
        with col3:
            avg_effectiveness = sum(a.effectiveness_score for a in all_artifacts) / len(all_artifacts)
            st.metric("Avg Effectiveness", f"{avg_effectiveness:.2f}")
        
        # Artifact type distribution
        st.subheader("Artifact Type Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=list(artifact_types.keys()),
                values=list(artifact_types.values())
            )])
            fig.update_layout(title="Artifacts by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if domains:
                fig = go.Figure(data=[go.Pie(
                    labels=list(domains.keys()),
                    values=list(domains.values())
                )])
                fig.update_layout(title="Artifacts by Domain")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No domain information available.")
        
        # Most used artifacts
        st.subheader("Most Used Knowledge Artifacts")
        most_used = sorted(all_artifacts, key=lambda x: x.usage_count, reverse=True)[:10]
        
        usage_data = []
        for artifact in most_used:
            usage_data.append({
                "ID": artifact.id[:8],
                "Type": artifact.artifact_type,
                "Domain": artifact.domain or "N/A",
                "Usage Count": artifact.usage_count,
                "Effectiveness": f"{artifact.effectiveness_score:.2f}"
            })
        
        st.dataframe(pd.DataFrame(usage_data), use_container_width=True)
    
    # Helper methods
    
    def _calculate_success_rate(self, workflow_metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall success rate from workflow metrics."""
        if not workflow_metrics:
            return 0.0
        
        successes = sum(1 for m in workflow_metrics if m.metrics.get("success", False))
        return (successes / len(workflow_metrics)) * 100
    
    def _calculate_average_duration(self, workflow_metrics: List[PerformanceMetrics]) -> float:
        """Calculate average workflow duration in minutes."""
        if not workflow_metrics:
            return 0.0
        
        durations = [m.metrics.get("duration_minutes", 0) for m in workflow_metrics]
        return sum(durations) / len(durations) if durations else 0.0
    
    def _prepare_activity_data(self, workflow_metrics: List[PerformanceMetrics]) -> pd.DataFrame:
        """Prepare activity data for time series chart."""
        activity = {}
        
        for metric in workflow_metrics:
            date = datetime.fromtimestamp(metric.timestamp).date()
            activity[date] = activity.get(date, 0) + 1
        
        # Sort by date
        sorted_dates = sorted(activity.keys())
        
        return pd.DataFrame({
            "date": sorted_dates,
            "count": [activity[date] for date in sorted_dates]
        })


def render_analytics_dashboard():
    """Convenience function to render the analytics dashboard."""
    # Initialize knowledge manager
    knowledge_manager = KnowledgeManager()
    
    # Create and render dashboard
    dashboard = AnalyticsDashboard(knowledge_manager)
    dashboard.render_analytics_dashboard()


def render_openevolve_metrics_dashboard(metrics_data: Dict[str, Any]):
    """Render comprehensive OpenEvolve metrics dashboard with detailed visualizations"""
    st.subheader("🧬 OpenEvolve Metrics Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Operations", metrics_data.get('total_operations', 0))
    with col2:
        st.metric("Avg Fitness", f"{metrics_data.get('avg_fitness', 0.0):.3f}")
    with col3:
        st.metric("Best Fitness", f"{metrics_data.get('best_fitness', 0.0):.3f}")
    with col4:
        st.metric("Total Cost", f"${metrics_data.get('total_cost', 0.0):.2f}")
    with col5:
        st.metric("Success Rate", f"{metrics_data.get('success_rate', 0.0)*100:.1f}%")
    
    # Fitness evolution chart
    if 'fitness_history' in metrics_data and metrics_data['fitness_history']:
        st.subheader("Fitness Evolution Over Time")
        
        fitness_df = pd.DataFrame({
            'Iteration': range(len(metrics_data['fitness_history'])),
            'Fitness': metrics_data['fitness_history']
        })
        
        fig = px.line(
            fitness_df,
            x='Iteration',
            y='Fitness',
            title='Fitness Improvement Across Iterations',
            labels={'Fitness': 'Fitness Score', 'Iteration': 'Iteration Number'}
        )
        fig.add_hline(
            y=metrics_data.get('avg_fitness', 0),
            line_dash="dash",
            annotation_text="Average",
            line_color="orange"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Evolution mode comparison
    if 'by_evolution_mode' in metrics_data and metrics_data['by_evolution_mode']:
        st.subheader("Performance by Evolution Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode_df = pd.DataFrame(metrics_data['by_evolution_mode']).T
            mode_df = mode_df.reset_index()
            mode_df.columns = ['Mode'] + list(mode_df.columns[1:])
            
            if 'avg_fitness' in mode_df.columns:
                fig = px.bar(
                    mode_df,
                    x='Mode',
                    y='avg_fitness',
                    title='Average Fitness by Evolution Mode',
                    labels={'avg_fitness': 'Average Fitness', 'Mode': 'Evolution Mode'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'total_cost' in mode_df.columns:
                fig = px.bar(
                    mode_df,
                    x='Mode',
                    y='total_cost',
                    title='Total Cost by Evolution Mode',
                    labels={'total_cost': 'Total Cost ($)', 'Mode': 'Evolution Mode'},
                    color='total_cost',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(mode_df, use_container_width=True)
    
    # Population diversity metrics
    if 'diversity_metrics' in metrics_data:
        st.subheader("Population Diversity")
        
        diversity = metrics_data['diversity_metrics']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Diversity", f"{diversity.get('avg_diversity', 0.0):.3f}")
        with col2:
            st.metric("Max Diversity", f"{diversity.get('max_diversity', 0.0):.3f}")
        with col3:
            st.metric("Archive Coverage", f"{diversity.get('archive_coverage', 0.0)*100:.1f}%")
        
        if 'diversity_history' in diversity:
            diversity_df = pd.DataFrame({
                'Iteration': range(len(diversity['diversity_history'])),
                'Diversity': diversity['diversity_history']
            })
            
            fig = px.line(
                diversity_df,
                x='Iteration',
                y='Diversity',
                title='Diversity Evolution',
                labels={'Diversity': 'Diversity Score', 'Iteration': 'Iteration Number'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Resource usage breakdown
    if 'resource_usage' in metrics_data:
        st.subheader("Resource Usage")
        
        resources = metrics_data['resource_usage']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Calls", resources.get('api_calls', 0))
        with col2:
            st.metric("Tokens Used", f"{resources.get('tokens_used', 0):,}")
        with col3:
            st.metric("Execution Time", f"{resources.get('execution_time', 0.0):.1f}s")
        with col4:
            st.metric("Memory Peak", f"{resources.get('memory_peak_mb', 0.0):.1f} MB")
    
    # Parameter configuration summary
    if 'parameters' in metrics_data:
        with st.expander("📋 Configuration Parameters"):
            params = metrics_data['parameters']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Evolution Settings:**")
                st.write(f"- Max Iterations: {params.get('max_iterations', 'N/A')}")
                st.write(f"- Population Size: {params.get('population_size', 'N/A')}")
                st.write(f"- Archive Size: {params.get('archive_size', 'N/A')}")
                st.write(f"- Temperature: {params.get('temperature', 'N/A')}")
            
            with col2:
                st.write("**Selection Ratios:**")
                st.write(f"- Elite Ratio: {params.get('elite_ratio', 'N/A')}")
                st.write(f"- Exploration Ratio: {params.get('exploration_ratio', 'N/A')}")
                st.write(f"- Exploitation Ratio: {params.get('exploitation_ratio', 'N/A')}")
                
                if params.get('enable_quality_diversity'):
                    st.write("**Quality Diversity:** [OK] Enabled")
                if params.get('enable_cascade_evaluation'):
                    st.write("**Cascade Evaluation:** [OK] Enabled")

def render_diversity_heatmap(archive_data: List[Dict[str, Any]], feature_dimensions: Optional[List[str]] = None):
    """Render comprehensive quality diversity archive heatmap with interactive features"""
    st.subheader("🗺️ Quality Diversity Archive Heatmap")
    
    if not archive_data:
        st.info("No archive data available")
        return
    
    import numpy as np
    
    # Get feature dimensions
    if not feature_dimensions and archive_data:
        first_behavior = archive_data[0].get('behavior', {})
        feature_dimensions = list(first_behavior.keys())
    
    if not feature_dimensions or len(feature_dimensions) < 2:
        st.warning("Need at least 2 feature dimensions for heatmap visualization")
        return
    
    # Allow user to select dimensions to visualize
    col1, col2 = st.columns(2)
    with col1:
        dim_x = st.selectbox("X-Axis Dimension", feature_dimensions, index=0)
    with col2:
        dim_y = st.selectbox("Y-Axis Dimension", feature_dimensions, index=min(1, len(feature_dimensions)-1))
    
    # Grid size selection
    grid_size = st.slider("Grid Resolution", min_value=5, max_value=20, value=10, step=1)
    
    # Create heatmap data
    heatmap = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    
    for entry in archive_data:
        behavior = entry.get('behavior', {})
        if dim_x in behavior and dim_y in behavior:
            x = int(behavior[dim_x] * grid_size) % grid_size
            y = int(behavior[dim_y] * grid_size) % grid_size
            fitness = entry.get('fitness', 0)
            
            # Average fitness if multiple entries in same cell
            heatmap[y][x] = (heatmap[y][x] * counts[y][x] + fitness) / (counts[y][x] + 1)
            counts[y][x] += 1
    
    # Calculate coverage
    coverage = np.count_nonzero(counts) / (grid_size * grid_size) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Archive Size", len(archive_data))
    with col2:
        st.metric("Coverage", f"{coverage:.1f}%")
    with col3:
        st.metric("Avg Fitness", f"{np.mean([e.get('fitness', 0) for e in archive_data]):.3f}")
    
    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        colorscale='Viridis',
        colorbar=dict(title="Fitness"),
        hovertemplate=f'{dim_x}: %{{x}}<br>{dim_y}: %{{y}}<br>Fitness: %{{z:.3f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Archive Coverage: {dim_x} vs {dim_y}",
        xaxis_title=dim_x,
        yaxis_title=dim_y,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show filled cells distribution
    st.subheader("Cell Occupancy Distribution")
    
    occupancy_data = counts.flatten()
    occupancy_data = occupancy_data[occupancy_data > 0]  # Only filled cells
    
    if len(occupancy_data) > 0:
        fig = px.histogram(
            occupancy_data,
            nbins=20,
            title="Distribution of Solutions per Cell",
            labels={'value': 'Solutions per Cell', 'count': 'Number of Cells'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performers in archive
    with st.expander("🏆 Top Performers in Archive"):
        top_n = st.slider("Number of top solutions to show", 5, 20, 10)
        top_solutions = sorted(archive_data, key=lambda x: x.get('fitness', 0), reverse=True)[:top_n]
        
        top_data = []
        for i, sol in enumerate(top_solutions, 1):
            behavior = sol.get('behavior', {})
            top_data.append({
                'Rank': i,
                'Fitness': f"{sol.get('fitness', 0):.3f}",
                **{dim: f"{behavior.get(dim, 0):.3f}" for dim in feature_dimensions}
            })
        
        st.dataframe(pd.DataFrame(top_data), use_container_width=True)

def render_fitness_evolution_plot(evolution_history: List[Dict[str, Any]]):
    """Render detailed fitness evolution plot with statistics"""
    st.subheader("📊 Fitness Evolution Analysis")
    
    if not evolution_history:
        st.info("No evolution history available")
        return
    
    # Extract data
    iterations = [entry.get('iteration', i) for i, entry in enumerate(evolution_history)]
    best_fitness = [entry.get('best_fitness', 0) for entry in evolution_history]
    avg_fitness = [entry.get('avg_fitness', 0) for entry in evolution_history]
    worst_fitness = [entry.get('worst_fitness', 0) for entry in evolution_history]
    
    # Calculate improvement metrics
    if len(best_fitness) > 1:
        total_improvement = best_fitness[-1] - best_fitness[0]
        improvement_rate = total_improvement / len(best_fitness)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Fitness", f"{best_fitness[0]:.3f}")
        with col2:
            st.metric("Final Fitness", f"{best_fitness[-1]:.3f}")
        with col3:
            st.metric("Total Improvement", f"{total_improvement:.3f}")
        with col4:
            st.metric("Improvement Rate", f"{improvement_rate:.4f}/iter")
    
    # Create evolution plot
    fig = go.Figure()
    
    # Best fitness line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=best_fitness,
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='green', width=3),
        marker=dict(size=6)
    ))
    
    # Average fitness line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=avg_fitness,
        mode='lines',
        name='Average Fitness',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Worst fitness line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=worst_fitness,
        mode='lines',
        name='Worst Fitness',
        line=dict(color='red', width=1, dash='dot'),
        opacity=0.5
    ))
    
    # Add shaded area between best and worst
    fig.add_trace(go.Scatter(
        x=iterations + iterations[::-1],
        y=best_fitness + worst_fitness[::-1],
        fill='toself',
        fillcolor='rgba(0,100,200,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fitness Range'
    ))
    
    fig.update_layout(
        title='Fitness Evolution Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Fitness Score',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Convergence analysis
    st.subheader("Convergence Analysis")
    
    # Calculate convergence rate (change in best fitness)
    if len(best_fitness) > 1:
        convergence_rates = [best_fitness[i] - best_fitness[i-1] for i in range(1, len(best_fitness))]
        
        fig = px.bar(
            x=iterations[1:],
            y=convergence_rates,
            title='Fitness Improvement per Iteration',
            labels={'x': 'Iteration', 'y': 'Fitness Improvement'},
            color=convergence_rates,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify stagnation periods
        stagnation_threshold = 0.001
        stagnant_iterations = sum(1 for rate in convergence_rates if abs(rate) < stagnation_threshold)
        stagnation_percentage = (stagnant_iterations / len(convergence_rates)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stagnant Iterations", stagnant_iterations)
        with col2:
            st.metric("Stagnation %", f"{stagnation_percentage:.1f}%")


def render_population_diversity_plot(diversity_history: List[Dict[str, Any]]):
    """Render population diversity evolution plot"""
    st.subheader("🌈 Population Diversity Analysis")
    
    if not diversity_history:
        st.info("No diversity history available")
        return
    
    # Extract data
    iterations = [entry.get('iteration', i) for i, entry in enumerate(diversity_history)]
    diversity_scores = [entry.get('diversity', 0) for entry in diversity_history]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Initial Diversity", f"{diversity_scores[0]:.3f}")
    with col2:
        st.metric("Final Diversity", f"{diversity_scores[-1]:.3f}")
    with col3:
        avg_diversity = sum(diversity_scores) / len(diversity_scores)
        st.metric("Average Diversity", f"{avg_diversity:.3f}")
    
    # Create diversity plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=diversity_scores,
        mode='lines+markers',
        name='Diversity Score',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(128,0,128,0.1)'
    ))
    
    # Add average line
    fig.add_hline(
        y=avg_diversity,
        line_dash="dash",
        annotation_text="Average",
        line_color="orange"
    )
    
    fig.update_layout(
        title='Population Diversity Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Diversity Score',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Diversity trend analysis
    if len(diversity_scores) > 1:
        diversity_trend = "increasing" if diversity_scores[-1] > diversity_scores[0] else "decreasing"
        trend_color = "green" if diversity_trend == "increasing" else "red"
        
        st.markdown(f"**Diversity Trend:** :{trend_color}[{diversity_trend.upper()}]")
        
        if diversity_trend == "decreasing":
            st.warning("[WARN] Decreasing diversity may indicate premature convergence. Consider increasing exploration ratio or population size.")
        else:
            st.success("[OK] Increasing diversity indicates good exploration of the solution space.")


def render_pareto_front(solutions: List[Dict[str, Any]], objective_names: Optional[List[str]] = None):
    """Render interactive Pareto front visualization for multi-objective optimization"""
    st.subheader("📈 Pareto Front Visualization")
    
    if not solutions:
        st.info("No Pareto front data available")
        return
    
    # Detect objectives
    if not objective_names and solutions:
        first_sol = solutions[0]
        objective_names = [k for k in first_sol.keys() if k.startswith('objective_')]
    
    if not objective_names or len(objective_names) < 2:
        st.warning("Need at least 2 objectives for Pareto front visualization")
        return
    
    # Allow user to select objectives to visualize
    col1, col2 = st.columns(2)
    with col1:
        obj_x = st.selectbox("X-Axis Objective", objective_names, index=0)
    with col2:
        obj_y = st.selectbox("Y-Axis Objective", objective_names, index=min(1, len(objective_names)-1))
    
    # Extract objective values
    obj_x_vals = [s.get(obj_x, 0) for s in solutions]
    obj_y_vals = [s.get(obj_y, 0) for s in solutions]
    
    # Identify Pareto-optimal solutions
    pareto_optimal = []
    for i, sol in enumerate(solutions):
        is_dominated = False
        for j, other in enumerate(solutions):
            if i != j:
                # Check if other dominates sol
                if (other.get(obj_x, 0) >= sol.get(obj_x, 0) and 
                    other.get(obj_y, 0) >= sol.get(obj_y, 0) and
                    (other.get(obj_x, 0) > sol.get(obj_x, 0) or other.get(obj_y, 0) > sol.get(obj_y, 0))):
                    is_dominated = True
                    break
        pareto_optimal.append(not is_dominated)
    
    # Calculate metrics
    num_pareto = sum(pareto_optimal)
    pareto_percentage = (num_pareto / len(solutions)) * 100 if solutions else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Solutions", len(solutions))
    with col2:
        st.metric("Pareto-Optimal", num_pareto)
    with col3:
        st.metric("Pareto %", f"{pareto_percentage:.1f}%")
    
    # Create scatter plot
    colors = ['red' if p else 'blue' for p in pareto_optimal]
    sizes = [12 if p else 6 for p in pareto_optimal]
    
    fig = go.Figure()
    
    # Non-Pareto solutions
    non_pareto_x = [obj_x_vals[i] for i in range(len(solutions)) if not pareto_optimal[i]]
    non_pareto_y = [obj_y_vals[i] for i in range(len(solutions)) if not pareto_optimal[i]]
    
    if non_pareto_x:
        fig.add_trace(go.Scatter(
            x=non_pareto_x,
            y=non_pareto_y,
            mode='markers',
            name='Dominated',
            marker=dict(size=6, color='lightblue', opacity=0.6),
            hovertemplate=f'{obj_x}: %{{x:.3f}}<br>{obj_y}: %{{y:.3f}}<extra></extra>'
        ))
    
    # Pareto-optimal solutions
    pareto_x = [obj_x_vals[i] for i in range(len(solutions)) if pareto_optimal[i]]
    pareto_y = [obj_y_vals[i] for i in range(len(solutions)) if pareto_optimal[i]]
    
    if pareto_x:
        # Sort for line connection
        pareto_points = sorted(zip(pareto_x, pareto_y), key=lambda p: p[0])
        pareto_x_sorted = [p[0] for p in pareto_points]
        pareto_y_sorted = [p[1] for p in pareto_points]
        
        fig.add_trace(go.Scatter(
            x=pareto_x_sorted,
            y=pareto_y_sorted,
            mode='markers+lines',
            name='Pareto Front',
            marker=dict(size=12, color='red', symbol='star'),
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate=f'{obj_x}: %{{x:.3f}}<br>{obj_y}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Pareto Front: {obj_x} vs {obj_y}",
        xaxis_title=obj_x.replace('_', ' ').title(),
        yaxis_title=obj_y.replace('_', ' ').title(),
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hypervolume indicator (if 2D)
    if len(objective_names) == 2:
        st.subheader("Hypervolume Indicator")
        
        # Calculate hypervolume (simplified 2D case)
        if pareto_x:
            # Use reference point as minimum of all objectives
            ref_x = min(obj_x_vals) - 0.1 * (max(obj_x_vals) - min(obj_x_vals))
            ref_y = min(obj_y_vals) - 0.1 * (max(obj_y_vals) - min(obj_y_vals))
            
            # Calculate area under Pareto front
            hypervolume = 0
            sorted_pareto = sorted(zip(pareto_x, pareto_y), key=lambda p: p[0])
            
            for i in range(len(sorted_pareto)):
                x, y = sorted_pareto[i]
                if i == 0:
                    width = x - ref_x
                else:
                    width = x - sorted_pareto[i-1][0]
                height = y - ref_y
                hypervolume += width * height
            
            st.metric("Hypervolume", f"{hypervolume:.3f}")
            st.caption("Higher hypervolume indicates better coverage of the objective space")
    
    # Show Pareto-optimal solutions table
    with st.expander("🏆 Pareto-Optimal Solutions"):
        pareto_solutions = [solutions[i] for i in range(len(solutions)) if pareto_optimal[i]]
        
        if pareto_solutions:
            pareto_data = []
            for i, sol in enumerate(pareto_solutions, 1):
                row = {'Rank': i}
                for obj in objective_names:
                    row[obj.replace('_', ' ').title()] = f"{sol.get(obj, 0):.3f}"
                pareto_data.append(row)
            
            st.dataframe(pd.DataFrame(pareto_data), use_container_width=True)
        else:
            st.info("No Pareto-optimal solutions found")
