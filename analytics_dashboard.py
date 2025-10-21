"""
Analytics Dashboard Module

This module provides comprehensive analytics and visualization for the Decomposition Workflow,
including workflow performance, team effectiveness, gauntlet performance, and solution quality metrics.
"""

import streamlit as st
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
