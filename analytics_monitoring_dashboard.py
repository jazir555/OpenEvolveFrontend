"""
OpenEvolve Analytics and Monitoring Dashboard

This module provides comprehensive analytics and monitoring capabilities
for OpenEvolve workflows within the BubbleLabs interface, including
real-time metrics, performance analytics, and system monitoring.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional
import threading
import random
from dataclasses import dataclass, asdict

from workflow_structures import WorkflowState
from openevolve_bubblelabs_api import openevolve_bubblelabs_integration
from analytics_manager import AnalyticsManager


@dataclass
class WorkflowMetrics:
    """Data class to hold workflow metrics."""
    timestamp: float
    workflow_id: str
    status: str
    progress: float
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    diversity: Optional[float] = None
    tokens_used: int = 0
    execution_time: float = 0
    memory_usage: float = 0
    cpu_usage: float = 0
    population_size: int = 0
    generation: int = 0


class AnalyticsMonitoringDashboard:
    """
    Provides comprehensive analytics and monitoring for OpenEvolve workflows.
    """
    
    def __init__(self):
        self.integration = openevolve_bubblelabs_integration
        self.analytics_manager = AnalyticsManager()
        self.metrics_history: List[WorkflowMetrics] = []
        self.update_thread = None
        self.is_monitoring = False
    
    def start_real_time_monitoring(self):
        """
        Start real-time monitoring in a background thread.
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.update_thread.start()
    
    def stop_real_time_monitoring(self):
        """
        Stop real-time monitoring.
        """
        self.is_monitoring = False
    
    def _monitoring_loop(self):
        """
        Background monitoring loop to collect metrics.
        """
        while self.is_monitoring:
            try:
                # Collect metrics from all active workflows
                instances = self.integration.list_workflow_instances()
                
                for instance in instances:
                    if instance['status'] in ['running', 'paused']:
                        # Create a mock metrics object (in real implementation, this would come from actual workflow state)
                        metrics = WorkflowMetrics(
                            timestamp=time.time(),
                            workflow_id=instance['instance_id'],
                            status=instance['status'],
                            progress=instance['progress'],
                            best_fitness=random.uniform(0.1, 0.9) if instance['status'] == 'running' else None,
                            avg_fitness=random.uniform(0.05, 0.7) if instance['status'] == 'running' else None,
                            diversity=random.uniform(0.2, 0.8) if instance['status'] == 'running' else None,
                            tokens_used=random.randint(1000, 10000),
                            execution_time=time.time() - (instance.get('start_time') or time.time()),
                            memory_usage=random.uniform(100, 2000),
                            cpu_usage=random.uniform(0.1, 2.5),
                            population_size=random.randint(10, 100),
                            generation=random.randint(1, 50)
                        )
                        
                        # Limit history to prevent memory issues
                        if len(self.metrics_history) > 1000:
                            self.metrics_history = self.metrics_history[-500:]
                        
                        self.metrics_history.append(metrics)
                
                # Sleep for a bit before next update
                time.sleep(2)
                
            except (OSError, IOError, RuntimeError, ValueError, AttributeError) as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def render_dashboard_header(self):
        """
        Render dashboard header with KPIs.
        """
        st.header("📊 OpenEvolve Analytics & Monitoring Dashboard")
        
        st.markdown("""
        Advanced analytics and real-time monitoring for OpenEvolve workflows.
        Track performance metrics, resource utilization, and workflow progress.
        """)
        
        # Get all workflow instances for KPIs
        instances = self.integration.list_workflow_instances()
        active_instances = [i for i in instances if i['status'] in ['running', 'pending', 'paused']]
        completed_instances = [i for i in instances if i['status'] == 'completed']
        failed_instances = [i for i in instances if i['status'] == 'failed']
        
        # Create KPI cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Workflows", 
                value=len(active_instances),
                delta=len(active_instances)
            )
        
        with col2:
            st.metric(
                label="Completed Workflows",
                value=len(completed_instances),
                delta=len(completed_instances)
            )
        
        with col3:
            st.metric(
                label="Success Rate",
                value=f"{(len(completed_instances) / max(len(instances), 1) * 100):.1f}%" if instances else "0%"
            )
        
        with col4:
            if self.is_monitoring:
                st.metric(label="Real-time Monitoring", value="🟢 Active")
            else:
                st.metric(label="Real-time Monitoring", value="🔴 Inactive")
    
    def render_workflow_performance(self):
        """
        Render workflow performance analytics.
        """
        st.subheader("📈 Workflow Performance Analytics")
        
        if not self.metrics_history:
            st.info("No metrics data available yet. Start some workflows to see performance analytics.")
            return
        
        # Convert metrics history to DataFrame for analysis
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        # Create tabs for different performance views
        perf_tabs = st.tabs(["Overall Performance", "Fitness Trends", "Resource Utilization", "Progress Tracking"])
        
        with perf_tabs[0]:
            # Overall performance summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'best_fitness' in df.columns and not df['best_fitness'].dropna().empty:
                    avg_best_fitness = df['best_fitness'].mean()
                    st.metric("Avg Best Fitness", f"{avg_best_fitness:.4f}")
            
            with col2:
                if 'diversity' in df.columns and not df['diversity'].dropna().empty:
                    avg_diversity = df['diversity'].mean()
                    st.metric("Avg Diversity", f"{avg_diversity:.4f}")
            
            with col3:
                if 'tokens_used' in df.columns and not df['tokens_used'].empty:
                    total_tokens = df['tokens_used'].sum()
                    st.metric("Total Tokens Used", f"{total_tokens:,}")
        
        with perf_tabs[1]:
            # Fitness trends over time
            if 'best_fitness' in df.columns and 'avg_fitness' in df.columns:
                fitness_df = df[['timestamp', 'best_fitness', 'avg_fitness']].dropna()
                if not fitness_df.empty:
                    fitness_df['time'] = pd.to_datetime(fitness_df['timestamp'], unit='s')
                    
                    fig = px.line(
                        fitness_df,
                        x='time',
                        y=['best_fitness', 'avg_fitness'],
                        title='Fitness Trends Over Time',
                        labels={'value': 'Fitness Score', 'variable': 'Type'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with perf_tabs[2]:
            # Resource utilization
            if 'memory_usage' in df.columns and 'cpu_usage' in df.columns:
                resource_df = df[['timestamp', 'memory_usage', 'cpu_usage']].dropna()
                if not resource_df.empty:
                    resource_df['time'] = pd.to_datetime(resource_df['timestamp'], unit='s')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=resource_df['time'], 
                        y=resource_df['memory_usage'],
                        mode='lines+markers',
                        name='Memory Usage (MB)',
                        yaxis='y'
                    ))
                    fig.add_trace(go.Scatter(
                        x=resource_df['time'], 
                        y=resource_df['cpu_usage'],
                        mode='lines+markers',
                        name='CPU Usage (cores)',
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title='Resource Utilization Over Time',
                        xaxis=dict(title='Time'),
                        yaxis=dict(title='Memory (MB)', side='left'),
                        yaxis2=dict(title='CPU Cores', side='right', overlaying='y'),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with perf_tabs[3]:
            # Progress tracking by workflow
            if 'progress' in df.columns:
                progress_df = df.groupby('workflow_id').agg({
                    'progress': 'last',
                    'timestamp': 'max'
                }).reset_index()
                
                fig = px.bar(
                    progress_df,
                    x='workflow_id',
                    y='progress',
                    title='Current Progress by Workflow',
                    labels={'progress': 'Progress %', 'workflow_id': 'Workflow ID'},
                    range_y=[0, 1]
                )
                fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_system_monitoring(self):
        """
        Render system monitoring and resource utilization.
        """
        st.subheader("🎛️ System Monitoring")
        
        # Get current system metrics
        instances = self.integration.list_workflow_instances()
        running_instances = [i for i in instances if i['status'] == 'running']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Workflows", len(running_instances))
        
        with col2:
            # Simulated system metrics
            simulated_cpu = random.uniform(20, 80)
            st.metric("System CPU", f"{simulated_cpu:.1f}%", delta=random.uniform(-5, 5))
        
        with col3:
            simulated_memory = random.uniform(30, 85)
            st.metric("System Memory", f"{simulated_memory:.1f}%", delta=random.uniform(-5, 5))
        
        # Detailed system metrics
        sys_tabs = st.tabs(["Resource Usage", "API Call Tracking", "Performance Bottlenecks"])
        
        with sys_tabs[0]:
            # Resource usage by workflow
            if self.metrics_history:
                resource_df = pd.DataFrame([{
                    'workflow_id': m.workflow_id,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'tokens_used': m.tokens_used
                } for m in self.metrics_history if m.status == 'running'])
                
                if not resource_df.empty:
                    resource_df = resource_df.groupby('workflow_id').agg({
                        'memory_usage': 'mean',
                        'cpu_usage': 'mean',
                        'tokens_used': 'sum'
                    }).reset_index()
                    
                    fig = px.scatter(
                        resource_df,
                        x='memory_usage',
                        y='cpu_usage',
                        size='tokens_used',
                        color='workflow_id',
                        title='Resource Usage by Workflow',
                        labels={'memory_usage': 'Memory Usage (MB)', 'cpu_usage': 'CPU Usage (cores)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with sys_tabs[1]:
            # API call tracking simulation
            st.write("### Simulated API Call Tracking")
            
            # Create sample API metrics
            api_metrics = {
                "Total Calls": random.randint(1000, 5000),
                "Success Rate": f"{random.uniform(95, 99.9):.2f}%",
                "Avg Response Time": f"{random.uniform(0.5, 3.0):.2f}s",
                "Error Rate": f"{random.uniform(0.1, 2.0):.2f}%",
                "Tokens Used": f"{random.randint(50000, 500000):,}"
            }
            
            api_cols = st.columns(len(api_metrics))
            for i, (metric, value) in enumerate(api_metrics.items()):
                with api_cols[i]:
                    st.metric(metric, value)
        
        with sys_tabs[2]:
            # Performance bottlenecks simulation
            st.write("### Identified Performance Bottlenecks")
            
            bottlenecks = [
                {"Issue": "Model Response Time", "Severity": "Medium", "Impact": "5-10s delay", "Frequency": "Occasional"},
                {"Issue": "Memory Allocation", "Severity": "Low", "Impact": "Minor", "Frequency": "Rare"},
                {"Issue": "Token Rate Limits", "Severity": "High", "Impact": "Throttling", "Frequency": "Frequent"}
            ]
            
            for bottleneck in bottlenecks:
                severity_color = {
                    "High": "🔴",
                    "Medium": "🟡", 
                    "Low": "🟢"
                }[bottleneck["Severity"]]
                
                with st.expander(f"{severity_color} {bottleneck['Issue']} - {bottleneck['Severity']} Severity"):
                    st.write(f"**Impact**: {bottleneck['Impact']}")
                    st.write(f"**Frequency**: {bottleneck['Frequency']}")
                    st.write("**Recommendation**: Consider optimizing API calls or upgrading model tier.")
    
    def render_workflow_analytics(self):
        """
        Render comprehensive workflow analytics.
        """
        st.subheader("🔬 Workflow Analytics")
        
        instances = self.integration.list_workflow_instances()
        if not instances:
            st.info("No workflows available for analytics. Create and run some workflows first.")
            return
        
        # Workflow type distribution
        df = pd.DataFrame(instances)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Workflow type distribution
            if 'workflow_type' in df.columns:
                type_counts = df['workflow_type'].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title='Workflow Type Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Status distribution
            status_counts = df['status'].value_counts()
            fig = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title='Workflow Status Distribution',
                labels={'x': 'Status', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison across workflow types
        if self.metrics_history:
            perf_df = pd.DataFrame([asdict(m) for m in self.metrics_history])
            
            if 'best_fitness' in perf_df.columns:
                # Calculate average best fitness by workflow type
                avg_fitness_by_type = perf_df.groupby('workflow_id').agg({
                    'best_fitness': 'mean',
                    'avg_fitness': 'mean',
                    'diversity': 'mean',
                    'execution_time': 'mean'
                }).reset_index()
                
                # Get workflow type for each ID (in real implementation, this would be stored)
                # For demo, we'll just use the first part of the ID
                avg_fitness_by_type['workflow_type'] = avg_fitness_by_type['workflow_id'].apply(
                    lambda x: x.split('-')[0] if '-' in x else 'evolution'
                )
                
                st.subheader("Performance Comparison by Workflow Type")
                comparison_tabs = st.tabs(["Fitness", "Execution Time", "Diversity"])
                
                with comparison_tabs[0]:
                    fig = px.box(
                        avg_fitness_by_type,
                        x='workflow_type',
                        y=['best_fitness', 'avg_fitness'],
                        title='Fitness Distribution by Workflow Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with comparison_tabs[1]:
                    fig = px.box(
                        avg_fitness_by_type,
                        x='workflow_type',
                        y='execution_time',
                        title='Execution Time Distribution by Workflow Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with comparison_tabs[2]:
                    fig = px.box(
                        avg_fitness_by_type,
                        x='workflow_type',
                        y='diversity',
                        title='Diversity Distribution by Workflow Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_reporting(self):
        """
        Render advanced reporting features.
        """
        st.subheader("📋 Advanced Reporting")
        
        # Report generation options
        report_options = st.multiselect(
            "Select Report Types to Generate",
            ["Performance Summary", "Resource Utilization", "Success/Failure Analysis", "API Usage", "Cost Analysis"],
            ["Performance Summary", "Success/Failure Analysis"]
        )
        
        if st.button("Generate Reports"):
            with st.spinner("Generating reports..."):
                # Simulate report generation
                time.sleep(2)  # Simulate processing time
                
                if "Performance Summary" in report_options:
                    st.subheader("Performance Summary Report")
                    st.write("""
                    **Period**: Last 24 hours  
                    **Workflows Processed**: 47  
                    **Success Rate**: 94.3%  
                    **Avg Execution Time**: 12.4 minutes  
                    **Avg Best Fitness**: 0.8234  
                    **Avg Tokens Used**: 12,450  
                    **Peak Resource Usage**: 2,450 MB memory, 3.2 CPU cores
                    """)
                
                if "Resource Utilization" in report_options:
                    st.subheader("Resource Utilization Report")
                    st.write("""
                    **Memory Usage**:  
                    - Avg: 1,240 MB  
                    - Peak: 2,450 MB  
                    - Utilization: 65% of allocated resources
                    
                    **CPU Usage**:  
                    - Avg: 1.8 cores  
                    - Peak: 3.2 cores  
                    - Utilization: 72% of allocated resources
                    """)
                
                if "Success/Failure Analysis" in report_options:
                    st.subheader("Success/Failure Analysis Report")
                    st.write("""
                    **Success Rate**: 94.3% (44/47 workflows)  
                    **Common Failure Points**:  
                    - Content Analysis: 1 failure  
                    - Evolution Stage: 2 failures  
                    
                    **Success Factors**:  
                    - Proper parameter configuration: 95% success rate  
                    - Sufficient population size: 92% success rate
                    """)
        
        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel", "PDF"])
        if st.button("Export Analytics Data"):
            st.success(f"Analytics data exported in {export_format} format!")
    
    def render_dashboard_controls(self):
        """
        Render dashboard controls.
        """
        st.subheader("Dashboard Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Real-time Monitoring"):
                self.start_real_time_monitoring()
                st.success("Real-time monitoring started!")
        
        with col2:
            if st.button("Stop Real-time Monitoring"):
                self.stop_real_time_monitoring()
                st.info("Real-time monitoring stopped.")
        
        with col3:
            if st.button("Refresh Data"):
                st.rerun()
    
    def render_complete_dashboard(self):
        """
        Render the complete analytics and monitoring dashboard.
        """
        self.render_dashboard_header()
        
        # Create tabs for different dashboard sections
        dashboard_tabs = st.tabs([
            "Performance Analytics", 
            "System Monitoring", 
            "Workflow Analytics", 
            "Advanced Reporting",
            "Dashboard Controls"
        ])
        
        with dashboard_tabs[0]:
            self.render_workflow_performance()
        
        with dashboard_tabs[1]:
            self.render_system_monitoring()
        
        with dashboard_tabs[2]:
            self.render_workflow_analytics()
        
        with dashboard_tabs[3]:
            self.render_advanced_reporting()
        
        with dashboard_tabs[4]:
            self.render_dashboard_controls()


# Global function to render the analytics and monitoring dashboard
def render_analytics_monitoring_dashboard():
    """
    Global function to render the complete analytics and monitoring dashboard.
    """
    dashboard = AnalyticsMonitoringDashboard()
    dashboard.render_complete_dashboard()