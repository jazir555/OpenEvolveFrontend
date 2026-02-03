"""
OpenEvolve Workflow Visualization Module

This module provides comprehensive visualization for OpenEvolve workflow execution
within the BubbleLabs interface, including real-time metrics, progress tracking,
and workflow graph visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from workflow_structures import WorkflowState
from openevolve_bubblelabs_api import openevolve_bubblelabs_integration

class OpenEvolveVisualizer:
    """
    Provides visualization for OpenEvolve workflow execution.
    """
    
    def __init__(self):
        self.integration = openevolve_bubblelabs_integration
    
    def render_workflow_graph(self, workflow_state: WorkflowState):
        """
        Render a visual representation of the workflow graph.
        """
        st.subheader(f"{workflow_state.workflow_type.capitalize()} Workflow Graph")
        
        # Define the workflow stages based on the workflow type
        if workflow_state.workflow_type == "evolution":
            stages = ["Input", "Analysis", "Evolution", "Evaluation", "Output"]
        elif workflow_state.workflow_type == "adversarial":
            stages = ["Input", "Red Team", "Blue Team", "Evaluator", "Output"]
        elif workflow_state.workflow_type == "sovereign":
            stages = ["Input", "Analysis", "Decomposition", "Solving", "Assembly", "Verification", "Output"]
        else:
            stages = ["Input", "Processing", "Output"]
        
        # Create a Sankey diagram to represent the workflow
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=stages,
                color="blue"
            ),
            link=dict(
                source=[i for i in range(len(stages)-1)],
                target=[i+1 for i in range(len(stages)-1)],
                value=[100] * (len(stages)-1)
            )
        )])
        
        fig.update_layout(title_text=f"{workflow_state.workflow_type.capitalize()} Workflow Flow", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_execution_metrics(self, workflow_state: WorkflowState):
        """
        Render execution metrics for the workflow.
        """
        st.subheader("📈 Execution Metrics")
        
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if hasattr(workflow_state, 'execution_time') and workflow_state.execution_time:
                execution_time = workflow_state.execution_time
                minutes = int(execution_time // 60)
                seconds = int(execution_time % 60)
                st.metric("Execution Time", f"{minutes}m {seconds}s")
            else:
                st.metric("Execution Time", "N/A")
        
        with col2:
            progress = getattr(workflow_state, 'progress', 0) * 100
            st.metric("Progress", f"{progress:.1f}%")
        
        with col3:
            status = getattr(workflow_state, 'status', 'unknown')
            status_icon = self._get_status_icon(status)
            st.metric("Status", f"{status_icon} {status.upper()}")
        
        with col4:
            current_stage = getattr(workflow_state, 'current_stage', 'initializing')
            st.metric("Current Stage", current_stage)
        
        # Evolution-specific metrics
        if hasattr(workflow_state, 'best_fitness'):
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                best_fitness = getattr(workflow_state, 'best_fitness', 0)
                st.metric("Best Fitness", f"{best_fitness:.4f}")
            
            with col6:
                avg_fitness = getattr(workflow_state, 'avg_fitness', 0)
                st.metric("Avg Fitness", f"{avg_fitness:.4f}")
            
            with col7:
                diversity = getattr(workflow_state, 'diversity', 0)
                st.metric("Diversity", f"{diversity:.4f}")
            
            with col8:
                population_size = getattr(workflow_state, 'population_size', 0)
                st.metric("Population Size", population_size)
    
    def render_progress_timeline(self, workflow_state: WorkflowState):
        """
        Render a timeline of workflow progress.
        """
        st.subheader("📊 Progress Timeline")
        
        # Create a timeline of stages
        if workflow_state.workflow_type == "evolution":
            stages = ["Initialization", "Analysis", "Iteration 1", "Iteration 2", "Iteration 3", "Final Evaluation"]
            progress_values = [10, 25, 40, 60, 80, 100]
        elif workflow_state.workflow_type == "adversarial":
            stages = ["Initialization", "Red Team", "Blue Team", "Evaluation", "Iteration 2", "Final Assessment"]
            progress_values = [15, 30, 45, 65, 85, 100]
        elif workflow_state.workflow_type == "sovereign":
            stages = ["Input", "Analysis", "Decomposition", "Sub-solving", "Assembly", "Verification", "Output"]
            progress_values = [10, 20, 35, 55, 75, 90, 100]
        else:
            stages = ["Input", "Processing", "Output"]
            progress_values = [30, 60, 100]
        
        # Create the progress timeline
        fig = go.Figure()
        
        # Add progress line
        fig.add_trace(go.Scatter(
            x=list(range(len(stages))),
            y=progress_values,
            mode='lines+markers',
            name='Progress',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        # Add current progress line if available
        current_progress = getattr(workflow_state, 'progress', 0) * 100
        if current_progress > 0:
            fig.add_hline(y=current_progress, line_dash="dash", line_color="red", 
                         annotation_text=f"Current: {current_progress:.1f}%")
        
        fig.update_layout(
            title="Workflow Progress Timeline",
            xaxis_title="Stage",
            yaxis_title="Progress %",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(stages))),
                ticktext=stages
            ),
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_evolution_chart(self, workflow_state: WorkflowState):
        """
        Render evolution progress chart showing fitness over generations.
        """
        if workflow_state.workflow_type not in ["evolution", "sovereign"]:
            return
            
        st.subheader("📈 Evolution Progress")
        
        # Generate sample data for demonstration
        # In a real implementation, this would come from the actual workflow metrics
        generations = list(range(1, 21))  # 20 generations
        best_fitness = [0.1 + 0.7 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.05) for i in generations]
        avg_fitness = [0.05 + 0.6 * (1 - np.exp(-i/6)) + np.random.normal(0, 0.03) for i in generations]
        
        # Ensure values don't go below 0 or above 1
        best_fitness = [max(0, min(1, x)) for x in best_fitness]
        avg_fitness = [max(0, min(1, x)) for x in avg_fitness]
        
        df = pd.DataFrame({
            'Generation': generations,
            'Best Fitness': best_fitness,
            'Average Fitness': avg_fitness
        })
        
        fig = px.line(df, x='Generation', y=['Best Fitness', 'Average Fitness'],
                     title='Fitness Progress Over Generations',
                     line_shape='linear')
        
        fig.update_layout(
            xaxis_title='Generation',
            yaxis_title='Fitness Score',
            yaxis=dict(range=[0, 1.1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_diversity_chart(self, workflow_state: WorkflowState):
        """
        Render diversity metrics chart.
        """
        st.subheader("🔍 Diversity Analysis")
        
        if workflow_state.workflow_type in ["evolution", "sovereign"]:
            # Generate sample diversity data
            generations = list(range(1, 11))
            diversity_scores = [0.3 + 0.5 * np.sin(i/2) + np.random.normal(0, 0.1) for i in generations]
            diversity_scores = [max(0, min(1, x)) for x in diversity_scores]  # Clamp to [0, 1]
            
            df = pd.DataFrame({
                'Generation': generations,
                'Diversity Score': diversity_scores
            })
            
            fig = px.bar(df, x='Generation', y='Diversity Score',
                        title='Population Diversity Over Generations',
                        color='Diversity Score',
                        color_continuous_scale='viridis')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For non-evolution workflows, show a simple status chart
            stages = ["Input", "Processing", "Output"]
            status = [100, 50, 0]  # Example: input complete, processing 50%, output pending
            
            df = pd.DataFrame({
                'Stage': stages,
                'Completion %': status
            })
            
            fig = px.bar(df, x='Stage', y='Completion %',
                        title='Workflow Stage Completion',
                        color='Completion %',
                        color_continuous_scale='blues')
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_resource_utilization(self, workflow_state: WorkflowState):
        """
        Render resource utilization metrics.
        """
        st.subheader("🎛️ Resource Utilization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            memory_limit = getattr(workflow_state, 'memory_limit_mb', 2048)
            current_memory = min(memory_limit * 0.7, memory_limit)  # Example: 70% usage
            st.metric("Memory Usage", f"{current_memory:.0f} MB / {memory_limit} MB")
            st.progress(current_memory / memory_limit)
        
        with col2:
            cpu_limit = getattr(workflow_state, 'cpu_limit', 1.0)
            current_cpu = min(cpu_limit * 0.6, cpu_limit)  # Example: 60% usage
            st.metric("CPU Usage", f"{current_cpu:.1f} / {cpu_limit} cores")
            st.progress(current_cpu / cpu_limit)
        
        with col3:
            parallel_evals = getattr(workflow_state, 'parallel_evaluations', 4)
            active_evals = min(parallel_evals, 3)  # Example: 3 of 4 evals active
            st.metric("Active Evaluations", f"{active_evals} / {parallel_evals}")
            st.progress(active_evals / parallel_evals if parallel_evals > 0 else 0)
    
    def render_workflow_status_pane(self, workflow_state: WorkflowState):
        """
        Render a comprehensive workflow status pane.
        """
        st.subheader("📋 Workflow Status")
        
        # Create status cards
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.write("### 📊 Basic Info")
            st.write(f"**Workflow ID:** `{workflow_state.workflow_id}`")
            st.write(f"**Type:** {workflow_state.workflow_type}")
            st.write(f"**Problem:** {workflow_state.problem_statement[:50]}{'...' if len(workflow_state.problem_statement) > 50 else ''}")
            st.write(f"**Created:** {datetime.fromtimestamp(getattr(workflow_state, 'start_time', time.time())).strftime('%Y-%m-%d %H:%M:%S')}")
        
        with status_col2:
            st.write("### 🔄 Execution Info")
            st.write(f"**Status:** {self._get_status_icon(workflow_state.status)} {workflow_state.status.upper()}")
            st.write(f"**Stage:** {workflow_state.current_stage}")
            st.write(f"**Progress:** {getattr(workflow_state, 'progress', 0) * 100:.1f}%")
            if hasattr(workflow_state, 'execution_time') and workflow_state.execution_time:
                st.write(f"**Runtime:** {workflow_state.execution_time:.2f}s")
    
    def _get_status_icon(self, status: str) -> str:
        """
        Get appropriate icon for workflow status.
        """
        status_icons = {
            'created': '🆕',
            'pending': '⏳',
            'running': '🏃',
            'paused': '⏸️',
            'stopping': '🛑',
            'stopped': '⏹️',
            'completed': '✅',
            'failed': '❌',
            'cancelled': '🚫'
        }
        return status_icons.get(status.lower(), '❓')
    
    def render_real_time_monitoring(self, workflow_state: WorkflowState):
        """
        Render real-time monitoring dashboard for the workflow.
        """
        st.subheader("📡 Real-time Monitoring")
        
        # Create real-time metrics that update
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Threads", f"{len(self.integration.running_threads)}")
        
        with col2:
            st.metric("Tokens Used", f"{getattr(workflow_state, 'tokens_used', 0):,}")
        
        with col3:
            st.metric("Iterations", getattr(workflow_state, 'iterations_completed', 0))
        
        with col4:
            if hasattr(workflow_state, 'convergence') and workflow_state.convergence:
                st.metric("Convergence", f"{workflow_state.convergence:.3f}")
            else:
                st.metric("Convergence", "N/A")
        
        # Show live log of workflow events
        st.subheader("Workflow Events")

        # In a real implementation, this would show actual workflow events
        # Use monitoring events when available
        events = st.session_state.get("sgd_monitoring_data", [])
        if not events:
            events = st.session_state.get("workflow_events", [])

        for event in events:
            if "timestamp" in event and isinstance(event["timestamp"], str):
                timestamp = event["timestamp"]
            else:
                timestamp = datetime.fromtimestamp(event.get("timestamp", time.time())).strftime('%H:%M:%S')

            status = event.get("status") or event.get("level", "INFO")
            if status in ["failure", "error", "ERROR"]:
                level_icon = "[X]"
            elif status in ["warning", "WARN"]:
                level_icon = "[!]"
            else:
                level_icon = "[i]"

            message = event.get("message") or event.get("event") or "Workflow event"
            st.write(f"{level_icon} `{timestamp}` - {message}")

        if not events:
            st.info("No workflow events recorded yet.")

    def render_complete_dashboard(self, workflow_state: Optional[WorkflowState] = None):
        """
        Render the complete visualization dashboard.
        """
        if workflow_state is None:
            # Create a dummy workflow state for demonstration
            from workflow_structures import WorkflowState
            workflow_state = WorkflowState(
                workflow_id="demo-123",
                workflow_type="evolution",
                problem_statement="Optimize neural network architecture for image classification",
                current_stage="evolving",
                status="running"
            )
        
        # Main visualization sections
        self.render_workflow_status_pane(workflow_state)
        
        # Execution metrics
        self.render_execution_metrics(workflow_state)
        
        # Resource utilization
        self.render_resource_utilization(workflow_state)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_evolution_chart(workflow_state)
        
        with col2:
            self.render_diversity_chart(workflow_state)
        
        # Progress and monitoring
        self.render_progress_timeline(workflow_state)
        self.render_real_time_monitoring(workflow_state)
        
        # Workflow graph visualization
        self.render_workflow_graph(workflow_state)


# Global function to render visualization
def render_workflow_visualization(workflow_state: Optional[WorkflowState] = None):
    """
    Global function to render the workflow visualization.
    """
    visualizer = OpenEvolveVisualizer()
    visualizer.render_complete_dashboard(workflow_state)
