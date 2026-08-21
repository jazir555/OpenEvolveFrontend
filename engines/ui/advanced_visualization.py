"""
Advanced Visualization and Reporting Module

This module provides advanced visualization and reporting features for workflows.
"""
from __future__ import annotations


import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import networkx as nx
import io
import base64
import time

from workflow_structures import WorkflowState, DecompositionPlan, SubProblem

class DependencyGraphVisualizer:
    """Visualizes dependency graphs for sub-problems."""
    
    def __init__(self):
        """Initialize dependency graph visualizer."""
        self.node_colors = {
            "pending": "#FFA500",  # Orange
            "in_progress": "#4169E1",  # Royal Blue
            "solved": "#32CD32",  # Lime Green
            "failed": "#DC143C",  # Crimson
            "requires_rework": "#FFD700"  # Gold
        }
    
    def generate_graph(self, sub_problems: List[SubProblem]) -> nx.DiGraph:
        """
        Generate networkx directed graph from sub-problems.
        
        Args:
            sub_problems: List of sub-problems
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for sp in sub_problems:
            G.add_node(
                sp.id,
                description=sp.description[:50] + "..." if len(sp.description) > 50 else sp.description,
                status=sp.status,
                complexity=sp.ai_suggested_complexity_score
            )
        
        # Add edges (dependencies)
        for sp in sub_problems:
            for dep_id in sp.dependencies:
                G.add_edge(dep_id, sp.id)
        
        return G
    
    def create_interactive_graph(self, sub_problems: List[SubProblem]) -> go.Figure:
        """
        Create interactive dependency graph using Plotly.
        
        Args:
            sub_problems: List of sub-problems
            
        Returns:
            Plotly figure
        """
        G = self.generate_graph(sub_problems)
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            status = node_data.get('status', 'pending')
            complexity = node_data.get('complexity', 5)
            description = node_data.get('description', '')
            
            node_text.append(f"{node}<br>{description}<br>Status: {status}<br>Complexity: {complexity}")
            node_colors.append(self.node_colors.get(status, '#808080'))
            node_sizes.append(10 + complexity * 3)  # Size based on complexity
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white')
            ),
            name='Sub-Problems'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Sub-Problem Dependency Graph',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def export_graph(self, sub_problems: List[SubProblem], format: str = 'png') -> bytes:
        """
        Export dependency graph to image format.
        
        Args:
            sub_problems: List of sub-problems
            format: Export format ('png', 'svg', 'pdf')
            
        Returns:
            Image bytes
        """
        import matplotlib.pyplot as plt
        
        G = self.generate_graph(sub_problems)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Get node colors based on status
        node_colors_list = [
            self.node_colors.get(G.nodes[node].get('status', 'pending'), '#808080')
            for node in G.nodes()
        ]
        
        # Draw graph
        nx.draw(
            G, pos,
            node_color=node_colors_list,
            with_labels=True,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='#888',
            ax=ax
        )
        
        ax.set_title('Sub-Problem Dependency Graph', fontsize=16, fontweight='bold')
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format=format, dpi=300, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.read()
        plt.close(fig)
        
        return image_bytes
    
    def get_critical_path(self, sub_problems: List[SubProblem]) -> List[str]:
        """
        Identify the critical path (longest path) through the dependency graph.
        
        Args:
            sub_problems: List of sub-problems
            
        Returns:
            List of sub-problem IDs in the critical path
        """
        G = self.generate_graph(sub_problems)
        
        try:
            # Find longest path using topological sort
            longest_path = nx.dag_longest_path(G)
            return longest_path
        except Exception as e:
            # Log the specific error for debugging
            import logging
            logging.exception(f"Error in advanced_visualization longest path calculation: {e}")
            # If graph has cycles or other issues, return empty list
            return []


class AdvancedVisualizer:
    """Provides advanced visualization capabilities."""
    
    def __init__(self):
        """Initialize advanced visualizer."""
        self.color_scheme = px.colors.qualitative.Set3
        self.dependency_visualizer = DependencyGraphVisualizer()
    
    def create_workflow_timeline(self, workflow_state: WorkflowState) -> go.Figure:
        """Create interactive timeline visualization of workflow execution."""
        # Create timeline data
        events = []
        
        # Add workflow start
        events.append({
            "Task": "Workflow Start",
            "Start": datetime.fromtimestamp(workflow_state.start_time),
            "Finish": datetime.fromtimestamp(workflow_state.start_time + 60),
            "Stage": "Initialization"
        })
        
        # Add sub-problem solving events
        if workflow_state.decomposition_plan:
            for i, sp in enumerate(workflow_state.decomposition_plan.sub_problems):
                if sp.id in workflow_state.solved_sub_problem_ids:
                    events.append({
                        "Task": f"Sub-Problem {sp.id}",
                        "Start": datetime.fromtimestamp(workflow_state.start_time + (i * 300)),
                        "Finish": datetime.fromtimestamp(workflow_state.start_time + ((i + 1) * 300)),
                        "Stage": "Sub-Problem Solving"
                    })
        
        # Add workflow end
        if workflow_state.end_time:
            events.append({
                "Task": "Workflow Complete",
                "Start": datetime.fromtimestamp(workflow_state.end_time - 60),
                "Finish": datetime.fromtimestamp(workflow_state.end_time),
                "Stage": "Completion"
            })
        
        df = pd.DataFrame(events)
        
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Stage",
            title="Workflow Execution Timeline"
        )
        
        fig.update_yaxes(autorange="reversed")
        return fig
    
    def create_complexity_heatmap(self, plan: DecompositionPlan) -> go.Figure:
        """Create heatmap of sub-problem complexity."""
        if not plan.sub_problems:
            return go.Figure()
        
        # Create matrix data
        sp_ids = [sp.id for sp in plan.sub_problems]
        complexities = [sp.ai_suggested_complexity_score for sp in plan.sub_problems]
        
        fig = go.Figure(data=go.Heatmap(
            z=[complexities],
            x=sp_ids,
            y=["Complexity"],
            colorscale="Viridis",
            text=[[f"{c}" for c in complexities]],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Sub-Problem Complexity Heatmap",
            xaxis_title="Sub-Problem ID",
            height=200
        )
        
        return fig
    
    def create_workflow_flow_diagram(self, workflow_state: WorkflowState) -> go.Figure:
        """
        Create real-time workflow execution flow diagram.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            Plotly figure showing workflow flow
        """
        # Define workflow stages
        stages = [
            "Content Analysis",
            "Decomposition",
            "Manual Review",
            "Sub-Problem Solving",
            "Reassembly",
            "Final Verification",
            "Completed"
        ]
        
        # Map current stage to index
        stage_map = {
            "INITIALIZING": 0,
            "Content Analysis": 0,
            "AI-Assisted Decomposition": 1,
            "Manual Review & Override": 2,
            "Sub-Problem Solving Loop": 3,
            "Configurable Reassembly": 4,
            "Final Verification & Self-Healing Loop": 5,
            "Knowledge Extraction & Learning": 6,
            "COMPLETED": 6
        }
        
        current_stage_idx = stage_map.get(workflow_state.current_stage, 0)
        
        # Create flow diagram
        fig = go.Figure()
        
        # Add stages as nodes
        for i, stage in enumerate(stages):
            # Determine color based on status
            if i < current_stage_idx:
                color = '#32CD32'  # Green - completed
                symbol = 'circle'
            elif i == current_stage_idx:
                color = '#4169E1'  # Blue - current
                symbol = 'diamond'
            else:
                color = '#D3D3D3'  # Gray - pending
                symbol = 'circle'
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=30, color=color, symbol=symbol, line=dict(width=2, color='white')),
                text=[stage],
                textposition="bottom center",
                name=stage,
                hovertext=f"Stage: {stage}<br>Status: {'Completed' if i < current_stage_idx else 'Current' if i == current_stage_idx else 'Pending'}",
                hoverinfo='text'
            ))
        
        # Add connecting lines
        for i in range(len(stages) - 1):
            color = '#32CD32' if i < current_stage_idx else '#D3D3D3'
            fig.add_trace(go.Scatter(
                x=[i, i + 1],
                y=[0, 0],
                mode='lines',
                line=dict(width=3, color=color),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'Workflow Execution Flow - {workflow_state.status.upper()}',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, len(stages) - 0.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            plot_bgcolor='white',
            height=300,
            showlegend=False
        )
        
        return fig
    
    def update_flow_real_time(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """
        Update flow visualization in real-time.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            Dictionary with update data
        """
        return {
            "current_stage": workflow_state.current_stage,
            "status": workflow_state.status,
            "progress": workflow_state.progress,
            "timestamp": time.time()
        }
    
    def create_performance_dashboard(self, workflow_state: WorkflowState) -> Dict[str, go.Figure]:
        """Create comprehensive performance dashboard."""
        figures = {}
        
        # 1. Progress gauge
        figures["progress"] = self._create_progress_gauge(workflow_state)
        
        # 2. Sub-problem status pie chart
        figures["status"] = self._create_status_pie_chart(workflow_state)
        
        # 3. Refinement loops bar chart
        figures["refinement"] = self._create_refinement_chart(workflow_state)
        
        # 4. Timeline
        figures["timeline"] = self.create_workflow_timeline(workflow_state)
        
        # 5. Workflow flow diagram
        figures["flow"] = self.create_workflow_flow_diagram(workflow_state)
        
        return figures
    
    def _create_progress_gauge(self, workflow_state: WorkflowState) -> go.Figure:
        """Create progress gauge."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=workflow_state.progress * 100,
            title={"text": "Workflow Progress"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 75], "color": "gray"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90
                }
            }
        ))
        
        return fig
    
    def _create_status_pie_chart(self, workflow_state: WorkflowState) -> go.Figure:
        """Create pie chart of sub-problem statuses."""
        if not workflow_state.decomposition_plan:
            return go.Figure()
        
        status_counts = {}
        for sp in workflow_state.decomposition_plan.sub_problems:
            status = sp.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3
        )])
        
        fig.update_layout(title="Sub-Problem Status Distribution")
        
        return fig
    
    def _create_refinement_chart(self, workflow_state: WorkflowState) -> go.Figure:
        """Create bar chart of refinement loops."""
        fig = go.Figure(data=[
            go.Bar(
                x=["Refinement Loops"],
                y=[workflow_state.refinement_loop_count],
                text=[workflow_state.refinement_loop_count],
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Refinement Loop Count",
            yaxis_title="Count",
            showlegend=False
        )
        
        return fig
    
    def create_performance_metrics_chart(self, workflow_state: WorkflowState) -> go.Figure:
        """
        Create performance metric charts.
        
        Args:
            workflow_state: Workflow state with performance metrics
            
        Returns:
            Plotly figure with performance metrics
        """
        # Calculate metrics
        metrics = {}
        
        # Execution time
        if workflow_state.start_time and workflow_state.end_time:
            metrics["Execution Time (s)"] = workflow_state.end_time - workflow_state.start_time
        elif workflow_state.start_time:
            metrics["Execution Time (s)"] = time.time() - workflow_state.start_time
        
        # Sub-problems metrics
        if workflow_state.decomposition_plan:
            total_sp = len(workflow_state.decomposition_plan.sub_problems)
            solved_sp = len(workflow_state.solved_sub_problem_ids)
            metrics["Sub-Problems Solved"] = solved_sp
            metrics["Sub-Problems Total"] = total_sp
            if total_sp > 0:
                metrics["Success Rate (%)"] = (solved_sp / total_sp) * 100
        
        # Refinement metrics
        metrics["Refinement Loops"] = workflow_state.refinement_loop_count
        
        # Resource usage from performance_metrics if available
        if workflow_state.performance_metrics:
            for key, value in workflow_state.performance_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in metrics.values()],
                textposition="auto",
                marker=dict(color='#4169E1')
            )
        ])
        
        fig.update_layout(
            title="Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_resource_usage_chart(self, workflow_state: WorkflowState) -> go.Figure:
        """
        Create resource usage visualization.
        
        Args:
            workflow_state: Workflow state with resource usage data
            
        Returns:
            Plotly figure showing resource usage
        """
        # Extract resource usage data
        resource_data = workflow_state.resource_usage if workflow_state.resource_usage else {}
        
        if not resource_data:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No resource usage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(title="Resource Usage")
            return fig
        
        # Create subplots for different resource types
        categories = list(resource_data.keys())
        values = list(resource_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker=dict(
                    color=values,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in values],
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Resource Usage",
            xaxis_title="Resource Type",
            yaxis_title="Usage",
            height=400
        )
        
        return fig
    
    def create_quality_scores_chart(self, workflow_state: WorkflowState) -> go.Figure:
        """
        Create quality scores visualization.
        
        Args:
            workflow_state: Workflow state
            
        Returns:
            Plotly figure showing quality scores
        """
        if not workflow_state.decomposition_plan:
            fig = go.Figure()
            fig.add_annotation(
                text="No quality data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Collect quality scores from verification reports
        sp_ids = []
        quality_scores = []
        
        for sp_id, solution in workflow_state.sub_problem_solutions.items():
            if solution.verification_reports:
                # Get average score from verification reports
                avg_score = sum(vr.average_score for vr in solution.verification_reports) / len(solution.verification_reports)
                sp_ids.append(sp_id)
                quality_scores.append(avg_score * 100)  # Convert to percentage
        
        if not sp_ids:
            fig = go.Figure()
            fig.add_annotation(
                text="No quality scores available yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure(data=[
            go.Bar(
                x=sp_ids,
                y=quality_scores,
                marker=dict(
                    color=quality_scores,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(title="Quality %")
                ),
                text=[f"{score:.1f}%" for score in quality_scores],
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Solution Quality Scores by Sub-Problem",
            xaxis_title="Sub-Problem ID",
            yaxis_title="Quality Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig


class ReportGenerator:
    """Generates comprehensive reports for workflows."""
    
    def generate_executive_summary(self, workflow_state: WorkflowState) -> str:
        """Generate executive summary report."""
        report = []
        report.append("="*70)
        report.append("WORKFLOW EXECUTIVE SUMMARY")
        report.append("="*70)
        report.append(f"Workflow ID: {workflow_state.workflow_id}")
        report.append(f"Problem: {workflow_state.problem_statement[:100]}...")
        report.append(f"Status: {workflow_state.status.upper()}")
        report.append(f"Progress: {workflow_state.progress * 100:.1f}%")
        report.append("")
        
        if workflow_state.decomposition_plan:
            total_sp = len(workflow_state.decomposition_plan.sub_problems)
            solved_sp = len(workflow_state.solved_sub_problem_ids)
            report.append(f"Sub-Problems: {solved_sp}/{total_sp} solved")
        
        report.append(f"Refinement Loops: {workflow_state.refinement_loop_count}")
        
        if workflow_state.end_time:
            duration = workflow_state.end_time - workflow_state.start_time
            report.append(f"Duration: {duration:.1f} seconds")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def generate_detailed_report(self, workflow_state: WorkflowState) -> str:
        """Generate detailed workflow report."""
        report = []
        report.append("="*70)
        report.append("DETAILED WORKFLOW REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Executive summary
        report.append(self.generate_executive_summary(workflow_state))
        report.append("")
        
        # Sub-problem details
        if workflow_state.decomposition_plan:
            report.append("SUB-PROBLEM DETAILS:")
            report.append("-"*70)
            for sp in workflow_state.decomposition_plan.sub_problems:
                report.append(f"\n{sp.id}: {sp.description[:80]}...")
                report.append(f"  Status: {sp.status}")
                report.append(f"  Complexity: {sp.ai_suggested_complexity_score}/10")
                report.append(f"  Dependencies: {', '.join(sp.dependencies) if sp.dependencies else 'None'}")
                
                if sp.id in workflow_state.sub_problem_solutions:
                    solution = workflow_state.sub_problem_solutions[sp.id]
                    report.append(f"  Solution Length: {len(solution.content)} chars")
                    report.append(f"  Critiques: {len(solution.critique_reports)}")
                    report.append(f"  Verifications: {len(solution.verification_reports)}")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def export_to_json(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Export workflow state to JSON format."""
        return {
            "workflow_id": workflow_state.workflow_id,
            "problem_statement": workflow_state.problem_statement,
            "status": workflow_state.status,
            "current_stage": workflow_state.current_stage,
            "progress": workflow_state.progress,
            "start_time": workflow_state.start_time,
            "end_time": workflow_state.end_time,
            "refinement_loop_count": workflow_state.refinement_loop_count,
            "solved_sub_problems": list(workflow_state.solved_sub_problem_ids),
            "total_sub_problems": len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0
        }
    
    def generate_comprehensive_report(
        self,
        workflow_state: WorkflowState,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report with all data and visualizations.
        
        Args:
            workflow_state: Workflow state
            include_visualizations: Whether to include visualization figures
            
        Returns:
            Dictionary containing report data
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "workflow_id": workflow_state.workflow_id,
                "report_type": "comprehensive"
            },
            "executive_summary": self.generate_executive_summary(workflow_state),
            "detailed_report": self.generate_detailed_report(workflow_state),
            "json_data": self.export_to_json(workflow_state)
        }
        
        if include_visualizations:
            visualizer = get_advanced_visualizer()
            report["visualizations"] = {
                "dependency_graph": visualizer.dependency_visualizer.create_interactive_graph(
                    workflow_state.decomposition_plan.sub_problems
                ) if workflow_state.decomposition_plan else None,
                "workflow_flow": visualizer.create_workflow_flow_diagram(workflow_state),
                "performance_metrics": visualizer.create_performance_metrics_chart(workflow_state),
                "quality_scores": visualizer.create_quality_scores_chart(workflow_state),
                "dashboard": visualizer.create_performance_dashboard(workflow_state)
            }
        
        return report
    
    def export_report(
        self,
        workflow_state: WorkflowState,
        format: str = 'html'
    ) -> bytes:
        """
        Export report to specified format.
        
        Args:
            workflow_state: Workflow state
            format: Export format ('html', 'pdf', 'json')
            
        Returns:
            Report bytes
        """
        if format == 'json':
            import json
            data = self.generate_comprehensive_report(workflow_state, include_visualizations=False)
            return json.dumps(data, indent=2).encode('utf-8')
        
        elif format == 'html':
            # Generate HTML report with embedded visualizations
            report = self.generate_comprehensive_report(workflow_state, include_visualizations=True)
            
            html_parts = [
                "<html><head>",
                "<title>Workflow Report - {}</title>".format(workflow_state.workflow_id),
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 20px; }",
                "h1 { color: #333; }",
                "h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }",
                "pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }",
                ".metric { display: inline-block; margin: 10px; padding: 15px; background: #e3f2fd; border-radius: 5px; }",
                "</style>",
                "</head><body>",
                "<h1>Workflow Comprehensive Report</h1>",
                "<p><strong>Generated:</strong> {}</p>".format(report["metadata"]["generated_at"]),
                "<h2>Executive Summary</h2>",
                "<pre>{}</pre>".format(report["executive_summary"]),
                "<h2>Detailed Report</h2>",
                "<pre>{}</pre>".format(report["detailed_report"]),
                "</body></html>"
            ]
            
            return "".join(html_parts).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global instances
_global_visualizer: Optional[AdvancedVisualizer] = None
_global_report_generator: Optional[ReportGenerator] = None


def get_advanced_visualizer() -> AdvancedVisualizer:
    """Get or create the global advanced visualizer."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = AdvancedVisualizer()
    return _global_visualizer


def get_report_generator() -> ReportGenerator:
    """Get or create the global report generator."""
    global _global_report_generator
    if _global_report_generator is None:
        _global_report_generator = ReportGenerator()
    return _global_report_generator
