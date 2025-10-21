"""
Dependency Visualizer Module

This module provides visualization and analysis of sub-problem dependencies
in the decomposition workflow, including circular dependency detection and
execution order suggestions.
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from workflow_structures import DecompositionPlan, SubProblem


class DependencyVisualizer:
    """Visualizes and analyzes sub-problem dependencies."""
    
    def __init__(self, plan: DecompositionPlan):
        """
        Initialize the dependency visualizer.
        
        Args:
            plan: The decomposition plan to visualize
        """
        self.plan = plan
        self.graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build a directed graph from sub-problem dependencies."""
        G = nx.DiGraph()
        
        # Add nodes
        for sp in self.plan.sub_problems:
            G.add_node(
                sp.id,
                description=sp.description,
                complexity=sp.ai_suggested_complexity_score,
                solver_team=sp.solver_team_name,
                status=sp.status
            )
        
        # Add edges (dependencies)
        for sp in self.plan.sub_problems:
            for dep in sp.dependencies:
                if dep in [s.id for s in self.plan.sub_problems]:
                    G.add_edge(dep, sp.id)  # dep -> sp (dep must be solved first)
        
        return G
    
    def render_dependency_graph(self):
        """Render interactive dependency graph visualization."""
        st.subheader("📊 Dependency Graph")
        
        if not self.plan.sub_problems:
            st.info("No sub-problems to visualize.")
            return
        
        # Check for issues
        cycles = self.detect_circular_dependencies()
        if cycles:
            st.error(f"⚠️ Circular dependencies detected! ({len(cycles)} cycles)")
            with st.expander("View Circular Dependencies"):
                for i, cycle in enumerate(cycles, 1):
                    st.write(f"**Cycle {i}:** {' → '.join(cycle)}")
        
        # Calculate layout
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        except:
            pos = nx.random_layout(self.graph, seed=42)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Add arrow annotations
        annotations = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate arrow position (80% along the edge)
            ax = x0 * 0.2 + x1 * 0.8
            ay = y0 * 0.2 + y1 * 0.8
            
            annotations.append(
                dict(
                    x=x1, y=y1,
                    ax=ax, ay=ay,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#888'
                )
            )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            desc = node_data['description'][:50] + "..." if len(node_data['description']) > 50 else node_data['description']
            
            node_text.append(
                f"<b>{node}</b><br>"
                f"{desc}<br>"
                f"Complexity: {node_data['complexity']}<br>"
                f"Team: {node_data['solver_team'] or 'Not assigned'}<br>"
                f"Status: {node_data['status']}"
            )
            
            # Color by status
            status_colors = {
                'pending': '#FFA500',
                'in_progress': '#4169E1',
                'solved': '#32CD32',
                'failed': '#DC143C',
                'requires_rework': '#FF6347'
            }
            node_color.append(status_colors.get(node_data['status'], '#808080'))
            
            # Size by complexity
            node_size.append(15 + node_data['complexity'] * 3)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in self.graph.nodes()],
            hovertext=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Sub-Problem Dependency Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                annotations=annotations
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.write("**Status Colors:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("🟠 Pending")
        with col2:
            st.markdown("🔵 In Progress")
        with col3:
            st.markdown("🟢 Solved")
        with col4:
            st.markdown("🔴 Failed")
        with col5:
            st.markdown("🟡 Requires Rework")
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the plan.
        
        Returns:
            List of cycles (each cycle is a list of sub-problem IDs)
        """
        try:
            # NetworkX will raise an exception if there are cycles
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []
    
    def suggest_execution_order(self) -> List[str]:
        """
        Suggest execution order using topological sort.
        
        Returns:
            List of sub-problem IDs in suggested execution order
        """
        try:
            # Topological sort
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Graph has cycles, return original order
            return [sp.id for sp in self.plan.sub_problems]
    
    def get_dependency_statistics(self) -> Dict[str, Any]:
        """Get statistics about dependencies."""
        stats = {
            'total_sub_problems': len(self.plan.sub_problems),
            'total_dependencies': self.graph.number_of_edges(),
            'avg_dependencies_per_sub_problem': 0,
            'max_dependencies': 0,
            'sub_problems_with_no_dependencies': 0,
            'sub_problems_with_no_dependents': 0,
            'longest_dependency_chain': 0,
            'has_circular_dependencies': False,
            'number_of_cycles': 0
        }
        
        if not self.plan.sub_problems:
            return stats
        
        # Calculate statistics
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        stats['avg_dependencies_per_sub_problem'] = sum(in_degrees.values()) / len(in_degrees)
        stats['max_dependencies'] = max(in_degrees.values()) if in_degrees else 0
        stats['sub_problems_with_no_dependencies'] = sum(1 for d in in_degrees.values() if d == 0)
        stats['sub_problems_with_no_dependents'] = sum(1 for d in out_degrees.values() if d == 0)
        
        # Longest path
        try:
            stats['longest_dependency_chain'] = nx.dag_longest_path_length(self.graph)
        except:
            stats['longest_dependency_chain'] = 0
        
        # Circular dependencies
        cycles = self.detect_circular_dependencies()
        stats['has_circular_dependencies'] = len(cycles) > 0
        stats['number_of_cycles'] = len(cycles)
        
        return stats
    
    def render_dependency_analysis(self):
        """Render dependency analysis panel."""
        st.subheader("📈 Dependency Analysis")
        
        stats = self.get_dependency_statistics()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sub-Problems", stats['total_sub_problems'])
            st.metric("Total Dependencies", stats['total_dependencies'])
        
        with col2:
            st.metric("Avg Dependencies", f"{stats['avg_dependencies_per_sub_problem']:.1f}")
            st.metric("Max Dependencies", stats['max_dependencies'])
        
        with col3:
            st.metric("No Dependencies", stats['sub_problems_with_no_dependencies'])
            st.metric("No Dependents", stats['sub_problems_with_no_dependents'])
        
        with col4:
            st.metric("Longest Chain", stats['longest_dependency_chain'])
            if stats['has_circular_dependencies']:
                st.error(f"⚠️ {stats['number_of_cycles']} Cycles")
            else:
                st.success("✓ No Cycles")
        
        # Execution order
        st.subheader("🔄 Suggested Execution Order")
        
        if stats['has_circular_dependencies']:
            st.error("Cannot suggest execution order due to circular dependencies. Please resolve cycles first.")
        else:
            execution_order = self.suggest_execution_order()
            
            st.write("Sub-problems should be executed in the following order:")
            
            # Display as a flow
            for i, sp_id in enumerate(execution_order, 1):
                sp = next((s for s in self.plan.sub_problems if s.id == sp_id), None)
                if sp:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{i}.**")
                    with col2:
                        st.write(f"**{sp_id}**: {sp.description[:80]}...")
            
            # Export execution order
            if st.button("Copy Execution Order"):
                order_text = "\n".join(f"{i}. {sp_id}" for i, sp_id in enumerate(execution_order, 1))
                st.code(order_text)
    
    def render_dependency_matrix(self):
        """Render dependency matrix visualization."""
        st.subheader("📋 Dependency Matrix")
        
        if not self.plan.sub_problems:
            st.info("No sub-problems to display.")
            return
        
        # Build adjacency matrix
        sp_ids = [sp.id for sp in self.plan.sub_problems]
        n = len(sp_ids)
        
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for i, sp in enumerate(self.plan.sub_problems):
            for dep in sp.dependencies:
                if dep in sp_ids:
                    j = sp_ids.index(dep)
                    matrix[i][j] = 1
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=sp_ids,
            y=sp_ids,
            colorscale=[[0, 'white'], [1, 'blue']],
            showscale=False,
            hovertemplate='%{y} depends on %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Dependency Matrix (Row depends on Column)',
            xaxis_title='Dependency',
            yaxis_title='Sub-Problem',
            height=max(400, n * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_dependency_visualization(plan: DecompositionPlan):
    """
    Convenience function to render dependency visualization.
    
    Args:
        plan: The decomposition plan to visualize
    """
    visualizer = DependencyVisualizer(plan)
    
    # Create tabs for different views
    tabs = st.tabs(["Graph View", "Analysis", "Matrix View"])
    
    with tabs[0]:
        visualizer.render_dependency_graph()
    
    with tabs[1]:
        visualizer.render_dependency_analysis()
    
    with tabs[2]:
        visualizer.render_dependency_matrix()


def detect_and_fix_circular_dependencies(plan: DecompositionPlan) -> Tuple[bool, List[str], DecompositionPlan]:
    """
    Detect circular dependencies and suggest fixes.
    
    Args:
        plan: The decomposition plan to check
        
    Returns:
        Tuple of (has_cycles, suggestions, fixed_plan)
    """
    visualizer = DependencyVisualizer(plan)
    cycles = visualizer.detect_circular_dependencies()
    
    if not cycles:
        return False, [], plan
    
    suggestions = []
    fixed_plan = plan
    
    # Suggest removing dependencies to break cycles
    for cycle in cycles:
        # Suggest removing the last dependency in the cycle
        if len(cycle) >= 2:
            from_sp = cycle[-1]
            to_sp = cycle[0]
            suggestions.append(
                f"Remove dependency: {from_sp} → {to_sp} (breaks cycle: {' → '.join(cycle)})"
            )
    
    return True, suggestions, fixed_plan


def validate_dependencies(plan: DecompositionPlan) -> Tuple[bool, List[str]]:
    """
    Validate all dependencies in a plan.
    
    Args:
        plan: The decomposition plan to validate
        
    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []
    valid_ids = {sp.id for sp in plan.sub_problems}
    
    # Check for invalid dependency references
    for sp in plan.sub_problems:
        for dep in sp.dependencies:
            if dep not in valid_ids:
                issues.append(f"Sub-problem {sp.id} has invalid dependency: {dep}")
    
    # Check for circular dependencies
    visualizer = DependencyVisualizer(plan)
    cycles = visualizer.detect_circular_dependencies()
    
    if cycles:
        for cycle in cycles:
            issues.append(f"Circular dependency: {' → '.join(cycle)}")
    
    # Check for self-dependencies
    for sp in plan.sub_problems:
        if sp.id in sp.dependencies:
            issues.append(f"Sub-problem {sp.id} depends on itself")
    
    return len(issues) == 0, issues
