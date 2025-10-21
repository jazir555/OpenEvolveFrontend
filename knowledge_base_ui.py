"""
Knowledge Base UI Module

This module provides the user interface for exploring, managing, and visualizing
the knowledge base of artifacts extracted from workflow executions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from workflow_structures import KnowledgeArtifact
from knowledge_manager import KnowledgeManager


class KnowledgeBaseUI:
    """Manages the knowledge base user interface."""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        """
        Initialize the Knowledge Base UI.
        
        Args:
            knowledge_manager: KnowledgeManager instance for accessing artifacts
        """
        self.knowledge_manager = knowledge_manager
    
    def render_knowledge_base(self):
        """Render the main knowledge base interface."""
        st.title("🧠 Knowledge Base")
        
        # Create tabs for different views
        tabs = st.tabs([
            "Browse Artifacts",
            "Search & Filter",
            "Knowledge Graph",
            "Recommendations",
            "Import/Export"
        ])
        
        with tabs[0]:
            self._render_browse_artifacts()
        
        with tabs[1]:
            self._render_search_filter()
        
        with tabs[2]:
            self._render_knowledge_graph()
        
        with tabs[3]:
            self._render_recommendations()
        
        with tabs[4]:
            self._render_import_export()
    
    def _render_browse_artifacts(self):
        """Render artifact browser."""
        st.header("Browse Knowledge Artifacts")
        
        all_artifacts = self.knowledge_manager.get_all_artifacts()
        
        if not all_artifacts:
            st.info("No knowledge artifacts available. Complete some workflows to build the knowledge base.")
            return
        
        # Display summary
        st.write(f"Total artifacts: **{len(all_artifacts)}**")
        
        # Group by type
        artifact_types = {}
        for artifact in all_artifacts:
            artifact_types[artifact.artifact_type] = artifact_types.get(artifact.artifact_type, 0) + 1
        
        st.write("Artifacts by type:")
        for artifact_type, count in artifact_types.items():
            st.write(f"- {artifact_type}: {count}")
        
        st.divider()
        
        # Artifact list with expandable details
        st.subheader("Artifact List")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Most Recent", "Most Used", "Most Effective", "Type"]
        )
        
        # Sort artifacts
        if sort_by == "Most Recent":
            sorted_artifacts = sorted(all_artifacts, key=lambda x: x.extraction_timestamp, reverse=True)
        elif sort_by == "Most Used":
            sorted_artifacts = sorted(all_artifacts, key=lambda x: x.usage_count, reverse=True)
        elif sort_by == "Most Effective":
            sorted_artifacts = sorted(all_artifacts, key=lambda x: x.effectiveness_score, reverse=True)
        else:  # Type
            sorted_artifacts = sorted(all_artifacts, key=lambda x: x.artifact_type)
        
        # Display artifacts
        for artifact in sorted_artifacts:
            with st.expander(
                f"🔹 {artifact.artifact_type} | ID: {artifact.id[:8]} | "
                f"Effectiveness: {artifact.effectiveness_score:.2f} | "
                f"Used: {artifact.usage_count} times"
            ):
                self._render_artifact_details(artifact)
    
    def _render_artifact_details(self, artifact: KnowledgeArtifact):
        """Render detailed view of a single artifact."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Metadata:**")
            st.write(f"- ID: `{artifact.id}`")
            st.write(f"- Type: {artifact.artifact_type}")
            st.write(f"- Source Workflow: `{artifact.source_workflow_id[:8]}`")
            st.write(f"- Extracted: {datetime.fromtimestamp(artifact.extraction_timestamp).strftime('%Y-%m-%d %H:%M')}")
            st.write(f"- Domain: {artifact.domain or 'N/A'}")
            st.write(f"- Problem Type: {artifact.problem_type or 'N/A'}")
        
        with col2:
            st.write("**Performance:**")
            st.write(f"- Usage Count: {artifact.usage_count}")
            st.write(f"- Effectiveness Score: {artifact.effectiveness_score:.2f}")
            
            # Effectiveness indicator
            if artifact.effectiveness_score >= 0.8:
                st.success("🟢 Highly Effective")
            elif artifact.effectiveness_score >= 0.5:
                st.info("🟡 Moderately Effective")
            else:
                st.warning("🔴 Low Effectiveness")
        
        st.write("**Content:**")
        st.json(artifact.content)
        
        # Related artifacts
        if artifact.related_artifacts:
            st.write("**Related Artifacts:**")
            for related_id in artifact.related_artifacts:
                st.write(f"- `{related_id[:8]}`")
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Delete Artifact", key=f"delete_{artifact.id}"):
                if self.knowledge_manager.delete_artifact(artifact.id):
                    st.success("Artifact deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete artifact.")
        
        with col2:
            if st.button(f"Export Artifact", key=f"export_{artifact.id}"):
                artifact_json = json.dumps({
                    'id': artifact.id,
                    'artifact_type': artifact.artifact_type,
                    'content': artifact.content,
                    'source_workflow_id': artifact.source_workflow_id,
                    'extraction_timestamp': artifact.extraction_timestamp,
                    'domain': artifact.domain,
                    'problem_type': artifact.problem_type,
                    'usage_count': artifact.usage_count,
                    'effectiveness_score': artifact.effectiveness_score,
                    'related_artifacts': artifact.related_artifacts
                }, indent=2)
                st.download_button(
                    "Download JSON",
                    artifact_json,
                    file_name=f"artifact_{artifact.id[:8]}.json",
                    mime="application/json"
                )
    
    def _render_search_filter(self):
        """Render search and filter interface."""
        st.header("Search & Filter Artifacts")
        
        # Search inputs
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input(
                "Search by keywords",
                placeholder="Enter keywords to search in artifact content..."
            )
        
        with col2:
            artifact_type_filter = st.multiselect(
                "Filter by type",
                ["solution_pattern", "problem_solution_mapping", "critique_insight", 
                 "team_performance", "gauntlet_effectiveness"],
                default=[]
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            domain_filter = st.text_input("Filter by domain", placeholder="e.g., Software Development")
        
        with col4:
            min_effectiveness = st.slider(
                "Minimum effectiveness score",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Search button
        if st.button("Search", type="primary"):
            # Perform search
            results = self._search_artifacts(
                search_query,
                artifact_type_filter,
                domain_filter,
                min_effectiveness
            )
            
            if results:
                st.success(f"Found {len(results)} matching artifacts")
                
                # Display results
                for artifact in results:
                    with st.expander(
                        f"🔹 {artifact.artifact_type} | ID: {artifact.id[:8]} | "
                        f"Effectiveness: {artifact.effectiveness_score:.2f}"
                    ):
                        self._render_artifact_details(artifact)
            else:
                st.info("No artifacts match your search criteria.")
    
    def _search_artifacts(
        self,
        query: str,
        artifact_types: List[str],
        domain: str,
        min_effectiveness: float
    ) -> List[KnowledgeArtifact]:
        """Search artifacts based on criteria."""
        all_artifacts = self.knowledge_manager.get_all_artifacts()
        results = []
        
        for artifact in all_artifacts:
            # Filter by type
            if artifact_types and artifact.artifact_type not in artifact_types:
                continue
            
            # Filter by domain
            if domain and (not artifact.domain or domain.lower() not in artifact.domain.lower()):
                continue
            
            # Filter by effectiveness
            if artifact.effectiveness_score < min_effectiveness:
                continue
            
            # Filter by query
            if query:
                artifact_text = json.dumps(artifact.content).lower()
                if query.lower() not in artifact_text:
                    continue
            
            results.append(artifact)
        
        return results
    
    def _render_knowledge_graph(self):
        """Render knowledge graph visualization."""
        st.header("Knowledge Graph")
        
        all_artifacts = self.knowledge_manager.get_all_artifacts()
        
        if not all_artifacts:
            st.info("No artifacts available to visualize.")
            return
        
        st.write("Visualizing relationships between knowledge artifacts...")
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for artifact in all_artifacts:
            G.add_node(
                artifact.id[:8],
                type=artifact.artifact_type,
                effectiveness=artifact.effectiveness_score,
                usage=artifact.usage_count
            )
        
        # Add edges for related artifacts
        for artifact in all_artifacts:
            for related_id in artifact.related_artifacts:
                if related_id in [a.id for a in all_artifacts]:
                    G.add_edge(artifact.id[:8], related_id[:8])
        
        # Create visualization
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Prepare edge trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Prepare node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = G.nodes[node]
                node_text.append(
                    f"ID: {node}<br>"
                    f"Type: {node_data['type']}<br>"
                    f"Effectiveness: {node_data['effectiveness']:.2f}<br>"
                    f"Usage: {node_data['usage']}"
                )
                node_color.append(node_data['effectiveness'])
                node_size.append(10 + node_data['usage'] * 2)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_color,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title='Effectiveness',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2
                )
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Knowledge Artifact Relationships',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph statistics
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Nodes", len(G.nodes()))
            
            with col2:
                st.metric("Total Edges", len(G.edges()))
            
            with col3:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Connections", f"{avg_degree:.1f}")
        else:
            st.info("No relationships to visualize yet.")
    
    def _render_recommendations(self):
        """Render recommendations based on knowledge base."""
        st.header("Knowledge-Based Recommendations")
        
        st.write("Get recommendations for solving new problems based on learned patterns.")
        
        # Input for new problem
        problem_statement = st.text_area(
            "Describe your problem",
            placeholder="Enter a problem description to get recommendations...",
            height=150
        )
        
        domain = st.text_input("Domain (optional)", placeholder="e.g., Software Development")
        
        if st.button("Get Recommendations", type="primary"):
            if not problem_statement:
                st.warning("Please enter a problem description.")
                return
            
            with st.spinner("Analyzing problem and searching knowledge base..."):
                suggestions = self.knowledge_manager.apply_learned_patterns(
                    problem_statement,
                    domain=domain if domain else None
                )
            
            # Display recommendations
            if any(suggestions.values()):
                st.success("Found recommendations based on past experience!")
                
                # Recommended approaches
                if suggestions["recommended_approaches"]:
                    st.subheader("🎯 Recommended Approaches")
                    for i, approach in enumerate(suggestions["recommended_approaches"], 1):
                        with st.expander(f"Approach {i} (Effectiveness: {approach['effectiveness']:.2f})"):
                            st.write(f"**Approach:** {approach['approach']}")
                            st.write(f"**Source:** `{approach['source'][:8]}`")
                
                # Similar problems
                if suggestions["similar_problems"]:
                    st.subheader("📚 Similar Problems Solved")
                    for i, similar in enumerate(suggestions["similar_problems"], 1):
                        with st.expander(f"Similar Problem {i}"):
                            st.write(f"**Problem:** {similar['problem'][:200]}...")
                            st.write(f"**Decomposition Strategy:**")
                            st.json(similar['decomposition_strategy'])
                            st.write(f"**Source:** `{similar['source'][:8]}`")
                
                # Team recommendations
                if suggestions["team_recommendations"]:
                    st.subheader("👥 Recommended Teams")
                    for i, team_rec in enumerate(suggestions["team_recommendations"], 1):
                        with st.expander(f"Team: {team_rec['team_name']}"):
                            st.write(f"**Performance:**")
                            st.json(team_rec['performance'])
                            st.write(f"**Effectiveness:** {team_rec['effectiveness']:.2f}")
                
                # Gauntlet recommendations
                if suggestions["gauntlet_recommendations"]:
                    st.subheader("🛡️ Recommended Gauntlets")
                    for i, gauntlet_rec in enumerate(suggestions["gauntlet_recommendations"], 1):
                        with st.expander(f"Gauntlet: {gauntlet_rec['gauntlet_name']}"):
                            st.write(f"**Effectiveness:**")
                            st.json(gauntlet_rec['effectiveness'])
                            st.write(f"**Score:** {gauntlet_rec['score']:.2f}")
            else:
                st.info("No recommendations found. The knowledge base may not have relevant experience yet.")
    
    def _render_import_export(self):
        """Render import/export interface."""
        st.header("Import/Export Knowledge Base")
        
        # Export section
        st.subheader("📤 Export Knowledge Base")
        st.write("Export the entire knowledge base to a JSON file for backup or sharing.")
        
        if st.button("Export Knowledge Base"):
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            # Export to temp file
            self.knowledge_manager.export_knowledge_base(temp_path)
            
            # Read and offer download
            with open(temp_path, 'r') as f:
                knowledge_json = f.read()
            
            st.download_button(
                "Download Knowledge Base",
                knowledge_json,
                file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            st.success("Knowledge base exported successfully!")
        
        st.divider()
        
        # Import section
        st.subheader("📥 Import Knowledge Base")
        st.write("Import a knowledge base from a JSON file.")
        
        uploaded_file = st.file_uploader("Choose a knowledge base file", type=['json'])
        
        if uploaded_file is not None:
            if st.button("Import Knowledge Base", type="primary"):
                import tempfile
                import os
                
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as f:
                    f.write(uploaded_file.getvalue())
                    temp_path = f.name
                
                try:
                    # Import from temp file
                    self.knowledge_manager.import_knowledge_base(temp_path)
                    st.success("Knowledge base imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to import knowledge base: {e}")
                finally:
                    # Clean up temp file
                    os.unlink(temp_path)
        
        st.divider()
        
        # Clear knowledge base
        st.subheader("🗑️ Clear Knowledge Base")
        st.warning("⚠️ This will permanently delete all knowledge artifacts!")
        
        if st.button("Clear All Artifacts", type="secondary"):
            if st.checkbox("I understand this action cannot be undone"):
                self.knowledge_manager.clear_all_artifacts()
                st.success("Knowledge base cleared.")
                st.rerun()


def render_knowledge_base():
    """Convenience function to render the knowledge base UI."""
    # Initialize knowledge manager
    knowledge_manager = KnowledgeManager()
    
    # Create and render UI
    kb_ui = KnowledgeBaseUI(knowledge_manager)
    kb_ui.render_knowledge_base()
