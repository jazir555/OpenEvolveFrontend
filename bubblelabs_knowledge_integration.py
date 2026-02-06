"""
BubbleLabs Knowledge Engine Integration

This module provides comprehensive integration between BubbleLabs UI and the
OpenEvolve Knowledge Engine, including:

- Bedrock knowledge base querying
- Knowledge graph visualization
- Knowledge extraction workflows
- Knowledge search and retrieval
- Entity and relationship exploration

Author: OpenEvolve Integration Team
Created: 2026-01-03
Status: Production Ready
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ui_shim import ui as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Knowledge Engine imports
from knowledge_engine.engine import KnowledgeEngine
from knowledge_engine.core import KnowledgeState, EntityKnowledgeGraph
from knowledge_engine.bedrock_kb import BedrockKnowledgeBaseClient
from integrations import IntegrationFactory

# Logging configuration
logger = logging.getLogger(__name__)

# =============================================================================
# KNOWLEDGE GRAPH VISUALIZATION COMPONENTS
# =============================================================================

class KnowledgeGraphVisualizer:
    """
    Interactive knowledge graph visualization for BubbleLabs.

    Features:
    - Network graph visualization with force-directed layout
    - Entity filtering and search
    - Relationship exploration
    - Confidence score display
    - Temporal knowledge tracking
    """

    def __init__(self):
        """Initialize the knowledge graph visualizer."""
        self.graph = nx.DiGraph()
        self.entity_colors = px.colors.qualitative.Set3
        self.relation_colors = px.colors.qualitative.Pastel1

    def build_graph_from_data(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        Build NetworkX graph from knowledge data.

        Args:
            entities: List of entity dictionaries with 'name', 'type', 'attributes'
            relationships: List of relationship dictionaries with 'source', 'relation', 'target'

        Returns:
            NetworkX DiGraph object
        """
        self.graph.clear()

        # Add nodes (entities)
        for entity in entities:
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'default')
            attributes = entity.get('attributes', {})

            self.graph.add_node(
                name,
                type=entity_type,
                **attributes
            )

        # Add edges (relationships)
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            relation = rel.get('relation', 'related_to')
            attributes = rel.get('attributes', {})

            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    **attributes
                )

        return self.graph

    def create_interactive_plot(
        self,
        layout: str = 'spring',
        node_size_multiplier: float = 1.0,
        edge_width_multiplier: float = 1.0,
        show_labels: bool = True,
        filter_entities: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> go.Figure:
        """
        Create interactive Plotly visualization of the knowledge graph.

        Args:
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'random')
            node_size_multiplier: Scale factor for node sizes
            edge_width_multiplier: Scale factor for edge widths
            show_labels: Whether to show node labels
            filter_entities: List of entity names to include (None = all)
            min_confidence: Minimum confidence score for edges

        Returns:
            Plotly figure object
        """
        # Filter graph if needed
        if filter_entities:
            subgraph = self.graph.subgraph(filter_entities)
        else:
            subgraph = self.graph

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            pos = nx.random_layout(subgraph)

        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_info.append(f"Entity: {node}<br>Type: {subgraph.nodes[node].get('type', 'N/A')}")

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_text if show_labels else [],
            textposition='middle center',
            marker=dict(
                size=[10 * node_size_multiplier] * len(node_x),
                color=[hash(subgraph.nodes[n].get('type', 'default')) % len(self.entity_colors)
                       for n in subgraph.nodes()],
                colorscale=self.entity_colors,
                line=dict(width=2, color='#888')
            ),
            hovertext=node_info,
            hoverinfo='text',
            customdata=list(subgraph.nodes())
        )

        # Extract edge positions
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            relation = edge[2].get('relation', 'related_to')
            confidence = edge[2].get('confidence', 1.0)
            edge_info.append(f"{edge[0]} -> {relation} -> {edge[1]}<br>Confidence: {confidence:.2f}")

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2 * edge_width_multiplier, color='#888'),
            hoverinfo='none'
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph Visualization',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               text="Knowledge Graph",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))

        return fig

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()),
            'strongly_connected_components': nx.number_strongly_connected_components(self.graph),
        }

    def find_shortest_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.

        Args:
            source: Source entity name
            target: Target entity name

        Returns:
            List of entity names in path or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def get_entity_neighbors(
        self,
        entity: str,
        depth: int = 1
    ) -> Dict[str, List[str]]:
        """
        Get neighbors of an entity at specified depth.

        Args:
            entity: Entity name
            depth: Neighborhood depth

        Returns:
            Dictionary with 'predecessors' and 'successors' lists
        """
        if entity not in self.graph.nodes:
            return {'predecessors': [], 'successors': []}

        predecessors = list(self.graph.predecessors(entity))
        successors = list(self.graph.successors(entity))

        return {
            'predecessors': predecessors[:depth * 10],
            'successors': successors[:depth * 10]
        }


# =============================================================================
# KNOWLEDGE QUERY INTERFACE
# =============================================================================

class KnowledgeQueryInterface:
    """
    Interface for querying multiple knowledge sources.

    Supports:
    - Bedrock Knowledge Base
    - Graphiti temporal knowledge graph
    - Elasticsearch indices
    - Local code indexes
    """

    def __init__(self, knowledge_engine: KnowledgeEngine):
        """
        Initialize the query interface.

        Args:
            knowledge_engine: KnowledgeEngine instance
        """
        self.engine = knowledge_engine
        self.query_history: List[Dict[str, Any]] = []
        self.cache: Dict[str, Any] = {}

    async def query_bedrock(
        self,
        knowledge_base_id: str,
        query_text: str,
        use_temporal_search: bool = True,
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query Bedrock Knowledge Base with optional temporal search.

        Args:
            knowledge_base_id: Bedrock KB ID
            query_text: Query text
            use_temporal_search: Include Graphiti temporal results
            num_results: Maximum number of results

        Returns:
            Query results dictionary
        """
        logger.info(f"Querying Bedrock KB '{knowledge_base_id}' with: {query_text}")

        try:
            results = await self.engine.query_bedrock_knowledge_base(
                knowledge_base_id=knowledge_base_id,
                query=query_text
            )

            # Log query
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'source': 'bedrock',
                'query': query_text,
                'results_count': len(results.get('bedrock_results', {})),
                'success': True
            })

            return results

        except Exception as e:
            logger.error(f"Bedrock query failed: {e}")
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'source': 'bedrock',
                'query': query_text,
                'error': str(e),
                'success': False
            })
            return {'error': str(e)}

    async def query_graphiti(
        self,
        query: str,
        temporal_filters: Optional[Dict[str, Any]] = None,
        num_results: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Query Graphiti temporal knowledge graph.

        Args:
            query: Search query
            temporal_filters: Optional temporal filters
            num_results: Maximum results

        Returns:
            Graphiti search results or None
        """
        logger.info(f"Querying Graphiti with: {query}")

        try:
            if not self.engine.bedrock_client:
                logger.warning("Bedrock client not initialized, cannot query Graphiti")
                return None

            results = await self.engine.bedrock_client.search_graphiti(
                query=query,
                temporal_filters=temporal_filters,
                num_results=num_results
            )

            # Log query
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'source': 'graphiti',
                'query': query,
                'results_count': len(results.get('nodes', [])) if results else 0,
                'success': results is not None
            })

            return results

        except Exception as e:
            logger.error(f"Graphiti query failed: {e}")
            return None

    async def query_local_index(
        self,
        index_path: str,
        keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Query local code index.

        Args:
            index_path: Path to index JSON file
            keyword: Search keyword

        Returns:
            List of matching file summaries
        """
        logger.info(f"Querying local index '{index_path}' for: {keyword}")

        try:
            # Load index
            index_data = self.engine.load_index(index_path)

            if not index_data:
                return []

            # Search
            results = self.engine.query_index_by_keyword(index_data, keyword)

            # Log query
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'source': 'local_index',
                'query': keyword,
                'index_path': index_path,
                'results_count': len(results),
                'success': True
            })

            return results

        except Exception as e:
            logger.error(f"Local index query failed: {e}")
            return []

    async def unified_query(
        self,
        query: str,
        sources: List[str] = ['bedrock', 'graphiti', 'local'],
        bedrock_kb_id: Optional[str] = None,
        index_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute unified query across multiple knowledge sources.

        Args:
            query: Query text
            sources: List of sources to query ('bedrock', 'graphiti', 'local')
            bedrock_kb_id: Bedrock KB ID (required if 'bedrock' in sources)
            index_path: Local index path (required if 'local' in sources)

        Returns:
            Unified results dictionary
        """
        logger.info(f"Executing unified query across {len(sources)} sources: {query}")

        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }

        # Query Bedrock
        if 'bedrock' in sources and bedrock_kb_id:
            results['sources']['bedrock'] = await self.query_bedrock(
                bedrock_kb_id, query
            )

        # Query Graphiti
        if 'graphiti' in sources:
            results['sources']['graphiti'] = await self.query_graphiti(query)

        # Query local index
        if 'local' in sources and index_path:
            results['sources']['local'] = await self.query_local_index(
                index_path, query
            )

        return results

    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get query history."""
        return self.query_history[-limit:]


# =============================================================================
# KNOWLEDGE EXTRACTION WORKFLOW
# =============================================================================

class KnowledgeExtractionWorkflow:
    """
    Workflow for extracting knowledge from documents and code.

    Features:
    - Document loading (PDF, Office, text)
    - Knowledge extraction using LLM
    - Entity and relationship extraction
    - Knowledge graph construction
    """

    def __init__(self, knowledge_engine: KnowledgeEngine):
        """
        Initialize the extraction workflow.

        Args:
            knowledge_engine: KnowledgeEngine instance
        """
        self.engine = knowledge_engine
        self.extraction_history: List[Dict[str, Any]] = []

    async def extract_from_document(
        self,
        document_path_or_url: str,
        extraction_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge from a document.

        Args:
            document_path_or_url: Path or URL to document
            extraction_config: Optional extraction configuration

        Returns:
            Extraction results with entities and relationships
        """
        logger.info(f"Extracting knowledge from: {document_path_or_url}")

        try:
            # Load document
            text_content = await self.engine.add_document(document_path_or_url)

            if not text_content:
                return {'error': 'Failed to load document'}

            # Extract entities and relationships
            entities, relationships = await self._extract_knowledge(text_content)

            # Build knowledge graph
            kg = EntityKnowledgeGraph()
            for entity in entities:
                await kg.add_entity(entity['name'], entity.get('attributes'))

            for rel in relationships:
                await kg.add_relationship(
                    rel['source'],
                    rel['relation'],
                    rel['target'],
                    rel.get('attributes')
                )

            # Log extraction
            self.extraction_history.append({
                'timestamp': datetime.now().isoformat(),
                'source': document_path_or_url,
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'success': True
            })

            return {
                'text_content': text_content[:1000],  # Preview
                'entities': entities,
                'relationships': relationships,
                'knowledge_graph': kg.to_dict(),
                'statistics': {
                    'total_entities': len(entities),
                    'total_relationships': len(relationships)
                }
            }

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {'error': str(e)}

    async def _extract_knowledge(
        self,
        text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relationships from text using LLM.

        Args:
            text: Input text

        Returns:
            Tuple of (entities, relationships)
        """
        extraction_prompt = f"""
        Extract knowledge from the following text in JSON format:

        TEXT:
        {text[:3000]}

        Extract:
        1. Entities (people, organizations, concepts, technologies)
        2. Relationships between entities

        Return format:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "Type", "attributes": {{"key": "value"}}}}
            ],
            "relationships": [
                {{"source": "Entity1", "relation": "relationship_type", "target": "Entity2", "attributes": {{}}}}
            ]
        }}

        Focus on concrete, specific entities and clear relationships.
        """

        try:
            response = await self.engine._call_llm(
                prompt=extraction_prompt,
                system_prompt="You are a knowledge extraction expert. Extract entities and relationships in structured JSON format.",
                max_tokens=2000
            )

            # Parse JSON response
            import json
            import re

            # Find JSON block
            start = response.find('{')
            end = response.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                entities = data.get('entities', [])
                relationships = data.get('relationships', [])

                return entities, relationships

            return [], []

        except Exception as e:
            logger.error(f"LLM knowledge extraction failed: {e}")
            return [], []

    def get_extraction_history(self) -> List[Dict[str, Any]]:
        """Get extraction history."""
        return self.extraction_history


# =============================================================================
# BUBBLELABS KNOWLEDGE UI COMPONENTS
# =============================================================================

class BubbleLabsKnowledgeUI:
    """
    Streamlit UI components for knowledge exploration in BubbleLabs.

    Provides:
    - Knowledge graph visualization
    - Knowledge query interface
    - Knowledge extraction workflows
    - Knowledge statistics dashboard
    """

    def __init__(self):
        """Initialize the BubbleLabs Knowledge UI."""
        self.engine = None
        self.visualizer = KnowledgeGraphVisualizer()
        self.query_interface = None
        self.extraction_workflow = None

        # Initialize session state
        if 'knowledge_engine_initialized' not in st.session_state:
            st.session_state.knowledge_engine_initialized = False

        if 'knowledge_graph_data' not in st.session_state:
            st.session_state.knowledge_graph_data = {
                'entities': [],
                'relationships': []
            }

        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

    def initialize_engine(self):
        """Initialize the Knowledge Engine if not already initialized."""
        if not st.session_state.knowledge_engine_initialized:
            with st.spinner("Initializing Knowledge Engine..."):
                try:
                    self.engine = KnowledgeEngine()
                    self.query_interface = KnowledgeQueryInterface(self.engine)
                    self.extraction_workflow = KnowledgeExtractionWorkflow(self.engine)
                    st.session_state.knowledge_engine_initialized = True
                    st.success("Knowledge Engine initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize Knowledge Engine: {e}")
                    logger.error(f"Knowledge Engine initialization failed: {e}")

    def render_knowledge_explorer(self):
        """Render the main knowledge exploration interface."""
        st.markdown("## 🔍 Knowledge Explorer")

        # Initialize engine if needed
        if not st.session_state.knowledge_engine_initialized:
            self.initialize_engine()

        if not self.engine:
            st.warning("Knowledge Engine not available. Please check configuration.")
            return

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔎 Query Knowledge",
            "📊 Knowledge Graph",
            "📄 Extract Knowledge",
            "📈 Statistics"
        ])

        with tab1:
            self.render_query_interface()

        with tab2:
            self.render_graph_visualization()

        with tab3:
            self.render_extraction_workflow()

        with tab4:
            self.render_statistics_dashboard()

    def render_query_interface(self):
        """Render knowledge query interface."""
        st.markdown("### Query Knowledge Bases")

        # Query input
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., How does the decomposition workflow handle adversarial validation?",
            key='knowledge_query_input'
        )

        # Source selection
        col1, col2, col3 = st.columns(3)
        with col1:
            use_bedrock = st.checkbox("Bedrock KB", value=True)
        with col2:
            use_graphiti = st.checkbox("Graphiti (Temporal)", value=True)
        with col3:
            use_local = st.checkbox("Local Index", value=False)

        # Configuration
        if use_bedrock:
            bedrock_kb_id = st.text_input(
                "Bedrock Knowledge Base ID:",
                value="",
                key='bedrock_kb_id'
            )

        if use_local:
            index_path = st.text_input(
                "Local Index Path:",
                value="knowledge_index",
                key='local_index_path'
            )

        # Query button
        if st.button("🔍 Execute Query", type="primary"):
            if not query:
                st.warning("Please enter a query")
                return

            with st.spinner("Querying knowledge bases..."):
                sources = []
                if use_bedrock and bedrock_kb_id:
                    sources.append('bedrock')
                if use_graphiti:
                    sources.append('graphiti')
                if use_local:
                    sources.append('local')

                results = asyncio.run(self.query_interface.unified_query(
                    query=query,
                    sources=sources,
                    bedrock_kb_id=bedrock_kb_id if use_bedrock else None,
                    index_path=index_path if use_local else None
                ))

                # Display results
                st.session_state.query_history.append(results)
                self._display_query_results(results)

        # Display query history
        if st.session_state.query_history:
            st.markdown("#### Query History")
            for i, hist_result in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {hist_result['query'][:50]}..."):
                    st.json(hist_result)

    def _display_query_results(self, results: Dict[str, Any]):
        """Display query results."""
        st.markdown("#### Results")

        # Bedrock results
        if 'bedrock' in results.get('sources', {}):
            bedrock_results = results['sources']['bedrock']
            st.markdown("##### Bedrock Knowledge Base")

            if 'merged_context' in bedrock_results:
                st.info(bedrock_results['merged_context'])

            if 'temporal_metadata' in bedrock_results:
                with st.expander("Temporal Metadata"):
                    st.json(bedrock_results['temporal_metadata'])

        # Graphiti results
        if 'graphiti' in results.get('sources', {}):
            graphiti_results = results['sources']['graphiti']
            if graphiti_results:
                st.markdown("##### Graphiti Temporal Knowledge Graph")

                nodes = graphiti_results.get('nodes', [])
                edges = graphiti_results.get('edges', [])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entities Found", len(nodes))
                with col2:
                    st.metric("Relationships Found", len(edges))

                if nodes:
                    st.markdown("**Related Entities:**")
                    for node in nodes[:10]:
                        st.write(f"- **{node.get('name')}**: {node.get('summary', '')[:100]}")

                if edges:
                    st.markdown("**Relationships:**")
                    for edge in edges[:10]:
                        st.write(f"- {edge.get('fact', 'Unknown relationship')}")

        # Local index results
        if 'local' in results.get('sources', {}):
            local_results = results['sources']['local']
            if local_results:
                st.markdown("##### Local Index Results")
                st.metric("Files Found", len(local_results))

                for result in local_results[:10]:
                    with st.expander(f"📄 {result['file_path']}"):
                        st.write(f"**Type:** {result['file_type']}")
                        st.write(f"**Summary:** {result['summary']}")
                        st.write(f"**Key Concepts:** {', '.join(result['key_concepts'][:5])}")

    def render_graph_visualization(self):
        """Render knowledge graph visualization."""
        st.markdown("### Knowledge Graph Visualization")

        # Load sample data or use extracted data
        if st.session_state.knowledge_graph_data['entities']:
            entities = st.session_state.knowledge_graph_data['entities']
            relationships = st.session_state.knowledge_graph_data['relationships']
        else:
            st.info("No knowledge graph data available. Extract knowledge from a document first.")
            return

        # Build graph
        self.visualizer.build_graph_from_data(entities, relationships)

        # Visualization options
        col1, col2, col3 = st.columns(3)
        with col1:
            layout = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"])
        with col2:
            show_labels = st.checkbox("Show Labels", value=True)
        with col3:
            node_size = st.slider("Node Size", 0.5, 3.0, 1.0, 0.1)

        # Create visualization
        fig = self.visualizer.create_interactive_plot(
            layout=layout,
            node_size_multiplier=node_size,
            show_labels=show_labels
        )

        st.plotly_chart(fig, use_container_width=True)

        # Graph statistics
        stats = self.visualizer.get_graph_statistics()
        st.markdown("#### Graph Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entities", stats['total_nodes'])
        with col2:
            st.metric("Total Relationships", stats['total_edges'])
        with col3:
            st.metric("Graph Density", f"{stats['density']:.4f}")
        with col4:
            st.metric("Connected", "Yes" if stats['is_connected'] else "No")

        # Entity search
        st.markdown("#### Entity Exploration")
        entity_list = [e['name'] for e in entities]
        if entity_list:
            selected_entity = st.selectbox("Select entity to explore", entity_list)

            if selected_entity:
                neighbors = self.visualizer.get_entity_neighbors(selected_entity)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Predecessors (points to entity):**")
                    for pred in neighbors['predecessors']:
                        st.write(f"- {pred}")

                with col2:
                    st.markdown("**Successors (entity points to):**")
                    for succ in neighbors['successors']:
                        st.write(f"- {succ}")

    def render_extraction_workflow(self):
        """Render knowledge extraction workflow."""
        st.markdown("### Knowledge Extraction")

        # Input source
        source_type = st.radio(
            "Select input source:",
            ["URL", "Local File", "Text Input"]
        )

        document_source = None

        if source_type == "URL":
            document_source = st.text_input(
                "Enter document URL:",
                placeholder="https://arxiv.org/pdf/2301.07041",
                key='doc_url'
            )
        elif source_type == "Local File":
            uploaded_file = st.file_uploader(
                "Upload document (PDF, DOCX, TXT):",
                type=['pdf', 'docx', 'txt'],
                key='doc_upload'
            )
            if uploaded_file:
                # Save uploaded file temporarily
                temp_dir = Path("temp_docs")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_file.name

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                document_source = str(temp_path)
        else:
            text_input = st.text_area(
                "Enter text to extract knowledge from:",
                height=200,
                key='text_input'
            )

        # Extract button
        if st.button("🚀 Extract Knowledge", type="primary"):
            if source_type == "Text Input" and not text_input:
                st.warning("Please enter some text")
                return
            elif source_type != "Text Input" and not document_source:
                st.warning("Please provide a document source")
                return

            with st.spinner("Extracting knowledge..."):
                if source_type == "Text Input":
                    # Direct text extraction
                    entities, relationships = asyncio.run(
                        self.extraction_workflow._extract_knowledge(text_input)
                    )

                    results = {
                        'entities': entities,
                        'relationships': relationships,
                        'statistics': {
                            'total_entities': len(entities),
                            'total_relationships': len(relationships)
                        }
                    }
                else:
                    # Document-based extraction
                    results = asyncio.run(
                        self.extraction_workflow.extract_from_document(document_source)
                    )

                # Store results in session state
                if 'error' not in results:
                    st.session_state.knowledge_graph_data = {
                        'entities': results.get('entities', []),
                        'relationships': results.get('relationships', [])
                    }

                    # Display results
                    st.success("Knowledge extraction completed!")

                    st.markdown("#### Extraction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Entities Extracted", results['statistics']['total_entities'])
                    with col2:
                        st.metric("Relationships Extracted", results['statistics']['total_relationships'])

                    # Show entities
                    if results['entities']:
                        st.markdown("**Entities:**")
                        for entity in results['entities'][:20]:
                            st.write(f"- **{entity['name']}** ({entity.get('type', 'Unknown')})")

                    # Show relationships
                    if results['relationships']:
                        st.markdown("**Relationships:**")
                        for rel in results['relationships'][:20]:
                            st.write(f"- {rel['source']} -> {rel['relation']} -> {rel['target']}")
                else:
                    st.error(f"Extraction failed: {results.get('error')}")

    def render_statistics_dashboard(self):
        """Render knowledge statistics dashboard."""
        st.markdown("### Knowledge Statistics Dashboard")

        # Overall statistics
        st.markdown("#### Knowledge Base Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Entities",
                len(st.session_state.knowledge_graph_data['entities'])
            )

        with col2:
            st.metric(
                "Total Relationships",
                len(st.session_state.knowledge_graph_data['relationships'])
            )

        with col3:
            st.metric(
                "Queries Executed",
                len(st.session_state.query_history)
            )

        with col4:
            if self.engine:
                st.metric("Engine Status", "[OK] Active")
            else:
                st.metric("Engine Status", "[FAIL] Inactive")

        # Entity type distribution
        if st.session_state.knowledge_graph_data['entities']:
            entities = st.session_state.knowledge_graph_data['entities']

            # Count entity types
            type_counts = {}
            for entity in entities:
                entity_type = entity.get('type', 'Unknown')
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

            if type_counts:
                st.markdown("#### Entity Type Distribution")

                # Create bar chart
                fig = px.bar(
                    x=list(type_counts.keys()),
                    y=list(type_counts.values()),
                    title="Entity Types",
                    labels={'x': 'Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Relationship type distribution
        if st.session_state.knowledge_graph_data['relationships']:
            relationships = st.session_state.knowledge_graph_data['relationships']

            # Count relationship types
            rel_counts = {}
            for rel in relationships:
                rel_type = rel.get('relation', 'related_to')
                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

            if rel_counts:
                st.markdown("#### Relationship Type Distribution")

                # Create bar chart
                fig = px.bar(
                    x=list(rel_counts.keys()),
                    y=list(rel_counts.values()),
                    title="Relationship Types",
                    labels={'x': 'Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for BubbleLabs Knowledge Integration."""
    st.set_page_config(
        page_title="BubbleLabs Knowledge Integration",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 BubbleLabs Knowledge Engine Integration")
    st.markdown("""
    Explore knowledge graphs, query knowledge bases, and extract insights from documents.

    Features:
    - 🔍 Query multiple knowledge sources
    - 📊 Interactive knowledge graph visualization
    - 📄 Extract knowledge from documents
    - 📈 Comprehensive statistics dashboard
    """)

    # Create UI
    ui = BubbleLabsKnowledgeUI()
    ui.render_knowledge_explorer()


if __name__ == "__main__":
    main()
