"""
Knowledge Analytics Node for BubbleLabs Integration

Provides comprehensive analytics and metrics for knowledge graphs including:
- Graph statistics (nodes, edges, density, diameter)
- Centrality metrics (degree, betweenness, closeness, pagerank)
- Entity distribution analysis
- Confidence statistics and quality metrics
- Growth and change metrics over time
- Exportable analytics reports

The node can work with knowledge graph IDs from the UnifiedKGIntegrationHub
or process knowledge graph data directly from the workflow context.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import json
import csv
import io
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeAnalyticsNode(BubbleLabsNode):
    """
    Knowledge Analytics Node for generating statistics and metrics for knowledge graphs.

    Provides comprehensive analytics including:
    - Basic statistics: node count, edge count, density, component analysis
    - Centrality metrics: degree, betweenness, closeness, pagerank
    - Distribution analysis: entity types, relationship types
    - Quality metrics: confidence statistics, coverage scores
    - Growth metrics: temporal changes, growth rates
    - Comprehensive reports combining all metrics

    Supports multiple export formats (JSON, CSV, formatted report) for downstream use.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Analytics"
    DESCRIPTION = "Generate statistics, metrics, and analytics for knowledge graphs"
    ICON = "knowledge-analytics"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of UnifiedKGIntegrationHub
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for KnowledgeAnalyticsNode"
        )

        self.UnifiedKGIntegrationHub = None
        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)

        # Safe import of KarateClubIntegration
        karateclub_module = self.safe_import(
            'knowledge_engine.integrations.karateclub_integration',
            fallback_value=None,
            error_msg="KarateClubIntegration not available for KnowledgeAnalyticsNode"
        )

        self.KarateClubIntegration = None
        self.karateclub = None
        if karateclub_module:
            self.KarateClubIntegration = getattr(karateclub_module, 'KarateClubIntegration', None)
            if self.KarateClubIntegration:
                try:
                    self.karateclub = self.KarateClubIntegration()
                    self.logger.info("KarateClub integration initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Could not initialize KarateClub integration: {e}")
                    self.karateclub = None

        # Initialize hub instance if available
        self.hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Try to import networkx for graph analysis
        self.nx = self.safe_import(
            'networkx',
            fallback_value=None,
            error_msg="NetworkX not available, some metrics will use fallback calculations"
        )

        # Track available capabilities
        self.has_networkx = self.nx is not None
        self.has_karateclub = self.karateclub is not None and (
            self.karateclub.is_available() if hasattr(self.karateclub, 'is_available') else True
        )

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (one of):
            - knowledge_graph_id: str - ID of the knowledge graph to analyze
            - knowledge_graph: dict - Knowledge graph data directly

        Optional:
            - analysis_type: str - Override the configured analysis type
            - entity_types: list - Filter by specific entity types
            - time_range: str - Time range for temporal analysis
        """
        errors = []

        # Check that we have either knowledge_graph_id or knowledge_graph
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_kg = 'knowledge_graph' in inputs and inputs['knowledge_graph']

        if not has_kg_id and not has_kg:
            errors.append("Missing required input: either 'knowledge_graph_id' or 'knowledge_graph' must be provided")

        # Validate analysis_type if provided
        if 'analysis_type' in inputs:
            valid_types = ['statistics', 'centrality', 'distribution', 'quality', 'growth', 'comprehensive']
            if inputs['analysis_type'] not in valid_types:
                errors.append(f"Invalid analysis_type: '{inputs['analysis_type']}'. Must be one of: {', '.join(valid_types)}")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be a list of strings")
            elif not all(isinstance(et, str) for et in inputs['entity_types']):
                errors.append("All items in 'entity_types' must be strings")

        # Validate metrics if provided
        if 'metrics' in inputs:
            if not isinstance(inputs['metrics'], list):
                errors.append("'metrics' must be a list of strings")
            elif not all(isinstance(m, str) for m in inputs['metrics']):
                errors.append("All items in 'metrics' must be strings")

        # Validate compare_with_previous if provided
        if 'compare_with_previous' in inputs:
            if not isinstance(inputs['compare_with_previous'], bool):
                errors.append("'compare_with_previous' must be a boolean")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge graph analytics.

        Args:
            inputs: Contains knowledge_graph_id or knowledge_graph, plus optional parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - statistics: Basic graph statistics
                - centrality: Centrality metrics (if requested)
                - distribution: Distribution analysis (if requested)
                - quality: Quality metrics (if requested)
                - growth: Growth metrics (if requested)
                - report: Formatted analytics report

        Raises:
            NodeExecutionError: If analytics fails
        """
        # Get configuration
        analysis_type = inputs.get('analysis_type', self.config.get('analysis_type', 'comprehensive'))
        metrics = inputs.get('metrics', self.config.get('metrics', []))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        time_range = inputs.get('time_range', self.config.get('time_range', None))
        compare_with_previous = inputs.get(
            'compare_with_previous',
            self.config.get('compare_with_previous', False)
        )
        export_format = inputs.get('export_format', self.config.get('export_format', 'json'))

        context.update_progress(10, f"Initializing {analysis_type} analysis")
        self.logger.info(f"Starting knowledge analytics: type={analysis_type}")

        try:
            # Retrieve knowledge graph data
            kg_data = self._get_knowledge_graph_data(inputs, context)

            if not kg_data:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="Could not retrieve knowledge graph data",
                    details={'inputs': list(inputs.keys())}
                )

            context.update_progress(25, "Knowledge graph retrieved, preprocessing data")

            # Filter by entity types if specified
            if entity_types:
                kg_data = self._filter_by_entity_types(kg_data, entity_types)
                context.update_progress(30, f"Filtered to {len(kg_data.get('nodes', []))} nodes of specified types")

            # Filter by time range if specified
            if time_range:
                kg_data = self._filter_by_time_range(kg_data, time_range)
                context.update_progress(35, f"Filtered by time range: {time_range}")

            context.update_progress(40, f"Executing {analysis_type} analytics")

            # Build result based on analysis type
            result = {
                'analysis_type': analysis_type,
                'metadata': {
                    'executed_at': datetime.now().isoformat(),
                    'execution_id': self.execution_id,
                    'entity_types_filter': entity_types,
                    'time_range_filter': time_range,
                    'compare_with_previous': compare_with_previous,
                    'capabilities': {
                        'networkx': self.has_networkx,
                        'karateclub': self.has_karateclub
                    }
                }
            }

            # Always calculate basic statistics
            result['statistics'] = self._calculate_statistics(kg_data, context)
            context.update_progress(50, "Statistics calculated")

            # Calculate specific metrics based on analysis type
            if analysis_type in ['centrality', 'comprehensive']:
                result['centrality'] = self._calculate_centrality(kg_data, metrics, context)
                context.update_progress(65, "Centrality metrics calculated")

            if analysis_type in ['distribution', 'comprehensive']:
                result['distribution'] = self._analyze_distribution(kg_data, context)
                context.update_progress(75, "Distribution analysis complete")

            if analysis_type in ['quality', 'comprehensive']:
                result['quality'] = self._analyze_quality(kg_data, context)
                context.update_progress(85, "Quality metrics calculated")

            if analysis_type in ['growth', 'comprehensive']:
                result['growth'] = self._analyze_growth(kg_data, time_range, compare_with_previous, context)
                context.update_progress(90, "Growth metrics calculated")

            # Generate report
            result['report'] = self._generate_report(result, export_format, context)
            context.update_progress(95, "Report generated")

            # Store artifacts in context
            context.add_artifact('knowledge_analytics', {
                'analysis_type': analysis_type,
                'node_count': result['statistics'].get('node_count', 0),
                'edge_count': result['statistics'].get('edge_count', 0),
                'density': result['statistics'].get('density', 0),
                'export_format': export_format
            })

            context.update_progress(100, "Analytics complete")

            self.logger.info(
                f"Knowledge analytics completed: "
                f"nodes={result['statistics'].get('node_count', 0)}, "
                f"edges={result['statistics'].get('edge_count', 0)}"
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge analytics failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge analytics failed: {str(e)}",
                details={
                    'analysis_type': analysis_type,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph_data(self, inputs: Dict, context) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge graph data from inputs or hub.

        Priority:
        1. knowledge_graph from inputs (direct data)
        2. knowledge_graph_id from inputs (fetch from hub)
        3. kg_instance from inputs
        """
        # Direct knowledge graph data
        if 'knowledge_graph' in inputs and inputs['knowledge_graph']:
            return inputs['knowledge_graph']

        # Fetch from hub using knowledge_graph_id
        kg_id = inputs.get('knowledge_graph_id')
        if kg_id and self.hub:
            try:
                # Try to get from hub
                if hasattr(self.hub, 'get_knowledge_graph'):
                    return self.hub.get_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'export_graph'):
                    return self.hub.export_graph(kg_id)
            except Exception as e:
                self.logger.warning(f"Could not fetch graph from hub: {e}")

        # Check for kg_instance in inputs
        if 'kg_instance' in inputs and inputs['kg_instance']:
            kg = inputs['kg_instance']
            if hasattr(kg, 'export_to_dict'):
                return kg.export_to_dict()
            elif hasattr(kg, 'to_dict'):
                return kg.to_dict()

        return None

    def _filter_by_entity_types(self, kg_data: Dict[str, Any], entity_types: List[str]) -> Dict[str, Any]:
        """Filter knowledge graph by entity types."""
        if not entity_types:
            return kg_data

        # Filter nodes
        filtered_nodes = [
            node for node in kg_data.get('nodes', [])
            if node.get('type') in entity_types or node.get('entity_type') in entity_types
        ]

        # Get IDs of filtered nodes
        node_ids = {node.get('id') for node in filtered_nodes}

        # Filter edges to only include connections between filtered nodes
        filtered_edges = [
            edge for edge in kg_data.get('edges', [])
            if edge.get('source') in node_ids and edge.get('target') in node_ids
        ]

        # Filter triples if present
        filtered_triples = []
        if 'triples' in kg_data:
            filtered_triples = [
                triple for triple in kg_data['triples']
                if triple.get('subject') in node_ids and triple.get('object') in node_ids
            ]

        result = dict(kg_data)
        result['nodes'] = filtered_nodes
        result['edges'] = filtered_edges
        if filtered_triples:
            result['triples'] = filtered_triples

        return result

    def _filter_by_time_range(self, kg_data: Dict[str, Any], time_range: str) -> Dict[str, Any]:
        """Filter knowledge graph by time range."""
        # Parse time range (simplified implementation)
        # Expected formats: "7d", "30d", "1m", "1y", or ISO date range "2024-01-01/2024-12-31"
        try:
            if '/' in time_range:
                # Date range format
                start_str, end_str = time_range.split('/')
                start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            else:
                # Relative time format
                end_date = datetime.now()
                if time_range.endswith('d'):
                    days = int(time_range[:-1])
                    start_date = end_date - timedelta(days=days)
                elif time_range.endswith('m'):
                    months = int(time_range[:-1])
                    start_date = end_date - timedelta(days=months * 30)
                elif time_range.endswith('y'):
                    years = int(time_range[:-1])
                    start_date = end_date - timedelta(days=years * 365)
                else:
                    return kg_data

            # Filter nodes by timestamp
            def is_in_range(item):
                timestamp = item.get('timestamp') or item.get('created_at')
                if not timestamp:
                    return True  # Include items without timestamps
                try:
                    if isinstance(timestamp, str):
                        item_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        item_date = timestamp
                    return start_date <= item_date <= end_date
                except:
                    return True

            result = dict(kg_data)
            result['nodes'] = [node for node in kg_data.get('nodes', []) if is_in_range(node)]
            result['edges'] = [edge for edge in kg_data.get('edges', []) if is_in_range(edge)]
            if 'triples' in kg_data:
                result['triples'] = [triple for triple in kg_data['triples'] if is_in_range(triple)]

            return result

        except Exception as e:
            self.logger.warning(f"Could not parse time range '{time_range}': {e}")
            return kg_data

    def _calculate_statistics(self, kg_data: Dict[str, Any], context) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        triples = kg_data.get('triples', [])

        node_count = len(nodes)
        edge_count = len(edges)
        triple_count = len(triples)

        # Calculate density
        if node_count > 1:
            max_edges = node_count * (node_count - 1) / 2  # For undirected graph
            density = edge_count / max_edges if max_edges > 0 else 0
        else:
            density = 0

        # Calculate average degree
        if node_count > 0:
            avg_degree = (2 * edge_count) / node_count
        else:
            avg_degree = 0

        # Component analysis (if networkx available)
        components = {'count': 0, 'largest': 0, 'sizes': []}
        if self.has_networkx:
            try:
                G = self._build_networkx_graph(kg_data)
                if G:
                    if G.is_directed():
                        component_list = list(self.nx.weakly_connected_components(G))
                    else:
                        component_list = list(self.nx.connected_components(G))
                    components['count'] = len(component_list)
                    components['sizes'] = [len(c) for c in component_list]
                    components['largest'] = max(components['sizes']) if components['sizes'] else 0
            except Exception as e:
                self.logger.warning(f"Component analysis failed: {e}")

        # Diameter (if networkx available and graph is connected)
        diameter = None
        radius = None
        if self.has_networkx and components['count'] == 1:
            try:
                G = self._build_networkx_graph(kg_data)
                if G and not G.is_directed():
                    diameter = self.nx.diameter(G)
                    radius = self.nx.radius(G)
            except Exception as e:
                self.logger.warning(f"Diameter calculation failed: {e}")

        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'triple_count': triple_count,
            'density': round(density, 6),
            'avg_degree': round(avg_degree, 4),
            'components': components,
            'diameter': diameter,
            'radius': radius,
            'is_directed': kg_data.get('directed', False)
        }

    def _calculate_centrality(self, kg_data: Dict[str, Any], requested_metrics: List[str], context) -> Dict[str, Any]:
        """Calculate centrality metrics."""
        if not self.has_networkx:
            return self._calculate_centrality_fallback(kg_data, requested_metrics)

        try:
            G = self._build_networkx_graph(kg_data)
            if not G:
                return self._calculate_centrality_fallback(kg_data, requested_metrics)

            centrality = {}

            # Determine which metrics to calculate
            all_metrics = ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']
            metrics_to_calc = requested_metrics if requested_metrics else all_metrics

            # Degree centrality
            if 'degree' in metrics_to_calc:
                if G.is_directed():
                    in_degree = self.nx.in_degree_centrality(G)
                    out_degree = self.nx.out_degree_centrality(G)
                    centrality['degree'] = {
                        'in_degree': self._summarize_centrality(in_degree),
                        'out_degree': self._summarize_centrality(out_degree),
                        'combined': self._summarize_centrality(
                            {k: (in_degree.get(k, 0) + out_degree.get(k, 0)) / 2 for k in set(in_degree) | set(out_degree)}
                        )
                    }
                else:
                    degree = self.nx.degree_centrality(G)
                    centrality['degree'] = self._summarize_centrality(degree)

            # Betweenness centrality
            if 'betweenness' in metrics_to_calc:
                betweenness = self.nx.betweenness_centrality(G)
                centrality['betweenness'] = self._summarize_centrality(betweenness)

            # Closeness centrality
            if 'closeness' in metrics_to_calc:
                try:
                    closeness = self.nx.closeness_centrality(G)
                    centrality['closeness'] = self._summarize_centrality(closeness)
                except Exception as e:
                    self.logger.warning(f"Closeness centrality failed: {e}")
                    centrality['closeness'] = {'error': str(e)}

            # PageRank
            if 'pagerank' in metrics_to_calc:
                try:
                    if G.is_directed():
                        pagerank = self.nx.pagerank(G)
                    else:
                        pagerank = self.nx.pagerank(G)
                    centrality['pagerank'] = self._summarize_centrality(pagerank)
                except Exception as e:
                    self.logger.warning(f"PageRank calculation failed: {e}")
                    centrality['pagerank'] = {'error': str(e)}

            # Eigenvector centrality
            if 'eigenvector' in metrics_to_calc:
                try:
                    eigenvector = self.nx.eigenvector_centrality(G, max_iter=1000)
                    centrality['eigenvector'] = self._summarize_centrality(eigenvector)
                except Exception as e:
                    self.logger.warning(f"Eigenvector centrality failed: {e}")
                    centrality['eigenvector'] = {'error': str(e)}

            return centrality

        except Exception as e:
            self.logger.warning(f"Centrality calculation failed: {e}, using fallback")
            return self._calculate_centrality_fallback(kg_data, requested_metrics)

    def _calculate_centrality_fallback(self, kg_data: Dict[str, Any], requested_metrics: List[str]) -> Dict[str, Any]:
        """Fallback centrality calculation using simple degree-based metrics."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                adjacency[source].add(target)
                adjacency[target].add(source)

        # Calculate degree centrality
        node_ids = [node.get('id') for node in nodes if node.get('id')]
        n = len(node_ids)
        max_possible = n - 1 if n > 1 else 1

        degree_centrality = {}
        for node_id in node_ids:
            degree = len(adjacency.get(node_id, set()))
            degree_centrality[node_id] = degree / max_possible

        return {
            'degree': self._summarize_centrality(degree_centrality),
            'betweenness': {'note': 'Betweenness requires NetworkX'},
            'closeness': {'note': 'Closeness requires NetworkX'},
            'pagerank': {'note': 'PageRank requires NetworkX'},
            'fallback': True
        }

    def _summarize_centrality(self, centrality_dict: Dict[str, float]) -> Dict[str, Any]:
        """Summarize centrality values."""
        if not centrality_dict:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'top_nodes': []}

        values = list(centrality_dict.values())

        # Get top 5 nodes by centrality
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        top_nodes = [{'node_id': node_id, 'score': round(score, 6)} for node_id, score in sorted_nodes]

        return {
            'mean': round(statistics.mean(values), 6),
            'max': round(max(values), 6),
            'min': round(min(values), 6),
            'std': round(statistics.stdev(values), 6) if len(values) > 1 else 0,
            'top_nodes': top_nodes
        }

    def _analyze_distribution(self, kg_data: Dict[str, Any], context) -> Dict[str, Any]:
        """Analyze entity and relationship distributions."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        triples = kg_data.get('triples', [])

        # Entity type distribution
        entity_types = Counter()
        for node in nodes:
            entity_type = node.get('type') or node.get('entity_type') or 'unknown'
            entity_types[entity_type] += 1

        # Relationship type distribution
        relationship_types = Counter()
        for edge in edges:
            rel_type = edge.get('type') or edge.get('relationship') or edge.get('predicate') or 'unknown'
            relationship_types[rel_type] += 1

        # Also check triples for relationship types
        for triple in triples:
            rel_type = triple.get('predicate') or 'unknown'
            relationship_types[rel_type] += 1

        # Calculate distribution statistics
        entity_type_dist = dict(entity_types.most_common())
        relationship_type_dist = dict(relationship_types.most_common())

        # Calculate entropy (measure of distribution uniformity)
        entity_entropy = self._calculate_entropy(list(entity_types.values()))
        relationship_entropy = self._calculate_entropy(list(relationship_types.values()))

        return {
            'entity_type_distribution': entity_type_dist,
            'relationship_distribution': relationship_type_dist,
            'entity_type_count': len(entity_types),
            'relationship_type_count': len(relationship_types),
            'entity_entropy': round(entity_entropy, 4),
            'relationship_entropy': round(relationship_entropy, 4),
            'dominant_entity_type': entity_types.most_common(1)[0][0] if entity_types else None,
            'dominant_relationship': relationship_types.most_common(1)[0][0] if relationship_types else None
        }

    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy for a distribution."""
        if not counts:
            return 0

        total = sum(counts)
        if total == 0:
            return 0

        import math
        entropy = 0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _analyze_quality(self, kg_data: Dict[str, Any], context) -> Dict[str, Any]:
        """Analyze quality metrics including confidence statistics."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        triples = kg_data.get('triples', [])

        # Collect confidence values
        node_confidences = [n.get('confidence', 1.0) for n in nodes if n.get('confidence') is not None]
        edge_confidences = [e.get('confidence', 1.0) for e in edges if e.get('confidence') is not None]
        triple_confidences = [t.get('confidence', 1.0) for t in triples if t.get('confidence') is not None]

        all_confidences = node_confidences + edge_confidences + triple_confidences

        quality = {}

        # Confidence statistics
        if all_confidences:
            quality['avg_confidence'] = round(statistics.mean(all_confidences), 4)
            quality['median_confidence'] = round(statistics.median(all_confidences), 4)
            quality['min_confidence'] = round(min(all_confidences), 4)
            quality['max_confidence'] = round(max(all_confidences), 4)
            if len(all_confidences) > 1:
                quality['std_confidence'] = round(statistics.stdev(all_confidences), 4)

            # Confidence distribution
            buckets = {'high': 0, 'medium': 0, 'low': 0}
            for c in all_confidences:
                if c >= 0.8:
                    buckets['high'] += 1
                elif c >= 0.5:
                    buckets['medium'] += 1
                else:
                    buckets['low'] += 1
            quality['confidence_distribution'] = buckets
        else:
            quality['note'] = 'No confidence data available'

        # Coverage analysis
        nodes_with_metadata = sum(1 for n in nodes if n.get('metadata'))
        edges_with_metadata = sum(1 for e in edges if e.get('metadata'))

        quality['coverage'] = {
            'nodes_with_metadata': nodes_with_metadata,
            'edges_with_metadata': edges_with_metadata,
            'node_coverage_pct': round(nodes_with_metadata / len(nodes) * 100, 2) if nodes else 0,
            'edge_coverage_pct': round(edges_with_metadata / len(edges) * 100, 2) if edges else 0
        }

        # Consistency score (based on duplicate detection)
        node_signatures = [json.dumps({k: v for k, v in n.items() if k != 'id'}, sort_keys=True) for n in nodes]
        duplicates = len(node_signatures) - len(set(node_signatures))
        quality['consistency_score'] = round(1 - (duplicates / len(nodes) if nodes else 0), 4)

        return quality

    def _analyze_growth(self, kg_data: Dict[str, Any], time_range: Optional[str], 
                        compare_with_previous: bool, context) -> Dict[str, Any]:
        """Analyze growth and change metrics."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])

        growth = {
            'current_metrics': {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'timestamp': datetime.now().isoformat()
            }
        }

        # Try to extract temporal information
        timestamps = []
        for item in nodes + edges:
            ts = item.get('timestamp') or item.get('created_at')
            if ts:
                try:
                    if isinstance(ts, str):
                        timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    else:
                        timestamps.append(ts)
                except:
                    pass

        if timestamps:
            timestamps.sort()
            growth['temporal_range'] = {
                'earliest': timestamps[0].isoformat() if timestamps else None,
                'latest': timestamps[-1].isoformat() if timestamps else None,
                'span_days': (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0
            }

            # Calculate additions over time (simplified)
            if len(timestamps) > 1:
                mid_point = timestamps[len(timestamps) // 2]
                early_count = sum(1 for t in timestamps if t < mid_point)
                late_count = sum(1 for t in timestamps if t >= mid_point)

                growth_rate = ((late_count - early_count) / early_count * 100) if early_count > 0 else 0
                growth['growth_rate'] = round(growth_rate, 2)
                growth['new_entities'] = late_count
                growth['modified_entities'] = early_count  # Simplified

        # Comparison placeholder (would require historical data)
        if compare_with_previous:
            growth['comparison_note'] = 'Historical comparison requires stored analytics data'

        return growth

    def _generate_report(self, result: Dict[str, Any], export_format: str, context) -> Dict[str, Any]:
        """Generate formatted analytics report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'format': export_format,
            'summary': {
                'analysis_type': result['analysis_type'],
                'node_count': result.get('statistics', {}).get('node_count', 0),
                'edge_count': result.get('statistics', {}).get('edge_count', 0),
                'density': result.get('statistics', {}).get('density', 0)
            }
        }

        if export_format == 'json':
            report['data'] = {
                'statistics': result.get('statistics'),
                'centrality': result.get('centrality'),
                'distribution': result.get('distribution'),
                'quality': result.get('quality'),
                'growth': result.get('growth')
            }

        elif export_format == 'csv':
            # Generate CSV format for key metrics
            csv_data = self._generate_csv_report(result)
            report['csv_data'] = csv_data

        elif export_format == 'report':
            # Generate human-readable report
            report['text'] = self._generate_text_report(result)

        return report

    def _generate_csv_report(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate CSV format report."""
        csv_data = {}

        # Statistics CSV
        stats = result.get('statistics', {})
        stats_buffer = io.StringIO()
        writer = csv.writer(stats_buffer)
        writer.writerow(['Metric', 'Value'])
        for key, value in stats.items():
            if isinstance(value, (int, float, str)):
                writer.writerow([key, value])
        csv_data['statistics'] = stats_buffer.getvalue()

        # Distribution CSV
        dist = result.get('distribution', {})
        if dist:
            dist_buffer = io.StringIO()
            writer = csv.writer(dist_buffer)
            writer.writerow(['Type', 'Count'])
            for entity_type, count in dist.get('entity_type_distribution', {}).items():
                writer.writerow([entity_type, count])
            csv_data['distribution'] = dist_buffer.getvalue()

        return csv_data

    def _generate_text_report(self, result: Dict[str, Any]) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 60,
            "KNOWLEDGE GRAPH ANALYTICS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Type: {result['analysis_type']}",
            ""
        ]

        # Statistics section
        stats = result.get('statistics', {})
        lines.extend([
            "-" * 40,
            "GRAPH STATISTICS",
            "-" * 40,
            f"Nodes: {stats.get('node_count', 'N/A')}",
            f"Edges: {stats.get('edge_count', 'N/A')}",
            f"Triples: {stats.get('triple_count', 'N/A')}",
            f"Density: {stats.get('density', 'N/A'):.6f}",
            f"Average Degree: {stats.get('avg_degree', 'N/A'):.4f}",
            ""
        ])

        # Centrality section
        centrality = result.get('centrality', {})
        if centrality:
            lines.extend([
                "-" * 40,
                "CENTRALITY METRICS",
                "-" * 40
            ])
            for metric, values in centrality.items():
                if isinstance(values, dict) and 'mean' in values:
                    lines.append(f"{metric.capitalize()}: mean={values['mean']:.4f}, max={values['max']:.4f}")
            lines.append("")

        # Distribution section
        distribution = result.get('distribution', {})
        if distribution:
            lines.extend([
                "-" * 40,
                "DISTRIBUTION ANALYSIS",
                "-" * 40,
                f"Entity Types: {distribution.get('entity_type_count', 'N/A')}",
                f"Relationship Types: {distribution.get('relationship_type_count', 'N/A')}",
                f"Dominant Entity Type: {distribution.get('dominant_entity_type', 'N/A')}",
                ""
            ])

        # Quality section
        quality = result.get('quality', {})
        if quality:
            lines.extend([
                "-" * 40,
                "QUALITY METRICS",
                "-" * 40,
                f"Average Confidence: {quality.get('avg_confidence', 'N/A')}",
                f"Consistency Score: {quality.get('consistency_score', 'N/A')}",
                ""
            ])

        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])

        return '\n'.join(lines)

    def _build_networkx_graph(self, kg_data: Dict[str, Any]) -> Optional[Any]:
        """Build a NetworkX graph from knowledge graph data."""
        if not self.has_networkx:
            return None

        try:
            # Determine if graph should be directed
            is_directed = kg_data.get('directed', False)
            G = self.nx.DiGraph() if is_directed else self.nx.Graph()

            # Add nodes
            for node in kg_data.get('nodes', []):
                node_id = node.get('id')
                if node_id:
                    G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})

            # Add edges
            for edge in kg_data.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                if source and target:
                    G.add_edge(source, target, **{k: v for k, v in edge.items() if k not in ['source', 'target']})

            # Also process triples if no edges
            if not kg_data.get('edges') and kg_data.get('triples'):
                for triple in kg_data.get('triples', []):
                    subject = triple.get('subject')
                    obj = triple.get('object')
                    if subject and obj:
                        G.add_edge(subject, obj, predicate=triple.get('predicate'))

            return G

        except Exception as e:
            self.logger.warning(f"Failed to build NetworkX graph: {e}")
            return None

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Knowledge Analytics Configuration",
            "description": "Configure knowledge graph analytics parameters",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "title": "Analysis Type",
                    "description": "Type of analytics to perform",
                    "enum": ["statistics", "centrality", "distribution", "quality", "growth", "comprehensive"],
                    "enumNames": [
                        "Statistics - Basic graph metrics",
                        "Centrality - Node importance metrics",
                        "Distribution - Entity and relationship distributions",
                        "Quality - Confidence and quality metrics",
                        "Growth - Temporal change analysis",
                        "Comprehensive - All analytics"
                    ],
                    "default": "comprehensive"
                },
                "metrics": {
                    "type": "array",
                    "title": "Metrics to Calculate",
                    "description": "Specific centrality metrics to calculate (empty for all)",
                    "items": {
                        "type": "string",
                        "enum": ["degree", "betweenness", "closeness", "pagerank", "eigenvector"]
                    },
                    "default": []
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types Filter",
                    "description": "Filter analysis to specific entity types (empty for all)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "time_range": {
                    "type": "string",
                    "title": "Time Range",
                    "description": "Time range for temporal analysis (e.g., '7d', '30d', '2024-01-01/2024-12-31')",
                    "default": ""
                },
                "compare_with_previous": {
                    "type": "boolean",
                    "title": "Compare with Previous",
                    "description": "Compare with historical analytics data",
                    "default": False
                },
                "export_format": {
                    "type": "string",
                    "title": "Export Format",
                    "description": "Format for analytics report export",
                    "enum": ["json", "csv", "report"],
                    "enumNames": [
                        "JSON - Structured data",
                        "CSV - Spreadsheet format",
                        "Report - Human-readable text"
                    ],
                    "default": "json"
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node can perform basic analytics
        """
        # Node can work with or without optional dependencies
        # Basic statistics don't require NetworkX
        return True

    def get_available_capabilities(self) -> Dict[str, bool]:
        """
        Get available analytics capabilities.

        Returns:
            Dictionary of capability name -> availability
        """
        return {
            'basic_statistics': True,
            'centrality_metrics': self.has_networkx,
            'community_detection': self.has_karateclub,
            'distribution_analysis': True,
            'quality_metrics': True,
            'growth_analysis': True
        }
