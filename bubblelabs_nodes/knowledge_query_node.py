"""
Knowledge Query Node for BubbleLabs Integration

Provides querying capabilities for the Unified Knowledge Graph:
- Query triples by subject, predicate, or object
- Find paths between entities
- Get entity neighbors
- Export knowledge as JSON
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeQueryNode(BubbleLabsNode):
    """
    Query the unified knowledge graph for entities, relationships, and paths.

    Supports four query types:
    - triples: Query triples by subject, predicate, or object with confidence filtering
    - paths: Find paths between two entities with configurable max length
    - neighbors: Get neighborhood of an entity with configurable depth
    - export: Export the entire knowledge graph as JSON

    All queries support minimum confidence thresholds for quality control.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Query"
    DESCRIPTION = "Query the unified knowledge graph for entities, relationships, and paths"
    ICON = "knowledge-query"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import UnifiedKnowledgeGraph using safe import pattern
        UnifiedKnowledgeGraph = self.safe_import(
            'knowledge_engine.graph.unified_kg.UnifiedKnowledgeGraph',
            fallback_value=None,
            error_msg="UnifiedKnowledgeGraph not available for KnowledgeQueryNode"
        )

        # Also try alternative import paths
        if UnifiedKnowledgeGraph is None:
            UnifiedKnowledgeGraph = self.safe_import(
                'graph.unified_kg.UnifiedKnowledgeGraph',
                fallback_value=None,
                error_msg="UnifiedKnowledgeGraph not found in alternate path"
            )

        if UnifiedKnowledgeGraph is None:
            # Try one more common location
            unified_kg_module = self.safe_import(
                'unified_kg',
                fallback_value=None,
                error_msg="unified_kg module not available"
            )
            if unified_kg_module:
                UnifiedKnowledgeGraph = getattr(unified_kg_module, 'UnifiedKnowledgeGraph', None)

        self.UnifiedKnowledgeGraph = UnifiedKnowledgeGraph

        # Initialize knowledge graph instance if available
        self.kg_instance = None
        if UnifiedKnowledgeGraph:
            try:
                backend = self.config.get('backend', 'networkx')
                self.kg_instance = UnifiedKnowledgeGraph(backend=backend)
                self.logger.info(f"KnowledgeQueryNode initialized with {backend} backend")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKnowledgeGraph: {e}")
                self.kg_instance = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on query type.

        Required fields vary by query_type:
        - triples: No required fields (optional filters: subject, predicate, object)
        - paths: Requires source_entity and target_entity
        - neighbors: Requires source_entity (the center entity)
        - export: No required fields
        """
        errors = []

        # Get query type from inputs or config
        query_type = inputs.get('query_type')
        if query_type is None:
            query_type = self.config.get('query_type')

        if query_type is None:
            errors.append("Missing required field: query_type (must be 'triples', 'paths', 'neighbors', or 'export')")
            return errors

        valid_query_types = ['triples', 'paths', 'neighbors', 'export']
        if query_type not in valid_query_types:
            errors.append(f"Invalid query_type: {query_type}. Must be one of: {', '.join(valid_query_types)}")
            return errors

        # Validate query-type specific requirements
        if query_type == 'paths':
            source = inputs.get('source_entity') or self.config.get('source_entity')
            target = inputs.get('target_entity') or self.config.get('target_entity')
            if not source:
                errors.append("Path queries require 'source_entity' (in inputs or config)")
            if not target:
                errors.append("Path queries require 'target_entity' (in inputs or config)")

        elif query_type == 'neighbors':
            source = inputs.get('source_entity') or self.config.get('source_entity')
            if not source:
                errors.append("Neighbor queries require 'source_entity' (in inputs or config)")

        # Validate numeric parameters if provided
        if 'min_confidence' in inputs:
            try:
                confidence = float(inputs['min_confidence'])
                if not (0.0 <= confidence <= 1.0):
                    errors.append("min_confidence must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("min_confidence must be a number")

        if 'max_path_length' in inputs:
            try:
                length = int(inputs['max_path_length'])
                if length < 1:
                    errors.append("max_path_length must be at least 1")
            except (ValueError, TypeError):
                errors.append("max_path_length must be an integer")

        if 'depth' in inputs:
            try:
                depth = int(inputs['depth'])
                if depth < 1:
                    errors.append("depth must be at least 1")
            except (ValueError, TypeError):
                errors.append("depth must be an integer")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge query based on query_type.

        Args:
            inputs: Query specification including query_type and parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing query results:
                - query_type: The type of query executed
                - results: Query-specific results (triples, paths, neighbors, or export data)
                - metadata: Query execution metadata
                - total_count: Number of results returned

        Raises:
            NodeExecutionError: If query execution fails
        """
        # Get query parameters
        query_type = inputs.get('query_type', self.config.get('query_type'))

        # Merge inputs with config (inputs take precedence)
        subject = inputs.get('subject') or self.config.get('subject')
        predicate = inputs.get('predicate') or self.config.get('predicate')
        obj = inputs.get('object') or self.config.get('object')
        min_confidence = inputs.get('min_confidence', self.config.get('min_confidence', 0.0))
        source_entity = inputs.get('source_entity') or self.config.get('source_entity')
        target_entity = inputs.get('target_entity') or self.config.get('target_entity')
        max_path_length = inputs.get('max_path_length', self.config.get('max_path_length', 3))
        depth = inputs.get('depth', self.config.get('depth', 1))

        context.update_progress(10, f"Initializing {query_type} query")
        self.logger.info(f"Executing {query_type} query")

        try:
            # Get the knowledge graph instance to use
            kg = self._get_knowledge_graph(inputs)

            if kg is None:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="No knowledge graph available for querying",
                    details={'query_type': query_type}
                )

            context.update_progress(30, "Executing query")

            # Execute the appropriate query
            if query_type == 'triples':
                result = self._query_triples(kg, subject, predicate, obj, min_confidence)
            elif query_type == 'paths':
                result = self._query_paths(kg, source_entity, target_entity, max_path_length, min_confidence)
            elif query_type == 'neighbors':
                result = self._query_neighbors(kg, source_entity, depth, min_confidence)
            elif query_type == 'export':
                result = self._export_knowledge(kg, min_confidence)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown query type: {query_type}",
                    details={'valid_types': ['triples', 'paths', 'neighbors', 'export']}
                )

            context.update_progress(90, "Processing results")

            # Add execution metadata
            result['metadata'] = {
                'query_type': query_type,
                'executed_at': datetime.now().isoformat(),
                'execution_id': self.execution_id,
                'parameters': {
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'min_confidence': min_confidence,
                    'source_entity': source_entity,
                    'target_entity': target_entity,
                    'max_path_length': max_path_length,
                    'depth': depth
                }
            }

            context.update_progress(100, f"Query complete: {result.get('total_count', 0)} results")
            self.logger.info(f"Knowledge query completed: {result.get('total_count', 0)} results")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge query failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge query failed: {str(e)}",
                details={
                    'query_type': query_type,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph(self, inputs: Dict) -> Optional[Any]:
        """
        Get the knowledge graph instance to use.

        Priority:
        1. kg_instance from inputs
        2. self.kg_instance (initialized in __init__)
        3. None (if not available)
        """
        # Check if a knowledge graph instance was provided in inputs
        if 'kg_instance' in inputs:
            return inputs['kg_instance']

        # Use the instance created in __init__
        return self.kg_instance

    def _query_triples(
        self,
        kg: Any,
        subject: Optional[str],
        predicate: Optional[str],
        obj: Optional[str],
        min_confidence: float
    ) -> Dict[str, Any]:
        """Query triples by subject, predicate, or object."""
        triples = kg.get_triples(
            subject=subject,
            predicate=predicate,
            object=obj,
            min_confidence=min_confidence
        )

        # Convert triples to dictionaries
        triples_data = []
        for t in triples:
            triples_data.append({
                'subject': t.subject,
                'predicate': t.predicate,
                'object': t.object,
                'confidence': t.confidence,
                'source': t.source,
                'timestamp': t.timestamp.isoformat() if hasattr(t.timestamp, 'isoformat') else str(t.timestamp),
                'metadata': t.metadata
            })

        return {
            'query_type': 'triples',
            'results': triples_data,
            'total_count': len(triples_data),
            'filters': {
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'min_confidence': min_confidence
            }
        }

    def _query_paths(
        self,
        kg: Any,
        source: str,
        target: str,
        max_length: int,
        min_confidence: float
    ) -> Dict[str, Any]:
        """Find paths between two entities."""
        paths = kg.find_paths(source, target, max_length)

        # Filter paths by confidence if needed
        filtered_paths = []
        for path in paths:
            # Check if all edges in path meet confidence threshold
            if all(edge.get('confidence', 1.0) >= min_confidence for edge in path):
                filtered_paths.append(path)

        return {
            'query_type': 'paths',
            'source_entity': source,
            'target_entity': target,
            'results': filtered_paths,
            'total_count': len(filtered_paths),
            'path_lengths': [len(p) for p in filtered_paths],
            'filters': {
                'max_path_length': max_length,
                'min_confidence': min_confidence
            }
        }

    def _query_neighbors(
        self,
        kg: Any,
        entity: str,
        depth: int,
        min_confidence: float
    ) -> Dict[str, Any]:
        """Get neighborhood of an entity."""
        neighbors = kg.get_neighbors(entity, depth)

        # Filter edges by confidence if needed
        if min_confidence > 0.0 and 'edges' in neighbors:
            neighbors['edges'] = [
                e for e in neighbors['edges']
                if e.get('confidence', 1.0) >= min_confidence
            ]
            # Recalculate node count based on filtered edges
            connected_nodes = set()
            for edge in neighbors['edges']:
                connected_nodes.add(edge.get('from'))
                connected_nodes.add(edge.get('to'))
            neighbors['node_count'] = len(connected_nodes)
            neighbors['edge_count'] = len(neighbors['edges'])

        return {
            'query_type': 'neighbors',
            'center_entity': entity,
            'results': neighbors,
            'total_count': neighbors.get('node_count', 0) - 1,  # Exclude center
            'filters': {
                'depth': depth,
                'min_confidence': min_confidence
            }
        }

    def _export_knowledge(
        self,
        kg: Any,
        min_confidence: float
    ) -> Dict[str, Any]:
        """Export knowledge graph as JSON."""
        export_data = kg.export_to_dict()

        # Filter by confidence if needed
        if min_confidence > 0.0 and 'triples' in export_data:
            original_count = len(export_data['triples'])
            export_data['triples'] = [
                t for t in export_data['triples']
                if t.get('confidence', 1.0) >= min_confidence
            ]
            export_data['filtered_count'] = len(export_data['triples'])
            export_data['filtered_out'] = original_count - len(export_data['triples'])

            # Update statistics
            if 'statistics' in export_data:
                export_data['statistics']['filtered_triple_count'] = len(export_data['triples'])

        return {
            'query_type': 'export',
            'results': export_data,
            'total_count': len(export_data.get('triples', [])),
            'filters': {
                'min_confidence': min_confidence
            }
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration including:
        - query_type: Type of query to execute
        - subject/predicate/object: Filters for triple queries
        - min_confidence: Minimum confidence threshold
        - source_entity/target_entity: For path queries
        - max_path_length: Maximum path length for path queries
        - depth: Neighborhood depth for neighbor queries
        """
        return {
            "type": "object",
            "title": "Knowledge Query Configuration",
            "description": "Configure knowledge graph query parameters",
            "properties": {
                "query_type": {
                    "type": "string",
                    "title": "Query Type",
                    "description": "Type of query to execute against the knowledge graph",
                    "enum": ["triples", "paths", "neighbors", "export"],
                    "enumNames": [
                        "Query Triples",
                        "Find Paths",
                        "Get Neighbors",
                        "Export Knowledge"
                    ],
                    "default": "triples"
                },
                "subject": {
                    "type": "string",
                    "title": "Subject",
                    "description": "Filter triples by subject entity (optional)",
                    "default": ""
                },
                "predicate": {
                    "type": "string",
                    "title": "Predicate",
                    "description": "Filter triples by predicate/relationship (optional)",
                    "default": ""
                },
                "object": {
                    "type": "string",
                    "title": "Object",
                    "description": "Filter triples by object entity (optional)",
                    "default": ""
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "description": "Minimum confidence threshold for results (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0
                },
                "source_entity": {
                    "type": "string",
                    "title": "Source Entity",
                    "description": "Starting entity for path or neighbor queries",
                    "default": ""
                },
                "target_entity": {
                    "type": "string",
                    "title": "Target Entity",
                    "description": "Target entity for path queries",
                    "default": ""
                },
                "max_path_length": {
                    "type": "number",
                    "title": "Maximum Path Length",
                    "description": "Maximum path length for path queries",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3
                },
                "depth": {
                    "type": "number",
                    "title": "Neighborhood Depth",
                    "description": "Depth of neighborhood for neighbor queries",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1
                },
                "backend": {
                    "type": "string",
                    "title": "Storage Backend",
                    "description": "Backend storage for knowledge graph",
                    "enum": ["networkx", "memory"],
                    "enumNames": [
                        "NetworkX (recommended)",
                        "In-Memory"
                    ],
                    "default": "networkx"
                }
            },
            "required": ["query_type"],
            "dependencies": {
                "query_type": {
                    "oneOf": [
                        {
                            "properties": {
                                "query_type": {"enum": ["triples"]}
                            },
                            "description": "Query triples by subject, predicate, or object"
                        },
                        {
                            "properties": {
                                "query_type": {"enum": ["paths"]}
                            },
                            "required": ["source_entity", "target_entity"],
                            "description": "Find paths between two entities"
                        },
                        {
                            "properties": {
                                "query_type": {"enum": ["neighbors"]}
                            },
                            "required": ["source_entity"],
                            "description": "Get neighborhood of an entity"
                        },
                        {
                            "properties": {
                                "query_type": {"enum": ["export"]}
                            },
                            "description": "Export entire knowledge graph as JSON"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if UnifiedKnowledgeGraph is available, False otherwise
        """
        return self.UnifiedKnowledgeGraph is not None
