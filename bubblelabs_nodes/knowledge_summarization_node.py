"""
Knowledge Summarization Node for BubbleLabs Integration

Generate human-readable summaries of knowledge graph content including:
- Entity summaries with descriptions and key facts
- Subgraph summaries for connected entities
- Path summaries explaining relationships between entities
- Topic summaries for thematic exploration
- Change summaries (diffs) for tracking knowledge evolution

Supports multi-level summarization: brief, detailed, comprehensive
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeSummarizationNode(BubbleLabsNode):
    """
    Generate human-readable summaries of knowledge graph content.

    Features:
    - Entity summaries: Describe individual entities with key facts and relationships
    - Subgraph summaries: Summarize connected entity clusters
    - Path summaries: Explain relationship paths between two entities
    - Topic summaries: Discover and summarize knowledge around topics
    - Change summaries: Compare knowledge states and summarize differences

    Multi-level output:
    - brief: Concise 1-2 sentence summary
    - detailed: Paragraph with key facts and statistics
    - comprehensive: Full analysis with all supporting details
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Summarization"
    DESCRIPTION = "Generate human-readable summaries of knowledge graph content"
    ICON = "knowledge-summarization"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import knowledge engine classes using safe_import pattern
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for KnowledgeSummarizationNode"
        )

        self.UnifiedKnowledgeGraph = self.safe_import(
            'knowledge_engine.graph.unified_kg.UnifiedKnowledgeGraph',
            fallback_value=None,
            error_msg="UnifiedKnowledgeGraph not available for KnowledgeSummarizationNode"
        )

        # Try alternative import paths if needed
        if self.UnifiedKnowledgeGraph is None:
            self.UnifiedKnowledgeGraph = self.safe_import(
                'knowledge_engine.core.unified_knowledge_graph.UnifiedKnowledgeGraph',
                fallback_value=None,
                error_msg="UnifiedKnowledgeGraph not found in alternate path"
            )

        # Initialize knowledge graph instances if available
        self.kg_hub = None
        self.kg_instance = None

        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("KnowledgeSummarizationNode initialized with UnifiedKGIntegrationHub")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

        if self.kg_hub is None and self.UnifiedKnowledgeGraph:
            try:
                backend = self.config.get('backend', 'networkx')
                self.kg_instance = self.UnifiedKnowledgeGraph(backend=backend)
                self.logger.info(f"KnowledgeSummarizationNode initialized with {backend} backend")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKnowledgeGraph: {e}")
                self.kg_instance = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields by operation:
        - entity_summary: entity_id
        - subgraph_summary: entity_ids (array)
        - path_summary: source_entity, target_entity
        - topic_summary: topic_query
        - change_summary: previous_state, current_state (or kg_instance with history)
        """
        errors = []

        # Get operation from inputs or config
        operation = inputs.get('operation')
        if operation is None:
            operation = self.config.get('operation')

        if operation is None:
            errors.append("Missing required field: operation (must be 'entity_summary', 'subgraph_summary', 'path_summary', 'topic_summary', or 'change_summary')")
            return errors

        valid_operations = ['entity_summary', 'subgraph_summary', 'path_summary', 'topic_summary', 'change_summary']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Validate operation-specific requirements
        if operation == 'entity_summary':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            if not entity_id:
                errors.append("Entity summary requires 'entity_id' (in inputs or config)")

        elif operation == 'subgraph_summary':
            entity_ids = inputs.get('entity_ids') or self.config.get('entity_ids')
            if not entity_ids:
                errors.append("Subgraph summary requires 'entity_ids' (array of entity IDs)")
            elif not isinstance(entity_ids, list):
                errors.append("entity_ids must be a list of strings")
            elif len(entity_ids) < 1:
                errors.append("entity_ids must contain at least one entity ID")

        elif operation == 'path_summary':
            source = inputs.get('source_entity') or self.config.get('source_entity')
            target = inputs.get('target_entity') or self.config.get('target_entity')
            if not source:
                errors.append("Path summary requires 'source_entity' (in inputs or config)")
            if not target:
                errors.append("Path summary requires 'target_entity' (in inputs or config)")

        elif operation == 'topic_summary':
            topic_query = inputs.get('topic_query') or self.config.get('topic_query')
            if not topic_query:
                errors.append("Topic summary requires 'topic_query' (in inputs or config)")

        elif operation == 'change_summary':
            # Change summary needs either explicit states or a knowledge graph with history
            has_explicit_states = (
                inputs.get('previous_state') is not None and
                inputs.get('current_state') is not None
            )
            has_kg_with_history = (
                inputs.get('kg_instance') is not None or
                self.kg_hub is not None or
                self.kg_instance is not None
            )
            if not has_explicit_states and not has_kg_with_history:
                errors.append("Change summary requires 'previous_state' and 'current_state', or a knowledge graph with history")

        # Validate summary_level if provided
        summary_level = inputs.get('summary_level') or self.config.get('summary_level', 'detailed')
        valid_levels = ['brief', 'detailed', 'comprehensive']
        if summary_level not in valid_levels:
            errors.append(f"Invalid summary_level: {summary_level}. Must be one of: {', '.join(valid_levels)}")

        # Validate max_length if provided
        if 'max_length' in inputs:
            try:
                max_len = int(inputs['max_length'])
                if max_len < 50:
                    errors.append("max_length must be at least 50 characters")
                if max_len > 10000:
                    errors.append("max_length cannot exceed 10000 characters")
            except (ValueError, TypeError):
                errors.append("max_length must be an integer")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge summarization based on operation type.

        Args:
            inputs: Summarization specification including operation and parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing summarization results:
                - summary: Human-readable summary text
                - key_facts: List of extracted key facts
                - statistics: Numerical summary statistics
                - related_entities: Related entity references
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If summarization fails
        """
        # Get operation and parameters
        operation = inputs.get('operation', self.config.get('operation'))
        summary_level = inputs.get('summary_level', self.config.get('summary_level', 'detailed'))
        max_length = inputs.get('max_length', self.config.get('max_length', 500))
        include_statistics = inputs.get('include_statistics', self.config.get('include_statistics', True))
        include_key_facts = inputs.get('include_key_facts', self.config.get('include_key_facts', True))
        language = inputs.get('language', self.config.get('language', 'en'))

        context.update_progress(10, f"Initializing {operation} with {summary_level} level")
        self.logger.info(f"Executing {operation} at {summary_level} level")

        try:
            # Get the knowledge graph to use
            kg = self._get_knowledge_graph(inputs)

            # Execute the appropriate summarization operation
            if operation == 'entity_summary':
                entity_id = inputs.get('entity_id') or self.config.get('entity_id')
                result = self._summarize_entity(
                    kg, entity_id, summary_level, max_length,
                    include_statistics, include_key_facts, language, context
                )
            elif operation == 'subgraph_summary':
                entity_ids = inputs.get('entity_ids') or self.config.get('entity_ids', [])
                result = self._summarize_subgraph(
                    kg, entity_ids, summary_level, max_length,
                    include_statistics, include_key_facts, language, context
                )
            elif operation == 'path_summary':
                source = inputs.get('source_entity') or self.config.get('source_entity')
                target = inputs.get('target_entity') or self.config.get('target_entity')
                result = self._summarize_path(
                    kg, source, target, summary_level, max_length,
                    include_statistics, include_key_facts, language, context
                )
            elif operation == 'topic_summary':
                topic_query = inputs.get('topic_query') or self.config.get('topic_query')
                result = self._summarize_topic(
                    kg, topic_query, summary_level, max_length,
                    include_statistics, include_key_facts, language, context
                )
            elif operation == 'change_summary':
                previous_state = inputs.get('previous_state')
                current_state = inputs.get('current_state')
                result = self._summarize_changes(
                    kg, previous_state, current_state, summary_level, max_length,
                    include_statistics, include_key_facts, language, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['entity_summary', 'subgraph_summary', 'path_summary', 'topic_summary', 'change_summary']}
                )

            context.update_progress(100, f"Summarization complete: {len(result.get('summary', ''))} chars")
            self.logger.info(f"Knowledge summarization completed: {len(result.get('summary', ''))} characters")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge summarization failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge summarization failed: {str(e)}",
                details={
                    'operation': operation,
                    'summary_level': summary_level,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_knowledge_graph(self, inputs: Dict) -> Optional[Any]:
        """
        Get the knowledge graph instance to use.

        Priority:
        1. kg_instance from inputs
        2. self.kg_hub (UnifiedKGIntegrationHub)
        3. self.kg_instance (UnifiedKnowledgeGraph)
        4. None (fallback mode)
        """
        # Check if a knowledge graph instance was provided in inputs
        if 'kg_instance' in inputs:
            return inputs['kg_instance']

        # Use the hub if available
        if self.kg_hub is not None:
            return self.kg_hub

        # Use the instance created in __init__
        return self.kg_instance

    def _summarize_entity(
        self,
        kg: Any,
        entity_id: str,
        summary_level: str,
        max_length: int,
        include_statistics: bool,
        include_key_facts: bool,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Generate summary for a single entity."""
        context.update_progress(30, f"Retrieving entity data for: {entity_id}")

        # Get entity data from knowledge graph or use fallback
        if kg is not None:
            entity_data = self._get_entity_from_kg(kg, entity_id)
        else:
            entity_data = self._generate_fallback_entity_data(entity_id)

        context.update_progress(50, "Generating entity summary")

        # Generate summary based on level
        summary_text = self._generate_entity_summary_text(
            entity_data, summary_level, max_length, language
        )

        # Extract key facts
        key_facts = []
        if include_key_facts:
            key_facts = self._extract_entity_key_facts(entity_data)

        # Calculate statistics
        statistics = {}
        if include_statistics:
            statistics = self._calculate_entity_statistics(entity_data)

        # Get related entities
        related_entities = entity_data.get('related_entities', [])

        context.update_progress(90, "Entity summary complete")

        return {
            'operation': 'entity_summary',
            'entity_id': entity_id,
            'summary': summary_text,
            'key_facts': key_facts,
            'statistics': statistics,
            'related_entities': related_entities,
            'metadata': {
                'summary_level': summary_level,
                'language': language,
                'max_length': max_length,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id,
                'entity_types': entity_data.get('types', []),
                'confidence': entity_data.get('confidence', 1.0)
            }
        }

    def _summarize_subgraph(
        self,
        kg: Any,
        entity_ids: List[str],
        summary_level: str,
        max_length: int,
        include_statistics: bool,
        include_key_facts: bool,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Generate summary for a subgraph of connected entities."""
        context.update_progress(30, f"Retrieving subgraph data for {len(entity_ids)} entities")

        # Get subgraph data
        if kg is not None:
            subgraph_data = self._get_subgraph_from_kg(kg, entity_ids)
        else:
            subgraph_data = self._generate_fallback_subgraph_data(entity_ids)

        context.update_progress(50, "Analyzing subgraph connections")

        # Generate summary based on level
        summary_text = self._generate_subgraph_summary_text(
            subgraph_data, summary_level, max_length, language
        )

        # Extract key facts
        key_facts = []
        if include_key_facts:
            key_facts = self._extract_subgraph_key_facts(subgraph_data)

        # Calculate statistics
        statistics = {}
        if include_statistics:
            statistics = self._calculate_subgraph_statistics(subgraph_data)

        context.update_progress(90, "Subgraph summary complete")

        return {
            'operation': 'subgraph_summary',
            'entity_ids': entity_ids,
            'summary': summary_text,
            'key_facts': key_facts,
            'statistics': statistics,
            'related_entities': subgraph_data.get('boundary_entities', []),
            'metadata': {
                'summary_level': summary_level,
                'language': language,
                'max_length': max_length,
                'entity_count': len(entity_ids),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id
            }
        }

    def _summarize_path(
        self,
        kg: Any,
        source: str,
        target: str,
        summary_level: str,
        max_length: int,
        include_statistics: bool,
        include_key_facts: bool,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Generate summary for paths between two entities."""
        context.update_progress(30, f"Finding paths from '{source}' to '{target}'")

        # Get path data
        if kg is not None:
            path_data = self._get_paths_from_kg(kg, source, target)
        else:
            path_data = self._generate_fallback_path_data(source, target)

        context.update_progress(50, "Analyzing relationship paths")

        # Generate summary based on level
        summary_text = self._generate_path_summary_text(
            path_data, summary_level, max_length, language
        )

        # Extract key facts
        key_facts = []
        if include_key_facts:
            key_facts = self._extract_path_key_facts(path_data)

        # Calculate statistics
        statistics = {}
        if include_statistics:
            statistics = self._calculate_path_statistics(path_data)

        context.update_progress(90, "Path summary complete")

        return {
            'operation': 'path_summary',
            'source_entity': source,
            'target_entity': target,
            'summary': summary_text,
            'key_facts': key_facts,
            'statistics': statistics,
            'related_entities': path_data.get('intermediate_entities', []),
            'metadata': {
                'summary_level': summary_level,
                'language': language,
                'max_length': max_length,
                'path_count': len(path_data.get('paths', [])),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id
            }
        }

    def _summarize_topic(
        self,
        kg: Any,
        topic_query: str,
        summary_level: str,
        max_length: int,
        include_statistics: bool,
        include_key_facts: bool,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Generate summary for a topic query."""
        context.update_progress(30, f"Searching for topic: {topic_query}")

        # Get topic-related entities
        if kg is not None:
            topic_data = self._get_topic_from_kg(kg, topic_query)
        else:
            topic_data = self._generate_fallback_topic_data(topic_query)

        context.update_progress(50, "Synthesizing topic summary")

        # Generate summary based on level
        summary_text = self._generate_topic_summary_text(
            topic_data, topic_query, summary_level, max_length, language
        )

        # Extract key facts
        key_facts = []
        if include_key_facts:
            key_facts = self._extract_topic_key_facts(topic_data)

        # Calculate statistics
        statistics = {}
        if include_statistics:
            statistics = self._calculate_topic_statistics(topic_data)

        context.update_progress(90, "Topic summary complete")

        return {
            'operation': 'topic_summary',
            'topic_query': topic_query,
            'summary': summary_text,
            'key_facts': key_facts,
            'statistics': statistics,
            'related_entities': topic_data.get('related_entities', []),
            'metadata': {
                'summary_level': summary_level,
                'language': language,
                'max_length': max_length,
                'entity_count': len(topic_data.get('entities', [])),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id
            }
        }

    def _summarize_changes(
        self,
        kg: Any,
        previous_state: Optional[Dict],
        current_state: Optional[Dict],
        summary_level: str,
        max_length: int,
        include_statistics: bool,
        include_key_facts: bool,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Generate summary of changes between two knowledge states."""
        context.update_progress(30, "Comparing knowledge states")

        # Get change data
        if previous_state is not None and current_state is not None:
            change_data = self._compare_states(previous_state, current_state)
        elif kg is not None:
            change_data = self._get_changes_from_kg(kg)
        else:
            change_data = self._generate_fallback_change_data()

        context.update_progress(50, "Analyzing changes")

        # Generate summary based on level
        summary_text = self._generate_change_summary_text(
            change_data, summary_level, max_length, language
        )

        # Extract key facts
        key_facts = []
        if include_key_facts:
            key_facts = self._extract_change_key_facts(change_data)

        # Calculate statistics
        statistics = {}
        if include_statistics:
            statistics = self._calculate_change_statistics(change_data)

        context.update_progress(90, "Change summary complete")

        return {
            'operation': 'change_summary',
            'summary': summary_text,
            'key_facts': key_facts,
            'statistics': statistics,
            'related_entities': change_data.get('affected_entities', []),
            'metadata': {
                'summary_level': summary_level,
                'language': language,
                'max_length': max_length,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'execution_id': self.execution_id,
                'changes_detected': change_data.get('total_changes', 0) > 0
            }
        }

    # Knowledge Graph Data Access Methods

    def _get_entity_from_kg(self, kg: Any, entity_id: str) -> Dict[str, Any]:
        """Retrieve entity data from knowledge graph."""
        try:
            # Try different methods based on available APIs
            if hasattr(kg, 'get_entity_profile'):
                profile = kg.get_entity_profile(entity_id)
                if profile:
                    if hasattr(profile, 'to_dict'):
                        return profile.to_dict()
                    return profile

            if hasattr(kg, 'get_entity'):
                entity = kg.get_entity(entity_id)
                if entity:
                    return entity

            if hasattr(kg, 'get_neighbors'):
                neighbors = kg.get_neighbors(entity_id, depth=1)
                return {
                    'id': entity_id,
                    'name': entity_id,
                    'types': [],
                    'properties': {},
                    'related_entities': list(neighbors.get('nodes', {}).keys()) if isinstance(neighbors, dict) else [],
                    'relationships': neighbors.get('edges', []) if isinstance(neighbors, dict) else []
                }

            # Fallback: create minimal structure
            return self._generate_fallback_entity_data(entity_id)

        except Exception as e:
            self.logger.warning(f"Error retrieving entity from KG: {e}")
            return self._generate_fallback_entity_data(entity_id)

    def _get_subgraph_from_kg(self, kg: Any, entity_ids: List[str]) -> Dict[str, Any]:
        """Retrieve subgraph data from knowledge graph."""
        try:
            entities = []
            relationships = []
            boundary_entities = set()

            for entity_id in entity_ids:
                entity_data = self._get_entity_from_kg(kg, entity_id)
                entities.append(entity_data)

                # Collect relationships within the subgraph
                rels = entity_data.get('relationships', [])
                for rel in rels:
                    target = rel.get('target') if isinstance(rel, dict) else getattr(rel, 'target', None)
                    if target and target in entity_ids:
                        relationships.append(rel)
                    elif target:
                        boundary_entities.add(target)

            return {
                'entities': entities,
                'relationships': relationships,
                'boundary_entities': list(boundary_entities),
                'internal_connections': len(relationships)
            }

        except Exception as e:
            self.logger.warning(f"Error retrieving subgraph from KG: {e}")
            return self._generate_fallback_subgraph_data(entity_ids)

    def _get_paths_from_kg(self, kg: Any, source: str, target: str) -> Dict[str, Any]:
        """Retrieve path data from knowledge graph."""
        try:
            if hasattr(kg, 'find_paths'):
                paths = kg.find_paths(source, target, max_length=5)
            elif hasattr(kg, 'get_paths'):
                paths = kg.get_paths(source, target)
            else:
                paths = []

            # Extract intermediate entities
            intermediate_entities = set()
            for path in paths:
                path_nodes = path if isinstance(path, list) else path.get('nodes', [])
                for node in path_nodes:
                    node_id = node if isinstance(node, str) else node.get('id', node.get('name', ''))
                    if node_id and node_id != source and node_id != target:
                        intermediate_entities.add(node_id)

            return {
                'source': source,
                'target': target,
                'paths': paths,
                'intermediate_entities': list(intermediate_entities),
                'path_count': len(paths)
            }

        except Exception as e:
            self.logger.warning(f"Error retrieving paths from KG: {e}")
            return self._generate_fallback_path_data(source, target)

    def _get_topic_from_kg(self, kg: Any, topic_query: str) -> Dict[str, Any]:
        """Retrieve topic-related data from knowledge graph."""
        try:
            # Try semantic search if available
            if hasattr(kg, 'semantic_search'):
                results = kg.semantic_search(topic_query, limit=20)
                entities = results if isinstance(results, list) else results.get('results', [])
            elif hasattr(kg, 'search'):
                results = kg.search(topic_query)
                entities = results if isinstance(results, list) else results.get('results', [])
            else:
                # Fallback: query triples
                entities = []

            # Extract related entities by relationship
            related_entities = set()
            for entity in entities:
                rels = entity.get('relationships', []) if isinstance(entity, dict) else []
                for rel in rels:
                    target = rel.get('target') if isinstance(rel, dict) else getattr(rel, 'target', None)
                    if target:
                        related_entities.add(target)

            return {
                'topic': topic_query,
                'entities': entities,
                'related_entities': list(related_entities),
                'entity_count': len(entities)
            }

        except Exception as e:
            self.logger.warning(f"Error retrieving topic from KG: {e}")
            return self._generate_fallback_topic_data(topic_query)

    def _get_changes_from_kg(self, kg: Any) -> Dict[str, Any]:
        """Retrieve change history from knowledge graph."""
        try:
            if hasattr(kg, 'get_change_history'):
                changes = kg.get_change_history()
            elif hasattr(kg, 'get_changes'):
                changes = kg.get_changes()
            else:
                changes = []

            return {
                'changes': changes,
                'total_changes': len(changes),
                'affected_entities': list(set(
                    c.get('entity_id') for c in changes if isinstance(c, dict) and 'entity_id' in c
                ))
            }

        except Exception as e:
            self.logger.warning(f"Error retrieving changes from KG: {e}")
            return self._generate_fallback_change_data()

    def _compare_states(self, previous: Dict, current: Dict) -> Dict[str, Any]:
        """Compare two knowledge states and identify changes."""
        changes = []
        affected_entities = set()

        prev_entities = {e.get('id', e.get('name', '')): e for e in previous.get('entities', [])}
        curr_entities = {e.get('id', e.get('name', '')): e for e in current.get('entities', [])}

        # Find added entities
        for entity_id in curr_entities:
            if entity_id not in prev_entities:
                changes.append({
                    'type': 'added',
                    'entity_id': entity_id,
                    'entity': curr_entities[entity_id]
                })
                affected_entities.add(entity_id)

        # Find removed entities
        for entity_id in prev_entities:
            if entity_id not in curr_entities:
                changes.append({
                    'type': 'removed',
                    'entity_id': entity_id,
                    'entity': prev_entities[entity_id]
                })
                affected_entities.add(entity_id)

        # Find modified entities
        for entity_id in curr_entities:
            if entity_id in prev_entities:
                prev_props = prev_entities[entity_id].get('properties', {})
                curr_props = curr_entities[entity_id].get('properties', {})
                if prev_props != curr_props:
                    changes.append({
                        'type': 'modified',
                        'entity_id': entity_id,
                        'previous': prev_entities[entity_id],
                        'current': curr_entities[entity_id]
                    })
                    affected_entities.add(entity_id)

        return {
            'changes': changes,
            'total_changes': len(changes),
            'affected_entities': list(affected_entities)
        }

    # Fallback Data Generation Methods

    def _generate_fallback_entity_data(self, entity_id: str) -> Dict[str, Any]:
        """Generate fallback entity data when KG is unavailable."""
        return {
            'id': entity_id,
            'name': entity_id,
            'types': ['Unknown'],
            'properties': {
                'description': f'Entity {entity_id} (knowledge graph unavailable)',
                'fallback': True
            },
            'relationships': [],
            'related_entities': [],
            'sources': ['fallback'],
            'confidence': 0.5,
            'fallback': True
        }

    def _generate_fallback_subgraph_data(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Generate fallback subgraph data when KG is unavailable."""
        entities = [self._generate_fallback_entity_data(eid) for eid in entity_ids]
        return {
            'entities': entities,
            'relationships': [],
            'boundary_entities': [],
            'internal_connections': 0,
            'fallback': True
        }

    def _generate_fallback_path_data(self, source: str, target: str) -> Dict[str, Any]:
        """Generate fallback path data when KG is unavailable."""
        return {
            'source': source,
            'target': target,
            'paths': [],
            'intermediate_entities': [],
            'path_count': 0,
            'fallback': True
        }

    def _generate_fallback_topic_data(self, topic_query: str) -> Dict[str, Any]:
        """Generate fallback topic data when KG is unavailable."""
        return {
            'topic': topic_query,
            'entities': [],
            'related_entities': [],
            'entity_count': 0,
            'fallback': True
        }

    def _generate_fallback_change_data(self) -> Dict[str, Any]:
        """Generate fallback change data when KG is unavailable."""
        return {
            'changes': [],
            'total_changes': 0,
            'affected_entities': [],
            'fallback': True
        }

    # Summary Text Generation Methods

    def _generate_entity_summary_text(self, entity_data: Dict, level: str, max_length: int, language: str) -> str:
        """Generate summary text for an entity."""
        name = entity_data.get('name', entity_data.get('id', 'Unknown'))
        types = entity_data.get('types', [])
        properties = entity_data.get('properties', {})
        relationships = entity_data.get('relationships', [])
        is_fallback = entity_data.get('fallback', False)

        if level == 'brief':
            type_str = f", a {', '.join(types)}" if types else ""
            rel_count = len(relationships)
            summary = f"{name}{type_str} with {rel_count} relationship(s)."

        elif level == 'detailed':
            lines = [f"## {name}"]
            if types:
                lines.append(f"**Type(s):** {', '.join(types)}")

            # Add description if available
            desc = properties.get('description', properties.get('summary', ''))
            if desc:
                lines.append(f"\n{desc}")
            else:
                lines.append(f"\nEntity with {len(relationships)} relationships.")

            # Add key properties (excluding internal/private ones)
            key_props = {k: v for k, v in properties.items() if not k.startswith('_') and k != 'description'}
            if key_props:
                lines.append("\n**Key Properties:**")
                for k, v in list(key_props.items())[:5]:
                    lines.append(f"- {k}: {v}")

            if is_fallback:
                lines.append("\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        else:  # comprehensive
            lines = [f"# {name}", f"**ID:** {entity_data.get('id', 'N/A')}"]

            if types:
                lines.append(f"**Types:** {', '.join(types)}")

            lines.append(f"**Confidence:** {entity_data.get('confidence', 'N/A')}")

            desc = properties.get('description', properties.get('summary', ''))
            if desc:
                lines.append(f"\n## Description\n{desc}")

            # All properties
            key_props = {k: v for k, v in properties.items() if not k.startswith('_')}
            if key_props:
                lines.append("\n## Properties")
                for k, v in key_props.items():
                    lines.append(f"- **{k}:** {v}")

            # Relationships
            if relationships:
                lines.append(f"\n## Relationships ({len(relationships)})")
                for rel in relationships[:20]:  # Limit to 20
                    if isinstance(rel, dict):
                        pred = rel.get('predicate', 'related_to')
                        target = rel.get('target', 'unknown')
                        lines.append(f"- {pred} -> {target}")
                    else:
                        lines.append(f"- {str(rel)}")
                if len(relationships) > 20:
                    lines.append(f"- ... and {len(relationships) - 20} more")

            # Sources
            sources = entity_data.get('sources', [])
            if sources:
                lines.append(f"\n## Sources")
                lines.append(f"{', '.join(str(s) for s in sources)}")

            if is_fallback:
                lines.append("\n---\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    def _generate_subgraph_summary_text(self, subgraph_data: Dict, level: str, max_length: int, language: str) -> str:
        """Generate summary text for a subgraph."""
        entities = subgraph_data.get('entities', [])
        relationships = subgraph_data.get('relationships', [])
        boundary = subgraph_data.get('boundary_entities', [])
        is_fallback = subgraph_data.get('fallback', False)

        entity_names = [e.get('name', e.get('id', 'Unknown')) for e in entities]

        if level == 'brief':
            summary = f"Subgraph with {len(entities)} entities and {len(relationships)} internal connections."

        elif level == 'detailed':
            lines = ["## Subgraph Summary", f"**Entities:** {len(entities)}", f"**Internal Connections:** {len(relationships)}"]

            if entity_names:
                lines.append(f"\n**Core Entities:** {', '.join(entity_names[:10])}")
                if len(entity_names) > 10:
                    lines.append(f"*... and {len(entity_names) - 10} more*")

            if boundary:
                lines.append(f"\n**External Connections:** {len(boundary)} boundary entities")

            if is_fallback:
                lines.append("\n*Note: Running in fallback mode*")

            summary = '\n'.join(lines)

        else:  # comprehensive
            lines = ["# Subgraph Analysis"]
            lines.append(f"**Total Entities:** {len(entities)}")
            lines.append(f"**Internal Relationships:** {len(relationships)}")
            lines.append(f"**Boundary Entities:** {len(boundary)}")

            # List all entities
            lines.append("\n## Entities")
            for entity in entities:
                name = entity.get('name', entity.get('id', 'Unknown'))
                types = entity.get('types', [])
                type_str = f" ({', '.join(types)})" if types else ""
                lines.append(f"- {name}{type_str}")

            # Relationship summary
            if relationships:
                lines.append(f"\n## Relationships")
                rel_types = defaultdict(int)
                for rel in relationships:
                    pred = rel.get('predicate', 'related_to') if isinstance(rel, dict) else 'related_to'
                    rel_types[pred] += 1
                for pred, count in sorted(rel_types.items(), key=lambda x: -x[1]):
                    lines.append(f"- {pred}: {count}")

            if boundary:
                lines.append(f"\n## Boundary Connections")
                lines.append(f"Connected to: {', '.join(boundary[:20])}")
                if len(boundary) > 20:
                    lines.append(f"*... and {len(boundary) - 20} more*")

            if is_fallback:
                lines.append("\n---\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    def _generate_path_summary_text(self, path_data: Dict, level: str, max_length: int, language: str) -> str:
        """Generate summary text for paths between entities."""
        source = path_data.get('source', 'Unknown')
        target = path_data.get('target', 'Unknown')
        paths = path_data.get('paths', [])
        is_fallback = path_data.get('fallback', False)

        if level == 'brief':
            if paths:
                summary = f"Found {len(paths)} path(s) from '{source}' to '{target}'."
            else:
                summary = f"No paths found from '{source}' to '{target}'."

        elif level == 'detailed':
            lines = [f"## Path Summary: {source} -> {target}"]

            if paths:
                lines.append(f"**Total Paths:** {len(paths)}")

                # Describe shortest path
                shortest = min(paths, key=lambda p: len(p) if isinstance(p, list) else len(p.get('nodes', [])))
                path_len = len(shortest) if isinstance(shortest, list) else len(shortest.get('nodes', []))
                lines.append(f"**Shortest Path Length:** {path_len} hops")

                # List intermediate entities
                intermediate = path_data.get('intermediate_entities', [])
                if intermediate:
                    lines.append(f"\n**Key Connectors:** {', '.join(intermediate[:10])}")

                lines.append(f"\n**Relationship Chain:** The entities are connected through a chain of relationships spanning {path_len} steps.")
            else:
                lines.append("No connecting paths found between these entities.")

            if is_fallback:
                lines.append("\n*Note: Running in fallback mode*")

            summary = '\n'.join(lines)

        else:  # comprehensive
            lines = [f"# Path Analysis: {source} -> {target}"]

            if paths:
                lines.append(f"**Total Paths Discovered:** {len(paths)}")

                # Path statistics
                path_lengths = [len(p) if isinstance(p, list) else len(p.get('nodes', [])) for p in paths]
                lines.append(f"**Path Length Range:** {min(path_lengths)} - {max(path_lengths)} hops")
                lines.append(f"**Average Path Length:** {sum(path_lengths) / len(path_lengths):.1f} hops")

                # Detailed path descriptions
                lines.append("\n## Paths")
                for i, path in enumerate(paths[:5], 1):  # Show first 5 paths
                    if isinstance(path, list):
                        path_str = ' -> '.join(str(node) for node in path)
                    else:
                        nodes = path.get('nodes', [])
                        path_str = ' -> '.join(str(n.get('name', n.get('id', str(n)))) for n in nodes)
                    lines.append(f"\n### Path {i}")
                    lines.append(path_str)

                if len(paths) > 5:
                    lines.append(f"\n*... and {len(paths) - 5} additional paths*")

                # Intermediate entities
                intermediate = path_data.get('intermediate_entities', [])
                if intermediate:
                    lines.append(f"\n## Bridge Entities ({len(intermediate)})")
                    for entity in intermediate:
                        lines.append(f"- {entity}")
            else:
                lines.append("No connecting paths found between these entities.")
                lines.append("\nPossible reasons:")
                lines.append("- Entities are in disconnected graph components")
                lines.append("- Maximum path length limit reached")
                lines.append("- Insufficient relationship data")

            if is_fallback:
                lines.append("\n---\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    def _generate_topic_summary_text(self, topic_data: Dict, topic_query: str, level: str, max_length: int, language: str) -> str:
        """Generate summary text for a topic."""
        entities = topic_data.get('entities', [])
        related = topic_data.get('related_entities', [])
        is_fallback = topic_data.get('fallback', False)

        if level == 'brief':
            summary = f"Topic '{topic_query}' involves {len(entities)} entities with {len(related)} related connections."

        elif level == 'detailed':
            lines = [f"## Topic: {topic_query}"]
            lines.append(f"**Primary Entities:** {len(entities)}")
            lines.append(f"**Related Entities:** {len(related)}")

            if entities:
                lines.append("\n**Key Entities:**")
                for entity in entities[:10]:
                    if isinstance(entity, dict):
                        name = entity.get('name', entity.get('id', 'Unknown'))
                        desc = entity.get('properties', {}).get('description', '')
                        if desc:
                            lines.append(f"- **{name}:** {desc[:100]}..." if len(desc) > 100 else f"- **{name}:** {desc}")
                        else:
                            lines.append(f"- {name}")
                    else:
                        lines.append(f"- {str(entity)}")

            if is_fallback:
                lines.append("\n*Note: Running in fallback mode*")

            summary = '\n'.join(lines)

        else:  # comprehensive
            lines = [f"# Topic Analysis: {topic_query}"]
            lines.append(f"**Query:** {topic_query}")
            lines.append(f"**Matching Entities:** {len(entities)}")
            lines.append(f"**Extended Network:** {len(related)} related entities")

            # Categorize entities by type if available
            if entities:
                by_type = defaultdict(list)
                for entity in entities:
                    if isinstance(entity, dict):
                        types = entity.get('types', ['Unknown'])
                        name = entity.get('name', entity.get('id', 'Unknown'))
                        for t in types:
                            by_type[t].append(name)

                lines.append("\n## Entity Categories")
                for type_name, names in sorted(by_type.items(), key=lambda x: -len(x[1])):
                    lines.append(f"\n### {type_name} ({len(names)})")
                    for name in names[:15]:
                        lines.append(f"- {name}")
                    if len(names) > 15:
                        lines.append(f"- ... and {len(names) - 15} more")

            # Related entities
            if related:
                lines.append(f"\n## Extended Network ({len(related)} entities)")
                lines.append("Entities connected to the primary topic entities:")
                for entity_id in related[:20]:
                    lines.append(f"- {entity_id}")
                if len(related) > 20:
                    lines.append(f"- ... and {len(related) - 20} more")

            if is_fallback:
                lines.append("\n---\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    def _generate_change_summary_text(self, change_data: Dict, level: str, max_length: int, language: str) -> str:
        """Generate summary text for changes."""
        changes = change_data.get('changes', [])
        total = change_data.get('total_changes', 0)
        affected = change_data.get('affected_entities', [])
        is_fallback = change_data.get('fallback', False)

        # Count by type
        added = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'added')
        removed = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'removed')
        modified = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'modified')

        if level == 'brief':
            if total > 0:
                summary = f"Detected {total} change(s): {added} added, {removed} removed, {modified} modified."
            else:
                summary = "No changes detected between knowledge states."

        elif level == 'detailed':
            lines = ["## Change Summary"]

            if total > 0:
                lines.append(f"**Total Changes:** {total}")
                lines.append(f"- Added: {added}")
                lines.append(f"- Removed: {removed}")
                lines.append(f"- Modified: {modified}")
                lines.append(f"**Affected Entities:** {len(affected)}")

                # List recent changes
                lines.append("\n**Recent Changes:**")
                for change in changes[:10]:
                    if isinstance(change, dict):
                        change_type = change.get('type', 'unknown')
                        entity_id = change.get('entity_id', 'unknown')
                        lines.append(f"- {change_type.upper()}: {entity_id}")
            else:
                lines.append("No changes detected.")
                lines.append("\nThe knowledge graph states are identical.")

            if is_fallback:
                lines.append("\n*Note: Running in fallback mode*")

            summary = '\n'.join(lines)

        else:  # comprehensive
            lines = ["# Knowledge Graph Change Analysis"]

            # Overview
            lines.append("## Overview")
            lines.append(f"**Total Changes:** {total}")
            if total > 0:
                lines.append(f"- **Entities Added:** {added}")
                lines.append(f"- **Entities Removed:** {removed}")
                lines.append(f"- **Entities Modified:** {modified}")
                lines.append(f"**Unique Affected Entities:** {len(affected)}")

                # Detailed change list
                if added > 0:
                    lines.append("\n## Added Entities")
                    for change in changes:
                        if isinstance(change, dict) and change.get('type') == 'added':
                            entity_id = change.get('entity_id', 'unknown')
                            lines.append(f"- {entity_id}")

                if removed > 0:
                    lines.append("\n## Removed Entities")
                    for change in changes:
                        if isinstance(change, dict) and change.get('type') == 'removed':
                            entity_id = change.get('entity_id', 'unknown')
                            lines.append(f"- {entity_id}")

                if modified > 0:
                    lines.append("\n## Modified Entities")
                    for change in changes:
                        if isinstance(change, dict) and change.get('type') == 'modified':
                            entity_id = change.get('entity_id', 'unknown')
                            lines.append(f"- {entity_id}")

                # Affected entities summary
                lines.append(f"\n## Affected Entities ({len(affected)})")
                for entity_id in affected[:30]:
                    lines.append(f"- {entity_id}")
                if len(affected) > 30:
                    lines.append(f"- ... and {len(affected) - 30} more")
            else:
                lines.append("\nNo changes detected between the compared knowledge states.")
                lines.append("\n### Possible Reasons:")
                lines.append("- The knowledge graph has not been modified")
                lines.append("- The comparison window is too narrow")
                lines.append("- Changes are in metadata not tracked")

            if is_fallback:
                lines.append("\n---\n*Note: Running in fallback mode (knowledge graph unavailable)*")

            summary = '\n'.join(lines)

        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    # Key Facts Extraction Methods

    def _extract_entity_key_facts(self, entity_data: Dict) -> List[str]:
        """Extract key facts from entity data."""
        facts = []

        name = entity_data.get('name', entity_data.get('id', 'Unknown'))
        types = entity_data.get('types', [])

        if types:
            facts.append(f"{name} is classified as: {', '.join(types)}")

        relationships = entity_data.get('relationships', [])
        if relationships:
            facts.append(f"Has {len(relationships)} relationships with other entities")

        properties = entity_data.get('properties', {})
        for key, value in list(properties.items())[:5]:
            if not key.startswith('_') and key != 'description':
                facts.append(f"{key}: {value}")

        return facts

    def _extract_subgraph_key_facts(self, subgraph_data: Dict) -> List[str]:
        """Extract key facts from subgraph data."""
        facts = []

        entities = subgraph_data.get('entities', [])
        relationships = subgraph_data.get('relationships', [])
        boundary = subgraph_data.get('boundary_entities', [])

        facts.append(f"Subgraph contains {len(entities)} entities")
        facts.append(f"{len(relationships)} internal relationships connect these entities")

        if boundary:
            facts.append(f"Connects to {len(boundary)} external entities")

        # Most connected entities
        if entities:
            connection_counts = defaultdict(int)
            for rel in relationships:
                if isinstance(rel, dict):
                    target = rel.get('target')
                    if target:
                        connection_counts[target] += 1

            if connection_counts:
                most_connected = max(connection_counts.items(), key=lambda x: x[1])
                facts.append(f"Most connected: {most_connected[0]} with {most_connected[1]} relationships")

        return facts

    def _extract_path_key_facts(self, path_data: Dict) -> List[str]:
        """Extract key facts from path data."""
        facts = []

        paths = path_data.get('paths', [])
        source = path_data.get('source', 'Unknown')
        target = path_data.get('target', 'Unknown')
        intermediate = path_data.get('intermediate_entities', [])

        if paths:
            facts.append(f"Found {len(paths)} path(s) from '{source}' to '{target}'")

            path_lengths = [len(p) if isinstance(p, list) else len(p.get('nodes', [])) for p in paths]
            facts.append(f"Shortest path: {min(path_lengths)} hops")
            facts.append(f"Longest path: {max(path_lengths)} hops")

            if intermediate:
                facts.append(f"{len(intermediate)} intermediate entities connect these nodes")
        else:
            facts.append(f"No connecting paths found between '{source}' and '{target}'")

        return facts

    def _extract_topic_key_facts(self, topic_data: Dict) -> List[str]:
        """Extract key facts from topic data."""
        facts = []

        entities = topic_data.get('entities', [])
        related = topic_data.get('related_entities', [])

        facts.append(f"Topic matches {len(entities)} primary entities")
        facts.append(f"Extended network includes {len(related)} related entities")

        # Type distribution
        if entities:
            type_counts = defaultdict(int)
            for entity in entities:
                if isinstance(entity, dict):
                    for t in entity.get('types', ['Unknown']):
                        type_counts[t] += 1

            if type_counts:
                most_common = max(type_counts.items(), key=lambda x: x[1])
                facts.append(f"Most common entity type: {most_common[0]} ({most_common[1]} entities)")

        return facts

    def _extract_change_key_facts(self, change_data: Dict) -> List[str]:
        """Extract key facts from change data."""
        facts = []

        changes = change_data.get('changes', [])
        total = change_data.get('total_changes', 0)
        affected = change_data.get('affected_entities', [])

        if total > 0:
            facts.append(f"{total} total changes detected")

            added = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'added')
            removed = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'removed')
            modified = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'modified')

            if added:
                facts.append(f"{added} new entities added")
            if removed:
                facts.append(f"{removed} entities removed")
            if modified:
                facts.append(f"{modified} entities modified")

            facts.append(f"{len(affected)} unique entities affected")
        else:
            facts.append("No changes detected")

        return facts

    # Statistics Calculation Methods

    def _calculate_entity_statistics(self, entity_data: Dict) -> Dict[str, Any]:
        """Calculate statistics for entity data."""
        return {
            'relationship_count': len(entity_data.get('relationships', [])),
            'property_count': len(entity_data.get('properties', {})),
            'type_count': len(entity_data.get('types', [])),
            'source_count': len(entity_data.get('sources', [])),
            'confidence_score': entity_data.get('confidence', 0),
            'has_description': 'description' in entity_data.get('properties', {}),
            'is_fallback': entity_data.get('fallback', False)
        }

    def _calculate_subgraph_statistics(self, subgraph_data: Dict) -> Dict[str, Any]:
        """Calculate statistics for subgraph data."""
        entities = subgraph_data.get('entities', [])
        relationships = subgraph_data.get('relationships', [])
        boundary = subgraph_data.get('boundary_entities', [])

        # Calculate density
        n = len(entities)
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = len(relationships) / max_edges if max_edges > 0 else 0

        return {
            'entity_count': n,
            'relationship_count': len(relationships),
            'boundary_count': len(boundary),
            'internal_connections': subgraph_data.get('internal_connections', 0),
            'density': round(density, 3),
            'is_fallback': subgraph_data.get('fallback', False)
        }

    def _calculate_path_statistics(self, path_data: Dict) -> Dict[str, Any]:
        """Calculate statistics for path data."""
        paths = path_data.get('paths', [])

        if not paths:
            return {
                'path_count': 0,
                'shortest_path_length': 0,
                'longest_path_length': 0,
                'average_path_length': 0,
                'intermediate_entities': 0,
                'is_fallback': path_data.get('fallback', False)
            }

        path_lengths = [len(p) if isinstance(p, list) else len(p.get('nodes', [])) for p in paths]

        return {
            'path_count': len(paths),
            'shortest_path_length': min(path_lengths),
            'longest_path_length': max(path_lengths),
            'average_path_length': round(sum(path_lengths) / len(path_lengths), 2),
            'intermediate_entities': len(path_data.get('intermediate_entities', [])),
            'is_fallback': path_data.get('fallback', False)
        }

    def _calculate_topic_statistics(self, topic_data: Dict) -> Dict[str, Any]:
        """Calculate statistics for topic data."""
        entities = topic_data.get('entities', [])

        # Type distribution
        type_counts = defaultdict(int)
        for entity in entities:
            if isinstance(entity, dict):
                for t in entity.get('types', ['Unknown']):
                    type_counts[t] += 1

        return {
            'entity_count': len(entities),
            'related_entity_count': len(topic_data.get('related_entities', [])),
            'type_distribution': dict(type_counts),
            'primary_types': sorted(type_counts.keys())[:5],
            'is_fallback': topic_data.get('fallback', False)
        }

    def _calculate_change_statistics(self, change_data: Dict) -> Dict[str, Any]:
        """Calculate statistics for change data."""
        changes = change_data.get('changes', [])

        added = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'added')
        removed = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'removed')
        modified = sum(1 for c in changes if isinstance(c, dict) and c.get('type') == 'modified')

        return {
            'total_changes': len(changes),
            'added_count': added,
            'removed_count': removed,
            'modified_count': modified,
            'affected_entities': len(change_data.get('affected_entities', [])),
            'net_change': added - removed,
            'is_fallback': change_data.get('fallback', False)
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration including:
        - operation: Type of summarization to perform
        - summary_level: Detail level of the output
        - entity_id/entity_ids: Target entity specification
        - source_entity/target_entity: For path summaries
        - topic_query: For topic summaries
        - max_length: Maximum summary length
        - include_statistics: Whether to include statistics
        - include_key_facts: Whether to extract key facts
        - language: Output language code
        """
        return {
            "type": "object",
            "title": "Knowledge Summarization Configuration",
            "description": "Configure knowledge graph summarization parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of summarization to perform",
                    "enum": ["entity_summary", "subgraph_summary", "path_summary", "topic_summary", "change_summary"],
                    "enumNames": [
                        "Entity Summary - Summarize a single entity",
                        "Subgraph Summary - Summarize a group of connected entities",
                        "Path Summary - Summarize paths between two entities",
                        "Topic Summary - Summarize knowledge around a topic",
                        "Change Summary - Summarize changes between knowledge states"
                    ],
                    "default": "entity_summary"
                },
                "summary_level": {
                    "type": "string",
                    "title": "Summary Level",
                    "description": "Detail level of the generated summary",
                    "enum": ["brief", "detailed", "comprehensive"],
                    "enumNames": [
                        "Brief - 1-2 sentence summary",
                        "Detailed - Paragraph with key facts",
                        "Comprehensive - Full analysis with all details"
                    ],
                    "default": "detailed"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "ID of the entity to summarize (for entity_summary operation)"
                },
                "entity_ids": {
                    "type": "array",
                    "title": "Entity IDs",
                    "description": "List of entity IDs to include in subgraph summary",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "source_entity": {
                    "type": "string",
                    "title": "Source Entity",
                    "description": "Starting entity for path summary"
                },
                "target_entity": {
                    "type": "string",
                    "title": "Target Entity",
                    "description": "Target entity for path summary"
                },
                "topic_query": {
                    "type": "string",
                    "title": "Topic Query",
                    "description": "Topic or search query for topic summary"
                },
                "max_length": {
                    "type": "integer",
                    "title": "Maximum Length",
                    "description": "Maximum summary length in characters",
                    "minimum": 50,
                    "maximum": 10000,
                    "default": 500
                },
                "include_statistics": {
                    "type": "boolean",
                    "title": "Include Statistics",
                    "description": "Include numerical statistics in the output",
                    "default": True
                },
                "include_key_facts": {
                    "type": "boolean",
                    "title": "Include Key Facts",
                    "description": "Extract and include key facts separately",
                    "default": True
                },
                "language": {
                    "type": "string",
                    "title": "Output Language",
                    "description": "Language code for summary output (e.g., 'en', 'es', 'fr')",
                    "default": "en"
                },
                "backend": {
                    "type": "string",
                    "title": "Storage Backend",
                    "description": "Backend storage for knowledge graph (when creating new instance)",
                    "enum": ["networkx", "memory"],
                    "enumNames": [
                        "NetworkX (recommended)",
                        "In-Memory"
                    ],
                    "default": "networkx"
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["entity_summary"]}
                            },
                            "required": ["entity_id"],
                            "description": "Summarize a single entity's properties and relationships"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["subgraph_summary"]}
                            },
                            "required": ["entity_ids"],
                            "description": "Summarize a connected subgraph of entities"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["path_summary"]}
                            },
                            "required": ["source_entity", "target_entity"],
                            "description": "Summarize relationship paths between two entities"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["topic_summary"]}
                            },
                            "required": ["topic_query"],
                            "description": "Summarize knowledge related to a topic"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["change_summary"]}
                            },
                            "description": "Summarize changes between knowledge states (requires previous_state and current_state inputs, or knowledge graph with history)"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least one knowledge graph interface is available,
            or if fallback mode is functional. False only if critical errors occur.
        """
        try:
            # Node can work in fallback mode, so it's generally healthy
            # Check if we have KG access for enhanced functionality
            return True
        except Exception:
            return False
