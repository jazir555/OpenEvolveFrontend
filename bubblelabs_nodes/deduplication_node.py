"""
Deduplication Node for BubbleLabs Integration

Finds and merges duplicate entities and triples in knowledge graphs using
multiple deduplication strategies including semantic matching, clustering,
and rule-based standardization.

Features:
- Find duplicate entities by name, aliases, and similarity
- Find duplicate triples across the knowledge graph
- Merge duplicate entities with configurable strategies
- Resolve conflicts during merge operations
- Generate comprehensive deduplication reports
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import asyncio
from difflib import SequenceMatcher
from collections import defaultdict

from .base_node import BubbleLabsNode, NodeExecutionError


class DeduplicationNode(BubbleLabsNode):
    """
    Find and merge duplicate entities and triples in knowledge graphs.

    Supports multiple deduplication strategies:
    - Semantic matching using embeddings
    - Rule-based name standardization
    - LM-based clustering for large datasets
    - Fast hash-based matching

    Provides configurable merge strategies and comprehensive reporting.
    """

    # Node metadata
    DISPLAY_NAME = "Deduplication"
    DESCRIPTION = "Find and merge duplicate entities and triples in knowledge graphs"
    ICON = "deduplication"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports with fallbacks
        self.UnifiedDeduplicationManager = self._safe_import_dedup_manager()
        self.UnifiedKGIntegrationHub = self._safe_import_kg_hub()

        # Initialize dedup manager if available
        self.dedup_manager = None
        if self.UnifiedDeduplicationManager:
            try:
                self.dedup_manager = self.UnifiedDeduplicationManager()
                self.logger.info("UnifiedDeduplicationManager initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedDeduplicationManager: {e}")
                self.dedup_manager = None

        # Initialize KG hub if available
        self.kg_hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

    def _safe_import_dedup_manager(self):
        """Safely import UnifiedDeduplicationManager."""
        module = self.safe_import(
            'knowledge_engine.deduplication.unified_manager',
            fallback_value=None,
            error_msg="Deduplication manager not available"
        )
        if module:
            return getattr(module, 'UnifiedDeduplicationManager', None)
        return None

    def _safe_import_kg_hub(self):
        """Safely import UnifiedKGIntegrationHub."""
        module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="KG Integration Hub not available"
        )
        if module:
            return getattr(module, 'UnifiedKGIntegrationHub', None)
        return None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (one of):
            - knowledge_graph_id: str - ID of knowledge graph to deduplicate
            - entities: list - List of entities to deduplicate
            - triples: list - List of triples to deduplicate

        Optional:
            - operation: str - Override configured operation
            - similarity_threshold: float - Override similarity threshold
        """
        errors = []

        # Check that at least one input source is provided
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_entities = 'entities' in inputs and isinstance(inputs['entities'], list)
        has_triples = 'triples' in inputs and isinstance(inputs['triples'], list)

        if not (has_kg_id or has_entities or has_triples):
            errors.append(
                "Must provide one of: 'knowledge_graph_id', 'entities', or 'triples'"
            )

        # Validate operation if provided
        if 'operation' in inputs:
            valid_ops = ['find_duplicates', 'merge', 'auto_deduplicate', 'report']
            if inputs['operation'] not in valid_ops:
                errors.append(
                    f"Invalid operation: '{inputs['operation']}'. "
                    f"Must be one of: {', '.join(valid_ops)}"
                )

        # Validate similarity_threshold if provided
        if 'similarity_threshold' in inputs:
            try:
                threshold = float(inputs['similarity_threshold'])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("'similarity_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'similarity_threshold' must be a number")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be a list of strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute deduplication based on operation type.

        Args:
            inputs: Input data containing entities/triples and operation parameters
            context: Workflow state for tracking progress

        Returns:
            Dict containing:
                - duplicates: List of duplicate groups found
                - merged: List of merged entities
                - conflicts: List of unresolved conflicts
                - report: Comprehensive deduplication report

        Raises:
            NodeExecutionError: If deduplication fails
        """
        operation = inputs.get('operation', self.config.get('operation', 'auto_deduplicate'))
        similarity_threshold = inputs.get(
            'similarity_threshold',
            self.config.get('similarity_threshold', 0.85)
        )
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        merge_strategy = inputs.get('merge_strategy', self.config.get('merge_strategy', 'merge_properties'))
        auto_merge = inputs.get('auto_merge', self.config.get('auto_merge', False))
        properties_to_compare = inputs.get(
            'properties_to_compare',
            self.config.get('properties_to_compare', ['name', 'aliases', 'description'])
        )

        context.update_progress(10, f"Starting deduplication operation: {operation}")
        self.logger.info(f"Executing deduplication: operation={operation}, threshold={similarity_threshold}")

        try:
            # Load entities/triples from inputs
            entities, triples = self._load_data(inputs, context)

            if not entities and not triples:
                return {
                    'duplicates': [],
                    'merged': [],
                    'conflicts': [],
                    'report': {
                        'operation': operation,
                        'status': 'no_data',
                        'message': 'No entities or triples to deduplicate'
                    }
                }

            context.update_progress(30, f"Loaded {len(entities)} entities, {len(triples)} triples")

            # Filter by entity types if specified
            if entity_types and entities:
                entities = [e for e in entities if self._get_entity_type(e) in entity_types]
                context.update_progress(35, f"Filtered to {len(entities)} entities by type")

            # Execute operation
            if operation == 'find_duplicates':
                result = self._execute_find_duplicates(
                    entities, triples, similarity_threshold, properties_to_compare, context
                )
            elif operation == 'merge':
                result = self._execute_merge(
                    entities, merge_strategy, auto_merge, properties_to_compare, context
                )
            elif operation == 'auto_deduplicate':
                result = self._execute_auto_deduplicate(
                    entities, triples, similarity_threshold, merge_strategy, 
                    auto_merge, properties_to_compare, context
                )
            elif operation == 'report':
                result = self._execute_report(
                    entities, triples, similarity_threshold, properties_to_compare, context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['find_duplicates', 'merge', 'auto_deduplicate', 'report']}
                )

            context.update_progress(100, f"Deduplication complete: {operation}")

            # Store artifact in context
            context.add_artifact('deduplication', {
                'operation': operation,
                'entities_processed': len(entities),
                'triples_processed': len(triples),
                'duplicates_found': len(result.get('duplicates', [])),
                'entities_merged': len(result.get('merged', [])),
                'conflicts': len(result.get('conflicts', []))
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Deduplication failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Deduplication failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': {k: v for k, v in inputs.items() if k != 'entities'},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _load_data(self, inputs: Dict, context) -> Tuple[List[Dict], List[Dict]]:
        """Load entities and triples from inputs or KG hub."""
        entities = []
        triples = []

        # Load from direct input
        if 'entities' in inputs:
            entities = inputs['entities'] if isinstance(inputs['entities'], list) else []
        if 'triples' in inputs:
            triples = inputs['triples'] if isinstance(inputs['triples'], list) else []

        # Load from KG hub if ID provided
        if 'knowledge_graph_id' in inputs and self.kg_hub:
            kg_id = inputs['knowledge_graph_id']
            context.update_progress(15, f"Loading data from knowledge graph: {kg_id}")
            try:
                # Try to get entities from hub
                if hasattr(self.kg_hub, 'entities'):
                    hub_entities = self.kg_hub.entities
                    if hub_entities:
                        entities.extend(self._convert_hub_entities(hub_entities))
                if hasattr(self.kg_hub, 'triples'):
                    hub_triples = self.kg_hub.triples
                    if hub_triples:
                        triples.extend(self._convert_hub_triples(hub_triples))
            except Exception as e:
                self.logger.warning(f"Could not load from KG hub: {e}")

        return entities, triples

    def _convert_hub_entities(self, hub_entities) -> List[Dict]:
        """Convert hub entities to dictionary format."""
        entities = []
        for e in hub_entities:
            if isinstance(e, dict):
                entities.append(e)
            else:
                # Convert object to dict
                entities.append({
                    'id': getattr(e, 'id', str(hash(e))),
                    'name': getattr(e, 'name', getattr(e, 'subject', 'Unknown')),
                    'type': getattr(e, 'entity_type', getattr(e, 'type', 'entity')),
                    'description': getattr(e, 'description', None),
                    'properties': getattr(e, 'properties', {}),
                    'aliases': getattr(e, 'aliases', []),
                    'confidence': getattr(e, 'confidence', 1.0)
                })
        return entities

    def _convert_hub_triples(self, hub_triples) -> List[Dict]:
        """Convert hub triples to dictionary format."""
        triples = []
        for t in hub_triples:
            if isinstance(t, dict):
                triples.append(t)
            else:
                triples.append({
                    'subject': getattr(t, 'subject', getattr(t, 'head', '')),
                    'predicate': getattr(t, 'predicate', getattr(t, 'relation', '')),
                    'object': getattr(t, 'object', getattr(t, 'tail', '')),
                    'confidence': getattr(t, 'confidence', 1.0)
                })
        return triples

    def _get_entity_type(self, entity: Dict) -> str:
        """Extract entity type from entity dict."""
        return entity.get('type', entity.get('entity_type', 'unknown'))

    def _execute_find_duplicates(
        self,
        entities: List[Dict],
        triples: List[Dict],
        similarity_threshold: float,
        properties_to_compare: List[str],
        context
    ) -> Dict[str, Any]:
        """Find duplicate entities and triples without merging."""
        context.update_progress(40, "Finding duplicate entities")

        # Find entity duplicates
        entity_duplicates = self._find_entity_duplicates(
            entities, similarity_threshold, properties_to_compare
        )

        context.update_progress(70, f"Found {len(entity_duplicates)} duplicate groups")

        # Find triple duplicates
        context.update_progress(75, "Finding duplicate triples")
        triple_duplicates = self._find_triple_duplicates(triples)

        context.update_progress(90, f"Found {len(triple_duplicates)} duplicate triples")

        return {
            'duplicates': entity_duplicates,
            'triple_duplicates': triple_duplicates,
            'merged': [],
            'conflicts': [],
            'report': {
                'operation': 'find_duplicates',
                'entities_processed': len(entities),
                'triples_processed': len(triples),
                'duplicate_entity_groups': len(entity_duplicates),
                'duplicate_triples': len(triple_duplicates),
                'similarity_threshold': similarity_threshold,
                'properties_compared': properties_to_compare
            }
        }

    def _execute_merge(
        self,
        entities: List[Dict],
        merge_strategy: str,
        auto_merge: bool,
        properties_to_compare: List[str],
        context
    ) -> Dict[str, Any]:
        """Merge duplicate entities based on strategy."""
        context.update_progress(40, f"Merging entities with strategy: {merge_strategy}")

        # First find duplicates
        duplicate_groups = self._find_entity_duplicates(
            entities, 0.85, properties_to_compare
        )

        context.update_progress(60, f"Found {len(duplicate_groups)} groups to merge")

        merged = []
        conflicts = []

        for i, group in enumerate(duplicate_groups):
            progress = 60 + (i / len(duplicate_groups)) * 30 if duplicate_groups else 90
            context.update_progress(int(progress), f"Merging group {i+1}/{len(duplicate_groups)}")

            if len(group) < 2:
                continue

            try:
                merged_entity = self._merge_entity_group(group, merge_strategy)
                merged.append({
                    'canonical': merged_entity,
                    'merged_from': group,
                    'strategy': merge_strategy
                })
            except Exception as e:
                conflicts.append({
                    'group': group,
                    'error': str(e),
                    'strategy': merge_strategy
                })

        context.update_progress(95, f"Merged {len(merged)} groups, {len(conflicts)} conflicts")

        return {
            'duplicates': duplicate_groups,
            'merged': merged,
            'conflicts': conflicts,
            'report': {
                'operation': 'merge',
                'merge_strategy': merge_strategy,
                'auto_merge': auto_merge,
                'entities_merged': len(merged),
                'conflicts': len(conflicts)
            }
        }

    def _execute_auto_deduplicate(
        self,
        entities: List[Dict],
        triples: List[Dict],
        similarity_threshold: float,
        merge_strategy: str,
        auto_merge: bool,
        properties_to_compare: List[str],
        context
    ) -> Dict[str, Any]:
        """Automatically find and merge duplicates."""
        context.update_progress(40, "Auto-deduplicating entities")

        # Use dedup manager if available
        if self.dedup_manager and entities:
            try:
                return self._auto_dedup_with_manager(
                    entities, similarity_threshold, merge_strategy, context
                )
            except Exception as e:
                self.logger.warning(f"Manager dedup failed, using fallback: {e}")

        # Fallback: manual deduplication
        return self._execute_merge(
            entities, merge_strategy, auto_merge, properties_to_compare, context
        )

    def _auto_dedup_with_manager(
        self,
        entities: List[Dict],
        similarity_threshold: float,
        merge_strategy: str,
        context
    ) -> Dict[str, Any]:
        """Use UnifiedDeduplicationManager for deduplication."""
        from knowledge_engine.deduplication.base import Entity

        context.update_progress(50, "Converting to manager format")

        # Convert dict entities to Entity objects
        entity_objects = []
        for e in entities:
            entity_objects.append(Entity(
                id=e.get('id', str(hash(str(e)))),
                name=e.get('name', 'Unknown'),
                entity_type=e.get('type', e.get('entity_type', 'unknown')),
                description=e.get('description'),
                properties=e.get('properties', {}),
                source=e.get('source'),
                timestamp=datetime.utcnow()
            ))

        context.update_progress(60, f"Running deduplication on {len(entity_objects)} entities")

        # Run async deduplication
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.dedup_manager.deduplicate(entity_objects, strategy='auto')
            )

            context.update_progress(80, "Processing deduplication results")

            # Convert result back to dicts
            duplicates = []
            for group in result.duplicate_groups:
                duplicates.append([{
                    'id': e.id,
                    'name': e.name,
                    'type': e.entity_type,
                    'description': e.description
                } for e in group])

            merged = [{
                'id': e.id,
                'name': e.name,
                'type': e.entity_type,
                'description': e.description,
                'properties': e.properties
            } for e in result.canonical_entities]

            return {
                'duplicates': duplicates,
                'merged': merged,
                'conflicts': [],
                'report': {
                    'operation': 'auto_deduplicate',
                    'strategy_used': result.strategy_used,
                    'processing_time_ms': result.processing_time_ms,
                    'entities_before': len(entity_objects),
                    'entities_after': len(merged),
                    'reduction_percent': (
                        (1 - len(merged) / len(entity_objects)) * 100
                        if entity_objects else 0
                    )
                }
            }
        except Exception as e:
            self.logger.error(f"Manager deduplication failed: {e}")
            raise

    def _execute_report(
        self,
        entities: List[Dict],
        triples: List[Dict],
        similarity_threshold: float,
        properties_to_compare: List[str],
        context
    ) -> Dict[str, Any]:
        """Generate comprehensive deduplication report."""
        context.update_progress(40, "Analyzing entities for report")

        # Find duplicates at various thresholds
        thresholds = [0.7, 0.8, 0.9, 0.95]
        duplicate_analysis = {}

        for threshold in thresholds:
            dup_groups = self._find_entity_duplicates(
                entities, threshold, properties_to_compare
            )
            duplicate_analysis[f'threshold_{threshold}'] = {
                'duplicate_groups': len(dup_groups),
                'affected_entities': sum(len(g) for g in dup_groups)
            }

        context.update_progress(70, "Analyzing entity types")

        # Analyze entity types
        type_distribution = defaultdict(int)
        for e in entities:
            type_distribution[self._get_entity_type(e)] += 1

        context.update_progress(80, "Finding potential duplicates")

        # Find potential duplicates at default threshold
        potential_duplicates = self._find_entity_duplicates(
            entities, similarity_threshold, properties_to_compare
        )

        # Calculate statistics
        total_entities = len(entities)
        total_triples = len(triples)
        affected_entities = sum(len(g) for g in potential_duplicates)

        context.update_progress(95, "Generating final report")

        report = {
            'summary': {
                'total_entities': total_entities,
                'total_triples': total_triples,
                'entity_types': dict(type_distribution),
                'potential_duplicates': len(potential_duplicates),
                'affected_entities': affected_entities,
                'deduplication_potential': (
                    affected_entities / total_entities * 100
                    if total_entities > 0 else 0
                )
            },
            'duplicate_analysis': duplicate_analysis,
            'recommendations': self._generate_recommendations(
                entities, potential_duplicates, similarity_threshold
            ),
            'sample_duplicates': potential_duplicates[:5] if potential_duplicates else []
        }

        return {
            'duplicates': potential_duplicates,
            'merged': [],
            'conflicts': [],
            'report': report
        }

    def _find_entity_duplicates(
        self,
        entities: List[Dict],
        similarity_threshold: float,
        properties_to_compare: List[str]
    ) -> List[List[Dict]]:
        """Find groups of duplicate entities using similarity metrics."""
        if not entities:
            return []

        # Build similarity graph
        n = len(entities)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue

            group = [entities[i]]
            visited.add(i)

            for j in range(i + 1, n):
                if j in visited:
                    continue

                similarity = self._calculate_similarity(
                    entities[i], entities[j], properties_to_compare
                )

                if similarity >= similarity_threshold:
                    group.append(entities[j])
                    visited.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _find_triple_duplicates(self, triples: List[Dict]) -> List[List[Dict]]:
        """Find duplicate triples."""
        if not triples:
            return []

        # Group by normalized subject-predicate-object
        triple_groups = defaultdict(list)

        for t in triples:
            # Normalize triple for comparison
            subj = str(t.get('subject', '')).lower().strip()
            pred = str(t.get('predicate', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()
            key = f"{subj}|{pred}|{obj}"
            triple_groups[key].append(t)

        # Return groups with more than one triple
        return [group for group in triple_groups.values() if len(group) > 1]

    def _calculate_similarity(
        self,
        entity1: Dict,
        entity2: Dict,
        properties_to_compare: List[str]
    ) -> float:
        """Calculate similarity score between two entities."""
        scores = []

        # Compare names
        if 'name' in properties_to_compare:
            name1 = str(entity1.get('name', '')).lower()
            name2 = str(entity2.get('name', '')).lower()
            if name1 and name2:
                scores.append(self._string_similarity(name1, name2))

        # Compare aliases
        if 'aliases' in properties_to_compare:
            aliases1 = set(a.lower() for a in entity1.get('aliases', []))
            aliases2 = set(a.lower() for a in entity2.get('aliases', []))
            if aliases1 and aliases2:
                intersection = len(aliases1 & aliases2)
                union = len(aliases1 | aliases2)
                if union > 0:
                    scores.append(intersection / union)

        # Compare descriptions
        if 'description' in properties_to_compare:
            desc1 = str(entity1.get('description', '')).lower()
            desc2 = str(entity2.get('description', '')).lower()
            if desc1 and desc2:
                scores.append(self._string_similarity(desc1, desc2))

        # Compare type
        type1 = self._get_entity_type(entity1)
        type2 = self._get_entity_type(entity2)
        if type1 == type2:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Return average score
        return sum(scores) / len(scores) if scores else 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _merge_entity_group(
        self,
        group: List[Dict],
        merge_strategy: str
    ) -> Dict:
        """Merge a group of duplicate entities into a canonical form."""
        if not group:
            raise ValueError("Cannot merge empty group")

        if len(group) == 1:
            return group[0]

        if merge_strategy == 'keep_first':
            return group[0].copy()

        elif merge_strategy == 'keep_best_confidence':
            # Sort by confidence and return highest
            sorted_group = sorted(
                group,
                key=lambda e: e.get('confidence', 0),
                reverse=True
            )
            return sorted_group[0].copy()

        elif merge_strategy == 'merge_properties':
            # Start with most complete entity
            sorted_group = sorted(
                group,
                key=lambda e: len(e.get('properties', {})),
                reverse=True
            )
            canonical = sorted_group[0].copy()

            # Merge properties from all entities
            merged_properties = dict(canonical.get('properties', {}))
            all_aliases = set(canonical.get('aliases', []))
            all_sources = set()

            for entity in group:
                # Merge properties
                merged_properties.update(entity.get('properties', {}))
                # Collect aliases
                all_aliases.update(entity.get('aliases', []))
                # Collect sources
                if entity.get('source'):
                    all_sources.add(entity['source'])

            canonical['properties'] = merged_properties
            canonical['aliases'] = list(all_aliases)
            if all_sources:
                canonical['source'] = ', '.join(sorted(all_sources))

            return canonical

        elif merge_strategy == 'manual':
            # Return with metadata for manual review
            return {
                'id': group[0].get('id'),
                'name': group[0].get('name'),
                'type': self._get_entity_type(group[0]),
                'candidates': group,
                'requires_manual_review': True
            }

        else:
            # Default: return first
            return group[0].copy()

    def _generate_recommendations(
        self,
        entities: List[Dict],
        duplicates: List[List[Dict]],
        threshold: float
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        total = len(entities)
        affected = sum(len(g) for g in duplicates)

        if affected == 0:
            recommendations.append("No duplicates found. The knowledge graph appears clean.")
        elif affected / total < 0.1:
            recommendations.append("Low duplication rate detected. Manual review may be sufficient.")
        elif affected / total < 0.3:
            recommendations.append("Moderate duplication detected. Consider running auto-deduplication.")
        else:
            recommendations.append("High duplication rate detected. Auto-deduplication strongly recommended.")

        if threshold < 0.8:
            recommendations.append("Current similarity threshold is low. Consider increasing to 0.85-0.9 for stricter matching.")
        elif threshold > 0.95:
            recommendations.append("Current similarity threshold is very high. Consider lowering to 0.85-0.9 to catch more duplicates.")

        if len(duplicates) > 100:
            recommendations.append("Large number of duplicate groups found. Consider using 'merge_properties' strategy for batch processing.")

        return recommendations

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Deduplication Configuration",
            "description": "Configure deduplication of entities and triples in knowledge graphs",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Deduplication operation to perform",
                    "enum": ["find_duplicates", "merge", "auto_deduplicate", "report"],
                    "enumNames": [
                        "Find Duplicates - Identify duplicate entities and triples without merging",
                        "Merge - Merge identified duplicate entities",
                        "Auto Deduplicate - Automatically find and merge duplicates",
                        "Report - Generate comprehensive deduplication report"
                    ],
                    "default": "auto_deduplicate"
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Similarity Threshold",
                    "description": "Minimum similarity score for matching entities (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.85
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Limit deduplication to specific entity types (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "merge_strategy": {
                    "type": "string",
                    "title": "Merge Strategy",
                    "description": "Strategy for merging duplicate entities",
                    "enum": ["keep_first", "keep_best_confidence", "merge_properties", "manual"],
                    "enumNames": [
                        "Keep First - Keep the first entity in each duplicate group",
                        "Keep Best Confidence - Keep the entity with highest confidence score",
                        "Merge Properties - Combine properties from all duplicate entities",
                        "Manual - Flag for manual review without automatic merging"
                    ],
                    "default": "merge_properties"
                },
                "properties_to_compare": {
                    "type": "array",
                    "title": "Properties to Compare",
                    "description": "Entity properties to check for similarity",
                    "items": {
                        "type": "string",
                        "enum": ["name", "aliases", "description", "type"]
                    },
                    "default": ["name", "aliases", "description"]
                },
                "auto_merge": {
                    "type": "boolean",
                    "title": "Auto Merge",
                    "description": "Automatically merge duplicates without review (use with caution)",
                    "default": False
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (fallback available even without managers)
        """
        try:
            # Node can work with or without the managers (has fallback)
            return True
        except Exception:
            return False

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available deduplication strategies.

        Returns:
            List of strategy names available
        """
        strategies = ['fallback_string_matching']

        if self.dedup_manager:
            try:
                stats = self.dedup_manager.get_stats()
                strategies.extend(stats.get('strategies_available', []))
            except Exception:
                pass

        return strategies
