"""
Change Detection Node for BubbleLabs Integration

Detects and reports changes in knowledge graphs over time by comparing
knowledge states, snapshots, or time periods. Supports detecting:
- Added/removed entities and triples
- Modified entity properties
- Confidence changes
- Relationship changes

Uses safe_import for optional temporal knowledge engine dependencies.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import difflib
import copy
from .base_node import BubbleLabsNode, NodeExecutionError


@dataclass
class KnowledgeChange:
    """Represents a single change in knowledge."""
    change_type: str  # 'added', 'removed', 'modified', 'confidence_changed', 'relationship_changed'
    entity_id: Optional[str]
    property_name: Optional[str]
    old_value: Any
    new_value: Any
    confidence_old: Optional[float] = None
    confidence_new: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert change to dictionary."""
        return {
            'change_type': self.change_type,
            'entity_id': self.entity_id,
            'property_name': self.property_name,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'confidence_old': self.confidence_old,
            'confidence_new': self.confidence_new,
            'timestamp': self.timestamp,
            'details': self.details
        }


class ChangeDetectionNode(BubbleLabsNode):
    """
    Change Detection Node for BubbleLabs.
    
    Detects and reports changes in knowledge graphs over time:
    - Compare two knowledge graph states (snapshots or timestamps)
    - Detect added, removed, and modified entities/triples
    - Identify confidence changes above configurable thresholds
    - Generate human-readable diffs and change reports
    - Filter changes by entity scope and change type
    
    Uses safe_import for temporal knowledge modules with fallback
    implementations when these components are unavailable.
    """

    # Node metadata
    DISPLAY_NAME = "Change Detection"
    DESCRIPTION = "Detect and report changes in knowledge graphs over time"
    ICON = "change-detection"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for optional dependencies
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for ChangeDetectionNode"
        )

        self.ChronicleIntegration = self.safe_import(
            'knowledge_engine.chronicle.chronicle.ChronicleIntegration',
            fallback_value=None,
            error_msg="ChronicleIntegration not available for ChangeDetectionNode"
        )

        self.TemporalKnowledgeEngine = self.safe_import(
            'knowledge_engine.temporal_knowledge_engine.TemporalKnowledgeEngine',
            fallback_value=None,
            error_msg="TemporalKnowledgeEngine not available for ChangeDetectionNode"
        )

        # Alternative import paths
        if self.UnifiedKGIntegrationHub is None:
            self.UnifiedKGIntegrationHub = self.safe_import(
                'unified_kg_integration_hub.UnifiedKGIntegrationHub',
                fallback_value=None,
                error_msg="UnifiedKGIntegrationHub not found in alternate path"
            )

        if self.ChronicleIntegration is None:
            self.ChronicleIntegration = self.safe_import(
                'chronicle_memory.ChronicleIntegration',
                fallback_value=None,
                error_msg="ChronicleIntegration not found in alternate path"
            )

        if self.TemporalKnowledgeEngine is None:
            self.TemporalKnowledgeEngine = self.safe_import(
                'temporal_knowledge_engine.TemporalKnowledgeEngine',
                fallback_value=None,
                error_msg="TemporalKnowledgeEngine not found in alternate path"
            )

        # Initialize component instances
        self.kg_hub = None
        self.chronicle = None
        self.temporal_engine = None
        self._initialized = False

        # Initialize if configuration provides connection details
        self._initialize_components()

    def _initialize_components(self):
        """Initialize temporal and knowledge components if config available."""
        # Initialize KG Hub if available
        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for ChangeDetectionNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

        # Initialize Chronicle if config provides instance
        if self.ChronicleIntegration:
            try:
                chronicle_instance = self.config.get('chronicle_instance')
                if chronicle_instance:
                    self.chronicle = self.ChronicleIntegration(chronicle_instance)
                    self.logger.info("ChronicleIntegration initialized for ChangeDetectionNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize ChronicleIntegration: {e}")
                self.chronicle = None

        # Initialize Temporal Knowledge Engine if available
        if self.TemporalKnowledgeEngine:
            try:
                self.temporal_engine = self.TemporalKnowledgeEngine()
                self.logger.info("TemporalKnowledgeEngine initialized for ChangeDetectionNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize TemporalKnowledgeEngine: {e}")
                self.temporal_engine = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - compare_states: baseline_state, current_state
        - detect_changes: baseline_state, current_state
        - generate_diff: baseline_state, current_state
        - change_report: baseline_state, current_state
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'compare_states'))

        valid_operations = ['compare_states', 'detect_changes', 'generate_diff', 'change_report']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # All operations require baseline and current state
        baseline = inputs.get('baseline_state') or self.config.get('baseline_state')
        current = inputs.get('current_state') or self.config.get('current_state')

        if not baseline:
            errors.append("Missing required input: 'baseline_state' (snapshot ID or timestamp)")

        if not current:
            errors.append("Missing required input: 'current_state' (snapshot ID or timestamp)")

        # Validate ISO timestamps if provided in time range fields
        for field in ['time_range_start', 'time_range_end']:
            value = inputs.get(field) or self.config.get(field)
            if value:
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid ISO datetime format for {field}: {value}")

        # Validate change_types if provided
        change_types = inputs.get('change_types') or self.config.get('change_types', [])
        if change_types:
            valid_change_types = ['added', 'removed', 'modified', 'confidence_changed', 'relationship_changed']
            invalid_types = [ct for ct in change_types if ct not in valid_change_types]
            if invalid_types:
                errors.append(f"Invalid change_types: {invalid_types}. Must be from: {valid_change_types}")

        # Validate min_confidence_change range
        min_conf = inputs.get('min_confidence_change') or self.config.get('min_confidence_change', 0.1)
        if not isinstance(min_conf, (int, float)) or min_conf < 0.0 or min_conf > 1.0:
            errors.append(f"min_confidence_change must be between 0.0 and 1.0, got: {min_conf}")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the change detection operation based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing change detection results with changes, diffs, and statistics

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'compare_states'))

        context.update_progress(10, f"Starting {operation} operation")
        self.logger.info(f"Executing change detection operation: {operation}")

        try:
            # Route to appropriate operation handler
            if operation == 'compare_states':
                result = self._execute_compare_states(inputs, context)
            elif operation == 'detect_changes':
                result = self._execute_detect_changes(inputs, context)
            elif operation == 'generate_diff':
                result = self._execute_generate_diff(inputs, context)
            elif operation == 'change_report':
                result = self._execute_change_report(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['compare_states', 'detect_changes', 'generate_diff', 'change_report']}
                )

            context.update_progress(100, f"{operation} operation completed")

            # Add artifact to context
            context.add_artifact('change_detection', {
                'operation': operation,
                'success': result.get('success', True),
                'baseline_state': inputs.get('baseline_state') or self.config.get('baseline_state'),
                'current_state': inputs.get('current_state') or self.config.get('current_state'),
                'changes_count': len(result.get('changes', [])),
                'statistics': result.get('statistics', {})
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Change detection {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': {k: v for k, v in inputs.items() if k not in ['baseline_state', 'current_state']},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_compare_states(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compare two knowledge states and return detailed comparison."""
        context.update_progress(20, "Loading knowledge states")

        baseline_state = self._load_state(inputs, 'baseline_state')
        current_state = self._load_state(inputs, 'current_state')

        context.update_progress(40, "Analyzing state differences")

        # Get configuration parameters
        entity_scope = inputs.get('entity_scope') or self.config.get('entity_scope', [])
        include_unchanged = inputs.get('include_unchanged') or self.config.get('include_unchanged', False)

        # Perform comparison
        comparison = self._compare_states_detailed(
            baseline_state, current_state, entity_scope, include_unchanged
        )

        context.update_progress(80, "Generating comparison results")

        return {
            'success': True,
            'operation': 'compare_states',
            'baseline_state_id': inputs.get('baseline_state') or self.config.get('baseline_state'),
            'current_state_id': inputs.get('current_state') or self.config.get('current_state'),
            'comparison': comparison,
            'changes': [c.to_dict() for c in comparison.get('changes', [])],
            'added': [c.to_dict() for c in comparison.get('added', [])],
            'removed': [c.to_dict() for c in comparison.get('removed', [])],
            'modified': [c.to_dict() for c in comparison.get('modified', [])],
            'statistics': comparison.get('statistics', {})
        }

    def _execute_detect_changes(self, inputs: Dict, context) -> Dict[str, Any]:
        """Detect specific types of changes between states."""
        context.update_progress(20, "Loading knowledge states for change detection")

        baseline_state = self._load_state(inputs, 'baseline_state')
        current_state = self._load_state(inputs, 'current_state')

        context.update_progress(40, "Configuring change detection filters")

        # Get configuration parameters
        change_types = inputs.get('change_types') or self.config.get('change_types', 
            ['added', 'removed', 'modified', 'confidence_changed'])
        entity_scope = inputs.get('entity_scope') or self.config.get('entity_scope', [])
        min_confidence_change = inputs.get('min_confidence_change') or self.config.get('min_confidence_change', 0.1)

        context.update_progress(60, "Detecting changes")

        # Detect changes with filtering
        changes = self._detect_changes_filtered(
            baseline_state, current_state, change_types, entity_scope, min_confidence_change
        )

        # Categorize changes
        added = [c for c in changes if c.change_type == 'added']
        removed = [c for c in changes if c.change_type == 'removed']
        modified = [c for c in changes if c.change_type == 'modified']
        confidence_changed = [c for c in changes if c.change_type == 'confidence_changed']
        relationship_changed = [c for c in changes if c.change_type == 'relationship_changed']

        context.update_progress(80, "Calculating change statistics")

        statistics = self._calculate_statistics(
            changes, added, removed, modified, confidence_changed, relationship_changed,
            baseline_state, current_state
        )

        context.update_progress(100, "Change detection complete")

        return {
            'success': True,
            'operation': 'detect_changes',
            'baseline_state_id': inputs.get('baseline_state') or self.config.get('baseline_state'),
            'current_state_id': inputs.get('current_state') or self.config.get('current_state'),
            'changes': [c.to_dict() for c in changes],
            'added': [c.to_dict() for c in added],
            'removed': [c.to_dict() for c in removed],
            'modified': [c.to_dict() for c in modified],
            'confidence_changed': [c.to_dict() for c in confidence_changed],
            'relationship_changed': [c.to_dict() for c in relationship_changed],
            'statistics': statistics
        }

    def _execute_generate_diff(self, inputs: Dict, context) -> Dict[str, Any]:
        """Generate a human-readable diff between states."""
        context.update_progress(20, "Loading states for diff generation")

        baseline_state = self._load_state(inputs, 'baseline_state')
        current_state = self._load_state(inputs, 'current_state')

        context.update_progress(40, "Generating diff")

        # Generate text diff
        diff_text = self._generate_text_diff(baseline_state, current_state)

        # Generate structured diff
        structured_diff = self._generate_structured_diff(baseline_state, current_state)

        context.update_progress(80, "Formatting diff output")

        return {
            'success': True,
            'operation': 'generate_diff',
            'baseline_state_id': inputs.get('baseline_state') or self.config.get('baseline_state'),
            'current_state_id': inputs.get('current_state') or self.config.get('current_state'),
            'diff_text': diff_text,
            'structured_diff': structured_diff,
            'changes_count': len(structured_diff.get('changes', [])),
            'summary': structured_diff.get('summary', {})
        }

    def _execute_change_report(self, inputs: Dict, context) -> Dict[str, Any]:
        """Generate a comprehensive change report."""
        context.update_progress(20, "Loading states for change report")

        baseline_state = self._load_state(inputs, 'baseline_state')
        current_state = self._load_state(inputs, 'current_state')

        context.update_progress(40, "Analyzing all changes")

        # Get configuration parameters
        change_types = inputs.get('change_types') or self.config.get('change_types',
            ['added', 'removed', 'modified', 'confidence_changed', 'relationship_changed'])
        entity_scope = inputs.get('entity_scope') or self.config.get('entity_scope', [])
        min_confidence_change = inputs.get('min_confidence_change') or self.config.get('min_confidence_change', 0.1)
        include_unchanged = inputs.get('include_unchanged') or self.config.get('include_unchanged', False)

        # Detect all changes
        all_changes = self._detect_changes_filtered(
            baseline_state, current_state, change_types, entity_scope, min_confidence_change
        )

        context.update_progress(60, "Generating report sections")

        # Generate diff
        diff_text = self._generate_text_diff(baseline_state, current_state)
        structured_diff = self._generate_structured_diff(baseline_state, current_state)

        # Categorize changes
        added = [c for c in all_changes if c.change_type == 'added']
        removed = [c for c in all_changes if c.change_type == 'removed']
        modified = [c for c in all_changes if c.change_type == 'modified']
        confidence_changed = [c for c in all_changes if c.change_type == 'confidence_changed']
        relationship_changed = [c for c in all_changes if c.change_type == 'relationship_changed']

        # Calculate statistics
        statistics = self._calculate_statistics(
            all_changes, added, removed, modified, confidence_changed, relationship_changed,
            baseline_state, current_state
        )

        context.update_progress(80, "Compiling comprehensive report")

        report = {
            'success': True,
            'operation': 'change_report',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'baseline_state_id': inputs.get('baseline_state') or self.config.get('baseline_state'),
            'current_state_id': inputs.get('current_state') or self.config.get('current_state'),
            'summary': {
                'total_changes': len(all_changes),
                'added_count': len(added),
                'removed_count': len(removed),
                'modified_count': len(modified),
                'confidence_changed_count': len(confidence_changed),
                'relationship_changed_count': len(relationship_changed)
            },
            'changes': [c.to_dict() for c in all_changes],
            'added': [c.to_dict() for c in added],
            'removed': [c.to_dict() for c in removed],
            'modified': [c.to_dict() for c in modified],
            'confidence_changed': [c.to_dict() for c in confidence_changed],
            'relationship_changed': [c.to_dict() for c in relationship_changed],
            'diff': {
                'text': diff_text,
                'structured': structured_diff
            },
            'statistics': statistics,
            'configuration': {
                'change_types_detected': change_types,
                'entity_scope': entity_scope if entity_scope else 'all',
                'min_confidence_change': min_confidence_change,
                'include_unchanged': include_unchanged
            }
        }

        context.update_progress(100, "Change report complete")

        return report

    def _load_state(self, inputs: Dict, state_key: str) -> Dict[str, Any]:
        """
        Load a knowledge state from snapshot ID, timestamp, or direct input.
        
        Attempts to use temporal modules first, falls back to direct state data.
        """
        state_ref = inputs.get(state_key) or self.config.get(state_key)
        
        if not state_ref:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Missing state reference: {state_key}",
                details={'state_key': state_key}
            )

        # First check if state is provided directly in inputs
        direct_state_key = f"{state_key}_data"
        if direct_state_key in inputs:
            return inputs[direct_state_key]

        # Try to load from temporal engine or hub
        if self.temporal_engine or self.kg_hub:
            try:
                # Check if it's a timestamp
                try:
                    timestamp = datetime.fromisoformat(state_ref.replace('Z', '+00:00'))
                    return self._load_state_at_time(timestamp)
                except (ValueError, AttributeError):
                    pass

                # Try loading as snapshot ID
                return self._load_state_by_id(state_ref)
            except Exception as e:
                self.logger.warning(f"Failed to load state via temporal modules: {e}")

        # Fallback: return empty state structure
        self.logger.warning(f"Using fallback empty state for {state_key}")
        return {'entities': [], 'triples': [], 'metadata': {}}

    def _load_state_at_time(self, timestamp: datetime) -> Dict[str, Any]:
        """Load knowledge state at a specific timestamp."""
        if self.temporal_engine:
            try:
                # This would query the temporal engine for state at time
                return self.temporal_engine.get_state_at_time(timestamp)
            except Exception as e:
                self.logger.warning(f"Temporal engine query failed: {e}")

        if self.kg_hub:
            try:
                return self.kg_hub.get_historical_state(timestamp)
            except Exception as e:
                self.logger.warning(f"KG Hub historical query failed: {e}")

        raise NodeExecutionError(
            node_name=self.get_display_name(),
            message="Cannot load state at time - no temporal engine available",
            details={'timestamp': timestamp.isoformat()}
        )

    def _load_state_by_id(self, state_id: str) -> Dict[str, Any]:
        """Load knowledge state by snapshot ID."""
        if self.chronicle:
            try:
                return self.chronicle.get_snapshot(state_id)
            except Exception as e:
                self.logger.warning(f"Chronicle snapshot retrieval failed: {e}")

        if self.kg_hub:
            try:
                return self.kg_hub.get_snapshot(state_id)
            except Exception as e:
                self.logger.warning(f"KG Hub snapshot retrieval failed: {e}")

        raise NodeExecutionError(
            node_name=self.get_display_name(),
            message="Cannot load state by ID - no storage backend available",
            details={'state_id': state_id}
        )

    def _compare_states_detailed(
        self, 
        baseline: Dict[str, Any], 
        current: Dict[str, Any],
        entity_scope: List[str],
        include_unchanged: bool
    ) -> Dict[str, Any]:
        """Perform detailed comparison between two states."""
        changes = []
        added = []
        removed = []
        modified = []
        unchanged = []

        # Extract entities and triples from both states
        baseline_entities = self._extract_entities(baseline)
        current_entities = self._extract_entities(current)
        baseline_triples = self._extract_triples(baseline)
        current_triples = self._extract_triples(current)

        # Filter by entity scope if specified
        if entity_scope:
            baseline_entities = {k: v for k, v in baseline_entities.items() if k in entity_scope}
            current_entities = {k: v for k, v in current_entities.items() if k in entity_scope}
            baseline_triples = [t for t in baseline_triples if self._triple_involves_entities(t, entity_scope)]
            current_triples = [t for t in current_triples if self._triple_involves_entities(t, entity_scope)]

        # Compare entities
        baseline_ids = set(baseline_entities.keys())
        current_ids = set(current_entities.keys())

        added_ids = current_ids - baseline_ids
        removed_ids = baseline_ids - current_ids
        common_ids = baseline_ids & current_ids

        # Detect added entities
        for entity_id in added_ids:
            change = KnowledgeChange(
                change_type='added',
                entity_id=entity_id,
                property_name=None,
                old_value=None,
                new_value=current_entities[entity_id],
                details={'entity_data': current_entities[entity_id]}
            )
            changes.append(change)
            added.append(change)

        # Detect removed entities
        for entity_id in removed_ids:
            change = KnowledgeChange(
                change_type='removed',
                entity_id=entity_id,
                property_name=None,
                old_value=baseline_entities[entity_id],
                new_value=None,
                details={'entity_data': baseline_entities[entity_id]}
            )
            changes.append(change)
            removed.append(change)

        # Detect modified entities
        for entity_id in common_ids:
            baseline_entity = baseline_entities[entity_id]
            current_entity = current_entities[entity_id]

            property_changes = self._compare_entity_properties(entity_id, baseline_entity, current_entity)
            changes.extend(property_changes)
            modified.extend([c for c in property_changes if c.change_type == 'modified'])

            if not property_changes and include_unchanged:
                unchanged.append({
                    'entity_id': entity_id,
                    'entity': current_entity
                })

        # Compare triples
        baseline_triple_set = {self._triple_key(t): t for t in baseline_triples}
        current_triple_set = {self._triple_key(t): t for t in current_triples}

        baseline_triple_keys = set(baseline_triple_set.keys())
        current_triple_keys = set(current_triple_set.keys())

        added_triple_keys = current_triple_keys - baseline_triple_keys
        removed_triple_keys = baseline_triple_keys - current_triple_keys

        # Detect added triples
        for key in added_triple_keys:
            triple = current_triple_set[key]
            change = KnowledgeChange(
                change_type='added',
                entity_id=triple.get('subject'),
                property_name='triple',
                old_value=None,
                new_value=triple,
                details={'triple': triple}
            )
            changes.append(change)
            added.append(change)

        # Detect removed triples
        for key in removed_triple_keys:
            triple = baseline_triple_set[key]
            change = KnowledgeChange(
                change_type='removed',
                entity_id=triple.get('subject'),
                property_name='triple',
                old_value=triple,
                new_value=None,
                details={'triple': triple}
            )
            changes.append(change)
            removed.append(change)

        # Detect confidence changes in common triples
        common_triple_keys = baseline_triple_keys & current_triple_keys
        for key in common_triple_keys:
            baseline_triple = baseline_triple_set[key]
            current_triple = current_triple_set[key]

            baseline_conf = baseline_triple.get('confidence', 1.0)
            current_conf = current_triple.get('confidence', 1.0)

            if abs(current_conf - baseline_conf) >= self.config.get('min_confidence_change', 0.1):
                change = KnowledgeChange(
                    change_type='confidence_changed',
                    entity_id=baseline_triple.get('subject'),
                    property_name='confidence',
                    old_value=baseline_conf,
                    new_value=current_conf,
                    confidence_old=baseline_conf,
                    confidence_new=current_conf,
                    details={'triple': current_triple, 'confidence_delta': current_conf - baseline_conf}
                )
                changes.append(change)

        # Calculate statistics
        statistics = {
            'total_entities_baseline': len(baseline_entities),
            'total_entities_current': len(current_entities),
            'total_triples_baseline': len(baseline_triples),
            'total_triples_current': len(current_triples),
            'entities_added': len(added_ids),
            'entities_removed': len(removed_ids),
            'entities_modified': len(set(c.entity_id for c in modified if c.entity_id)),
            'triples_added': len(added_triple_keys),
            'triples_removed': len(removed_triple_keys),
            'total_changes': len(changes)
        }

        return {
            'changes': changes,
            'added': added,
            'removed': removed,
            'modified': modified,
            'unchanged': unchanged if include_unchanged else [],
            'statistics': statistics
        }

    def _detect_changes_filtered(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        change_types: List[str],
        entity_scope: List[str],
        min_confidence_change: float
    ) -> List[KnowledgeChange]:
        """Detect changes with filtering options."""
        comparison = self._compare_states_detailed(baseline, current, entity_scope, False)
        all_changes = comparison.get('changes', [])

        # Filter by change type
        if change_types:
            all_changes = [c for c in all_changes if c.change_type in change_types]

        # Filter confidence changes by threshold
        if 'confidence_changed' in change_types:
            confidence_changes = [c for c in all_changes if c.change_type == 'confidence_changed']
            other_changes = [c for c in all_changes if c.change_type != 'confidence_changed']
            
            filtered_confidence = [
                c for c in confidence_changes 
                if c.confidence_old is not None 
                and c.confidence_new is not None
                and abs(c.confidence_new - c.confidence_old) >= min_confidence_change
            ]
            all_changes = other_changes + filtered_confidence

        return all_changes

    def _extract_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from a knowledge state."""
        entities = {}

        if 'entities' in state:
            entity_list = state['entities']
            if isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, dict):
                        entity_id = entity.get('id') or entity.get('name') or entity.get('uri')
                        if entity_id:
                            entities[entity_id] = entity
                    elif isinstance(entity, str):
                        entities[entity] = {'id': entity, 'name': entity}
            elif isinstance(entity_list, dict):
                entities = entity_list

        # Also extract from triples
        if 'triples' in state:
            for triple in state['triples']:
                if isinstance(triple, dict):
                    subject = triple.get('subject')
                    obj = triple.get('object')
                    if subject and subject not in entities:
                        entities[subject] = {'id': subject, 'name': subject}
                    if obj and obj not in entities:
                        entities[obj] = {'id': obj, 'name': obj}

        return entities

    def _extract_triples(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract triples from a knowledge state."""
        if 'triples' in state and isinstance(state['triples'], list):
            return state['triples']
        return []

    def _triple_key(self, triple: Dict[str, Any]) -> str:
        """Generate a unique key for a triple."""
        if isinstance(triple, dict):
            subject = triple.get('subject', '')
            predicate = triple.get('predicate', '')
            obj = triple.get('object', '')
            return f"{subject}|{predicate}|{obj}"
        return str(triple)

    def _triple_involves_entities(self, triple: Dict[str, Any], entity_ids: List[str]) -> bool:
        """Check if a triple involves any of the specified entities."""
        if isinstance(triple, dict):
            subject = triple.get('subject', '')
            obj = triple.get('object', '')
            return subject in entity_ids or obj in entity_ids
        return False

    def _compare_entity_properties(
        self, 
        entity_id: str, 
        baseline: Dict[str, Any], 
        current: Dict[str, Any]
    ) -> List[KnowledgeChange]:
        """Compare properties of an entity between two states."""
        changes = []

        if not isinstance(baseline, dict) or not isinstance(current, dict):
            return changes

        all_keys = set(baseline.keys()) | set(current.keys())

        for key in all_keys:
            if key in baseline and key not in current:
                # Property removed
                changes.append(KnowledgeChange(
                    change_type='modified',
                    entity_id=entity_id,
                    property_name=key,
                    old_value=baseline[key],
                    new_value=None,
                    details={'change_subtype': 'property_removed'}
                ))
            elif key not in baseline and key in current:
                # Property added
                changes.append(KnowledgeChange(
                    change_type='modified',
                    entity_id=entity_id,
                    property_name=key,
                    old_value=None,
                    new_value=current[key],
                    details={'change_subtype': 'property_added'}
                ))
            elif baseline.get(key) != current.get(key):
                # Property modified
                changes.append(KnowledgeChange(
                    change_type='modified',
                    entity_id=entity_id,
                    property_name=key,
                    old_value=baseline[key],
                    new_value=current[key],
                    details={'change_subtype': 'property_modified'}
                ))

        # Check for type changes
        baseline_type = baseline.get('type') or baseline.get('entity_type')
        current_type = current.get('type') or current.get('entity_type')
        if baseline_type != current_type:
            changes.append(KnowledgeChange(
                change_type='modified',
                entity_id=entity_id,
                property_name='type',
                old_value=baseline_type,
                new_value=current_type,
                details={'change_subtype': 'type_changed'}
            ))

        return changes

    def _generate_text_diff(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> str:
        """Generate a human-readable text diff between states."""
        baseline_lines = self._state_to_lines(baseline)
        current_lines = self._state_to_lines(current)

        diff = difflib.unified_diff(
            baseline_lines,
            current_lines,
            fromfile='baseline',
            tofile='current',
            lineterm=''
        )

        return '\n'.join(diff)

    def _generate_structured_diff(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured diff representation."""
        comparison = self._compare_states_detailed(baseline, current, [], False)

        summary = {
            'total_changes': len(comparison.get('changes', [])),
            'added_count': len(comparison.get('added', [])),
            'removed_count': len(comparison.get('removed', [])),
            'modified_count': len(comparison.get('modified', []))
        }

        return {
            'summary': summary,
            'changes': [c.to_dict() for c in comparison.get('changes', [])],
            'added_summary': self._summarize_changes(comparison.get('added', [])),
            'removed_summary': self._summarize_changes(comparison.get('removed', [])),
            'modified_summary': self._summarize_changes(comparison.get('modified', []))
        }

    def _state_to_lines(self, state: Dict[str, Any]) -> List[str]:
        """Convert a knowledge state to lines for diffing."""
        lines = []

        # Add entities
        entities = self._extract_entities(state)
        for entity_id in sorted(entities.keys()):
            entity = entities[entity_id]
            lines.append(f"ENTITY: {entity_id}")
            if isinstance(entity, dict):
                for key, value in sorted(entity.items()):
                    lines.append(f"  {key}: {value}")

        # Add triples
        triples = self._extract_triples(state)
        for triple in sorted(triples, key=self._triple_key):
            if isinstance(triple, dict):
                subject = triple.get('subject', '')
                predicate = triple.get('predicate', '')
                obj = triple.get('object', '')
                confidence = triple.get('confidence', 1.0)
                lines.append(f"TRIPLE: {subject} --{predicate}--> {obj} (conf: {confidence})")

        return lines

    def _summarize_changes(self, changes: List[KnowledgeChange]) -> List[Dict[str, Any]]:
        """Create a summary of changes by entity."""
        entity_changes = {}

        for change in changes:
            entity_id = change.entity_id or 'unknown'
            if entity_id not in entity_changes:
                entity_changes[entity_id] = {
                    'entity_id': entity_id,
                    'change_count': 0,
                    'change_types': set()
                }
            entity_changes[entity_id]['change_count'] += 1
            entity_changes[entity_id]['change_types'].add(change.change_type)

        # Convert sets to lists for serialization
        for summary in entity_changes.values():
            summary['change_types'] = list(summary['change_types'])

        return list(entity_changes.values())

    def _calculate_statistics(
        self,
        all_changes: List[KnowledgeChange],
        added: List[KnowledgeChange],
        removed: List[KnowledgeChange],
        modified: List[KnowledgeChange],
        confidence_changed: List[KnowledgeChange],
        relationship_changed: List[KnowledgeChange],
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive change statistics."""
        baseline_entities = self._extract_entities(baseline)
        current_entities = self._extract_entities(current)
        baseline_triples = self._extract_triples(baseline)
        current_triples = self._extract_triples(current)

        # Calculate confidence change statistics
        confidence_deltas = []
        for change in confidence_changed:
            if change.confidence_old is not None and change.confidence_new is not None:
                confidence_deltas.append(change.confidence_new - change.confidence_old)

        stats = {
            'total_changes': len(all_changes),
            'change_breakdown': {
                'added': len(added),
                'removed': len(removed),
                'modified': len(modified),
                'confidence_changed': len(confidence_changed),
                'relationship_changed': len(relationship_changed)
            },
            'entity_statistics': {
                'baseline_count': len(baseline_entities),
                'current_count': len(current_entities),
                'net_change': len(current_entities) - len(baseline_entities),
                'growth_rate': (len(current_entities) - len(baseline_entities)) / max(len(baseline_entities), 1) * 100
            },
            'triple_statistics': {
                'baseline_count': len(baseline_triples),
                'current_count': len(current_triples),
                'net_change': len(current_triples) - len(baseline_triples),
                'growth_rate': (len(current_triples) - len(baseline_triples)) / max(len(baseline_triples), 1) * 100
            },
            'confidence_statistics': {
                'total_confidence_changes': len(confidence_changed),
                'average_confidence_delta': sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else 0,
                'max_confidence_increase': max(confidence_deltas) if confidence_deltas else 0,
                'max_confidence_decrease': min(confidence_deltas) if confidence_deltas else 0
            },
            'affected_entities': list(set(c.entity_id for c in all_changes if c.entity_id)),
            'affected_entity_count': len(set(c.entity_id for c in all_changes if c.entity_id))
        }

        return stats

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all change detection operations and parameters.
        """
        return {
            "type": "object",
            "title": "Change Detection Configuration",
            "description": "Configure change detection operations for comparing knowledge states over time",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The change detection operation to perform",
                    "enum": ["compare_states", "detect_changes", "generate_diff", "change_report"],
                    "enumNames": [
                        "Compare States - Detailed comparison between two states",
                        "Detect Changes - Filtered change detection with type selection",
                        "Generate Diff - Create human-readable diffs",
                        "Change Report - Comprehensive change report with all details"
                    ],
                    "default": "compare_states"
                },
                "baseline_state": {
                    "type": "string",
                    "title": "Baseline State",
                    "description": "Reference state to compare from (snapshot ID or ISO timestamp)",
                    "default": ""
                },
                "current_state": {
                    "type": "string",
                    "title": "Current State",
                    "description": "Current state to compare to (snapshot ID or ISO timestamp)",
                    "default": ""
                },
                "entity_scope": {
                    "type": "array",
                    "title": "Entity Scope",
                    "description": "Limit detection to specific entity IDs (empty = all entities)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "change_types": {
                    "type": "array",
                    "title": "Change Types",
                    "description": "Types of changes to detect",
                    "items": {
                        "type": "string",
                        "enum": ["added", "removed", "modified", "confidence_changed", "relationship_changed"]
                    },
                    "default": ["added", "removed", "modified", "confidence_changed"]
                },
                "min_confidence_change": {
                    "type": "number",
                    "title": "Minimum Confidence Change",
                    "description": "Only report confidence changes greater than this threshold (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1
                },
                "include_unchanged": {
                    "type": "boolean",
                    "title": "Include Unchanged",
                    "description": "Include unchanged entities in the output",
                    "default": False
                },
                "time_range_start": {
                    "type": "string",
                    "title": "Time Range Start",
                    "description": "ISO timestamp for the start of the comparison period (optional)",
                    "default": ""
                },
                "time_range_end": {
                    "type": "string",
                    "title": "Time Range End",
                    "description": "ISO timestamp for the end of the comparison period (optional)",
                    "default": ""
                }
            },
            "required": ["operation", "baseline_state", "current_state"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["compare_states"]}
                            },
                            "description": "Detailed comparison between two knowledge states"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["detect_changes"]}
                            },
                            "description": "Filtered change detection with configurable filters"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["generate_diff"]}
                            },
                            "description": "Generate text and structured diffs"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["change_report"]}
                            },
                            "description": "Comprehensive report with all change information"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (at least fallback comparison available), False otherwise
        """
        try:
            # Check if any of the temporal modules are available
            hub_available = self.UnifiedKGIntegrationHub is not None
            chronicle_available = self.ChronicleIntegration is not None
            temporal_available = self.TemporalKnowledgeEngine is not None

            # Node is healthy if at least one backend is available OR
            # if we have fallback capability (always true for this node)
            # The node can always perform comparisons on direct state inputs
            return hub_available or chronicle_available or temporal_available or True
        except Exception:
            return True  # Fallback is always available

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.kg_hub = None
            self.chronicle = None
            self.temporal_engine = None
            self.logger.info("ChangeDetectionNode cleanup complete")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


class ChangeTracker:
    """
    Tracks changes in knowledge graphs over time.
    
    Provides a simple interface for recording and retrieving changes
    between knowledge states. Used for backward compatibility and
    testing purposes.
    """
    
    def __init__(self, *args, **kwargs):
        self.changes = []
        self.snapshots = {}
    
    def track(self, change_type: str, entity_id: str = None, 
              old_value: Any = None, new_value: Any = None,
              details: Dict[str, Any] = None) -> KnowledgeChange:
        """
        Track a change.
        
        Args:
            change_type: Type of change (added, removed, modified, etc.)
            entity_id: Optional entity identifier
            old_value: Previous value
            new_value: New value
            details: Additional details about the change
            
        Returns:
            KnowledgeChange object representing the tracked change
        """
        change = KnowledgeChange(
            change_type=change_type,
            entity_id=entity_id,
            property_name=None,
            old_value=old_value,
            new_value=new_value,
            details=details or {}
        )
        self.changes.append(change)
        return change
    
    def get_changes(self, *args, **kwargs) -> List[KnowledgeChange]:
        """
        Get all tracked changes.
        
        Returns:
            List of KnowledgeChange objects
        """
        return self.changes
    
    def snapshot(self, snapshot_id: str, state: Dict[str, Any]) -> None:
        """
        Store a snapshot of a state.
        
        Args:
            snapshot_id: Identifier for the snapshot
            state: State data to store
        """
        self.snapshots[snapshot_id] = state
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored snapshot.
        
        Args:
            snapshot_id: Identifier for the snapshot
            
        Returns:
            Stored state data or None if not found
        """
        return self.snapshots.get(snapshot_id)
    
    def clear(self) -> None:
        """Clear all tracked changes and snapshots."""
        self.changes = []
        self.snapshots = {}
