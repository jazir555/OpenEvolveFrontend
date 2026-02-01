"""
Temporal Knowledge Node for BubbleLabs Integration

Provides temporal knowledge tracking capabilities using Graphiti:
- Store knowledge with timestamps (valid_from, valid_at)
- Query knowledge at specific points in time
- Track knowledge evolution/changes over time
- Find what was known at a specific date
- Compare knowledge between two time periods
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .base_node import BubbleLabsNode, NodeExecutionError


class TemporalKnowledgeNode(BubbleLabsNode):
    """
    Track and query knowledge changes over time with temporal awareness.

    Integrates with Graphiti temporal knowledge graph to provide:
    - Time-aware knowledge storage with validity periods
    - Point-in-time queries to retrieve historical knowledge states
    - Change tracking to see how knowledge evolved
    - Period comparison to analyze differences between time ranges
    - History retrieval for complete entity timelines

    Uses safe imports for GraphitiIntegration, ChronicleIntegration,
    and UnifiedKGIntegrationHub with fallback implementations when
    these components are not available.
    """

    # Node metadata
    DISPLAY_NAME = "Temporal Knowledge"
    DESCRIPTION = "Track and query knowledge changes over time with temporal awareness"
    ICON = "temporal-knowledge"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for optional dependencies
        self.GraphitiIntegration = self.safe_import(
            'knowledge_engine.integrations.graphiti_integration.GraphitiIntegration',
            fallback_value=None,
            error_msg="GraphitiIntegration not available for TemporalKnowledgeNode"
        )

        self.ChronicleIntegration = self.safe_import(
            'knowledge_engine.chronicle.chronicle.ChronicleIntegration',
            fallback_value=None,
            error_msg="ChronicleIntegration not available for TemporalKnowledgeNode"
        )

        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for TemporalKnowledgeNode"
        )

        # Alternative import paths
        if self.GraphitiIntegration is None:
            self.GraphitiIntegration = self.safe_import(
                'integrations.graphiti_integration.GraphitiIntegration',
                fallback_value=None,
                error_msg="GraphitiIntegration not found in alternate path"
            )

        if self.ChronicleIntegration is None:
            self.ChronicleIntegration = self.safe_import(
                'chronicle_memory.ChronicleIntegration',
                fallback_value=None,
                error_msg="ChronicleIntegration not found in alternate path"
            )

        if self.UnifiedKGIntegrationHub is None:
            self.UnifiedKGIntegrationHub = self.safe_import(
                'unified_kg_integration_hub.UnifiedKGIntegrationHub',
                fallback_value=None,
                error_msg="UnifiedKGIntegrationHub not found in alternate path"
            )

        # Initialize component instances
        self.graphiti_client = None
        self.chronicle_integration = None
        self.kg_hub = None
        self._initialized = False

        # Initialize if configuration provides connection details
        self._initialize_components()

    def _initialize_components(self):
        """Initialize Graphiti and Chronicle components if config available."""
        # Initialize Graphiti if credentials provided
        if self.GraphitiIntegration:
            uri = self.config.get('graphiti_uri')
            user = self.config.get('graphiti_user')
            password = self.config.get('graphiti_password')

            if uri and user and password:
                try:
                    self.graphiti_client = self.GraphitiIntegration(
                        uri=uri,
                        user=user,
                        password=password
                    )
                    self.logger.info("GraphitiIntegration initialized for TemporalKnowledgeNode")
                except Exception as e:
                    self.logger.warning(f"Could not initialize GraphitiIntegration: {e}")
                    self.graphiti_client = None

        # Initialize Chronicle if available
        if self.ChronicleIntegration:
            try:
                # Try to get chronicle from config or create new
                chronicle = self.config.get('chronicle_instance')
                if chronicle:
                    self.chronicle_integration = self.ChronicleIntegration(chronicle)
                    self.logger.info("ChronicleIntegration initialized for TemporalKnowledgeNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize ChronicleIntegration: {e}")
                self.chronicle_integration = None

        # Initialize KG Hub if available
        if self.UnifiedKGIntegrationHub:
            try:
                self.kg_hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for TemporalKnowledgeNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.kg_hub = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - store: knowledge (dict) or content (str)
        - query_at_time: timestamp (ISO datetime)
        - track_changes: entity_id (str)
        - compare_periods: start_time, end_time (ISO datetime)
        - get_history: entity_id (str)
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'store'))

        valid_operations = ['store', 'query_at_time', 'track_changes', 'compare_periods', 'get_history']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Operation-specific validation
        if operation == 'store':
            if 'knowledge' not in inputs and 'content' not in inputs:
                errors.append("Store operation requires 'knowledge' (dict) or 'content' (str) input")

            # Validate timestamps if provided
            for field in ['valid_from', 'valid_until']:
                if field in inputs:
                    try:
                        datetime.fromisoformat(inputs[field].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        errors.append(f"Invalid ISO datetime format for {field}: {inputs[field]}")

        elif operation == 'query_at_time':
            timestamp = inputs.get('timestamp') or self.config.get('timestamp')
            if not timestamp:
                errors.append("Query at time operation requires 'timestamp' (ISO datetime)")
            else:
                try:
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid ISO datetime format for timestamp: {timestamp}")

        elif operation == 'track_changes':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            if not entity_id:
                errors.append("Track changes operation requires 'entity_id'")

        elif operation == 'compare_periods':
            start_time = inputs.get('start_time') or self.config.get('start_time')
            end_time = inputs.get('end_time') or self.config.get('end_time')

            if not start_time:
                errors.append("Compare periods operation requires 'start_time' (ISO datetime)")
            else:
                try:
                    datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid ISO datetime format for start_time: {start_time}")

            if not end_time:
                errors.append("Compare periods operation requires 'end_time' (ISO datetime)")
            else:
                try:
                    datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid ISO datetime format for end_time: {end_time}")

        elif operation == 'get_history':
            entity_id = inputs.get('entity_id') or self.config.get('entity_id')
            if not entity_id:
                errors.append("Get history operation requires 'entity_id'")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the temporal knowledge operation based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing operation results with temporal data, changes, or historical states

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'store'))

        context.update_progress(10, f"Starting {operation} operation")
        self.logger.info(f"Executing temporal knowledge operation: {operation}")

        try:
            # Route to appropriate operation handler
            if operation == 'store':
                result = self._execute_store(inputs, context)
            elif operation == 'query_at_time':
                result = self._execute_query_at_time(inputs, context)
            elif operation == 'track_changes':
                result = self._execute_track_changes(inputs, context)
            elif operation == 'compare_periods':
                result = self._execute_compare_periods(inputs, context)
            elif operation == 'get_history':
                result = self._execute_get_history(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['store', 'query_at_time', 'track_changes', 'compare_periods', 'get_history']}
                )

            context.update_progress(100, f"{operation} operation completed")

            # Add artifact to context
            context.add_artifact('temporal_knowledge', {
                'operation': operation,
                'success': result.get('success', True),
                'result_summary': self._summarize_result(result)
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Temporal knowledge {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': inputs,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_store(self, inputs: Dict, context) -> Dict[str, Any]:
        """Store knowledge with temporal metadata."""
        context.update_progress(30, "Preparing knowledge for temporal storage")

        # Extract knowledge data
        knowledge = inputs.get('knowledge', {})
        content = inputs.get('content', '')
        entity_id = inputs.get('entity_id', self.config.get('entity_id', 'unknown'))

        # Parse timestamps
        valid_from_str = inputs.get('valid_from') or self.config.get('valid_from')
        valid_until_str = inputs.get('valid_until') or self.config.get('valid_until')

        valid_from = self._parse_timestamp(valid_from_str) or datetime.now(timezone.utc)
        valid_until = self._parse_timestamp(valid_until_str)

        context.update_progress(50, "Storing knowledge with temporal metadata")

        # Try Graphiti first, fallback to in-memory
        if self.graphiti_client:
            try:
                import asyncio
                import uuid

                # Create knowledge artifact
                artifact_data = {
                    'id': str(uuid.uuid4()),
                    'content': content or knowledge.get('content', str(knowledge)),
                    'artifact_type': knowledge.get('type', 'knowledge'),
                    'valid_at': valid_from,
                    'invalid_at': valid_until,
                    'metadata': {
                        'entity_id': entity_id,
                        'source': knowledge.get('source', 'temporal_node'),
                        **knowledge.get('metadata', {})
                    },
                    'source': knowledge.get('source', 'temporal_node'),
                    'group_id': knowledge.get('group_id', 'default')
                }

                # Get KnowledgeArtifact class
                KnowledgeArtifact = self.safe_import(
                    'knowledge_engine.integrations.graphiti_integration.KnowledgeArtifact',
                    fallback_value=None
                )

                if KnowledgeArtifact:
                    artifact = KnowledgeArtifact(**artifact_data)

                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(
                        self.graphiti_client.add_artifact(artifact)
                    )

                    context.update_progress(90, "Knowledge stored in Graphiti")

                    return {
                        'success': result.get('success', True),
                        'artifact_id': result.get('artifact_id'),
                        'episode_id': result.get('episode_id'),
                        'entities_extracted': result.get('entities_extracted', 0),
                        'relationships_extracted': result.get('relationships_extracted', 0),
                        'valid_from': valid_from.isoformat(),
                        'valid_until': valid_until.isoformat() if valid_until else None,
                        'entity_id': entity_id,
                        'storage_backend': 'graphiti'
                    }
            except Exception as e:
                self.logger.warning(f"Graphiti store failed, using fallback: {e}")

        # Fallback: Store in-memory
        context.update_progress(60, "Using in-memory storage fallback")

        stored_knowledge = {
            'entity_id': entity_id,
            'content': content or knowledge.get('content', str(knowledge)),
            'knowledge': knowledge,
            'valid_from': valid_from.isoformat(),
            'valid_until': valid_until.isoformat() if valid_until else None,
            'stored_at': datetime.now(timezone.utc).isoformat()
        }

        # Store in context for retrieval
        temporal_store = context.get_artifact('temporal_store') or []
        temporal_store.append(stored_knowledge)
        context.add_artifact('temporal_store', temporal_store)

        context.update_progress(90, "Knowledge stored in memory")

        return {
            'success': True,
            'artifact_id': f"mem_{entity_id}_{valid_from.timestamp()}",
            'entity_id': entity_id,
            'valid_from': valid_from.isoformat(),
            'valid_until': valid_until.isoformat() if valid_until else None,
            'storage_backend': 'memory',
            'warning': 'Graphiti not available, stored in memory only'
        }

    def _execute_query_at_time(self, inputs: Dict, context) -> Dict[str, Any]:
        """Query knowledge at a specific point in time."""
        context.update_progress(30, "Parsing query timestamp")

        timestamp_str = inputs.get('timestamp') or self.config.get('timestamp')
        timestamp = self._parse_timestamp(timestamp_str)

        if not timestamp:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Invalid timestamp for query_at_time operation",
                details={'timestamp': timestamp_str}
            )

        query = inputs.get('query', '*')  # Default query matches all
        entity_id = inputs.get('entity_id') or self.config.get('entity_id')

        context.update_progress(50, f"Querying knowledge at {timestamp.isoformat()}")

        # Try Graphiti first
        if self.graphiti_client:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                artifacts = loop.run_until_complete(
                    self.graphiti_client.query_at_point_in_time(
                        query=query,
                        timestamp=timestamp,
                        max_results=inputs.get('max_results', 10)
                    )
                )

                context.update_progress(90, "Retrieved knowledge from Graphiti")

                return {
                    'success': True,
                    'query_time': timestamp.isoformat(),
                    'query': query,
                    'entity_id': entity_id,
                    'results_count': len(artifacts),
                    'knowledge_items': [a.to_dict() if hasattr(a, 'to_dict') else str(a) for a in artifacts],
                    'storage_backend': 'graphiti'
                }
            except Exception as e:
                self.logger.warning(f"Graphiti query failed, using fallback: {e}")

        # Fallback: Query in-memory store
        context.update_progress(60, "Using in-memory query fallback")

        temporal_store = context.get_artifact('temporal_store') or []
        matching_items = []

        for item in temporal_store:
            item_valid_from = self._parse_timestamp(item.get('valid_from'))
            item_valid_until = self._parse_timestamp(item.get('valid_until'))

            # Check if item was valid at the query time
            if item_valid_from and item_valid_from <= timestamp:
                if not item_valid_until or item_valid_until >= timestamp:
                    # Filter by entity_id if specified
                    if not entity_id or item.get('entity_id') == entity_id:
                        matching_items.append(item)

        context.update_progress(90, f"Retrieved {len(matching_items)} items from memory")

        return {
            'success': True,
            'query_time': timestamp.isoformat(),
            'query': query,
            'entity_id': entity_id,
            'results_count': len(matching_items),
            'knowledge_items': matching_items,
            'storage_backend': 'memory',
            'warning': 'Graphiti not available, queried memory only'
        }

    def _execute_track_changes(self, inputs: Dict, context) -> Dict[str, Any]:
        """Track knowledge evolution/changes over time for an entity."""
        context.update_progress(30, "Preparing change tracking")

        entity_id = inputs.get('entity_id') or self.config.get('entity_id')
        if not entity_id:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="entity_id is required for track_changes operation"
            )

        time_window_days = inputs.get('time_window_days', 30)

        context.update_progress(50, f"Tracking changes for entity: {entity_id}")

        # Try Graphiti first
        if self.graphiti_client:
            try:
                import asyncio

                end_time = datetime.now(timezone.utc)
                start_time = end_time - __import__('datetime').timedelta(days=time_window_days)

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                timeline = loop.run_until_complete(
                    self.graphiti_client.get_entity_timeline(
                        entity_name=entity_id,
                        start_time=start_time,
                        end_time=end_time
                    )
                )

                context.update_progress(90, "Retrieved change history from Graphiti")

                # Analyze changes
                changes = self._analyze_changes(timeline)

                return {
                    'success': True,
                    'entity_id': entity_id,
                    'time_window_days': time_window_days,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_events': len(timeline),
                    'changes': changes,
                    'timeline': timeline,
                    'storage_backend': 'graphiti'
                }
            except Exception as e:
                self.logger.warning(f"Graphiti timeline failed, using fallback: {e}")

        # Fallback: Track changes from in-memory store
        context.update_progress(60, "Using in-memory change tracking fallback")

        temporal_store = context.get_artifact('temporal_store') or []
        entity_items = [item for item in temporal_store if item.get('entity_id') == entity_id]

        # Sort by valid_from
        entity_items.sort(key=lambda x: x.get('valid_from', ''))

        changes = self._analyze_changes(entity_items)

        context.update_progress(90, f"Tracked {len(changes)} changes from memory")

        return {
            'success': True,
            'entity_id': entity_id,
            'time_window_days': time_window_days,
            'total_events': len(entity_items),
            'changes': changes,
            'timeline': entity_items,
            'storage_backend': 'memory',
            'warning': 'Graphiti not available, tracked from memory only'
        }

    def _execute_compare_periods(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compare knowledge between two time periods."""
        context.update_progress(30, "Parsing period boundaries")

        start_time_str = inputs.get('start_time') or self.config.get('start_time')
        end_time_str = inputs.get('end_time') or self.config.get('end_time')
        entity_id = inputs.get('entity_id') or self.config.get('entity_id')

        start_time = self._parse_timestamp(start_time_str)
        end_time = self._parse_timestamp(end_time_str)

        if not start_time or not end_time:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Invalid start_time or end_time for compare_periods operation",
                details={'start_time': start_time_str, 'end_time': end_time_str}
            )

        context.update_progress(50, f"Comparing periods: {start_time.isoformat()} to {end_time.isoformat()}")

        # Try Graphiti first
        if self.graphiti_client:
            try:
                import asyncio

                # Query at start and end of period
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                start_artifacts = loop.run_until_complete(
                    self.graphiti_client.query_at_point_in_time(
                        query=entity_id or '*',
                        timestamp=start_time,
                        max_results=50
                    )
                )

                end_artifacts = loop.run_until_complete(
                    self.graphiti_client.query_at_point_in_time(
                        query=entity_id or '*',
                        timestamp=end_time,
                        max_results=50
                    )
                )

                comparison = self._compare_artifact_sets(start_artifacts, end_artifacts)

                context.update_progress(90, "Period comparison complete")

                return {
                    'success': True,
                    'entity_id': entity_id,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'start_count': len(start_artifacts),
                    'end_count': len(end_artifacts),
                    'comparison': comparison,
                    'storage_backend': 'graphiti'
                }
            except Exception as e:
                self.logger.warning(f"Graphiti comparison failed, using fallback: {e}")

        # Fallback: Compare from in-memory store
        context.update_progress(60, "Using in-memory comparison fallback")

        temporal_store = context.get_artifact('temporal_store') or []

        # Filter items valid at start and end times
        start_items = []
        end_items = []

        for item in temporal_store:
            item_valid_from = self._parse_timestamp(item.get('valid_from'))
            item_valid_until = self._parse_timestamp(item.get('valid_until'))

            if entity_id and item.get('entity_id') != entity_id:
                continue

            # Check validity at start_time
            if item_valid_from and item_valid_from <= start_time:
                if not item_valid_until or item_valid_until >= start_time:
                    start_items.append(item)

            # Check validity at end_time
            if item_valid_from and item_valid_from <= end_time:
                if not item_valid_until or item_valid_until >= end_time:
                    end_items.append(item)

        comparison = self._compare_item_sets(start_items, end_items)

        context.update_progress(90, "Period comparison complete")

        return {
            'success': True,
            'entity_id': entity_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'start_count': len(start_items),
            'end_count': len(end_items),
            'comparison': comparison,
            'storage_backend': 'memory',
            'warning': 'Graphiti not available, compared from memory only'
        }

    def _execute_get_history(self, inputs: Dict, context) -> Dict[str, Any]:
        """Get complete history for an entity."""
        context.update_progress(30, "Preparing history retrieval")

        entity_id = inputs.get('entity_id') or self.config.get('entity_id')
        if not entity_id:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="entity_id is required for get_history operation"
            )

        context.update_progress(50, f"Retrieving history for entity: {entity_id}")

        # Try Graphiti first
        if self.graphiti_client:
            try:
                import asyncio
                from datetime import datetime, timezone, timedelta

                # Get full timeline
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=365 * 10)  # 10 years back

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                timeline = loop.run_until_complete(
                    self.graphiti_client.get_entity_timeline(
                        entity_name=entity_id,
                        start_time=start_time,
                        end_time=end_time
                    )
                )

                context.update_progress(90, "History retrieved from Graphiti")

                return {
                    'success': True,
                    'entity_id': entity_id,
                    'total_events': len(timeline),
                    'history': timeline,
                    'first_seen': timeline[0].get('timestamp') if timeline else None,
                    'last_seen': timeline[-1].get('timestamp') if timeline else None,
                    'storage_backend': 'graphiti'
                }
            except Exception as e:
                self.logger.warning(f"Graphiti history failed, using fallback: {e}")

        # Fallback: Get history from in-memory store
        context.update_progress(60, "Using in-memory history fallback")

        temporal_store = context.get_artifact('temporal_store') or []
        entity_items = [item for item in temporal_store if item.get('entity_id') == entity_id]

        # Sort by valid_from chronologically
        entity_items.sort(key=lambda x: x.get('valid_from', ''))

        context.update_progress(90, f"Retrieved {len(entity_items)} history items from memory")

        return {
            'success': True,
            'entity_id': entity_id,
            'total_events': len(entity_items),
            'history': entity_items,
            'first_seen': entity_items[0].get('valid_from') if entity_items else None,
            'last_seen': entity_items[-1].get('valid_from') if entity_items else None,
            'storage_backend': 'memory',
            'warning': 'Graphiti not available, retrieved from memory only'
        }

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO format timestamp string to datetime."""
        if not timestamp_str:
            return None
        try:
            # Handle Z suffix
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, AttributeError):
            return None

    def _analyze_changes(self, timeline: List[Dict]) -> List[Dict]:
        """Analyze changes from a timeline."""
        changes = []

        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]

            change = {
                'change_type': 'update',
                'timestamp': curr.get('timestamp') or curr.get('valid_from'),
                'previous_state': prev,
                'current_state': curr
            }
            changes.append(change)

        return changes

    def _compare_artifact_sets(self, start_artifacts: List[Any], end_artifacts: List[Any]) -> Dict[str, Any]:
        """Compare two sets of artifacts."""
        start_ids = {getattr(a, 'id', str(a)) for a in start_artifacts}
        end_ids = {getattr(a, 'id', str(a)) for a in end_artifacts}

        added = end_ids - start_ids
        removed = start_ids - end_ids
        common = start_ids & end_ids

        return {
            'added_count': len(added),
            'removed_count': len(removed),
            'common_count': len(common),
            'added_ids': list(added),
            'removed_ids': list(removed),
            'net_change': len(added) - len(removed)
        }

    def _compare_item_sets(self, start_items: List[Dict], end_items: List[Dict]) -> Dict[str, Any]:
        """Compare two sets of items."""
        start_ids = {item.get('artifact_id', str(item)) for item in start_items}
        end_ids = {item.get('artifact_id', str(item)) for item in end_items}

        added = end_ids - start_ids
        removed = start_ids - end_ids
        common = start_ids & end_ids

        return {
            'added_count': len(added),
            'removed_count': len(removed),
            'common_count': len(common),
            'added_ids': list(added),
            'removed_ids': list(removed),
            'net_change': len(added) - len(removed)
        }

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the result for artifact storage."""
        summary = {'success': result.get('success', False)}

        if 'entity_id' in result:
            summary['entity_id'] = result['entity_id']
        if 'results_count' in result:
            summary['results_count'] = result['results_count']
        if 'total_events' in result:
            summary['total_events'] = result['total_events']
        if 'storage_backend' in result:
            summary['storage_backend'] = result['storage_backend']

        return summary

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all operation types and parameters.
        """
        return {
            "type": "object",
            "title": "Temporal Knowledge Configuration",
            "description": "Configure temporal knowledge tracking and querying with Graphiti",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The temporal knowledge operation to perform",
                    "enum": ["store", "query_at_time", "track_changes", "compare_periods", "get_history"],
                    "enumNames": [
                        "Store - Save knowledge with temporal metadata",
                        "Query at Time - Retrieve knowledge valid at a specific time",
                        "Track Changes - Monitor knowledge evolution over time",
                        "Compare Periods - Compare knowledge between two time periods",
                        "Get History - Retrieve complete history for an entity"
                    ],
                    "default": "store"
                },
                "timestamp": {
                    "type": "string",
                    "title": "Timestamp",
                    "description": "ISO datetime for query_at_time operation (e.g., 2026-01-31T12:00:00Z)",
                    "default": ""
                },
                "valid_from": {
                    "type": "string",
                    "title": "Valid From",
                    "description": "ISO datetime when knowledge becomes valid (for store operation)",
                    "default": ""
                },
                "valid_until": {
                    "type": "string",
                    "title": "Valid Until",
                    "description": "ISO datetime when knowledge expires (optional, for store operation)",
                    "default": ""
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "Entity identifier to track or query",
                    "default": ""
                },
                "start_time": {
                    "type": "string",
                    "title": "Start Time",
                    "description": "ISO datetime for period start (for compare_periods operation)",
                    "default": ""
                },
                "end_time": {
                    "type": "string",
                    "title": "End Time",
                    "description": "ISO datetime for period end (for compare_periods operation)",
                    "default": ""
                },
                "graphiti_uri": {
                    "type": "string",
                    "title": "Graphiti URI",
                    "description": "Neo4j connection URI for Graphiti (e.g., bolt://localhost:7687)",
                    "default": ""
                },
                "graphiti_user": {
                    "type": "string",
                    "title": "Graphiti User",
                    "description": "Neo4j username for Graphiti connection",
                    "default": ""
                },
                "graphiti_password": {
                    "type": "string",
                    "title": "Graphiti Password",
                    "description": "Neo4j password for Graphiti connection",
                    "default": ""
                },
                "time_window_days": {
                    "type": "number",
                    "title": "Time Window (Days)",
                    "description": "Number of days to look back for track_changes operation",
                    "minimum": 1,
                    "maximum": 3650,
                    "default": 30
                },
                "max_results": {
                    "type": "number",
                    "title": "Maximum Results",
                    "description": "Maximum number of results to return for queries",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 10
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["store"]}
                            },
                            "description": "Store knowledge with temporal metadata"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["query_at_time"]}
                            },
                            "required": ["timestamp"],
                            "description": "Query knowledge at a specific point in time"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["track_changes"]}
                            },
                            "required": ["entity_id"],
                            "description": "Track knowledge changes over time for an entity"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["compare_periods"]}
                            },
                            "required": ["start_time", "end_time"],
                            "description": "Compare knowledge between two time periods"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["get_history"]}
                            },
                            "required": ["entity_id"],
                            "description": "Retrieve complete history for an entity"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least one storage backend is available, False otherwise
        """
        try:
            # Check if Graphiti is available
            graphiti_available = self.GraphitiIntegration is not None

            # Check if Chronicle is available
            chronicle_available = self.ChronicleIntegration is not None

            # Check if KG Hub is available
            kg_hub_available = self.UnifiedKGIntegrationHub is not None

            # Node is healthy if at least Graphiti class is available
            # (fallback to memory will work for basic operations)
            return graphiti_available or chronicle_available or kg_hub_available
        except Exception:
            return False
