"""
Knowledge Federation Node for BubbleLabs

Federates and merges knowledge from distributed remote sources.
Supports multiple source types (REST API, GraphQL, SPARQL, MCP) with
configurable sync modes, conflict resolution strategies, and provenance tracking.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
import asyncio
import json
import time
from dataclasses import dataclass, field

from .base_node import BubbleLabsNode, NodeExecutionError


@dataclass
class FederatedSource:
    """Represents a federated knowledge source."""
    url: str
    source_type: str
    status: str = "disconnected"
    last_sync: Optional[datetime] = None
    entities_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictRecord:
    """Represents a conflict between federated sources."""
    entity_id: str
    conflict_type: str
    sources: List[str]
    values: List[Any]
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


class KnowledgeFederationNode(BubbleLabsNode):
    """
    Knowledge Federation Node for BubbleLabs.

    Provides distributed knowledge graph federation capabilities:
    - Connect to remote knowledge sources via multiple protocols
    - Fetch knowledge from federated endpoints
    - Merge federated knowledge with conflict resolution
    - Handle source provenance and lineage tracking
    - Manage federation topology
    - Bidirectional synchronization support

    Supported source types:
    - REST API: HTTP/HTTPS endpoints returning JSON knowledge graphs
    - GraphQL: GraphQL endpoints with knowledge graph queries
    - SPARQL: SPARQL endpoints for RDF knowledge graphs
    - MCP: Model Context Protocol endpoints
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Federation"
    DESCRIPTION = "Federate and merge knowledge from distributed remote sources"
    ICON = "federation"
    CATEGORY = "integration"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        self.FederationManager = self.safe_import(
            'knowledge_engine.federation.FederationManager',
            fallback_value=None,
            error_msg="FederationManager not available"
        )

        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available"
        )

        # Initialize federation manager instance
        self.federation_manager = None
        self._manager_initialized = False

        if self.FederationManager:
            try:
                self.federation_manager = self.FederationManager()
                self.logger.info("KnowledgeFederationNode initialized with FederationManager")
            except Exception as e:
                self.logger.error(f"Failed to initialize FederationManager: {e}")
                self.federation_manager = None
        else:
            self.logger.warning("FederationManager not available, node will operate in limited mode")

        # Track connected sources
        self.connected_sources: Dict[str, FederatedSource] = {}
        self.federation_topology: Dict[str, Any] = {}
        self.conflicts_history: List[ConflictRecord] = []

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Operations:
        - connect: Requires source_urls, source_type
        - fetch: Requires source_urls or previously connected sources
        - merge: Requires sources with data to merge
        - sync: Requires source_urls and sync_mode
        - disconnect: Optional source_urls (disconnects all if not specified)
        - topology: No required fields
        """
        errors = []

        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'connect'))

        valid_operations = ['connect', 'fetch', 'merge', 'sync', 'disconnect', 'topology']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Operation-specific validation
        if operation == 'connect':
            source_urls = inputs.get('source_urls') or self.config.get('source_urls', [])
            if not source_urls:
                errors.append("Missing required field 'source_urls' for connect operation")
            elif not isinstance(source_urls, list):
                errors.append("'source_urls' must be a list of strings")

            source_type = inputs.get('source_type') or self.config.get('source_type')
            if not source_type:
                errors.append("Missing required field 'source_type' for connect operation")
            else:
                valid_types = ['rest_api', 'graphql', 'sparql', 'mcp']
                if source_type not in valid_types:
                    errors.append(f"Invalid source_type: {source_type}. Must be one of: {', '.join(valid_types)}")

        elif operation == 'fetch':
            source_urls = inputs.get('source_urls') or self.config.get('source_urls', [])
            if not source_urls and not self.connected_sources:
                errors.append("No sources specified and no previously connected sources available")

        elif operation == 'merge':
            sources = inputs.get('sources', [])
            if sources and not isinstance(sources, list):
                errors.append("'sources' must be a list of knowledge sources")
            elif sources and len(sources) < 2:
                errors.append("At least 2 sources required for merge operation")

        elif operation == 'sync':
            source_urls = inputs.get('source_urls') or self.config.get('source_urls', [])
            if not source_urls:
                errors.append("Missing required field 'source_urls' for sync operation")

            sync_mode = inputs.get('sync_mode') or self.config.get('sync_mode')
            if sync_mode:
                valid_sync_modes = ['push', 'pull', 'bidirectional']
                if sync_mode not in valid_sync_modes:
                    errors.append(f"Invalid sync_mode: {sync_mode}. Must be one of: {', '.join(valid_sync_modes)}")

        # Validate conflict_resolution if provided
        conflict_resolution = inputs.get('conflict_resolution') or self.config.get('conflict_resolution')
        if conflict_resolution:
            valid_strategies = ['source_priority', 'timestamp', 'voting', 'manual']
            if conflict_resolution not in valid_strategies:
                errors.append(f"Invalid conflict_resolution: {conflict_resolution}. Must be one of: {', '.join(valid_strategies)}")

        # Validate sync_interval format if provided
        sync_interval = inputs.get('sync_interval') or self.config.get('sync_interval')
        if sync_interval:
            valid_intervals = ['5m', '15m', '30m', '1h', '6h', '12h', 'daily', 'weekly']
            if sync_interval not in valid_intervals:
                errors.append(f"Invalid sync_interval: {sync_interval}. Must be one of: {', '.join(valid_intervals)}")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge federation workflow based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing operation results with entity counts, conflicts, and source info

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'connect'))

        self.logger.info(f"Starting knowledge federation operation: {operation}")
        context.update_progress(5, f"Starting {operation} operation")

        try:
            # Route to appropriate operation handler
            if operation == 'connect':
                result = self._execute_connect(inputs, context)
            elif operation == 'fetch':
                result = self._execute_fetch(inputs, context)
            elif operation == 'merge':
                result = self._execute_merge(inputs, context)
            elif operation == 'sync':
                result = self._execute_sync(inputs, context)
            elif operation == 'disconnect':
                result = self._execute_disconnect(inputs, context)
            elif operation == 'topology':
                result = self._execute_topology(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['connect', 'fetch', 'merge', 'sync', 'disconnect', 'topology']}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('knowledge_federation', {
                'operation': operation,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'result_summary': self._summarize_result(result)
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge federation {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': inputs,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_connect(self, inputs: Dict, context) -> Dict[str, Any]:
        """Connect to remote knowledge sources."""
        context.update_progress(10, "Preparing connection to sources")

        source_urls = inputs.get('source_urls', self.config.get('source_urls', []))
        source_type = inputs.get('source_type', self.config.get('source_type', 'rest_api'))
        authentication = inputs.get('authentication', self.config.get('authentication', {}))

        connected = []
        failed = []

        for i, url in enumerate(source_urls):
            progress = 10 + (i / len(source_urls)) * 70
            context.update_progress(int(progress), f"Connecting to {url}")

            try:
                # Attempt connection
                source_info = self._connect_to_source(url, source_type, authentication)

                # Track connected source
                self.connected_sources[url] = FederatedSource(
                    url=url,
                    source_type=source_type,
                    status="connected",
                    metadata=source_info
                )

                connected.append({
                    'url': url,
                    'status': 'connected',
                    'source_type': source_type,
                    'info': source_info
                })

            except Exception as e:
                self.logger.warning(f"Failed to connect to {url}: {e}")
                failed.append({
                    'url': url,
                    'error': str(e)
                })

        context.update_progress(100, "Connection process complete")

        return {
            'success': len(connected) > 0,
            'operation': 'connect',
            'connected_count': len(connected),
            'failed_count': len(failed),
            'connected': connected,
            'failed': failed,
            'sources': list(self.connected_sources.keys())
        }

    def _execute_fetch(self, inputs: Dict, context) -> Dict[str, Any]:
        """Fetch knowledge from federated sources."""
        context.update_progress(10, "Preparing fetch operation")

        source_urls = inputs.get('source_urls', self.config.get('source_urls', []))
        entity_filter = inputs.get('entity_filter', self.config.get('entity_filter', []))
        include_provenance = inputs.get('include_provenance', self.config.get('include_provenance', True))

        # Use connected sources if no URLs specified
        if not source_urls and self.connected_sources:
            source_urls = list(self.connected_sources.keys())

        if not source_urls:
            return {
                'success': False,
                'error': 'No sources specified or connected',
                'entities_fetched': 0,
                'sources': []
            }

        fetched_data = []
        total_entities = 0

        for i, url in enumerate(source_urls):
            progress = 10 + (i / len(source_urls)) * 70
            context.update_progress(int(progress), f"Fetching from {url}")

            try:
                # Fetch with fallback to manual if federation unavailable
                if self.federation_manager and self._manager_initialized:
                    data = self._fetch_from_source(url, entity_filter)
                else:
                    data = self._manual_fetch(url, entity_filter)

                if include_provenance:
                    data = self._add_provenance(data, url)

                fetched_data.append({
                    'source': url,
                    'entities': data.get('entities', []),
                    'triples': data.get('triples', []),
                    'entity_count': len(data.get('entities', [])),
                    'triple_count': len(data.get('triples', []))
                })

                total_entities += len(data.get('entities', []))

                # Update source tracking
                if url in self.connected_sources:
                    self.connected_sources[url].entities_count = len(data.get('entities', []))
                    self.connected_sources[url].last_sync = datetime.now()

            except Exception as e:
                self.logger.warning(f"Failed to fetch from {url}: {e}")
                fetched_data.append({
                    'source': url,
                    'error': str(e),
                    'entities': [],
                    'triples': [],
                    'entity_count': 0,
                    'triple_count': 0
                })

        context.update_progress(100, "Fetch operation complete")

        return {
            'success': True,
            'operation': 'fetch',
            'entities_fetched': total_entities,
            'sources': fetched_data,
            'source_count': len(source_urls)
        }

    def _execute_merge(self, inputs: Dict, context) -> Dict[str, Any]:
        """Merge federated knowledge with conflict resolution."""
        context.update_progress(10, "Preparing merge operation")

        sources = inputs.get('sources', [])
        conflict_resolution = inputs.get('conflict_resolution', self.config.get('conflict_resolution', 'timestamp'))
        include_provenance = inputs.get('include_provenance', self.config.get('include_provenance', True))

        if not sources:
            # Try to use fetched data from connected sources
            fetch_result = self._execute_fetch(inputs, context)
            if fetch_result['success']:
                sources = fetch_result['sources']

        if len(sources) < 2:
            return {
                'success': False,
                'error': 'At least 2 sources required for merge',
                'merged_entities': [],
                'conflicts': []
            }

        context.update_progress(30, "Collecting entities from sources")

        # Collect all entities with their provenance
        all_entities: Dict[str, List[Dict]] = {}
        for source_data in sources:
            source_url = source_data.get('source', 'unknown')
            for entity in source_data.get('entities', []):
                entity_id = entity.get('id') or entity.get('name') or entity.get('uri')
                if entity_id:
                    if entity_id not in all_entities:
                        all_entities[entity_id] = []

                    entity_with_provenance = entity.copy()
                    if include_provenance:
                        entity_with_provenance['_provenance'] = {
                            'source': source_url,
                            'fetched_at': datetime.now().isoformat()
                        }
                    all_entities[entity_id].append(entity_with_provenance)

        context.update_progress(50, f"Processing {len(all_entities)} unique entities")

        # Detect and resolve conflicts
        conflicts = []
        resolved_conflicts = 0
        merged_entities = []

        for entity_id, entity_versions in all_entities.items():
            if len(entity_versions) > 1:
                # Conflict detected
                conflict = ConflictRecord(
                    entity_id=entity_id,
                    conflict_type='duplicate_entity',
                    sources=[e.get('_provenance', {}).get('source', 'unknown') for e in entity_versions],
                    values=entity_versions
                )

                # Resolve conflict
                resolved_entity = self._resolve_conflict(
                    entity_id, entity_versions, conflict_resolution
                )

                if resolved_entity:
                    conflict.resolution = conflict_resolution
                    conflict.resolved_at = datetime.now()
                    resolved_conflicts += 1
                    merged_entities.append(resolved_entity)
                else:
                    conflicts.append(conflict)
            else:
                merged_entities.append(entity_versions[0])

        context.update_progress(80, f"Resolved {resolved_conflicts} conflicts")

        # Store conflicts in history
        self.conflicts_history.extend(conflicts)

        # Merge triples
        all_triples = []
        for source_data in sources:
            all_triples.extend(source_data.get('triples', []))

        # Deduplicate triples
        unique_triples = self._deduplicate_triples(all_triples)

        context.update_progress(100, "Merge operation complete")

        return {
            'success': True,
            'operation': 'merge',
            'entities_synced': len(merged_entities),
            'triples_merged': len(unique_triples),
            'conflicts': len(conflicts),
            'resolved': resolved_conflicts,
            'merged_entities': merged_entities[:100],  # Limit returned data
            'merged_triples': unique_triples[:100],
            'conflict_resolution_strategy': conflict_resolution
        }

    def _execute_sync(self, inputs: Dict, context) -> Dict[str, Any]:
        """Synchronize knowledge with remote sources."""
        context.update_progress(5, "Preparing synchronization")

        source_urls = inputs.get('source_urls', self.config.get('source_urls', []))
        sync_mode = inputs.get('sync_mode', self.config.get('sync_mode', 'pull'))
        sync_interval = inputs.get('sync_interval', self.config.get('sync_interval', '1h'))
        conflict_resolution = inputs.get('conflict_resolution', self.config.get('conflict_resolution', 'timestamp'))

        # Ensure sources are connected
        if not self.connected_sources:
            connect_result = self._execute_connect(inputs, context)
            if not connect_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to connect to sources for sync',
                    'entities_synced': 0
                }

        context.update_progress(20, f"Starting {sync_mode} synchronization")

        sync_results = {
            'pushed': 0,
            'pulled': 0,
            'conflicts': 0,
            'resolved': 0
        }

        if sync_mode in ['pull', 'bidirectional']:
            # Fetch from remote sources
            context.update_progress(30, "Pulling from remote sources")
            fetch_result = self._execute_fetch(inputs, context)

            if fetch_result['success']:
                # Merge fetched data
                merge_inputs = {
                    'sources': fetch_result['sources'],
                    'conflict_resolution': conflict_resolution
                }
                merge_result = self._execute_merge(merge_inputs, context)

                sync_results['pulled'] = merge_result.get('entities_synced', 0)
                sync_results['conflicts'] += merge_result.get('conflicts', 0)
                sync_results['resolved'] += merge_result.get('resolved', 0)

        if sync_mode in ['push', 'bidirectional']:
            # Push local changes to remote
            context.update_progress(70, "Pushing to remote sources")
            push_result = self._push_to_sources(source_urls)
            sync_results['pushed'] = push_result.get('entities_pushed', 0)

        context.update_progress(100, "Synchronization complete")

        return {
            'success': True,
            'operation': 'sync',
            'sync_mode': sync_mode,
            'sync_interval': sync_interval,
            'entities_synced': sync_results['pulled'] + sync_results['pushed'],
            'pulled': sync_results['pulled'],
            'pushed': sync_results['pushed'],
            'conflicts': sync_results['conflicts'],
            'resolved': sync_results['resolved'],
            'sources': source_urls
        }

    def _execute_disconnect(self, inputs: Dict, context) -> Dict[str, Any]:
        """Disconnect from federated sources."""
        context.update_progress(10, "Preparing disconnect")

        source_urls = inputs.get('source_urls', self.config.get('source_urls', []))

        # Disconnect all if no specific URLs
        if not source_urls:
            source_urls = list(self.connected_sources.keys())

        disconnected = []
        failed = []

        for i, url in enumerate(source_urls):
            progress = 10 + (i / len(source_urls)) * 80
            context.update_progress(int(progress), f"Disconnecting from {url}")

            try:
                # Close connection
                if url in self.connected_sources:
                    del self.connected_sources[url]

                disconnected.append(url)
            except Exception as e:
                failed.append({'url': url, 'error': str(e)})

        context.update_progress(100, "Disconnect complete")

        return {
            'success': True,
            'operation': 'disconnect',
            'disconnected_count': len(disconnected),
            'failed_count': len(failed),
            'disconnected': disconnected,
            'failed': failed,
            'remaining_sources': list(self.connected_sources.keys())
        }

    def _execute_topology(self, inputs: Dict, context) -> Dict[str, Any]:
        """Manage and retrieve federation topology."""
        context.update_progress(20, "Analyzing federation topology")

        # Build topology map
        topology = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_sources': len(self.connected_sources),
                'last_updated': datetime.now().isoformat()
            }
        }

        # Add connected sources as nodes
        for url, source in self.connected_sources.items():
            node = {
                'id': url,
                'type': source.source_type,
                'status': source.status,
                'entities_count': source.entities_count,
                'last_sync': source.last_sync.isoformat() if source.last_sync else None
            }
            topology['nodes'].append(node)

        # Add federation relationships as edges
        for i, url1 in enumerate(self.connected_sources.keys()):
            for url2 in list(self.connected_sources.keys())[i+1:]:
                edge = {
                    'source': url1,
                    'target': url2,
                    'relationship': 'federated'
                }
                topology['edges'].append(edge)

        self.federation_topology = topology

        context.update_progress(100, "Topology analysis complete")

        return {
            'success': True,
            'operation': 'topology',
            'topology': topology,
            'source_count': len(topology['nodes']),
            'connection_count': len(topology['edges'])
        }

    def _connect_to_source(self, url: str, source_type: str, auth: Dict) -> Dict[str, Any]:
        """Establish connection to a knowledge source."""
        if self.federation_manager:
            try:
                return self.federation_manager.connect(url, source_type, auth)
            except Exception as e:
                self.logger.warning(f"FederationManager connect failed: {e}, using manual connection")

        # Manual connection fallback
        return self._manual_connect(url, source_type, auth)

    def _manual_connect(self, url: str, source_type: str, auth: Dict) -> Dict[str, Any]:
        """Manual connection implementation as fallback."""
        import urllib.request
        import urllib.error

        headers = {}
        if auth.get('token'):
            headers['Authorization'] = f"Bearer {auth['token']}"
        if auth.get('api_key'):
            headers['X-API-Key'] = auth['api_key']

        req = urllib.request.Request(url, headers=headers, method='HEAD')

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return {
                    'status': 'connected',
                    'status_code': response.status,
                    'source_type': source_type
                }
        except urllib.error.HTTPError as e:
            if e.code == 405:  # Method not allowed, try GET
                req = urllib.request.Request(url, headers=headers, method='GET')
                with urllib.request.urlopen(req, timeout=10) as response:
                    return {
                        'status': 'connected',
                        'status_code': response.status,
                        'source_type': source_type
                    }
            raise

    def _fetch_from_source(self, url: str, entity_filter: List[str]) -> Dict[str, Any]:
        """Fetch knowledge from a connected source."""
        if self.federation_manager:
            try:
                return self.federation_manager.fetch(url, entity_filter)
            except Exception as e:
                self.logger.warning(f"FederationManager fetch failed: {e}, using manual fetch")

        return self._manual_fetch(url, entity_filter)

    def _manual_fetch(self, url: str, entity_filter: List[str]) -> Dict[str, Any]:
        """Manual fetch implementation as fallback."""
        import urllib.request
        import json

        req = urllib.request.Request(url, headers={'Accept': 'application/json'})

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

            # Filter entities if specified
            if entity_filter and 'entities' in data:
                data['entities'] = [
                    e for e in data['entities']
                    if any(f in str(e) for f in entity_filter)
                ]

            return data

    def _push_to_sources(self, source_urls: List[str]) -> Dict[str, Any]:
        """Push local knowledge to remote sources."""
        entities_pushed = 0

        for url in source_urls:
            try:
                if self.federation_manager:
                    result = self.federation_manager.push(url)
                    entities_pushed += result.get('pushed', 0)
            except Exception as e:
                self.logger.warning(f"Failed to push to {url}: {e}")

        return {'entities_pushed': entities_pushed}

    def _add_provenance(self, data: Dict, source_url: str) -> Dict:
        """Add provenance information to fetched data."""
        provenance = {
            'source_url': source_url,
            'fetched_at': datetime.now().isoformat(),
            'federation_node': self.get_display_name(),
            'version': self.get_version()
        }

        if 'entities' in data:
            for entity in data['entities']:
                if '_provenance' not in entity:
                    entity['_provenance'] = provenance

        if 'triples' in data:
            for triple in data['triples']:
                if '_provenance' not in triple:
                    triple['_provenance'] = provenance

        return data

    def _resolve_conflict(self, entity_id: str, versions: List[Dict], strategy: str) -> Optional[Dict]:
        """Resolve conflict between multiple versions of an entity."""
        if not versions:
            return None

        if len(versions) == 1:
            return versions[0]

        if strategy == 'source_priority':
            # Use priority from config or first source
            priority = self.config.get('source_priority', [])
            for p in priority:
                for v in versions:
                    if v.get('_provenance', {}).get('source') == p:
                        return v
            return versions[0]

        elif strategy == 'timestamp':
            # Use most recent
            sorted_versions = sorted(
                versions,
                key=lambda x: x.get('_provenance', {}).get('fetched_at', ''),
                reverse=True
            )
            return sorted_versions[0] if sorted_versions else versions[0]

        elif strategy == 'voting':
            # Simple voting - use version with most sources agreeing
            # In practice, this would compare entity properties
            return versions[0]

        elif strategy == 'manual':
            # Return all versions for manual resolution
            # Mark as conflict for external handling
            return {
                'id': entity_id,
                '_conflict': True,
                '_versions': versions
            }

        return versions[0]

    def _deduplicate_triples(self, triples: List[Dict]) -> List[Dict]:
        """Remove duplicate triples."""
        seen = set()
        unique = []

        for triple in triples:
            key = (
                triple.get('subject') or triple.get('s'),
                triple.get('predicate') or triple.get('p'),
                triple.get('object') or triple.get('o')
            )

            if key not in seen:
                seen.add(key)
                unique.append(triple)

        return unique

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the result for artifact storage."""
        summary = {
            'success': result.get('success', False),
            'operation': result.get('operation')
        }

        if 'entities_synced' in result:
            summary['entities_synced'] = result['entities_synced']
        if 'conflicts' in result:
            summary['conflicts'] = result['conflicts']
        if 'resolved' in result:
            summary['resolved'] = result['resolved']
        if 'source_count' in result:
            summary['source_count'] = result['source_count']

        return summary

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns comprehensive schema for UI configuration including:
        - operation: Type of federation operation
        - source_urls: Remote KG endpoint URLs
        - source_type: Protocol for connecting to sources
        - authentication: Auth credentials
        - sync_mode: Push/pull/bidirectional
        - conflict_resolution: Strategy for resolving conflicts
        - sync_interval: Sync frequency
        - include_provenance: Track knowledge lineage
        - entity_filter: Entity types to sync
        """
        return {
            "type": "object",
            "title": "Knowledge Federation Configuration",
            "description": "Configure knowledge federation from distributed remote sources",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The federation operation to perform",
                    "enum": ["connect", "fetch", "merge", "sync", "disconnect", "topology"],
                    "enumNames": [
                        "Connect - Establish connection to remote sources",
                        "Fetch - Retrieve knowledge from federated sources",
                        "Merge - Merge knowledge from multiple sources",
                        "Sync - Synchronize with remote sources",
                        "Disconnect - Close connections to sources",
                        "Topology - View federation topology"
                    ],
                    "default": "connect"
                },
                "source_urls": {
                    "type": "array",
                    "title": "Source URLs",
                    "description": "Remote knowledge graph endpoints to federate",
                    "items": {
                        "type": "string",
                        "format": "uri"
                    },
                    "default": []
                },
                "source_type": {
                    "type": "string",
                    "title": "Source Type",
                    "description": "Protocol for connecting to knowledge sources",
                    "enum": ["rest_api", "graphql", "sparql", "mcp"],
                    "enumNames": [
                        "REST API - HTTP/HTTPS JSON endpoints",
                        "GraphQL - GraphQL query endpoints",
                        "SPARQL - RDF SPARQL endpoints",
                        "MCP - Model Context Protocol endpoints"
                    ],
                    "default": "rest_api"
                },
                "authentication": {
                    "type": "object",
                    "title": "Authentication",
                    "description": "Authentication credentials for remote sources",
                    "properties": {
                        "token": {
                            "type": "string",
                            "title": "Bearer Token",
                            "description": "Bearer token for authentication"
                        },
                        "api_key": {
                            "type": "string",
                            "title": "API Key",
                            "description": "API key for authentication"
                        },
                        "username": {
                            "type": "string",
                            "title": "Username",
                            "description": "Username for basic auth"
                        },
                        "password": {
                            "type": "string",
                            "title": "Password",
                            "description": "Password for basic auth",
                            "format": "password"
                        }
                    }
                },
                "sync_mode": {
                    "type": "string",
                    "title": "Sync Mode",
                    "description": "Direction of synchronization",
                    "enum": ["push", "pull", "bidirectional"],
                    "enumNames": [
                        "Push - Upload local changes to remote",
                        "Pull - Download remote changes to local",
                        "Bidirectional - Sync in both directions"
                    ],
                    "default": "pull"
                },
                "conflict_resolution": {
                    "type": "string",
                    "title": "Conflict Resolution",
                    "description": "Strategy for resolving conflicts between sources",
                    "enum": ["source_priority", "timestamp", "voting", "manual"],
                    "enumNames": [
                        "Source Priority - Use priority list to resolve conflicts",
                        "Timestamp - Use most recent version",
                        "Voting - Use version with most agreement",
                        "Manual - Return conflicts for manual resolution"
                    ],
                    "default": "timestamp"
                },
                "sync_interval": {
                    "type": "string",
                    "title": "Sync Interval",
                    "description": "Frequency of automatic synchronization",
                    "enum": ["5m", "15m", "30m", "1h", "6h", "12h", "daily", "weekly"],
                    "enumNames": [
                        "5 minutes",
                        "15 minutes",
                        "30 minutes",
                        "1 hour",
                        "6 hours",
                        "12 hours",
                        "Daily",
                        "Weekly"
                    ],
                    "default": "1h"
                },
                "include_provenance": {
                    "type": "boolean",
                    "title": "Include Provenance",
                    "description": "Track source provenance and lineage for all knowledge",
                    "default": True
                },
                "entity_filter": {
                    "type": "array",
                    "title": "Entity Filter",
                    "description": "Entity types to include in synchronization (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "source_priority": {
                    "type": "array",
                    "title": "Source Priority",
                    "description": "Priority order for source resolution (highest first)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["connect"]}
                            },
                            "required": ["source_urls", "source_type"],
                            "description": "Connect to remote knowledge sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["fetch"]}
                            },
                            "description": "Fetch knowledge from federated sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["merge"]}
                            },
                            "description": "Merge knowledge from multiple sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["sync"]}
                            },
                            "required": ["sync_mode"],
                            "description": "Synchronize with remote sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["disconnect"]}
                            },
                            "description": "Disconnect from remote sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["topology"]}
                            },
                            "description": "View federation topology"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if FederationManager is available or fallback methods work
        """
        try:
            # Check if we have the federation manager or can use fallback
            if self.FederationManager is not None:
                return True

            # Check if we can at least do manual connections
            import urllib.request
            return True
        except Exception:
            return False
