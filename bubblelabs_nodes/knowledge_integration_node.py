"""
Knowledge Integration Node for BubbleLabs

Integrates knowledge from multiple sources using the Unified KG Integration Hub.
Supports extraction, merging, and export of knowledge graphs from 40+ integrated systems.
"""

from typing import Dict, Any, List, Optional
import asyncio
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeIntegrationNode(BubbleLabsNode):
    """
    Knowledge Integration Node for BubbleLabs.

    Wraps the Unified KG Integration Hub to provide:
    - Multi-source knowledge extraction (DeepKE, OneKE, KG-Gen)
    - Knowledge merging and deduplication
    - Export in multiple formats (JSON, triples, NetworkX)
    - Health checking for all integrations
    - Reasoning and temporal tracking support
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Integration"
    DESCRIPTION = "Integrate knowledge from multiple sources using the Unified KG Integration Hub"
    ICON = "knowledge-integration"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.UnifiedKGIntegrationHub',
            error_msg="Knowledge Engine not available"
        )
        UnifiedKGConfig = self.safe_import(
            'knowledge_engine.UnifiedKGConfig',
            error_msg="Knowledge Engine config not available"
        )

        # Store references
        self.UnifiedKGIntegrationHub = UnifiedKGIntegrationHub
        self.UnifiedKGConfig = UnifiedKGConfig

        # Initialize hub instance
        self.hub = None
        self._hub_initialized = False

        if UnifiedKGIntegrationHub and UnifiedKGConfig:
            try:
                # Create config with user preferences
                kg_config = UnifiedKGConfig(
                    enable_deepke=True,
                    enable_oneke=True,
                    enable_kg_gen=True,
                    enable_reasoning=self.config.get('enable_reasoning', True),
                    enable_temporal_tracking=self.config.get('enable_temporal', True),
                    enable_verification=self.config.get('enable_reasoning', True),
                    enable_causal_analysis=self.config.get('enable_reasoning', True)
                )

                self.hub = UnifiedKGIntegrationHub(config=kg_config)
                self.logger.info("KnowledgeIntegrationNode initialized with UnifiedKGIntegrationHub")
            except Exception as e:
                self.logger.error(f"Failed to initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None
        else:
            self.logger.warning("Knowledge Engine components not available, node will operate in limited mode")

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - initialize: None (uses config)
        - extract: text (str)
        - merge: sources (list of knowledge sources)
        - export: None (optional format override)
        - health_check: None
        """
        errors = []

        # Check for operation type in inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'initialize'))

        valid_operations = ['initialize', 'extract', 'merge', 'export', 'health_check']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Operation-specific validation
        if operation == 'extract':
            if 'text' not in inputs:
                errors.append("Missing required field 'text' for extract operation")
            elif not isinstance(inputs['text'], str):
                errors.append("'text' must be a string")
            elif len(inputs['text'].strip()) == 0:
                errors.append("'text' cannot be empty")

            # Validate extractors if provided
            if 'extractors' in inputs:
                valid_extractors = ['deepke', 'oneke', 'kg_gen']
                extractors = inputs['extractors']
                if not isinstance(extractors, list):
                    errors.append("'extractors' must be a list")
                else:
                    for ext in extractors:
                        if ext not in valid_extractors:
                            errors.append(f"Invalid extractor: {ext}. Must be one of: {', '.join(valid_extractors)}")

        elif operation == 'merge':
            if 'sources' not in inputs:
                errors.append("Missing required field 'sources' for merge operation")
            elif not isinstance(inputs['sources'], list):
                errors.append("'sources' must be a list of knowledge sources")
            elif len(inputs['sources']) < 2:
                errors.append("At least 2 sources required for merge operation")

        elif operation == 'export':
            if 'export_format' in inputs:
                valid_formats = ['json', 'triples', 'networkx']
                if inputs['export_format'] not in valid_formats:
                    errors.append(f"Invalid export_format: {inputs['export_format']}. Must be one of: {', '.join(valid_formats)}")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge integration workflow based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress

        Returns:
            Dict containing operation results
        """
        if not self.UnifiedKGIntegrationHub:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Knowledge Engine not available",
                details={
                    'error': 'The knowledge_engine module must be available',
                    'hint': 'Ensure knowledge_engine is installed and accessible'
                }
            )

        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'initialize'))

        try:
            context.update_progress(10, f"Starting {operation} operation")

            # Run the appropriate operation
            if operation == 'initialize':
                result = self._execute_initialize(inputs, context)
            elif operation == 'extract':
                result = self._execute_extract(inputs, context)
            elif operation == 'merge':
                result = self._execute_merge(inputs, context)
            elif operation == 'export':
                result = self._execute_export(inputs, context)
            elif operation == 'health_check':
                result = self._execute_health_check(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['initialize', 'extract', 'merge', 'export', 'health_check']}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('knowledge_integration', {
                'operation': operation,
                'success': True,
                'result_summary': self._summarize_result(result)
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge integration {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': inputs,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_initialize(self, inputs: Dict, context) -> Dict[str, Any]:
        """Initialize the knowledge integration hub."""
        context.update_progress(30, "Initializing knowledge hub")

        if not self.hub:
            return {
                'success': False,
                'error': 'Knowledge hub not available',
                'initialized': False
            }

        try:
            # Run initialization in async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            initialized = loop.run_until_complete(self.hub.initialize())
            self._hub_initialized = initialized

            context.update_progress(70, "Hub initialization complete")

            # Get initialized integrations
            initialized_integrations = self.hub.registry.get_initialized()

            return {
                'success': initialized,
                'initialized': initialized,
                'integrations_count': len(initialized_integrations),
                'integrations': initialized_integrations,
                'config': {
                    'enable_deepke': self.hub.config.enable_deepke,
                    'enable_oneke': self.hub.config.enable_oneke,
                    'enable_kg_gen': self.hub.config.enable_kg_gen,
                    'enable_reasoning': self.hub.config.enable_verification,
                    'enable_temporal': self.hub.config.enable_temporal_tracking
                }
            }
        except Exception as e:
            return {
                'success': False,
                'initialized': False,
                'error': str(e),
                'integrations_count': 0,
                'integrations': []
            }

    def _execute_extract(self, inputs: Dict, context) -> Dict[str, Any]:
        """Extract knowledge from text using multiple extractors."""
        context.update_progress(20, "Preparing extraction")

        if not self.hub:
            return {
                'success': False,
                'error': 'Knowledge hub not available',
                'triples': [],
                'entity_count': 0,
                'relation_count': 0
            }

        # Initialize if not done
        if not self._hub_initialized:
            context.update_progress(30, "Initializing hub for extraction")
            init_result = self._execute_initialize({}, context)
            if not init_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to initialize hub for extraction',
                    'triples': []
                }

        text = inputs['text']
        extractors = inputs.get('extractors', ['deepke', 'oneke', 'kg_gen'])
        merge_results = inputs.get('merge_results', True)

        context.update_progress(40, f"Extracting with {len(extractors)} extractors")

        try:
            # Run extraction in async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            triples = loop.run_until_complete(
                self.hub.extract_knowledge(text, extractors, merge_results)
            )

            context.update_progress(80, "Processing extraction results")

            # Convert triples to dictionaries
            triples_dict = [t.to_dict() for t in triples]

            # Extract entities and relations
            entities = set()
            relations = set()
            for t in triples:
                entities.add(t.subject)
                entities.add(t.object)
                relations.add(t.predicate)

            context.update_progress(100, "Extraction complete")

            return {
                'success': True,
                'triples': triples_dict,
                'triple_count': len(triples),
                'entity_count': len(entities),
                'relation_count': len(relations),
                'entities': list(entities),
                'relations': list(relations),
                'extractors_used': extractors,
                'merged': merge_results
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'triples': [],
                'triple_count': 0,
                'entity_count': 0,
                'relation_count': 0
            }

    def _execute_merge(self, inputs: Dict, context) -> Dict[str, Any]:
        """Merge knowledge from multiple sources."""
        context.update_progress(20, "Preparing merge operation")

        if not self.hub:
            return {
                'success': False,
                'error': 'Knowledge hub not available',
                'merged_triples': [],
                'duplicates_removed': 0
            }

        sources = inputs['sources']
        context.update_progress(40, f"Processing {len(sources)} sources")

        try:
            # Import knowledge from each source
            total_imported = 0
            for i, source in enumerate(sources):
                progress = 40 + (i / len(sources)) * 30
                context.update_progress(int(progress), f"Importing source {i+1}/{len(sources)}")

                if isinstance(source, dict) and 'data' in source:
                    format_type = source.get('format', 'json')
                    data = source['data']
                    if isinstance(data, str):
                        success = self.hub.import_knowledge(data, format=format_type)
                        if success:
                            total_imported += 1

            context.update_progress(70, "Merging and deduplicating")

            # Merge triples
            merged_triples = self.hub._merge_triples(self.hub.triples)
            self.hub.triples = merged_triples

            # Run deduplication if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            dedup_result = loop.run_until_complete(self.hub.deduplicate_knowledge())
            duplicates_removed = dedup_result.get('duplicates_removed', 0)

            context.update_progress(100, "Merge complete")

            # Convert merged triples
            merged_dicts = [t.to_dict() for t in merged_triples]

            return {
                'success': True,
                'merged_triples': merged_dicts,
                'triple_count': len(merged_triples),
                'duplicates_removed': duplicates_removed,
                'sources_processed': total_imported
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'merged_triples': [],
                'triple_count': 0,
                'duplicates_removed': 0
            }

    def _execute_export(self, inputs: Dict, context) -> Dict[str, Any]:
        """Export integrated knowledge."""
        context.update_progress(30, "Preparing export")

        if not self.hub:
            return {
                'success': False,
                'error': 'Knowledge hub not available',
                'export_data': None
            }

        export_format = inputs.get('export_format', self.config.get('export_format', 'json'))
        include_metadata = inputs.get('include_metadata', True)

        context.update_progress(50, f"Exporting in {export_format} format")

        try:
            export_result = self.hub.export_knowledge(
                format=export_format,
                include_metadata=include_metadata
            )

            context.update_progress(100, "Export complete")

            if export_format == 'json' and isinstance(export_result, str):
                import json
                export_data = json.loads(export_result)
            else:
                export_data = export_result

            return {
                'success': True,
                'export_format': export_format,
                'export_data': export_data,
                'triple_count': len(self.hub.triples),
                'entity_count': len(self.hub.entities),
                'relation_count': len(self.hub.relations)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'export_data': None
            }

    def _execute_health_check(self, inputs: Dict, context) -> Dict[str, Any]:
        """Check health of all integrations."""
        context.update_progress(30, "Running health checks")

        if not self.hub:
            return {
                'success': False,
                'error': 'Knowledge hub not available',
                'healthy': False,
                'integrations': {}
            }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            health_result = loop.run_until_complete(self.hub.health_check())

            context.update_progress(100, "Health check complete")

            # Determine overall health
            initialized = health_result.get('initialized', [])
            overall_healthy = len(initialized) > 0

            return {
                'success': True,
                'healthy': overall_healthy,
                'initialized_integrations': initialized,
                'initialized_count': len(initialized),
                'details': health_result
            }

        except Exception as e:
            return {
                'success': False,
                'healthy': False,
                'error': str(e),
                'initialized_integrations': [],
                'initialized_count': 0
            }

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the result for artifact storage."""
        summary = {'success': result.get('success', False)}

        if 'triple_count' in result:
            summary['triple_count'] = result['triple_count']
        if 'entity_count' in result:
            summary['entity_count'] = result['entity_count']
        if 'relation_count' in result:
            summary['relation_count'] = result['relation_count']
        if 'initialized_count' in result:
            summary['initialized_count'] = result['initialized_count']
        if 'duplicates_removed' in result:
            summary['duplicates_removed'] = result['duplicates_removed']

        return summary

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all operation types and parameters.
        """
        return {
            "type": "object",
            "title": "Knowledge Integration Configuration",
            "description": "Configure knowledge extraction, merging, and export from multiple sources",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The knowledge integration operation to perform",
                    "enum": ["initialize", "extract", "merge", "export", "health_check"],
                    "enumNames": [
                        "Initialize - Setup and initialize the knowledge hub",
                        "Extract - Extract knowledge from text using multiple extractors",
                        "Merge - Merge knowledge from multiple sources",
                        "Export - Export integrated knowledge to various formats",
                        "Health Check - Check health of all integrations"
                    ],
                    "default": "initialize"
                },
                "extractors": {
                    "type": "array",
                    "title": "Extractors",
                    "description": "Knowledge extractors to use for extraction operation",
                    "items": {
                        "type": "string",
                        "enum": ["deepke", "oneke", "kg_gen"]
                    },
                    "default": ["deepke", "oneke", "kg_gen"]
                },
                "sources": {
                    "type": "array",
                    "title": "Knowledge Sources",
                    "description": "List of knowledge sources to merge (for merge operation)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "title": "Knowledge Data",
                                "description": "Knowledge data in the specified format"
                            },
                            "format": {
                                "type": "string",
                                "title": "Format",
                                "enum": ["json", "triples"],
                                "default": "json"
                            }
                        }
                    },
                    "default": []
                },
                "export_format": {
                    "type": "string",
                    "title": "Export Format",
                    "description": "Format for exporting knowledge",
                    "enum": ["json", "triples", "networkx"],
                    "enumNames": [
                        "JSON - Complete knowledge export as JSON",
                        "Triples - Simple triple format",
                        "NetworkX - NetworkX graph format"
                    ],
                    "default": "json"
                },
                "enable_reasoning": {
                    "type": "boolean",
                    "title": "Enable Reasoning",
                    "description": "Enable reasoning systems (verification, causal analysis)",
                    "default": True
                },
                "enable_temporal": {
                    "type": "boolean",
                    "title": "Enable Temporal Tracking",
                    "description": "Enable temporal tracking for knowledge",
                    "default": True
                }
            }
        }

    def is_healthy(self) -> bool:
        """Check if the node is healthy and ready to execute."""
        try:
            return self.UnifiedKGIntegrationHub is not None
        except Exception:
            return False
