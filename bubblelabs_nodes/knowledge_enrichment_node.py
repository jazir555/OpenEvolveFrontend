"""
Knowledge Enrichment Node for BubbleLabs Integration

Enrich entities with additional data from external sources and APIs.

Features:
- Web search for entity information
- Wikidata/DBpedia lookups
- Similar entity discovery
- Property inference from related entities
- Cross-reference validation
- Source tracking with confidence scores
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import asyncio
import json
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeEnrichmentNode(BubbleLabsNode):
    """
    Enrich entities with additional data from external sources and APIs.

    Supports five operations:
    - enrich_entity: Enrich a single entity with external data
    - batch_enrich: Enrich multiple entities in batch
    - find_related: Discover similar/related entities
    - cross_reference: Validate entity data against multiple sources
    - web_lookup: Search web for entity information

    All enrichment operations track sources and provide confidence scores.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Enrichment"
    DESCRIPTION = "Enrich entities with additional data from external sources and APIs"
    ICON = "knowledge-enrichment"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        self.UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available"
        )
        self.KnowledgeGraphModels = self.safe_import(
            'knowledge_engine.graph.kg_models.KnowledgeGraphModels',
            fallback_value=None,
            error_msg="KnowledgeGraphModels not available"
        )
        self.ExternalKnowledgeIntegration = self.safe_import(
            'knowledge_engine.external_knowledge_integration.ExternalKnowledgeIntegration',
            fallback_value=None,
            error_msg="ExternalKnowledgeIntegration not available"
        )

        # Initialize enrichment cache
        self._enrichment_cache: Dict[str, Dict] = {}
        self._api_call_count = 0
        self._max_external_calls = self.config.get('max_external_calls', 10)

        # Track enrichment statistics
        self._enrichment_stats = {
            'successful_enrichments': 0,
            'failed_enrichments': 0,
            'cache_hits': 0,
            'api_calls': 0
        }

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - enrich_entity: entity_id (str)
        - batch_enrich: entity_ids (list)
        - find_related: entity_id (str)
        - cross_reference: entity_id (str) and reference_data
        - web_lookup: entity_id (str) or query (str)
        """
        errors = []

        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'enrich_entity'))

        valid_operations = ['enrich_entity', 'batch_enrich', 'find_related', 'cross_reference', 'web_lookup']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Operation-specific validation
        if operation == 'enrich_entity':
            if 'entity_id' not in inputs and 'entity_id' not in self.config:
                errors.append("Missing required field 'entity_id' for enrich_entity operation")

        elif operation == 'batch_enrich':
            entity_ids = inputs.get('entity_ids', self.config.get('entity_ids', []))
            if not entity_ids:
                errors.append("Missing required field 'entity_ids' for batch_enrich operation")
            elif not isinstance(entity_ids, list):
                errors.append("'entity_ids' must be a list of strings")
            elif len(entity_ids) > 100:
                errors.append("Maximum 100 entities allowed for batch enrichment")

        elif operation == 'find_related':
            if 'entity_id' not in inputs and 'entity_id' not in self.config:
                errors.append("Missing required field 'entity_id' for find_related operation")

        elif operation == 'cross_reference':
            if 'entity_id' not in inputs and 'entity_id' not in self.config:
                errors.append("Missing required field 'entity_id' for cross_reference operation")

        elif operation == 'web_lookup':
            if 'entity_id' not in inputs and 'query' not in inputs:
                if 'entity_id' not in self.config:
                    errors.append("Missing required field 'entity_id' or 'query' for web_lookup operation")

        # Validate enrichment_sources if provided
        if 'enrichment_sources' in inputs:
            valid_sources = ['web', 'wikidata', 'dbpedia', 'custom_api']
            sources = inputs['enrichment_sources']
            if not isinstance(sources, list):
                errors.append("'enrichment_sources' must be a list")
            else:
                for source in sources:
                    if source not in valid_sources:
                        errors.append(f"Invalid enrichment source: {source}. Must be one of: {', '.join(valid_sources)}")

        # Validate similarity_threshold if provided
        if 'similarity_threshold' in inputs:
            try:
                threshold = float(inputs['similarity_threshold'])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("'similarity_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'similarity_threshold' must be a number")

        # Validate properties_to_enrich if provided
        if 'properties_to_enrich' in inputs:
            if not isinstance(inputs['properties_to_enrich'], list):
                errors.append("'properties_to_enrich' must be a list of strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the knowledge enrichment operation based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing enrichment results with sources and confidence scores

        Raises:
            NodeExecutionError: If enrichment fails
        """
        operation = inputs.get('operation', self.config.get('operation', 'enrich_entity'))

        try:
            context.update_progress(10, f"Starting {operation} operation")

            # Execute the appropriate operation
            if operation == 'enrich_entity':
                result = self._execute_enrich_entity(inputs, context)
            elif operation == 'batch_enrich':
                result = self._execute_batch_enrich(inputs, context)
            elif operation == 'find_related':
                result = self._execute_find_related(inputs, context)
            elif operation == 'cross_reference':
                result = self._execute_cross_reference(inputs, context)
            elif operation == 'web_lookup':
                result = self._execute_web_lookup(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['enrich_entity', 'batch_enrich', 'find_related', 'cross_reference', 'web_lookup']}
                )

            context.update_progress(100, f"{operation} operation completed")

            # Add artifact to context
            context.add_artifact('knowledge_enrichment', {
                'operation': operation,
                'success': True,
                'result_summary': self._summarize_result(result)
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge enrichment {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': inputs,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_enrich_entity(self, inputs: Dict, context) -> Dict[str, Any]:
        """Enrich a single entity with external data."""
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        enrichment_sources = inputs.get(
            'enrichment_sources',
            self.config.get('enrichment_sources', ['web', 'wikidata'])
        )
        properties_to_enrich = inputs.get(
            'properties_to_enrich',
            self.config.get('properties_to_enrich', [])
        )
        include_confidence = inputs.get(
            'include_confidence',
            self.config.get('include_confidence', True)
        )

        context.update_progress(20, f"Enriching entity: {entity_id}")

        # Check cache first
        cache_key = f"{entity_id}:{','.join(sorted(enrichment_sources))}"
        if cache_key in self._enrichment_cache:
            self._enrichment_stats['cache_hits'] += 1
            context.update_progress(50, "Returning cached enrichment")
            cached = self._enrichment_cache[cache_key].copy()
            cached['from_cache'] = True
            return cached

        # Collect enrichment data from sources
        enriched_properties = {}
        added_relationships = []
        sources = []
        total_confidence = 0.0
        source_count = 0

        for source in enrichment_sources:
            if self._api_call_count >= self._max_external_calls:
                self.logger.warning(f"Max external calls ({self._max_external_calls}) reached")
                break

            context.update_progress(
                30 + (source_count * 15),
                f"Querying {source} for entity data"
            )

            source_data = self._query_source(entity_id, source, properties_to_enrich)

            if source_data and source_data.get('found'):
                self._api_call_count += 1
                self._enrichment_stats['api_calls'] += 1

                # Merge properties
                for prop, value in source_data.get('properties', {}).items():
                    if prop not in enriched_properties or source_data.get('confidence', 0) > enriched_properties[prop].get('confidence', 0):
                        enriched_properties[prop] = {
                            'value': value,
                            'source': source,
                            'confidence': source_data.get('confidence', 0.5)
                        }

                # Add relationships
                for rel in source_data.get('relationships', []):
                    rel['source'] = source
                    added_relationships.append(rel)

                sources.append({
                    'name': source,
                    'url': source_data.get('url', ''),
                    'retrieved_at': datetime.now().isoformat()
                })

                total_confidence += source_data.get('confidence', 0.5)
                source_count += 1

        context.update_progress(80, "Processing enrichment results")

        # Calculate overall confidence
        overall_confidence = total_confidence / source_count if source_count > 0 else 0.0

        result = {
            'success': source_count > 0,
            'entity_id': entity_id,
            'enriched_properties': enriched_properties,
            'added_relationships': added_relationships,
            'sources': sources,
            'source_count': source_count,
            'confidence': round(overall_confidence, 4) if include_confidence else None,
            'properties_enriched': len(enriched_properties),
            'relationships_added': len(added_relationships),
            'from_cache': False
        }

        # Cache the result
        self._enrichment_cache[cache_key] = result.copy()
        self._enrichment_stats['successful_enrichments'] += 1

        return result

    def _execute_batch_enrich(self, inputs: Dict, context) -> Dict[str, Any]:
        """Enrich multiple entities in batch."""
        entity_ids = inputs.get('entity_ids', self.config.get('entity_ids', []))
        enrichment_sources = inputs.get(
            'enrichment_sources',
            self.config.get('enrichment_sources', ['web', 'wikidata'])
        )

        context.update_progress(10, f"Starting batch enrichment for {len(entity_ids)} entities")

        results = []
        successful = 0
        failed = 0

        for i, entity_id in enumerate(entity_ids):
            progress = 20 + (i / len(entity_ids)) * 60
            context.update_progress(int(progress), f"Enriching entity {i+1}/{len(entity_ids)}: {entity_id}")

            try:
                single_input = {
                    'entity_id': entity_id,
                    'enrichment_sources': enrichment_sources
                }
                result = self._execute_enrich_entity(single_input, context)
                results.append(result)
                if result['success']:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                self.logger.warning(f"Failed to enrich {entity_id}: {e}")
                results.append({
                    'success': False,
                    'entity_id': entity_id,
                    'error': str(e)
                })
                failed += 1

        context.update_progress(90, "Batch enrichment complete")

        return {
            'success': successful > 0,
            'operation': 'batch_enrich',
            'entity_count': len(entity_ids),
            'successful': successful,
            'failed': failed,
            'results': results,
            'enrichment_rate': round(successful / len(entity_ids), 4) if entity_ids else 0.0
        }

    def _execute_find_related(self, inputs: Dict, context) -> Dict[str, Any]:
        """Find similar/related entities."""
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        similarity_threshold = inputs.get(
            'similarity_threshold',
            self.config.get('similarity_threshold', 0.8)
        )
        max_results = inputs.get('max_results', 10)

        context.update_progress(20, f"Finding related entities for: {entity_id}")

        related_entities = []

        # Query external sources for related entities
        for source in ['wikidata', 'dbpedia']:
            if self._api_call_count >= self._max_external_calls:
                break

            context.update_progress(40, f"Querying {source} for related entities")

            source_related = self._find_related_in_source(entity_id, source, similarity_threshold)
            related_entities.extend(source_related)
            self._api_call_count += len(source_related)

        # Sort by similarity and deduplicate
        related_entities.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        seen = set()
        unique_related = []
        for entity in related_entities:
            key = entity.get('id', entity.get('name', ''))
            if key and key not in seen:
                seen.add(key)
                unique_related.append(entity)

        context.update_progress(80, "Processing related entity results")

        # Filter by threshold
        filtered = [e for e in unique_related if e.get('similarity', 0) >= similarity_threshold]

        return {
            'success': len(filtered) > 0,
            'entity_id': entity_id,
            'related_entities': filtered[:max_results],
            'total_found': len(filtered),
            'similarity_threshold': similarity_threshold,
            'sources_queried': ['wikidata', 'dbpedia']
        }

    def _execute_cross_reference(self, inputs: Dict, context) -> Dict[str, Any]:
        """Cross-reference entity data against multiple sources."""
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        reference_data = inputs.get('reference_data', {})

        context.update_progress(20, f"Cross-referencing entity: {entity_id}")

        # Get data from multiple sources
        source_data_map = {}
        conflicts = []
        agreements = []

        sources = ['wikidata', 'dbpedia', 'web']

        for source in sources:
            if self._api_call_count >= self._max_external_calls:
                break

            context.update_progress(30 + (sources.index(source) * 20), f"Querying {source}")

            data = self._query_source(entity_id, source, [])
            if data and data.get('found'):
                source_data_map[source] = data
                self._api_call_count += 1

        context.update_progress(80, "Analyzing cross-reference results")

        # Compare properties across sources
        all_properties = set()
        for data in source_data_map.values():
            all_properties.update(data.get('properties', {}).keys())

        for prop in all_properties:
            values = {}
            for source, data in source_data_map.items():
                if prop in data.get('properties', {}):
                    values[source] = data['properties'][prop]

            if len(values) > 1:
                # Check for conflicts
                unique_values = set(str(v) for v in values.values())
                if len(unique_values) > 1:
                    conflicts.append({
                        'property': prop,
                        'values': values,
                        'type': 'conflict'
                    })
                else:
                    agreements.append({
                        'property': prop,
                        'value': list(unique_values)[0],
                        'sources': list(values.keys()),
                        'agreement_count': len(values)
                    })
            elif len(values) == 1:
                agreements.append({
                    'property': prop,
                    'value': list(values.values())[0],
                    'sources': list(values.keys()),
                    'agreement_count': 1
                })

        # Validate against reference data if provided
        validation_results = []
        if reference_data:
            for prop, expected_value in reference_data.items():
                actual_values = {}
                for source, data in source_data_map.items():
                    if prop in data.get('properties', {}):
                        actual_values[source] = data['properties'][prop]

                if actual_values:
                    matches = any(str(v) == str(expected_value) for v in actual_values.values())
                    validation_results.append({
                        'property': prop,
                        'expected': expected_value,
                        'actual': actual_values,
                        'valid': matches
                    })

        context.update_progress(100, "Cross-reference complete")

        return {
            'success': len(source_data_map) > 0,
            'entity_id': entity_id,
            'sources_queried': list(source_data_map.keys()),
            'source_count': len(source_data_map),
            'agreements': agreements,
            'conflicts': conflicts,
            'agreement_count': len(agreements),
            'conflict_count': len(conflicts),
            'validation_results': validation_results,
            'consistency_score': round(len(agreements) / (len(agreements) + len(conflicts)), 4) if (agreements or conflicts) else 1.0
        }

    def _execute_web_lookup(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search web for entity information."""
        entity_id = inputs.get('entity_id', self.config.get('entity_id'))
        query = inputs.get('query', entity_id)

        context.update_progress(20, f"Performing web lookup for: {query}")

        if self._api_call_count >= self._max_external_calls:
            return {
                'success': False,
                'error': 'Maximum external API calls reached',
                'query': query
            }

        # Perform web search
        web_results = self._perform_web_search(query)
        self._api_call_count += 1
        self._enrichment_stats['api_calls'] += 1

        context.update_progress(60, "Processing web search results")

        # Extract structured information
        extracted_info = self._extract_from_web_results(web_results, entity_id)

        context.update_progress(100, "Web lookup complete")

        return {
            'success': len(web_results) > 0,
            'query': query,
            'entity_id': entity_id,
            'search_results': web_results[:5],
            'extracted_properties': extracted_info.get('properties', {}),
            'extracted_relationships': extracted_info.get('relationships', []),
            'result_count': len(web_results),
            'source': 'web_search',
            'retrieved_at': datetime.now().isoformat()
        }

    def _query_source(self, entity_id: str, source: str, properties: List[str]) -> Optional[Dict]:
        """Query a specific enrichment source for entity data."""
        if source == 'wikidata':
            return self._query_wikidata(entity_id, properties)
        elif source == 'dbpedia':
            return self._query_dbpedia(entity_id, properties)
        elif source == 'web':
            return self._query_web(entity_id, properties)
        elif source == 'custom_api':
            return self._query_custom_api(entity_id, properties)
        return None

    def _query_wikidata(self, entity_id: str, properties: List[str]) -> Optional[Dict]:
        """Query Wikidata for entity information."""
        try:
            # Simulated Wikidata query (in production, use Wikidata API)
            # This is a placeholder that returns mock data for demonstration
            mock_data = {
                'found': True,
                'entity_id': entity_id,
                'properties': {
                    'label': entity_id.replace('_', ' '),
                    'description': f'Entity representing {entity_id}',
                    'instance_of': 'organization' if 'Inc' in entity_id or 'Corp' in entity_id else 'concept'
                },
                'relationships': [
                    {'type': 'instance_of', 'target': 'organization', 'confidence': 0.9}
                ],
                'confidence': 0.85,
                'url': f'https://www.wikidata.org/wiki/{entity_id}'
            }

            # Filter properties if specified
            if properties:
                mock_data['properties'] = {k: v for k, v in mock_data['properties'].items() if k in properties}

            return mock_data

        except Exception as e:
            self.logger.warning(f"Wikidata query failed for {entity_id}: {e}")
            return {'found': False, 'error': str(e)}

    def _query_dbpedia(self, entity_id: str, properties: List[str]) -> Optional[Dict]:
        """Query DBpedia for entity information."""
        try:
            # Simulated DBpedia query
            mock_data = {
                'found': True,
                'entity_id': entity_id,
                'properties': {
                    'name': entity_id.replace('_', ' '),
                    'abstract': f'{entity_id} is a notable entity in the knowledge base.',
                    'type': 'Thing'
                },
                'relationships': [
                    {'type': 'type', 'target': 'Thing', 'confidence': 0.8}
                ],
                'confidence': 0.75,
                'url': f'http://dbpedia.org/resource/{entity_id}'
            }

            if properties:
                mock_data['properties'] = {k: v for k, v in mock_data['properties'].items() if k in properties}

            return mock_data

        except Exception as e:
            self.logger.warning(f"DBpedia query failed for {entity_id}: {e}")
            return {'found': False, 'error': str(e)}

    def _query_web(self, entity_id: str, properties: List[str]) -> Optional[Dict]:
        """Query web sources for entity information."""
        try:
            # Perform web search
            search_results = self._perform_web_search(entity_id)

            if not search_results:
                return {'found': False}

            # Extract information from results
            extracted = self._extract_from_web_results(search_results, entity_id)

            return {
                'found': True,
                'entity_id': entity_id,
                'properties': extracted.get('properties', {}),
                'relationships': extracted.get('relationships', []),
                'confidence': 0.6,
                'url': search_results[0].get('url', '') if search_results else ''
            }

        except Exception as e:
            self.logger.warning(f"Web query failed for {entity_id}: {e}")
            return {'found': False, 'error': str(e)}

    def _query_custom_api(self, entity_id: str, properties: List[str]) -> Optional[Dict]:
        """Query custom API endpoints for entity information."""
        api_endpoints = self.config.get('api_endpoints', [])

        if not api_endpoints:
            return {'found': False, 'error': 'No custom API endpoints configured'}

        combined_properties = {}
        combined_relationships = []
        max_confidence = 0.0

        for api_config in api_endpoints:
            try:
                result = self._call_custom_api(api_config, entity_id, properties)
                if result and result.get('found'):
                    combined_properties.update(result.get('properties', {}))
                    combined_relationships.extend(result.get('relationships', []))
                    max_confidence = max(max_confidence, result.get('confidence', 0.5))
            except Exception as e:
                self.logger.warning(f"Custom API call failed for {entity_id}: {e}")

        if combined_properties or combined_relationships:
            return {
                'found': True,
                'entity_id': entity_id,
                'properties': combined_properties,
                'relationships': combined_relationships,
                'confidence': max_confidence,
                'url': api_endpoints[0].get('url', '') if api_endpoints else ''
            }

        return {'found': False}

    def _perform_web_search(self, query: str) -> List[Dict]:
        """Perform web search for query."""
        # In production, this would use a search API
        # For now, return mock results
        return [
            {
                'title': f'Information about {query}',
                'url': f'https://example.com/info/{query.replace(" ", "-")}',
                'snippet': f'{query} is an important entity with various properties and relationships.',
                'source': 'web'
            },
            {
                'title': f'{query} - Wikipedia',
                'url': f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}',
                'snippet': f'Learn about {query}, its history, and significance.',
                'source': 'wikipedia'
            }
        ]

    def _extract_from_web_results(self, results: List[Dict], entity_id: str) -> Dict[str, Any]:
        """Extract structured information from web search results."""
        properties = {}
        relationships = []

        for result in results:
            snippet = result.get('snippet', '')
            title = result.get('title', '')

            # Simple property extraction patterns
            if 'is a' in snippet:
                parts = snippet.split('is a')
                if len(parts) > 1:
                    properties['type'] = parts[1].strip().split('.')[0].split(',')[0]

            if 'founded' in snippet.lower():
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', snippet)
                if year_match:
                    properties['founded_year'] = year_match.group()

            # Add source as relationship
            relationships.append({
                'type': 'mentioned_in',
                'target': result.get('source', 'web'),
                'confidence': 0.5,
                'url': result.get('url', '')
            })

        return {
            'properties': properties,
            'relationships': relationships
        }

    def _find_related_in_source(self, entity_id: str, source: str, threshold: float) -> List[Dict]:
        """Find related entities in a specific source."""
        related = []

        try:
            if source == 'wikidata':
                # Mock related entities from Wikidata
                related = [
                    {'id': f'{entity_id}_related_1', 'name': f'Related to {entity_id}', 'similarity': 0.85, 'source': 'wikidata'},
                    {'id': f'{entity_id}_related_2', 'name': f'Similar to {entity_id}', 'similarity': 0.75, 'source': 'wikidata'}
                ]
            elif source == 'dbpedia':
                # Mock related entities from DBpedia
                related = [
                    {'id': f'{entity_id}_dbp_related_1', 'name': f'Associated with {entity_id}', 'similarity': 0.80, 'source': 'dbpedia'}
                ]

        except Exception as e:
            self.logger.warning(f"Finding related entities failed for {source}: {e}")

        return [r for r in related if r.get('similarity', 0) >= threshold]

    def _call_custom_api(self, api_config: Dict, entity_id: str, properties: List[str]) -> Optional[Dict]:
        """Call a custom API endpoint."""
        try:
            # In production, this would make actual HTTP requests
            # For now, return mock response
            return {
                'found': True,
                'entity_id': entity_id,
                'properties': api_config.get('default_properties', {}),
                'relationships': [],
                'confidence': 0.7,
                'url': api_config.get('url', '')
            }
        except Exception as e:
            self.logger.warning(f"Custom API call failed: {e}")
            return {'found': False, 'error': str(e)}

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the result for artifact storage."""
        summary = {'success': result.get('success', False)}

        if 'entity_id' in result:
            summary['entity_id'] = result['entity_id']
        if 'entity_count' in result:
            summary['entity_count'] = result['entity_count']
        if 'source_count' in result:
            summary['source_count'] = result['source_count']
        if 'confidence' in result and result['confidence'] is not None:
            summary['confidence'] = result['confidence']
        if 'properties_enriched' in result:
            summary['properties_enriched'] = result['properties_enriched']
        if 'relationships_added' in result:
            summary['relationships_added'] = result['relationships_added']
        if 'total_found' in result:
            summary['related_entities_found'] = result['total_found']
        if 'agreement_count' in result:
            summary['agreements'] = result['agreement_count']
        if 'conflict_count' in result:
            summary['conflicts'] = result['conflict_count']

        return summary

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration with all operation types and parameters.
        """
        return {
            "type": "object",
            "title": "Knowledge Enrichment Configuration",
            "description": "Configure knowledge enrichment with external data sources",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The enrichment operation to perform",
                    "enum": ["enrich_entity", "batch_enrich", "find_related", "cross_reference", "web_lookup"],
                    "enumNames": [
                        "Enrich Entity - Enrich a single entity with external data",
                        "Batch Enrich - Enrich multiple entities in batch",
                        "Find Related - Discover similar/related entities",
                        "Cross Reference - Validate entity data against multiple sources",
                        "Web Lookup - Search web for entity information"
                    ],
                    "default": "enrich_entity"
                },
                "entity_id": {
                    "type": "string",
                    "title": "Entity ID",
                    "description": "The entity to enrich (for single-entity operations)",
                    "default": ""
                },
                "entity_ids": {
                    "type": "array",
                    "title": "Entity IDs",
                    "description": "List of entities to enrich (for batch operations)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "enrichment_sources": {
                    "type": "array",
                    "title": "Enrichment Sources",
                    "description": "External sources to query for enrichment data",
                    "items": {
                        "type": "string",
                        "enum": ["web", "wikidata", "dbpedia", "custom_api"]
                    },
                    "default": ["web", "wikidata"]
                },
                "properties_to_enrich": {
                    "type": "array",
                    "title": "Properties to Enrich",
                    "description": "Specific properties to find (empty = all available)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "similarity_threshold": {
                    "type": "number",
                    "title": "Similarity Threshold",
                    "description": "Minimum similarity score for related entities (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8
                },
                "max_external_calls": {
                    "type": "integer",
                    "title": "Max External Calls",
                    "description": "Maximum number of external API calls per execution",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "api_endpoints": {
                    "type": "array",
                    "title": "Custom API Endpoints",
                    "description": "Custom API configurations for enrichment",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "title": "API Name"
                            },
                            "url": {
                                "type": "string",
                                "title": "API URL"
                            },
                            "auth_token": {
                                "type": "string",
                                "title": "Authentication Token"
                            },
                            "default_properties": {
                                "type": "object",
                                "title": "Default Properties"
                            }
                        },
                        "required": ["name", "url"]
                    },
                    "default": []
                },
                "include_confidence": {
                    "type": "boolean",
                    "title": "Include Confidence",
                    "description": "Include confidence scores in enrichment results",
                    "default": True
                }
            },
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["enrich_entity"]}
                            },
                            "required": ["entity_id"],
                            "description": "Enrich a single entity with external data"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["batch_enrich"]}
                            },
                            "required": ["entity_ids"],
                            "description": "Enrich multiple entities in batch"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["find_related"]}
                            },
                            "required": ["entity_id"],
                            "description": "Discover similar/related entities"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["cross_reference"]}
                            },
                            "required": ["entity_id"],
                            "description": "Validate entity data against multiple sources"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["web_lookup"]}
                            },
                            "required": ["entity_id"],
                            "description": "Search web for entity information"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least fallback enrichment is available
        """
        try:
            # Node can operate even without external dependencies
            # (has fallback/mock implementations)
            return True
        except Exception:
            return False

    def get_enrichment_stats(self) -> Dict[str, Any]:
        """
        Get enrichment statistics.

        Returns:
            Dict containing enrichment statistics
        """
        return {
            **self._enrichment_stats,
            'cache_size': len(self._enrichment_cache),
            'api_calls_remaining': max(0, self._max_external_calls - self._api_call_count)
        }

    def clear_cache(self):
        """Clear the enrichment cache."""
        self._enrichment_cache.clear()
        self.logger.info("Enrichment cache cleared")
