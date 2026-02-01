"""
Entity Profile Node for BubbleLabs Integration

Creates and manages rich entity profiles with relationships and provenance.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .base_node import BubbleLabsNode, NodeExecutionError


class EntityProfileNode(BubbleLabsNode):
    """
    Create and manage rich entity profiles with relationships and provenance.

    Features:
    - Create rich entity profiles
    - Add relationships and properties to entities
    - Merge entity information from multiple sources
    - Query entity profiles
    """

    # Node metadata
    DISPLAY_NAME = "Entity Profile"
    DESCRIPTION = "Create and manage rich entity profiles with relationships and provenance"
    ICON = "entity-profile"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import models using safe_import pattern
        self.KnowledgeGraphModels = self.safe_import(
            'knowledge_engine.graph.kg_models.KnowledgeGraphModels',
            fallback_value=None,
            error_msg="KnowledgeGraphModels not available"
        )
        self.EntityProfile = self.safe_import(
            'knowledge_engine.graph.kg_models.EntityProfile',
            fallback_value=None,
            error_msg="EntityProfile not available"
        )
        self.KnowledgeSource = self.safe_import(
            'knowledge_engine.graph.kg_models.KnowledgeSource',
            fallback_value=None,
            error_msg="KnowledgeSource not available"
        )

        # Initialize models instance if available
        if self.KnowledgeGraphModels:
            try:
                self.kg_models = self.KnowledgeGraphModels()
            except Exception as e:
                self.logger.warning(f"Could not instantiate KnowledgeGraphModels: {e}")
                self.kg_models = None
        else:
            self.kg_models = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - operation: One of ["create", "update", "get", "merge"]
            - entity_name: Name of the entity

        Optional (depending on operation):
            - entity_types: List of entity types
            - properties: Dictionary of key-value properties
            - relationships: List of relationship objects
            - confidence: Confidence level (0.0-1.0)
            - source: Source of the information
            - merge_profiles: List of profiles to merge (for merge operation)
        """
        errors = []

        # Check required fields
        if 'operation' not in inputs:
            errors.append("Missing required field: operation")
        else:
            valid_operations = ['create', 'update', 'get', 'merge']
            if inputs['operation'] not in valid_operations:
                errors.append(f"Invalid operation: {inputs['operation']}. Must be one of {valid_operations}")

        if 'entity_name' not in inputs:
            errors.append("Missing required field: entity_name")
        elif not isinstance(inputs['entity_name'], str) or not inputs['entity_name'].strip():
            errors.append("entity_name must be a non-empty string")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("entity_types must be a list of strings")
            else:
                for et in inputs['entity_types']:
                    if not isinstance(et, str):
                        errors.append(f"All entity_types must be strings, got: {type(et)}")
                        break

        # Validate properties if provided
        if 'properties' in inputs:
            if not isinstance(inputs['properties'], dict):
                errors.append("properties must be a dictionary")

        # Validate relationships if provided
        if 'relationships' in inputs:
            if not isinstance(inputs['relationships'], list):
                errors.append("relationships must be a list of objects")
            else:
                for idx, rel in enumerate(inputs['relationships']):
                    if not isinstance(rel, dict):
                        errors.append(f"Relationship at index {idx} must be an object")
                        continue
                    if 'predicate' not in rel:
                        errors.append(f"Relationship at index {idx} missing required field: predicate")
                    if 'target' not in rel:
                        errors.append(f"Relationship at index {idx} missing required field: target")

        # Validate confidence if provided
        if 'confidence' in inputs:
            try:
                conf = float(inputs['confidence'])
                if not 0.0 <= conf <= 1.0:
                    errors.append("confidence must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("confidence must be a number between 0.0 and 1.0")

        # Validate merge_profiles if merge operation
        if inputs.get('operation') == 'merge':
            if 'merge_profiles' not in inputs:
                errors.append("Missing required field for merge operation: merge_profiles")
            elif not isinstance(inputs['merge_profiles'], list):
                errors.append("merge_profiles must be a list of entity profiles")
            elif len(inputs['merge_profiles']) < 2:
                errors.append("merge_profiles must contain at least 2 profiles to merge")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the entity profile operation.

        Args:
            inputs: Operation parameters including:
                - operation: Type of operation
                - entity_name: Name of the entity
                - entity_types: List of types
                - properties: Key-value properties
                - relationships: Relationships to add
                - confidence: Confidence level
                - source: Information source
            context: Workflow state for tracking

        Returns:
            Dict containing operation result and entity profile data
        """
        operation = inputs['operation']
        entity_name = inputs['entity_name']
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        properties = inputs.get('properties', self.config.get('properties', {}))
        relationships = inputs.get('relationships', self.config.get('relationships', []))
        confidence = float(inputs.get('confidence', self.config.get('confidence', 1.0)))
        source = inputs.get('source', self.config.get('source', 'bubblelabs'))

        context.update_progress(10, f"Starting {operation} operation for entity: {entity_name}")
        self.logger.info(f"Executing {operation} on entity: {entity_name}")

        try:
            if operation == 'create':
                result = self._create_profile(
                    entity_name, entity_types, properties, relationships, confidence, source, context
                )
            elif operation == 'update':
                result = self._update_profile(
                    entity_name, entity_types, properties, relationships, confidence, source, context
                )
            elif operation == 'get':
                result = self._get_profile(entity_name, context)
            elif operation == 'merge':
                merge_profiles = inputs.get('merge_profiles', [])
                result = self._merge_profiles(entity_name, merge_profiles, confidence, source, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['create', 'update', 'get', 'merge']}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed successfully")
            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Entity profile operation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Entity profile operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'entity_name': entity_name,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _create_profile(
        self,
        name: str,
        types: List[str],
        properties: Dict[str, Any],
        relationships: List[Dict],
        confidence: float,
        source: str,
        context
    ) -> Dict[str, Any]:
        """Create a new entity profile."""
        context.update_progress(30, "Creating entity profile")

        if self.kg_models:
            # Check if profile already exists
            existing = self.kg_models.get_entity_profile(name)
            if existing:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Entity profile already exists: {name}",
                    details={'entity_name': name, 'suggestion': 'Use update operation to modify existing profile'}
                )

            # Create profile using KnowledgeGraphModels
            profile = self.kg_models.create_entity_profile(
                name=name,
                types=types,
                properties=properties
            )

            # Add relationships
            for rel in relationships:
                profile.add_relationship(
                    predicate=rel['predicate'],
                    target=rel['target'],
                    confidence=rel.get('confidence', confidence),
                    source=rel.get('source', source)
                )

            # Set confidence scores
            profile.confidence_scores['overall'] = confidence
            profile.sources.add(source)
            profile.update_timestamp()

            result = {
                'success': True,
                'operation': 'create',
                'profile': profile.to_dict(),
                'metadata': {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'relationships_added': len(relationships)
                }
            }
        else:
            # Fallback when models not available
            result = {
                'success': True,
                'operation': 'create',
                'profile': {
                    'id': f"fallback_{name}_{datetime.now(timezone.utc).timestamp()}",
                    'name': name,
                    'types': types,
                    'properties': properties,
                    'relationships': relationships,
                    'sources': [source],
                    'confidence_scores': {'overall': confidence},
                    'first_seen': datetime.now(timezone.utc).isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'metadata': {}
                },
                'metadata': {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'fallback': True,
                    'warning': 'KnowledgeGraphModels not available, using fallback implementation'
                }
            }

        context.update_progress(80, "Entity profile created")
        return result

    def _update_profile(
        self,
        name: str,
        types: List[str],
        properties: Dict[str, Any],
        relationships: List[Dict],
        confidence: float,
        source: str,
        context
    ) -> Dict[str, Any]:
        """Update an existing entity profile."""
        context.update_progress(30, "Updating entity profile")

        if self.kg_models:
            profile = self.kg_models.get_entity_profile(name)
            if not profile:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Entity profile not found: {name}",
                    details={'entity_name': name, 'suggestion': 'Use create operation to create new profile'}
                )

            # Update types if provided
            if types:
                profile.types = list(set(profile.types + types))

            # Update properties
            profile.properties.update(properties)

            # Add new relationships
            for rel in relationships:
                profile.add_relationship(
                    predicate=rel['predicate'],
                    target=rel['target'],
                    confidence=rel.get('confidence', confidence),
                    source=rel.get('source', source)
                )

            profile.confidence_scores['overall'] = confidence
            profile.sources.add(source)
            profile.update_timestamp()

            result = {
                'success': True,
                'operation': 'update',
                'profile': profile.to_dict(),
                'metadata': {
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'relationships_added': len(relationships)
                }
            }
        else:
            # Fallback
            result = {
                'success': True,
                'operation': 'update',
                'profile': {
                    'name': name,
                    'types': types,
                    'properties': properties,
                    'relationships': relationships,
                    'sources': [source],
                    'confidence_scores': {'overall': confidence},
                    'last_updated': datetime.now(timezone.utc).isoformat()
                },
                'metadata': {
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'fallback': True,
                    'warning': 'KnowledgeGraphModels not available, using fallback implementation'
                }
            }

        context.update_progress(80, "Entity profile updated")
        return result

    def _get_profile(self, name: str, context) -> Dict[str, Any]:
        """Retrieve an entity profile."""
        context.update_progress(30, "Retrieving entity profile")

        if self.kg_models:
            profile = self.kg_models.get_entity_profile(name)
            if not profile:
                return {
                    'success': False,
                    'operation': 'get',
                    'profile': None,
                    'error': f"Entity profile not found: {name}",
                    'metadata': {
                        'queried_at': datetime.now(timezone.utc).isoformat(),
                        'entity_name': name
                    }
                }

            result = {
                'success': True,
                'operation': 'get',
                'profile': profile.to_dict(),
                'metadata': {
                    'queried_at': datetime.now(timezone.utc).isoformat(),
                    'entity_name': name,
                    'profile_id': profile.id
                }
            }
        else:
            # Fallback - return empty profile structure
            result = {
                'success': False,
                'operation': 'get',
                'profile': None,
                'error': f"KnowledgeGraphModels not available, cannot retrieve profile: {name}",
                'metadata': {
                    'queried_at': datetime.now(timezone.utc).isoformat(),
                    'entity_name': name,
                    'fallback': True
                }
            }

        context.update_progress(80, "Entity profile retrieved")
        return result

    def _merge_profiles(
        self,
        name: str,
        merge_profiles: List[Dict],
        confidence: float,
        source: str,
        context
    ) -> Dict[str, Any]:
        """Merge multiple entity profiles into one."""
        context.update_progress(30, "Merging entity profiles")

        # Initialize merged data
        merged_types = []
        merged_properties = {}
        merged_relationships = []
        merged_sources = set()
        merged_confidence_scores = {}

        # Process each profile to merge
        for idx, profile_data in enumerate(merge_profiles):
            context.update_progress(30 + (idx * 40 // len(merge_profiles)), f"Processing profile {idx + 1}/{len(merge_profiles)}")

            # Handle both EntityProfile objects and dictionaries
            if isinstance(profile_data, dict):
                profile_dict = profile_data
            elif hasattr(profile_data, 'to_dict'):
                profile_dict = profile_data.to_dict()
            else:
                profile_dict = {'name': str(profile_data)}

            # Merge types
            types = profile_dict.get('types', [])
            merged_types.extend(types)

            # Merge properties
            properties = profile_dict.get('properties', {})
            merged_properties.update(properties)

            # Merge relationships
            relationships = profile_dict.get('relationships', [])
            merged_relationships.extend(relationships)

            # Merge sources
            sources = profile_dict.get('sources', [])
            merged_sources.update(sources)

            # Merge confidence scores
            confidence_scores = profile_dict.get('confidence_scores', {})
            merged_confidence_scores.update(confidence_scores)

        # Remove duplicates from types
        merged_types = list(set(merged_types))

        # Add source
        merged_sources.add(source)

        # Set overall confidence
        merged_confidence_scores['overall'] = confidence
        merged_confidence_scores['merge_count'] = len(merge_profiles)

        if self.kg_models:
            # Check if profile exists, update if so, create if not
            existing = self.kg_models.get_entity_profile(name)
            if existing:
                profile = self.kg_models.update_entity_profile(name, {
                    'types': list(set(existing.types + merged_types)),
                    'properties': {**existing.properties, **merged_properties},
                    'relationships': existing.relationships + merged_relationships,
                    'sources': existing.sources.union(merged_sources),
                    'confidence_scores': {**existing.confidence_scores, **merged_confidence_scores}
                })
                operation = 'merge_update'
            else:
                profile = self.kg_models.create_entity_profile(
                    name=name,
                    types=merged_types,
                    properties=merged_properties
                )
                profile.relationships = merged_relationships
                profile.sources = merged_sources
                profile.confidence_scores = merged_confidence_scores
                profile.update_timestamp()
                operation = 'merge_create'

            result = {
                'success': True,
                'operation': 'merge',
                'profile': profile.to_dict(),
                'metadata': {
                    'merged_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'profiles_merged': len(merge_profiles),
                    'merge_operation': operation,
                    'types_count': len(merged_types),
                    'properties_count': len(merged_properties),
                    'relationships_count': len(merged_relationships)
                }
            }
        else:
            # Fallback
            result = {
                'success': True,
                'operation': 'merge',
                'profile': {
                    'id': f"merged_{name}_{datetime.now(timezone.utc).timestamp()}",
                    'name': name,
                    'types': merged_types,
                    'properties': merged_properties,
                    'relationships': merged_relationships,
                    'sources': list(merged_sources),
                    'confidence_scores': merged_confidence_scores,
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'metadata': {'merge_operation': True}
                },
                'metadata': {
                    'merged_at': datetime.now(timezone.utc).isoformat(),
                    'source': source,
                    'confidence': confidence,
                    'profiles_merged': len(merge_profiles),
                    'types_count': len(merged_types),
                    'properties_count': len(merged_properties),
                    'relationships_count': len(merged_relationships),
                    'fallback': True,
                    'warning': 'KnowledgeGraphModels not available, using fallback implementation'
                }
            }

        context.update_progress(80, "Entity profiles merged")
        return result

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns:
            JSON schema dictionary for UI configuration
        """
        return {
            "type": "object",
            "title": "Entity Profile Configuration",
            "description": "Configure entity profile operations",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Type of entity profile operation to perform",
                    "enum": ["create", "update", "get", "merge"],
                    "enumNames": [
                        "Create - Create a new entity profile",
                        "Update - Update an existing profile",
                        "Get - Retrieve an entity profile",
                        "Merge - Merge multiple profiles"
                    ],
                    "default": "create"
                },
                "entity_name": {
                    "type": "string",
                    "title": "Entity Name",
                    "description": "Name of the entity (required for all operations)"
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Types or categories for the entity (e.g., Person, Researcher, Organization)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "properties": {
                    "type": "object",
                    "title": "Properties",
                    "description": "Key-value properties for the entity",
                    "additionalProperties": True,
                    "default": {}
                },
                "relationships": {
                    "type": "array",
                    "title": "Relationships",
                    "description": "Relationships to add to the entity profile",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predicate": {
                                "type": "string",
                                "title": "Predicate",
                                "description": "Relationship type (e.g., works_for, located_in, related_to)"
                            },
                            "target": {
                                "type": "string",
                                "title": "Target",
                                "description": "Target entity name or ID"
                            },
                            "confidence": {
                                "type": "number",
                                "title": "Confidence",
                                "description": "Confidence level for this specific relationship",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 1.0
                            },
                            "source": {
                                "type": "string",
                                "title": "Source",
                                "description": "Source of this relationship information"
                            }
                        },
                        "required": ["predicate", "target"]
                    },
                    "default": []
                },
                "confidence": {
                    "type": "number",
                    "title": "Confidence Level",
                    "description": "Overall confidence level for the entity information (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0
                },
                "source": {
                    "type": "string",
                    "title": "Source",
                    "description": "Source of the entity information",
                    "default": "bubblelabs"
                }
            },
            "required": ["operation", "entity_name"]
        }
