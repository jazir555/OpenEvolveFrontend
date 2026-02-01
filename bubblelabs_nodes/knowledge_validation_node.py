"""
Knowledge Validation Node for BubbleLabs Integration

Validates knowledge quality, completeness, and schema compliance.

Features:
- Schema validation (type checking, required fields, value ranges)
- Completeness checks (missing properties, null values, empty strings)
- Quality assessment (confidence scores, source attribution, evidence)
- Reference validation (broken links, dangling references)
- Format compliance (data types, string patterns, date formats)
- Comprehensive validation reports with scores
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import re
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeValidationNode(BubbleLabsNode):
    """
    Validates knowledge quality, completeness, and schema compliance.

    Supports multiple validation types:
    - Schema: Validate against defined schemas (type checking, required fields)
    - Completeness: Check for missing properties, null values, empty strings
    - Quality: Assess confidence scores, source attribution, evidence quality
    - References: Validate links and references to other entities
    - Format: Check data types, string patterns, date formats
    - Comprehensive: Run all validation types and aggregate results
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Validation"
    DESCRIPTION = "Validate knowledge quality, completeness, and schema compliance"
    ICON = "knowledge-validation"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    # Validation types
    VALIDATION_TYPES = ["schema", "completeness", "quality", "references", "format", "comprehensive"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="Knowledge Engine (unified_kg_integration_hub) not available"
        )

        quality_assurance_module = self.safe_import(
            'knowledge_engine.quality_assurance',
            fallback_value=None,
            error_msg="Quality Assurance module not available"
        )

        # Store module references
        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None
        self.QualityAssurance = None

        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)

        if quality_assurance_module:
            self.QualityAssurance = getattr(quality_assurance_module, 'QualityAssurance', None)

        # Initialize hub instance
        self.hub = None
        if self.UnifiedKGIntegrationHub and self.UnifiedKGConfig:
            try:
                config_obj = self.UnifiedKGConfig(
                    enable_deepke=True,
                    enable_oneke=True,
                    enable_kg_gen=True
                )
                self.hub = self.UnifiedKGIntegrationHub(config=config_obj)
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (one of):
            - knowledge_graph_id: str - ID of knowledge graph to validate
            - entities: List[Dict] - List of entities to validate directly

        Optional:
            - validation_type: str - Override configured validation type
            - schema_id: str - Schema to validate against
            - required_properties: List[str] - Properties that must exist
            - quality_threshold: float - Minimum quality score (0.0-1.0)
        """
        errors = []

        # Check required fields - must have either knowledge_graph_id or entities
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_entities = 'entities' in inputs and isinstance(inputs['entities'], list) and len(inputs['entities']) > 0

        if not has_kg_id and not has_entities:
            errors.append("Must provide either 'knowledge_graph_id' or 'entities'")

        # Validate knowledge_graph_id if provided
        if 'knowledge_graph_id' in inputs:
            if not isinstance(inputs['knowledge_graph_id'], str):
                errors.append("'knowledge_graph_id' must be a string")
            elif len(inputs['knowledge_graph_id'].strip()) == 0:
                errors.append("'knowledge_graph_id' cannot be empty")

        # Validate entities if provided
        if 'entities' in inputs:
            if not isinstance(inputs['entities'], list):
                errors.append("'entities' must be a list")
            elif len(inputs['entities']) == 0:
                errors.append("'entities' list cannot be empty")
            else:
                for i, entity in enumerate(inputs['entities']):
                    if not isinstance(entity, dict):
                        errors.append(f"Entity at index {i} must be a dictionary")
                        break

        # Validate validation_type override if provided
        if 'validation_type' in inputs:
            if inputs['validation_type'] not in self.VALIDATION_TYPES:
                errors.append(
                    f"Invalid validation_type: '{inputs['validation_type']}'. "
                    f"Must be one of: {', '.join(self.VALIDATION_TYPES)}"
                )

        # Validate quality_threshold override if provided
        if 'quality_threshold' in inputs:
            try:
                threshold = float(inputs['quality_threshold'])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("'quality_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'quality_threshold' must be a number")

        # Validate required_properties if provided
        if 'required_properties' in inputs:
            if not isinstance(inputs['required_properties'], list):
                errors.append("'required_properties' must be a list of strings")
            else:
                for prop in inputs['required_properties']:
                    if not isinstance(prop, str):
                        errors.append("All items in 'required_properties' must be strings")
                        break

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be a list of strings")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge validation.

        Args:
            inputs: Contains knowledge_graph_id or entities, plus validation parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - valid: Whether validation passed
                - score: Overall quality score (0.0-1.0)
                - errors: List of validation errors
                - warnings: List of validation warnings
                - validation_type: Type of validation performed
                - report: Detailed validation report

        Raises:
            NodeExecutionError: If validation fails unexpectedly
        """
        # Get configuration
        validation_type = inputs.get('validation_type', self.config.get('validation_type', 'comprehensive'))
        schema_id = inputs.get('schema_id', self.config.get('schema_id', None))
        required_properties = inputs.get(
            'required_properties',
            self.config.get('required_properties', [])
        )
        quality_threshold = inputs.get(
            'quality_threshold',
            self.config.get('quality_threshold', 0.7)
        )
        check_references = inputs.get(
            'check_references',
            self.config.get('check_references', True)
        )
        strict_mode = inputs.get(
            'strict_mode',
            self.config.get('strict_mode', False)
        )
        entity_types = inputs.get(
            'entity_types',
            self.config.get('entity_types', [])
        )

        context.update_progress(10, f"Initializing {validation_type} validation")
        self.logger.info(f"Starting knowledge validation: type={validation_type}")

        try:
            # Retrieve entities to validate
            context.update_progress(20, "Retrieving entities for validation")
            entities = self._get_entities(inputs)

            if entity_types:
                entities = [e for e in entities if e.get('type') in entity_types]

            if not entities:
                return {
                    'valid': True,
                    'score': 1.0,
                    'errors': [],
                    'warnings': ["No entities to validate"],
                    'validation_type': validation_type,
                    'report': {
                        'entities_checked': 0,
                        'checks_performed': [],
                        'timestamp': datetime.now().isoformat()
                    }
                }

            context.update_progress(30, f"Validating {len(entities)} entities")

            # Run validation based on type
            if validation_type == 'comprehensive':
                result = self._run_comprehensive_validation(
                    entities=entities,
                    schema_id=schema_id,
                    required_properties=required_properties,
                    quality_threshold=quality_threshold,
                    check_references=check_references,
                    strict_mode=strict_mode,
                    context=context
                )
            elif validation_type == 'schema':
                result = self._validate_schema(entities, schema_id, context)
            elif validation_type == 'completeness':
                result = self._validate_completeness(entities, required_properties, context)
            elif validation_type == 'quality':
                result = self._validate_quality(entities, quality_threshold, context)
            elif validation_type == 'references':
                result = self._validate_references(entities, context)
            elif validation_type == 'format':
                result = self._validate_format(entities, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown validation type: {validation_type}",
                    details={'valid_types': self.VALIDATION_TYPES}
                )

            # Add metadata
            result['validation_type'] = validation_type
            result['report']['timestamp'] = datetime.now().isoformat()
            result['report']['entities_checked'] = len(entities)
            result['report']['configuration'] = {
                'schema_id': schema_id,
                'required_properties': required_properties,
                'quality_threshold': quality_threshold,
                'check_references': check_references,
                'strict_mode': strict_mode,
                'entity_types': entity_types
            }

            # Determine overall validity
            if strict_mode:
                result['valid'] = len(result['errors']) == 0 and len(result['warnings']) == 0
            else:
                result['valid'] = len(result['errors']) == 0 and result['score'] >= quality_threshold

            # Store artifact
            context.add_artifact('knowledge_validation', {
                'valid': result['valid'],
                'score': result['score'],
                'error_count': len(result['errors']),
                'warning_count': len(result['warnings']),
                'entities_validated': len(entities)
            })

            status_msg = "PASSED" if result['valid'] else "FAILED"
            context.update_progress(
                100,
                f"Validation {status_msg}: score={result['score']:.2f}, "
                f"errors={len(result['errors'])}, warnings={len(result['warnings'])}"
            )

            self.logger.info(
                f"Knowledge validation completed: {status_msg}, "
                f"score={result['score']:.2f}"
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge validation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge validation failed: {str(e)}",
                details={
                    'validation_type': validation_type,
                    'inputs': {k: v for k, v in inputs.items() if k != 'entities'},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _get_entities(self, inputs: Dict) -> List[Dict[str, Any]]:
        """Retrieve entities from inputs or knowledge graph."""
        if 'entities' in inputs and inputs['entities']:
            return inputs['entities']

        if 'knowledge_graph_id' in inputs and self.hub:
            try:
                # Try to get entities from hub
                kg_id = inputs['knowledge_graph_id']
                # This is a placeholder - actual implementation depends on hub API
                if hasattr(self.hub, 'get_entities'):
                    return self.hub.get_entities(kg_id)
                elif hasattr(self.hub, 'entities'):
                    return [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.hub.entities]
            except Exception as e:
                self.logger.warning(f"Could not retrieve entities from hub: {e}")

        return []

    def _run_comprehensive_validation(
        self,
        entities: List[Dict],
        schema_id: Optional[str],
        required_properties: List[str],
        quality_threshold: float,
        check_references: bool,
        strict_mode: bool,
        context
    ) -> Dict[str, Any]:
        """Run all validation types and aggregate results."""
        context.update_progress(35, "Running schema validation")
        schema_result = self._validate_schema(entities, schema_id, context)

        context.update_progress(50, "Running completeness validation")
        completeness_result = self._validate_completeness(entities, required_properties, context)

        context.update_progress(65, "Running quality validation")
        quality_result = self._validate_quality(entities, quality_threshold, context)

        context.update_progress(80, "Running reference validation")
        references_result = self._validate_references(entities, context) if check_references else {
            'errors': [], 'warnings': [], 'score': 1.0, 'report': {'checks': []}
        }

        context.update_progress(90, "Running format validation")
        format_result = self._validate_format(entities, context)

        # Aggregate results
        all_errors = (
            schema_result['errors'] +
            completeness_result['errors'] +
            quality_result['errors'] +
            references_result['errors'] +
            format_result['errors']
        )

        all_warnings = (
            schema_result['warnings'] +
            completeness_result['warnings'] +
            quality_result['warnings'] +
            references_result['warnings'] +
            format_result['warnings']
        )

        # Calculate overall score (weighted average)
        scores = [
            schema_result['score'] * 0.25,
            completeness_result['score'] * 0.25,
            quality_result['score'] * 0.30,
            references_result['score'] * 0.10,
            format_result['score'] * 0.10
        ]
        overall_score = sum(scores)

        return {
            'valid': False,  # Will be determined by caller
            'score': round(overall_score, 4),
            'errors': all_errors,
            'warnings': all_warnings,
            'report': {
                'checks_performed': ['schema', 'completeness', 'quality', 'references', 'format'],
                'schema_validation': schema_result['report'],
                'completeness_validation': completeness_result['report'],
                'quality_validation': quality_result['report'],
                'references_validation': references_result['report'],
                'format_validation': format_result['report'],
                'summary': {
                    'total_errors': len(all_errors),
                    'total_warnings': len(all_warnings),
                    'schema_score': schema_result['score'],
                    'completeness_score': completeness_result['score'],
                    'quality_score': quality_result['score'],
                    'references_score': references_result['score'],
                    'format_score': format_result['score']
                }
            }
        }

    def _validate_schema(
        self,
        entities: List[Dict],
        schema_id: Optional[str],
        context
    ) -> Dict[str, Any]:
        """Validate entities against schema."""
        errors = []
        warnings = []
        checks = []

        # Load schema if specified
        schema = self._load_schema(schema_id) if schema_id else self._get_default_schema()

        for entity in entities:
            entity_id = entity.get('id', 'unknown')
            entity_type = entity.get('type', 'unknown')

            # Check required fields based on type
            type_schema = schema.get(entity_type, schema.get('default', {}))
            required = type_schema.get('required', ['id', 'type'])

            for field in required:
                if field not in entity:
                    errors.append({
                        'entity_id': entity_id,
                        'field': field,
                        'message': f"Missing required field: {field}",
                        'severity': 'error'
                    })

            # Check field types
            properties = type_schema.get('properties', {})
            for field, value in entity.items():
                if field in properties:
                    expected_type = properties[field].get('type')
                    if expected_type and not self._check_type(value, expected_type):
                        errors.append({
                            'entity_id': entity_id,
                            'field': field,
                            'message': f"Expected type {expected_type}, got {type(value).__name__}",
                            'severity': 'error'
                        })

            # Check value ranges if specified
            for field, prop_def in properties.items():
                if field in entity:
                    value = entity[field]
                    if 'minimum' in prop_def and value < prop_def['minimum']:
                        errors.append({
                            'entity_id': entity_id,
                            'field': field,
                            'message': f"Value {value} below minimum {prop_def['minimum']}",
                            'severity': 'error'
                        })
                    if 'maximum' in prop_def and value > prop_def['maximum']:
                        errors.append({
                            'entity_id': entity_id,
                            'field': field,
                            'message': f"Value {value} above maximum {prop_def['maximum']}",
                            'severity': 'error'
                        })

        checks.append({
            'check': 'schema_compliance',
            'entities_checked': len(entities),
            'errors_found': len(errors),
            'warnings_found': len(warnings)
        })

        score = max(0.0, 1.0 - (len(errors) / max(len(entities), 1)) * 0.5)

        return {
            'errors': errors,
            'warnings': warnings,
            'score': round(score, 4),
            'report': {'checks': checks, 'schema_used': schema_id or 'default'}
        }

    def _validate_completeness(
        self,
        entities: List[Dict],
        required_properties: List[str],
        context
    ) -> Dict[str, Any]:
        """Validate completeness of entities."""
        errors = []
        warnings = []
        checks = []

        # Default required properties if none specified
        if not required_properties:
            required_properties = ['id', 'type', 'name', 'description']

        for entity in entities:
            entity_id = entity.get('id', 'unknown')

            # Check for missing properties
            for prop in required_properties:
                if prop not in entity:
                    errors.append({
                        'entity_id': entity_id,
                        'property': prop,
                        'message': f"Missing required property: {prop}",
                        'severity': 'error'
                    })

            # Check for null or empty values
            for key, value in entity.items():
                if value is None:
                    warnings.append({
                        'entity_id': entity_id,
                        'property': key,
                        'message': f"Property '{key}' has null value",
                        'severity': 'warning'
                    })
                elif isinstance(value, str) and not value.strip():
                    warnings.append({
                        'entity_id': entity_id,
                        'property': key,
                        'message': f"Property '{key}' has empty string value",
                        'severity': 'warning'
                    })
                elif isinstance(value, (list, dict)) and len(value) == 0:
                    warnings.append({
                        'entity_id': entity_id,
                        'property': key,
                        'message': f"Property '{key}' has empty value",
                        'severity': 'warning'
                    })

            # Check for properties that look like they should have values
            completeness_indicators = ['source', 'confidence', 'evidence', 'author', 'timestamp']
            for indicator in completeness_indicators:
                if indicator not in entity:
                    warnings.append({
                        'entity_id': entity_id,
                        'property': indicator,
                        'message': f"Missing recommended property: {indicator}",
                        'severity': 'warning'
                    })

        checks.append({
            'check': 'completeness',
            'entities_checked': len(entities),
            'required_properties': required_properties,
            'errors_found': len(errors),
            'warnings_found': len(warnings)
        })

        score = max(0.0, 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1) / max(len(entities), 1))

        return {
            'errors': errors,
            'warnings': warnings,
            'score': round(score, 4),
            'report': {'checks': checks}
        }

    def _validate_quality(
        self,
        entities: List[Dict],
        quality_threshold: float,
        context
    ) -> Dict[str, Any]:
        """Validate quality of entities."""
        errors = []
        warnings = []
        checks = []

        quality_scores = []

        for entity in entities:
            entity_id = entity.get('id', 'unknown')
            entity_quality = 1.0

            # Check confidence score
            confidence = entity.get('confidence')
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    if conf_val < quality_threshold:
                        warnings.append({
                            'entity_id': entity_id,
                            'message': f"Confidence score {conf_val:.2f} below threshold {quality_threshold}",
                            'severity': 'warning'
                        })
                        entity_quality -= 0.2
                except (TypeError, ValueError):
                    warnings.append({
                        'entity_id': entity_id,
                        'message': f"Invalid confidence value: {confidence}",
                        'severity': 'warning'
                    })
                    entity_quality -= 0.1
            else:
                warnings.append({
                    'entity_id': entity_id,
                    'message': "Missing confidence score",
                    'severity': 'warning'
                })
                entity_quality -= 0.15

            # Check source attribution
            source = entity.get('source')
            if not source:
                warnings.append({
                    'entity_id': entity_id,
                    'message': "Missing source attribution",
                    'severity': 'warning'
                })
                entity_quality -= 0.1

            # Check evidence
            evidence = entity.get('evidence')
            if not evidence:
                warnings.append({
                    'entity_id': entity_id,
                    'message': "Missing evidence/supporting data",
                    'severity': 'warning'
                })
                entity_quality -= 0.1

            # Check timestamp
            timestamp = entity.get('timestamp')
            if not timestamp:
                warnings.append({
                    'entity_id': entity_id,
                    'message': "Missing timestamp",
                    'severity': 'warning'
                })
                entity_quality -= 0.05

            # Check for sufficient description
            description = entity.get('description', '')
            if isinstance(description, str):
                if len(description) < 10:
                    warnings.append({
                        'entity_id': entity_id,
                        'message': "Description is too short (minimum 10 characters)",
                        'severity': 'warning'
                    })
                    entity_quality -= 0.1

            quality_scores.append(max(0.0, entity_quality))

        # Calculate overall quality score
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        checks.append({
            'check': 'quality',
            'entities_checked': len(entities),
            'average_quality_score': overall_score,
            'quality_threshold': quality_threshold,
            'errors_found': len(errors),
            'warnings_found': len(warnings)
        })

        return {
            'errors': errors,
            'warnings': warnings,
            'score': round(overall_score, 4),
            'report': {'checks': checks}
        }

    def _validate_references(
        self,
        entities: List[Dict],
        context
    ) -> Dict[str, Any]:
        """Validate references and links between entities."""
        errors = []
        warnings = []
        checks = []

        # Build set of valid entity IDs
        valid_ids: Set[str] = set()
        for entity in entities:
            entity_id = entity.get('id')
            if entity_id:
                valid_ids.add(entity_id)

        for entity in entities:
            entity_id = entity.get('id', 'unknown')

            # Check relations/references
            relations = entity.get('relations', entity.get('edges', []))
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, dict):
                        target_id = relation.get('target') or relation.get('object') or relation.get('to')
                        if target_id and target_id not in valid_ids:
                            errors.append({
                                'entity_id': entity_id,
                                'reference': target_id,
                                'message': f"Dangling reference to non-existent entity: {target_id}",
                                'severity': 'error'
                            })

            # Check external links
            links = entity.get('links', entity.get('urls', []))
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, str):
                        # Basic URL validation
                        if not link.startswith(('http://', 'https://', 'ftp://', 'file://')):
                            warnings.append({
                                'entity_id': entity_id,
                                'link': link,
                                'message': f"Potentially invalid URL format: {link}",
                                'severity': 'warning'
                            })

            # Check parent/child references
            parent_id = entity.get('parent_id')
            if parent_id and parent_id not in valid_ids:
                warnings.append({
                    'entity_id': entity_id,
                    'reference': parent_id,
                    'message': f"Parent reference to non-existent entity: {parent_id}",
                    'severity': 'warning'
                })

        checks.append({
            'check': 'references',
            'entities_checked': len(entities),
            'valid_references': len(valid_ids),
            'errors_found': len(errors),
            'warnings_found': len(warnings)
        })

        score = max(0.0, 1.0 - len(errors) * 0.2)

        return {
            'errors': errors,
            'warnings': warnings,
            'score': round(score, 4),
            'report': {'checks': checks}
        }

    def _validate_format(
        self,
        entities: List[Dict],
        context
    ) -> Dict[str, Any]:
        """Validate format compliance of entity data."""
        errors = []
        warnings = []
        checks = []

        # Common patterns
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        date_iso_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
        email_pattern = re.compile(r'^[^@]+@[^@]+\.[^@]+$')

        for entity in entities:
            entity_id = entity.get('id', 'unknown')

            for key, value in entity.items():
                # Check ID format
                if key == 'id' and isinstance(value, str):
                    if not uuid_pattern.match(value) and not value.replace('_', '').replace('-', '').isalnum():
                        warnings.append({
                            'entity_id': entity_id,
                            'field': key,
                            'message': f"ID '{value}' may not follow recommended format",
                            'severity': 'warning'
                        })

                # Check date formats
                if 'date' in key.lower() or 'time' in key.lower() or 'timestamp' in key.lower():
                    if isinstance(value, str) and not date_iso_pattern.match(value):
                        warnings.append({
                            'entity_id': entity_id,
                            'field': key,
                            'message': f"Date field '{key}' should use ISO 8601 format",
                            'severity': 'warning'
                        })

                # Check email formats
                if 'email' in key.lower() and isinstance(value, str):
                    if not email_pattern.match(value):
                        warnings.append({
                            'entity_id': entity_id,
                            'field': key,
                            'message': f"Email '{value}' appears to be invalid",
                            'severity': 'warning'
                        })

                # Check numeric ranges
                if isinstance(value, (int, float)):
                    if value < 0 and 'count' in key.lower():
                        errors.append({
                            'entity_id': entity_id,
                            'field': key,
                            'message': f"Count field '{key}' cannot be negative",
                            'severity': 'error'
                        })

        checks.append({
            'check': 'format',
            'entities_checked': len(entities),
            'errors_found': len(errors),
            'warnings_found': len(warnings)
        })

        score = max(0.0, 1.0 - (len(errors) * 0.3 + len(warnings) * 0.05) / max(len(entities), 1))

        return {
            'errors': errors,
            'warnings': warnings,
            'score': round(score, 4),
            'report': {'checks': checks}
        }

    def _load_schema(self, schema_id: str) -> Dict[str, Any]:
        """Load schema definition by ID."""
        # Placeholder - would load from schema registry
        self.logger.info(f"Loading schema: {schema_id}")
        return self._get_default_schema()

    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default schema definition."""
        return {
            'default': {
                'required': ['id', 'type'],
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'timestamp': {'type': 'string'},
                    'source': {'type': 'string'},
                    'evidence': {'type': 'array'}
                }
            },
            'entity': {
                'required': ['id', 'type', 'name'],
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'name': {'type': 'string'},
                    'description': {'type': 'string'}
                }
            },
            'relation': {
                'required': ['id', 'type', 'source', 'target'],
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'source': {'type': 'string'},
                    'target': {'type': 'string'},
                    'relation_type': {'type': 'string'}
                }
            }
        }

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }

        expected = type_mapping.get(expected_type)
        if expected is None:
            return True  # Unknown type, allow

        return isinstance(value, expected)

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Knowledge Validation Configuration",
            "description": "Configure knowledge validation parameters",
            "properties": {
                "validation_type": {
                    "type": "string",
                    "title": "Validation Type",
                    "description": "Type of validation to perform",
                    "enum": ["schema", "completeness", "quality", "references", "format", "comprehensive"],
                    "enumNames": [
                        "Schema - Validate against defined schemas",
                        "Completeness - Check for missing properties",
                        "Quality - Assess confidence and source quality",
                        "References - Validate links and references",
                        "Format - Check data type and format compliance",
                        "Comprehensive - Run all validation types"
                    ],
                    "default": "comprehensive"
                },
                "schema_id": {
                    "type": "string",
                    "title": "Schema ID",
                    "description": "Schema to validate against (optional)",
                    "default": ""
                },
                "required_properties": {
                    "type": "array",
                    "title": "Required Properties",
                    "description": "Properties that must exist in each entity",
                    "items": {
                        "type": "string"
                    },
                    "default": ["id", "type", "name"]
                },
                "quality_threshold": {
                    "type": "number",
                    "title": "Quality Threshold",
                    "description": "Minimum quality score for validation to pass (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                },
                "check_references": {
                    "type": "boolean",
                    "title": "Check References",
                    "description": "Validate links and references between entities",
                    "default": True
                },
                "strict_mode": {
                    "type": "boolean",
                    "title": "Strict Mode",
                    "description": "Fail validation on warnings in addition to errors",
                    "default": False
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Limit validation to specific entity types (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (can run validation without external dependencies)
        """
        try:
            # Node can work without external dependencies (has internal validation logic)
            return True
        except Exception:
            return False

    def get_supported_validation_types(self) -> List[str]:
        """
        Get list of supported validation types.

        Returns:
            List of validation type names
        """
        return self.VALIDATION_TYPES.copy()
