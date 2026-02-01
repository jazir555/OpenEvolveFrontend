"""
Knowledge Extraction Node for BubbleLabs Integration

Extracts structured knowledge (triples) from unstructured text using multiple
NLP extraction strategies including DeepKE, OneKE, and KG-Gen.

Features:
- Multiple extraction backends (DeepKE, OneKE, KG-Gen)
- Automatic extractor selection (auto mode)
- Confidence threshold filtering
- Domain-aware extraction hints
- Structured output with provenance metadata
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeExtractionNode(BubbleLabsNode):
    """
    Extract structured knowledge from unstructured text using multiple NLP strategies.

    Supports:
    - DeepKE: Deep learning-based relation extraction
    - OneKE: One-stop knowledge extraction with schema guidance
    - KG-Gen: Knowledge graph generation
    - Auto: Automatic selection of best extractor

    Output includes structured triples (subject-predicate-object) with
    confidence scores and provenance information.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Extraction"
    DESCRIPTION = "Extract structured knowledge from unstructured text using multiple NLP strategies"
    ICON = "knowledge-extraction"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of UnifiedKGIntegrationHub
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="Knowledge Engine (unified_kg_integration_hub) not available"
        )

        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None
        self.KGSource = None
        self.KnowledgeTriple = None

        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)
            self.KGSource = getattr(unified_hub_module, 'KGSource', None)
            self.KnowledgeTriple = getattr(unified_hub_module, 'KnowledgeTriple', None)

        # Initialize hub instance
        self.hub = None
        if self.UnifiedKGIntegrationHub and self.UnifiedKGConfig:
            try:
                config_obj = self._create_hub_config()
                self.hub = self.UnifiedKGIntegrationHub(config=config_obj)
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Track available extractors
        self.available_extractors = ['deepke', 'oneke', 'kg_gen', 'auto']

    def _create_hub_config(self):
        """Create UnifiedKGConfig based on node configuration."""
        extractor = self.config.get('extractor', 'auto')

        # Base configuration
        config_kwargs = {
            'enable_deepke': extractor in ['deepke', 'auto'],
            'enable_oneke': extractor in ['oneke', 'auto'],
            'enable_kg_gen': extractor in ['kg_gen', 'auto'],
            'enable_ai_kg': False,
            'enable_unified_extraction': extractor == 'auto',
        }

        return self.UnifiedKGConfig(**config_kwargs)

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - text: str - The text to extract knowledge from

        Optional:
            - extractor: str - Override the configured extractor
            - min_confidence: float - Override the configured confidence threshold
            - domain: str - Domain hint for extraction
        """
        errors = []

        # Check required fields
        if 'text' not in inputs:
            errors.append("Missing required field: 'text'")
        elif not isinstance(inputs['text'], str):
            errors.append("'text' must be a string")
        elif len(inputs['text'].strip()) == 0:
            errors.append("'text' cannot be empty")
        elif len(inputs['text']) > 100000:
            errors.append("'text' exceeds maximum length of 100,000 characters")

        # Validate extractor override if provided
        if 'extractor' in inputs:
            if inputs['extractor'] not in self.available_extractors:
                errors.append(
                    f"Invalid extractor: '{inputs['extractor']}'. "
                    f"Must be one of: {', '.join(self.available_extractors)}"
                )

        # Validate min_confidence override if provided
        if 'min_confidence' in inputs:
            try:
                conf = float(inputs['min_confidence'])
                if not 0.0 <= conf <= 1.0:
                    errors.append("'min_confidence' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'min_confidence' must be a number")

        # Validate domain if provided
        if 'domain' in inputs:
            if not isinstance(inputs['domain'], str):
                errors.append("'domain' must be a string")
            elif len(inputs['domain']) > 100:
                errors.append("'domain' exceeds maximum length of 100 characters")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Extract knowledge from input text.

        Args:
            inputs: Must contain 'text' and optional extraction parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - triples: List of extracted knowledge triples
                - entities: List of extracted entities
                - confidence: Overall extraction confidence score
                - metadata: Extraction provenance and statistics

        Raises:
            NodeExecutionError: If extraction fails
        """
        text = inputs['text']
        extractor = inputs.get('extractor', self.config.get('extractor', 'auto'))
        min_confidence = inputs.get(
            'min_confidence',
            self.config.get('min_confidence', 0.7)
        )
        include_metadata = inputs.get(
            'include_metadata',
            self.config.get('include_metadata', True)
        )
        domain = inputs.get('domain', self.config.get('domain', None))

        context.update_progress(10, f"Initializing knowledge extraction with {extractor} extractor")
        self.logger.info(f"Extracting knowledge using extractor: {extractor}")

        try:
            # Use hub if available, otherwise use fallback
            if self.hub:
                result = self._extract_with_hub(
                    text=text,
                    extractor=extractor,
                    min_confidence=min_confidence,
                    domain=domain,
                    context=context
                )
            else:
                result = self._extract_fallback(
                    text=text,
                    min_confidence=min_confidence,
                    context=context
                )

            # Add provenance metadata if requested
            if include_metadata:
                result['metadata'] = {
                    'extractor_used': extractor,
                    'min_confidence_threshold': min_confidence,
                    'domain_hint': domain,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'text_length': len(text),
                    'hub_available': self.hub is not None
                }

            # Store artifacts in context
            context.add_artifact('knowledge_extraction', {
                'triples_count': len(result.get('triples', [])),
                'entities_count': len(result.get('entities', [])),
                'extractor': extractor,
                'confidence': result.get('confidence', 0.0)
            })

            context.update_progress(
                100,
                f"Extraction complete: {len(result.get('triples', []))} triples, "
                f"{len(result.get('entities', []))} entities"
            )

            self.logger.info(
                f"Knowledge extraction completed: {len(result.get('triples', []))} triples, "
                f"confidence={result.get('confidence', 0.0):.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge extraction failed: {str(e)}",
                details={
                    'extractor': extractor,
                    'text_length': len(text),
                    'domain': domain,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _extract_with_hub(
        self,
        text: str,
        extractor: str,
        min_confidence: float,
        domain: Optional[str],
        context
    ) -> Dict[str, Any]:
        """Extract knowledge using UnifiedKGIntegrationHub."""
        context.update_progress(20, "Configuring extraction pipeline")

        # Determine which extractors to use
        if extractor == 'auto':
            extractors = ['deepke', 'oneke', 'kg_gen']
        else:
            extractors = [extractor]

        context.update_progress(30, f"Running extraction with {', '.join(extractors)}")

        # Run async extraction
        try:
            triples = asyncio.run(self.hub.extract_knowledge(
                text=text,
                extractors=extractors,
                merge_results=True
            ))
        except Exception as e:
            self.logger.warning(f"Async extraction failed: {e}, trying sync fallback")
            triples = []

        # Filter by confidence
        filtered_triples = [
            t for t in triples
            if t.confidence >= min_confidence
        ]

        context.update_progress(70, f"Filtered to {len(filtered_triples)} high-confidence triples")

        # Convert triples to dictionary format
        triples_list = []
        entities_set = set()

        for triple in filtered_triples:
            triple_dict = {
                'subject': triple.subject,
                'predicate': triple.predicate,
                'object': triple.object,
                'confidence': triple.confidence,
                'provenance': {
                    'source': triple.source.value if hasattr(triple.source, 'value') else str(triple.source),
                    'timestamp': triple.timestamp.isoformat() if hasattr(triple.timestamp, 'isoformat') else str(triple.timestamp)
                }
            }

            if triple.metadata:
                triple_dict['metadata'] = triple.metadata

            triples_list.append(triple_dict)

            # Extract entities
            entities_set.add(triple.subject)
            entities_set.add(triple.object)

        context.update_progress(90, "Formatting extraction results")

        # Build entities list
        entities_list = [
            {
                'name': entity,
                'mentions': 1,
                'types': []  # Could be enhanced with entity typing
            }
            for entity in sorted(entities_set)
        ]

        # Calculate overall confidence
        overall_confidence = (
            sum(t['confidence'] for t in triples_list) / len(triples_list)
            if triples_list else 0.0
        )

        return {
            'triples': triples_list,
            'entities': entities_list,
            'confidence': round(overall_confidence, 4),
            'statistics': {
                'total_triples': len(triples),
                'filtered_triples': len(filtered_triples),
                'unique_entities': len(entities_list),
                'extractors_used': extractors
            }
        }

    def _extract_fallback(
        self,
        text: str,
        min_confidence: float,
        context
    ) -> Dict[str, Any]:
        """Fallback extraction when UnifiedKGIntegrationHub is not available."""
        context.update_progress(20, "Using fallback extraction (hub not available)")

        # Simple pattern-based extraction as fallback
        triples = []
        entities = set()

        # Basic entity extraction (capitalized phrases)
        import re

        # Simple regex for capitalized phrases (potential entities)
        entity_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        found_entities = set(re.findall(entity_pattern, text))

        # Simple pattern: "X is Y" -> (X, is, Y)
        is_pattern = r'([A-Z][a-zA-Z]+(?:\s+[a-z]+){0,3})\s+is\s+(?:a|an|the)?\s*([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){0,5})'
        for match in re.finditer(is_pattern, text):
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            triples.append({
                'subject': subject,
                'predicate': 'is',
                'object': obj,
                'confidence': 0.5,
                'provenance': {
                    'source': 'fallback_pattern',
                    'timestamp': datetime.now().isoformat()
                }
            })
            entities.add(subject)
            entities.add(obj)

        # Filter by confidence
        filtered_triples = [t for t in triples if t['confidence'] >= min_confidence]

        context.update_progress(90, "Fallback extraction complete")

        entities_list = [
            {'name': e, 'mentions': 1, 'types': []}
            for e in sorted(found_entities | entities)
        ]

        overall_confidence = (
            sum(t['confidence'] for t in filtered_triples) / len(filtered_triples)
            if filtered_triples else 0.0
        )

        return {
            'triples': filtered_triples,
            'entities': entities_list,
            'confidence': round(overall_confidence, 4),
            'statistics': {
                'total_triples': len(triples),
                'filtered_triples': len(filtered_triples),
                'unique_entities': len(entities_list),
                'extractors_used': ['fallback_pattern'],
                'warning': 'UnifiedKGIntegrationHub not available, using fallback extraction'
            }
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Knowledge Extraction Configuration",
            "description": "Configure knowledge extraction from unstructured text",
            "properties": {
                "extractor": {
                    "type": "string",
                    "title": "Extractor",
                    "description": "Which extraction engine to use",
                    "enum": ["deepke", "oneke", "kg_gen", "auto"],
                    "enumNames": [
                        "DeepKE - Deep learning relation extraction",
                        "OneKE - Schema-guided knowledge extraction",
                        "KG-Gen - Knowledge graph generation",
                        "Auto - Automatic selection of best extractor"
                    ],
                    "default": "auto"
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "description": "Minimum confidence threshold for extracted triples (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                },
                "include_metadata": {
                    "type": "boolean",
                    "title": "Include Metadata",
                    "description": "Include extraction metadata and provenance information",
                    "default": True
                },
                "domain": {
                    "type": "string",
                    "title": "Domain Hint",
                    "description": "Optional domain hint to improve extraction (e.g., 'medical', 'finance', 'legal')",
                    "default": ""
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least fallback extraction is available
        """
        # Node can work with or without the hub (has fallback)
        return True

    def get_available_extractors(self) -> List[str]:
        """
        Get list of available extraction backends.

        Returns:
            List of extractor names that are currently available
        """
        available = []

        if self.hub:
            available = ['deepke', 'oneke', 'kg_gen', 'auto']
        else:
            available = ['auto (fallback)']

        return available
