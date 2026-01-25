"""
Knowledge Extraction Node for BubbleLabs Integration

Extracts and packages knowledge artifacts from workflow execution.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeExtractionNode(BubbleLabsNode):
    """
    Extracts knowledge artifacts and learns from workflow execution.

    Extracts:
    - Patterns and best practices
    - Lessons learned
    - Performance metrics
    - Knowledge artifacts for reuse
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Extraction"
    DESCRIPTION = (
        "Extract knowledge artifacts, patterns, and lessons learned "
        "from workflow execution for future reuse."
    )
    ICON = "knowledge"
    CATEGORY = "learning"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import knowledge extractor (safe import)
        WorkflowKnowledgeExtractor = self.safe_import(
            'workflow_knowledge_extractor.WorkflowKnowledgeExtractor',
            fallback_value=None,
            error_msg="WorkflowKnowledgeExtractor not available for KnowledgeExtractionNode"
        )

        if WorkflowKnowledgeExtractor:
            try:
                self.extractor = WorkflowKnowledgeExtractor()
            except Exception as e:
                self.logger.warning(f"Could not instantiate WorkflowKnowledgeExtractor: {e}")
                self.extractor = None
        else:
            self.extractor = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - workflow_state: WorkflowState object or dict

        Optional:
            - extraction_types: List[str]
            - artifact_format: str
            - store_in_kb: bool
        """
        errors = []

        # Check required fields
        if 'workflow_state' not in inputs:
            errors.append("Missing required field: workflow_state")
        elif not isinstance(inputs['workflow_state'], (dict, object)):
            errors.append("workflow_state must be a WorkflowState or dictionary")

        # Validate extraction_types
        if 'extraction_types' in inputs:
            if not isinstance(inputs['extraction_types'], list):
                errors.append("extraction_types must be a list")
            else:
                valid_types = ['patterns', 'lessons_learned', 'best_practices', 'metrics', 'artifacts']
                for et in inputs['extraction_types']:
                    if et not in valid_types:
                        errors.append(f"Invalid extraction type: {et}. Must be one of {valid_types}")

        # Validate artifact_format
        if 'artifact_format' in inputs:
            valid_formats = ['structured', 'unstructured', 'both']
            if inputs['artifact_format'] not in valid_formats:
                errors.append(f"artifact_format must be one of: {', '.join(valid_formats)}")

        # Validate store_in_kb
        if 'store_in_kb' in inputs:
            if not isinstance(inputs['store_in_kb'], bool):
                errors.append("store_in_kb must be a boolean")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Extract knowledge from workflow execution.

        Args:
            inputs: Must contain 'workflow_state' and optional extraction parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - artifacts: List of knowledge artifacts
                - patterns_found: List of discovered patterns
                - lessons_learned: List of lessons learned
                - metrics_summary: Summary of workflow metrics
                - knowledge_base_links: Links to stored knowledge
        """
        if not self.extractor:
            return self._extract_simple(inputs, context)

        workflow_state = inputs['workflow_state']
        extraction_types = inputs.get('extraction_types', self.config.get('extraction_types', [
            'patterns',
            'lessons_learned',
            'best_practices',
            'metrics'
        ]))
        artifact_format = inputs.get('artifact_format', self.config.get('artifact_format', 'structured'))
        store_in_kb = inputs.get('store_in_kb', self.config.get('store_in_kb', True))

        # Update progress
        context.update_progress(10, "Initializing knowledge extractor")
        self.logger.info(f"Extracting knowledge types: {', '.join(extraction_types)}")

        try:
            # Extract knowledge
            context.update_progress(20, "Analyzing workflow state")

            extraction_result = self.extractor.extract(
                workflow_state=workflow_state,
                extraction_types=extraction_types,
                artifact_format=artifact_format,
                store_in_kb=store_in_kb,
                callback=lambda p, m: context.update_progress(20 + p * 0.7, m)
            )

            # Update progress
            context.update_progress(90, "Processing extracted knowledge")

            # Extract and format results
            result = {
                'artifacts': self._format_artifacts(extraction_result.artifacts),
                'patterns_found': extraction_result.patterns,
                'lessons_learned': extraction_result.lessons_learned,
                'best_practices': extraction_result.best_practices,
                'metrics_summary': extraction_result.metrics_summary,
                'knowledge_base_links': extraction_result.kb_links if store_in_kb else [],
                'extraction_metadata': {
                    'types_processed': extraction_types,
                    'format': artifact_format,
                    'stored_in_kb': store_in_kb,
                    'extraction_time': extraction_result.extraction_time,
                    'artifacts_count': len(extraction_result.artifacts)
                }
            }

            # Add artifacts to context
            context.add_artifact('knowledge_extraction', {
                'result': result,
                'extraction_types': extraction_types,
                'timestamp': context.generate_execution_id() if hasattr(context, 'generate_execution_id') else None
            })

            context.update_progress(
                100,
                f"Knowledge extraction complete: {result['extraction_metadata']['artifacts_count']} artifacts, "
                f"{len(result['patterns_found'])} patterns, "
                f"{len(result['lessons_learned'])} lessons"
            )

            self.logger.info(
                f"Knowledge extraction completed: {result['extraction_metadata']['artifacts_count']} artifacts, "
                f"{len(result['patterns_found'])} patterns found"
            )

            return result

        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge extraction failed: {str(e)}",
                details={
                    'extraction_types': extraction_types,
                    'artifact_format': artifact_format,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _extract_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple extraction fallback when extractor not available"""
        workflow_state = inputs['workflow_state']
        extraction_types = inputs.get('extraction_types', ['patterns', 'metrics'])

        context.update_progress(10, "Using simple extraction (extractor not available)")

        # Convert workflow state to dict if it's an object
        if hasattr(workflow_state, '__dict__'):
            state_dict = workflow_state.__dict__
        elif isinstance(workflow_state, dict):
            state_dict = workflow_state
        else:
            state_dict = {'state': str(workflow_state)}

        context.update_progress(30, "Analyzing workflow state")

        # Simple pattern extraction
        patterns = []
        lessons = []
        metrics = {}

        # Extract basic metrics
        if 'progress' in state_dict:
            metrics['progress'] = state_dict['progress']
        if 'artifacts' in state_dict:
            metrics['artifacts_count'] = len(state_dict['artifacts'])

        # Extract patterns from artifacts
        if 'artifacts' in state_dict:
            for artifact_name, artifact_data in state_dict['artifacts'].items():
                patterns.append({
                    'type': 'artifact_pattern',
                    'name': artifact_name,
                    'description': f"Artifact created: {artifact_name}"
                })

        # Generate basic lessons
        if metrics.get('artifacts_count', 0) > 0:
            lessons.append("Workflow produced multiple knowledge artifacts")

        result = {
            'artifacts': [
                {
                    'type': 'simple_extraction',
                    'data': state_dict,
                    'metadata': {'note': 'Simple extraction performed'}
                }
            ],
            'patterns_found': patterns,
            'lessons_learned': lessons,
            'best_practices': [],
            'metrics_summary': metrics,
            'knowledge_base_links': [],
            'extraction_metadata': {
                'types_processed': extraction_types,
                'format': 'structured',
                'stored_in_kb': False,
                'extraction_time': 0.1,
                'artifacts_count': 1,
                'warning': 'Full extractor not available, using simple extraction'
            }
        }

        context.update_progress(100, "Simple extraction complete")
        return result

    def _format_artifacts(self, artifacts: List) -> List[Dict[str, Any]]:
        """Format artifacts for output"""
        formatted = []

        for artifact in artifacts:
            formatted.append({
                'id': getattr(artifact, 'id', 'unknown'),
                'type': getattr(artifact, 'type', 'general'),
                'title': getattr(artifact, 'title', 'Untitled'),
                'description': getattr(artifact, 'description', ''),
                'data': getattr(artifact, 'data', {}),
                'metadata': getattr(artifact, 'metadata', {}),
                'created_at': getattr(artifact, 'created_at', None)
            })

        return formatted

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Knowledge Extraction Configuration",
            "description": "Configure knowledge extraction and learning parameters",
            "properties": {
                "extraction_types": {
                    "type": "array",
                    "title": "Extraction Types",
                    "description": "Types of knowledge to extract",
                    "items": {
                        "type": "string",
                        "enum": ["patterns", "lessons_learned", "best_practices", "metrics", "artifacts"]
                    },
                    "uniqueItems": True,
                    "default": ["patterns", "lessons_learned", "best_practices", "metrics"]
                },
                "artifact_format": {
                    "type": "string",
                    "title": "Artifact Format",
                    "description": "Format for extracted artifacts",
                    "enum": ["structured", "unstructured", "both"],
                    "enumNames": [
                        "Structured (JSON)",
                        "Unstructured (Text)",
                        "Both Formats"
                    ],
                    "default": "structured"
                },
                "store_in_kb": {
                    "type": "boolean",
                    "title": "Store in Knowledge Base",
                    "description": "Store extracted artifacts in knowledge base for future reference",
                    "default": True
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "description": "Minimum confidence threshold for pattern extraction (0-1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                }
            },
            "required": ["extraction_types"]
        }
