"""
Output Node for BubbleLabs Integration

Implements SOP (Standard Operating Procedure) generation and output formatting.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class OutputNode(BubbleLabsNode):
    """
    Generates formatted output and Standard Operating Procedures (SOPs).

    Supports multiple output formats:
    - Markdown
    - HTML
    - JSON
    - Plain text
    """

    # Node metadata
    DISPLAY_NAME = "Output & SOP Generation"
    DESCRIPTION = (
        "Generate formatted outputs and Standard Operating Procedures "
        "from solution data."
    )
    ICON = "output"
    CATEGORY = "output"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import SOP generator (safe import)
        SOPGenerator = self.safe_import(
            'sop_generator.SOPGenerator',
            fallback_value=None,
            error_msg="SOPGenerator not available for OutputNode"
        )

        if SOPGenerator:
            try:
                self.generator = SOPGenerator()
            except Exception as e:
                self.logger.warning(f"Could not instantiate SOPGenerator: {e}")
                self.generator = None
        else:
            self.generator = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - solution: Dict containing solution data

        Optional:
            - output_format: str (markdown, html, json, text)
            - template: str
            - include_sections: List[str]
            - language: str
        """
        errors = []

        # Check required fields
        if 'solution' not in inputs:
            errors.append("Missing required field: solution")
        elif not isinstance(inputs['solution'], dict):
            errors.append("solution must be a dictionary")

        # Validate output_format
        if 'output_format' in inputs:
            valid_formats = ['markdown', 'html', 'json', 'text']
            if inputs['output_format'] not in valid_formats:
                errors.append(f"output_format must be one of: {', '.join(valid_formats)}")

        # Validate include_sections
        if 'include_sections' in inputs:
            if not isinstance(inputs['include_sections'], list):
                errors.append("include_sections must be a list")
            elif not all(isinstance(s, str) for s in inputs['include_sections']):
                errors.append("All items in include_sections must be strings")

        # Validate language
        if 'language' in inputs:
            if not isinstance(inputs['language'], str) or len(inputs['language']) != 2:
                errors.append("language must be a 2-letter ISO code (e.g., 'en')")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Generate formatted output from solution data.

        Args:
            inputs: Must contain 'solution' and optional formatting parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - output: Formatted output string
                - metadata: Output metadata
                - word_count: Number of words in output
                - sections: List of section information
                - preview: Preview of output (first 500 chars)
        """
        if not self.generator:
            # Fallback to simple formatting if generator not available
            return self._format_output_simple(inputs, context)

        solution = inputs['solution']
        output_format = inputs.get('output_format', self.config.get('output_format', 'markdown'))
        template = inputs.get('template', self.config.get('template', 'standard'))
        include_sections = inputs.get('include_sections', self.config.get('include_sections', [
            'executive_summary',
            'detailed_steps',
            'references'
        ]))
        language = inputs.get('language', self.config.get('language', 'en'))

        # Update progress
        context.update_progress(10, "Initializing output generator")
        self.logger.info(f"Generating {output_format.upper()} output using template: {template}")

        try:
            # Generate output
            context.update_progress(20, "Processing solution data")

            output_result = self.generator.generate(
                solution=solution,
                output_format=output_format,
                template=template,
                sections=include_sections,
                language=language,
                callback=lambda p, m: context.update_progress(20 + p * 0.7, m)
            )

            # Update progress
            context.update_progress(90, "Finalizing output format")

            # Extract and format results
            result = {
                'output': output_result.content,
                'metadata': {
                    'format': output_format,
                    'template': template,
                    'language': language,
                    'generated_at': output_result.timestamp,
                    'generator_version': output_result.version
                },
                'word_count': output_result.word_count,
                'sections': output_result.sections,
                'preview': self._generate_preview(output_result.content),
                'size_bytes': len(output_result.content.encode('utf-8'))
            }

            # Add artifacts to context
            context.add_artifact('output', {
                'result': result,
                'solution_id': solution.get('id', 'unknown'),
                'format': output_format
            })

            context.update_progress(100, f"Output generation complete: {result['word_count']} words")

            self.logger.info(
                f"Output generated: {output_format} format, "
                f"{result['word_count']} words, "
                f"{len(result['sections'])} sections"
            )

            return result

        except Exception as e:
            self.logger.error(f"Output generation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Output generation failed: {str(e)}",
                details={
                    'solution_id': solution.get('id', 'unknown'),
                    'output_format': output_format,
                    'template': template,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _format_output_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple formatting fallback when generator not available"""
        solution = inputs['solution']
        output_format = inputs.get('output_format', 'markdown')

        context.update_progress(10, "Using simple formatter (generator not available)")

        # Generate simple output based on format
        if output_format == 'json':
            import json
            content = json.dumps(solution, indent=2)
        elif output_format == 'html':
            import json
            content = f"<html><body><pre>{json.dumps(solution, indent=2)}</pre></body></html>"
        else:  # markdown or text
            import json
            content = f"# Solution Output\n\n```\n{json.dumps(solution, indent=2)}\n```"

        result = {
            'output': content,
            'metadata': {
                'format': output_format,
                'template': 'simple',
                'language': 'en',
                'warning': 'Full generator not available, using simple formatter'
            },
            'word_count': len(content.split()),
            'sections': [{'name': 'main', 'length': len(content)}],
            'preview': self._generate_preview(content),
            'size_bytes': len(content.encode('utf-8'))
        }

        context.update_progress(100, "Simple formatting complete")
        return result

    def _generate_preview(self, content: str, max_length: int = 500) -> str:
        """Generate a preview of the output"""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Output Configuration",
            "description": "Configure output formatting and SOP generation",
            "properties": {
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "description": "Format for the generated output",
                    "enum": ["markdown", "html", "json", "text"],
                    "enumNames": [
                        "Markdown (.md)",
                        "HTML (.html)",
                        "JSON (.json)",
                        "Plain Text (.txt)"
                    ],
                    "default": "markdown"
                },
                "template": {
                    "type": "string",
                    "title": "Template",
                    "description": "Template to use for formatting",
                    "enum": ["standard", "detailed", "concise", "technical", "executive"],
                    "default": "standard"
                },
                "language": {
                    "type": "string",
                    "title": "Language",
                    "description": "Output language (2-letter ISO code)",
                    "minLength": 2,
                    "maxLength": 2,
                    "pattern": "^[a-z]{2}$",
                    "default": "en"
                },
                "include_sections": {
                    "type": "array",
                    "title": "Include Sections",
                    "description": "Sections to include in output",
                    "items": {
                        "type": "string",
                        "enum": [
                            "executive_summary",
                            "problem_statement",
                            "solution_overview",
                            "detailed_steps",
                            "code_examples",
                            "diagrams",
                            "references",
                            "appendix"
                        ]
                    },
                    "uniqueItems": True,
                    "default": [
                        "executive_summary",
                        "detailed_steps",
                        "references"
                    ]
                }
            },
            "required": ["output_format"]
        }
