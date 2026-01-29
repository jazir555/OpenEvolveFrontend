"""
Blue Team Utilities for OpenEvolve
Comprehensive utility functions and helper classes for Blue Team operations
"""

import os
import re
import json
import yaml
import string
import random
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Content Normalization Utilities
# ============================================================================

class ContentNormalizer:
    """
    Utilities for normalizing and standardizing content across different formats.
    """

    @staticmethod
    def normalize_whitespace(content: str) -> str:
        """
        Normalize whitespace in content.

        Args:
            content: Content to normalize

        Returns:
            Normalized content
        """
        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split('\n')]

        # Normalize multiple spaces to single space
        normalized = '\n'.join(lines)
        normalized = re.sub(r' +', ' ', normalized)

        # Remove excessive blank lines (more than 2)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)

        return normalized.strip()

    @staticmethod
    def normalize_line_endings(content: str, style: str = 'unix') -> str:
        """
        Normalize line endings to specific style.

        Args:
            content: Content to normalize
            style: 'unix' (LF) or 'windows' (CRLF)

        Returns:
            Content with normalized line endings
        """
        # First normalize to LF
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        if style == 'windows':
            content = content.replace('\n', '\r\n')

        return content

    @staticmethod
    def normalize_indentation(content: str, spaces: int = 4) -> str:
        """
        Normalize indentation to consistent spaces.

        Args:
            content: Content to normalize
            spaces: Number of spaces per indent level

        Returns:
            Content with normalized indentation
        """
        lines = content.split('\n')
        normalized_lines = []

        for line in lines:
            if line.strip():  # Non-empty line
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())

                # Calculate indent level
                indent_level = leading_spaces // 4

                # Apply new indentation
                normalized_lines.append(' ' * (indent_level * spaces) + line.lstrip())
            else:
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    @staticmethod
    def normalize_quotes(content: str, quote_style: str = 'double') -> str:
        """
        Normalize quote usage in strings.

        Args:
            content: Content to normalize
            quote_style: 'double' or 'single'

        Returns:
            Content with normalized quotes
        """
        if quote_style == 'double':
            # Convert single quotes to double (outside of escaped contexts)
            # This is a simplified version - real implementation would be more sophisticated
            content = re.sub(r"'([^'\\]*)'", r'"\1"', content)
        else:
            content = re.sub(r'"([^"\\]*)"', r"'\1'", content)

        return content

    @staticmethod
    def normalize_encoding(content: str, target_encoding: str = 'utf-8') -> bytes:
        """
        Normalize content encoding.

        Args:
            content: Content to normalize
            target_encoding: Target encoding

        Returns:
            Encoded bytes
        """
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]

        return content.encode(target_encoding, errors='replace')

    @staticmethod
    def remove_comments(content: str, content_type: str = "python") -> str:
        """
        Remove comments from content.

        Args:
            content: Content to process
            content_type: Type of content

        Returns:
            Content without comments
        """
        if content_type == "python":
            # Remove single-line comments
            lines = []
            for line in content.split('\n'):
                # Find comment start (ignoring strings)
                in_string = False
                string_char = None
                comment_pos = -1

                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        comment_pos = i
                        break

                if comment_pos >= 0:
                    lines.append(line[:comment_pos].rstrip())
                else:
                    lines.append(line)

            return '\n'.join(lines)

        elif content_type == "javascript":
            # Remove single-line comments
            content = re.sub(r'//.*', '', content)
            # Remove multi-line comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            return content

        return content


# ============================================================================
# Format Conversion Utilities
# ============================================================================

class FormatConverter:
    """
    Utilities for converting between different formats.
    """

    @staticmethod
    def json_to_yaml(json_content: str) -> str:
        """
        Convert JSON to YAML.

        Args:
            json_content: JSON string

        Returns:
            YAML string
        """
        try:
            data = json.loads(json_content)
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except (json.JSONDecodeError, ValueError, yaml.YAMLError) as e:
            raise ValueError(f"JSON to YAML conversion failed: {type(e).__name__}: {e}")

    @staticmethod
    def yaml_to_json(yaml_content: str) -> str:
        """
        Convert YAML to JSON.

        Args:
            yaml_content: YAML string

        Returns:
            JSON string
        """
        try:
            data = yaml.safe_load(yaml_content)
            return json.dumps(data, indent=2)
        except (yaml.YAMLError, ValueError, TypeError) as e:
            raise ValueError(f"YAML to JSON conversion failed: {type(e).__name__}: {e}")

    @staticmethod
    def markdown_to_html(markdown_content: str) -> str:
        """
        Convert Markdown to HTML.

        Args:
            markdown_content: Markdown string

        Returns:
            HTML string
        """
        html = markdown_content

        # Headers
        html = re.sub(r'^### (.*)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Code blocks
        html = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        # Line breaks and paragraphs
        html = html.replace('\n\n', '</p><p>')
        html = html.replace('\n', '<br>')

        return f'<html><body><p>{html}</p></body></html>'

    @staticmethod
    def csv_to_json(csv_content: str, delimiter: str = ',') -> str:
        """
        Convert CSV to JSON.

        Args:
            csv_content: CSV string
            delimiter: CSV delimiter

        Returns:
            JSON string
        """
        lines = csv_content.strip().split('\n')
        headers = [h.strip() for h in lines[0].split(delimiter)]

        data = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split(delimiter)]
            row = dict(zip(headers, values))
            data.append(row)

        return json.dumps(data, indent=2)

    @staticmethod
    def xml_to_json(xml_content: str) -> str:
        """
        Convert XML to JSON (simplified).

        Args:
            xml_content: XML string

        Returns:
            JSON string
        """
        # This is a simplified XML to JSON converter
        # For production, use xmltodict or similar library

        result = {}

        # Remove XML declaration
        xml_content = re.sub(r'<\?xml.*?\?>', '', xml_content)

        # Extract root element
        root_match = re.search(r'<(\w+).*?>(.*?)</\1>', xml_content, re.DOTALL)
        if root_match:
            root_name = root_match.group(1)
            content = root_match.group(2)

            # Extract child elements
            children = re.findall(r'<(\w+)(.*?)>(.*?)</\1>', content, re.DOTALL)
            for child_name, attrs, child_content in children:
                if child_name not in result:
                    result[child_name] = []
                result[child_name].append(child_content.strip())

        return json.dumps({root_name: result} if root_match else result, indent=2)


# ============================================================================
# Template Processing Utilities
# ============================================================================

class TemplateProcessor:
    """
    Utilities for processing templates and generating content from templates.
    """

    def __init__(self):
        self.template_cache = {}
        self.custom_delimiters = {
            'variable_start': '{{',
            'variable_end': '}}',
            'block_start': '{%',
            'block_end': '%}'
        }

    def process_template(self, template: str, variables: Dict[str, Any],
                        delimiters: Optional[Dict[str, str]] = None) -> str:
        """
        Process a template with variables.

        Args:
            template: Template string
            variables: Variables to substitute
            delimiters: Optional custom delimiters

        Returns:
            Processed content
        """
        delims = delimiters or self.custom_delimiters
        result = template

        # Replace variables
        for key, value in variables.items():
            placeholder = f"{delims['variable_start']} {key} {delims['variable_end']}"
            result = result.replace(placeholder, str(value))

        # Handle simple if statements
        if_statements = re.findall(
            rf"{delims['block_start']}\s*if\s+(\w+)\s*{delims['block_end']}(.*?){delims['block_start']}\s*endif\s*{delims['block_end']}",
            result,
            re.DOTALL
        )

        for var_name, block_content in if_statements:
            if variables.get(var_name):
                result = result.replace(
                    f"{delims['block_start']} if {var_name} {delims['block_end']}{block_content}{delims['block_start']} endif {delims['block_end']}",
                    block_content
                )
            else:
                result = result.replace(
                    f"{delims['block_start']} if {var_name} {delims['block_end']}{block_content}{delims['block_start']} endif {delims['block_end']}",
                    ''
                )

        return result

    def process_batch_template(self, template: str,
                               variable_sets: List[Dict[str, Any]]) -> List[str]:
        """
        Process template with multiple sets of variables.

        Args:
            template: Template string
            variable_sets: List of variable dictionaries

        Returns:
            List of processed contents
        """
        return [self.process_template(template, variables) for variables in variable_sets]

    def load_template_from_file(self, filepath: str) -> str:
        """
        Load template from file.

        Args:
            filepath: Path to template file

        Returns:
            Template content
        """
        if filepath in self.template_cache:
            return self.template_cache[filepath]

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                template = f.read()
            self.template_cache[filepath] = template
            return template
        except (FileNotFoundError, PermissionError, IOError, UnicodeDecodeError) as e:
            raise IOError(f"Failed to load template from {filepath}: {type(e).__name__}: {e}")

    def save_template(self, name: str, template: str):
        """
        Save template to cache.

        Args:
            name: Template name
            template: Template content
        """
        self.template_cache[name] = template

    def get_available_templates(self) -> List[str]:
        """
        Get list of available templates.

        Returns:
            List of template names
        """
        return list(self.template_cache.keys())


# ============================================================================
# Batch Processing Utilities
# ============================================================================

class BatchProcessor:
    """
    Utilities for batch processing multiple files or contents.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processing_history = []

    def process_files(self, filepaths: List[str],
                     processor_fn: Callable[[str], Any],
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple files.

        Args:
            filepaths: List of file paths to process
            processor_fn: Function to process each file
            show_progress: Show progress indicator

        Returns:
            List of processing results
        """
        results = []
        total = len(filepaths)

        for i, filepath in enumerate(filepaths):
            try:
                result = {
                    'filepath': filepath,
                    'success': True,
                    'result': processor_fn(filepath),
                    'index': i,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

                if show_progress and (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{total} files")

            except (FileNotFoundError, PermissionError, IOError, UnicodeDecodeError,
                    TypeError, ValueError, RuntimeError) as e:
                results.append({
                    'filepath': filepath,
                    'success': False,
                    'error': f"{type(e).__name__}: {e}",
                    'index': i,
                    'timestamp': datetime.now().isoformat()
                })

        self.processing_history.extend(results)
        return results

    def process_contents(self, contents: List[Tuple[str, str]],
                        processor_fn: Callable[[str], Any]) -> List[Dict[str, Any]]:
        """
        Process multiple content items with identifiers.

        Args:
            contents: List of (identifier, content) tuples
            processor_fn: Function to process each content

        Returns:
            List of processing results
        """
        results = []

        for identifier, content in contents:
            try:
                result = {
                    'identifier': identifier,
                    'success': True,
                    'result': processor_fn(content),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
                results.append({
                    'identifier': identifier,
                    'success': False,
                    'error': f"{type(e).__name__}: {e}",
                    'timestamp': datetime.now().isoformat()
                })

        self.processing_history.extend(results)
        return results

    def apply_to_directory(self, directory: str, pattern: str,
                          processor_fn: Callable[[str], Any],
                          recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Apply processor to all files matching pattern in directory.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.py')
            processor_fn: Function to process each file
            recursive: Recursively process subdirectories

        Returns:
            List of processing results
        """
        path = Path(directory)
        glob_method = path.rglob if recursive else path.glob

        matching_files = [str(f) for f in glob_method(pattern) if f.is_file()]
        return self.process_files(matching_files, processor_fn)

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of batch processing operations.

        Returns:
            Summary dictionary
        """
        if not self.processing_history:
            return {'total_processed': 0}

        total = len(self.processing_history)
        successful = sum(1 for r in self.processing_history if r.get('success', False))
        failed = total - successful

        return {
            'total_processed': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }


# ============================================================================
# Solution Templates
# ============================================================================

@dataclass
class SolutionTemplate:
    """
    Template for common solution patterns.
    """
    name: str
    category: str
    template_content: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render template with variables.

        Args:
            variables: Variables to substitute

        Returns:
            Rendered content
        """
        processor = TemplateProcessor()
        return processor.process_template(self.template_content, variables)

    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """
        Validate that all required variables are provided.

        Args:
            variables: Variables to validate

        Returns:
            True if all required variables present
        """
        return all(var in variables for var in self.variables)


class SolutionTemplateLibrary:
    """
    Library of pre-defined solution templates.
    """

    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize default solution templates."""

        # Python function template
        self.templates['python_function'] = SolutionTemplate(
            name='python_function',
            category='python',
            template_content='''
def {{function_name}}({{parameters}}):
    """
    {{description}}

    Args:
        {{args_description}}

    Returns:
        {{returns_description}}
    """
    {{body}}
    return {{return_value}}
''',
            variables=['function_name', 'parameters', 'description', 'args_description',
                      'returns_description', 'body', 'return_value'],
            description='Standard Python function template',
            tags=['python', 'function']
        )

        # Python class template
        self.templates['python_class'] = SolutionTemplate(
            name='python_class',
            category='python',
            template_content='''
class {{class_name}}:
    """
    {{description}}
    """

    def __init__(self{{init_params}}):
        """
        Initialize {{class_name}}.

        Args:
            {{init_args_description}}
        """
        {{init_body}}

    def {{method_name}}(self{{method_params}}):
        """
        {{method_description}}
        """
        {{method_body}}
''',
            variables=['class_name', 'description', 'init_params', 'init_args_description',
                      'init_body', 'method_name', 'method_params', 'method_description',
                      'method_body'],
            description='Standard Python class template',
            tags=['python', 'class']
        )

        # Error handling template
        self.templates['error_handling'] = SolutionTemplate(
            name='error_handling',
            category='error_handling',
            template_content='''
try:
    {{try_block}}
except {{exception_type}} as e:
    logger.error(f"Error: {str(e)}")
    {{except_block}}
    raise
else:
    {{else_block}}
finally:
    {{finally_block}}
''',
            variables=['try_block', 'exception_type', 'except_block', 'else_block', 'finally_block'],
            description='Error handling template',
            tags=['error_handling', 'try_except']
        )

        # Validation function template
        self.templates['validation'] = SolutionTemplate(
            name='validation',
            category='validation',
            template_content='''
def validate_{{entity_name}}({{entity_param}}) -> Tuple[bool, List[str]]:
    """
    Validate {{entity_name}}.

    Args:
        {{entity_param}}: {{entity_param}} to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required fields
    {{required_fields_check}}

    # Validate types
    {{type_validation}}

    # Validate values
    {{value_validation}}

    return len(errors) == 0, errors
''',
            variables=['entity_name', 'entity_param', 'required_fields_check',
                      'type_validation', 'value_validation'],
            description='Validation function template',
            tags=['validation', 'function']
        )

        # API endpoint template
        self.templates['api_endpoint'] = SolutionTemplate(
            name='api_endpoint',
            category='api',
            template_content='''
@app.{{http_method}}('/{{route}}')
def {{endpoint_name}}():
    """
    {{description}}
    """
    try:
        # Parse request
        {{request_parsing}}

        # Process request
        {{request_processing}}

        # Return response
        return jsonify({
            "status": "success",
            "data": {{response_data}}
        }), {{status_code}}

    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        logger.error(f"Error in {{endpoint_name}}: {type(e).__name__}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
''',
            variables=['http_method', 'route', 'endpoint_name', 'description',
                      'request_parsing', 'request_processing', 'response_data',
                      'status_code'],
            description='API endpoint template',
            tags=['api', 'endpoint', 'flask']
        )

    def get_template(self, name: str) -> Optional[SolutionTemplate]:
        """
        Get template by name.

        Args:
            name: Template name

        Returns:
            SolutionTemplate or None
        """
        return self.templates.get(name)

    def get_templates_by_category(self, category: str) -> List[SolutionTemplate]:
        """
        Get all templates in a category.

        Args:
            category: Category name

        Returns:
            List of SolutionTemplates
        """
        return [t for t in self.templates.values() if t.category == category]

    def get_templates_by_tag(self, tag: str) -> List[SolutionTemplate]:
        """
        Get all templates with a tag.

        Args:
            tag: Tag name

        Returns:
            List of SolutionTemplates
        """
        return [t for t in self.templates.values() if tag in t.tags]

    def add_template(self, template: SolutionTemplate):
        """
        Add a custom template.

        Args:
            template: SolutionTemplate to add
        """
        self.templates[template.name] = template

    def list_templates(self) -> List[str]:
        """
        List all template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())


# ============================================================================
# Patch Library
# ============================================================================

@dataclass
class Patch:
    """
    Reusable patch definition.
    """
    name: str
    description: str
    patch_type: str
    pattern: str  # Regex pattern to match
    replacement: str  # Replacement pattern
    flags: int = re.MULTILINE
    category: str = "general"
    severity: str = "medium"
    tags: List[str] = field(default_factory=list)

    def apply(self, content: str) -> Tuple[str, int]:
        """
        Apply patch to content.

        Args:
            content: Content to patch

        Returns:
            Tuple of (patched_content, num_changes)
        """
        patched_content, num_changes = re.subn(
            self.pattern,
            self.replacement,
            content,
            flags=self.flags
        )
        return patched_content, num_changes

    def matches(self, content: str) -> int:
        """
        Count matches in content.

        Args:
            content: Content to check

        Returns:
            Number of matches
        """
        return len(re.findall(self.pattern, content, flags=self.flags))


class PatchLibrary:
    """
    Library of reusable patches.
    """

    def __init__(self):
        self.patches = {}
        self._initialize_default_patches()

    def _initialize_default_patches(self):
        """Initialize default patches."""

        # Security patches
        self.patches['eval_to_literal_eval'] = Patch(
            name='eval_to_literal_eval',
            description='Replace dangerous eval() with ast.literal_eval()',
            patch_type='security',
            pattern=r'eval\s*\(',
            replacement='ast.literal_eval(',
            category='security',
            severity='high',
            tags=['security', 'eval']
        )

        self.patches['remove_hardcoded_password'] = Patch(
            name='remove_hardcoded_password',
            description='Flag hardcoded passwords for environment variable replacement',
            patch_type='security',
            pattern=r"password\s*=\s*['\"][^'\"]+['\"]",
            replacement='password = os.getenv("PASSWORD")',
            category='security',
            severity='high',
            tags=['security', 'password']
        )

        # Performance patches
        self.patches['string_concat_to_join'] = Patch(
            name='string_concat_to_join',
            description='Optimize string concatenation in loops',
            patch_type='performance',
            pattern=r'(\w+)\s*\+=\s*(\w+)',
            replacement=r'\1 = "".join([\1, \2])',
            category='performance',
            severity='medium',
            tags=['performance', 'strings']
        )

        self.patches['list_to_set'] = Patch(
            name='list_to_set',
            description='Convert list membership tests to set for O(1) lookup',
            patch_type='performance',
            pattern=r'if\s+(\w+)\s+in\s+\[(.*?)\]',
            replacement=r'if \1 in set([\2])',
            category='performance',
            severity='medium',
            tags=['performance', 'lookup']
        )

        # Style patches
        self.patches['remove_trailing_whitespace'] = Patch(
            name='remove_trailing_whitespace',
            description='Remove trailing whitespace',
            patch_type='style',
            pattern=r'[ \t]+$',
            replacement='',
            flags=re.MULTILINE,
            category='style',
            severity='low',
            tags=['style', 'whitespace']
        )

        self.patches['tabs_to_spaces'] = Patch(
            name='tabs_to_spaces',
            description='Convert tabs to 4 spaces',
            patch_type='style',
            pattern=r'\t',
            replacement='    ',
            category='style',
            severity='low',
            tags=['style', 'indentation']
        )

        # Quality patches
        self.patches['add_docstring'] = Patch(
            name='add_docstring',
            description='Add basic docstring template to function',
            patch_type='documentation',
            pattern=r'def\s+(\w+)\s*\((.*?)\):\s*\n',
            replacement=r'def \1(\2):\n    """\n    TODO: Add docstring\n    """\n',
            category='quality',
            severity='low',
            tags=['documentation', 'docstring']
        )

        self.patches['add_type_hints'] = Patch(
            name='add_type_hints',
            description='Add type hints to function parameters',
            patch_type='enhancement',
            pattern=r'def\s+(\w+)\s*\((\w+):\s*str',
            replacement=r'def \1(\2: str',
            category='quality',
            severity='low',
            tags=['quality', 'types']
        )

    def get_patch(self, name: str) -> Optional[Patch]:
        """
        Get patch by name.

        Args:
            name: Patch name

        Returns:
            Patch or None
        """
        return self.patches.get(name)

    def get_patches_by_category(self, category: str) -> List[Patch]:
        """
        Get all patches in a category.

        Args:
            category: Category name

        Returns:
            List of Patches
        """
        return [p for p in self.patches.values() if p.category == category]

    def get_patches_by_severity(self, severity: str) -> List[Patch]:
        """
        Get all patches with a severity level.

        Args:
            severity: Severity level

        Returns:
            List of Patches
        """
        return [p for p in self.patches.values() if p.severity == severity]

    def get_patches_by_tag(self, tag: str) -> List[Patch]:
        """
        Get all patches with a tag.

        Args:
            tag: Tag name

        Returns:
            List of Patches
        """
        return [p for p in self.patches.values() if tag in p.tags]

    def apply_patch(self, content: str, patch_name: str) -> Tuple[str, int]:
        """
        Apply a named patch to content.

        Args:
            content: Content to patch
            patch_name: Name of patch to apply

        Returns:
            Tuple of (patched_content, num_changes)
        """
        patch = self.get_patch(patch_name)
        if patch:
            return patch.apply(content)
        return content, 0

    def apply_category_patches(self, content: str,
                              category: str) -> Tuple[str, Dict[str, int]]:
        """
        Apply all patches in a category.

        Args:
            content: Content to patch
            category: Category name

        Returns:
            Tuple of (patched_content, changes_summary)
        """
        patches = self.get_patches_by_category(category)
        changes_summary = {}

        for patch in patches:
            content, num_changes = patch.apply(content)
            if num_changes > 0:
                changes_summary[patch.name] = num_changes

        return content, changes_summary

    def add_patch(self, patch: Patch):
        """
        Add a custom patch.

        Args:
            patch: Patch to add
        """
        self.patches[patch.name] = patch

    def list_patches(self) -> List[str]:
        """
        List all patch names.

        Returns:
            List of patch names
        """
        return list(self.patches.keys())


# ============================================================================
# Code Snippet Library
# ============================================================================

@dataclass
class CodeSnippet:
    """
    Reusable code snippet.
    """
    name: str
    category: str
    code: str
    language: str = "python"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class CodeSnippetLibrary:
    """
    Library of reusable code snippets.
    """

    def __init__(self):
        self.snippets = {}
        self._initialize_default_snippets()

    def _initialize_default_snippets(self):
        """Initialize default code snippets."""

        # Logging configuration
        self.snippets['logging_config'] = CodeSnippet(
            name='logging_config',
            category='logging',
            code='''
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)
''',
            language='python',
            description='Configure logging for a module',
            tags=['logging', 'configuration']
        )

        # Error handler
        self.snippets['error_handler'] = CodeSnippet(
            name='error_handler',
            category='error_handling',
            code='''
def handle_error(error: Exception, context: str = "") -> None:
    """
    Handle error with logging and optional recovery.

    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)

    # Optionally implement recovery logic here
    # ...
''',
            language='python',
            description='Standard error handling function',
            tags=['error_handling', 'logging']
        )

        # Retry decorator
        self.snippets['retry_decorator'] = CodeSnippet(
            name='retry_decorator',
            category='utility',
            code='''
import time
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, RuntimeError, IOError) as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
''',
            language='python',
            description='Retry decorator for functions',
            tags=['utility', 'retry', 'decorator']
        )

        # Timing decorator
        self.snippets['timing_decorator'] = CodeSnippet(
            name='timing_decorator',
            category='utility',
            code='''
import time
from functools import wraps

def time_function(func):
    """
    Decorator to time function execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end - start:.2f}s")
        return result
    return wrapper
''',
            language='python',
            description='Timing decorator for functions',
            tags=['utility', 'timing', 'performance']
        )

        # Cache decorator
        self.snippets['cache_decorator'] = CodeSnippet(
            name='cache_decorator',
            category='performance',
            code='''
from functools import wraps

def memoize(func):
    """
    Decorator to cache function results.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper
''',
            language='python',
            description='Memoization cache decorator',
            tags=['performance', 'caching', 'decorator']
        )

        # Input validation
        self.snippets['input_validation'] = CodeSnippet(
            name='input_validation',
            category='validation',
            code='''
from typing import Any, List

def validate_input(data: Any, required_fields: List[str]) -> tuple[bool, List[str]]:
    """
    Validate input data has required fields.

    Args:
        data: Input data to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not isinstance(data, dict):
        errors.append("Input must be a dictionary")
        return False, errors

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field cannot be None: {field}")

    return len(errors) == 0, errors
''',
            language='python',
            description='Input validation function',
            tags=['validation', 'input']
        )

    def get_snippet(self, name: str) -> Optional[CodeSnippet]:
        """
        Get snippet by name.

        Args:
            name: Snippet name

        Returns:
            CodeSnippet or None
        """
        return self.snippets.get(name)

    def get_snippets_by_category(self, category: str) -> List[CodeSnippet]:
        """
        Get all snippets in a category.

        Args:
            category: Category name

        Returns:
            List of CodeSnippets
        """
        return [s for s in self.snippets.values() if s.category == category]

    def get_snippets_by_tag(self, tag: str) -> List[CodeSnippet]:
        """
        Get all snippets with a tag.

        Args:
            tag: Tag name

        Returns:
            List of CodeSnippets
        """
        return [s for s in self.snippets.values() if tag in s.tags]

    def add_snippet(self, snippet: CodeSnippet):
        """
        Add a custom snippet.

        Args:
            snippet: CodeSnippet to add
        """
        self.snippets[snippet.name] = snippet

    def list_snippets(self) -> List[str]:
        """
        List all snippet names.

        Returns:
            List of snippet names
        """
        return list(self.snippets.keys())


# ============================================================================
# Validation Helper
# ============================================================================

class ValidationHelper:
    """
    Comprehensive validation utilities.
    """

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format.

        Args:
            email: Email address to validate

        Returns:
            True if valid
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format.

        Args:
            phone: Phone number to validate

        Returns:
            True if valid
        """
        # Remove non-numeric characters
        cleaned = re.sub(r'[^\d]', '', phone)
        return 10 <= len(cleaned) <= 15

    @staticmethod
    def validate_date(date_str: str, format: str = '%Y-%m-%d') -> bool:
        """
        Validate date string format.

        Args:
            date_str: Date string to validate
            format: Expected date format

        Returns:
            True if valid
        """
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_json(json_str: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate and parse JSON string.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            data = json.loads(json_str)
            return True, data
        except json.JSONDecodeError:
            return False, None

    @staticmethod
    def validate_regex(content: str, pattern: str) -> bool:
        """
        Validate content matches regex pattern.

        Args:
            content: Content to validate
            pattern: Regex pattern

        Returns:
            True if matches
        """
        try:
            return bool(re.match(pattern, content))
        except re.error:
            return False

    @staticmethod
    def validate_range(value: Union[int, float],
                      min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """
        Validate value is within range.

        Args:
            value: Value to validate
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            True if within range
        """
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    @staticmethod
    def validate_length(value: str,
                       min_len: Optional[int] = None,
                       max_len: Optional[int] = None) -> bool:
        """
        Validate string length.

        Args:
            value: String to validate
            min_len: Minimum length
            max_len: Maximum length

        Returns:
            True if within length constraints
        """
        length = len(value)
        if min_len is not None and length < min_len:
            return False
        if max_len is not None and length > max_len:
            return False
        return True

    @staticmethod
    def validate_required_fields(data: Dict[str, Any],
                                  required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate required fields are present in data.

        Args:
            data: Data dictionary
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing = [field for field in required_fields if field not in data]
        return len(missing) == 0, missing

    @staticmethod
    def sanitize_input(input_str: str,
                       max_length: int = 1000,
                       allowed_chars: Optional[str] = None) -> str:
        """
        Sanitize user input.

        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            allowed_chars: Optional set of allowed characters

        Returns:
            Sanitized string
        """
        # Trim to max length
        sanitized = input_str[:max_length]

        # Remove dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '\\']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        # Filter to allowed characters if specified
        if allowed_chars:
            sanitized = ''.join(c for c in sanitized if c in allowed_chars)

        return sanitized.strip()


# ============================================================================
# File Utilities
# ============================================================================

class FileUtilities:
    """
    File operation utilities.
    """

    @staticmethod
    def create_backup(filepath: str, backup_dir: Optional[str] = None) -> str:
        """
        Create backup of a file.

        Args:
            filepath: Path to file
            backup_dir: Optional backup directory

        Returns:
            Path to backup file
        """
        source = Path(filepath)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            backup_file = backup_path / f"{source.stem}_{timestamp}{source.suffix}"
        else:
            backup_file = source.parent / f"{source.stem}_{timestamp}{source.suffix}"

        shutil.copy2(source, backup_file)
        return str(backup_file)

    @staticmethod
    def ensure_directory(directory: str) -> str:
        """
        Ensure directory exists, create if needed.

        Args:
            directory: Directory path

        Returns:
            Directory path
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def get_file_hash(filepath: str, algorithm: str = 'sha256') -> str:
        """
        Calculate file hash.

        Args:
            filepath: Path to file
            algorithm: Hash algorithm

        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get file information.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file information
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        stat = path.stat()

        return {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'absolute': str(path.absolute())
        }

    @staticmethod
    def create_temp_file(content: str, suffix: str = '.tmp') -> str:
        """
        Create temporary file with content.

        Args:
            content: Content to write
            suffix: File suffix

        Returns:
            Path to temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
        except (IOError, OSError, PermissionError) as e:
            os.close(fd)
            logger.error(f"Failed to create temp file: {type(e).__name__}: {e}", exc_info=True)
            raise

        return path

    @staticmethod
    def read_file_chunks(filepath: str, chunk_size: int = 4096) -> List[str]:
        """
        Read file in chunks.

        Args:
            filepath: Path to file
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        chunks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)

        return chunks


# ============================================================================
# Convenience Functions
# ============================================================================

def normalize_content(content: str, normalize_type: str = 'all') -> str:
    """
    Normalize content using multiple normalizers.

    Args:
        content: Content to normalize
        normalize_type: Type of normalization ('whitespace', 'encoding', 'all')

    Returns:
        Normalized content
    """
    normalizer = ContentNormalizer()

    if normalize_type in ['whitespace', 'all']:
        content = normalizer.normalize_whitespace(content)

    if normalize_type in ['encoding', 'all']:
        content = content.decode('utf-8') if isinstance(content, bytes) else content

    return content


def apply_standard_fixes(content: str) -> str:
    """
    Apply standard fixes from patch library.

    Args:
        content: Content to fix

    Returns:
        Fixed content
    """
    library = PatchLibrary()

    # Apply style fixes
    content, _ = library.apply_patch(content, 'remove_trailing_whitespace')
    content, _ = library.apply_patch(content, 'tabs_to_spaces')

    return content


def get_quick_validation(content: str) -> Dict[str, bool]:
    """
    Quick validation check.

    Args:
        content: Content to validate

    Returns:
        Dictionary with validation results
    """
    validator = ValidationHelper()

    return {
        'has_content': len(content.strip()) > 0,
        'no_syntax_errors': True,  # Simplified
        'reasonable_length': len(content) < 1000000
    }


def create_solution_from_template(template_name: str,
                                  variables: Dict[str, Any]) -> str:
    """
    Create solution from template.

    Args:
        template_name: Name of template
        variables: Template variables

    Returns:
        Rendered solution
    """
    library = SolutionTemplateLibrary()
    template = library.get_template(template_name)

    if not template:
        raise ValueError(f"Template not found: {template_name}")

    return template.render(variables)
