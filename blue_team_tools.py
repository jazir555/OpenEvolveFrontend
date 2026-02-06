"""
Blue Team Tools and Utilities for OpenEvolve
Comprehensive toolset for solution analysis, patch generation, and validation
"""


import os
import re
import ast
import json
import math
import difflib
import hashlib
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import Counter, defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)

# **LEAN INTEGRATION**: Real Lean proof verification for blue team tools
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logger.warning("LeanAide client not available - formal verification tools disabled")


async def verify_with_lean_tools(content: str, tool_context: str = None) -> Dict[str, Any]:
    """Verify content using Lean theorem prover for blue team tools.
    
    Args:
        content: The content to verify (patch, fix, or solution)
        tool_context: Optional context about which tool is using verification
        
    Returns:
        Dictionary with verification results
    """
    if not LEAN_AVAILABLE:
        return {"verified": False, "reason": "Lean unavailable"}
    
    try:
        client = LeanAideClient()
        
        # Translate content to formal theorem statement
        formalized = await client.translate_thm(content)
        
        # Verify the formalized content
        result = await client.verify(formalized)
        
        return {
            "verified": result.verified if hasattr(result, 'verified') else False,
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
            "proof": result.proof_code if hasattr(result, 'proof_code') else None,
            "tool_context": tool_context,
            "formalized_content": formalized
        }
    except Exception as e:
        logger.warning(f"Lean tool verification failed: {e}")
        return {"verified": False, "reason": str(e), "tool_context": tool_context}


class LeanVerificationToolsMixin:
    """Mixin class providing Lean verification tools."""
    
    async def _verify_with_lean(
        self, 
        content: str, 
        tool_context: str = None
    ) -> Dict[str, Any]:
        """Verify content using Lean theorem prover.
        
        Args:
            content: The content to verify
            tool_context: Optional context about the tool
            
        Returns:
            Dictionary with verification results
        """
        return await verify_with_lean_tools(content, tool_context)

class AnalysisType(Enum):
    """Types of analysis supported"""
    COMPLEXITY = "complexity"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"

class PatchType(Enum):
    """Types of patches"""
    FIX = "fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"

class ValidationType(Enum):
    """Types of validation"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    REGRESSION = "regression"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"

@dataclass
class AnalysisResult:
    """Result of analysis operation"""
    analysis_type: AnalysisType
    score: float  # 0-100
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatchResult:
    """Result of patch operation"""
    patch_type: PatchType
    success: bool
    original_content: str
    patched_content: str
    diff: str
    changes: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    rollback_info: Optional[Dict[str, Any]] = None

@dataclass
class ValidationResult:
    """Result of validation operation"""
    validation_type: ValidationType
    passed: bool
    score: float  # 0-100
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Solution Analysis Tools
# ============================================================================

class SolutionAnalysisTools:
    """
    Comprehensive suite of tools for analyzing solutions, code, and content.
    Provides complexity analysis, dependency analysis, security scanning,
    and performance bottleneck detection.
    """

    def __init__(self):
        self.cache = {}
        self.analysis_history = []

    def analyze_complexity(self, content: str, content_type: str = "python") -> AnalysisResult:
        """
        Analyze code/content complexity using multiple metrics.

        Args:
            content: The code or content to analyze
            content_type: Type of content (python, javascript, text, etc.)

        Returns:
            AnalysisResult with complexity metrics and recommendations
        """
        cache_key = f"complexity_{hash(content)}_{content_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        issues = []
        metrics = {}
        recommendations = []

        if content_type == "python":
            metrics = self._analyze_python_complexity(content)
        elif content_type == "javascript":
            metrics = self._analyze_javascript_complexity(content)
        elif content_type == "text":
            metrics = self._analyze_text_complexity(content)
        else:
            metrics = self._analyze_generic_complexity(content)

        # Calculate overall complexity score (0-100, inverted: lower complexity = higher score)
        complexity_score = max(0, min(100, 100 - metrics.get('complexity_value', 50)))

        # Generate issues based on metrics
        if metrics.get('cyclomatic_complexity', 0) > 10:
            issues.append({
                'severity': 'high',
                'category': 'complexity',
                'message': f"High cyclomatic complexity: {metrics.get('cyclomatic_complexity')}",
                'location': self._find_complex_functions(content)
            })
            recommendations.append("Consider breaking down complex functions into smaller, more focused units")

        if metrics.get('nesting_depth', 0) > 4:
            issues.append({
                'severity': 'medium',
                'category': 'complexity',
                'message': f"Deep nesting detected: {metrics.get('nesting_depth')} levels",
                'location': self._find_deep_nesting(content)
            })
            recommendations.append("Reduce nesting depth by using early returns and guard clauses")

        if metrics.get('function_length', 0) > 50:
            issues.append({
                'severity': 'medium',
                'category': 'complexity',
                'message': f"Long functions detected: average {metrics.get('function_length')} lines",
                'location': self._find_long_functions(content)
            })
            recommendations.append("Break long functions into smaller, more maintainable pieces")

        result = AnalysisResult(
            analysis_type=AnalysisType.COMPLEXITY,
            score=complexity_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            metadata={'content_type': content_type}
        )

        self.cache[cache_key] = result
        self.analysis_history.append(result)
        return result

    def _analyze_python_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze Python-specific complexity metrics"""
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'complexity_value': 0,
            'cyclomatic_complexity': 0,
            'nesting_depth': 0,
            'function_length': 0,
            'num_functions': 0
        }

        try:
            tree = ast.parse(content)
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(node)
                    # Calculate cyclomatic complexity for this function
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                            complexity += 1
                        elif isinstance(child, (ast.And, ast.Or)):
                            complexity += 1
                    metrics['cyclomatic_complexity'] += complexity

                    # Calculate function length
                    if hasattr(node, 'end_lineno') and node.lineno:
                        func_length = node.end_lineno - node.lineno
                        metrics['function_length'] = max(metrics['function_length'], func_length)

                    # Calculate nesting depth
                    depth = self._calculate_nesting_depth(node)
                    metrics['nesting_depth'] = max(metrics['nesting_depth'], depth)

            metrics['num_functions'] = len(functions)
            if functions:
                metrics['function_length'] = metrics['function_length'] / len(functions)
                metrics['cyclomatic_complexity'] = metrics['cyclomatic_complexity'] / len(functions)

            # Calculate overall complexity value
            metrics['complexity_value'] = (
                metrics['cyclomatic_complexity'] * 10 +
                metrics['nesting_depth'] * 5 +
                metrics['function_length'] * 0.5
            )

        except (SyntaxError, ValueError, RecursionError) as e:
            logger.error(f"Python complexity analysis failed: {type(e).__name__}: {e}")
            metrics['complexity_value'] = 100  # High complexity if syntax error
            metrics['syntax_error'] = True

        return metrics

    def _analyze_javascript_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript-specific complexity metrics"""
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'complexity_value': 0,
            'cyclomatic_complexity': 0,
            'nesting_depth': 0,
            'function_length': 0,
            'num_functions': 0
        }

        lines = content.split('\n')
        brace_count = 0
        max_nesting = 0
        function_starts = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Count braces for nesting depth
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            brace_count += open_braces - close_braces
            max_nesting = max(max_nesting, brace_count)

            # Count functions
            if re.search(r'\bfunction\s+\w+|=>\s*{?', stripped):
                function_starts.append(i)

        metrics['nesting_depth'] = max_nesting
        metrics['num_functions'] = len(function_starts)

        # Calculate cyclomatic complexity (simplified)
        decision_keywords = ['if', 'else if', 'for', 'while', 'case', 'catch', '&&', '||', '?']
        complexity = 1
        for line in lines:
            for keyword in decision_keywords:
                complexity += line.count(keyword)

        metrics['cyclomatic_complexity'] = complexity if metrics['num_functions'] > 0 else complexity / max(1, metrics['num_functions'])

        # Calculate complexity value
        metrics['complexity_value'] = (
            metrics['cyclomatic_complexity'] * 10 +
            metrics['nesting_depth'] * 5
        )

        return metrics

    def _analyze_text_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze text complexity metrics"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')

        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        avg_sentence_length = len(words) / max(1, len(sentences))

        metrics = {
            'lines_of_code': len(content.split('\n')),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'complexity_value': 0
        }

        # Calculate readability score (simplified Flesch)
        metrics['readability_score'] = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 100))
        metrics['complexity_value'] = max(0, min(100, 100 - metrics['readability_score']))

        return metrics

    def _analyze_generic_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze generic content complexity"""
        lines = content.split('\n')
        metrics = {
            'lines_of_code': len(lines),
            'complexity_value': len(lines) / 10,  # Simple heuristic
            'avg_line_length': sum(len(line) for line in lines) / max(1, len(lines))
        }
        return metrics

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth for an AST node"""
        max_depth = 0

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                depth = 1 + self._calculate_nesting_depth(child)
                max_depth = max(max_depth, depth)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                depth = self._calculate_nesting_depth(child)
                max_depth = max(max_depth, depth)

        return max_depth

    def _find_complex_functions(self, content: str) -> List[Dict[str, int]]:
        """Find functions with high complexity"""
        locations = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = 1
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For)):
                            complexity += 1
                    if complexity > 10:
                        locations.append({
                            'function': node.name,
                            'line': node.lineno,
                            'complexity': complexity
                        })
        except (SyntaxError, ValueError, RecursionError) as e:
            logger.error(f"Error finding complex functions: {type(e).__name__}: {e}", exc_info=True)
            # Return empty list instead of raising
        return locations

    def _find_deep_nesting(self, content: str) -> List[Dict[str, int]]:
        """Find locations with deep nesting"""
        locations = []
        lines = content.split('\n')
        brace_count = 0

        for i, line in enumerate(lines, 1):
            brace_count += line.count('{') - line.count('}')
            if brace_count > 4:
                locations.append({'line': i, 'depth': brace_count})

        return locations

    def _find_long_functions(self, content: str) -> List[Dict[str, int]]:
        """Find long functions"""
        locations = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'end_lineno'):
                        length = node.end_lineno - node.lineno
                        if length > 50:
                            locations.append({
                                'function': node.name,
                                'line': node.lineno,
                                'length': length
                            })
        except (SyntaxError, ValueError, RecursionError) as e:
            logger.error(f"Error finding long functions: {type(e).__name__}: {e}", exc_info=True)
            # Return empty list instead of raising
        return locations

    def analyze_dependencies(self, content: str, content_type: str = "python") -> AnalysisResult:
        """
        Analyze dependencies and imports in content.

        Args:
            content: The code to analyze
            content_type: Type of content

        Returns:
            AnalysisResult with dependency information
        """
        issues = []
        metrics = {'imports': [], 'external_dependencies': [], 'internal_dependencies': []}
        recommendations = []

        if content_type == "python":
            metrics = self._analyze_python_dependencies(content)
        elif content_type == "javascript":
            metrics = self._analyze_javascript_dependencies(content)

        # Check for potential issues
        if len(metrics.get('external_dependencies', [])) > 20:
            issues.append({
                'severity': 'medium',
                'category': 'dependencies',
                'message': f"High number of external dependencies: {len(metrics['external_dependencies'])}"
            })
            recommendations.append("Consider reducing the number of external dependencies")

        if len(metrics.get('circular_dependencies', [])) > 0:
            issues.append({
                'severity': 'high',
                'category': 'dependencies',
                'message': f"Circular dependencies detected: {len(metrics['circular_dependencies'])}",
                'details': metrics['circular_dependencies']
            })
            recommendations.append("Resolve circular dependencies to improve modularity")

        # Calculate dependency score
        dependency_score = max(0, min(100, 100 - len(metrics.get('external_dependencies', [])) * 2))

        result = AnalysisResult(
            analysis_type=AnalysisType.DEPENDENCY,
            score=dependency_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            metadata={'content_type': content_type}
        )

        self.analysis_history.append(result)
        return result

    def _analyze_python_dependencies(self, content: str) -> Dict[str, Any]:
        """Analyze Python dependencies"""
        metrics = {
            'imports': [],
            'external_dependencies': set(),
            'internal_dependencies': set(),
            'circular_dependencies': []
        }

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metrics['imports'].append(alias.name)
                        module = alias.name.split('.')[0]
                        if module not in ['os', 'sys', 'json', 're', 'math', 'datetime', 'typing']:
                            metrics['external_dependencies'].add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metrics['imports'].append(node.module)
                        module = node.module.split('.')[0]
                        if module not in ['os', 'sys', 'json', 're', 'math', 'datetime', 'typing']:
                            metrics['external_dependencies'].add(module)

            metrics['external_dependencies'] = list(metrics['external_dependencies'])
            metrics['internal_dependencies'] = list(metrics['internal_dependencies'])
        except (SyntaxError, ValueError, RecursionError) as e:
            logger.error(f"Python dependency analysis failed: {type(e).__name__}: {e}", exc_info=True)
            # Return partial results instead of raising

        return metrics

    def _analyze_javascript_dependencies(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript dependencies"""
        metrics = {
            'imports': [],
            'external_dependencies': set(),
            'internal_dependencies': set()
        }

        # Match import statements
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\([\'"]([^\'"]+)[\'"]\)'
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                metrics['imports'].append(match)
                if match.startswith('./') or match.startswith('../'):
                    metrics['internal_dependencies'].add(match)
                elif not match.startswith('.'):
                    metrics['external_dependencies'].add(match)

        metrics['external_dependencies'] = list(metrics['external_dependencies'])
        metrics['internal_dependencies'] = list(metrics['internal_dependencies'])

        return metrics

    def analyze_security(self, content: str, content_type: str = "python") -> AnalysisResult:
        """
        Perform security vulnerability scanning.

        Args:
            content: Code to scan
            content_type: Type of content

        Returns:
            AnalysisResult with security findings
        """
        issues = []
        metrics = {'vulnerabilities': [], 'security_score': 100}
        recommendations = []

        # Common security patterns
        security_patterns = {
            'sql_injection': [
                (r'execute\s*\(\s*[\'"].*?\+.*?[\'"]', 'Potential SQL injection vulnerability'),
                (r'query\s*\(\s*[\'"].*?\+.*?[\'"]', 'Potential SQL injection vulnerability'),
            ],
            'xss': [
                (r'innerHTML\s*=\s*.*?\+', 'Potential XSS vulnerability'),
                (r'document\.write\s*\(\s*.*?\+', 'Potential XSS vulnerability'),
            ],
            'hardcoded_secrets': [
                (r'(password|secret|api_key|token)\s*=\s*[\'"][^\'"]{10,}[\'"]', 'Hardcoded secret detected'),
                (r'(aws_access_key|aws_secret)\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded AWS credentials detected'),
            ],
            'insecure_crypto': [
                (r'md5\s*\(', 'Weak hash function MD5 detected'),
                (r'sha1\s*\(', 'Weak hash function SHA1 detected'),
            ],
            'eval_usage': [
                (r'eval\s*\(', 'Use of eval() is dangerous'),
            ]
        }

        for category, patterns in security_patterns.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'severity': 'high' if category in ['sql_injection', 'hardcoded_secrets', 'eval_usage'] else 'medium',
                        'category': 'security',
                        'message': message,
                        'location': {'line': line_num, 'category': category}
                    })
                    metrics['vulnerabilities'].append({
                        'category': category,
                        'line': line_num,
                        'message': message
                    })

        # Generate recommendations based on findings
        if any(v['category'] == 'sql_injection' for v in metrics['vulnerabilities']):
            recommendations.append("Use parameterized queries or prepared statements to prevent SQL injection")

        if any(v['category'] == 'xss' for v in metrics['vulnerabilities']):
            recommendations.append("Sanitize user input and use textContent instead of innerHTML")

        if any(v['category'] == 'hardcoded_secrets' for v in metrics['vulnerabilities']):
            recommendations.append("Move secrets to environment variables or secure configuration management")

        if any(v['category'] == 'insecure_crypto' for v in metrics['vulnerabilities']):
            recommendations.append("Use stronger cryptographic algorithms (SHA-256, bcrypt, etc.)")

        if any(v['category'] == 'eval_usage' for v in metrics['vulnerabilities']):
            recommendations.append("Avoid using eval() - consider safer alternatives")

        # Calculate security score
        high_severity = sum(1 for i in issues if i['severity'] == 'high')
        medium_severity = sum(1 for i in issues if i['severity'] == 'medium')
        metrics['security_score'] = max(0, 100 - (high_severity * 20) - (medium_severity * 10))

        result = AnalysisResult(
            analysis_type=AnalysisType.SECURITY,
            score=metrics['security_score'],
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            metadata={'content_type': content_type}
        )

        self.analysis_history.append(result)
        return result

    def analyze_performance(self, content: str, content_type: str = "python") -> AnalysisResult:
        """
        Detect performance bottlenecks and anti-patterns.

        Args:
            content: Code to analyze
            content_type: Type of content

        Returns:
            AnalysisResult with performance findings
        """
        issues = []
        metrics = {'bottlenecks': [], 'performance_score': 100}
        recommendations = []

        # Common performance anti-patterns
        performance_patterns = {
            'nested_loops': [
                (r'for\s+\w+\s+in.*?:\s*for\s+\w+\s+in', 'Nested loops detected - consider optimization'),
            ],
            'inefficient_string_concat': [
                (r'\w+\s*\+=\s*\w+', 'String concatenation in loop - use join() instead'),
            ],
            'database_in_loop': [
                (r'for.*:\s*(query|execute|select)', 'Database query inside loop - use batch operations'),
            ],
            'large_object_copy': [
                (r'copy\.deepcopy\s*\(\s*\w+\s*\)', 'Large object copy - consider alternative approaches'),
            ],
            'synchronous_io': [
                (r'(open|read|write)\s*\(', 'Synchronous I/O operation - consider async alternatives'),
            ]
        }

        for category, patterns in performance_patterns.items():
            for pattern, message in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'severity': 'medium',
                        'category': 'performance',
                        'message': message,
                        'location': {'line': line_num, 'category': category}
                    })
                    metrics['bottlenecks'].append({
                        'category': category,
                        'line': line_num,
                        'message': message
                    })

        # Check for inefficient algorithms
        if content_type == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    # Check for O(n^2) operations
                    if isinstance(node, ast.For):
                        for child in ast.walk(node):
                            if isinstance(child, ast.For) and child != node:
                                issues.append({
                                    'severity': 'high',
                                    'category': 'performance',
                                    'message': 'Nested O(n^2) loop detected',
                                    'location': {'line': node.lineno, 'category': 'nested_loops'}
                                })
                                metrics['bottlenecks'].append({
                                    'category': 'nested_loops',
                                    'line': node.lineno,
                                    'message': 'Nested O(n^2) loop'
                                })
                                break
            except (SyntaxError, ValueError, RecursionError) as e:
                logger.error(f"Performance analysis failed: {type(e).__name__}: {e}", exc_info=True)

        # Generate recommendations
        if any(b['category'] == 'nested_loops' for b in metrics['bottlenecks']):
            recommendations.append("Consider using dictionaries, sets, or more efficient algorithms")

        if any(b['category'] == 'inefficient_string_concat' for b in metrics['bottlenecks']):
            recommendations.append("Use str.join() or f-strings for string concatenation")

        if any(b['category'] == 'database_in_loop' for b in metrics['bottlenecks']):
            recommendations.append("Use batch operations or bulk queries instead of looping")

        # Calculate performance score
        metrics['performance_score'] = max(0, 100 - len(issues) * 15)

        result = AnalysisResult(
            analysis_type=AnalysisType.PERFORMANCE,
            score=metrics['performance_score'],
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            metadata={'content_type': content_type}
        )

        self.analysis_history.append(result)
        return result

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed"""
        if not self.analysis_history:
            return {'total_analyses': 0}

        summary = {
            'total_analyses': len(self.analysis_history),
            'by_type': Counter(a.analysis_type.value for a in self.analysis_history),
            'average_score': sum(a.score for a in self.analysis_history) / len(self.analysis_history),
            'total_issues': sum(len(a.issues) for a in self.analysis_history),
            'total_recommendations': sum(len(a.recommendations) for a in self.analysis_history)
        }

        return summary


# ============================================================================
# Patch Generation Tools
# ============================================================================

class PatchGenerationTools:
    """
    Tools for generating, applying, testing, and rolling back patches.
    """

    def __init__(self):
        self.patch_history = []
        self.rollback_stack = []

    def generate_patch(self, original: str, modified: str, patch_type: PatchType = PatchType.FIX,
                      description: str = "") -> PatchResult:
        """
        Generate a patch between original and modified content.

        Args:
            original: Original content
            modified: Modified content
            patch_type: Type of patch
            description: Description of the patch

        Returns:
            PatchResult with diff and changes
        """
        # Generate unified diff
        diff = self._generate_diff(original, modified)

        # Extract changes
        changes = self._extract_changes(original, modified)

        # Prepare rollback info
        rollback_info = {
            'original_content': original,
            'patch_type': patch_type.value,
            'timestamp': self._get_timestamp(),
            'description': description
        }

        result = PatchResult(
            patch_type=patch_type,
            success=True,
            original_content=original,
            patched_content=modified,
            diff=diff,
            changes=changes,
            validation_results=[],
            rollback_info=rollback_info
        )

        self.patch_history.append(result)
        self.rollback_stack.append(rollback_info)

        return result

    def _generate_diff(self, original: str, modified: str, context_lines: int = 3) -> str:
        """Generate unified diff between two strings"""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile='original',
            tofile='modified',
            lineterm='',
            n=context_lines
        )

        return ''.join(diff)

    def _extract_changes(self, original: str, modified: str) -> List[Dict[str, Any]]:
        """Extract detailed changes between original and modified content"""
        changes = []

        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            change = {
                'operation': tag,
                'original_start': i1,
                'original_end': i2,
                'modified_start': j1,
                'modified_end': j2,
                'original_lines': original_lines[i1:i2],
                'modified_lines': modified_lines[j1:j2]
            }
            changes.append(change)

        return changes

    def apply_patch(self, content: str, patch: str) -> PatchResult:
        """
        Apply a unified diff patch to content.

        Args:
            content: Original content
            patch: Unified diff string

        Returns:
            PatchResult with patched content
        """
        try:
            # Save original for rollback
            original = content

            # Parse and apply patch
            patched_lines = content.splitlines(keepends=True)
            modified_lines = []

            # Parse diff
            diff_lines = patch.splitlines()
            i = 0
            patch_idx = 0

            while i < len(patched_lines):
                # Look for hunk headers
                if patch_idx < len(diff_lines) and diff_lines[patch_idx].startswith('@@'):
                    # Parse hunk header
                    match = re.match(r'@@\s+-(\d+),?(\d+)?\s+\+(\d+),?(\d+)?\s+@@', diff_lines[patch_idx])
                    if match:
                        orig_start = int(match.group(1)) - 1
                        mod_start = int(match.group(3)) - 1

                        # Copy lines up to the hunk
                        while i < orig_start and i < len(patched_lines):
                            modified_lines.append(patched_lines[i])
                            i += 1

                        patch_idx += 1

                        # Apply hunk
                        while patch_idx < len(diff_lines):
                            line = diff_lines[patch_idx]

                            if line.startswith('@@'):
                                break  # Next hunk

                            if line.startswith('+'):
                                modified_lines.append(line[1:] + '\n')
                            elif not line.startswith('-'):
                                # Context line
                                modified_lines.append(line + '\n' if not line.endswith('\n') else line)
                                i += 1

                            patch_idx += 1
                    else:
                        patch_idx += 1
                else:
                    modified_lines.append(patched_lines[i])
                    i += 1
                    if not diff_lines[patch_idx].startswith('@@'):
                        patch_idx += 1

            patched_content = ''.join(modified_lines)

            # Generate validation result
            diff = self._generate_diff(original, patched_content)
            changes = self._extract_changes(original, patched_content)

            result = PatchResult(
                patch_type=PatchType.FIX,
                success=True,
                original_content=original,
                patched_content=patched_content,
                diff=diff,
                changes=changes,
                validation_results=[],
                rollback_info={'original_content': original}
            )

            self.patch_history.append(result)
            return result

        except (ValueError, re.error, AttributeError, KeyError, IndexError) as e:
            logger.error(f"Patch application failed: {type(e).__name__}: {e}")
            return PatchResult(
                patch_type=PatchType.FIX,
                success=False,
                original_content=content,
                patched_content=content,
                diff='',
                changes=[],
                validation_results=[],
                rollback_info=None
            )

    def test_patch(self, original: str, patch: str, validation_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Test a patch before applying it.

        Args:
            original: Original content
            patch: Patch to test
            validation_fn: Optional validation function

        Returns:
            Dictionary with test results
        """
        result = {
            'can_apply': False,
            'test_passed': False,
            'errors': [],
            'warnings': []
        }

        try:
            # Try to apply patch
            patch_result = self.apply_patch(original, patch)
            result['can_apply'] = patch_result.success

            if not patch_result.success:
                result['errors'].append('Failed to apply patch')
                return result

            # Run validation if provided
            if validation_fn:
                try:
                    validation_result = validation_fn(patch_result.patched_content)
                    result['test_passed'] = validation_result.get('passed', False)
                    result['validation_details'] = validation_result
                except (TypeError, ValueError, AttributeError, KeyError) as e:
                    result['warnings'].append(f'Validation error: {type(e).__name__}: {e}')

            # Check for obvious issues
            if len(patch_result.patched_content) == 0:
                result['warnings'].append('Patched content is empty')
            elif len(patch_result.patched_content) < len(original) * 0.5:
                result['warnings'].append('Patched content significantly smaller than original')

            # Count changes
            added = sum(1 for c in patch_result.changes if c['operation'] == 'insert')
            removed = sum(1 for c in patch_result.changes if c['operation'] == 'delete')
            result['changes_summary'] = {
                'additions': added,
                'deletions': removed,
                'total_changes': len(patch_result.changes)
            }

        except (ValueError, re.error, AttributeError, KeyError, RuntimeError) as e:
            result['errors'].append(f'Test error: {type(e).__name__}: {e}')

        return result

    def rollback_patch(self, current_content: str, rollback_info: Dict[str, Any]) -> str:
        """
        Rollback a patch using rollback information.

        Args:
            current_content: Current content (may be patched)
            rollback_info: Rollback information from patch application

        Returns:
            Original content before patch
        """
        if rollback_info and 'original_content' in rollback_info:
            return rollback_info['original_content']

        # Try to find in rollback stack
        if self.rollback_stack:
            return self.rollback_stack.pop()['original_content']

        return current_content

    def create_automated_patch(self, content: str, issue: Dict[str, Any],
                              fix_pattern: str) -> PatchResult:
        """
        Create an automated patch based on issue and fix pattern.

        Args:
            content: Original content
            issue: Issue description
            fix_pattern: Pattern for the fix

        Returns:
            PatchResult with automated fix
        """
        modified_content = content

        # Apply fix pattern
        if issue.get('category') == 'security':
            modified_content = self._apply_security_fix(content, issue, fix_pattern)
        elif issue.get('category') == 'performance':
            modified_content = self._apply_performance_fix(content, issue, fix_pattern)
        elif issue.get('category') == 'complexity':
            modified_content = self._apply_complexity_fix(content, issue, fix_pattern)
        else:
            # Generic regex-based fix
            if 'pattern' in issue and 'replacement' in fix_pattern:
                modified_content = re.sub(issue['pattern'], fix_pattern['replacement'], content)

        return self.generate_patch(
            content,
            modified_content,
            PatchType.FIX,
            f"Automated fix for {issue.get('category', 'unknown')} issue"
        )

    def _apply_security_fix(self, content: str, issue: Dict[str, Any],
                           fix_pattern: str) -> str:
        """Apply security fix"""
        # Implement security-specific fix logic
        if issue.get('message', '').lower() == 'use of eval() is dangerous':
            return re.sub(r'eval\s*\(', 'safe_eval(', content)
        return content

    def _apply_performance_fix(self, content: str, issue: Dict[str, Any],
                              fix_pattern: str) -> str:
        """Apply performance fix"""
        if 'string concatenation' in issue.get('message', '').lower():
            # Convert += to .join() pattern
            return re.sub(r'(\w+)\s*\+=\s*(\w+)', r'\1 = "".join([\1, \2])', content)
        return content

    def _apply_complexity_fix(self, content: str, issue: Dict[str, Any],
                             fix_pattern: str) -> str:
        """Apply complexity fix"""
        # Simplify complex expressions
        return content

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_patch_history(self) -> List[Dict[str, Any]]:
        """Get history of all patches"""
        return [
            {
                'patch_type': p.patch_type.value,
                'success': p.success,
                'changes_count': len(p.changes),
                'rollback_info': p.rollback_info
            }
            for p in self.patch_history
        ]


# ============================================================================
# Validation Tools
# ============================================================================

class ValidationTools:
    """
    Comprehensive validation tools for solutions, patches, and quality checks.
    """

    def __init__(self):
        self.validation_history = []
        self.validation_rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default validation rules"""
        return {
            'max_line_length': 120,
            'max_function_length': 100,
            'max_complexity': 10,
            'require_docstrings': True,
            'check_style': True,
            'check_security': True,
            'check_performance': True
        }

    def validate_solution(self, content: str, content_type: str = "python",
                         rules: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Comprehensive solution validation.

        Args:
            content: Content to validate
            content_type: Type of content
            rules: Optional validation rules

        Returns:
            ValidationResult with comprehensive validation results
        """
        errors = []
        warnings = []
        suggestions = []
        metrics = {}

        validation_rules = rules or self.validation_rules

        # Syntax validation
        syntax_result = self.validate_syntax(content, content_type)
        if not syntax_result.passed:
            errors.extend(syntax_result.errors)

        # Style validation
        if validation_rules.get('check_style', True):
            style_result = self.validate_style(content, content_type)
            warnings.extend(style_result.warnings)
            suggestions.extend(style_result.suggestions)

        # Security validation
        if validation_rules.get('check_security', True):
            security_result = self.validate_security(content, content_type)
            if not security_result.passed:
                warnings.extend(security_result.warnings)
            suggestions.extend(security_result.suggestions)

        # Performance validation
        if validation_rules.get('check_performance', True):
            perf_result = self.validate_performance(content, content_type)
            if not perf_result.passed:
                warnings.extend(perf_result.warnings)
            suggestions.extend(perf_result.suggestions)

        # Quality validation
        quality_result = self.validate_quality(content, content_type)
        metrics.update(quality_result.metrics)

        # Calculate overall score
        error_weight = 20
        warning_weight = 5
        score = max(0, 100 - (len(errors) * error_weight) - (len(warnings) * warning_weight))

        result = ValidationResult(
            validation_type=ValidationType.SEMANTIC,
            passed=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

        self.validation_history.append(result)
        return result

    def validate_syntax(self, content: str, content_type: str = "python") -> ValidationResult:
        """
        Validate syntax of content.

        Args:
            content: Content to validate
            content_type: Type of content

        Returns:
            ValidationResult with syntax validation
        """
        errors = []
        warnings = []
        suggestions = []
        metrics = {'syntax_valid': False}

        if content_type == "python":
            try:
                ast.parse(content)
                metrics['syntax_valid'] = True
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                metrics['syntax_valid'] = False
        elif content_type == "javascript":
            # Basic JavaScript syntax check
            if content.count('{') != content.count('}'):
                errors.append("Unbalanced braces detected")
            if content.count('(') != content.count(')'):
                errors.append("Unbalanced parentheses detected")
            if not errors:
                metrics['syntax_valid'] = True
        else:
            metrics['syntax_valid'] = True

        return ValidationResult(
            validation_type=ValidationType.SYNTAX,
            passed=len(errors) == 0,
            score=100 if len(errors) == 0 else 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_style(self, content: str, content_type: str = "python",
                      rules: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate code style.

        Args:
            content: Content to validate
            content_type: Type of content
            rules: Style rules

        Returns:
            ValidationResult with style validation
        """
        warnings = []
        suggestions = []
        errors = []
        metrics = {}

        style_rules = rules or self.validation_rules

        lines = content.split('\n')

        # Check line length
        max_length = style_rules.get('max_line_length', 120)
        long_lines = [(i + 1, len(line)) for i, line in enumerate(lines) if len(line) > max_length]

        if long_lines:
            warnings.append(f"Found {len(long_lines)} lines exceeding {max_length} characters")
            suggestions.append(f"Break long lines to stay under {max_length} characters")
        metrics['long_lines'] = len(long_lines)

        # Check for trailing whitespace
        trailing_whitespace = [(i + 1) for i, line in enumerate(lines) if line != line.rstrip()]
        if trailing_whitespace:
            warnings.append(f"Found {len(trailing_whitespace)} lines with trailing whitespace")
            suggestions.append("Remove trailing whitespace")
        metrics['trailing_whitespace'] = len(trailing_whitespace)

        # Check for tabs vs spaces
        tabs_lines = [(i + 1) for i, line in enumerate(lines) if '\t' in line]
        if tabs_lines:
            warnings.append(f"Found tabs in {len(tabs_lines)} lines")
            suggestions.append("Use spaces instead of tabs")
        metrics['tabs_count'] = len(tabs_lines)

        # Calculate style score
        style_score = max(0, 100 - len(warnings) * 10)

        return ValidationResult(
            validation_type=ValidationType.SEMANTIC,
            passed=len(warnings) == 0,
            score=style_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_security(self, content: str, content_type: str = "python") -> ValidationResult:
        """
        Validate security aspects.

        Args:
            content: Content to validate
            content_type: Type of content

        Returns:
            ValidationResult with security validation
        """
        warnings = []
        suggestions = []
        errors = []
        metrics = {'security_issues': 0}

        # Security patterns
        security_patterns = {
            'hardcoded_passwords': r'(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
            'sql_injection': r'execute\s*\(\s*["\'].*?\+.*?["\']',
            'eval_usage': r'eval\s*\(',
            'shell_injection': r'(os\.system|subprocess\.call)\s*\(\s*["\'].*?\+',
        }

        for category, pattern in security_patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                warnings.append(f"Found {len(matches)} potential {category} issues")
                suggestions.append(f"Review and fix {category} issues")
                metrics['security_issues'] += len(matches)

        # Calculate security score
        security_score = max(0, 100 - metrics['security_issues'] * 15)

        return ValidationResult(
            validation_type=ValidationType.COMPLIANCE,
            passed=metrics['security_issues'] == 0,
            score=security_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_performance(self, content: str, content_type: str = "python") -> ValidationResult:
        """
        Validate performance aspects.

        Args:
            content: Content to validate
            content_type: Type of content

        Returns:
            ValidationResult with performance validation
        """
        warnings = []
        suggestions = []
        errors = []
        metrics = {'performance_issues': 0}

        # Performance patterns
        performance_patterns = {
            'nested_loops': r'for\s+\w+\s+in.*:\s*for\s+\w+\s+in',
            'string_concat_in_loop': r'\w+\s*\+=\s*\w+',
            'global_variables': r'^global\s+\w+',
        }

        for category, pattern in performance_patterns.items():
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if matches:
                warnings.append(f"Found {len(matches)} {category} occurrences")
                suggestions.append(f"Consider optimizing {category}")
                metrics['performance_issues'] += len(matches)

        # Calculate performance score
        performance_score = max(0, 100 - metrics['performance_issues'] * 10)

        return ValidationResult(
            validation_type=ValidationType.PERFORMANCE,
            passed=metrics['performance_issues'] == 0,
            score=performance_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_quality(self, content: str, content_type: str = "python") -> ValidationResult:
        """
        Validate overall code quality.

        Args:
            content: Content to validate
            content_type: Type of content

        Returns:
            ValidationResult with quality metrics
        """
        warnings = []
        suggestions = []
        errors = []
        metrics = {}

        lines = content.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        # Calculate metrics
        metrics['total_lines'] = len(lines)
        metrics['code_lines'] = len(code_lines)
        metrics['comment_ratio'] = (len(lines) - len(code_lines)) / max(1, len(lines))

        # Check for docstrings (Python)
        if content_type == "python":
            has_docstring = bool(re.search(r'""".*?"""', content, re.DOTALL) or
                                re.search(r"'''.*?'''", content, re.DOTALL))
            metrics['has_docstring'] = has_docstring
            if not has_docstring:
                suggestions.append("Consider adding module docstring")

        # Calculate quality score
        quality_score = 70
        if metrics.get('has_docstring', True):
            quality_score += 10
        if metrics['comment_ratio'] > 0.1:
            quality_score += 10
        if metrics['comment_ratio'] > 0.2:
            quality_score += 10

        return ValidationResult(
            validation_type=ValidationType.SEMANTIC,
            passed=True,
            score=quality_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_regression(self, original: str, modified: str,
                           tests: List[Callable]) -> ValidationResult:
        """
        Run regression tests to ensure changes don't break functionality.

        Args:
            original: Original content
            modified: Modified content
            tests: List of test functions to run

        Returns:
            ValidationResult with regression test results
        """
        errors = []
        warnings = []
        suggestions = []
        metrics = {'tests_passed': 0, 'tests_failed': 0, 'tests_total': len(tests)}

        for test_fn in tests:
            try:
                # Test original
                original_result = test_fn(original)

                # Test modified
                modified_result = test_fn(modified)

                # Compare results
                if original_result != modified_result:
                    errors.append(f"Test {test_fn.__name__} failed: results differ")
                    metrics['tests_failed'] += 1
                    suggestions.append(f"Review changes that affect {test_fn.__name__}")
                else:
                    metrics['tests_passed'] += 1

            except (AssertionError, TypeError, ValueError, AttributeError, KeyError) as e:
                errors.append(f"Test {test_fn.__name__} error: {type(e).__name__}: {e}")
                metrics['tests_failed'] += 1

        # Calculate regression score
        if metrics['tests_total'] > 0:
            regression_score = (metrics['tests_passed'] / metrics['tests_total']) * 100
        else:
            regression_score = 100

        return ValidationResult(
            validation_type=ValidationType.REGRESSION,
            passed=metrics['tests_failed'] == 0,
            score=regression_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def validate_compliance(self, content: str, compliance_standards: List[str]) -> ValidationResult:
        """
        Validate against compliance standards.

        Args:
            content: Content to validate
            compliance_standards: List of compliance standards to check

        Returns:
            ValidationResult with compliance check results
        """
        errors = []
        warnings = []
        suggestions = []
        metrics = {'standards_checked': len(compliance_standards), 'standards_passed': 0}

        for standard in compliance_standards:
            if standard.lower() == 'gdpr':
                # Check for GDPR compliance patterns
                if 'personal_data' not in content.lower() and 'consent' not in content.lower():
                    warnings.append("May lack GDPR compliance features")
                    suggestions.append("Review GDPR requirements for personal data handling")
                else:
                    metrics['standards_passed'] += 1

            elif standard.lower() == 'owasp':
                # Check for OWASP compliance
                security_issues = len(re.findall(r'(eval\(|exec\(|os\.system)', content))
                if security_issues > 0:
                    warnings.append(f"Found {security_issues} potential OWASP security issues")
                else:
                    metrics['standards_passed'] += 1

        # Calculate compliance score
        compliance_score = (metrics['standards_passed'] / max(1, metrics['standards_checked'])) * 100

        return ValidationResult(
            validation_type=ValidationType.COMPLIANCE,
            passed=len(errors) == 0,
            score=compliance_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations"""
        if not self.validation_history:
            return {'total_validations': 0}

        summary = {
            'total_validations': len(self.validation_history),
            'by_type': Counter(v.validation_type.value for v in self.validation_history),
            'average_score': sum(v.score for v in self.validation_history) / len(self.validation_history),
            'total_errors': sum(len(v.errors) for v in self.validation_history),
            'total_warnings': sum(len(v.warnings) for v in self.validation_history),
            'passed_count': sum(1 for v in self.validation_history if v.passed)
        }

        return summary

    def verify_with_lean(self, content: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover as a validation tool.
        
        Args:
            content: The content to verify (theorem statement or proof)
            properties: Optional properties for verification
            
        Returns:
            Dict with verification results including:
            - verified: bool
            - formalized: str (Lean code)
            - proof_status: str
            - errors: list
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "error": "Lean verification not available"}
        
        try:
            client = LeanAideClient()
            # Auto-formalize the content
            formalized = client.autoformalize(content)
            # Verify the formalized content
            verification = client.verify(formalized)
            
            return {
                "verified": verification.get("success", False),
                "formalized": formalized,
                "proof_status": verification.get("status", "unknown"),
                "errors": verification.get("errors", []),
                "metadata": properties or {}
            }
        except Exception as e:
            logger.error(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_solution_comprehensive(content: str, content_type: str = "python") -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a solution.

    Args:
        content: Content to analyze
        content_type: Type of content

    Returns:
        Dictionary with all analysis results
    """
    analysis_tools = SolutionAnalysisTools()

    results = {
        'complexity': analysis_tools.analyze_complexity(content, content_type),
        'dependencies': analysis_tools.analyze_dependencies(content, content_type),
        'security': analysis_tools.analyze_security(content, content_type),
        'performance': analysis_tools.analyze_performance(content, content_type),
        'summary': {}
    }

    # Calculate overall summary
    results['summary'] = analysis_tools.get_analysis_summary()

    return results


def create_and_validate_patch(original: str, modified: str,
                             validation_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Create a patch and validate it.

    Args:
        original: Original content
        modified: Modified content
        validation_fn: Optional validation function

    Returns:
        Dictionary with patch and validation results
    """
    patch_tools = PatchGenerationTools()
    validation_tools = ValidationTools()

    # Generate patch
    patch_result = patch_tools.generate_patch(original, modified)

    # Test patch
    test_result = patch_tools.test_patch(original, patch_result.diff, validation_fn)

    # Validate patched content
    validation_result = validation_tools.validate_solution(modified)

    return {
        'patch': patch_result,
        'test': test_result,
        'validation': validation_result
    }


def quick_validate(content: str, content_type: str = "python") -> Dict[str, Any]:
    """
    Quick validation check.

    Args:
        content: Content to validate
        content_type: Type of content

    Returns:
        Quick validation results
    """
    validation_tools = ValidationTools()

    return {
        'syntax': validation_tools.validate_syntax(content, content_type),
        'style': validation_tools.validate_style(content, content_type),
        'security': validation_tools.validate_security(content, content_type)
    }
