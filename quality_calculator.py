"""
Quality Calculator for Sovereign Solutions

This module provides comprehensive quality assessment for solution attempts,
analyzing correctness, completeness, efficiency, and maintainability.

Production-ready features:
- AST-based code analysis
- Requirement validation
- Complexity analysis
- Code smell detection
- Comprehensive error handling
- Full type hints
- Unit tests included
"""

import ast
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SolutionQualityMetrics:
    """Quality metrics for solution attempts.

    Attributes:
        correctness: Degree to which solution meets requirements (0.0-1.0)
        completeness: Extent to which all components are present (0.0-1.0)
        efficiency: Resource usage and performance quality (0.0-1.0)
        maintainability: Code quality and readability (0.0-1.0)
    """
    correctness: float
    completeness: float
    efficiency: float
    maintainability: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "efficiency": self.efficiency,
            "maintainability": self.maintainability
        }

    def __post_init__(self):
        """Validate metric ranges."""
        for metric_name, value in self.to_dict().items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0, got {value}")


@dataclass
class CodeQualityAnalysis:
    """Detailed code quality analysis results."""
    complexity_score: float
    documentation_score: float
    naming_score: float
    structure_score: float
    code_smells: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequirementMatch:
    """Result of requirement matching."""
    requirement: str
    matched: bool
    confidence: float
    evidence: str
    line_numbers: List[int] = field(default_factory=list)


class QualityCalculator:
    """Calculator for solution quality metrics.

    This class provides comprehensive quality assessment capabilities including:
    - Requirement validation
    - Code quality analysis using AST
    - Complexity metrics
    - Code smell detection
    - Overall scoring with customizable weights
    """

    # Default weights for overall score calculation
    DEFAULT_WEIGHTS = {
        "correctness": 0.35,
        "completeness": 0.25,
        "efficiency": 0.20,
        "maintainability": 0.20
    }

    # Code smell patterns
    CODE_SMELL_PATTERNS = {
        "long_function": (r"def \w+\([^)]*\):\s*\"\"\"[^\"]{200,}\"\"\"", "Function too long"),
        "magic_numbers": (r"\b(?!0|1\b)\d{2,}\b", "Magic number detected"),
        "deep_nesting": (r"\t{16,}| {32,}", "Deep nesting detected"),
        "global_variables": (r"^global\s+\w+", "Global variable usage"),
        "bare_except": (r"except\s*:", "Bare except clause"),
        "print_debugging": (r"print\s*\(", "Print statement (use logger)"),
    }

    # Good practice patterns
    GOOD_PRACTICE_PATTERNS = {
        "docstring": (r'""".*?"""', "Documentation present"),
        "type_hints": (r":\s*(str|int|float|bool|List|Dict|Optional|Tuple|Set)", "Type hints present"),
        "error_handling": (r"\btry:\s*.*?\s*except", "Error handling present"),
        "logging": (r"\blogger\.\w+\(", "Logging present"),
        "context_manager": (r"\bwith\s+\w+\s+as\s+", "Context manager usage"),
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize the quality calculator.

        Args:
            weights: Custom weights for overall score calculation.
                    If None, uses DEFAULT_WEIGHTS.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

        # Cache for AST analysis results
        self._ast_cache: Dict[str, ast.AST] = {}

    def _validate_weights(self) -> None:
        """Validate weight configuration."""
        if not abs(sum(self.weights.values()) - 1.0) < 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")

        required_keys = {"correctness", "completeness", "efficiency", "maintainability"}
        missing_keys = required_keys - set(self.weights.keys())
        if missing_keys:
            raise ValueError(f"Missing required weight keys: {missing_keys}")

    def calculate_quality(
        self,
        solution: Any,  # SolutionAttempt from sovereign_data_models
        requirements: List[str]
    ) -> SolutionQualityMetrics:
        """Calculate comprehensive quality metrics for a solution.

        Args:
            solution: SolutionAttempt object with solution code/content
            requirements: List of requirements the solution should meet

        Returns:
            SolutionQualityMetrics object with all four dimensions

        Raises:
            ValueError: If solution or requirements are invalid
        """
        if not requirements:
            raise ValueError("Requirements list cannot be empty")

        # Extract solution content
        solution_content = self._extract_solution_content(solution)
        if not solution_content:
            logger.warning("Empty solution content provided")
            return SolutionQualityMetrics(0.0, 0.0, 0.0, 0.0)

        try:
            # Calculate each dimension
            correctness = self.calculate_correctness(solution, requirements)
            completeness = self.calculate_completeness(solution, requirements)
            efficiency = self.calculate_efficiency(solution)
            maintainability = self.calculate_maintainability(solution)

            metrics = SolutionQualityMetrics(
                correctness=correctness,
                completeness=completeness,
                efficiency=efficiency,
                maintainability=maintainability
            )

            logger.info(
                f"Quality calculated: correct={correctness:.2f}, "
                f"complete={completeness:.2f}, efficient={efficiency:.2f}, "
                f"maintainable={maintainability:.2f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating quality: {e}", exc_info=True)
            raise

    def calculate_correctness(
        self,
        solution: Any,
        requirements: List[str]
    ) -> float:
        """Calculate correctness score based on requirement satisfaction.

        Correctness measures how well the solution addresses the stated requirements.

        Args:
            solution: SolutionAttempt object
            requirements: List of requirements to validate against

        Returns:
            Correctness score between 0.0 and 1.0
        """
        solution_content = self._extract_solution_content(solution)
        if not solution_content:
            return 0.0

        requirement_matches = self._match_requirements(solution_content, requirements)

        if not requirement_matches:
            return 0.0

        # Calculate weighted correctness based on match confidence
        total_confidence = sum(match.confidence for match in requirement_matches)
        max_confidence = len(requirement_matches)
        correctness = total_confidence / max_confidence if max_confidence > 0 else 0.0

        logger.debug(f"Correctness: {correctness:.2f} ({len(requirement_matches)}/{len(requirements)} requirements matched)")

        return min(1.0, max(0.0, correctness))

    def calculate_completeness(
        self,
        solution: Any,
        requirements: List[str]
    ) -> float:
        """Calculate completeness score based on component presence.

        Completeness checks if all necessary components are present and documented.

        Args:
            solution: SolutionAttempt object
            requirements: List of requirements defining completeness

        Returns:
            Completeness score between 0.0 and 1.0
        """
        solution_content = self._extract_solution_content(solution)
        if not solution_content:
            return 0.0

        completeness_factors = []

        # 1. Function/class definition presence
        try:
            tree = self._parse_ast(solution_content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            # Expected at least one function or class for non-trivial solutions
            if functions or classes:
                completeness_factors.append(1.0)
            elif len(solution_content.strip()) > 100:  # Non-trivial content but no structure
                completeness_factors.append(0.3)
            else:
                completeness_factors.append(0.0)

        except Exception:
            # If AST parsing fails, check for any structured content
            completeness_factors.append(0.5)

        # 2. Documentation coverage
        doc_score = self._calculate_documentation_coverage(solution_content)
        completeness_factors.append(doc_score)

        # 3. Requirement coverage
        requirement_matches = self._match_requirements(solution_content, requirements)
        requirement_coverage = len(requirement_matches) / len(requirements) if requirements else 1.0
        completeness_factors.append(requirement_coverage)

        # 4. Import/module organization
        import_score = self._calculate_import_score(solution_content)
        completeness_factors.append(import_score)

        # Average all factors
        completeness = sum(completeness_factors) / len(completeness_factors)

        logger.debug(f"Completeness: {completeness:.2f}")

        return min(1.0, max(0.0, completeness))

    def calculate_efficiency(self, solution: Any) -> float:
        """Calculate efficiency score based on resource usage patterns.

        Efficiency analyzes time/space complexity patterns and resource usage.

        Args:
            solution: SolutionAttempt object

        Returns:
            Efficiency score between 0.0 and 1.0
        """
        solution_content = self._extract_solution_content(solution)
        if not solution_content:
            return 0.0

        try:
            tree = self._parse_ast(solution_content)
            analysis = self._analyze_complexity(solution_content, tree)

            # Calculate efficiency score based on complexity analysis
            efficiency_factors = []

            # 1. Time complexity score (prefer O(n) or better)
            time_score = analysis.complexity_score
            efficiency_factors.append(time_score)

            # 2. Space complexity patterns
            space_score = self._calculate_space_efficiency(tree)
            efficiency_factors.append(space_score)

            # 3. Algorithmic efficiency
            algo_score = self._calculate_algorithmic_efficiency(solution_content)
            efficiency_factors.append(algo_score)

            # 4. Resource management (context managers, proper cleanup)
            resource_score = self._calculate_resource_management(solution_content)
            efficiency_factors.append(resource_score)

            efficiency = sum(efficiency_factors) / len(efficiency_factors)

            logger.debug(f"Efficiency: {efficiency:.2f}")

            return min(1.0, max(0.0, efficiency))

        except Exception as e:
            logger.error(f"Error calculating efficiency: {e}")
            return 0.5  # Return neutral score on error

    def calculate_maintainability(self, solution: Any) -> float:
        """Calculate maintainability score based on code quality.

        Maintainability analyzes code structure, naming, and documentation.

        Args:
            solution: SolutionAttempt object

        Returns:
            Maintainability score between 0.0 and 1.0
        """
        solution_content = self._extract_solution_content(solution)
        if not solution_content:
            return 0.0

        try:
            analysis = self.analyze_code_quality(solution_content)

            # Combine various maintainability aspects
            maintainability_factors = [
                analysis.documentation_score,
                analysis.naming_score,
                analysis.structure_score,
            ]

            # Penalty for code smells
            smell_penalty = min(0.3, len(analysis.code_smells) * 0.05)

            maintainability = (sum(maintainability_factors) / len(maintainability_factors)) - smell_penalty

            logger.debug(f"Maintainability: {maintainability:.2f} (penalty: {smell_penalty:.2f})")

            return min(1.0, max(0.0, maintainability))

        except Exception as e:
            logger.error(f"Error calculating maintainability: {e}")
            return 0.5  # Return neutral score on error

    def calculate_overall_score(
        self,
        metrics: SolutionQualityMetrics,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall quality score from component metrics.

        Args:
            metrics: SolutionQualityMetrics object
            weights: Optional custom weights (uses instance weights if None)

        Returns:
            Overall score between 0.0 and 1.0
        """
        if weights:
            # Validate custom weights
            if abs(sum(weights.values()) - 1.0) >= 0.01:
                raise ValueError(f"Custom weights must sum to 1.0, got {sum(weights.values())}")
            used_weights = weights
        else:
            used_weights = self.weights

        overall = (
            metrics.correctness * used_weights["correctness"] +
            metrics.completeness * used_weights["completeness"] +
            metrics.efficiency * used_weights["efficiency"] +
            metrics.maintainability * used_weights["maintainability"]
        )

        logger.info(f"Overall quality score: {overall:.2f}")

        return min(1.0, max(0.0, overall))

    def analyze_code_quality(self, content: str) -> CodeQualityAnalysis:
        """Perform comprehensive code quality analysis.

        Args:
            content: Source code to analyze

        Returns:
            CodeQualityAnalysis with detailed metrics
        """
        try:
            tree = self._parse_ast(content)
        except Exception as e:
            logger.warning(f"AST parsing failed, falling back to pattern analysis: {e}")
            return self._fallback_code_analysis(content)

        # Analyze various aspects
        complexity = self._calculate_cyclomatic_complexity(tree)
        documentation = self._calculate_documentation_coverage(content)
        naming = self._calculate_naming_score(tree)
        structure = self._calculate_structure_score(tree)
        code_smells = self.detect_code_smells(content)

        # Generate suggestions
        suggestions = self._generate_improvement_suggestions(
            complexity, documentation, naming, structure, code_smells
        )

        return CodeQualityAnalysis(
            complexity_score=complexity,
            documentation_score=documentation,
            naming_score=naming,
            structure_score=structure,
            code_smells=code_smells,
            suggestions=suggestions,
            metrics={
                "cyclomatic_complexity": complexity,
                "doc_coverage": documentation,
                "naming_convention_score": naming,
                "structure_quality": structure
            }
        )

    def detect_code_smells(self, content: str) -> List[str]:
        """Detect code smells using pattern matching and AST analysis.

        Args:
            content: Source code to analyze

        Returns:
            List of detected code smell descriptions
        """
        smells = []

        # Pattern-based detection
        for smell_name, (pattern, description) in self.CODE_SMELL_PATTERNS.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                count = len(matches)
                smells.append(f"{description} (found {count} time{'s' if count > 1 else ''})")

        # AST-based detection
        try:
            tree = self._parse_ast(content)
            ast_smells = self._detect_ast_code_smells(tree, content)
            smells.extend(ast_smells)
        except Exception as e:
            logger.debug(f"AST code smell detection failed: {e}")

        return list(set(smells))  # Remove duplicates

    # Private helper methods

    def _extract_solution_content(self, solution: Any) -> str:
        """Extract solution content from various solution object types.

        Args:
            solution: Solution object (SolutionAttempt, dict, or string)

        Returns:
            Extracted content as string
        """
        if isinstance(solution, str):
            return solution

        # Handle Pydantic models (crewai_state_management.SolutionAttempt)
        if hasattr(solution, 'solution_content'):
            return solution.solution_content

        # Handle dataclass objects with 'solution' field
        if hasattr(solution, 'solution'):
            return solution.solution

        # Handle dict-like objects
        if isinstance(solution, dict):
            return solution.get('solution', '') or solution.get('solution_content', '') or solution.get('content', '') or solution.get('code', '')

        # Handle objects with __dict__
        if hasattr(solution, '__dict__'):
            return str(solution)

        return str(solution)

    def _parse_ast(self, content: str) -> ast.AST:
        """Parse content into AST with caching.

        Args:
            content: Source code content

        Returns:
            AST tree

        Raises:
            SyntaxError: If content has invalid Python syntax
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash not in self._ast_cache:
            try:
                self._ast_cache[content_hash] = ast.parse(content)
            except SyntaxError as e:
                logger.error(f"Failed to parse AST: {e}")
                raise

        return self._ast_cache[content_hash]

    def _match_requirements(
        self,
        content: str,
        requirements: List[str]
    ) -> List[RequirementMatch]:
        """Match solution content against requirements.

        Args:
            content: Solution content
            requirements: List of requirements to match

        Returns:
            List of RequirementMatch objects
        """
        matches = []
        content_lower = content.lower()

        for req in requirements:
            req_lower = req.lower()

            # Extract key terms from requirement
            terms = self._extract_requirement_terms(req_lower)

            # Check for term presence in content
            matched_terms = [term for term in terms if term in content_lower]

            if matched_terms:
                confidence = len(matched_terms) / len(terms) if terms else 0.0

                # Find evidence (lines containing matched terms)
                lines = content.split('\n')
                line_numbers = []
                evidence_lines = []

                for i, line in enumerate(lines, 1):
                    if any(term in line.lower() for term in matched_terms):
                        line_numbers.append(i)
                        if len(evidence_lines) < 3:  # Limit evidence lines
                            evidence_lines.append(line.strip())

                evidence = ' | '.join(evidence_lines) if evidence_lines else "Terms matched but no specific lines"

                matches.append(RequirementMatch(
                    requirement=req,
                    matched=confidence > 0.3,  # Threshold for considering it matched
                    confidence=confidence,
                    evidence=evidence,
                    line_numbers=line_numbers
                ))
            else:
                matches.append(RequirementMatch(
                    requirement=req,
                    matched=False,
                    confidence=0.0,
                    evidence="No matching terms found"
                ))

        return matches

    def _extract_requirement_terms(self, requirement: str) -> List[str]:
        """Extract meaningful search terms from requirement.

        Args:
            requirement: Requirement text

        Returns:
            List of search terms
        """
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Extract words
        words = re.findall(r'\b\w+\b', requirement.lower())

        # Filter out stopwords and short words
        terms = [w for w in words if w not in stopwords and len(w) > 2]

        # Add technical terms (camelCase, snake_case, etc.)
        technical_terms = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-z]+_[a-z_]+\b', requirement)
        terms.extend(technical_terms)

        return list(set(terms))

    def _calculate_documentation_coverage(self, content: str) -> float:
        """Calculate documentation coverage score.

        Args:
            content: Source code content

        Returns:
            Documentation score between 0.0 and 1.0
        """
        if not content or len(content.strip()) < 10:
            return 0.0

        try:
            tree = self._parse_ast(content)

            # Count documented vs undocumented items
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            documented_functions = sum(1 for f in functions if ast.get_docstring(f))
            documented_classes = sum(1 for c in classes if ast.get_docstring(c))

            total_items = len(functions) + len(classes)
            if total_items == 0:
                return 0.5  # Neutral if no functions/classes

            documented_items = documented_functions + documented_classes
            coverage = documented_items / total_items

            # Bonus for module docstring
            if ast.get_docstring(tree):
                coverage = min(1.0, coverage + 0.1)

            return coverage

        except Exception:
            # Fallback: check for docstring patterns
            docstring_pattern = r'""".*?"""'
            matches = re.findall(docstring_pattern, content, re.DOTALL)
            return min(1.0, len(matches) * 0.3)

    def _calculate_import_score(self, content: str) -> float:
        """Calculate import organization score.

        Args:
            content: Source code content

        Returns:
            Import score between 0.0 and 1.0
        """
        try:
            tree = self._parse_ast(content)
            imports = [node for node in ast.walk(tree)
                      if isinstance(node, (ast.Import, ast.ImportFrom))]

            if not imports:
                return 0.5  # Neutral if no imports

            # Check for standard library vs third party organization
            # This is a simplified check
            score = 0.5

            # Bonus for explicit imports (not import *)
            for imp in imports:
                if isinstance(imp, ast.ImportFrom):
                    if imp.module and not any(alias.name == '*' for alias in imp.names):
                        score += 0.1

            return min(1.0, score)

        except Exception:
            return 0.5

    def _analyze_complexity(self, content: str, tree: ast.AST) -> CodeQualityAnalysis:
        """Analyze code complexity.

        Args:
            content: Source code
            tree: AST tree

        Returns:
            CodeQualityAnalysis with complexity metrics
        """
        complexity = self._calculate_cyclomatic_complexity(tree)

        # Analyze nested loops and conditionals
        nesting_score = self._calculate_nesting_score(tree)

        # Overall complexity score (inverse of complexity)
        complexity_score = max(0.0, 1.0 - (complexity / 20.0))  # Normalize around 20

        return CodeQualityAnalysis(
            complexity_score=complexity_score,
            documentation_score=0.0,
            naming_score=0.0,
            structure_score=nesting_score,
            metrics={"cyclomatic_complexity": complexity}
        )

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity.

        Args:
            tree: AST tree

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return float(complexity)

    def _calculate_nesting_score(self, tree: ast.AST) -> float:
        """Calculate nesting depth score.

        Args:
            tree: AST tree

        Returns:
            Nesting score (higher is better)
        """
        max_depth = 0

        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    calculate_depth(child, current_depth + 1)
                else:
                    calculate_depth(child, current_depth)

        calculate_depth(tree)

        # Score decreases with nesting depth
        # Depth 0-2: 1.0, Depth 3-4: 0.7, Depth 5+: 0.3
        if max_depth <= 2:
            return 1.0
        elif max_depth <= 4:
            return 0.7
        else:
            return 0.3

    def _calculate_space_efficiency(self, tree: ast.AST) -> float:
        """Calculate space efficiency score.

        Args:
            tree: AST tree

        Returns:
            Space efficiency score
        """
        # Check for memory-intensive patterns
        score = 1.0

        for node in ast.walk(tree):
            # List comprehensions are generally efficient
            if isinstance(node, ast.ListComp):
                score = min(1.0, score + 0.05)

            # Generator expressions are very efficient
            if isinstance(node, ast.GeneratorExp):
                score = min(1.0, score + 0.1)

            # Deep copies can be expensive
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'deepcopy':
                        score = max(0.5, score - 0.2)

        return score

    def _calculate_algorithmic_efficiency(self, content: str) -> float:
        """Calculate algorithmic efficiency based on patterns.

        Args:
            content: Source code

        Returns:
            Algorithmic efficiency score
        """
        score = 0.5

        # Good patterns
        good_patterns = {
            r"\bset\s*\(": 0.1,  # Using sets for O(1) lookup
            r"\bdict\s*\(": 0.1,  # Using dicts for O(1) lookup
            r"\.get\s*\(": 0.05,  # Using .get() for safe dict access
            r"\.join\s*\(": 0.1,  # Efficient string joining
        }

        # Bad patterns
        bad_patterns = {
            r"\bfor\s+\w+\s+in\s+.*:\s*if\s+\w+\s+==.*:": -0.2,  # O(n^2) nested
            r"\.append\s*\(.+\)\s*for\s+": -0.1,  # Potential inefficiency
        }

        for pattern, bonus in good_patterns.items():
            if re.search(pattern, content):
                score = min(1.0, score + bonus)

        for pattern, penalty in bad_patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                score = max(0.0, score + penalty)

        return score

    def _calculate_resource_management(self, content: str) -> float:
        """Calculate resource management score.

        Args:
            content: Source code

        Returns:
            Resource management score
        """
        score = 0.5

        # Check for context managers
        with_count = len(re.findall(r"\bwith\s+", content))
        score = min(1.0, score + (with_count * 0.15))

        # Check for explicit cleanup in __exit__ or __del__
        if re.search(r"def __(exit|del)__", content):
            score = min(1.0, score + 0.2)

        # Check for proper exception handling
        try_count = len(re.findall(r"\btry:\s*", content))
        score = min(1.0, score + (try_count * 0.05))

        return score

    def _calculate_naming_score(self, tree: ast.AST) -> float:
        """Calculate naming convention score.

        Args:
            tree: AST tree

        Returns:
            Naming score
        """
        score = 0.5
        total_names = 0
        good_names = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                # Function names should be snake_case
                if re.match(r'^[a-z][a-z0-9_]*$', node.name):
                    good_names += 1

                # Check parameter names
                for arg in node.args.args:
                    total_names += 1
                    if re.match(r'^[a-z][a-z0-9_]*$', arg.arg):
                        good_names += 1

            elif isinstance(node, ast.ClassDef):
                total_names += 1
                # Class names should be CamelCase
                if re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    good_names += 1

        if total_names > 0:
            score = good_names / total_names

        return score

    def _calculate_structure_score(self, tree: ast.AST) -> float:
        """Calculate code structure score.

        Args:
            tree: AST tree

        Returns:
            Structure score
        """
        score = 0.5

        # Check for appropriate function length
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if functions:
            avg_length = sum(len(ast.dump(f)) for f in functions) / len(functions)
            # Moderate length is good
            if 100 < avg_length < 1000:
                score = 1.0
            elif avg_length <= 100 or avg_length >= 1000:
                score = 0.7

        # Check for class organization
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if classes:
            # Having classes with methods is good
            methods_per_class = sum(
                len([n for n in ast.walk(c) if isinstance(n, ast.FunctionDef)])
                for c in classes
            ) / len(classes)

            if 2 <= methods_per_class <= 10:
                score = min(1.0, score + 0.2)

        return score

    def _detect_ast_code_smells(self, tree: ast.AST, content: str) -> List[str]:
        """Detect code smells using AST analysis.

        Args:
            tree: AST tree
            content: Source code

        Returns:
            List of code smell descriptions
        """
        smells = []

        for node in ast.walk(tree):
            # Long function
            if isinstance(node, ast.FunctionDef):
                func_length = len(ast.dump(node))
                if func_length > 2000:
                    smells.append(f"Function '{node.name}' is too long ({func_length} chars)")

                # Too many parameters
                if len(node.args.args) > 7:
                    smells.append(f"Function '{node.name}' has too many parameters ({len(node.args.args)})")

            # Empty class
            if isinstance(node, ast.ClassDef):
                methods = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                if len(methods) == 0:
                    smells.append(f"Class '{node.name}' has no methods")

        return smells

    def _generate_improvement_suggestions(
        self,
        complexity: float,
        documentation: float,
        naming: float,
        structure: float,
        code_smells: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on analysis.

        Args:
            complexity: Complexity score
            documentation: Documentation score
            naming: Naming score
            structure: Structure score
            code_smells: List of code smells

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if complexity < 0.5:
            suggestions.append("Consider breaking down complex functions into smaller units")

        if documentation < 0.5:
            suggestions.append("Add docstrings to functions and classes")

        if naming < 0.5:
            suggestions.append("Follow PEP 8 naming conventions (snake_case for functions, CamelCase for classes)")

        if structure < 0.5:
            suggestions.append("Improve code structure by organizing related functionality")

        if code_smells:
            suggestions.append(f"Address {len(code_smells)} code smell(s): {', '.join(code_smells[:3])}")

        return suggestions

    def _fallback_code_analysis(self, content: str) -> CodeQualityAnalysis:
        """Fallback code analysis when AST parsing fails.

        Args:
            content: Source code

        Returns:
            Basic CodeQualityAnalysis
        """
        # Use pattern-based analysis
        doc_score = self._calculate_documentation_coverage(content)
        code_smells = self.detect_code_smells(content)

        # Calculate basic metrics
        line_count = len(content.split('\n'))
        function_count = len(re.findall(r'\bdef\s+\w+', content))

        return CodeQualityAnalysis(
            complexity_score=0.5,
            documentation_score=doc_score,
            naming_score=0.5,
            structure_score=0.5,
            code_smells=code_smells,
            suggestions=["Consider improving code structure for better parsing"],
            metrics={"line_count": line_count, "function_count": function_count}
        )


# Singleton instance
_default_calculator = None


def get_quality_calculator(weights: Optional[Dict[str, float]] = None) -> QualityCalculator:
    """Get a singleton quality calculator instance.

    Args:
        weights: Optional custom weights

    Returns:
        QualityCalculator instance
    """
    global _default_calculator
    if _default_calculator is None or weights is not None:
        _default_calculator = QualityCalculator(weights)
    return _default_calculator


# Convenience functions
def calculate_quality(
    solution: Any,
    requirements: List[str],
    weights: Optional[Dict[str, float]] = None
) -> SolutionQualityMetrics:
    """Convenience function to calculate quality metrics.

    Args:
        solution: SolutionAttempt object
        requirements: List of requirements
        weights: Optional custom weights

    Returns:
        SolutionQualityMetrics
    """
    calculator = get_quality_calculator(weights)
    return calculator.calculate_quality(solution, requirements)


def analyze_code_quality(content: str) -> CodeQualityAnalysis:
    """Convenience function to analyze code quality.

    Args:
        content: Source code

    Returns:
        CodeQualityAnalysis
    """
    calculator = get_quality_calculator()
    return calculator.analyze_code_quality(content)


def detect_code_smells(content: str) -> List[str]:
    """Convenience function to detect code smells.

    Args:
        content: Source code

    Returns:
        List of code smell descriptions
    """
    calculator = get_quality_calculator()
    return calculator.detect_code_smells(content)


# Unit tests
if __name__ == "__main__":
    import unittest
    from dataclasses import dataclass

    @dataclass
    class MockSolutionAttempt:
        """Mock SolutionAttempt for testing."""
        id: str
        problem_id: str
        solution: str
        score: float
        timestamp: datetime

    class TestQualityCalculator(unittest.TestCase):
        """Unit tests for QualityCalculator."""

        def setUp(self):
            """Set up test fixtures."""
            self.calculator = QualityCalculator()
            self.sample_requirements = [
                "Implement a function to calculate fibonacci numbers",
                "Use dynamic programming for efficiency",
                "Include error handling for invalid inputs"
            ]

            self.good_solution = MockSolutionAttempt(
                id="test1",
                problem_id="prob1",
                solution='''"""
Fibonacci number calculator.

This module provides efficient fibonacci calculation using dynamic programming.
"""

def fibonacci(n: int) -> int:
    """Calculate the nth fibonacci number.

    Args:
        n: The position in fibonacci sequence (must be non-negative)

    Returns:
        The nth fibonacci number

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    # Dynamic programming approach
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b


if __name__ == "__main__":
    print(fibonacci(10))
''',
                score=0.9,
                timestamp=datetime.now()
            )

            self.poor_solution = MockSolutionAttempt(
                id="test2",
                problem_id="prob1",
                solution='''def fib(n):
    if n<=1:
        return n
    return fib(n-1)+fib(n-2)
''',
                score=0.3,
                timestamp=datetime.now()
            )

        def test_calculate_correctness(self):
            """Test correctness calculation."""
            good_correctness = self.calculator.calculate_correctness(
                self.good_solution, self.sample_requirements
            )
            # Good solution should have measurable correctness
            self.assertGreater(good_correctness, 0.0)

            poor_correctness = self.calculator.calculate_correctness(
                self.poor_solution, self.sample_requirements
            )
            # Both should produce valid scores
            self.assertGreaterEqual(poor_correctness, 0.0)
            self.assertLessEqual(good_correctness, 1.0)

        def test_calculate_completeness(self):
            """Test completeness calculation."""
            good_completeness = self.calculator.calculate_completeness(
                self.good_solution, self.sample_requirements
            )
            self.assertGreater(good_completeness, 0.5)

        def test_calculate_efficiency(self):
            """Test efficiency calculation."""
            good_efficiency = self.calculator.calculate_efficiency(self.good_solution)
            poor_efficiency = self.calculator.calculate_efficiency(self.poor_solution)

            # Both should produce valid scores in range [0, 1]
            self.assertGreaterEqual(good_efficiency, 0.0)
            self.assertLessEqual(good_efficiency, 1.0)
            self.assertGreaterEqual(poor_efficiency, 0.0)
            self.assertLessEqual(poor_efficiency, 1.0)

            # Good solution should have reasonable efficiency
            self.assertGreater(good_efficiency, 0.3)

        def test_calculate_maintainability(self):
            """Test maintainability calculation."""
            good_maintainability = self.calculator.calculate_maintainability(self.good_solution)
            poor_maintainability = self.calculator.calculate_maintainability(self.poor_solution)

            self.assertGreater(good_maintainability, poor_maintainability)

        def test_calculate_overall_score(self):
            """Test overall score calculation."""
            metrics = SolutionQualityMetrics(
                correctness=0.8,
                completeness=0.7,
                efficiency=0.9,
                maintainability=0.85
            )

            overall = self.calculator.calculate_overall_score(metrics)
            self.assertGreaterEqual(overall, 0.0)
            self.assertLessEqual(overall, 1.0)

        def test_analyze_code_quality(self):
            """Test code quality analysis."""
            analysis = self.calculator.analyze_code_quality(self.good_solution.solution)

            self.assertIsInstance(analysis, CodeQualityAnalysis)
            self.assertGreater(analysis.documentation_score, 0.5)
            self.assertGreater(analysis.naming_score, 0.5)

        def test_detect_code_smells(self):
            """Test code smell detection."""
            smells = self.calculator.detect_code_smells(self.poor_solution.solution)
            self.assertIsInstance(smells, list)

        def test_empty_solution(self):
            """Test handling of empty solutions."""
            empty_solution = MockSolutionAttempt(
                id="test3",
                problem_id="prob1",
                solution="",
                score=0.0,
                timestamp=datetime.now()
            )

            metrics = self.calculator.calculate_quality(
                empty_solution, self.sample_requirements
            )

            self.assertEqual(metrics.correctness, 0.0)
            self.assertEqual(metrics.completeness, 0.0)

        def test_requirement_matching(self):
            """Test requirement matching."""
            matches = self.calculator._match_requirements(
                self.good_solution.solution,
                self.sample_requirements
            )

            self.assertEqual(len(matches), len(self.sample_requirements))

            # At least some requirements should be matched
            matched_count = sum(1 for m in matches if m.matched)
            self.assertGreater(matched_count, 0)

        def test_invalid_weights(self):
            """Test weight validation."""
            with self.assertRaises(ValueError):
                QualityCalculator(weights={"correctness": 0.5, "completeness": 0.5})

        def test_metrics_validation(self):
            """Test metrics range validation."""
            with self.assertRaises(ValueError):
                SolutionQualityMetrics(1.5, 0.5, 0.5, 0.5)

            with self.assertRaises(ValueError):
                SolutionQualityMetrics(-0.1, 0.5, 0.5, 0.5)

    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)
