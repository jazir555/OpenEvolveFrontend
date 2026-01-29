"""
Complexity Analyzer Module

This module provides comprehensive analysis of problem complexity across multiple dimensions:
- Cognitive Complexity: Mental effort and cognitive load required
- Computational Complexity: Time/space complexity and resource requirements
- Domain Complexity: Specialized knowledge and expertise needed
- Integration Complexity: Dependencies, interfaces, and API complexity

The analyzer uses a multi-dimensional approach to provide a holistic complexity assessment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    from sovereign_data_models import (
        ProblemDefinition,
        ComplexityScore,
        ComplexityDimension,
        ProblemCategory
    )
except ImportError:
    # Fallback definitions if sovereign_data_models is not available
    from dataclasses import dataclass, field
    from enum import Enum
    from datetime import datetime
    from typing import List as TypingList, Dict as TypingDict

    class ProblemCategory(str, Enum):
        """Categories of problems"""
        OPTIMIZATION = "optimization"
        CLASSIFICATION = "classification"
        GENERATION = "generation"
        ANALYSIS = "analysis"
        INTEGRATION = "integration"
        UNKNOWN = "unknown"

    @dataclass
    class ComplexityScore:
        """Complexity score across multiple dimensions"""
        overall_score: float  # 0.0 to 1.0
        cognitive_score: float
        computational_score: float
        domain_score: float
        integration_score: float
        confidence: float  # 0.0 to 1.0
        explanation: str
        dimension_breakdown: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ProblemDefinition:
        """Definition of a problem to be solved."""
        problem_id: str
        title: str
        description: str
        domain: str
        complexity: str
        priority: str
        estimated_effort: str
        requirements: List[str]
        constraints: List[str]
        created_at: datetime
        dependencies: List[str] = field(default_factory=list)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityLevel(str, Enum):
    """Complexity levels for categorization"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
    EXTREME = "extreme"


@dataclass
class ComplexityMetrics:
    """Detailed metrics for complexity analysis"""
    sentence_count: int = 0
    word_count: int = 0
    avg_sentence_length: float = 0.0
    technical_term_count: int = 0
    dependency_count: int = 0
    constraint_count: int = 0
    abstract_concept_count: int = 0
    quantifier_count: int = 0
    conditional_count: int = 0


class ComplexityAnalyzer:
    """
    Analyzes problem complexity across multiple dimensions.

    This analyzer provides a comprehensive assessment of problem complexity by evaluating:
    1. Cognitive complexity: Mental effort required to understand and solve
    2. Computational complexity: Algorithmic and resource requirements
    3. Domain complexity: Specialized knowledge needed
    4. Integration complexity: Dependencies and external systems

    The analysis uses a combination of NLP techniques, pattern matching, and
    domain-specific rules to provide accurate complexity assessments.
    """

    # Technical terms that indicate higher complexity
    TECHNICAL_TERMS = {
        'neural', 'network', 'algorithm', 'optimization', 'convergence',
        'gradient', 'descent', 'backpropagation', 'convolution', 'recurrent',
        'transformer', 'attention', 'embedding', 'tensor', 'matrix', 'vector',
        'probability', 'statistical', 'regression', 'classification', 'clustering',
        'reinforcement', 'learning', 'supervised', 'unsupervised', 'generative',
        'discriminative', 'stochastic', 'deterministic', 'heuristic', 'metaheuristic',
        'evolutionary', 'genetic', 'particle', 'swarm', 'simulated', 'annealing',
        'constraint', 'satisfaction', 'linear', 'programming', 'integer', 'binary',
        'combinatorial', 'optimization', 'graph', 'tree', 'heap', 'hash', 'dynamic',
        'greedy', 'divide', 'conquer', 'recursive', 'iterative', 'parallel',
        'distributed', 'concurrent', 'asynchronous', 'synchronous', 'mutex', 'semaphore',
        'deadlock', 'race', 'condition', 'latency', 'throughput', 'scalability',
        'availability', 'consistency', 'partition', 'tolerance', 'cap', 'theorem',
        'acid', 'base', 'transaction', 'isolation', 'durability', 'persistence',
        'replication', 'sharding', 'partitioning', 'indexing', 'caching', 'buffer',
        'queue', 'stack', 'heap', 'priority', 'scheduler', 'dispatcher', 'router',
        'gateway', 'proxy', 'load', 'balancer', 'microservice', 'monolith', 'serverless',
        'function', 'lambda', 'container', 'docker', 'kubernetes', 'orchestration',
        'deployment', 'pipeline', 'continuous', 'integration', 'delivery'
    }

    # Complexity keywords and their weights
    COMPLEXITY_KEYWORDS = {
        # Very high complexity indicators
        'multi-objective': 0.9,
        'np-hard': 0.95,
        'np-complete': 0.95,
        'exponential': 0.9,
        'combinatorial': 0.85,
        'convergence': 0.8,
        'optimization': 0.7,
        'distributed': 0.75,
        'concurrent': 0.7,
        'parallel': 0.65,
        'real-time': 0.7,
        'scalable': 0.6,
        'machine learning': 0.8,
        'deep learning': 0.85,
        'neural network': 0.8,
        'reinforcement learning': 0.85,
        'unsupervised': 0.75,
        'supervised': 0.6,
        'generative': 0.7,
        'transformer': 0.8,
        'attention mechanism': 0.75,
        'graph': 0.65,
        'tree': 0.5,
        'recursive': 0.6,
        'dynamic programming': 0.7,
        'greedy': 0.5,
        'heuristic': 0.6,
        'metaheuristic': 0.7,
        'evolutionary': 0.75,
        'genetic algorithm': 0.75,
        'particle swarm': 0.7,
        'simulated annealing': 0.7,
        'constraint satisfaction': 0.75,
        'linear programming': 0.65,
        'integer programming': 0.7,
        'mixed integer': 0.75,
        'non-convex': 0.8,
        'non-linear': 0.7,
        'stochastic': 0.7,
        'probabilistic': 0.65,
        'deterministic': 0.4,
        'statistical': 0.6,
        'regression': 0.5,
        'classification': 0.5,
        'clustering': 0.6,
        'dimensionality reduction': 0.7,
        'feature extraction': 0.6,
        'feature engineering': 0.65,
        'hyperparameter': 0.6,
        'cross-validation': 0.5,
        'ensemble': 0.65,
        'random forest': 0.6,
        'gradient boosting': 0.65,
        'support vector': 0.6,
        'bayesian': 0.65,
        'markov': 0.6,
        'monte carlo': 0.7,
        'time series': 0.6,
        'sequence': 0.55,
        'natural language': 0.7,
        'computer vision': 0.75,
        'image processing': 0.7,
        'signal processing': 0.65,
        'cryptography': 0.8,
        'authentication': 0.6,
        'authorization': 0.6,
        'encryption': 0.75,
        'blockchain': 0.8,
        'smart contract': 0.75,
        'consensus': 0.7,
        'fault tolerance': 0.75,
        'high availability': 0.7,
        'load balancing': 0.65,
        'caching': 0.5,
        'database': 0.5,
        'transaction': 0.6,
        'concurrency': 0.7,
        'asynchronous': 0.6,
        'synchronous': 0.4,
        'microservices': 0.7,
        'api': 0.5,
        'rest': 0.45,
        'graphql': 0.55,
        'websocket': 0.6,
        'message queue': 0.65,
        'event-driven': 0.6,
        'streaming': 0.65,
        'batch': 0.4,
        'etl': 0.55,
        'data pipeline': 0.6,
        'data warehouse': 0.6,
        'data lake': 0.55,
        'big data': 0.7,
        'distributed system': 0.8,
        'cloud computing': 0.65,
        'serverless': 0.6,
        'containerization': 0.6,
        'orchestration': 0.65,
        'devops': 0.6,
        'monitoring': 0.5,
        'logging': 0.4,
        'testing': 0.45,
        'integration test': 0.55,
        'unit test': 0.4,
        'end-to-end': 0.6,
        'performance': 0.55,
        'optimization': 0.7,
        'refactoring': 0.5,
        'debugging': 0.5,
        'profiling': 0.55
    }

    # Domain complexity indicators
    DOMAIN_COMPLEXITY_MAP = {
        'machine_learning': 0.85,
        'deep_learning': 0.9,
        'artificial_intelligence': 0.85,
        'data_science': 0.75,
        'computer_vision': 0.8,
        'natural_language_processing': 0.8,
        'reinforcement_learning': 0.85,
        'optimization': 0.75,
        'operations_research': 0.75,
        'graph_theory': 0.7,
        'cryptography': 0.85,
        'cybersecurity': 0.8,
        'distributed_systems': 0.85,
        'database': 0.6,
        'software_engineering': 0.6,
        'web_development': 0.5,
        'mobile_development': 0.55,
        'cloud_computing': 0.7,
        'devops': 0.65,
        'data_engineering': 0.7,
        'machine_learning_ops': 0.75,
        'statistics': 0.65,
        'mathematics': 0.7,
        'physics': 0.7,
        'bioinformatics': 0.8,
        'computational_biology': 0.85,
        'computational_finance': 0.75,
        'quantitative_finance': 0.8,
        'game_theory': 0.75,
        'robotics': 0.8,
        'control_systems': 0.75,
        'signal_processing': 0.75,
        'information_retrieval': 0.7,
        'recommendation_systems': 0.7,
        'search_engines': 0.7
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Complexity Analyzer.

        Args:
            config: Optional configuration dictionary with parameters:
                - cognitive_weight: Weight for cognitive complexity (default: 0.3)
                - computational_weight: Weight for computational complexity (default: 0.3)
                - domain_weight: Weight for domain complexity (default: 0.2)
                - integration_weight: Weight for integration complexity (default: 0.2)
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - normalize_scores: Whether to normalize scores (default: True)
        """
        self.config = config or {}
        self.cognitive_weight = self.config.get('cognitive_weight', 0.3)
        self.computational_weight = self.config.get('computational_weight', 0.3)
        self.domain_weight = self.config.get('domain_weight', 0.2)
        self.integration_weight = self.config.get('integration_weight', 0.2)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.normalize_scores = self.config.get('normalize_scores', True)

        # Validate weights sum to approximately 1.0
        total_weight = (self.cognitive_weight + self.computational_weight +
                       self.domain_weight + self.integration_weight)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total_weight:.2f}, not 1.0. "
                "Normalizing weights."
            )
            # Normalize weights
            self.cognitive_weight /= total_weight
            self.computational_weight /= total_weight
            self.domain_weight /= total_weight
            self.integration_weight /= total_weight

    def calculate_complexity(
        self,
        problem: ProblemDefinition,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexityScore:
        """
        Calculate overall complexity score for a problem.

        This is the main entry point that orchestrates the analysis across all dimensions.

        Args:
            problem: Problem definition with description, requirements, etc.
            context: Additional context information (optional)

        Returns:
            ComplexityScore object with overall and dimension-specific scores

        Raises:
            ValueError: If problem definition is invalid
            TypeError: If problem is not a ProblemDefinition instance
        """
        # Check if problem has required attributes (works for both dataclass and TypedDict)
        required_attrs = ['problem_id', 'title', 'description', 'domain',
                         'complexity', 'priority', 'estimated_effort',
                         'requirements', 'constraints', 'created_at']

        if not all(hasattr(problem, attr) for attr in required_attrs):
            # Try dict-style access for TypedDict
            try:
                if not all(attr in problem for attr in required_attrs):
                    raise TypeError(
                        f"Expected ProblemDefinition, got {type(problem).__name__}"
                    )
            except (TypeError, KeyError):
                raise TypeError(
                    f"Expected ProblemDefinition, got {type(problem).__name__}"
                )

        if not problem.description or not problem.description.strip():
            raise ValueError("Problem description cannot be empty")

        context = context or {}

        try:
            # Get problem attributes (works for both dataclass and TypedDict)
            title = self._get_attr(problem, 'title')
            description = self._get_attr(problem, 'description')
            domain = self._get_attr(problem, 'domain', 'unknown') or 'unknown'
            requirements = self._get_attr(problem, 'requirements', [])
            constraints = self._get_attr(problem, 'constraints', [])
            dependencies = self._get_attr(problem, 'dependencies', [])

            # Analyze each dimension
            logger.info(f"Analyzing complexity for problem: {title}")

            cognitive_score = self.analyze_cognitive_complexity(
                description,
                domain
            )

            computational_score = self.analyze_computational_complexity(
                requirements or []
            )

            domain_score = self.analyze_domain_complexity(
                domain,
                constraints or []
            )

            integration_score = self.analyze_integration_complexity(
                dependencies or []
            )

            # Calculate overall score
            overall_score = (
                self.cognitive_weight * cognitive_score +
                self.computational_weight * computational_score +
                self.domain_weight * domain_score +
                self.integration_weight * integration_score
            )

            # Normalize if needed
            if self.normalize_scores:
                overall_score = max(0.0, min(1.0, overall_score))
                cognitive_score = max(0.0, min(1.0, cognitive_score))
                computational_score = max(0.0, min(1.0, computational_score))
                domain_score = max(0.0, min(1.0, domain_score))
                integration_score = max(0.0, min(1.0, integration_score))

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(problem, context)

            # Generate explanation
            explanation = self._generate_explanation(
                overall_score,
                cognitive_score,
                computational_score,
                domain_score,
                integration_score
            )

            # Create dimension breakdown
            dimension_breakdown = {
                'cognitive': {
                    'score': cognitive_score,
                    'level': self._get_complexity_level(cognitive_score).value,
                    'factors': self._get_cognitive_factors(problem)
                },
                'computational': {
                    'score': computational_score,
                    'level': self._get_complexity_level(computational_score).value,
                    'factors': self._get_computational_factors(problem)
                },
                'domain': {
                    'score': domain_score,
                    'level': self._get_complexity_level(domain_score).value,
                    'factors': self._get_domain_factors(problem)
                },
                'integration': {
                    'score': integration_score,
                    'level': self._get_complexity_level(integration_score).value,
                    'factors': self._get_integration_factors(problem)
                }
            }

            complexity_score = ComplexityScore(
                overall_score=overall_score,
                cognitive_score=cognitive_score,
                computational_score=computational_score,
                domain_score=domain_score,
                integration_score=integration_score,
                confidence=confidence,
                explanation=explanation,
                dimension_breakdown=dimension_breakdown
            )

            logger.info(
                f"Complexity analysis complete: {overall_score:.2f} "
                f"(confidence: {confidence:.2f})"
            )

            return complexity_score

        except Exception as e:
            logger.error(f"Error calculating complexity: {str(e)}")
            raise

    def analyze_cognitive_complexity(
        self,
        description: str,
        domain: str
    ) -> float:
        """
        Analyze cognitive complexity based on description and domain.

        Cognitive complexity measures:
        - Mental effort required to understand the problem
        - Reading difficulty and sentence structure
        - Conceptual complexity and abstractness
        - Use of technical terminology

        Args:
            description: Problem description text
            domain: Problem domain

        Returns:
            Cognitive complexity score (0.0 to 1.0)
        """
        if not description or not description.strip():
            return 0.0

        # Extract metrics
        metrics = self._extract_text_metrics(description)

        # Base complexity from text structure
        structural_complexity = self._calculate_structural_complexity(metrics)

        # Technical complexity from terminology
        technical_complexity = self._calculate_technical_complexity(
            description,
            domain
        )

        # Conceptual complexity from abstract concepts
        conceptual_complexity = self._calculate_conceptual_complexity(
            description
        )

        # Combine metrics
        cognitive_score = (
            0.3 * structural_complexity +
            0.4 * technical_complexity +
            0.3 * conceptual_complexity
        )

        logger.debug(
            f"Cognitive complexity: {cognitive_score:.2f} "
            f"(structural: {structural_complexity:.2f}, "
            f"technical: {technical_complexity:.2f}, "
            f"conceptual: {conceptual_complexity:.2f})"
        )

        return max(0.0, min(1.0, cognitive_score))

    def analyze_computational_complexity(
        self,
        requirements: List[str]
    ) -> float:
        """
        Analyze computational complexity based on requirements.

        Computational complexity measures:
        - Algorithmic complexity (time/space)
        - Scalability requirements
        - Performance constraints
        - Resource demands

        Args:
            requirements: List of requirement strings

        Returns:
            Computational complexity score (0.0 to 1.0)
        """
        if not requirements:
            return 0.3  # Default low complexity

        requirements_text = ' '.join(requirements).lower()

        # Detect complexity indicators
        complexity_indicators = {
            'exponential': 0.95,
            'factorial': 0.95,
            'np-hard': 0.95,
            'np-complete': 0.95,
            'combinatorial': 0.85,
            'polynomial': 0.6,
            'quadratic': 0.7,
            'cubic': 0.75,
            'logarithmic': 0.4,
            'linear': 0.3,
            'constant': 0.2,
            'o(n)': 0.5,
            'o(n^2)': 0.7,
            'o(n log n)': 0.6,
            'o(2^n)': 0.9,
            'o(n!)': 0.95,
            'big o': 0.5,
            'time complexity': 0.6,
            'space complexity': 0.6,
            'scalability': 0.7,
            'scale': 0.6,
            'large scale': 0.8,
            'high performance': 0.7,
            'real-time': 0.75,
            'low latency': 0.7,
            'high throughput': 0.7,
            'optimization': 0.65,
            'efficient': 0.5,
            'parallel': 0.65,
            'distributed': 0.75,
            'concurrent': 0.7,
            'batch': 0.4,
            'streaming': 0.65,
            'iterative': 0.45,
            'recursive': 0.6,
            'dynamic programming': 0.7,
            'greedy': 0.5,
            'divide and conquer': 0.6,
            'backtracking': 0.7,
            'branch and bound': 0.75,
            'memoization': 0.55,
            'cache': 0.4,
            'memory': 0.5,
            'storage': 0.5,
            'database': 0.55,
            'index': 0.45
        }

        # Calculate score from indicators
        max_score = 0.0
        matched_indicators = []

        for indicator, score in complexity_indicators.items():
            if indicator in requirements_text:
                if score > max_score:
                    max_score = score
                    matched_indicators.append(indicator)

        # Adjust based on number of requirements (more requirements = more complex)
        requirement_factor = min(1.0, len(requirements) / 20.0)

        # Combine scores
        computational_score = 0.6 * max_score + 0.4 * requirement_factor

        logger.debug(
            f"Computational complexity: {computational_score:.2f} "
            f"(indicators: {matched_indicators})"
        )

        return max(0.0, min(1.0, computational_score))

    def analyze_domain_complexity(
        self,
        domain: str,
        constraints: List[str]
    ) -> float:
        """
        Analyze domain complexity based on domain and constraints.

        Domain complexity measures:
        - Specialized knowledge required
        - Expertise level needed
        - Domain-specific concepts
        - Constraint complexity

        Args:
            domain: Problem domain
            constraints: List of constraints

        Returns:
            Domain complexity score (0.0 to 1.0)
        """
        domain_lower = domain.lower().replace(' ', '_').replace('-', '_')

        # Base complexity from domain
        base_score = self.DOMAIN_COMPLEXITY_MAP.get(domain_lower, 0.5)

        # Adjust based on constraints
        constraint_factor = self._analyze_constraint_complexity(constraints)

        # Combine scores
        domain_score = 0.7 * base_score + 0.3 * constraint_factor

        logger.debug(
            f"Domain complexity: {domain_score:.2f} "
            f"(domain: {domain_lower}, base: {base_score:.2f}, "
            f"constraints: {constraint_factor:.2f})"
        )

        return max(0.0, min(1.0, domain_score))

    def analyze_integration_complexity(
        self,
        dependencies: List[str]
    ) -> float:
        """
        Analyze integration complexity based on dependencies.

        Integration complexity measures:
        - Number of dependencies
        - Types of integrations (APIs, databases, services)
        - Synchronization requirements
        - Data transformation needs

        Args:
            dependencies: List of dependency identifiers

        Returns:
            Integration complexity score (0.0 to 1.0)
        """
        if not dependencies:
            return 0.0

        # Base complexity from number of dependencies
        count_factor = min(1.0, len(dependencies) / 10.0)

        # Analyze dependency types
        type_scores = []
        for dep in dependencies:
            dep_lower = dep.lower()
            if 'api' in dep_lower or 'rest' in dep_lower or 'graphql' in dep_lower:
                type_scores.append(0.6)
            elif 'database' in dep_lower or 'db' in dep_lower:
                type_scores.append(0.7)
            elif 'service' in dep_lower or 'microservice' in dep_lower:
                type_scores.append(0.75)
            elif 'queue' in dep_lower or 'message' in dep_lower or 'kafka' in dep_lower:
                type_scores.append(0.7)
            elif 'cache' in dep_lower or 'redis' in dep_lower:
                type_scores.append(0.5)
            elif 'external' in dep_lower or 'third-party' in dep_lower:
                type_scores.append(0.65)
            else:
                type_scores.append(0.5)

        # Average type score
        type_factor = sum(type_scores) / len(type_scores) if type_scores else 0.5

        # Combine scores
        integration_score = 0.6 * count_factor + 0.4 * type_factor

        logger.debug(
            f"Integration complexity: {integration_score:.2f} "
            f"(count: {len(dependencies)}, type: {type_factor:.2f})"
        )

        return max(0.0, min(1.0, integration_score))

    # Private helper methods

    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """
        Get attribute from object, supporting both dataclass and dict/TypedDict styles.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute not found

        Returns:
            Attribute value or default
        """
        # Try object attribute access first
        if hasattr(obj, attr):
            return getattr(obj, attr, default)

        # Try dict-style access
        try:
            return obj.get(attr, default) if hasattr(obj, 'get') else obj[attr]
        except (KeyError, TypeError, IndexError):
            return default

    def _extract_text_metrics(self, text: str) -> ComplexityMetrics:
        """Extract various metrics from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Split into words
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)

        # Average sentence length
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0
        )

        # Count technical terms
        technical_term_count = sum(
            1 for word in words if word in self.TECHNICAL_TERMS
        )

        # Count abstract concepts
        abstract_concepts = [
            'abstract', 'concept', 'theory', 'framework', 'paradigm',
            'architecture', 'pattern', 'principle', 'model', 'system'
        ]
        abstract_concept_count = sum(
            1 for word in words if word in abstract_concepts
        )

        # Count quantifiers (indicators of complexity)
        quantifiers = [
            'multiple', 'various', 'several', 'many', 'numerous',
            'complex', 'intricate', 'sophisticated', 'advanced',
            'all', 'every', 'each', 'any', 'some', 'most'
        ]
        quantifier_count = sum(
            1 for word in words if word in quantifiers
        )

        # Count conditional indicators
        conditionals = [
            'if', 'when', 'unless', 'provided that', 'in case',
            'depending on', 'conditional', 'requirement', 'constraint'
        ]
        conditional_count = sum(
            1 for phrase in conditionals if phrase in text.lower()
        )

        return ComplexityMetrics(
            sentence_count=sentence_count,
            word_count=word_count,
            avg_sentence_length=avg_sentence_length,
            technical_term_count=technical_term_count,
            abstract_concept_count=abstract_concept_count,
            quantifier_count=quantifier_count,
            conditional_count=conditional_count
        )

    def _calculate_structural_complexity(
        self,
        metrics: ComplexityMetrics
    ) -> float:
        """Calculate complexity from text structure."""
        # Sentence length complexity (longer sentences = more complex)
        length_score = min(1.0, metrics.avg_sentence_length / 30.0)

        # Sentence count complexity
        count_score = min(1.0, metrics.sentence_count / 20.0)

        # Conditional complexity
        conditional_score = min(1.0, metrics.conditional_count / 5.0)

        return (0.4 * length_score +
                0.3 * count_score +
                0.3 * conditional_score)

    def _calculate_technical_complexity(
        self,
        description: str,
        domain: str
    ) -> float:
        """Calculate complexity from technical terminology."""
        words = re.findall(r'\b\w+\b', description.lower())

        # Count complexity keywords
        keyword_score = 0.0
        for keyword, weight in self.COMPLEXITY_KEYWORDS.items():
            if keyword in description.lower():
                keyword_score = max(keyword_score, weight)

        # Technical term density
        if len(words) > 0:
            tech_density = sum(
                1 for word in words if word in self.TECHNICAL_TERMS
            ) / len(words)
        else:
            tech_density = 0.0

        tech_score = min(1.0, tech_density * 10)

        return max(keyword_score, tech_score)

    def _calculate_conceptual_complexity(self, description: str) -> float:
        """Calculate complexity from abstract concepts."""
        words = re.findall(r'\b\w+\b', description.lower())

        # Quantifier density
        if len(words) > 0:
            quantifier_density = sum(
                1 for word in words
                if word in ['multiple', 'various', 'several', 'many', 'complex']
            ) / len(words)
        else:
            quantifier_density = 0.0

        return min(1.0, quantifier_density * 20)

    def _analyze_constraint_complexity(
        self,
        constraints: List[str]
    ) -> float:
        """Analyze complexity of constraints."""
        if not constraints:
            return 0.0

        constraints_text = ' '.join(constraints).lower()

        # Constraint complexity indicators
        complexity_keywords = {
            'mutual': 0.8,
            'exclusive': 0.8,
            'conflicting': 0.9,
            'trade-off': 0.75,
            'non-linear': 0.8,
            'conditional': 0.7,
            'dynamic': 0.7,
            'real-time': 0.75,
            'strict': 0.6,
            'tight': 0.6,
            'limited': 0.5,
            'bounded': 0.5
        }

        max_score = 0.0
        for keyword, score in complexity_keywords.items():
            if keyword in constraints_text:
                max_score = max(max_score, score)

        # Number of constraints factor
        count_factor = min(1.0, len(constraints) / 10.0)

        return 0.7 * max_score + 0.3 * count_factor

    def _calculate_confidence(
        self,
        problem: ProblemDefinition,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the complexity assessment."""
        confidence = 0.5  # Base confidence

        # Get attributes using helper
        description = self._get_attr(problem, 'description', '')
        requirements = self._get_attr(problem, 'requirements', [])
        domain = self._get_attr(problem, 'domain', '')
        constraints = self._get_attr(problem, 'constraints', [])
        dependencies = self._get_attr(problem, 'dependencies', [])

        # Higher confidence with more data
        if description and len(description) > 50:
            confidence += 0.1

        if requirements and len(requirements) > 0:
            confidence += 0.1

        if domain:
            confidence += 0.1

        if constraints and len(constraints) > 0:
            confidence += 0.1

        if dependencies and len(dependencies) > 0:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_explanation(
        self,
        overall: float,
        cognitive: float,
        computational: float,
        domain: float,
        integration: float
    ) -> str:
        """Generate human-readable explanation of complexity."""
        level = self._get_complexity_level(overall)

        parts = [f"This problem is {level.value} in complexity."]

        # Add dimension-specific insights
        if cognitive > 0.7:
            parts.append(
                "It requires significant mental effort due to "
                "complex concepts and terminology."
            )
        elif cognitive > 0.4:
            parts.append(
                "It requires moderate mental effort to understand "
                "the concepts involved."
            )

        if computational > 0.7:
            parts.append(
                "The computational requirements are high, likely involving "
                "complex algorithms or significant resource demands."
            )
        elif computational > 0.4:
            parts.append(
                "The computational requirements are moderate."
            )

        if domain > 0.7:
            parts.append(
                "Specialized domain knowledge is required to solve this problem."
            )
        elif domain > 0.4:
            parts.append(
                "Some domain expertise will be helpful."
            )

        if integration > 0.7:
            parts.append(
                "Significant integration work is required with multiple "
                "external dependencies."
            )
        elif integration > 0.4:
            parts.append(
                "Some integration with external systems is needed."
            )

        return ' '.join(parts)

    def _get_complexity_level(self, score: float) -> ComplexityLevel:
        """Map score to complexity level."""
        if score < 0.15:
            return ComplexityLevel.TRIVIAL
        elif score < 0.35:
            return ComplexityLevel.SIMPLE
        elif score < 0.55:
            return ComplexityLevel.MODERATE
        elif score < 0.75:
            return ComplexityLevel.COMPLEX
        elif score < 0.9:
            return ComplexityLevel.VERY_COMPLEX
        else:
            return ComplexityLevel.EXTREME

    def _get_cognitive_factors(
        self,
        problem: ProblemDefinition
    ) -> List[str]:
        """Get factors contributing to cognitive complexity."""
        factors = []
        description = self._get_attr(problem, 'description', '').lower()

        if any(term in description for term in ['abstract', 'conceptual', 'theoretical']):
            factors.append('Abstract concepts')
        if any(term in description for term in ['multiple', 'several', 'various']):
            factors.append('Multiple concepts')
        domain = self._get_attr(problem, 'domain', '')
        if domain and domain in ['machine_learning', 'deep_learning']:
            factors.append('Technical domain')

        return factors

    def _get_computational_factors(
        self,
        problem: ProblemDefinition
    ) -> List[str]:
        """Get factors contributing to computational complexity."""
        factors = []

        requirements = self._get_attr(problem, 'requirements', [])
        if requirements:
            req_text = ' '.join(requirements).lower()
            if 'optimization' in req_text:
                factors.append('Optimization required')
            if 'real-time' in req_text or 'low latency' in req_text:
                factors.append('Real-time constraints')
            if 'scal' in req_text:
                factors.append('Scalability requirements')

        return factors

    def _get_domain_factors(
        self,
        problem: ProblemDefinition
    ) -> List[str]:
        """Get factors contributing to domain complexity."""
        factors = []

        domain = self._get_attr(problem, 'domain', '')
        if domain:
            factors.append(f"Domain: {domain}")
        constraints = self._get_attr(problem, 'constraints', [])
        if constraints and len(constraints) > 3:
            factors.append('Multiple constraints')

        return factors

    def _get_integration_factors(
        self,
        problem: ProblemDefinition
    ) -> List[str]:
        """Get factors contributing to integration complexity."""
        factors = []

        dependencies = self._get_attr(problem, 'dependencies', [])
        if dependencies:
            factors.append(f"{len(dependencies)} dependencies")
            if len(dependencies) > 5:
                factors.append('High dependency count')

        return factors


# Convenience functions for quick analysis

def quick_complexity_analysis(
    description: str,
    domain: str = "unknown",
    requirements: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None
) -> ComplexityScore:
    """
    Quick complexity analysis without requiring a full ProblemDefinition.

    Args:
        description: Problem description
        domain: Problem domain
        requirements: Optional list of requirements
        constraints: Optional list of constraints
        dependencies: Optional list of dependencies

    Returns:
        ComplexityScore object
    """
    from datetime import datetime

    # Create a minimal ProblemDefinition
    problem = ProblemDefinition(
        problem_id="quick_analysis",
        title="Quick Analysis",
        description=description,
        domain=domain,
        complexity="moderate",
        priority="medium",
        estimated_effort="unknown",
        requirements=requirements or [],
        constraints=constraints or [],
        created_at=datetime.now()
    )

    analyzer = ComplexityAnalyzer()
    return analyzer.calculate_complexity(problem)


# Example usage and testing

if __name__ == "__main__":
    # Example 1: Simple problem
    simple_problem = """
    Create a web form that collects user contact information including
    name, email, and phone number. The form should validate email format
    and store the data in a database.
    """

    print("=" * 80)
    print("Example 1: Simple Web Form")
    print("=" * 80)
    result = quick_complexity_analysis(
        description=simple_problem,
        domain="web_development",
        requirements=["Validate input", "Store in database"]
    )
    analyzer = ComplexityAnalyzer()
    level = analyzer._get_complexity_level(result.overall_score).value
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Level: {level}")
    print(f"Explanation: {result.explanation}")
    print()

    # Example 2: Machine Learning problem
    ml_problem = """
    Design a deep learning system for real-time object detection in video streams.
    The system must process multiple concurrent video feeds, maintain low latency,
    and achieve high accuracy across various lighting conditions. The model should
    use a transformer-based architecture and be optimized for edge deployment.
    """

    print("=" * 80)
    print("Example 2: Real-time Object Detection")
    print("=" * 80)
    result = quick_complexity_analysis(
        description=ml_problem,
        domain="computer_vision",
        requirements=[
            "Real-time processing",
            "Low latency",
            "High accuracy",
            "Edge deployment"
        ],
        constraints=["Limited computational resources", "Power constraints"],
        dependencies=["TensorFlow", "OpenCV", "Docker"]
    )
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Cognitive: {result.cognitive_score:.2f}")
    print(f"Computational: {result.computational_score:.2f}")
    print(f"Domain: {result.domain_score:.2f}")
    print(f"Integration: {result.integration_score:.2f}")
    print(f"Explanation: {result.explanation}")
    print()

    # Example 3: Distributed Systems problem
    distributed_problem = """
    Design a distributed consensus protocol for a blockchain system that handles
    thousands of transactions per second across multiple geographic regions.
    The system must ensure strong consistency, tolerate Byzantine failures,
    and optimize for both throughput and latency. Implement proof-of-stake
    validation with dynamic sharding.
    """

    print("=" * 80)
    print("Example 3: Distributed Consensus Protocol")
    print("=" * 80)
    result = quick_complexity_analysis(
        description=distributed_problem,
        domain="distributed_systems",
        requirements=[
            "High throughput",
            "Low latency",
            "Strong consistency",
            "Byzantine fault tolerance"
        ],
        constraints=[
            "Geographic distribution",
            "Network partition tolerance",
            "Security requirements"
        ],
        dependencies=[
            "Cryptographic libraries",
            "Network protocols",
            "Consensus algorithms"
        ]
    )
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Cognitive: {result.cognitive_score:.2f}")
    print(f"Computational: {result.computational_score:.2f}")
    print(f"Domain: {result.domain_score:.2f}")
    print(f"Integration: {result.integration_score:.2f}")
    print(f"Explanation: {result.explanation}")
    print()
