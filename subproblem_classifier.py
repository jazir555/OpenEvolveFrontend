"""
Sub-Problem Classifier Module

This module provides intelligent classification of sub-problems (SubProblem objects)
based on their descriptions, using keyword matching, NLP patterns, and confidence scoring.
It's designed to work seamlessly with sovereign_data_models.py and integrate with workflow files.

Author: OpenEvolve Frontend Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import logging
from collections import Counter
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubProblemType(Enum):
    """
    Classification types for sub-problems.

    Each type represents a different category of work:
    - IMPLEMENTATION: Writing code, creating features, building systems
    - ANALYSIS: Examining data, researching problems, understanding requirements
    - VALIDATION: Testing, verifying, reviewing, quality assurance
    """
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    VALIDATION = "validation"

    def __str__(self) -> str:
        """Return string representation of the type."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> 'SubProblemType':
        """
        Convert a string to SubProblemType enum.

        Args:
            value: String representation of the type

        Returns:
            SubProblemType enum value

        Raises:
            ValueError: If the string doesn't match any type
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = [t.value for t in cls]
            raise ValueError(
                f"Invalid SubProblemType: '{value}'. "
                f"Valid values are: {', '.join(valid_values)}"
            )


@dataclass
class ClassificationResult:
    """
    Result of problem classification.

    Attributes:
        problem_type: The classified SubProblemType
        confidence: Confidence score from 0.0 to 1.0
        keyword_scores: Dictionary of keyword category scores
        reasoning: Human-readable explanation of the classification
        alternative_types: Other types with their scores (sorted by score)
        classification_metadata: Additional metadata about the classification
    """
    problem_type: SubProblemType
    confidence: float
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    alternative_types: List[Tuple[SubProblemType, float]] = field(default_factory=list)
    classification_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the classification result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if isinstance(self.problem_type, str):
            self.problem_type = SubProblemType.from_string(self.problem_type)

        # Convert string tuples to SubProblemType tuples if needed
        converted_alternatives = []
        for alt in self.alternative_types:
            if isinstance(alt[0], str):
                converted_alternatives.append((SubProblemType.from_string(alt[0]), alt[1]))
            else:
                converted_alternatives.append(alt)
        self.alternative_types = sorted(
            converted_alternatives,
            key=lambda x: x[1],
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'problem_type': self.problem_type.value,
            'confidence': self.confidence,
            'keyword_scores': self.keyword_scores,
            'reasoning': self.reasoning,
            'alternative_types': [
                {'type': alt[0].value, 'score': alt[1]}
                for alt in self.alternative_types
            ],
            'classification_metadata': self.classification_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create ClassificationResult from dictionary."""
        problem_type = SubProblemType.from_string(data['problem_type'])

        alternatives = []
        for alt in data.get('alternative_types', []):
            if isinstance(alt, dict):
                alternatives.append((SubProblemType.from_string(alt['type']), alt['score']))
            else:
                alternatives.append((SubProblemType.from_string(alt[0]), alt[1]))

        return cls(
            problem_type=problem_type,
            confidence=data['confidence'],
            keyword_scores=data.get('keyword_scores', {}),
            reasoning=data.get('reasoning', ''),
            alternative_types=alternatives,
            classification_metadata=data.get('classification_metadata', {})
        )


@dataclass
class KeywordPattern:
    """
    A keyword pattern for classification.

    Attributes:
        keywords: List of keywords/phrases to match
        weight: Weight for this pattern (higher = more important)
        category: Category this pattern belongs to
        pattern_type: Type of pattern ('simple', 'regex', 'phrase')
    """
    keywords: List[str]
    weight: float = 1.0
    category: str = "general"
    pattern_type: str = "simple"

    def matches(self, text: str) -> Tuple[bool, float]:
        """
        Check if this pattern matches the given text.

        Args:
            text: Text to check against

        Returns:
            Tuple of (matches, score)
        """
        text_lower = text.lower()

        if self.pattern_type == "regex":
            # Use regex matching
            matches = sum(1 for kw in self.keywords if re.search(kw, text_lower))
        elif self.pattern_type == "phrase":
            # Match complete phrases
            matches = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        else:
            # Simple word matching
            words = set(text_lower.split())
            matches = sum(1 for kw in self.keywords if kw.lower() in words)

        score = (matches / len(self.keywords)) * self.weight if self.keywords else 0
        return matches > 0, score


class ProblemClassifier:
    """
    Intelligent classifier for sub-problems.

    This class analyzes problem descriptions and classifies them into types
    using keyword matching, pattern recognition, and confidence scoring.

    Features:
    - Keyword-based classification with weighted patterns
    - Confidence scoring based on multiple indicators
    - Support for custom classification rules
    - Handling of edge cases (mixed-type, ambiguous descriptions)
    - Extensible NLP pattern support
    """

    # Default keyword patterns for each type
    DEFAULT_PATTERNS: Dict[SubProblemType, List[KeywordPattern]] = {
        SubProblemType.IMPLEMENTATION: [
            KeywordPattern(
                keywords=[
                    "implement", "create", "build", "develop", "write", "code",
                    "program", "construct", "design", "architecture", "setup",
                    "configure", "deploy", "integrate", "interface", "api",
                    "function", "method", "class", "module", "component",
                    "add", "make", "generate", "produce"
                ],
                weight=1.0,
                category="action",
                pattern_type="simple"
            ),
            KeywordPattern(
                keywords=[
                    r"implement\s+\w+", r"create\s+\w+", r"build\s+\w+",
                    r"develop\s+\w+", r"write\s+\w+", r"add\s+\w+"
                ],
                weight=1.5,
                category="action_phrase",
                pattern_type="regex"
            ),
            KeywordPattern(
                keywords=[
                    "database", "backend", "frontend", "service", "system",
                    "application", "feature", "endpoint", "route", "controller",
                    "repository", "model", "view", "template", "schema"
                ],
                weight=0.8,
                category="technical",
                pattern_type="simple"
            ),
        ],
        SubProblemType.ANALYSIS: [
            KeywordPattern(
                keywords=[
                    "analyze", "investigate", "examine", "explore", "research",
                    "study", "review", "assess", "evaluate", "understand",
                    "identify", "discover", "determine", "figure out", "diagnose",
                    "find", "locate", "trace", "inspect"
                ],
                weight=1.0,
                category="action",
                pattern_type="simple"
            ),
            KeywordPattern(
                keywords=[
                    r"analyze\s+\w+", r"investigate\s+\w+", r"examine\s+\w+",
                    r"research\s+\w+", r"understand\s+\w+"
                ],
                weight=1.5,
                category="action_phrase",
                pattern_type="regex"
            ),
            KeywordPattern(
                keywords=[
                    "issue", "problem", "bug", "error", "behavior", "cause",
                    "root", "impact", "requirement", "specification", "data",
                    "why", "how", "what", "when", "where"
                ],
                weight=0.8,
                category="technical",
                pattern_type="simple"
            ),
        ],
        SubProblemType.VALIDATION: [
            KeywordPattern(
                keywords=[
                    "test", "verify", "validate", "check", "ensure", "confirm",
                    "inspect", "audit", "review", "quality", "assurance",
                    "assert", "expect", "coverage", "mock", "stub",
                    "prove", "certify", "guarantee"
                ],
                weight=1.0,
                category="action",
                pattern_type="simple"
            ),
            KeywordPattern(
                keywords=[
                    r"test\s+\w+", r"verify\s+\w+", r"validate\s+\w+",
                    r"check\s+\w+", r"ensure\s+\w+"
                ],
                weight=1.5,
                category="action_phrase",
                pattern_type="regex"
            ),
            KeywordPattern(
                keywords=[
                    "unit test", "integration test", "e2e test", "test case",
                    "test suite", "assertion", "fixture", "spec", "specification",
                    "quality assurance", "acceptance", "criteria"
                ],
                weight=1.2,
                category="technical",
                pattern_type="phrase"
            ),
        ],
    }

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.50
    LOW_CONFIDENCE_THRESHOLD = 0.25

    def __init__(
        self,
        custom_patterns: Optional[Dict[SubProblemType, List[KeywordPattern]]] = None,
        confidence_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
        enable_nlp_patterns: bool = True,
        handle_mixed_types: bool = True
    ):
        """
        Initialize the ProblemClassifier.

        Args:
            custom_patterns: Optional custom patterns to add/override defaults
            confidence_threshold: Minimum confidence threshold for classification
            enable_nlp_patterns: Whether to use NLP-based patterns
            handle_mixed_types: Whether to detect and handle mixed-type problems
        """
        self.patterns = {}
        for problem_type, patterns in self.DEFAULT_PATTERNS.items():
            self.patterns[problem_type] = [p for p in patterns]

        # Apply custom patterns
        if custom_patterns:
            for problem_type, patterns in custom_patterns.items():
                if problem_type in self.patterns:
                    self.patterns[problem_type].extend(patterns)
                else:
                    self.patterns[problem_type] = patterns

        self.confidence_threshold = confidence_threshold
        self.enable_nlp_patterns = enable_nlp_patterns
        self.handle_mixed_types = handle_mixed_types

        # Initialize NLP patterns
        self._initialize_nlp_patterns()

        logger.info(
            f"ProblemClassifier initialized with {len(custom_patterns) if custom_patterns else 0} "
            f"custom pattern sets, confidence threshold: {confidence_threshold}"
        )

    def _initialize_nlp_patterns(self):
        """Initialize advanced NLP-based patterns."""
        # Patterns based on linguistic structure
        self.nlp_action_patterns = {
            SubProblemType.IMPLEMENTATION: [
                r"^(create|build|develop|implement|write|add|make)\s+(\w+\s+){1,5}",
                r"need\s+to\s+(create|build|implement|develop|write)",
                r"add\s+(\w+\s+){1,5}(feature|function|capability)",
                r"should\s+(create|build|implement|make)",
            ],
            SubProblemType.ANALYSIS: [
                r"^(analyze|investigate|examine|explore|research|understand)\s+(\w+\s+){1,5}",
                r"(why|how|what)\s+(is|are|does|did)\s+\w+",
                r"understand\s+(\w+\s+){1,5}",
                r"figure\s+out\s+(\w+\s+){1,5}",
                r"determine\s+(\w+\s+){1,5}",
            ],
            SubProblemType.VALIDATION: [
                r"^(test|verify|validate|check|ensure)\s+(\w+\s+){1,5}",
                r"make\s+sure\s+(\w+\s+){1,5}",
                r"confirm\s+that\s+(\w+\s+){1,5}",
                r"ensure\s+that\s+(\w+\s+){1,5}",
            ],
        }

    def classify_problem(
        self,
        problem: Any,
        return_details: bool = False
    ) -> SubProblemType | ClassificationResult:
        """
        Classify a sub-problem into its type.

        Args:
            problem: SubProblem instance to classify (or dict with 'description' field)
            return_details: If True, return ClassificationResult with details

        Returns:
            SubProblemType or ClassificationResult (if return_details=True)

        Raises:
            ValueError: If problem description is empty or invalid
        """
        # Extract description from problem
        if isinstance(problem, dict):
            description = problem.get('description', '')
            title = problem.get('title', '')
        else:
            description = getattr(problem, 'description', None)
            title = getattr(problem, 'title', '')

        if not description or not isinstance(description, str):
            raise ValueError("Problem description must be a non-empty string")

        if len(description.strip()) < 3:
            raise ValueError("Problem description is too short to classify")

        # Combine title and description for better classification
        combined_text = f"{title} {description}".strip()

        # Analyze keywords
        keyword_scores = self.analyze_keywords(combined_text)

        # Determine type based on keywords
        problem_type = self.determine_type(keyword_scores)

        # Calculate confidence score
        confidence = self.get_confidence_score(keyword_scores)

        # Check for mixed-type problems
        is_mixed_type = False
        if self.handle_mixed_types and confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            is_mixed_type = self._is_mixed_type_problem(keyword_scores)
            if is_mixed_type:
                # Adjust reasoning to indicate mixed type
                pass

        # Generate reasoning
        reasoning = self._generate_reasoning(
            problem_type,
            keyword_scores,
            confidence,
            is_mixed_type
        )

        # Get alternative types
        alternative_types = self._get_alternative_types(problem_type, keyword_scores)

        # Create classification result
        result = ClassificationResult(
            problem_type=problem_type,
            confidence=confidence,
            keyword_scores=keyword_scores,
            reasoning=reasoning,
            alternative_types=alternative_types,
            classification_metadata={
                "description_length": len(combined_text),
                "word_count": len(combined_text.split()),
                "title_length": len(title),
                "classified_at": datetime.utcnow().isoformat(),
                "threshold_used": self.confidence_threshold,
                "is_high_confidence": confidence >= self.HIGH_CONFIDENCE_THRESHOLD,
                "is_low_confidence": confidence < self.MEDIUM_CONFIDENCE_THRESHOLD,
                "is_mixed_type": is_mixed_type,
                "has_multiple_types": len([s for s in keyword_scores.values() if s > 0]) > 1
            }
        )

        # Log classification
        log_title = title if title else problem.sub_problem_id if hasattr(problem, 'sub_problem_id') else "Unknown"
        logger.info(
            f"Classified problem '{log_title}' as {problem_type.value} "
            f"with confidence {confidence:.2f}"
        )

        return result if return_details else problem_type

    def analyze_keywords(self, description: str) -> Dict[str, float]:
        """
        Analyze keywords in the description and return category scores.

        Args:
            description: Problem description text

        Returns:
            Dictionary mapping categories/types to their scores
        """
        scores = {problem_type.value: 0.0 for problem_type in SubProblemType}

        description_lower = description.lower()

        # Analyze each type's patterns
        for problem_type, patterns in self.patterns.items():
            type_score = 0.0
            category_scores: Dict[str, float] = {}

            for pattern in patterns:
                matches, score = pattern.matches(description_lower)
                if matches:
                    type_score += score
                    category_scores[pattern.category] = (
                        category_scores.get(pattern.category, 0) + score
                    )

            scores[problem_type.value] = type_score

        # Apply NLP patterns if enabled
        if self.enable_nlp_patterns:
            nlp_scores = self._apply_nlp_patterns(description)
            for problem_type, nlp_score in nlp_scores.items():
                scores[problem_type.value] += nlp_score

        return scores

    def determine_type(self, keywords: Dict[str, float]) -> SubProblemType:
        """
        Determine the problem type based on keyword scores.

        Args:
            keywords: Dictionary of type scores from analyze_keywords

        Returns:
            SubProblemType with the highest score

        Raises:
            ValueError: If keywords dictionary is invalid or empty
        """
        if not keywords or not isinstance(keywords, dict):
            raise ValueError("Keywords must be a non-empty dictionary")

        # Normalize scores
        total_score = sum(keywords.values()) or 1.0  # Avoid division by zero
        normalized_scores = {
            problem_type: score / total_score
            for problem_type, score in keywords.items()
        }

        # Find the type with highest score
        max_score = max(normalized_scores.values())
        top_types = [
            pt for pt, score in normalized_scores.items() if score == max_score
        ]

        # Handle ties by preferring order: IMPLEMENTATION > ANALYSIS > VALIDATION
        type_preference = [
            SubProblemType.IMPLEMENTATION.value,
            SubProblemType.ANALYSIS.value,
            SubProblemType.VALIDATION.value
        ]

        for preferred_type in type_preference:
            if preferred_type in top_types:
                return SubProblemType.from_string(preferred_type)

        # Fallback (shouldn't reach here)
        return SubProblemType.from_string(top_types[0])

    def get_confidence_score(self, keywords: Dict[str, float]) -> float:
        """
        Calculate confidence score for the classification.

        Args:
            keywords: Dictionary of type scores from analyze_keywords

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not keywords:
            return 0.0

        # Calculate total score and max score
        total_score = sum(keywords.values())
        max_score = max(keywords.values())

        if total_score == 0:
            return 0.0

        # Base confidence: ratio of max score to total
        base_confidence = max_score / total_score

        # Boost confidence if one type dominates
        sorted_scores = sorted(keywords.values(), reverse=True)
        if len(sorted_scores) > 1:
            second_max = sorted_scores[1]
            dominance_ratio = (max_score - second_max) / (max_score + second_max) if max_score > 0 else 0
        else:
            dominance_ratio = 1.0

        # Combine metrics
        confidence = base_confidence * 0.7 + dominance_ratio * 0.3

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _is_mixed_type_problem(self, keyword_scores: Dict[str, float]) -> bool:
        """
        Detect if this is a mixed-type problem (indicators from multiple types).

        Args:
            keyword_scores: Scores for each type

        Returns:
            True if problem appears to be mixed-type
        """
        # Count types with significant scores
        significant_scores = [s for s in keyword_scores.values() if s > 0.5]

        # If 2+ types have significant scores, it's mixed
        return len(significant_scores) >= 2

    def _apply_nlp_patterns(self, description: str) -> Dict[SubProblemType, float]:
        """
        Apply NLP-based patterns to the description.

        Args:
            description: Problem description text

        Returns:
            Dictionary mapping SubProblemType to additional scores
        """
        scores = {problem_type: 0.0 for problem_type in SubProblemType}

        description_lower = description.lower()

        for problem_type, patterns in self.nlp_action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower, re.MULTILINE | re.IGNORECASE):
                    scores[problem_type] += 0.5  # Boost score for NLP match

        return scores

    def _generate_reasoning(
        self,
        problem_type: SubProblemType,
        keyword_scores: Dict[str, float],
        confidence: float,
        is_mixed_type: bool = False
    ) -> str:
        """
        Generate human-readable reasoning for the classification.

        Args:
            problem_type: The classified type
            keyword_scores: Scores for each type
            confidence: Confidence score
            is_mixed_type: Whether this is a mixed-type problem

        Returns:
            Human-readable explanation
        """
        # Find top scoring keywords
        sorted_scores = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build reasoning
        parts = []

        # Main classification
        if is_mixed_type:
            parts.append(
                f"Classified as '{problem_type.value}' (appears to be a mixed-type problem)."
            )
        else:
            parts.append(
                f"Classified as '{problem_type.value}' based on keyword analysis."
            )

        # Confidence level
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            parts.append("High confidence classification - clear indicators found.")
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            parts.append("Medium confidence classification - some ambiguity detected.")
        else:
            parts.append("Low confidence classification - description is ambiguous or mixed-type.")

        # Score breakdown
        parts.append("Score breakdown:")
        for type_name, score in sorted_scores:
            if score > 0:
                indicator = ">>>" if type_name == problem_type.value else "   "
                parts.append(f"  {indicator} {type_name}: {score:.2f}")

        return "\n".join(parts)

    def _get_alternative_types(
        self,
        primary_type: SubProblemType,
        keyword_scores: Dict[str, float]
    ) -> List[Tuple[SubProblemType, float]]:
        """
        Get alternative types with their scores.

        Args:
            primary_type: The primary classified type
            keyword_scores: Scores for each type

        Returns:
            List of (SubProblemType, score) tuples sorted by score
        """
        alternatives = []

        for type_name, score in keyword_scores.items():
            if score > 0:
                problem_type = SubProblemType.from_string(type_name)
                if problem_type != primary_type:
                    alternatives.append((problem_type, score))

        return sorted(alternatives, key=lambda x: x[1], reverse=True)

    def add_custom_pattern(
        self,
        problem_type: SubProblemType,
        keywords: List[str],
        weight: float = 1.0,
        category: str = "custom",
        pattern_type: str = "simple"
    ) -> None:
        """
        Add a custom classification pattern.

        Args:
            problem_type: Type to add pattern for
            keywords: List of keywords/phrases
            weight: Weight for this pattern
            category: Category for the pattern
            pattern_type: Type of pattern ('simple', 'regex', 'phrase')
        """
        pattern = KeywordPattern(
            keywords=keywords,
            weight=weight,
            category=category,
            pattern_type=pattern_type
        )

        if problem_type not in self.patterns:
            self.patterns[problem_type] = []

        self.patterns[problem_type].append(pattern)

        logger.info(f"Added custom pattern for {problem_type.value}: {category}")

    def classify_batch(
        self,
        problems: List[Any],
        return_details: bool = False
    ) -> List[SubProblemType | ClassificationResult]:
        """
        Classify multiple problems in batch.

        Args:
            problems: List of SubProblem instances (or dicts)
            return_details: If True, return ClassificationResult for each

        Returns:
            List of classifications
        """
        results = []

        for problem in problems:
            try:
                result = self.classify_problem(problem, return_details=return_details)
                results.append(result)
            except ValueError as e:
                # Extract title for logging
                if isinstance(problem, dict):
                    title = problem.get('title', 'Unknown')
                else:
                    title = getattr(problem, 'title', 'Unknown')

                logger.warning(f"Failed to classify problem '{title}': {e}")

                # Return fallback classification
                if return_details:
                    results.append(
                        ClassificationResult(
                            problem_type=SubProblemType.ANALYSIS,  # Default fallback
                            confidence=0.0,
                            reasoning=f"Classification failed: {str(e)}",
                            keyword_scores={}
                        )
                    )
                else:
                    results.append(SubProblemType.ANALYSIS)

        return results

    def get_type_distribution(
        self,
        problems: List[Any]
    ) -> Dict[SubProblemType, int]:
        """
        Get distribution of problem types in a list.

        Args:
            problems: List of SubProblem instances (or dicts)

        Returns:
            Dictionary mapping type to count
        """
        classifications = self.classify_batch(problems, return_details=False)

        distribution = Counter(classifications)

        # Convert to dict with all types
        return {
            ptype: distribution.get(ptype, 0)
            for ptype in SubProblemType
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier configuration and statistics.

        Returns:
            Dictionary with classifier info
        """
        return {
            'confidence_threshold': self.confidence_threshold,
            'nlp_patterns_enabled': self.enable_nlp_patterns,
            'mixed_type_handling': self.handle_mixed_types,
            'available_types': [t.value for t in SubProblemType],
            'patterns_count': {
                t.value: len(patterns)
                for t, patterns in self.patterns.items()
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def classify_problem_quick(
    description: str,
    title: str = ""
) -> SubProblemType:
    """
    Quick classification function for simple use cases.

    Args:
        description: Problem description
        title: Optional problem title

    Returns:
        SubProblemType classification
    """
    problem_dict = {
        'description': description,
        'title': title
    }

    classifier = ProblemClassifier()
    return classifier.classify_problem(problem_dict)


def classify_with_confidence(
    description: str,
    title: str = ""
) -> Tuple[SubProblemType, float]:
    """
    Classify a problem and return both type and confidence.

    Args:
        description: Problem description
        title: Optional problem title

    Returns:
        Tuple of (SubProblemType, confidence_score)
    """
    problem_dict = {
        'description': description,
        'title': title
    }

    classifier = ProblemClassifier()
    result = classifier.classify_problem(problem_dict, return_details=True)

    return result.problem_type, result.confidence


def classify_subproblem_from_model(sub_problem: Any) -> ClassificationResult:
    """
    Classify a SubProblem model instance (from sovereign_data_models).

    Args:
        sub_problem: SubProblem instance from sovereign_data_models

    Returns:
        ClassificationResult with full details
    """
    classifier = ProblemClassifier()
    return classifier.classify_problem(sub_problem, return_details=True)


def batch_classify_descriptions(
    descriptions: List[Tuple[str, str]]
) -> List[Tuple[str, SubProblemType, float]]:
    """
    Batch classify multiple (title, description) pairs.

    Args:
        descriptions: List of (title, description) tuples

    Returns:
        List of (title, type, confidence) tuples
    """
    problems = [
        {'title': title, 'description': desc}
        for title, desc in descriptions
    ]

    classifier = ProblemClassifier()
    results = classifier.classify_batch(problems, return_details=True)

    return [
        (desc['title'], r.problem_type, r.confidence)
        for desc, r in zip(problems, results)
    ]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SubProblemType',
    'ClassificationResult',
    'KeywordPattern',
    'ProblemClassifier',
    'classify_problem_quick',
    'classify_with_confidence',
    'classify_subproblem_from_model',
    'batch_classify_descriptions',
]
