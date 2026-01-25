"""
Automatic Problem Type Classification System

This module provides intelligent problem classification using both LLM-based
and keyword-based approaches with graceful fallback.

FEATURES:
- LLM-based classification for high accuracy (>80% target)
- Keyword-based fallback for speed and reliability
- Multi-dimensional confidence scoring
- Strategy suggestion based on problem type
- Integration with DecompositionEngine
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from sovereign_data_models import (
    ProblemDefinition, ProblemType, DomainContext, generate_id
)

# Try to import OpenEvolve for LLM-based classification
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    OpenEvolveClient = None  # type: ignore
    OPENEVOLVE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# KEYWORD SETS FOR FALLBACK CLASSIFICATION
# ============================================================================

IMPLEMENTATION_KEYWORDS = [
    "build", "create", "implement", "develop", "construct",
    "deploy", "setup", "install", "integrate", "write",
    "code", "program", "application", "system", "platform",
    "feature", "module", "component", "api", "interface"
]

ANALYSIS_KEYWORDS = [
    "analyze", "examine", "investigate", "understand", "evaluate",
    "assess", "review", "study", "compare", "measure",
    "audit", "inspect", "explore", "diagnose", "debug",
    "profile", "benchmark", "characterize"
]

RESEARCH_KEYWORDS = [
    "research", "explore", "discover", "investigate", "find",
    "identify", "search", "experiment", "study", "learn",
    "investigate", "survey", "literature", "novel", "new",
    "emerging", "state-of-the-art", "cutting-edge"
]

DESIGN_KEYWORDS = [
    "design", "architect", "plan", "structure", "framework",
    "blueprint", "schema", "model", "specify", "outline",
    "prototype", "mockup", "wireframe", "strategy", "approach",
    "methodology", "paradigm", "pattern"
]

OPTIMIZATION_KEYWORDS = [
    "optimize", "improve", "enhance", "refactor", "streamline",
    "accelerate", "reduce", "minimize", "maximize", "efficient",
    "performance", "scalability", "speed", "fast", "slow",
    "bottleneck", "latency", "throughput", "resource"
]

VALIDATION_KEYWORDS = [
    "validate", "verify", "test", "confirm", "check",
    "ensure", "guarantee", "prove", "benchmark", "certify",
    "quality assurance", "testing", "verification", "inspection",
    "audit", "compliance", "standard"
]

# Mapping of keywords to problem types
KEYWORD_MAP = {
    ProblemType.IMPLEMENTATION: IMPLEMENTATION_KEYWORDS,
    ProblemType.ANALYSIS: ANALYSIS_KEYWORDS,
    ProblemType.RESEARCH: RESEARCH_KEYWORDS,
    ProblemType.DESIGN: DESIGN_KEYWORDS,
    ProblemType.OPTIMIZATION: OPTIMIZATION_KEYWORDS,
    ProblemType.VALIDATION: VALIDATION_KEYWORDS
}


# ============================================================================
# LLM PROMPT TEMPLATES
# ============================================================================

CLASSIFICATION_PROMPT = """
You are an expert problem classifier. Analyze the following problem and classify its type.

PROBLEM TITLE: {title}
DESCRIPTION: {description}
DOMAIN: {domain}
SUBDOMAIN: {subdomain}

Classify this problem as ONE of the following types:

1. IMPLEMENTATION: Building or creating something new
   - Keywords: build, create, implement, develop, construct, deploy, code, write
   - Focus: Creating new software, systems, features, or components

2. ANALYSIS: Understanding or examining something existing
   - Keywords: analyze, examine, investigate, understand, evaluate, assess, study
   - Focus: Analyzing existing code, systems, data, or processes

3. RESEARCH: Exploring or discovering new knowledge
   - Keywords: research, explore, discover, investigate, find, identify, experiment
   - Focus: Learning about new technologies, approaches, or domains

4. DESIGN: Architecting or planning something
   - Keywords: design, architect, plan, structure, framework, blueprint, model
   - Focus: Creating architecture, specifications, or strategic plans

5. OPTIMIZATION: Improving something existing
   - Keywords: optimize, improve, enhance, refactor, streamline, accelerate
   - Focus: Making existing systems faster, more efficient, or more scalable

6. VALIDATION: Verifying or testing something
   - Keywords: validate, verify, test, confirm, check, ensure, guarantee
   - Focus: Testing, quality assurance, verification, or certification

Provide your analysis in the following JSON format:

{{
    "primary_type": "TYPE_NAME",
    "confidence": 0.0-1.0,
    "secondary_types": ["TYPE1", "TYPE2"],
    "reasoning": "1-2 sentence explanation of why this type was chosen",
    "indicators": ["word1", "phrase2", "pattern3"],
    "characteristics": {{
        "has_clear_requirements": true/false,
        "requires_creativity": true/false,
        "technically_complex": true/false,
        "time_critical": true/false
    }},
    "suggested_strategies": ["strategy1", "strategy2"]
}}

Respond ONLY with valid JSON. Do not include any explanatory text outside the JSON.
"""


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class ProblemClassification:
    """
    Result of problem classification analysis.

    Attributes:
        primary_type: Main problem type classification
        confidence: Confidence score (0.0-1.0)
        secondary_types: Additional applicable problem types
        reasoning: Explanation of classification decision
        suggested_strategies: Recommended decomposition strategies
        characteristics: Extracted problem characteristics
        indicators: Words/phrases that indicated this classification
        classification_method: "llm" or "keyword"
        timestamp: When classification was performed
        metadata: Additional classification metadata
    """
    primary_type: ProblemType
    confidence: float
    secondary_types: List[ProblemType] = field(default_factory=list)
    reasoning: str = ""
    suggested_strategies: List[str] = field(default_factory=list)
    characteristics: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    classification_method: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'primary_type': self.primary_type.value if isinstance(self.primary_type, ProblemType) else self.primary_type,
            'confidence': self.confidence,
            'secondary_types': [t.value if isinstance(t, ProblemType) else t for t in self.secondary_types],
            'reasoning': self.reasoning,
            'suggested_strategies': self.suggested_strategies,
            'characteristics': self.characteristics,
            'indicators': self.indicators,
            'classification_method': self.classification_method,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProblemClassification':
        """Create from dictionary."""
        data = data.copy()
        data['primary_type'] = ProblemType(data['primary_type'])
        data['secondary_types'] = [ProblemType(t) for t in data.get('secondary_types', [])]
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate classification data."""
        errors = []
        if not isinstance(self.primary_type, ProblemType):
            errors.append(f"primary_type must be ProblemType, got {type(self.primary_type)}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.classification_method not in ["llm", "keyword", "unknown"]:
            errors.append(f"classification_method must be 'llm', 'keyword', or 'unknown', got {self.classification_method}")
        for secondary_type in self.secondary_types:
            if not isinstance(secondary_type, ProblemType):
                errors.append(f"secondary_type must be ProblemType, got {type(secondary_type)}")
        return errors


# ============================================================================
# MAIN CLASSIFIER
# ============================================================================

class ProblemClassifier:
    """
    Automatic problem type classification system.

    Provides intelligent classification using:
    1. LLM-based classification (primary, high accuracy)
    2. Keyword-based classification (fallback, fast)

    The classifier automatically falls back to keyword-based classification
    if LLM is unavailable or fails.
    """

    def __init__(self, llm_client: Optional['OpenEvolveClient'] = None,
                 enable_llm: bool = True, llm_fallback_enabled: bool = True):
        """
        Initialize problem classifier.

        Args:
            llm_client: Optional LLM client for classification
            enable_llm: Whether to use LLM-based classification (default: True)
            llm_fallback_enabled: Whether to fallback to keyword-based on LLM failure (default: True)
        """
        self.llm_client = llm_client
        self.enable_llm = enable_llm and OPENEVOLVE_AVAILABLE
        self.llm_fallback_enabled = llm_fallback_enabled
        self.logger = logging.getLogger(__name__)

        # Initialize LLM client if not provided
        if self.enable_llm and not self.llm_client:
            try:
                self.llm_client = OpenEvolveClient()
                self.logger.info("Initialized OpenEvolve client for problem classification")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                self.logger.warning(f"Failed to initialize LLM client: {e}")
                self.enable_llm = False

        # Statistics
        self.classification_stats = {
            'llm_success': 0,
            'llm_failure': 0,
            'keyword_fallback': 0,
            'total': 0
        }

    def classify_problem(
        self,
        problem: ProblemDefinition,
        domain_context: Optional[DomainContext] = None,
        force_method: Optional[str] = None
    ) -> ProblemClassification:
        """
        Classify a problem into its appropriate type.

        Args:
            problem: The problem to classify
            domain_context: Optional domain context for better classification
            force_method: Force specific method ("llm" or "keyword")

        Returns:
            ProblemClassification with type, confidence, and reasoning

        Raises:
            ValueError: If forced_method is invalid
        """
        self.classification_stats['total'] += 1

        # Use domain context from problem if not provided
        if domain_context is None:
            domain_context = problem.domain_context

        # Force specific method if requested
        if force_method:
            if force_method == "llm":
                return self._classify_with_llm(problem, domain_context)
            elif force_method == "keyword":
                return self._classify_with_keywords(problem, domain_context)
            else:
                raise ValueError(f"Invalid force_method: {force_method}. Must be 'llm' or 'keyword'")

        # Try LLM-based classification first (more accurate)
        if self.enable_llm:
            try:
                classification = self._classify_with_llm(problem, domain_context)
                self.classification_stats['llm_success'] += 1
                return classification
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                self.classification_stats['llm_failure'] += 1
                self.logger.warning(f"LLM classification failed: {e}")

                # Fall back to keyword-based if enabled
                if self.llm_fallback_enabled:
                    self.logger.info("Falling back to keyword-based classification")
                    classification = self._classify_with_keywords(problem, domain_context)
                    self.classification_stats['keyword_fallback'] += 1
                    return classification
                else:
                    raise

        # Use keyword-based classification
        classification = self._classify_with_keywords(problem, domain_context)
        self.classification_stats['keyword_fallback'] += 1
        return classification

    def _classify_with_llm(
        self,
        problem: ProblemDefinition,
        domain_context: DomainContext
    ) -> ProblemClassification:
        """
        Classify problem using LLM analysis.

        Args:
            problem: Problem to classify
            domain_context: Domain context

        Returns:
            ProblemClassification from LLM analysis

        Raises:
            RuntimeError: If LLM is not available or classification fails
        """
        if not self.enable_llm or not self.llm_client:
            raise RuntimeError("LLM-based classification not available")

        self.logger.info(f"Classifying problem {problem.id} using LLM")

        # Prepare prompt
        prompt = CLASSIFICATION_PROMPT.format(
            title=problem.title,
            description=problem.description[:500],  # Limit length
            domain=domain_context.domain,
            subdomain=domain_context.subdomain or "N/A"
        )

        try:
            # Use OpenEvolve client to get classification
            # Note: We're using the client's query capability
            result = self._query_llm(prompt)

            # Parse JSON response
            classification_data = self._parse_llm_response(result)

            # Convert to ProblemClassification
            return self._create_classification_from_llm(classification_data)

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            self.logger.error(f"LLM classification error: {e}", exc_info=True)
            raise RuntimeError(f"LLM classification failed: {e}")

    def _query_llm(self, prompt: str) -> str:
        """
        Query LLM with classification prompt.

        Args:
            prompt: Classification prompt

        Returns:
            LLM response text
        """
        # Try to use OpenEvolve's evolution capabilities for classification
        if self.llm_client and hasattr(self.llm_client, 'evolve'):
            # Use evolve in "analysis" mode
            result = self.llm_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="classification"
            )

            if result.success:
                return result.best_code
            else:
                raise RuntimeError(f"Evolution failed: {result.error}")

        # Fallback: Try direct API call
        if OPENEVOLVE_AVAILABLE:
            try:
                from openevolve.api import run_evolution
                from openevolve.config import Config

                config = Config(
                    generations=1,  # Single generation needed
                    population_size=1,
                    llm_model="gpt-4"  # Use capable model
                )

                result = run_evolution(
                    initial_code=prompt,
                    config=config
                )

                return result.best_code
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                self.logger.warning(f"Direct OpenEvolve API call failed: {e}")

        raise RuntimeError("No LLM query method available")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed classification dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            # Extract JSON from response (handle potential markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)

            return json.loads(response)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response was: {response[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _create_classification_from_llm(self, data: Dict[str, Any]) -> ProblemClassification:
        """
        Create ProblemClassification from LLM response data.

        Args:
            data: Parsed LLM response

        Returns:
            ProblemClassification object
        """
        # Parse primary type
        primary_type_str = data.get('primary_type', 'IMPLEMENTATION').upper()
        try:
            primary_type = ProblemType[primary_type_str]
        except KeyError:
            self.logger.warning(f"Unknown problem type: {primary_type_str}, defaulting to IMPLEMENTATION")
            primary_type = ProblemType.IMPLEMENTATION

        # Parse secondary types
        secondary_types = []
        for type_str in data.get('secondary_types', []):
            try:
                secondary_types.append(ProblemType[type_str.upper()])
            except KeyError:
                self.logger.debug(f"Skipping unknown secondary type: {type_str}")

        # Create classification
        return ProblemClassification(
            primary_type=primary_type,
            confidence=float(data.get('confidence', 0.7)),
            secondary_types=secondary_types,
            reasoning=data.get('reasoning', ''),
            suggested_strategies=data.get('suggested_strategies', []),
            characteristics=data.get('characteristics', {}),
            indicators=data.get('indicators', []),
            classification_method="llm"
        )

    def _classify_with_keywords(
        self,
        problem: ProblemDefinition,
        domain_context: DomainContext
    ) -> ProblemClassification:
        """
        Classify problem using keyword matching (fallback method).

        Args:
            problem: Problem to classify
            domain_context: Domain context

        Returns:
            ProblemClassification based on keyword analysis
        """
        self.logger.info(f"Classifying problem {problem.id} using keywords")

        # Combine title and description for analysis
        text = f"{problem.title} {problem.description}".lower()

        # Score each problem type
        scores = {}
        all_indicators = []

        for problem_type, keywords in KEYWORD_MAP.items():
            score = 0
            type_indicators = []

            for keyword in keywords:
                if keyword.lower() in text:
                    # Count occurrences
                    count = text.count(keyword.lower())
                    score += count
                    type_indicators.append(keyword)

            scores[problem_type] = score
            all_indicators.extend(type_indicators)

        # Find primary type (highest score)
        if max(scores.values()) == 0:
            # No keywords found - default to IMPLEMENTATION
            primary_type = ProblemType.IMPLEMENTATION
            confidence = 0.3
            reasoning = "No clear indicators found, defaulting to IMPLEMENTATION"
        else:
            primary_type = max(scores, key=scores.get)
            max_score = scores[primary_type]

            # Calculate confidence based on score distribution
            total_score = sum(scores.values())
            if total_score > 0:
                confidence = max_score / total_score
                # Boost confidence if there's a clear winner
                if max_score > sum(v for k, v in scores.items() if k != primary_type):
                    confidence = min(confidence * 1.3, 1.0)
            else:
                confidence = 0.5

            reasoning = f"Identified {max_score} keywords matching {primary_type.value} type"

        # Find secondary types (any type with score > 0)
        secondary_types = [
            pt for pt, score in scores.items()
            if score > 0 and pt != primary_type
        ]

        # Suggest strategies based on type
        suggested_strategies = self._get_suggested_strategies(primary_type)

        # Extract characteristics
        characteristics = {
            'has_clear_requirements': any(word in text for word in ['requirement', 'spec', 'must', 'shall']),
            'requires_creativity': any(word in text for word in ['innovative', 'creative', 'novel', 'unique']),
            'technically_complex': any(word in text for word in ['complex', 'challenging', 'difficult', 'advanced']),
            'time_critical': any(word in text for word in ['urgent', 'deadline', 'asap', 'critical'])
        }

        # Get indicators for primary type
        indicators = [kw for kw in KEYWORD_MAP[primary_type] if kw.lower() in text]

        return ProblemClassification(
            primary_type=primary_type,
            confidence=round(confidence, 2),
            secondary_types=secondary_types[:3],  # Limit to top 3
            reasoning=reasoning,
            suggested_strategies=suggested_strategies,
            characteristics=characteristics,
            indicators=indicators[:10],  # Limit to top 10
            classification_method="keyword"
        )

    def _get_suggested_strategies(self, problem_type: ProblemType) -> List[str]:
        """
        Get suggested decomposition strategies for a problem type.

        Args:
            problem_type: The classified problem type

        Returns:
            List of recommended strategies
        """
        strategy_mapping = {
            ProblemType.IMPLEMENTATION: [
                "semantic",          # Break down by concepts
                "functional",        # Break down by features
                "technical_dependency"  # Respect technical dependencies
            ],
            ProblemType.ANALYSIS: [
                "semantic",          # Analyze by semantic clusters
                "complexity",        # Tackle complex parts first
                "risk_based"         # Focus on high-risk areas
            ],
            ProblemType.RESEARCH: [
                "research",          # Research-specific decomposition
                "semantic",          # Explore by concept
                "hybrid"             # Mix of approaches
            ],
            ProblemType.DESIGN: [
                "semantic",          # Design by concept
                "functional",        # Design by feature
                "hybrid"             # Multiple perspectives
            ],
            ProblemType.OPTIMIZATION: [
                "complexity",        # Focus on complex bottlenecks
                "semantic",          # Optimize by component
                "risk_based"         # Address critical paths
            ],
            ProblemType.VALIDATION: [
                "risk_based",        # Test high-risk areas
                "functional",        # Test by feature
                "temporal"           # Test in phases
            ]
        }

        return strategy_mapping.get(problem_type, ["hybrid"])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.

        Returns:
            Dictionary with classification metrics
        """
        stats = self.classification_stats.copy()

        # Calculate success rates
        if stats['total'] > 0:
            stats['llm_success_rate'] = stats['llm_success'] / stats['total']
            stats['keyword_fallback_rate'] = stats['keyword_fallback'] / stats['total']
        else:
            stats['llm_success_rate'] = 0.0
            stats['keyword_fallback_rate'] = 0.0

        # Check availability
        stats['llm_available'] = self.enable_llm
        stats['fallback_enabled'] = self.llm_fallback_enabled

        return stats

    def reset_statistics(self):
        """Reset classification statistics."""
        self.classification_stats = {
            'llm_success': 0,
            'llm_failure': 0,
            'keyword_fallback': 0,
            'total': 0
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def classify_problem_auto(
    problem: ProblemDefinition,
    domain_context: Optional[DomainContext] = None,
    llm_client: Optional['OpenEvolveClient'] = None
) -> ProblemClassification:
    """
    Convenience function to classify a problem with automatic method selection.

    Args:
        problem: Problem to classify
        domain_context: Optional domain context
        llm_client: Optional LLM client

    Returns:
        ProblemClassification result
    """
    classifier = ProblemClassifier(llm_client=llm_client)
    return classifier.classify_problem(problem, domain_context)


def get_problem_type_from_text(title: str, description: str) -> ProblemType:
    """
    Quick problem type detection from text (keyword-based only).

    Args:
        title: Problem title
        description: Problem description

    Returns:
        Detected ProblemType
    """
    text = f"{title} {description}".lower()

    # Score each type
    scores = {}
    for problem_type, keywords in KEYWORD_MAP.items():
        score = sum(text.count(kw.lower()) for kw in keywords)
        scores[problem_type] = score

    # Return highest scoring type
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        return ProblemType.IMPLEMENTATION  # Default


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ProblemClassifier',
    'ProblemClassification',
    'classify_problem_auto',
    'get_problem_type_from_text',
    'KEYWORD_MAP',
    'CLASSIFICATION_PROMPT'
]
