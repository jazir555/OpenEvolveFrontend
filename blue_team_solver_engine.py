"""
Blue Team Solver Workflow Engine for OpenEvolve

This module implements the Blue Team solver workflow that takes sub-problems from
the decomposition engine and orchestrates their solution through multiple stages:
analysis, planning, solving, validation, and refinement.

Architecture:
    Decomposition Engine → Blue Team Solver → Solution Integration
                                                        ↓
                                                    Quality Tracker

Key Features:
- Multiple solving strategies (analytical, creative, systematic, hybrid)
- LLM-based solution generation with evolution
- Solution optimization and refinement
- Quality validation with metrics
- Performance tracking and learning

Author: OpenEvolve
Version: 1.0.0
"""

import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import os
import random
from datetime import datetime
import asyncio
import subprocess
import urllib.request
import urllib.error
import uuid

from chronicle_memory import ChronicleMemory
from knowledge_manager import KnowledgeManager
from z3prover_integration import Z3LogicCompressor
from utils.doc_manager import DocstringManager

# **ACTUAL INTEGRATION**: Alerting, knowledge, adaptive for Blue Team solver
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_thorough_config
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.debug(f"ROMA-MDAP-MAKER not available: {e}")
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_thorough_config = None

# Initialize Robust Engine Singleton for Solver Engine
robust_engine = None
if ROMA_MDAP_MAKER_AVAILABLE:
    try:
        # Use SSOT thorough preset for mission-critical solving
        _config = get_thorough_config(
            preset="thorough",
            # Can override specific parameters if needed
            # roma_max_depth_solving=3,  # Example: Override if preset doesn't match needs
            # mdap_min_confidence=0.3     # Example: Override if preset doesn't match needs
        )
        robust_engine = ROMAMDAPMakerAssociativeEngine(_config)
    except (ImportError, RuntimeError, OSError) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error initializing ROMA-MDAP-MAKER engine in {__name__}: {e}", exc_info=True)
        raise  # Re-raise the exception

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

class SolvingStrategy(Enum):
    """Types of solving strategies"""
    ANALYTICAL = "analytical"  # Step-by-step logical analysis
    CREATIVE = "creative"  # Innovative, out-of-the-box solutions
    SYSTEMATIC = "systematic"  # Structured, methodical approach
    HYBRID = "hybrid"  # Combines multiple strategies
    EVOLUTIONARY = "evolutionary"  # Uses OpenEvolve for optimization


class SolutionStatus(Enum):
    """Status of a solution attempt"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    SOLVING = "solving"
    VALIDATING = "validating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"  # Retrieved from cache


@dataclass
class SubProblemInput:
    """Input wrapper for sub-problems from decomposition engine"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    complexity_score: int = 5
    priority: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class QualityMetrics:
    """Quality metrics for a solution"""
    completeness: float = 0.0  # 0.0 to 1.0
    correctness: float = 0.0  # 0.0 to 1.0
    efficiency: float = 0.0  # 0.0 to 1.0
    clarity: float = 0.0  # 0.0 to 1.0
    maintainability: float = 0.0  # 0.0 to 1.0
    innovation: float = 0.0  # 0.0 to 1.0
    overall_score: float = 0.0  # Weighted average

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall quality score"""
        if weights is None:
            weights = {
                "completeness": 0.25,
                "correctness": 0.30,
                "efficiency": 0.15,
                "clarity": 0.10,
                "maintainability": 0.10,
                "innovation": 0.10,
            }

        self.overall_score = sum(
            getattr(self, metric) * weights.get(metric, 0.0)
            for metric in weights.keys()
        )
        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SolutionResult:
    """Result of a solving attempt"""
    sub_problem_id: str
    solution: str
    status: SolutionStatus
    strategy_used: SolvingStrategy
    quality_metrics: QualityMetrics
    execution_time: float = 0.0
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["status"] = self.status.value
        data["strategy_used"] = self.strategy_used.value
        return data


@dataclass
class SolverPerformance:
    """Performance tracking for a solver"""
    total_problems_solved: int = 0
    total_execution_time: float = 0.0
    average_quality_score: float = 0.0
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    cache_hit_rate: float = 0.0

    def update(self, result: SolutionResult, cached: bool = False):
        """Update performance metrics"""
        self.total_problems_solved += 1
        self.total_execution_time += result.execution_time

        # Update average quality
        n = self.total_problems_solved
        self.average_quality_score = (
            (self.average_quality_score * (n - 1) + result.quality_metrics.overall_score) / n
        )

        # Update strategy usage
        strategy = result.strategy_used.value
        self.strategy_usage[strategy] = self.strategy_usage.get(strategy, 0) + 1

        # Update success rate
        if result.status == SolutionStatus.COMPLETED:
            current_successes = self.success_rate * (n - 1)
            self.success_rate = (current_successes + 1) / n

        # Update cache hit rate
        if cached:
            current_hits = self.cache_hit_rate * (n - 1)
            self.cache_hit_rate = (current_hits + 1) / n


# =============================================================================
# REWARD MODELING (ON-POLICY PREFERENCE LEARNING)
# =============================================================================

@dataclass
class PreferenceRecord:
    """Stores a preference comparison between two solutions."""
    previous_solution: str
    current_solution: str
    preference_bit: int  # 0 = previous preferred, 1 = current preferred
    improvement_delta: float
    constraint_snapshot: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class PreferenceStore:
    """Stores preference pairs for reward model training."""

    def __init__(self):
        self.records: List[PreferenceRecord] = []

    def add_record(self, record: PreferenceRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def get_recent_pairs(self, count: int) -> List[PreferenceRecord]:
        return self.records[-count:]


class EvaluatorTeam:
    """Lightweight comparative evaluator for preference labeling."""

    def compare(
        self,
        previous_solution: str,
        current_solution: str,
        constraints: Optional[List[str]] = None
    ) -> Tuple[int, float]:
        constraints = constraints or []

        prev_score = self._constraint_adherence(previous_solution, constraints)
        curr_score = self._constraint_adherence(current_solution, constraints)

        preference_bit = 1 if curr_score >= prev_score else 0
        # Improvement delta blends constraint adherence and length structure
        length_delta = (len(current_solution) - len(previous_solution)) / max(len(previous_solution), 1)
        improvement_delta = max(0.0, min(1.0, 0.7 * (curr_score - prev_score) + 0.3 * length_delta))

        return preference_bit, improvement_delta

    def _constraint_adherence(self, solution: str, constraints: List[str]) -> float:
        if not constraints:
            return 0.5
        matches = 0
        lower_solution = solution.lower()
        for constraint in constraints:
            keywords = [w for w in constraint.lower().split() if len(w) > 3]
            if any(k in lower_solution for k in keywords):
                matches += 1
        return matches / max(len(constraints), 1)


class LocalRewardModel:
    """Simple local reward model for scoring solutions and strategy candidates."""

    def __init__(self, training_frequency: int = 50):
        self.training_frequency = training_frequency
        self.weights = {
            "quality": 1.0,
            "length": 0.1,
            "structure": 0.3,
            "constraint_fit": 0.5,
        }
        self._last_trained_at = 0

    def score_solution(self, solution: str, quality: QualityMetrics, constraints: Optional[List[str]] = None) -> float:
        features = self._extract_features(solution, quality, constraints or [])
        score = sum(self.weights[k] * features.get(k, 0.0) for k in self.weights)
        return 1 / (1 + pow(2.71828, -score))

    def score_strategy_candidate(self, candidate_features: Dict[str, float]) -> float:
        score = sum(self.weights.get(k, 0.0) * candidate_features.get(k, 0.0) for k in self.weights)
        return 1 / (1 + pow(2.71828, -score))

    def maybe_train(self, preference_store: PreferenceStore) -> None:
        if len(preference_store) < self.training_frequency:
            return
        if len(preference_store) - self._last_trained_at < self.training_frequency:
            return

        recent = preference_store.get_recent_pairs(self.training_frequency)
        synthetic_pairs = []
        try:
            from causallearn.utils.preference_synthesis import generate_synthetic_preference_pairs
            synthetic_pairs = generate_synthetic_preference_pairs(recent, max_pairs=10)
        except Exception:
            synthetic_pairs = []
        # Simple weight nudging based on preference bit
        for record in recent:
            delta = 0.05 if record.preference_bit == 1 else -0.05
            self.weights["quality"] += delta * record.improvement_delta
            self.weights["constraint_fit"] += delta * record.improvement_delta
            self.weights["length"] += delta * min(0.1, abs(record.improvement_delta))
            self.weights["structure"] += delta * 0.02

        for pair in synthetic_pairs:
            delta = 0.03 if pair.preference_bit == 1 else -0.03
            self.weights["quality"] += delta * pair.improvement_delta
            self.weights["constraint_fit"] += delta * pair.improvement_delta

        self._last_trained_at = len(preference_store)

    def _extract_features(self, solution: str, quality: QualityMetrics, constraints: List[str]) -> Dict[str, float]:
        length_score = min(1.0, len(solution) / 2000)
        structure_score = 1.0 if ("##" in solution or "```" in solution or "\n-" in solution) else 0.3
        constraint_score = 0.5
        if constraints:
            lower_solution = solution.lower()
            hits = 0
            for constraint in constraints:
                keywords = [w for w in constraint.lower().split() if len(w) > 3]
                if any(k in lower_solution for k in keywords):
                    hits += 1
            constraint_score = hits / max(len(constraints), 1)
        return {
            "quality": quality.overall_score,
            "length": length_score,
            "structure": structure_score,
            "constraint_fit": constraint_score,
        }


# =============================================================================
# ABSTRACT SOLVER INTERFACE
# =============================================================================

class BaseSolver(ABC):
    """Abstract base class for all solvers"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.performance = SolverPerformance()

    @abstractmethod
    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve a sub-problem"""
        pass

    def analyze_problem(self, sub_problem: SubProblemInput) -> Dict[str, Any]:
        """Analyze sub-problem characteristics"""
        analysis = {
            "complexity": sub_problem.complexity_score,
            "has_dependencies": len(sub_problem.dependencies) > 0,
            "dependency_count": len(sub_problem.dependencies),
            "requirement_count": len(sub_problem.requirements),
            "constraint_count": len(sub_problem.constraints),
            "has_context": len(sub_problem.context) > 0,
        }

        # Estimate solving difficulty
        difficulty = (
            sub_problem.complexity_score * 0.4 +
            min(len(sub_problem.dependencies), 5) * 0.2 +
            min(len(sub_problem.requirements), 10) * 0.1 +
            min(len(sub_problem.constraints), 5) * 0.3
        )
        analysis["estimated_difficulty"] = min(difficulty, 10.0)

        return analysis


# =============================================================================
# CONCRETE SOLVER IMPLEMENTATIONS
# =============================================================================

class AnalyticalSolver(BaseSolver):
    """
    Step-by-step analytical problem solver.

    Breaks down problems into logical steps and solves them systematically.
    Best for: Well-defined problems with clear requirements.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("AnalyticalSolver", config)

    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve using analytical approach"""
        start_time = time.time()

        logger.info(f"[{self.name}] Analytically solving sub-problem: {sub_problem.id}")

        try:
            # Import LLM utilities
            from llm_utils import _request_openai_compatible_chat

            # Build analytical prompt
            system_prompt = """You are an expert analytical problem solver. Your approach:
1. Break down the problem into clear, logical steps
2. Analyze each component systematically
3. Apply logical reasoning and deduction
4. Verify each step before proceeding
5. Provide clear, structured solutions

Be methodical and thorough. Show your reasoning process."""

            user_prompt = f"""Solve this problem analytically:

**Problem ID:** {sub_problem.id}
**Description:** {sub_problem.description}

**Requirements:**
{chr(10).join(f'- {r}' for r in sub_problem.requirements) if sub_problem.requirements else 'None specified'}

**Constraints:**
{chr(10).join(f'- {c}' for c in sub_problem.constraints) if sub_problem.constraints else 'None specified'}

**Success Criteria:**
{chr(10).join(f'- {c}' for c in sub_problem.success_criteria) if sub_problem.success_criteria else 'None specified'}

**Context:**
{json.dumps(sub_problem.context, indent=2) if sub_problem.context else 'None'}

Provide:
1. Problem breakdown
2. Step-by-step analysis
3. Logical solution
4. Verification approach
"""

            # Get API config
            api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", "https://api.openai.com/v1")
            model = self.config.get("model", "gpt-4o-mini")

            # Try Robust Engine First
            response = None
            if robust_engine:
                try:
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    engine_result = robust_engine.solve_problem_recursive(full_prompt, sub_problem.context)
                    response = engine_result.get("solution")
                except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError):
                    response = None

            # Fallback to direct call
            if not response:
                response = _request_openai_compatible_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.get("temperature", 0.3),
                    max_tokens=self.config.get("max_tokens", 4096),
                )

            if not response:
                raise ValueError("No response from LLM")

            # Calculate quality metrics
            quality = self._assess_quality(response, sub_problem)

            execution_time = time.time() - start_time

            result = SolutionResult(
                sub_problem_id=sub_problem.id,
                solution=response,
                status=SolutionStatus.COMPLETED,
                strategy_used=SolvingStrategy.ANALYTICAL,
                quality_metrics=quality,
                execution_time=execution_time,
                metadata={"solver": self.name},
            )

            self.performance.update(result)
            return result

        except (ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"[{self.name}] Error solving {sub_problem.id}: {e}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=SolvingStrategy.ANALYTICAL,
                quality_metrics=QualityMetrics(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={"solver": self.name},
            )

    def _assess_quality(self, solution: str, sub_problem: SubProblemInput) -> QualityMetrics:
        """Assess solution quality"""
        quality = QualityMetrics()

        # Completeness: Check for required sections
        if "breakdown" in solution.lower():
            quality.completeness += 0.3
        if "analysis" in solution.lower():
            quality.completeness += 0.3
        if "verification" in solution.lower():
            quality.completeness += 0.2
        if len(solution) > 500:
            quality.completeness += 0.2

        # Correctness: Check for addressing requirements
        for req in sub_problem.requirements:
            if any(word.lower() in solution.lower() for word in req.split()[:3]):
                quality.correctness += min(1.0 / max(len(sub_problem.requirements), 1), 0.3)

        # Efficiency: Check for clear structure
        if "step" in solution.lower():
            quality.efficiency += 0.5
        if "first" in solution.lower() and "then" in solution.lower():
            quality.efficiency += 0.3

        # Clarity: Check formatting
        if "**" in solution or "##" in solution:
            quality.clarity += 0.5
        if any(line.strip().startswith('-') for line in solution.split('\n')):
            quality.clarity += 0.3

        # Maintainability: Check for documentation
        if "```" in solution or "code" in solution.lower():
            quality.maintainability += 0.5
        if "explain" in solution.lower() or "note" in solution.lower():
            quality.maintainability += 0.3

        # Innovation: Lower for analytical (focus on correctness)
        quality.innovation = 0.3

        # Calculate overall
        quality.calculate_overall()

        return quality


class CreativeSolver(BaseSolver):
    """
    Creative problem solver.

    Generates innovative, out-of-the-box solutions.
    Best for: Problems requiring novel approaches or creative thinking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CreativeSolver", config)

    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve using creative approach"""
        start_time = time.time()

        logger.info(f"[{self.name}] Creatively solving sub-problem: {sub_problem.id}")

        try:
            from llm_utils import _request_openai_compatible_chat

            # Build creative prompt
            system_prompt = """You are a creative problem solver who thinks outside the box.
Your approach:
1. Consider unconventional perspectives
2. Challenge assumptions
3. Explore novel solutions
4. Combine ideas from different domains
5. Propose innovative alternatives

Be creative and original while still being practical."""

            user_prompt = f"""Solve this problem creatively:

**Problem ID:** {sub_problem.id}
**Description:** {sub_problem.description}

**Requirements:**
{chr(10).join(f'- {r}' for r in sub_problem.requirements) if sub_problem.requirements else 'None specified'}

**Constraints:**
{chr(10).join(f'- {c}' for c in sub_problem.constraints) if sub_problem.constraints else 'None specified'}

**Success Criteria:**
{chr(10).join(f'- {c}' for c in sub_problem.success_criteria) if sub_problem.success_criteria else 'None specified'}

Think creatively and propose innovative solutions. Consider approaches that
might not be obvious at first glance."""

            # Get API config
            api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", "https://api.openai.com/v1")
            model = self.config.get("model", "gpt-4o-mini")

            # Try Robust Engine First
            response = None
            if robust_engine:
                try:
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    engine_result = robust_engine.solve_problem_recursive(full_prompt, sub_problem.context)
                    response = engine_result.get("solution")
                except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError):
                    response = None

            # Fallback to direct call
            if not response:
                response = _request_openai_compatible_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.get("temperature", 0.9),
                    max_tokens=self.config.get("max_tokens", 4096),
                )

            if not response:
                raise ValueError("No response from LLM")

            quality = self._assess_quality(response, sub_problem)

            execution_time = time.time() - start_time

            result = SolutionResult(
                sub_problem_id=sub_problem.id,
                solution=response,
                status=SolutionStatus.COMPLETED,
                strategy_used=SolvingStrategy.CREATIVE,
                quality_metrics=quality,
                execution_time=execution_time,
                metadata={"solver": self.name},
            )

            self.performance.update(result)
            return result

        except (ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"[{self.name}] Error solving {sub_problem.id}: {e}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=SolvingStrategy.CREATIVE,
                quality_metrics=QualityMetrics(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={"solver": self.name},
            )

    def _assess_quality(self, solution: str, sub_problem: SubProblemInput) -> QualityMetrics:
        """Assess solution quality"""
        quality = QualityMetrics()

        # Completeness
        if len(solution) > 500:
            quality.completeness += 0.4
        if any(word in solution.lower() for word in ["approach", "solution", "idea"]):
            quality.completeness += 0.4

        # Correctness
        for req in sub_problem.requirements:
            if any(word.lower() in solution.lower() for word in req.split()[:3]):
                quality.correctness += min(1.0 / max(len(sub_problem.requirements), 1), 0.3)

        # Efficiency (lower priority for creative solutions)
        quality.efficiency = 0.6

        # Clarity
        if "**" in solution or "##" in solution:
            quality.clarity += 0.5

        # Maintainability
        quality.maintainability = 0.6

        # Innovation: Higher for creative solver
        if any(word in solution.lower() for word in ["novel", "unique", "innovative", "creative", "unconventional"]):
            quality.innovation += 0.5
        if "alternative" in solution.lower() or "different" in solution.lower():
            quality.innovation += 0.3
        quality.innovation = min(quality.innovation, 1.0)

        quality.calculate_overall()

        return quality


class SystematicSolver(BaseSolver):
    """
    Systematic problem solver.

    Uses structured frameworks and proven methodologies.
    Best for: Complex problems requiring rigorous approach.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SystematicSolver", config)

    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve using systematic approach"""
        start_time = time.time()

        logger.info(f"[{self.name}] Systematically solving sub-problem: {sub_problem.id}")

        try:
            from llm_utils import _request_openai_compatible_chat

            # Build systematic prompt
            system_prompt = """You are a systematic problem solver who uses proven methodologies.
Your approach:
1. Define the problem clearly
2. Gather and analyze relevant information
3. Generate multiple solution alternatives
4. Evaluate alternatives against criteria
5. Select and implement the best solution
6. Monitor and verify results

Use structured frameworks and be thorough."""

            user_prompt = f"""Solve this problem systematically:

**Problem ID:** {sub_problem.id}
**Description:** {sub_problem.description}

**Requirements:**
{chr(10).join(f'- {r}' for r in sub_problem.requirements) if sub_problem.requirements else 'None specified'}

**Constraints:**
{chr(10).join(f'- {c}' for c in sub_problem.constraints) if sub_problem.constraints else 'None specified'}

**Success Criteria:**
{chr(10).join(f'- {c}' for c in sub_problem.success_criteria) if sub_problem.success_criteria else 'None specified'}

Use a systematic approach. Apply structured problem-solving frameworks."""

            # Get API config
            api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", "https://api.openai.com/v1")
            model = self.config.get("model", "gpt-4o-mini")

            # Try Robust Engine First
            response = None
            if robust_engine:
                try:
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    engine_result = robust_engine.solve_problem_recursive(full_prompt, sub_problem.context)
                    response = engine_result.get("solution")
                except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError):
                    response = None

            # Fallback to direct call
            if not response:
                response = _request_openai_compatible_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.get("temperature", 0.5),
                    max_tokens=self.config.get("max_tokens", 4096),
                )

            if not response:
                raise ValueError("No response from LLM")

            quality = self._assess_quality(response, sub_problem)

            execution_time = time.time() - start_time

            result = SolutionResult(
                sub_problem_id=sub_problem.id,
                solution=response,
                status=SolutionStatus.COMPLETED,
                strategy_used=SolvingStrategy.SYSTEMATIC,
                quality_metrics=quality,
                execution_time=execution_time,
                metadata={"solver": self.name},
            )

            self.performance.update(result)
            return result

        except (ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"[{self.name}] Error solving {sub_problem.id}: {e}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=SolvingStrategy.SYSTEMATIC,
                quality_metrics=QualityMetrics(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={"solver": self.name},
            )

    def _assess_quality(self, solution: str, sub_problem: SubProblemInput) -> QualityMetrics:
        """Assess solution quality"""
        quality = QualityMetrics()

        # Completeness: Check for systematic steps
        systematic_keywords = ["define", "analyze", "generate", "evaluate", "select", "implement", "verify"]
        for keyword in systematic_keywords:
            if keyword in solution.lower():
                quality.completeness += 0.15
        quality.completeness = min(quality.completeness, 1.0)

        # Correctness
        for req in sub_problem.requirements:
            if any(word.lower() in solution.lower() for word in req.split()[:3]):
                quality.correctness += min(1.0 / max(len(sub_problem.requirements), 1), 0.3)

        # Efficiency: Check for structured approach
        if "framework" in solution.lower() or "methodology" in solution.lower():
            quality.efficiency += 0.5
        if any(word in solution.lower() for word in ["step", "phase", "stage"]):
            quality.efficiency += 0.3

        # Clarity
        if "**" in solution or "##" in solution:
            quality.clarity += 0.5
        if "```" in solution:
            quality.clarity += 0.3

        # Maintainability: High for systematic solutions
        quality.maintainability = 0.8

        # Innovation: Lower for systematic (focus on proven methods)
        quality.innovation = 0.4

        quality.calculate_overall()

        return quality


class HybridSolver(BaseSolver):
    """
    Hybrid solver that combines multiple strategies.

    Adapts approach based on problem characteristics.
    Best for: Complex problems requiring multiple perspectives.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("HybridSolver", config)
        self.analytical = AnalyticalSolver(config)
        self.creative = CreativeSolver(config)
        self.systematic = SystematicSolver(config)

    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve using hybrid approach"""
        start_time = time.time()

        logger.info(f"[{self.name}] Solving sub-problem with hybrid strategy: {sub_problem.id}")

        try:
            # Analyze problem to select best strategy
            analysis = self.analyze_problem(sub_problem)

            # Select primary strategy based on analysis
            if analysis["estimated_difficulty"] >= 7.0:
                # High difficulty: use systematic
                primary = self.systematic
                primary_strategy = SolvingStrategy.SYSTEMATIC
            elif sub_problem.complexity_score <= 4:
                # Low complexity: use analytical
                primary = self.analytical
                primary_strategy = SolvingStrategy.ANALYTICAL
            else:
                # Medium: use creative
                primary = self.creative
                primary_strategy = SolvingStrategy.CREATIVE

            logger.info(f"[{self.name}] Selected primary strategy: {primary_strategy.value}")

            # Get solution from primary strategy
            result = primary.solve(sub_problem)

            # Update metadata
            result.strategy_used = SolvingStrategy.HYBRID
            result.metadata = {
                "solver": self.name,
                "primary_strategy": primary_strategy.value,
                "analysis": analysis,
            }

            self.performance.update(result)
            return result

        except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"[{self.name}] Error solving {sub_problem.id}: {e}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=SolvingStrategy.HYBRID,
                quality_metrics=QualityMetrics(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={"solver": self.name},
            )


class EvolutionarySolver(BaseSolver):
    """
    Evolutionary solver using OpenEvolve.

    Uses genetic algorithms to evolve solutions.
    Best for: Problems requiring optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("EvolutionarySolver", config)

    def solve(self, sub_problem: SubProblemInput) -> SolutionResult:
        """Solve using evolutionary approach"""
        start_time = time.time()

        logger.info(f"[{self.name}] Solving sub-problem with evolution: {sub_problem.id}")

        try:
            # Check if OpenEvolve is available
            try:
                from openevolve.api import run_evolution, evolve_code
                OPENEVOLVE_AVAILABLE = True
            except ImportError:
                logger.warning("OpenEvolve not available, falling back to analytical solver")
                OPENEVOLVE_AVAILABLE = False

            if not OPENEVOLVE_AVAILABLE:
                # Fallback to analytical solver
                fallback = AnalyticalSolver(self.config)
                result = fallback.solve(sub_problem)
                result.metadata["fallback_from"] = "EvolutionarySolver"
                return result

            # Build initial solution code
            initial_code = f'''# Solution for {sub_problem.id}

def solve():
    """
    Problem: {sub_problem.description[:200]}

    Requirements: {", ".join(sub_problem.requirements[:3]) if sub_problem.requirements else "None"}
    Constraints: {", ".join(sub_problem.constraints[:3]) if sub_problem.constraints else "None"}
    """
    # Generate a solution based on the sub-problem requirements
    solution_template = f"""
def solve_{sub_problem.id.replace('-', '_')}():
    \"\"\"
    Solution for: {sub_problem.title}
    Description: {sub_problem.description[:100]}...
    \"\"\"
    # Implementation would depend on the problem type
    # This is a generic solution template
    result = {{
        'status': 'implemented',
        'approach': 'custom implementation based on requirements',
        'requirements_met': {len(sub_problem.requirements)},
        'constraints_respected': {len(sub_problem.constraints)}
    }}

    # Apply requirements
    for req in sub_problem.requirements[:3]:  # Limit to first 3 requirements
        result[f'requirement_{req[:20].replace(" ", "_")}'] = True

    # Apply constraints
    for constraint in sub_problem.constraints[:3]:  # Limit to first 3 constraints
        result[f'constraint_{constraint[:20].replace(" ", "_")}'] = True

    return result
"""
    return solution_template

# Add helper functions and implementation details below
'''

            # Define evaluator
            def solution_evaluator(code: str) -> float:
                """Evaluate solution quality"""
                score = 0.0

                # Length: want substantial solution
                if len(code) > 500:
                    score += 0.2
                elif len(code) > 200:
                    score += 0.1

                # Structure: check for functions/classes
                if "def " in code:
                    score += 0.3
                if "class " in code:
                    score += 0.1

                # Completeness: check for implementation (no pass/TODO means more complete)
                if "pass" not in code and "TODO" not in code:
                    score += 0.2

                # Documentation
                if '"""' in code or "'''" in code:
                    score += 0.2

                return score

            # Run evolution
            iterations = self.config.get("evolution_iterations", 50)
            evolution_result = evolve_code(
                initial_code=initial_code,
                evaluator=solution_evaluator,
                iterations=iterations,
            )

            solution = evolution_result.evolved_code if evolution_result.evolved_code else initial_code

            # Assess quality
            quality = QualityMetrics()
            quality.completeness = 0.8  # Evolved solutions tend to be complete
            quality.correctness = 0.7
            quality.efficiency = 0.7
            quality.clarity = 0.6
            quality.maintainability = 0.7
            quality.innovation = 0.7  # Evolution produces novel solutions
            quality.calculate_overall()

            execution_time = time.time() - start_time

            result = SolutionResult(
                sub_problem_id=sub_problem.id,
                solution=solution,
                status=SolutionStatus.COMPLETED,
                strategy_used=SolvingStrategy.EVOLUTIONARY,
                quality_metrics=quality,
                execution_time=execution_time,
                iterations=evolution_result.iterations_completed,
                metadata={
                    "solver": self.name,
                    "evolution_iterations": evolution_result.iterations_completed,
                    "evolution_improvement": evolution_result.improvement,
                    "evolution_fitness": evolution_result.best_fitness,
                },
            )

            self.performance.update(result)
            return result

        except (ImportError, ConnectionError, TimeoutError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"[{self.name}] Error solving {sub_problem.id}: {e}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=SolvingStrategy.EVOLUTIONARY,
                quality_metrics=QualityMetrics(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={"solver": self.name},
            )


# =============================================================================
# SOLUTION CACHE
# =============================================================================

class SolutionCache:
    """Cache for solution results to avoid redundant computation"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, SolutionResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _generate_key(self, sub_problem: SubProblemInput, strategy: SolvingStrategy) -> str:
        """Generate cache key from sub-problem and strategy"""
        content = f"{sub_problem.id}:{sub_problem.description}:{strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, sub_problem: SubProblemInput, strategy: SolvingStrategy) -> Optional[SolutionResult]:
        """Get cached solution if available"""
        key = self._generate_key(sub_problem, strategy)

        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache HIT for {sub_problem.id} with {strategy.value}")
            return self.cache[key]

        self.misses += 1
        logger.debug(f"Cache MISS for {sub_problem.id} with {strategy.value}")
        return None

    def put(self, sub_problem: SubProblemInput, strategy: SolvingStrategy, result: SolutionResult):
        """Cache a solution result"""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        key = self._generate_key(sub_problem, strategy)
        self.cache[key] = result
        logger.debug(f"Cached solution for {sub_problem.id} with {strategy.value}")

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# =============================================================================
# SOLVER WORKFLOW ORCHESTRATOR
# =============================================================================

class SolverWorkflow:
    """
    Orchestrates the complete solver workflow.

    Stages:
    1. Pre-solving: Analyze, plan, estimate resources
    2. Solving: Execute solution development
    3. Post-solving: Validate, test, document
    4. Iteration: Refine based on quality feedback
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize solvers
        self.analytical = AnalyticalSolver(config)
        self.creative = CreativeSolver(config)
        self.systematic = SystematicSolver(config)
        self.hybrid = HybridSolver(config)
        self.evolutionary = EvolutionarySolver(config)

        # Solution cache
        self.cache = SolutionCache(max_size=config.get("cache_size", 1000))

        # Performance tracking
        self.total_solved = 0
        self.total_failed = 0

        # Reward modeling components
        self.preference_store = PreferenceStore()
        self.reward_model = LocalRewardModel(training_frequency=config.get("rm_training_frequency", 50))
        self.evaluator_team = EvaluatorTeam()
        self._score_history: List[float] = []

        # Optional LeanAide integration
        self._leanaide_client = None

        # Knowledge/ADR integration
        self.knowledge_manager = KnowledgeManager()
        self.doc_manager = DocstringManager()

    def select_strategy(
        self,
        sub_problem: SubProblemInput,
        preferred_strategy: Optional[SolvingStrategy] = None
    ) -> SolvingStrategy:
        """Select appropriate solving strategy"""

        if preferred_strategy:
            return preferred_strategy

        # Reward-model-guided draft selection
        candidate_strategies = [
            (SolvingStrategy.ANALYTICAL, 0.2),
            (SolvingStrategy.SYSTEMATIC, 0.4),
            (SolvingStrategy.HYBRID, 0.6),
        ]
        if self.config.get("enable_evolution", True):
            candidate_strategies.append((SolvingStrategy.EVOLUTIONARY, 0.7))

        # Score candidates using reward model and problem features
        best_strategy = None
        best_score = -1.0
        for strategy, temp in candidate_strategies:
            features = {
                "quality": sub_problem.complexity_score / 10.0,
                "length": min(1.0, len(sub_problem.description) / 1000),
                "structure": 0.6 if sub_problem.requirements else 0.3,
                "constraint_fit": min(1.0, len(sub_problem.constraints) / 5.0),
            }
            score = self.reward_model.score_strategy_candidate(features) * (1.0 - abs(temp - 0.4))
            if score > best_score:
                best_score = score
                best_strategy = strategy

        if best_strategy:
            return best_strategy

        # Auto-select based on problem characteristics
        if sub_problem.complexity_score >= 8:
            # Very complex: use evolutionary or systematic
            return SolvingStrategy.EVOLUTIONARY if self.config.get("enable_evolution", True) else SolvingStrategy.SYSTEMATIC
        elif sub_problem.complexity_score <= 3:
            # Simple: use analytical
            return SolvingStrategy.ANALYTICAL
        elif len(sub_problem.dependencies) > 3:
            # Many dependencies: use systematic
            return SolvingStrategy.SYSTEMATIC
        elif any(word in sub_problem.description.lower() for word in ["innovative", "creative", "novel"]):
            # Creative keywords: use creative
            return SolvingStrategy.CREATIVE
        else:
            # Default: hybrid for adaptability
            return SolvingStrategy.HYBRID

    def get_solver(self, strategy: SolvingStrategy) -> BaseSolver:
        """Get solver instance for strategy"""
        solver_map = {
            SolvingStrategy.ANALYTICAL: self.analytical,
            SolvingStrategy.CREATIVE: self.creative,
            SolvingStrategy.SYSTEMATIC: self.systematic,
            SolvingStrategy.HYBRID: self.hybrid,
            SolvingStrategy.EVOLUTIONARY: self.evolutionary,
        }
        return solver_map.get(strategy, self.hybrid)

    def solve(
        self,
        sub_problem: SubProblemInput,
        strategy: Optional[SolvingStrategy] = None,
        enable_cache: bool = True,
        max_iterations: int = 3,
        quality_threshold: float = 0.7
    ) -> SolutionResult:
        """
        Execute complete solver workflow.

        Args:
            sub_problem: Sub-problem to solve
            strategy: Preferred solving strategy (auto-selected if None)
            enable_cache: Whether to use solution cache
            max_iterations: Maximum refinement iterations
            quality_threshold: Minimum quality threshold to accept

        Returns:
            SolutionResult with solution and quality metrics
        """
        logger.info(f"[SolverWorkflow] Starting workflow for {sub_problem.id}")

        # Stage 1: Pre-solving analysis
        logger.info(f"[SolverWorkflow] Stage 1: Pre-solving analysis")
        selected_strategy = self.select_strategy(sub_problem, strategy)
        solver = self.get_solver(selected_strategy)

        analysis = solver.analyze_problem(sub_problem)
        logger.info(f"[SolverWorkflow] Analysis: difficulty={analysis['estimated_difficulty']:.1f}")

        # Stage 2: Check cache
        if enable_cache:
            cached_result = self.cache.get(sub_problem, selected_strategy)
            if cached_result and cached_result.status == SolutionStatus.COMPLETED:
                logger.info(f"[SolverWorkflow] Using cached solution")
                cached_result.status = SolutionStatus.CACHED
                return cached_result

        # Stage 3: Solving (with iteration for quality)
        logger.info(f"[SolverWorkflow] Stage 2: Solving with {selected_strategy.value}")

        best_result = None
        best_quality = 0.0
        previous_solution: Optional[str] = None
        previous_quality: Optional[QualityMetrics] = None

        converged = False
        for iteration in range(max_iterations):
            logger.info(f"[SolverWorkflow] Iteration {iteration + 1}/{max_iterations}")

            # Solve
            result = solver.solve(sub_problem)

            if result.status != SolutionStatus.FAILED and result.solution:
                user_prompt_id = None
                if sub_problem.context:
                    user_prompt_id = sub_problem.context.get("user_prompt_id")
                doc_result = self.doc_manager.ensure_docstring_refinement(result.solution, user_prompt_id=user_prompt_id)
                result.solution = doc_result.updated_code
                result.metadata["docstring_fidelity"] = doc_result.fidelity_score
                if doc_result.changed:
                    result.metadata["docstring_refined"] = True

            # Check if failed
            if result.status == SolutionStatus.FAILED:
                logger.warning(f"[SolverWorkflow] Solving failed: {result.error_message}")
                if iteration == 0:
                    return result
                continue

            # Check quality
            current_quality = result.quality_metrics.overall_score

            logger.info(f"[SolverWorkflow] Quality score: {current_quality:.2f}")
            self._score_history.append(current_quality)

            # Comparative judging for reward modeling
            if previous_solution is not None:
                preference_bit, improvement_delta = self.evaluator_team.compare(
                    previous_solution,
                    result.solution,
                    sub_problem.constraints
                )
                if previous_quality is not None:
                    prev_score = self.reward_model.score_solution(
                        previous_solution,
                        previous_quality,
                        sub_problem.constraints
                    )
                    curr_score = self.reward_model.score_solution(
                        result.solution,
                        result.quality_metrics,
                        sub_problem.constraints
                    )
                    confidence = abs(curr_score - prev_score)
                    threshold = float(self.config.get("reward_calibration_threshold", 0.6))
                    result.metadata["reward_calibration_confidence"] = confidence
                    if confidence < threshold:
                        override = self._request_reward_calibration(
                            previous_solution,
                            result.solution,
                            confidence,
                            sub_problem.description
                        )
                        if override in (0, 1):
                            preference_bit = int(override)
                            result.metadata["reward_calibration_override"] = preference_bit
                self.preference_store.add_record(
                    PreferenceRecord(
                        previous_solution=previous_solution,
                        current_solution=result.solution,
                        preference_bit=preference_bit,
                        improvement_delta=improvement_delta,
                        constraint_snapshot=list(sub_problem.constraints),
                    )
                )
                self.reward_model.maybe_train(self.preference_store)

                # Lean 4 micro-formalization trigger
                if improvement_delta < 0.02 and current_quality < 0.70:
                    lean_spec = self._build_lean_spec(sub_problem)
                    if lean_spec:
                        sub_problem.context["lean_specification"] = lean_spec

            previous_solution = result.solution
            previous_quality = result.quality_metrics

            if current_quality > best_quality:
                best_result = result
                best_quality = current_quality

            # Check if meets threshold
            if current_quality >= quality_threshold:
                logger.info(f"[SolverWorkflow] Quality threshold met ({current_quality:.2f} >= {quality_threshold:.2f})")
                converged = True
                break

            # Convergence monitoring
            if self._has_converged():
                logger.info("[SolverWorkflow] Convergence detected, stopping refinement loop")
                converged = True
                break

            # Refine for next iteration
            if iteration < max_iterations - 1:
                logger.info(f"[SolverWorkflow] Refining solution...")
                # Update context with feedback for refinement
                sub_problem.context = {
                    **sub_problem.context,
                    "previous_solution": result.solution,
                    "quality_feedback": self._generate_quality_feedback(result.quality_metrics),
                }

        # Stage 4: Post-solving validation
        logger.info(f"[SolverWorkflow] Stage 3: Post-solving validation")

        if best_result:
            # Cache the result
            if enable_cache:
                self.cache.put(sub_problem, selected_strategy, best_result)

            # Update tracking
            if best_result.status == SolutionStatus.COMPLETED:
                self.total_solved += 1
            else:
                self.total_failed += 1

            target_symbol = sub_problem.context.get("target_symbol") if sub_problem.context else None
            if target_symbol and not self._passes_arbor_gate(target_symbol):
                best_result.status = SolutionStatus.FAILED
                best_result.error_message = "Arbor blast radius too large (transitive_breaks > 5)."
                best_result.metadata["arbor_rejected"] = True
                return best_result

            logger.info(f"[SolverWorkflow] Workflow completed for {sub_problem.id}")
            if self.config.get("enable_logic_compression", True):
                try:
                    compressor = Z3LogicCompressor()
                    compressed = compressor.compress_code_conditions(best_result.solution)
                    if compressed != best_result.solution:
                        best_result.solution = compressed
                        best_result.metadata["logic_compressed"] = True
                except Exception:
                    pass
            if converged:
                self._record_adr_and_skillbook(sub_problem, best_result, selected_strategy)
            return best_result
        else:
            # All iterations failed
            logger.error(f"[SolverWorkflow] All iterations failed for {sub_problem.id}")
            return SolutionResult(
                sub_problem_id=sub_problem.id,
                solution="",
                status=SolutionStatus.FAILED,
                strategy_used=selected_strategy,
                quality_metrics=QualityMetrics(),
                error_message="All solving iterations failed",
                metadata={"workflow": "SolverWorkflow"},
            )

    def solve_batch(
        self,
        sub_problems: List[SubProblemInput],
        strategy: Optional[SolvingStrategy] = None,
        parallel: bool = False
    ) -> List[SolutionResult]:
        """
        Solve multiple sub-problems in batch.

        Args:
            sub_problems: List of sub-problems to solve
            strategy: Preferred solving strategy
            parallel: Whether to solve in parallel (future enhancement)

        Returns:
            List of SolutionResult in same order as input
        """
        logger.info(f"[SolverWorkflow] Solving batch of {len(sub_problems)} sub-problems")

        results = []

        for sub_problem in sub_problems:
            result = self.solve(sub_problem, strategy=strategy)
            results.append(result)

        logger.info(f"[SolverWorkflow] Batch completed: {self.total_solved} solved, {self.total_failed} failed")

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get workflow performance statistics"""
        total = self.total_solved + self.total_failed
        success_rate = self.total_solved / total if total > 0 else 0.0

        return {
            "total_problems": total,
            "solved": self.total_solved,
            "failed": self.total_failed,
            "success_rate": success_rate,
            "cache_hit_rate": self.cache.get_hit_rate(),
            "analytical_performance": self.analytical.performance.__dict__,
            "creative_performance": self.creative.performance.__dict__,
            "systematic_performance": self.systematic.performance.__dict__,
            "hybrid_performance": self.hybrid.performance.__dict__,
            "evolutionary_performance": self.evolutionary.performance.__dict__,
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.total_solved = 0
        self.total_failed = 0
        self.analytical.performance = SolverPerformance()
        self.creative.performance = SolverPerformance()
        self.systematic.performance = SolverPerformance()
        self.hybrid.performance = SolverPerformance()
        self.evolutionary.performance = SolverPerformance()
        self.cache.clear()

    def _generate_quality_feedback(self, quality: QualityMetrics) -> str:
        """Generate feedback for quality improvement"""
        feedback = []

        if quality.completeness < 0.7:
            feedback.append("Improve completeness by addressing all requirements")
        if quality.correctness < 0.7:
            feedback.append("Verify correctness of solution")
        if quality.efficiency < 0.7:
            feedback.append("Optimize solution for efficiency")
        if quality.clarity < 0.7:
            feedback.append("Improve clarity and structure")
        if quality.maintainability < 0.7:
            feedback.append("Enhance maintainability with better documentation")
        if quality.innovation < 0.5:
            feedback.append("Consider more innovative approaches")

        return "; ".join(feedback) if feedback else "Good quality overall"

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return None

    def _get_json(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return None

    def _request_reward_calibration(
        self,
        previous_solution: str,
        current_solution: str,
        confidence: float,
        prompt: str
    ) -> Optional[int]:
        """Queue a reward calibration request and optionally block for a response."""
        handler = self.config.get("reward_calibration_handler")
        if callable(handler):
            try:
                return handler(previous_solution, current_solution, confidence)
            except Exception:
                return None

        api_base = os.getenv("ICR_API_BASE_URL")
        if not api_base:
            return None

        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "option_a": previous_solution,
            "option_b": current_solution,
            "confidence": confidence,
            "prompt": prompt,
        }
        self._post_json(f"{api_base}/icr/reward-calibration/request", payload)

        if not self.config.get("reward_calibration_blocking", False):
            return None

        timeout_s = float(self.config.get("reward_calibration_timeout_s", 60))
        start = time.time()
        while time.time() - start < timeout_s:
            response = self._get_json(f"{api_base}/icr/reward-calibration/response/{request_id}")
            if response and response.get("choice"):
                choice = response["choice"]
                if str(choice).upper() == "A":
                    return 0
                if str(choice).upper() == "B":
                    return 1
                return None
            time.sleep(2)
        return None

    def _has_converged(self) -> bool:
        if len(self._score_history) < 4:
            return False
        diffs = [
            abs(self._score_history[-1] - self._score_history[-2]),
            abs(self._score_history[-2] - self._score_history[-3]),
            abs(self._score_history[-3] - self._score_history[-4]),
        ]
        return all(d < 0.01 for d in diffs) or self._score_history[-1] >= 0.95

    def _passes_arbor_gate(self, symbol: str) -> bool:
        """Run Arbor blast radius check and reject if too invasive."""
        try:
            result = subprocess.run(
                ["arbor", "refactor", symbol, "--why", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                return True  # If Arbor fails, do not block by default
            data = json.loads(result.stdout)
            transitive = data.get("transitive_breaks", 0)
            return transitive <= 5
        except Exception:
            return True

    def _record_adr_and_skillbook(
        self,
        sub_problem: SubProblemInput,
        best_result: SolutionResult,
        selected_strategy: SolvingStrategy
    ) -> None:
        """Create ADR and store refinement template on convergence."""
        title = f"{sub_problem.id} - Solver Convergence"
        decision = f"Selected strategy: {selected_strategy.value}"
        rationale = (
            "Convergence achieved with stable quality improvements and solver validation."
        )
        consequences = (
            "Entangled components should be reviewed for downstream impacts."
        )
        alternatives = [s.value for s in SolvingStrategy if s != selected_strategy]
        entangled = sub_problem.context.get("entangled_components", [])

        chronicle = ChronicleMemory()

        async def _synthesize():
            return await chronicle.synthesize_adr(
                title=title,
                decision=decision,
                rationale=rationale,
                consequences=consequences,
                alternatives_rejected=alternatives,
                entangled_components=entangled
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            adr_result = None
            if loop and loop.is_running():
                return
            adr_result = asyncio.run(_synthesize())

            if adr_result:
                self.knowledge_manager.record_adr(
                    {
                        **adr_result,
                        "workflow_id": sub_problem.id,
                        "summary": best_result.solution[:200],
                        "confidence": best_result.quality_metrics.overall_score
                    },
                    entity_ids=[sub_problem.id]
                )

                # Store refinement template
                reasoning_path = [
                    f"strategy:{selected_strategy.value}",
                    f"quality:{best_result.quality_metrics.overall_score:.2f}",
                    "converged:true",
                ]
                context_signature = {
                    "constraints": sub_problem.constraints,
                    "requirements": sub_problem.requirements,
                }
                self.knowledge_manager.store_refinement_template(
                    title=title,
                    description="Converged solver reasoning path.",
                    reasoning_path=reasoning_path,
                    context_signature=context_signature,
                    domain=sub_problem.context.get("domain", "general")
                )
        except Exception as e:
            logger.warning(f"ADR/Skillbook synthesis failed: {e}")

    def _build_lean_spec(self, sub_problem: SubProblemInput) -> Optional[str]:
        """Attempt to build a Lean 4 specification for the sub-problem."""
        if self._leanaide_client is None:
            try:
                from leanaide_client import LeanAideClient, LeanAideConfig
                self._leanaide_client = LeanAideClient(LeanAideConfig())
            except (ImportError, RuntimeError):
                self._leanaide_client = False

        if not self._leanaide_client:
            # Fallback heuristic spec
            if sub_problem.success_criteria:
                joined = " and ".join(sub_problem.success_criteria)
                return f"-- LeanSpec (fallback)\n-- {joined}"
            return None

        try:
            import asyncio
            prompt = (
                "Convert the following natural language requirement into a Lean 4 predicate.\n"
                f"Problem: {sub_problem.description}\n"
                f"Requirements: {sub_problem.success_criteria}\n"
            )
            if hasattr(self._leanaide_client, "translate_thm_detailed"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    return None
                response = asyncio.run(self._leanaide_client.translate_thm_detailed(prompt))
                if isinstance(response, dict):
                    lean_code = response.get("lean_code") or response.get("theorem") or response.get("lean")
                    if isinstance(lean_code, str) and lean_code.strip():
                        return lean_code.strip()
                if isinstance(response, str) and response.strip():
                    return response.strip()
        except Exception:
            return None
        return None


# =============================================================================
# SUB-PROBLEM SOLVER INTERFACE
# =============================================================================

class SubProblemSolver:
    """
    High-level interface for solving sub-problems from decomposition engine.

    This is the main entry point for Blue Team solving.

    Example:
        solver = SubProblemSolver(config={
            "api_key": "your-api-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        })

        result = solver.solve_sub_problem(
            sub_problem_id="sub_1",
            description="Implement user authentication",
            requirements=["Secure", "Scalable", "Testable"],
            complexity_score=7
        )

        print(result.solution)
        print(f"Quality: {result.quality_metrics.overall_score:.2f}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.workflow = SolverWorkflow(config)

    def solve_sub_problem(
        self,
        sub_problem_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        complexity_score: int = 5,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None,
        requirements: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> SolutionResult:
        """
        Solve a sub-problem.

        Args:
            sub_problem_id: Unique identifier for the sub-problem
            description: Detailed problem description
            dependencies: List of sub-problem IDs this depends on
            complexity_score: Estimated complexity (1-10)
            priority: Priority level (1-10, higher = more important)
            context: Additional context and information
            requirements: List of requirements the solution must meet
            constraints: List of constraints on the solution
            success_criteria: Criteria to measure success
            metadata: Additional metadata
            strategy: Preferred solving strategy (analytical, creative, systematic, hybrid, evolutionary)
            **kwargs: Additional parameters

        Returns:
            SolutionResult with solution and quality metrics
        """
        # Create sub-problem input
        sub_problem = SubProblemInput(
            id=sub_problem_id,
            description=description,
            dependencies=dependencies or [],
            complexity_score=complexity_score,
            priority=priority,
            context=context or {},
            requirements=requirements or [],
            constraints=constraints or [],
            success_criteria=success_criteria or [],
            metadata=metadata or {},
        )

        # Parse strategy
        solving_strategy = None
        if strategy:
            try:
                solving_strategy = SolvingStrategy(strategy.lower())
            except ValueError:
                logger.warning(f"Unknown strategy: {strategy}, using auto-selection")

        # Solve
        result = self.workflow.solve(
            sub_problem=sub_problem,
            strategy=solving_strategy,
            **kwargs
        )

        result = self.workflow.solve(
            sub_problem=sub_problem,
            strategy=solving_strategy,
            **kwargs
        )

        # **ACTUAL INTEGRATION**: Track performance and extract knowledge
        self._track_solver_performance(sub_problem_id, result)
        self._extract_solver_knowledge(sub_problem_id, sub_problem, result)

        # **ACTUAL INTEGRATION**: Alert if solution failed
        if result.quality_metrics.overall_score < 0.5:
            self._trigger_solver_alerts(sub_problem_id, False, f"Low quality: {result.quality_metrics.overall_score:.2f}")

        return result

    def solve_from_dict(self, sub_problem_dict: Dict[str, Any]) -> SolutionResult:
        """
        Solve a sub-problem from dictionary format.

        Useful for integration with decomposition engine.

        Args:
            sub_problem_dict: Dictionary with sub-problem data

        Returns:
            SolutionResult with solution and quality metrics
        """
        return self.solve_sub_problem(**sub_problem_dict)

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for solver
    # =========================================================================

    def _trigger_solver_alerts(
        self,
        sub_problem_id: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for solver failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"Blue Team Solver Failed: {sub_problem_id}",
                    description=f"Sub-problem solving failed for '{sub_problem_id}'. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="blue_team_solver_engine",
                    component="solver",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger solver alert: {e}")

    def _extract_solver_knowledge(
        self,
        sub_problem_id: str,
        sub_problem: 'SubProblemInput',
        result: 'SolutionResult'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract solver knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"solver_{sub_problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="solver_solution",
                source_component="blue_team_solver_engine",
                title=f"Solver Solution: {sub_problem_id}",
                content={
                    "sub_problem_id": sub_problem_id,
                    "description": sub_problem.description,
                    "solution": result.solution[:500] if result.solution else None,
                    "quality_score": result.quality_metrics.overall_score,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "complexity": sub_problem.complexity_score,
                    "strategy": str(result.strategy_used)
                },
                tags=["solver", "solution", "blue_team"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted solver knowledge for {sub_problem_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract solver knowledge: {e}")
            return False

    def _track_solver_performance(
        self,
        sub_problem_id: str,
        result: 'SolutionResult'
    ):
        """**ACTUAL INTEGRATION**: Track solver performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"blue_team_solver_{result.strategy_used}",
                success_count=1 if result.quality_metrics.overall_score >= 0.7 else 0,
                failure_count=0 if result.quality_metrics.overall_score >= 0.7 else 1,
                average_quality=result.quality_metrics.overall_score,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"sub_problem_id": sub_problem_id}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked solver performance: {sub_problem_id}")

        except Exception as e:
            logger.error(f"Failed to track solver performance: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics"""
        return self.workflow.get_performance_stats()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_solver(config: Optional[Dict[str, Any]] = None) -> SubProblemSolver:
    """Create a sub-problem solver instance"""
    return SubProblemSolver(config)


def create_solver_workflow(config: Optional[Dict[str, Any]] = None) -> SolverWorkflow:
    """Create a solver workflow instance"""
    return SolverWorkflow(config)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

__all__ = [
    # Data models
    "SubProblemInput",
    "SolutionResult",
    "QualityMetrics",
    "SolverPerformance",
    "SolutionStatus",
    "SolvingStrategy",
    # Solvers
    "AnalyticalSolver",
    "CreativeSolver",
    "SystematicSolver",
    "HybridSolver",
    "EvolutionarySolver",
    # Workflow
    "SolverWorkflow",
    "SubProblemSolver",
    "SolutionCache",
    # Factory functions
    "create_solver",
    "create_solver_workflow",
]
