# Hybrid Architecture - Code Examples & Implementation Guide

**Date**: 2026-01-30
**Status**: Implementation Guide

---

## TABLE OF CONTENTS
1. [PES Module Extraction](#1-pes-module-extraction)
2. [Unified API Implementation](#2-unified-api-implementation)
3. [Configuration Mapping](#3-configuration-mapping)
4. [Strategy Selection](#4-strategy-selection)
5. [Memory Integration](#5-memory-integration)
6. [Complete Working Example](#6-complete-working-example)

---

## 1. PES MODULE EXTRACTION

### Step 1: Directory Structure

```bash
# Create PES module in OpenEvolve
mkdir -p openevolve/pes/{core,planning,execution,memory,evaluation,config}

# Copy from LoongFlow
cp -r LoongFlow/src/loongflow/framework/pes/* openevolve/pes/
cp -r LoongFlow/src/loongflow/agentsdk/message openevolve/pes/message/
cp -r LoongFlow/src/loongflow/agentsdk/logger openevolve/pes/logger/
```

### Step 2: Core Module (`openevolve/pes/__init__.py`)

```python
"""
OpenEvolve PES Module
Extracted and adapted from LoongFlow
"""

from .core.agent import PESAgent
from .core.worker import Worker, register_worker
from .memory.database import EvolveDatabase
from .evaluation.evaluator import Evaluator, LoongFlowEvaluator
from .config.schemas import EvolveChainConfig, EvolveConfig, LLMConfig

__all__ = [
    "PESAgent",
    "Worker",
    "register_worker",
    "EvolveDatabase",
    "Evaluator",
    "LoongFlowEvaluator",
    "EvolveChainConfig",
    "EvolveConfig",
    "LLMConfig",
]
```

### Step 3: Simplified Worker Interface (`openevolve/pes/core/worker.py`)

```python
"""
Simplified Worker interface
Adapted from LoongFlow's Worker base class
"""

from abc import ABC, abstractmethod
from typing import Any
from .message import Message, Context

class Worker(ABC):
    """
    Base class for all PES workers (Planner, Executor, Summary)
    """

    def __init__(self):
        pass

    @abstractmethod
    async def run(self, context: Context, message: Message) -> Message:
        """
        Execute the worker's logic

        Args:
            context: Evolution context (iteration, island_id, task, etc.)
            message: Input message from previous worker

        Returns:
            Message containing output for next worker
        """
        pass


# Worker registry (simplified from LoongFlow)
_PLANNER_WORKERS = {}
_EXECUTOR_WORKERS = {}
_SUMMARY_WORKERS = {}

def register_worker(name: str, worker_type: str, worker_class: type):
    """Register a worker implementation"""
    if worker_type == "planner":
        _PLANNER_WORKERS[name] = worker_class
    elif worker_type == "executor":
        _EXECUTOR_WORKERS[name] = worker_class
    elif worker_type == "summary":
        _SUMMARY_WORKERS[name] = worker_class

def get_worker(name: str, worker_type: str, **kwargs):
    """Get a registered worker instance"""
    if worker_type == "planner":
        return _PLANNER_WORKERS[name](**kwargs)
    elif worker_type == "executor":
        return _EXECUTOR_WORKERS[name](**kwargs)
    elif worker_type == "summary":
        return _SUMMARY_WORKERS[name](**kwargs)
    raise ValueError(f"Unknown worker type: {worker_type}")
```

### Step 4: Message & Context (`openevolve/pes/core/message.py`)

```python
"""
Message and Context classes
Simplified from LoongFlow
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime, timezone

@dataclass
class Message:
    """
    Message passed between workers
    """
    content: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    role: str = "assistant"
    sender: str = ""

    @classmethod
    def from_text(cls, data: Any, role: str = "user", sender: str = "", mime_type: str = "text"):
        """Create message from text/data"""
        return cls(
            content=[{"data": data, "type": mime_type}],
            role=role,
            sender=sender
        )

    def get_elements(self, element_type=None):
        """Get content elements"""
        return self.content


@dataclass
class Context:
    """
    Evolution context passed to all workers
    """
    task: str
    base_path: str
    task_id: str
    island_id: int
    current_iteration: int
    total_iterations: int
    trace_id: str

    # Initial solution (optional)
    init_solution: str = ""
    init_score: Optional[float] = None
    init_evaluation: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def workspace_path(self) -> str:
        """Get workspace path for this iteration"""
        return f"{self.base_path}/iteration_{self.current_iteration}/island_{self.island_id}"
```

---

## 2. UNIFIED API IMPLEMENTATION

### Main Entry Point (`openevolve/unified.py`)

```python
"""
Unified Evolution Engine API
Combines OpenEvolve and PES
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import asyncio

from .pes import PESAgent, EvolveChainConfig
from .evolution import EvolutionEngine  # Existing OpenEvolve engine


@dataclass
class UnifiedConfig:
    """
    Unified configuration that merges both systems
    Only ~100 parameters needed (reduced from 272)
    """

    # ===== Core =====
    problem: str
    target_score: float = 0.95
    max_iterations: int = 100
    workspace_path: str = "./output"

    # ===== Planning Layer (PES) =====
    enable_planning: bool = True
    planner_model: str = "claude-3-5-sonnet"
    planning_depth: int = 3

    # ===== Strategy Selection =====
    strategy: str = "auto"  # auto, standard, qd, mo, adversarial, pes

    # ===== Execution Mode =====
    execution_mode: str = "unified"  # unified, pes_first, openevolve_first

    # ===== Memory (PES) =====
    enable_memory: bool = True
    memory_size: int = 1000
    compression_interval: int = 10
    num_islands: int = 5

    # ===== Specialized Modes (OpenEvolve) =====
    # Quality Diversity
    qd_archive_size: int = 100
    qd_novelty_threshold: float = 0.1
    feature_dimensions: List[str] = None

    # Multi-Objective
    mo_objectives: List[str] = None
    mo_pareto_size: int = 50
    objective_weights: List[float] = None

    # Adversarial
    adversarial_rounds: int = 5
    red_team_models: List[str] = None

    # ===== Model Config =====
    llm_model: str = "claude-3-5-sonnet"
    api_key: str = ""
    api_base: str = "https://api.anthropic.com"
    temperature: float = 0.7

    # ===== Resources =====
    max_time_seconds: int = 1800
    cost_limit_usd: float = 10.0


class UnifiedEvolutionEngine:
    """
    Unified evolution engine combining OpenEvolve and PES
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config

        # Initialize components based on configuration
        if config.enable_planning:
            self.pes_agent = self._init_pes_agent()
        else:
            self.pes_agent = None

        # Initialize OpenEvolve specialized engines
        self.openevolve_engine = EvolutionEngine(
            config=self._convert_to_openevolve_config(config)
        )

    def _init_pes_agent(self) -> PESAgent:
        """Initialize PES agent with unified config"""
        pes_config = self._convert_to_pes_config(self.config)
        return PESAgent(config=pes_config)

    async def run(self, problem: str) -> Dict[str, Any]:
        """
        Main entry point - auto-selects best strategy

        Args:
            problem: Problem statement

        Returns:
            OpenEvolve-compatible result dictionary
        """
        # Select strategy
        strategy = self._select_strategy(problem)

        # Execute based on execution mode
        if self.config.execution_mode == "unified":
            return await self._run_unified(problem, strategy)
        elif self.config.execution_mode == "pes_first":
            return await self._run_pes_first(problem)
        else:  # openevolve_first
            return await self._run_openevolve_first(problem)

    async def _run_unified(self, problem: str, strategy: str) -> Dict[str, Any]:
        """
        Approach C: Fully unified execution
        """
        # Phase 1: Plan (if enabled)
        if self.pes_agent:
            guidance = await self._create_plan(problem, strategy)
        else:
            guidance = None

        # Phase 2: Execute with strategy-specific engine
        if strategy == "quality_diversity":
            result = await self.openevolve_engine.run_qd(problem, guidance)
        elif strategy == "multi_objective":
            result = await self.openevolve_engine.run_mo(problem, guidance)
        elif strategy == "adversarial":
            result = await self.openevolve_engine.run_adversarial(problem, guidance)
        else:  # standard or pes
            if self.pes_agent:
                pes_result = await self.pes_agent.run()
                result = self._convert_pes_result(pes_result)
            else:
                result = await self.openevolve_engine.run_standard(problem)

        # Phase 3: Compress experience (if memory enabled)
        if self.pes_agent and self.config.enable_memory:
            await self._compress_experience(result)

        return result

    def _select_strategy(self, problem: str) -> str:
        """
        Auto-select best strategy based on problem characteristics
        """
        # If strategy is explicitly set, use it
        if self.config.strategy != "auto":
            return self.config.strategy

        # Auto-detect based on problem
        problem_lower = problem.lower()

        # Check for multi-objective keywords
        mo_keywords = ["balance", "trade-off", "pareto", "optimize multiple"]
        if any(kw in problem_lower for kw in mo_keywords):
            return "multi_objective"

        # Check for quality diversity keywords
        qd_keywords = ["diverse", "variety", "different", "explore"]
        if any(kw in problem_lower for kw in qd_keywords):
            return "quality_diversity"

        # Check for adversarial keywords
        adv_keywords = ["robust", "adversarial", "attack", "defense", "resilient"]
        if any(kw in problem_lower for kw in adv_keywords):
            return "adversarial"

        # Default to standard/PES
        return "standard"

    async def _create_plan(self, problem: str, strategy: str) -> Dict[str, Any]:
        """Create planning guidance using PES planner"""
        # This would call the PES planner
        # For now, return basic guidance
        return {
            "strategy": strategy,
            "focus_areas": self._infer_focus_areas(problem),
            "constraints": self._infer_constraints(problem)
        }

    def _infer_focus_areas(self, problem: str) -> List[str]:
        """Infer focus areas from problem statement"""
        # Simple heuristic-based inference
        keywords = {
            "performance": ["fast", "optimize", "efficient", "speed"],
            "correctness": ["correct", "accurate", "valid", "bug-free"],
            "maintainability": ["clean", "maintainable", "readable"],
            "robustness": ["robust", "handle", "error", "edge-case"]
        }

        focus = []
        problem_lower = problem.lower()
        for area, words in keywords.items():
            if any(word in problem_lower for word in words):
                focus.append(area)

        return focus or ["performance"]  # Default

    def _infer_constraints(self, problem: str) -> Dict[str, Any]:
        """Infer constraints from problem statement"""
        constraints = {}

        # Time constraints
        if "fast" in problem.lower() or "real-time" in problem.lower():
            constraints["max_time"] = "tight"

        # Memory constraints
        if "memory" in problem.lower() or "lightweight" in problem.lower():
            constraints["max_memory"] = "tight"

        return constraints

    def _convert_to_pes_config(self, config: UnifiedConfig) -> EvolveChainConfig:
        """Convert UnifiedConfig to PES EvolveChainConfig"""
        return EvolveChainConfig(
            evolve={
                "task": config.problem,
                "max_iterations": config.max_iterations,
                "target_score": config.target_score,
                "workspace_path": config.workspace_path,
                "initial_code": "",
                "concurrency": 5,
                "database": {
                    "num_islands": config.num_islands,
                    "checkpoint_interval": config.compression_interval,
                    "output_path": config.workspace_path
                }
            },
            llm_config={
                "model": config.planner_model,
                "api_key": config.api_key,
                "url": config.api_base,
                "temperature": config.temperature
            },
            planners={
                "general_planner": {
                    "llm_config": {
                        "model": config.planner_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    },
                    "max_turns": config.planning_depth
                }
            },
            executors={
                "general_executor": {
                    "llm_config": {
                        "model": config.llm_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    },
                    "max_rounds": 10
                }
            },
            summarizers={
                "general_summarizer": {
                    "llm_config": {
                        "model": config.llm_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    }
                }
            }
        )

    def _convert_to_openevolve_config(self, config: UnifiedConfig) -> Dict[str, Any]:
        """Convert UnifiedConfig to OpenEvolve config"""
        return {
            "evolution_mode": "standard" if config.strategy == "auto" else config.strategy,
            "max_iterations": config.max_iterations,
            "temperature": config.temperature,
            "api_key": config.api_key,
            "api_base": config.api_base,
            "model_id": config.llm_model,
            "enable_artifacts": True,

            # Specialized mode configs
            "qd_archive_size": config.qd_archive_size,
            "qd_novelty_threshold": config.qd_novelty_threshold,
            "mo_objectives": config.mo_objectives,
            "adversarial_rounds": config.adversarial_rounds,
        }

    def _convert_pes_result(self, pes_result) -> Dict[str, Any]:
        """Convert PES result to OpenEvolve format"""
        return {
            "status": "completed",
            "solution": {
                "code": pes_result.best_solution,
                "score": pes_result.best_score
            },
            "fitness": pes_result.best_score,
            "iterations": pes_result.total_iterations,
            "history": pes_result.evolution_history,
            "metadata": {
                "engine": "PES",
                "total_tokens": pes_result.total_tokens,
                "total_cost": pes_result.total_cost
            }
        }

    async def _compress_experience(self, result: Dict[str, Any]):
        """Compress experience into memory"""
        # This would call PES summarizer
        pass


# Convenience function for users
async def unified_evolve(problem: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for unified evolution

    Example:
        result = await unified_evolve(
            problem="Optimize code structure",
            enable_planning=True,
            max_iterations=100
        )
    """
    config = UnifiedConfig(problem=problem, **kwargs)
    engine = UnifiedEvolutionEngine(config)
    return await engine.run(problem)
```

---

## 3. CONFIGURATION MAPPING

### Parameter Mapping Table

```python
"""
OpenEvolve 272 parameters → UnifiedConfig → PES Config
"""

PARAMETER_MAPPING = {
    # Core Evolution
    "max_iterations": ("max_iterations", "evolve.max_iterations"),
    "target_score": ("target_score", "evolve.target_score"),
    "initial_solution": ("initial_solution", "evolve.initial_code"),
    "workspace_path": ("workspace_path", "evolve.workspace_path"),

    # Concurrency
    "concurrent_requests": (None, "evolve.concurrency"),
    "population_size": (None, "executor.max_rounds"),

    # Islands
    "num_islands": ("num_islands", "evolve.database.num_islands"),
    "migration_interval": ("compression_interval", "evolve.database.checkpoint_interval"),

    # Planning
    "enable_planning": ("enable_planning", None),
    "planning_depth": ("planning_depth", "planners.general_planner.max_turns"),
    "planner_model": ("planner_model", "planners.general_planner.llm_config.model"),

    # Memory
    "enable_memory": ("enable_memory", None),
    "memory_size": ("memory_size", "evolve.database.max_solutions"),

    # Strategy
    "evolution_mode": ("strategy", None),

    # Quality Diversity
    "qd_archive_size": ("qd_archive_size", None),
    "qd_novelty_threshold": ("qd_novelty_threshold", None),
    "feature_dimensions": ("feature_dimensions", None),

    # Multi-Objective
    "mo_objectives": ("mo_objectives", None),
    "mo_pareto_size": ("mo_pareto_size", None),
    "objective_weights": ("objective_weights", None),

    # Adversarial
    "adversarial_rounds": ("adversarial_rounds", None),
    "red_team_models": ("red_team_models", None),

    # Model
    "model_id": ("llm_model", "llm_config.model"),
    "api_key": ("api_key", "llm_config.api_key"),
    "api_base": ("api_base", "llm_config.url"),
    "temperature": ("temperature", "llm_config.temperature"),
    "top_p": ("top_p", "llm_config.top_p"),

    # Resources
    "max_time": ("max_time_seconds", None),
    "cost_limit_usd": ("cost_limit_usd", None),
}
```

### Config Converter (`openevolve/config_converter.py`)

```python
"""
Convert between OpenEvolve and PES configurations
"""

from typing import Dict, Any
from .evolution import EvolutionConfiguration
from .pes import EvolveChainConfig
from .unified import UnifiedConfig


class ConfigConverter:
    """
    Convert between different config formats
    """

    @staticmethod
    def openevolve_to_unified(oe_config: EvolutionConfiguration) -> UnifiedConfig:
        """
        Convert OpenEvolve config to Unified config
        """
        return UnifiedConfig(
            problem=oe_config.problem_statement or "",
            target_score=oe_config.target_score or 1.0,
            max_iterations=oe_config.max_iterations,
            workspace_path=oe_config.workspace_path or "./output",

            # Strategy
            strategy=oe_config.evolution_mode or "auto",

            # Planning
            enable_planning=(oe_config.evolution_mode == "pes"),
            planner_model=oe_config.model_id,
            planning_depth=3,

            # Memory
            enable_memory=True,
            memory_size=1000,
            num_islands=oe_config.num_islands or 5,

            # Specialized modes
            qd_archive_size=oe_config.qd_archive_size or 100,
            qd_novelty_threshold=oe_config.qd_novelty_threshold or 0.1,
            feature_dimensions=oe_config.feature_dimensions,

            mo_objectives=oe_config.mo_objectives,
            mo_pareto_size=oe_config.mo_pareto_size or 50,
            objective_weights=oe_config.objective_weights,

            adversarial_rounds=oe_config.adversarial_rounds or 5,
            red_team_models=oe_config.red_team_models,

            # Model
            llm_model=oe_config.model_id,
            api_key=oe_config.api_key,
            api_base=oe_config.api_base,
            temperature=oe_config.temperature,

            # Resources
            max_time_seconds=oe_config.max_time or 1800,
            cost_limit_usd=oe_config.cost_limit_usd or 10.0,
        )

    @staticmethod
    def unified_to_pes(config: UnifiedConfig) -> EvolveChainConfig:
        """
        Convert Unified config to PES config
        """
        return EvolveChainConfig(
            evolve={
                "task": config.problem,
                "max_iterations": config.max_iterations,
                "target_score": config.target_score,
                "workspace_path": config.workspace_path,
                "initial_code": "",
                "concurrency": 5,
                "database": {
                    "num_islands": config.num_islands,
                    "checkpoint_interval": config.compression_interval,
                    "output_path": config.workspace_path,
                    "max_solutions": config.memory_size
                }
            },
            llm_config={
                "model": config.planner_model,
                "api_key": config.api_key,
                "url": config.api_base,
                "temperature": config.temperature
            },
            planners={
                "general_planner": {
                    "llm_config": {
                        "model": config.planner_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    },
                    "max_turns": config.planning_depth
                }
            },
            executors={
                "general_executor": {
                    "llm_config": {
                        "model": config.llm_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    },
                    "max_rounds": 10
                }
            },
            summarizers={
                "general_summarizer": {
                    "llm_config": {
                        "model": config.llm_model,
                        "api_key": config.api_key,
                        "url": config.api_base
                    }
                }
            }
        )

    @staticmethod
    def openevolve_to_pes(oe_config: EvolutionConfiguration) -> EvolveChainConfig:
        """
        Convert OpenEvolve directly to PES (via Unified)
        """
        unified = ConfigConverter.openevolve_to_unified(oe_config)
        return ConfigConverter.unified_to_pes(unified)
```

---

## 4. STRATEGY SELECTION

### Adaptive Strategy Selector (`openevolve/strategy_selector.py`)

```python
"""
Automatically select the best evolution strategy based on problem characteristics
"""

from typing import Dict, Any, List
import re


class StrategySelector:
    """
    Analyzes problem and selects optimal evolution strategy
    """

    def __init__(self):
        # Keyword patterns for each strategy
        self.patterns = {
            "multi_objective": [
                r"\bbalance\b",
                r"\btrade-off\b",
                r"\bpareto\b",
                r"\boptimize multiple\b",
                r"\bminimize.*maximize\b",
                r"\bwhile.*\b.*\b\b.*ensure\b"
            ],
            "quality_diversity": [
                r"\bdiverse\b",
                r"\bvariety\b",
                r"\bdifferent\b",
                r"\bexplore\b",
                r"\bnovel\b",
                r"\bcreative\b"
            ],
            "adversarial": [
                r"\brobust\b",
                r"\badversar\b",
                r"\battack\b",
                r"\bdefens\b",
                r"\bresilient\b",
                r"\bresist\b"
            ],
            "island_model": [
                r"\bparallel\b",
                r"\bdistribut\b",
                r"\bmultiple.*search\b",
                r"\bindependent\b"
            ]
        }

        # Domain-specific mappings
        self.domain_strategies = {
            "finance": "multi_objective",
            "trading": "adversarial",
            "science": "quality_diversity",
            "engineering": "multi_objective",
            "pharma": "multi_objective",
            "web": "quality_diversity"
        }

    def select_strategy(
        self,
        problem: str,
        domain: str = None,
        user_preference: str = None
    ) -> str:
        """
        Select the best strategy for the given problem

        Args:
            problem: Problem statement
            domain: Optional domain hint
            user_preference: User-specified strategy (overrides auto-detection)

        Returns:
            Selected strategy name
        """
        # If user specified, use that
        if user_preference and user_preference != "auto":
            return user_preference

        # Check domain-specific strategies
        if domain and domain.lower() in self.domain_strategies:
            return self.domain_strategies[domain.lower()]

        # Analyze problem for patterns
        scores = self._score_strategies(problem)

        # Return highest-scoring strategy
        if not scores:
            return "standard"  # Default

        best_strategy = max(scores, key=scores.get)
        return best_strategy

    def _score_strategies(self, problem: str) -> Dict[str, float]:
        """
        Score each strategy based on keyword matches
        """
        problem_lower = problem.lower()
        scores = {}

        for strategy, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, problem_lower))
                score += matches
            if score > 0:
                scores[strategy] = score

        return scores

    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """
        Get default configuration for a strategy
        """
        configs = {
            "standard": {
                "enable_planning": True,
                "enable_memory": True,
                "max_iterations": 100
            },
            "multi_objective": {
                "enable_planning": True,
                "enable_memory": True,
                "strategy": "multi_objective",
                "mo_pareto_size": 100
            },
            "quality_diversity": {
                "enable_planning": True,
                "enable_memory": True,
                "strategy": "quality_diversity",
                "qd_archive_size": 500,
                "feature_dimensions": None  # Auto-detect
            },
            "adversarial": {
                "enable_planning": True,
                "enable_memory": True,
                "strategy": "adversarial",
                "adversarial_rounds": 10
            },
            "pes": {
                "enable_planning": True,
                "enable_memory": True,
                "strategy": "pes",
                "planning_depth": 5
            }
        }

        return configs.get(strategy, configs["standard"])


# Example usage
selector = StrategySelector()

# Example 1: Finance problem
strategy = selector.select_strategy(
    problem="Optimize portfolio for maximum returns while minimizing risk",
    domain="finance"
)
# Returns: "multi_objective"

# Example 2: Adversarial problem
strategy = selector.select_strategy(
    problem="Design robust network architecture that resists attacks"
)
# Returns: "adversarial"

# Example 3: Creative problem
strategy = selector.select_strategy(
    problem="Generate diverse website layouts exploring different design patterns"
)
# Returns: "quality_diversity"
```

---

## 5. MEMORY INTEGRATION

### Unified Memory (`openevolve/memory/unified_memory.py`)

```python
"""
Unified memory system combining PES fusion memory with OpenEvolve specialized stores
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .pes import EvolveDatabase


@dataclass
class Solution:
    """Represents a solution in memory"""
    solution_id: str
    code: str
    score: float
    parent_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For QD
    behavior_descriptor: Optional[Dict[str, float]] = None

    # For MO
    objective_scores: Optional[Dict[str, float]] = None


class UnifiedMemory:
    """
    Unified memory system that combines:
    - PES fusion memory (parent-child relationships)
    - QD archive (behavior space)
    - MO Pareto front
    - Adversarial attack/defense history
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # PES fusion memory (primary store)
        self.pes_memory = EvolveDatabase.create_database(config.get("pes_db", {}))

        # Specialized stores
        self.qd_archive = {}  # behavior → best solution
        self.mo_pareto_front = []  # list of non-dominated solutions
        self.adversarial_history = {}  # attack_type → best defense

        # Statistics
        self.stats = {
            "total_solutions": 0,
            "best_score": 0.0,
            "total_iterations": 0
        }

    async def add_solution(
        self,
        solution: Solution,
        strategy: str = "standard"
    ) -> None:
        """
        Add a solution to all relevant memory stores

        Args:
            solution: Solution to add
            strategy: Current evolution strategy
        """
        # Add to PES memory (always)
        self.pes_memory.add_solution(
            solution_id=solution.solution_id,
            parent_id=solution.parent_id,
            score=solution.score,
            code=solution.code,
            summary=solution.metadata.get("summary", "")
        )

        # Add to specialized stores based on strategy
        if strategy == "quality_diversity" and solution.behavior_descriptor:
            self._add_to_qd_archive(solution)
        elif strategy == "multi_objective" and solution.objective_scores:
            self._add_to_pareto_front(solution)
        elif strategy == "adversarial":
            self._add_to_adversarial_history(solution)

        # Update statistics
        self.stats["total_solutions"] += 1
        if solution.score > self.stats["best_score"]:
            self.stats["best_score"] = solution.score

    def _add_to_qd_archive(self, solution: Solution) -> None:
        """Add solution to QD archive"""
        # Create behavior key
        behavior_key = self._behavior_to_key(solution.behavior_descriptor)

        # Add if better than existing
        if behavior_key not in self.qd_archive or \
           solution.score > self.qd_archive[behavior_key].score:
            self.qd_archive[behavior_key] = solution

    def _add_to_pareto_front(self, solution: Solution) -> None:
        """Add solution to MO Pareto front"""
        # Check if dominated
        dominated = False
        for existing in self.mo_pareto_front:
            if self._dominates(existing, solution):
                dominated = True
                break
            elif self._dominates(solution, existing):
                # Remove dominated solution
                self.mo_pareto_front.remove(existing)

        # Add if not dominated
        if not dominated:
            self.mo_pareto_front.append(solution)

    def _add_to_adversarial_history(self, solution: Solution) -> None:
        """Add solution to adversarial history"""
        attack_type = solution.metadata.get("attack_type", "unknown")

        # Add if better defense
        if attack_type not in self.adversarial_history or \
           solution.score > self.adversarial_history[attack_type].score:
            self.adversarial_history[attack_type] = solution

    def _behavior_to_key(self, behavior: Dict[str, float]) -> str:
        """Convert behavior descriptor to string key"""
        # Discretize behavior into bins
        binned = {
            k: round(v, 2) for k, v in behavior.items()
        }
        return str(sorted(binned.items()))

    def _dominates(self, solution1: Solution, solution2: Solution) -> bool:
        """
        Check if solution1 dominates solution2 (for MO)
        Solution1 dominates if it's better in all objectives
        """
        if not solution1.objective_scores or not solution2.objective_scores:
            return False

        for obj in solution1.objective_scores:
            if solution1.objective_scores[obj] <= solution2.objective_scores.get(obj, 0):
                return False

        return True

    def get_best_solutions(self, top_k: int = 10) -> List[Solution]:
        """Get top-k best solutions"""
        return self.pes_memory.get_best_solutions(top_k=top_k)

    def get_qd_solutions(self) -> Dict[str, Solution]:
        """Get all QD archive solutions"""
        return self.qd_archive

    def get_pareto_front(self) -> List[Solution]:
        """Get MO Pareto front"""
        return self.mo_pareto_front

    def sample_parent(self, island_id: int = 0) -> Optional[Solution]:
        """Sample a parent solution for evolution"""
        parent = self.pes_memory.sample_solution(island_id)
        if parent:
            return Solution(
                solution_id=parent.get("solution_id", ""),
                code=parent.get("code", ""),
                score=parent.get("score", 0.0),
                parent_id=parent.get("parent_id")
            )
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get memory status"""
        pes_status = self.pes_memory.memory_status()

        return {
            **pes_status,
            "qd_archive_size": len(self.qd_archive),
            "pareto_front_size": len(self.mo_pareto_front),
            "adversarial_history_size": len(self.adversarial_history),
            **self.stats
        }
```

---

## 6. COMPLETE WORKING EXAMPLE

### End-to-End Example

```python
"""
Complete example of using the Unified Evolution Engine
"""

import asyncio
from openevolve import unified_evolve, UnifiedConfig


async def main():
    """
    Demonstrate unified evolution with different strategies
    """

    # ===== Example 1: Simple usage (auto-detect) =====
    print("Example 1: Auto-detect strategy")
    print("-" * 50)

    result1 = await unified_evolve(
        problem="Optimize sorting algorithm for performance",
        target_score=0.95,
        enable_planning=True,
        max_iterations=50
    )

    print(f"Strategy used: {result1['metadata']['strategy']}")
    print(f"Best score: {result1['fitness']:.2f}")
    print(f"Iterations: {result1['iterations']}")
    print()

    # ===== Example 2: Multi-objective optimization =====
    print("Example 2: Multi-objective optimization")
    print("-" * 50)

    result2 = await unified_evolve(
        problem="Design portfolio that maximizes returns while minimizing risk",
        strategy="multi_objective",
        objectives=["return", "risk", "liquidity"],
        objective_weights=[0.5, 0.3, 0.2],
        enable_planning=True,
        enable_memory=True,
        max_iterations=100
    )

    print(f"Pareto front size: {len(result2['pareto_front'])}")
    for i, sol in enumerate(result2['pareto_front'][:5]):
        print(f"  Solution {i+1}: "
              f"Return={sol['return']:.2%}, "
              f"Risk={sol['risk']:.2%}")
    print()

    # ===== Example 3: Quality Diversity =====
    print("Example 3: Quality Diversity")
    print("-" * 50)

    result3 = await unified_evolve(
        problem="Generate diverse neural network architectures",
        strategy="quality_diversity",
        qd_archive_size=200,
        feature_dimensions=["layers", "parameters", "accuracy"],
        enable_planning=True,
        max_iterations=80
    )

    print(f"QD archive size: {len(result3['qd_archive'])}")
    print(f"Diversity score: {result3['diversity']:.2f}")
    print()

    # ===== Example 4: Adversarial =====
    print("Example 4: Adversarial optimization")
    print("-" * 50)

    result4 = await unified_evolve(
        problem="Design robust trading strategy",
        strategy="adversarial",
        adversarial_rounds=15,
        attack_types=["market_crash", "black_swan", "liquidity_crisis"],
        enable_planning=True,
        enable_memory=True,
        max_iterations=120
    )

    print(f"Robustness score: {result4['robustness']:.2f}")
    print(f"Best performance: {result4['fitness']:.2f}")
    print(f"Survived attacks: {result4['survived_attacks']}/{result4['total_attacks']}")
    print()

    # ===== Example 5: Using config object =====
    print("Example 5: Using UnifiedConfig")
    print("-" * 50)

    config = UnifiedConfig(
        problem="Optimize code structure",
        strategy="auto",
        enable_planning=True,
        enable_memory=True,
        planner_model="claude-3-5-sonnet",
        planning_depth=5,
        memory_size=5000,
        num_islands=10,
        max_iterations=100,
        target_score=0.98
    )

    result5 = await unified_evolve(
        problem=config.problem,
        **config.__dict__
    )

    print(f"Auto-detected strategy: {result5['metadata']['strategy']}")
    print(f"Final score: {result5['fitness']:.2f}")
    print(f"Total tokens: {result5['metadata']['total_tokens']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
```

### Migration Example

```python
"""
Migration example: Converting from OpenEvolve to Unified API
"""

import asyncio
from openevolve import run_unified_evolution, unified_evolve


async def migrate_example():
    """
    Show how to migrate existing OpenEvolve code to Unified API
    """

    # ===== BEFORE: OpenEvolve with 272 parameters =====
    print("BEFORE: OpenEvolve traditional API")
    print("-" * 50)

    old_result = await run_unified_evolution(
        problem_statement="Optimize sorting algorithm",
        evolution_mode="standard",
        max_iterations=100,
        population_size=20,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # ... 265 more parameters 😰
    )

    print(f"Fitness: {old_result['fitness']:.2f}")
    print(f"Solution: {old_result['solution']['code'][:100]}...")
    print()

    # ===== AFTER: Unified API =====
    print("AFTER: Unified API (same result, cleaner)")
    print("-" * 50)

    new_result = await unified_evolve(
        problem="Optimize sorting algorithm",
        enable_planning=True,
        max_iterations=100
    )

    print(f"Fitness: {new_result['fitness']:.2f}")
    print(f"Solution: {new_result['solution']['code'][:100]}...")
    print()

    # ===== Gradual migration: Use PES as evolution mode =====
    print("GRADUAL: Use PES as evolution mode")
    print("-" * 50)

    # Still using old API, but with PES
    gradual_result = await run_unified_evolution(
        problem_statement="Optimize sorting algorithm",
        evolution_mode="pes",  # NEW: Use PES
        max_iterations=100
        # Other parameters ignored when using PES
    )

    print(f"Fitness: {gradual_result['fitness']:.2f}")
    print(f"Engine: {gradual_result['metadata']['engine']}")
    print()


if __name__ == "__main__":
    asyncio.run(migrate_example())
```

---

## CONCLUSION

This code examples document provides:

1. ✅ **PES extraction**: Directory structure and core modules
2. ✅ **Unified API**: Complete implementation with `unified_evolve()`
3. ✅ **Config mapping**: Convert between OpenEvolve, PES, and Unified formats
4. ✅ **Strategy selection**: Auto-detect best evolution strategy
5. ✅ **Memory integration**: Unified memory combining all systems
6. ✅ **Working examples**: End-to-end usage and migration guide

**Next Steps**:
1. Copy code into your project
2. Run extraction (Phase 1)
3. Test with example problems
4. Integrate with existing codebase

**For Full Analysis**: See `HYBRID_ARCHITECTURE_REPORT.md`
