"""ICR - Iterative Contextual Refinements.

A system for iterative refinement of outputs through multiple rounds
of self-improvement: Generate → Critique → Refine → Judge → Iterate.

Example:
    >>> from integrations.icr import ICREngine, Generator, Critic, Refiner, Judge
    >>> engine = ICREngine()
    >>> result = engine.refine("Write a function to sort a list", max_iterations=3)
    >>> print(result.final_output)
    >>> print(f"Achieved score: {result.final_score}")
"""

from integrations.icr.generator import Generator, GenerationResult
from integrations.icr.critic import (
    Critic,
    CritiqueResult,
    Issue,
    Suggestion,
)
from integrations.icr.refiner import (
    Refiner,
    RefinementStrategy,
    RefinementTracker,
    Change,
    RefinedOutput,
    ModifiedOutput,
    CombinedOutput,
)
from integrations.icr.judge import (
    Judge,
    EvaluationResult,
    ComparisonResult,
    Criteria,
)
from integrations.icr.iterative_engine import (
    ICREngine,
    RefinementResult,
    IterationResult,
)

__version__ = "1.0.0"

__all__ = [
    # Generator
    "Generator",
    "GenerationResult",
    # Critic
    "Critic",
    "CritiqueResult",
    "Issue",
    "Suggestion",
    # Refiner
    "Refiner",
    "RefinementStrategy",
    "RefinementTracker",
    "Change",
    "RefinedOutput",
    "ModifiedOutput",
    "CombinedOutput",
    # Judge
    "Judge",
    "EvaluationResult",
    "ComparisonResult",
    "Criteria",
    # Engine
    "ICREngine",
    "RefinementResult",
    "IterationResult",
]
