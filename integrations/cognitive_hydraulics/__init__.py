"""
Cognitive Hydraulics Integration for OpenEvolve Knowledge Engine.

A hybrid neuro-symbolic reasoning engine combining:
- Soar (System 2): Slow, deliberate, symbolic reasoning
- ACT-R (System 1): Fast, heuristic, utility-based reasoning
- LLM Intuition: Probability and cost estimation
- Evolutionary Fallback: Genetic algorithm when pressure is high
- Chunking: Learn from successful resolutions

Usage:
    from integrations.cognitive_hydraulics import CognitiveHydraulicsEngine
    
    engine = CognitiveHydraulicsEngine()
    result = engine.solve(problem_description="Find the shortest path",
                         goal={"type": "pathfinding", "start": "A", "end": "B"})
"""

# Version
__version__ = "1.0.0"

# Core engine exports
from .cognitive_hydraulics import (
    CognitiveHydraulicsEngine,
    ReasoningSession,
    SystemOrchestrator,
    ReasoningResult,
    SystemType,
)

# Soar (System 2) exports
from .soar_engine import (
    SoarEngine,
    SoarWorkingMemory,
    SoarProductionSystem,
    SoarDecisionCycle,
    ImpasseDetector,
    SubgoalManager,
    ChunkingSystem,
    SoarState,
    SoarOperator,
    SoarRule,
    Impasse,
    ImpasseType,
    TieImpasse,
    NoChangeImpasse,
    ConflictImpasse,
    ConstraintFailureImpasse,
)

# ACT-R (System 1) exports
from .actr_engine import (
    ACTREngine,
    ACTRDeclarativeMemory,
    ACTRProceduralMemory,
    UtilityCalculator,
    TabuSearch,
    NoiseGenerator,
    ACTRChunk,
    ACTRProduction,
    UtilityEquation,
    TabuList,
)

# Pressure Valve exports
from .pressure_valve import (
    PressureValve,
    PressureMonitor,
    SystemSwitcher,
    ThresholdConfig,
    PressureMetrics,
)

# LLM Intuition exports
from .llm_intuition import (
    LLMIntuitionEngine,
    IntuitionEngine,
    ProbabilityEstimator,
    CostEstimator,
    OperatorGenerator,
    ChunkEncoder,
    SuccessRating,
)

# Evolutionary Fallback exports
from .evolutionary_fallback import (
    EvolutionarySolver,
    Population,
    Individual,
    FitnessEvaluator,
    GeneticOperators,
    SolutionType,
)

# Chunking System exports
from .chunking_system import (
    ChunkingEngine,
    Chunk,
    ChunkRepository,
    Generalizer,
    ChunkType,
    ChunkQuality,
)

# Configuration exports
from .config import (
    CognitiveHydraulicsConfig,
    SoarConfig,
    ACTRConfig,
    PressureValveConfig,
    EvolutionaryConfig,
)

__all__ = [
    # Main engine
    "CognitiveHydraulicsEngine",
    "ReasoningSession",
    "SystemOrchestrator",
    "ReasoningResult",
    "SystemType",
    
    # Soar
    "SoarEngine",
    "SoarWorkingMemory",
    "SoarProductionSystem",
    "SoarDecisionCycle",
    "ImpasseDetector",
    "SubgoalManager",
    "ChunkingSystem",
    "SoarState",
    "SoarOperator",
    "SoarRule",
    "Impasse",
    "ImpasseType",
    "TieImpasse",
    "NoChangeImpasse",
    "ConflictImpasse",
    "ConstraintFailureImpasse",
    
    # ACT-R
    "ACTREngine",
    "ACTRDeclarativeMemory",
    "ACTRProceduralMemory",
    "UtilityCalculator",
    "TabuSearch",
    "NoiseGenerator",
    "ACTRChunk",
    "ACTRProduction",
    "UtilityEquation",
    "TabuList",
    
    # Pressure Valve
    "PressureValve",
    "PressureMonitor",
    "SystemSwitcher",
    "ThresholdConfig",
    "PressureMetrics",
    
    # LLM Intuition
    "LLMIntuitionEngine",
    "IntuitionEngine",
    "ProbabilityEstimator",
    "CostEstimator",
    "OperatorGenerator",
    "ChunkEncoder",
    "SuccessRating",
    
    # Evolutionary Fallback
    "EvolutionarySolver",
    "Population",
    "Individual",
    "FitnessEvaluator",
    "GeneticOperators",
    "SolutionType",
    
    # Chunking System
    "ChunkingEngine",
    "Chunk",
    "ChunkRepository",
    "Generalizer",
    "ChunkType",
    "ChunkQuality",
    
    # Configuration
    "CognitiveHydraulicsConfig",
    "SoarConfig",
    "ACTRConfig",
    "PressureValveConfig",
    "EvolutionaryConfig",
]
