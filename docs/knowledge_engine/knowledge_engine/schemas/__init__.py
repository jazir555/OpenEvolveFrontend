"""
Knowledge Engine Schemas

Exports all schema modules for the knowledge engine.
"""

from .evolutionary_artifacts import *
from .comparison_results import *
from .long_horizon import *

__all__ = [
    # Evolutionary Artifacts
    'ArtifactType',
    'SystemType',
    'DomainType',
    'SolutionPatternArtifact',
    'EvolutionaryTrajectoryArtifact',
    'MAPElitesArchiveArtifact',
    'PESPatternsArtifact',
    'ParameterEffectivenessArtifact',
    'PerformanceMetricsArtifact',
    'EvolutionaryTreeArtifact',
    'create_artifact_from_dict',

    # Comparison Results
    'ComparisonCategory',
    'WinnerType',
    'SynergyType',
    'ComplexityLevel',
    'CategoryComparison',
    'DetailedPerformanceComparison',
    'SynergyOpportunityDetailed',
    'BestPracticeDetailed',
    'HybridRecommendationDetailed',
    'DualRunAnalysisReport',

    # Long-Horizon Learning
    'OutcomeType',
    'AdaptationActionType',
    'ExperimentStatus',
    'ExplorationStrategy',
    'LearningOutcome',
    'StrategyPerformance',
    'AdaptationAction',
    'VariantStats',
    'Experiment',
    'ExperimentResults',
    'CausalRelationship',
    'CausalModel',
    'EffectPrediction',
    'Explanation',
    'MetaPattern',
    'StrategyRecommendation'
]
