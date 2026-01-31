"""
Knowledge Engine for Evolutionary Systems

Provides knowledge extraction, analysis, and fusion for OpenEvolve and LoongFlow,
extended with long-horizon learning capabilities.

Modules:
- integrations.unified_evolution_integration: Unified knowledge extraction
- online_learning: Continuous learning from streaming outcomes
- ab_testing: Statistical A/B testing framework
- causal_modeling: Causal model building and inference
- meta_learning: Cross-workflow pattern learning
- schemas: Data structures and canonical models
"""

__version__ = "1.0.0"

from .integrations import UnifiedEvolutionKnowledgeExtractor
from .online_learning import OnlineLearner
from .ab_testing import ABTestFramework
from .causal_modeling import CausalModelBuilder
from .meta_learning import MetaLearner, FeatureExtractor

__all__ = [
    # Main integration
    'UnifiedEvolutionKnowledgeExtractor',

    # Long-horizon learning
    'OnlineLearner',
    'ABTestFramework',
    'CausalModelBuilder',
    'MetaLearner',
    'FeatureExtractor',
]
