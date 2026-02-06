"""
OpenEvolve Configuration Stub Module

This module provides stub configuration classes for OpenEvolve.
When the actual OpenEvolve package is available, it will be used instead.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Flag to indicate if this is a stub implementation
IS_STUB = True


@dataclass
class LLMModelConfig:
    """Configuration for an LLM model."""
    name: str = "default"
    weight: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __init__(self, name: str = "default", weight: float = 1.0, **kwargs):
        self.name = name
        self.weight = weight
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class DatabaseConfig:
    """Configuration for the evolution database."""
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    num_islands: int = 5
    migration_rate: float = 0.1
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.7
    exploitation_ratio: float = 0.3


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.0
    cascade_evaluation: bool = False
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])


@dataclass
class PromptConfig:
    """Configuration for prompts."""
    system_message: str = ""
    evaluator_system_message: str = ""
    use_template_stochasticity: bool = False
    num_top_programs: int = 5
    num_diverse_programs: int = 5


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution tracing."""
    enabled: bool = True
    trace_dir: Optional[str] = None
    save_interval: int = 10


@dataclass
class Config:
    """Main configuration class for OpenEvolve."""
    llm_models: List[LLMModelConfig] = field(default_factory=list)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)
    
    def __init__(self, **kwargs):
        self.llm_models = kwargs.get('llm_models', [])
        self.database = kwargs.get('database', DatabaseConfig())
        self.evaluator = kwargs.get('evaluator', EvaluatorConfig())
        self.prompt = kwargs.get('prompt', PromptConfig())
        self.evolution_trace = kwargs.get('evolution_trace', EvolutionTraceConfig())
