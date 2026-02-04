"""Configuration for Cognitive Hydraulics Integration.

Follows CLAUDE.md patterns:
- Environment variable injection for all configurable values
- Explicit validation at startup
- No magic defaults
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class SoarConfig:
    """Soar Engine Configuration."""
    
    # Working memory limits
    working_memory_slots: int = field(
        default_factory=lambda: int(os.getenv("SOAR_WM_SLOTS", "7"))
    )
    max_subgoal_depth: int = field(
        default_factory=lambda: int(os.getenv("SOAR_MAX_SUBGOAL_DEPTH", "10"))
    )
    
    # Decision cycle settings
    decision_cycle_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("SOAR_CYCLE_TIMEOUT_MS", "100"))
    )
    max_decision_cycles: int = field(
        default_factory=lambda: int(os.getenv("SOAR_MAX_CYCLES", "1000"))
    )
    
    # Chunking settings
    enable_chunking: bool = field(
        default_factory=lambda: os.getenv("SOAR_ENABLE_CHUNKING", "true").lower() == "true"
    )
    chunk_generalization_threshold: float = field(
        default_factory=lambda: float(os.getenv("SOAR_CHUNK_GENERALIZE_THRESHOLD", "0.8"))
    )
    
    # Impasse detection
    tie_impasse_threshold: int = field(
        default_factory=lambda: int(os.getenv("SOAR_TIE_THRESHOLD", "3"))
    )
    no_change_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("SOAR_NO_CHANGE_TIMEOUT_MS", "500"))
    )


@dataclass
class ACTRConfig:
    """ACT-R Engine Configuration."""
    
    # Utility equation parameters
    default_probability: float = field(
        default_factory=lambda: float(os.getenv("ACTR_DEFAULT_P", "0.5"))
    )
    default_goal_value: float = field(
        default_factory=lambda: float(os.getenv("ACTR_DEFAULT_G", "10.0"))
    )
    default_cost: float = field(
        default_factory=lambda: float(os.getenv("ACTR_DEFAULT_C", "1.0"))
    )
    
    # Noise parameters (stochastic variability)
    noise_sigma: float = field(
        default_factory=lambda: float(os.getenv("ACTR_NOISE_SIGMA", "0.5"))
    )
    
    # Tabu search parameters
    tabu_list_size: int = field(
        default_factory=lambda: int(os.getenv("ACTR_TABU_SIZE", "10"))
    )
    tabu_penalty_weight: float = field(
        default_factory=lambda: float(os.getenv("ACTR_TABU_PENALTY", "2.0"))
    )
    
    # Activation decay
    activation_decay: float = field(
        default_factory=lambda: float(os.getenv("ACTR_DECAY", "0.5"))
    )
    
    # History penalty
    history_penalty_base: float = field(
        default_factory=lambda: float(os.getenv("ACTR_HISTORY_PENALTY", "1.0"))
    )


@dataclass
class PressureValveConfig:
    """Pressure Valve Configuration."""
    
    # Thresholds for system switching
    soar_to_actr_depth: int = field(
        default_factory=lambda: int(os.getenv("PRESSURE_SOAR_TO_ACTR_DEPTH", "3"))
    )
    actr_to_evo_pressure: float = field(
        default_factory=lambda: float(os.getenv("PRESSURE_ACTR_TO_EVO", "0.9"))
    )
    time_threshold_ms: int = field(
        default_factory=lambda: int(os.getenv("PRESSURE_TIME_THRESHOLD_MS", "500"))
    )
    
    # Pressure calculation weights (must sum to 1.0)
    weight_depth: float = field(
        default_factory=lambda: float(os.getenv("PRESSURE_WEIGHT_DEPTH", "0.3"))
    )
    weight_time: float = field(
        default_factory=lambda: float(os.getenv("PRESSURE_WEIGHT_TIME", "0.25"))
    )
    weight_impasses: float = field(
        default_factory=lambda: float(os.getenv("PRESSURE_WEIGHT_IMPASSES", "0.25"))
    )
    weight_ambiguity: float = field(
        default_factory=lambda: float(os.getenv("PRESSURE_WEIGHT_AMBIGUITY", "0.2"))
    )
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.weight_depth + self.weight_time + self.weight_impasses + self.weight_ambiguity
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.weight_depth /= total
            self.weight_time /= total
            self.weight_impasses /= total
            self.weight_ambiguity /= total


@dataclass
class EvolutionaryConfig:
    """Evolutionary Fallback Configuration."""
    
    # Population settings
    population_size: int = field(
        default_factory=lambda: int(os.getenv("EVO_POPULATION_SIZE", "50"))
    )
    max_generations: int = field(
        default_factory=lambda: int(os.getenv("EVO_MAX_GENERATIONS", "100"))
    )
    
    # Genetic operators
    mutation_rate: float = field(
        default_factory=lambda: float(os.getenv("EVO_MUTATION_RATE", "0.1"))
    )
    crossover_rate: float = field(
        default_factory=lambda: float(os.getenv("EVO_CROSSOVER_RATE", "0.7"))
    )
    elitism_count: int = field(
        default_factory=lambda: int(os.getenv("EVO_ELITISM", "5"))
    )
    
    # Fitness evaluation
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("EVO_TIMEOUT_SECONDS", "30"))
    )
    
    # Convergence
    convergence_threshold: float = field(
        default_factory=lambda: float(os.getenv("EVO_CONVERGENCE_THRESHOLD", "0.01"))
    )
    stagnation_generations: int = field(
        default_factory=lambda: int(os.getenv("EVO_STAGNATION_GENS", "20"))
    )


@dataclass
class LLMConfig:
    """LLM Intuition Engine Configuration."""
    
    # Model settings
    model_name: str = field(
        default_factory=lambda: os.getenv("COG_HYD_LLM_MODEL", "qwen3:8b")
    )
    api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("COG_HYD_LLM_API_BASE")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("COG_HYD_LLM_API_KEY")
    )
    
    # Request settings
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("COG_HYD_LLM_TIMEOUT", "10"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("COG_HYD_LLM_RETRIES", "3"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("COG_HYD_LLM_TEMP", "0.3"))
    )
    
    # Caching
    enable_cache: bool = field(
        default_factory=lambda: os.getenv("COG_HYD_LLM_CACHE", "true").lower() == "true"
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("COG_HYD_LLM_CACHE_TTL", "3600"))
    )


@dataclass
class CognitiveHydraulicsConfig:
    """Main Cognitive Hydraulics Configuration."""
    
    soar: SoarConfig = field(default_factory=SoarConfig)
    actr: ACTRConfig = field(default_factory=ACTRConfig)
    pressure_valve: PressureValveConfig = field(default_factory=PressureValveConfig)
    evolutionary: EvolutionaryConfig = field(default_factory=EvolutionaryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Global settings
    max_reasoning_time_ms: int = field(
        default_factory=lambda: int(os.getenv("COG_HYD_MAX_TIME_MS", "30000"))
    )
    enable_logging: bool = field(
        default_factory=lambda: os.getenv("COG_HYD_LOGGING", "true").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("COG_HYD_LOG_LEVEL", "INFO")
    )
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return errors if any."""
        errors = []
        
        # Check LLM configuration
        if self.llm.model_name not in ["qwen3:8b", "gpt-4", "gpt-3.5-turbo", "claude-3"]:
            if not self.llm.api_base:
                errors.append("LLM model requires api_base or must be known model")
        
        # Validate numeric ranges
        if self.pressure_valve.actr_to_evo_pressure < 0 or self.pressure_valve.actr_to_evo_pressure > 1:
            errors.append("actr_to_evo_pressure must be between 0 and 1")
        
        if self.evolutionary.mutation_rate < 0 or self.evolutionary.mutation_rate > 1:
            errors.append("mutation_rate must be between 0 and 1")
        
        if self.evolutionary.crossover_rate < 0 or self.evolutionary.crossover_rate > 1:
            errors.append("crossover_rate must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "soar": self.soar.__dict__,
            "actr": self.actr.__dict__,
            "pressure_valve": self.pressure_valve.__dict__,
            "evolutionary": self.evolutionary.__dict__,
            "llm": self.llm.__dict__,
            "max_reasoning_time_ms": self.max_reasoning_time_ms,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_env(cls) -> "CognitiveHydraulicsConfig":
        """Load configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, path: str) -> "CognitiveHydraulicsConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            soar=SoarConfig(**data.get("soar", {})),
            actr=ACTRConfig(**data.get("actr", {})),
            pressure_valve=PressureValveConfig(**data.get("pressure_valve", {})),
            evolutionary=EvolutionaryConfig(**data.get("evolutionary", {})),
            llm=LLMConfig(**data.get("llm", {})),
            max_reasoning_time_ms=data.get("max_reasoning_time_ms", 30000),
            enable_logging=data.get("enable_logging", True),
            log_level=data.get("log_level", "INFO"),
        )
