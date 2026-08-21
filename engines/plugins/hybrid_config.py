"""
Configuration Management for Hybrid MAKER System

This module provides validated configuration management:
- Schema validation
- Environment variable integration
- Configuration profiles
- Type safety

Author: OpenEvolve Hybrid Config Team
Created: 2025-01-07
Version: 1.0.0
"""
from __future__ import annotations


import os
import json
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATED CONFIGURATION
# =============================================================================

@dataclass
class ValidatedHybridConfig:
    """
    Validated configuration for hybrid MAKER system

    Type-safe configuration with validation
    """

    # Voting parameters
    enable_voting: bool = True
    voting_threshold: int = 3

    # Decomposition parameters
    enable_decomposition: bool = True
    decomposition_depth: int = 3
    max_subtasks: int = 10

    # Search parameters
    mcts_simulations: int = 100
    evolution_generations: int = 20
    population_size: int = 20

    # Adversarial parameters
    adversarial_rounds: int = 3
    red_team_size: int = 2
    blue_team_size: int = 2

    # Adaptive parameters
    adaptive_switching: bool = True
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95

    # Performance parameters
    enable_caching: bool = True
    cache_size: int = 1000
    max_workers: int = 4

    # Logging
    log_level: str = "INFO"

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors"""
        errors = []

        # Validate voting threshold
        if self.voting_threshold < 1 or self.voting_threshold > 10:
            errors.append("voting_threshold must be between 1 and 10")

        # Validate population size
        if self.population_size < 2:
            errors.append("population_size must be at least 2")

        # Validate thresholds
        if not 0.0 <= self.diversity_threshold <= 1.0:
            errors.append("diversity_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.convergence_threshold <= 1.0:
            errors.append("convergence_threshold must be between 0.0 and 1.0")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enable_voting": self.enable_voting,
            "voting_threshold": self.voting_threshold,
            "enable_decomposition": self.enable_decomposition,
            "decomposition_depth": self.decomposition_depth,
            "max_subtasks": self.max_subtasks,
            "mcts_simulations": self.mcts_simulations,
            "evolution_generations": self.evolution_generations,
            "population_size": self.population_size,
            "adversarial_rounds": self.adversarial_rounds,
            "red_team_size": self.red_team_size,
            "blue_team_size": self.blue_team_size,
            "adaptive_switching": self.adaptive_switching,
            "diversity_threshold": self.diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "max_workers": self.max_workers,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ValidatedHybridConfig":
        """Create from dictionary"""
        field_names = {f.name for f in fields(cls)}
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in field_names
        })

    @classmethod
    def from_env(cls) -> "ValidatedHybridConfig":
        """Load from environment variables"""
        return cls(
            enable_voting=os.getenv("HYBRID_ENABLE_VOTING", "true").lower() == "true",
            voting_threshold=int(os.getenv("HYBRID_VOTING_THRESHOLD", "3")),
            enable_decomposition=os.getenv("HYBRID_ENABLE_DECOMPOSITION", "true").lower() == "true",
            mcts_simulations=int(os.getenv("HYBRID_MCTS_SIMULATIONS", "100")),
            evolution_generations=int(os.getenv("HYBRID_EVOLUTION_GENERATIONS", "20")),
            population_size=int(os.getenv("HYBRID_POPULATION_SIZE", "20")),
            enable_caching=os.getenv("HYBRID_ENABLE_CACHING", "true").lower() == "true",
            cache_size=int(os.getenv("HYBRID_CACHE_SIZE", "1000")),
            max_workers=int(os.getenv("HYBRID_MAX_WORKERS", "4")),
            log_level=os.getenv("HYBRID_LOG_LEVEL", "INFO"),
        )

    def save(self, filepath: str):
        """Save to JSON file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ValidatedHybridConfig":
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# =============================================================================
# CONFIGURATION PROFILES
# =============================================================================

class HybridConfigProfiles:
    """Predefined configuration profiles"""

    @staticmethod
    def fast() -> ValidatedHybridConfig:
        """Fast configuration for quick prototyping"""
        return ValidatedHybridConfig(
            mcts_simulations=10,
            evolution_generations=5,
            population_size=10,
            adversarial_rounds=1,
            enable_caching=True,
            max_workers=2
        )

    @staticmethod
    def balanced() -> ValidatedHybridConfig:
        """Balanced configuration for general use"""
        return ValidatedHybridConfig(
            mcts_simulations=50,
            evolution_generations=15,
            population_size=20,
            adversarial_rounds=2,
            enable_caching=True,
            max_workers=4
        )

    @staticmethod
    def thorough() -> ValidatedHybridConfig:
        """Thorough configuration for production"""
        return ValidatedHybridConfig(
            mcts_simulations=200,
            evolution_generations=30,
            population_size=30,
            adversarial_rounds=5,
            enable_caching=True,
            max_workers=8
        )


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class HybridConfigManager:
    """Manage hybrid configuration with profiles"""

    def __init__(self, config_dir: str = "./config/hybrid"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, ValidatedHybridConfig] = {}

    def register_config(self, name: str, config: ValidatedHybridConfig):
        """Register a configuration"""
        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        self.configs[name] = config
        logger.info(f"Registered configuration: {name}")

    def get_config(self, name: str) -> Optional[ValidatedHybridConfig]:
        """Get registered configuration"""
        return self.configs.get(name)

    def save_config(self, name: str, config: ValidatedHybridConfig):
        """Save configuration to file"""
        filepath = self.config_dir / f"{name}.json"
        config.save(str(filepath))
        self.register_config(name, config)

    def load_config(self, name: str) -> ValidatedHybridConfig:
        """Load configuration from file"""
        filepath = self.config_dir / f"{name}.json"
        config = ValidatedHybridConfig.load(str(filepath))

        errors = config.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")

        self.register_config(name, config)
        return config


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("Hybrid MAKER Configuration Management")
    print("=" * 60)

    # Create configurations
    print("\n1. Configuration Profiles")
    print("-" * 40)

    fast_config = HybridConfigProfiles.fast()
    print(f"Fast config: mcts={fast_config.mcts_simulations}, pop={fast_config.population_size}")

    balanced_config = HybridConfigProfiles.balanced()
    print(f"Balanced config: mcts={balanced_config.mcts_simulations}, pop={balanced_config.population_size}")

    thorough_config = HybridConfigProfiles.thorough()
    print(f"Thorough config: mcts={thorough_config.mcts_simulations}, pop={thorough_config.population_size}")

    # Validation
    print("\n2. Configuration Validation")
    print("-" * 40)

    valid_config = ValidatedHybridConfig()
    errors = valid_config.validate()
    print(f"Valid config errors: {errors}")

    invalid_config = ValidatedHybridConfig(voting_threshold=-1)
    errors = invalid_config.validate()
    print(f"Invalid config errors: {errors}")

    # Manager
    print("\n3. Configuration Manager")
    print("-" * 40)

    manager = HybridConfigManager()
    manager.register_config("fast", fast_config)
    manager.register_config("balanced", balanced_config)

    retrieved = manager.get_config("fast")
    print(f"Retrieved config: {retrieved.mcts_simulations} simulations")

    # Save/load
    print("\n4. Save/Load Configuration")
    print("-" * 40)

    manager.save_config("production", thorough_config)
    loaded = manager.load_config("production")
    print(f"Loaded config: {loaded.evolution_generations} generations")

    # Environment variables
    print("\n5. Environment Variables")
    print("-" * 40)

    os.environ["HYBRID_MCTS_SIMULATIONS"] = "75"
    env_config = ValidatedHybridConfig.from_env()
    print(f"From env: {env_config.mcts_simulations} simulations")

    print("\n" + "=" * 60)
    print("Configuration management demo complete!")
