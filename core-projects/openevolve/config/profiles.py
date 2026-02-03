"""
Configuration Profiles

Pre-defined configuration profiles for common use cases:
- development: Fast iteration during development
- testing: Optimized for running tests
- production: Maximum quality for production
- benchmarking: Optimized for performance benchmarking
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProfileInfo:
    """Information about a configuration profile"""
    name: str
    description: str
    category: str  # 'development', 'testing', 'production', 'benchmarking'
    parameters: Dict[str, Any]


class BaseProfile:
    """
    Base class for configuration profiles.

    Profiles provide pre-configured parameter sets for common use cases.
    """

    def __init__(self):
        """Initialize profile with default parameters"""
        self.parameters = self.get_parameters()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get profile parameters.

        Returns:
            Dictionary of configuration parameters
        """
        raise NotImplementedError("Subclasses must implement get_parameters()")

    def to_dict(self) -> Dict[str, Any]:
        """Export profile as dictionary"""
        return self.parameters.copy()

    def save(self, filepath: str) -> None:
        """
        Save profile to file.

        Args:
            filepath: Path to save profile
        """
        import json
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.parameters, f, indent=2)

        logger.info(f"Saved profile '{self.__class__.__name__}' to {filepath}")


class DevelopmentProfile(BaseProfile):
    """
    Fast iteration during development.

    Characteristics:
    - Low iteration counts for quick feedback
    - Verbose logging
    - Gauntlet disabled for speed
    - Debug mode enabled
    - Intermediate results saved
    """

    def get_parameters(self) -> Dict[str, Any]:
        """Get development profile parameters"""
        return {
            # Core Evolution - Fast feedback
            'max_iterations': 20,
            'population_size': 10,
            'generations': 5,
            'evolution_mode': 'standard',

            # LLM - Faster models
            'model_id': 'gpt-4o-mini',  # Faster, cheaper
            'temperature': 0.7,
            'max_tokens': 2048,

            # PES - Enabled but minimal
            'enable_planning': True,
            'enable_memory': False,  # Skip for speed
            'plan_temperature': 0.5,
            'planner_model': 'gpt-4o-mini',  # Required when enable_planning=True

            # QD - Disabled for simplicity
            'qd_enabled': False,

            # Gauntlet - Disabled for speed
            'enable_gauntlet': False,

            # Logging - Verbose
            'log_level': 'DEBUG',
            'verbose': True,
            'debug': True,

            # Output - Save intermediate
            'save_intermediate_results': True,
            'save_final_results': True,
            'output_dir': './dev_output',

            # Performance - Single worker
            'parallel_workers': 1,

            # Validation - Lenient
            'validate_outputs': False,
            'validation_strictness': 'lenient',

            # Early stopping - Enabled
            'early_stopping': True,
            'early_stopping_patience': 3,

            # Tags
            'tags': ['development', 'testing']
        }


class TestingProfile(BaseProfile):
    """
    Optimized for running tests.

    Characteristics:
    - Minimal iteration counts
    - Deterministic (seed set)
    - Fast models
    - No expensive features
    - Minimal logging
    """

    def get_parameters(self) -> Dict[str, Any]:
        """Get testing profile parameters"""
        return {
            # Core Evolution - Minimal
            'max_iterations': 5,
            'population_size': 5,
            'generations': 2,
            'seed': 42,  # Deterministic
            'evolution_mode': 'standard',

            # LLM - Fast
            'model_id': 'gpt-4o-mini',
            'temperature': 0.0,  # Deterministic
            'max_tokens': 1024,

            # PES - Disabled
            'enable_planning': False,
            'enable_memory': False,

            # QD - Disabled
            'qd_enabled': False,

            # Gauntlet - Disabled
            'enable_gauntlet': False,

            # Logging - Minimal
            'log_level': 'WARNING',
            'verbose': False,
            'debug': False,

            # Output - Minimal
            'save_intermediate_results': False,
            'save_final_results': False,

            # Performance - Single worker
            'parallel_workers': 1,

            # Validation - Disabled for speed
            'validate_outputs': False,

            # Tags
            'tags': ['testing', 'ci']
        }


class ProductionProfile(BaseProfile):
    """
    Maximum quality for production runs.

    Characteristics:
    - High iteration counts
    - Full features enabled
    - Best models
    - Strict validation
    - Comprehensive logging
    - Fault tolerance
    """

    def get_parameters(self) -> Dict[str, Any]:
        """Get production profile parameters"""
        return {
            # Core Evolution - Maximum quality
            'max_iterations': 100,
            'population_size': 50,
            'generations': 50,
            'evolution_mode': 'standard',

            # LLM - Best models
            'model_id': 'gpt-4o',
            'temperature': 0.7,
            'max_tokens': 4096,
            'top_p': 0.9,

            # PES - Full features
            'enable_planning': True,
            'enable_memory': True,
            'memory_type': 'episodic',
            'plan_temperature': 0.7,
            'planner_model': 'gpt-4o',  # Required when enable_planning=True
            'memory_capacity': 1000,
            'memory_retention': 0.9,

            # QD - Enabled for diversity
            'qd_enabled': True,
            'qd_algorithm': 'map_elites',
            'qd_grid_resolution': 20,
            'qd_archive_size': 1000,

            # Gauntlet - Enabled
            'enable_gauntlet': True,
            'gauntlet_rounds': 10,
            'gauntlet_strictness': 0.8,

            # Logging - Comprehensive
            'log_level': 'INFO',
            'log_format': 'json',
            'verbose': False,
            'debug': False,

            # Output - Save everything
            'save_intermediate_results': True,
            'save_final_results': True,
            'save_frequency': 10,
            'output_dir': './production_output',
            'result_format': 'json',

            # Performance - Parallel
            'parallel_workers': 4,
            'batch_size': 10,
            'cache_enabled': True,
            'cache_size': 10000,
            'timeout': 300,
            'max_retries': 3,
            'retry_delay': 1.0,

            # Validation - Strict
            'validate_outputs': True,
            'validation_frequency': 5,
            'validation_strictness': 'strict',

            # Stopping - Don't stop early
            'early_stopping': False,

            # Diversity - High
            'diversity_weight': 0.5,
            'novelty_weight': 0.3,

            # Integrations - Enabled
            'enable_mlflow': True,
            'enable_wandb': False,

            # Tags
            'tags': ['production']
        }


class BenchmarkingProfile(BaseProfile):
    """
    Optimized for performance benchmarking.

    Characteristics:
    - High iteration counts
    - Parallel execution
    - Minimal overhead
    - Performance metrics
    - Reproducible (seed)
    """

    def get_parameters(self) -> Dict[str, Any]:
        """Get benchmarking profile parameters"""
        return {
            # Core Evolution - High for accurate benchmarking
            'max_iterations': 200,
            'population_size': 100,
            'generations': 100,
            'seed': 42,  # Reproducible
            'evolution_mode': 'standard',

            # LLM - Consistent
            'model_id': 'gpt-4o',
            'temperature': 0.7,
            'max_tokens': 2048,

            # PES - Enabled
            'enable_planning': True,
            'enable_memory': True,
            'planner_model': 'gpt-4o',  # Required when enable_planning=True

            # QD - Enabled
            'qd_enabled': True,

            # Gauntlet - Disabled (adds variability)
            'enable_gauntlet': False,

            # Logging - Minimal overhead
            'log_level': 'INFO',
            'log_format': 'json',
            'verbose': False,

            # Output - Minimal
            'save_intermediate_results': False,
            'save_final_results': True,

            # Performance - Max parallel
            'parallel_workers': 8,
            'batch_size': 20,
            'cache_enabled': True,
            'cache_size': 50000,

            # Validation - Disabled for speed
            'validate_outputs': False,

            # Early stopping - Disabled for consistency
            'early_stopping': False,

            # Tags
            'tags': ['benchmarking', 'performance']
        }


class QuickStartProfile(BaseProfile):
    """
    Quick start for new users.

    Characteristics:
    - Conservative defaults
    - Moderate iteration counts
    - Good model
    - Essential features only
    - Clear logging
    """

    def get_parameters(self) -> Dict[str, Any]:
        """Get quick start profile parameters"""
        return {
            # Core Evolution - Balanced
            'max_iterations': 30,
            'population_size': 20,
            'generations': 10,
            'evolution_mode': 'standard',

            # LLM - Good balance
            'model_id': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_tokens': 2048,

            # PES - Basic
            'enable_planning': True,
            'enable_memory': False,
            'planner_model': 'gpt-4o-mini',  # Required when enable_planning=True

            # QD - Disabled
            'qd_enabled': False,

            # Gauntlet - Disabled
            'enable_gauntlet': False,

            # Logging - Clear
            'log_level': 'INFO',
            'verbose': True,

            # Output - Save final
            'save_intermediate_results': False,
            'save_final_results': True,
            'output_dir': './output',

            # Performance - Single worker
            'parallel_workers': 1,

            # Tags
            'tags': ['quickstart']
        }


class ProfileManager:
    """
    Manage configuration profiles.

    Features:
    - Load built-in profiles
    - Save/load custom profiles
    - List available profiles
    - Create new profiles from base profiles
    - Validate profiles
    """

    # Built-in profiles
    PROFILES: Dict[str, Type[BaseProfile]] = {
        'development': DevelopmentProfile,
        'testing': TestingProfile,
        'production': ProductionProfile,
        'benchmarking': BenchmarkingProfile,
        'quickstart': QuickStartProfile,
    }

    def __init__(self, profile_dir: Optional[str] = None):
        """
        Initialize ProfileManager.

        Args:
            profile_dir: Directory to store custom profiles (default: ~/.evolve/profiles)
        """
        if profile_dir is None:
            home = os.path.expanduser('~')
            profile_dir = os.path.join(home, '.evolve', 'profiles')

        self.profile_dir = profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load profile by name.

        Args:
            profile_name: Name of profile ('development', 'testing', etc.)

        Returns:
            Dictionary of profile parameters

        Raises:
            ValueError: If profile not found
        """
        # Check built-in profiles first
        if profile_name in self.PROFILES:
            profile_class = self.PROFILES[profile_name]
            profile = profile_class()
            logger.info(f"Loaded built-in profile: {profile_name}")
            return profile.to_dict()

        # Check custom profiles
        profile_path = os.path.join(self.profile_dir, f'{profile_name}.json')
        if os.path.exists(profile_path):
            return self._load_profile_from_file(profile_path)

        raise ValueError(
            f"Profile '{profile_name}' not found. "
            f"Available: {', '.join(self.list_profiles())}"
        )

    def save_profile(
        self,
        profile_name: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Save custom profile.

        Args:
            profile_name: Name for the profile
            parameters: Parameter dictionary
        """
        profile_path = os.path.join(self.profile_dir, f'{profile_name}.json')

        with open(profile_path, 'w') as f:
            json.dump(parameters, f, indent=2)

        logger.info(f"Saved custom profile '{profile_name}' to {profile_path}")

    def create_profile(
        self,
        name: str,
        base: str = 'quickstart',
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create new profile from base profile with overrides.

        Args:
            name: Name for new profile
            base: Base profile name (default: 'quickstart')
            overrides: Parameter overrides

        Returns:
            New profile parameters
        """
        # Load base profile
        base_params = self.load_profile(base)

        # Apply overrides
        if overrides:
            base_params.update(overrides)

        # Save new profile
        self.save_profile(name, base_params)

        logger.info(f"Created profile '{name}' from base '{base}'")
        return base_params

    def delete_profile(self, name: str) -> None:
        """
        Delete custom profile.

        Args:
            name: Profile name to delete

        Raises:
            ValueError: If trying to delete built-in profile
        """
        if name in self.PROFILES:
            raise ValueError(f"Cannot delete built-in profile '{name}'")

        profile_path = os.path.join(self.profile_dir, f'{name}.json')

        if not os.path.exists(profile_path):
            raise ValueError(f"Profile '{name}' not found")

        os.remove(profile_path)
        logger.info(f"Deleted profile '{name}'")

    def list_profiles(self) -> List[str]:
        """
        List all available profiles.

        Returns:
            List of profile names
        """
        # Built-in profiles
        profiles = list(self.PROFILES.keys())

        # Custom profiles
        if os.path.exists(self.profile_dir):
            for file in os.listdir(self.profile_dir):
                if file.endswith('.json'):
                    profiles.append(file[:-5])  # Remove .json

        return sorted(profiles)

    def get_profile_info(self, profile_name: str) -> ProfileInfo:
        """
        Get information about a profile.

        Args:
            profile_name: Profile name

        Returns:
            ProfileInfo object

        Raises:
            ValueError: If profile not found
        """
        if profile_name not in self.list_profiles():
            raise ValueError(f"Profile '{profile_name}' not found")

        parameters = self.load_profile(profile_name)

        # Determine category
        if profile_name in self.PROFILES:
            category = profile_name
        else:
            category = 'custom'

        # Get description
        descriptions = {
            'development': 'Fast iteration during development',
            'testing': 'Optimized for running tests',
            'production': 'Maximum quality for production',
            'benchmarking': 'Optimized for performance benchmarking',
            'quickstart': 'Quick start for new users',
        }

        description = descriptions.get(profile_name, 'Custom profile')

        return ProfileInfo(
            name=profile_name,
            description=description,
            category=category,
            parameters=parameters
        )

    def _load_profile_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load profile from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


# Convenience functions
def load_profile(profile_name: str) -> Dict[str, Any]:
    """
    Quick function to load a profile.

    Args:
        profile_name: Name of profile

    Returns:
        Profile parameters
    """
    manager = ProfileManager()
    return manager.load_profile(profile_name)


def list_profiles() -> List[str]:
    """List all available profiles"""
    manager = ProfileManager()
    return manager.list_profiles()
