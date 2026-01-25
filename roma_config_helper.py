"""
ROMA Configuration Helper Utilities

Provides helper functions for creating and managing ROMA configurations
for use with problem_decomposition.py
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ROMAModelConfig:
    """
    Configuration for ROMA model settings.

    Attributes:
        provider: AI provider (openai, anthropic, google, openrouter)
        model: Model name
        api_key: API key (optional, will use env var if not provided)
        temperature: Temperature for generation (0.0-1.0)
        max_tokens: Maximum tokens to generate
        prediction_strategy: ROMA prediction strategy
    """
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    prediction_strategy: str = "chain_of_thought"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for ROMA"""
        config = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            config["api_key"] = self.api_key
        return config


@dataclass
class ROMAConfig:
    """
    Complete ROMA configuration for problem decomposition.

    This config provides a convenient way to set up ROMA with all
    the parameters needed for problem_decomposition.py

    Attributes:
        profile: Named profile from ROMA config
        config_path: Path to ROMA YAML config file
        overrides: Config override strings
        use_fractal: Use fractal decomposition
        max_depth: Maximum recursion depth
        max_nodes: Maximum nodes to create
        include_non_leaf: Include intermediate plan nodes
        atomizer_model: Model for atomizer (overrides model)
        planner_model: Model for planner (overrides model)
        enable_context: Enable domain context extraction
        enable_problem_analyzer: Use ProblemAnalyzer for domain context
    """
    # Config file settings
    profile: Optional[str] = None
    config_path: Optional[str] = None
    overrides: List[str] = field(default_factory=list)

    # Model settings
    model: Optional[str] = None
    atomizer_model: Optional[str] = None
    planner_model: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None

    # Decomposition settings
    use_fractal: bool = True
    max_depth: int = 3
    max_nodes: Optional[int] = None
    include_non_leaf: bool = False
    allow_small_components: bool = True

    # Context settings
    enable_context: bool = True
    enable_problem_analyzer: bool = True
    custom_context: Optional[str] = None

    # Performance settings
    cache_context: bool = True

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for problem_decomposition.decompose_content()

        Returns:
            Dictionary of kwargs ready to pass to decompose_content()
        """
        kwargs = {
            "roma_fractal": self.use_fractal,
            "roma_allow_small_components": self.allow_small_components,
            "roma_max_depth": self.max_depth,
            "roma_include_non_leaf": self.include_non_leaf,
            "use_problem_analyzer": self.enable_problem_analyzer,
        }

        # Add optional parameters
        if self.profile:
            kwargs["roma_profile"] = self.profile
        if self.config_path:
            kwargs["roma_config_path"] = self.config_path
        if self.overrides:
            kwargs["roma_overrides"] = self.overrides
        if self.model:
            kwargs["roma_model"] = self.model
        if self.atomizer_model:
            kwargs["roma_atomizer_model"] = self.atomizer_model
        if self.planner_model:
            kwargs["roma_planner_model"] = self.planner_model
        if self.model_config:
            kwargs["roma_model_config"] = self.model_config
        if self.max_nodes:
            kwargs["roma_max_nodes"] = self.max_nodes
        if self.custom_context:
            kwargs["roma_context"] = self.custom_context

        return kwargs


class ROMAConfigPresets:
    """Predefined ROMA configurations for common use cases"""

    @staticmethod
    def fast() -> ROMAConfig:
        """
        Fast configuration for quick decomposition.

        Uses lower depth and fewer nodes for speed.
        """
        return ROMAConfig(
            model="gpt-4o-mini",
            use_fractal=True,
            max_depth=2,
            max_nodes=20,
            allow_small_components=True,
            enable_problem_analyzer=False,  # Skip for speed
        )

    @staticmethod
    def balanced() -> ROMAConfig:
        """
        Balanced configuration for general use.

        Good balance between speed and quality.
        """
        return ROMAConfig(
            model="gpt-4o",
            use_fractal=True,
            max_depth=3,
            max_nodes=40,
            allow_small_components=True,
            enable_problem_analyzer=True,
        )

    @staticmethod
    def thorough() -> ROMAConfig:
        """
        Thorough configuration for complex problems.

        Deeper decomposition with more nodes.
        """
        return ROMAConfig(
            model="gpt-4o",
            use_fractal=True,
            max_depth=4,
            max_nodes=100,
            allow_small_components=False,  # Filter small components
            enable_problem_analyzer=True,
        )

    @staticmethod
    def hierarchical() -> ROMAConfig:
        """
        Hierarchical configuration for structured problems.

        Optimized for hierarchical decomposition.
        """
        return ROMAConfig(
            model="gpt-4o",
            use_fractal=True,
            max_depth=5,
            max_nodes=150,
            include_non_leaf=True,  # Include intermediate nodes
            allow_small_components=False,
            enable_problem_analyzer=True,
        )

    @staticmethod
    def code_focused() -> ROMAConfig:
        """
        Code-focused configuration for software problems.

        Optimized for code and algorithm decomposition.
        """
        return ROMAConfig(
            model="claude-3-5-sonnet-20241022",
            atomizer_model="claude-3-5-sonnet-20241022",
            planner_model="claude-3-5-sonnet-20241022",
            use_fractal=True,
            max_depth=3,
            max_nodes=50,
            allow_small_components=False,
            enable_problem_analyzer=True,
        )

    @staticmethod
    def research_focused() -> ROMAConfig:
        """
        Research-focused configuration for analysis tasks.

        Optimized for research and analysis problems.
        """
        return ROMAConfig(
            model="gpt-4o",
            use_fractal=True,
            max_depth=3,
            max_nodes=60,
            allow_small_components=True,
            enable_problem_analyzer=True,
            custom_context="Focus on thorough analysis and comprehensive coverage of topics.",
        )


def create_roma_config_from_env() -> ROMAConfig:
    """
    Create ROMA config from environment variables.

    Environment Variables:
        ROMA_MODEL: Model name
        ROMA_PROVIDER: Provider name
        ROMA_MAX_DEPTH: Maximum depth
        ROMA_MAX_NODES: Maximum nodes
        ROMA_PROFILE: Named profile
        ROMA_CONFIG_PATH: Path to config file

    Returns:
        ROMAConfig with settings from environment
    """
    config = ROMAConfig()

    if model := os.getenv("ROMA_MODEL"):
        config.model = model
    if profile := os.getenv("ROMA_PROFILE"):
        config.profile = profile
    if config_path := os.getenv("ROMA_CONFIG_PATH"):
        config.config_path = config_path
    if max_depth := os.getenv("ROMA_MAX_DEPTH"):
        config.max_depth = int(max_depth)
    if max_nodes := os.getenv("ROMA_MAX_NODES"):
        config.max_nodes = int(max_nodes)

    return config


def validate_roma_config(config: ROMAConfig) -> List[str]:
    """
    Validate ROMA configuration.

    Args:
        config: ROMAConfig to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check that at least one config method is specified
    if not any([config.profile, config.config_path, config.model]):
        errors.append(
            "Must specify at least one of: profile, config_path, or model"
        )

    # Validate depth
    if config.max_depth < 1:
        errors.append("max_depth must be at least 1")
    if config.max_depth > 10:
        errors.append("max_depth should not exceed 10 (performance risk)")

    # Validate nodes
    if config.max_nodes is not None and config.max_nodes < 1:
        errors.append("max_nodes must be at least 1")

    # Check for conflicting settings
    if config.profile and config.config_path:
        errors.append(
            "Cannot specify both profile and config_path (use one or the other)"
        )

    return errors


def merge_roma_configs(*configs: ROMAConfig) -> ROMAConfig:
    """
    Merge multiple ROMA configs, later configs override earlier ones.

    Args:
        *configs: ROMAConfig instances to merge

    Returns:
        Merged ROMAConfig
    """
    if not configs:
        return ROMAConfig()

    merged = configs[0]
    for config in configs[1:]:
        # Override non-None values
        if config.profile:
            merged.profile = config.profile
        if config.config_path:
            merged.config_path = config.config_path
        if config.overrides:
            merged.overrides = config.overrides
        if config.model:
            merged.model = config.model
        if config.atomizer_model:
            merged.atomizer_model = config.atomizer_model
        if config.planner_model:
            merged.planner_model = config.planner_model
        if config.model_config:
            merged.model_config = config.model_config
        if config.max_depth:
            merged.max_depth = config.max_depth
        if config.max_nodes:
            merged.max_nodes = config.max_nodes
        if config.custom_context:
            merged.custom_context = config.custom_context

        # Boolean flags
        merged.use_fractal = config.use_fractal
        merged.include_non_leaf = config.include_non_leaf
        merged.allow_small_components = config.allow_small_components
        merged.enable_context = config.enable_context
        merged.enable_problem_analyzer = config.enable_problem_analyzer
        merged.cache_context = config.cache_context

    return merged


# Example usage
if __name__ == "__main__":
    # Create a custom config
    config = ROMAConfig(
        model="gpt-4o",
        max_depth=3,
        use_fractal=True,
    )

    # Validate it
    errors = validate_roma_config(config)
    if errors:
        print("Config errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Config is valid!")

        # Get kwargs for decompose_content
        kwargs = config.to_kwargs()
        print("Kwargs:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    # Example using presets
    print("\n--- Presets ---")
    print("Fast config:", ROMAConfigPresets.fast().to_kwargs())
    print("Balanced config:", ROMAConfigPresets.balanced().to_kwargs())
    print("Thorough config:", ROMAConfigPresets.thorough().to_kwargs())
