"""
ROMA Recomposition Configuration Helper Utilities

Provides helper functions for creating and managing ROMA configurations
specifically for solution recomposition (assembling sub-solutions back into integrated solutions).
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ROMARecompositionConfig:
    """
    Configuration for ROMA-based solution recomposition.

    This config provides parameters for controlling how ROMA assembles
    sub-solutions back into an integrated solution.

    Attributes:
        enable_roma: Whether to use ROMA for recomposition
        deterministic: Use deterministic assembly (default: True)
            - True: ROMA decides structure only, sub-solutions inserted verbatim
            - False: ROMA may rewrite sub-solutions (creative mode)
        max_depth: Maximum recursion depth for recomposition (default: 2)
        execution_mode: ROMA execution mode ("recursive" or "iterative")
        provider: AI provider (openai, anthropic, etc.)
        model: Model name for recomposition
        api_key: API key (optional, will use env var if not provided)
        temperature: Temperature for generation (0.0-1.0, default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4000)
        strategy: Recomposition strategy to use
        track_in_crewai: Whether to track recomposition in CrewAI
        custom_context: Custom context string for ROMA
        extra_context: Extra context appended to auto-generated context
        enable_conflict_resolution: Enable LLM-mediated conflict resolution
        conflict_resolution_fallback: Fallback strategy if LLM fails
    """

    enable_roma: bool = True
    deterministic: bool = True  # NEW: Default to deterministic mode
    max_depth: int = 2
    execution_mode: str = "recursive"
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    strategy: str = "chain_of_thought"
    track_in_crewai: bool = False
    custom_context: Optional[str] = None
    extra_context: Optional[str] = None
    enable_conflict_resolution: bool = True
    conflict_resolution_fallback: str = "priority"

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for SolutionAssembler.assemble_solution()

        Returns:
            Dictionary of kwargs ready to pass to assemble_solution()
        """
        kwargs = {
            "assembly_strategy": "roma" if self.enable_roma else "hierarchical",
            "enable_roma": self.enable_roma,
            "roma_deterministic": self.deterministic,  # NEW: Include deterministic mode
            "roma_max_depth": self.max_depth,
            "roma_execution_mode": self.execution_mode,
        }

        # Add optional parameters
        if self.provider:
            kwargs["roma_provider"] = self.provider
        if self.model:
            kwargs["roma_model"] = self.model
        if self.custom_context:
            kwargs["roma_context"] = self.custom_context
        if self.extra_context:
            kwargs["roma_extra_context"] = self.extra_context
        if self.track_in_crewai:
            kwargs["track_in_crewai"] = True

        return kwargs

    def validate(self) -> List[str]:
        """
        Validate the recomposition configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate depth
        if self.max_depth < 1:
            errors.append("max_depth must be at least 1")
        if self.max_depth > 5:
            errors.append("max_depth should not exceed 5 (recomposition typically needs less depth)")

        # Validate temperature
        if not 0.0 <= self.temperature <= 1.0:
            errors.append("temperature must be between 0.0 and 1.0")

        # Validate max_tokens
        if self.max_tokens < 100:
            errors.append("max_tokens must be at least 100")

        # Validate execution mode
        if self.execution_mode not in ["recursive", "iterative"]:
            errors.append("execution_mode must be 'recursive' or 'iterative'")

        # Validate conflict resolution fallback
        if self.conflict_resolution_fallback not in ["priority", "merge", "manual"]:
            errors.append("conflict_resolution_fallback must be 'priority', 'merge', or 'manual'")

        return errors


class ROMARecompositionPresets:
    """Predefined ROMA recomposition configurations for common scenarios"""

    @staticmethod
    def fast() -> ROMARecompositionConfig:
        """
        Fast recomposition for quick integration.

        Uses lower depth and fewer tokens for speed.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=1,
            max_tokens=2000,
            temperature=0.5,
            execution_mode="iterative",
            track_in_crewai=False,
        )

    @staticmethod
    def balanced() -> ROMARecompositionConfig:
        """
        Balanced recomposition for general use.

        Good balance between speed and quality.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=2,
            max_tokens=4000,
            temperature=0.7,
            execution_mode="recursive",
            track_in_crewai=False,
        )

    @staticmethod
    def thorough() -> ROMARecompositionConfig:
        """
        Thorough recomposition for complex solutions.

        Deeper recomposition with more tokens.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=3,
            max_tokens=6000,
            temperature=0.8,
            execution_mode="recursive",
            track_in_crewai=False,
            enable_conflict_resolution=True,
        )

    @staticmethod
    def high_conflict() -> ROMARecompositionConfig:
        """
        Recomposition configuration for solutions with many conflicts.

        Emphasizes conflict resolution and coherence.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=2,
            max_tokens=5000,
            temperature=0.7,
            execution_mode="recursive",
            enable_conflict_resolution=True,
            conflict_resolution_fallback="merge",
            extra_context="Priority: Resolve all conflicts and create a coherent, unified solution.",
        )

    @staticmethod
    def code_focused() -> ROMARecompositionConfig:
        """
        Code-focused recomposition for software solutions.

        Optimized for integrating code and technical components.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=2,
            max_tokens=4000,
            temperature=0.3,  # Lower temperature for more deterministic code
            execution_mode="recursive",
            model="claude-3-5-sonnet-20241022",  # Good for code
            extra_context="Domain: Software Development. Focus on code integration, API compatibility, and technical correctness.",
        )

    @staticmethod
    def documentation_focused() -> ROMARecompositionConfig:
        """
        Documentation-focused recomposition for written content.

        Optimized for integrating documentation and prose.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=2,
            max_tokens=5000,
            temperature=0.8,  # Higher temperature for more natural language
            execution_mode="recursive",
            model="gpt-4o",
            extra_context="Domain: Technical Documentation. Focus on clarity, readability, and coherent narrative flow.",
        )

    @staticmethod
    def creative() -> ROMARecompositionConfig:
        """
        Creative recomposition for innovative solutions.

        Uses higher temperature for more creative integration.
        """
        return ROMARecompositionConfig(
            enable_roma=True,
            max_depth=3,
            max_tokens=6000,
            temperature=0.9,  # High temperature for creativity
            execution_mode="recursive",
            extra_context="Approach: Creative and innovative synthesis of solutions. Emphasize novel integration patterns.",
        )


def create_recomposition_config_from_env() -> ROMARecompositionConfig:
    """
    Create ROMA recomposition config from environment variables.

    Environment Variables:
        ROMA_RECOMPOSITION_MODEL: Model name
        ROMA_RECOMPOSITION_PROVIDER: Provider name
        ROMA_RECOMPOSITION_MAX_DEPTH: Maximum depth
        ROMA_RECOMPOSITION_MAX_TOKENS: Maximum tokens
        ROMA_RECOMPOSITION_TEMPERATURE: Temperature
        ROMA_RECOMPOSITION_STRATEGY: Strategy to use

    Returns:
        ROMARecompositionConfig with settings from environment
    """
    config = ROMARecompositionConfig()

    if model := os.getenv("ROMA_RECOMPOSITION_MODEL"):
        config.model = model
    if provider := os.getenv("ROMA_RECOMPOSITION_PROVIDER"):
        config.provider = provider
    if max_depth := os.getenv("ROMA_RECOMPOSITION_MAX_DEPTH"):
        config.max_depth = int(max_depth)
    if max_tokens := os.getenv("ROMA_RECOMPOSITION_MAX_TOKENS"):
        config.max_tokens = int(max_tokens)
    if temperature := os.getenv("ROMA_RECOMPOSITION_TEMPERATURE"):
        config.temperature = float(temperature)
    if strategy := os.getenv("ROMA_RECOMPOSITION_STRATEGY"):
        config.strategy = strategy

    return config


def get_recommended_recomposition_config(
    num_sub_solutions: int,
    num_conflicts: int,
    complexity: str = "medium",
    content_type: str = "general"
) -> ROMARecompositionConfig:
    """
    Get recommended ROMA recomposition configuration based on characteristics.

    Args:
        num_sub_solutions: Number of sub-solutions to integrate
        num_conflicts: Number of conflicts detected
        complexity: Problem complexity ("low", "medium", "high")
        content_type: Type of content ("code", "documentation", "general")

    Returns:
        Recommended ROMARecompositionConfig
    """
    # Base config
    config = ROMARecompositionConfig()

    # Adjust based on number of solutions
    if num_sub_solutions > 10:
        config.max_depth = 3
        config.max_tokens = 6000
    elif num_sub_solutions > 5:
        config.max_depth = 2
        config.max_tokens = 4000
    else:
        config.max_depth = 1
        config.max_tokens = 3000

    # Adjust based on conflicts
    if num_conflicts > 5:
        config.enable_conflict_resolution = True
        config.conflict_resolution_fallback = "merge"
        config.temperature = 0.8
    elif num_conflicts > 2:
        config.enable_conflict_resolution = True
        config.temperature = 0.7

    # Adjust based on complexity
    if complexity == "high":
        config.max_depth = min(config.max_depth + 1, 3)
        config.temperature = min(config.temperature + 0.1, 0.9)
    elif complexity == "low":
        config.max_depth = max(config.max_depth - 1, 1)
        config.temperature = max(config.temperature - 0.1, 0.5)

    # Adjust based on content type
    if content_type == "code":
        config.model = "claude-3-5-sonnet-20241022"
        config.temperature = 0.3
    elif content_type == "documentation":
        config.model = "gpt-4o"
        config.temperature = 0.8
    elif content_type == "creative":
        config.temperature = 0.9
        config.max_depth = 3

    return config


# Example usage
if __name__ == "__main__":
    # Create a custom config
    config = ROMARecompositionConfig(
        model="gpt-4o",
        max_depth=2,
        temperature=0.7,
    )

    # Validate it
    errors = config.validate()
    if errors:
        print("Config errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Config is valid!")

        # Get kwargs for assemble_solution
        kwargs = config.to_kwargs()
        print("Kwargs:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    # Example using presets
    print("\n--- Presets ---")
    print("Fast config:", ROMARecompositionPresets.fast().to_kwargs())
    print("Balanced config:", ROMARecompositionPresets.balanced().to_kwargs())
    print("Thorough config:", ROMARecompositionPresets.thorough().to_kwargs())
    print("High conflict config:", ROMARecompositionPresets.high_conflict().to_kwargs())
