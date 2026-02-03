"""
Preset Manager for configuration presets.

Provides functionality to:
- List all available presets
- Get specific presets
- Apply presets to configurations
- Validate presets
- Compare presets
- Save and load custom presets
"""

import json
import yaml
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
from pydantic import ValidationError

from .base import BasePreset, PresetInfo, ValidationResult, PresetComparison
from ..config import UnifiedEvolutionConfig


class PresetManager:
    """
    Manage and apply configuration presets.

    The preset manager provides a centralized interface for working with
    configuration presets, including listing, applying, validating, and
    comparing presets.
    """

    def __init__(self):
        """Initialize the preset manager and load all built-in presets."""
        self.presets: Dict[str, BasePreset] = {}
        self._load_builtin_presets()

    def _load_builtin_presets(self):
        """Load all built-in presets from submodules."""
        from . import performance, domains, use_cases, systems, problem_types

        # Load all preset classes
        preset_modules = [
            performance,
            domains,
            use_cases,
            systems,
            problem_types
        ]

        for module in preset_modules:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # Check if it's a preset class (not base class)
                if (isinstance(attr, type) and
                    issubclass(attr, BasePreset) and
                    attr != BasePreset):
                    try:
                        preset = attr()
                        self.presets[preset.name] = preset
                    except Exception as e:
                        # Skip presets that can't be instantiated
                        print(f"Warning: Could not instantiate preset {attr_name}: {e}")

    def list_presets(
        self,
        category: Optional[str] = None,
        evolution_mode: Optional[str] = None
    ) -> List[str]:
        """
        List available presets.

        Args:
            category: Filter by category (performance, domain, use_case, system, problem_type)
            evolution_mode: Filter by evolution mode (openevolve, pes, qd, mo, adversarial, hybrid)

        Returns:
            List of preset names
        """
        presets = list(self.presets.keys())

        if category:
            presets = [
                name for name in presets
                if self.presets[name].category == category
            ]

        if evolution_mode:
            presets = [
                name for name in presets
                if self.presets[name].evolution_mode == evolution_mode
            ]

        return sorted(presets)

    def list_categories(self) -> List[str]:
        """Get all available preset categories."""
        categories = set(preset.category for preset in self.presets.values())
        return sorted(categories)

    def get_preset(self, name: str) -> BasePreset:
        """
        Get a preset by name.

        Args:
            name: Preset name

        Returns:
            BasePreset instance

        Raises:
            ValueError: If preset not found
        """
        if name not in self.presets:
            available = ", ".join(self.list_presets())
            raise ValueError(
                f"Preset '{name}' not found. Available presets: {available}"
            )
        return self.presets[name]

    def get_preset_info(self, name: str) -> PresetInfo:
        """
        Get detailed information about a preset.

        Args:
            name: Preset name

        Returns:
            PresetInfo with detailed information
        """
        preset = self.get_preset(name)
        return preset.get_info()

    def apply_preset(
        self,
        name: str,
        base_config: Optional[UnifiedEvolutionConfig] = None
    ) -> UnifiedEvolutionConfig:
        """
        Apply a preset to a configuration.

        Args:
            name: Preset name
            base_config: Optional base configuration to merge with

        Returns:
            UnifiedEvolutionConfig with preset applied
        """
        preset = self.get_preset(name)
        config_dict = preset.to_unified_config()

        # If base config provided, merge it
        if base_config:
            base_dict = base_config.to_dict()
            # Deep merge preset into base
            config_dict = self._deep_merge(base_dict, config_dict)

        # Create and return unified config
        return UnifiedEvolutionConfig.from_dict(config_dict)

    def validate_preset(self, name: str) -> ValidationResult:
        """
        Validate a preset configuration.

        Args:
            name: Preset name

        Returns:
            ValidationResult with validation status
        """
        preset = self.get_preset(name)
        return preset.validate()

    def compare_presets(self, preset1: str, preset2: str) -> PresetComparison:
        """
        Compare two presets.

        Args:
            preset1: First preset name
            preset2: Second preset name

        Returns:
            PresetComparison with differences and similarities
        """
        p1 = self.get_preset(preset1)
        p2 = self.get_preset(preset2)

        # Get parameter summaries
        params1 = p1.get_parameter_summary()
        params2 = p2.get_parameter_summary()

        # Find differences
        differences = {}
        for key in set(params1.keys()) | set(params2.keys()):
            val1 = params1.get(key)
            val2 = params2.get(key)
            if val1 != val2:
                differences[key] = (val1, val2)

        # Find similarities
        similarities = [
            key for key in params1.keys()
            if key in params2 and params1[key] == params2[key]
        ]

        return PresetComparison(
            preset1=preset1,
            preset2=preset2,
            differences=differences,
            similarities=similarities
        )

    def create_preset(
        self,
        name: str,
        config: UnifiedEvolutionConfig,
        description: str,
        category: str = "custom"
    ) -> BasePreset:
        """
        Create a custom preset from a configuration.

        Args:
            name: Preset name
            config: Configuration to create preset from
            description: Preset description
            category: Preset category

        Returns:
            BasePreset instance
        """
        # Create a custom preset class
        class CustomPreset(BasePreset):
            name = name
            category = category
            description = description

        # Extract parameters from config
        common = config.common
        preset = CustomPreset(
            name=name,
            category=category,
            description=description,
            evolution_mode=config.evolution_mode,
            max_iterations=common.max_iterations,
            random_seed=common.random_seed,
            checkpoint_interval=common.checkpoint_interval,
            log_level=common.log_level,
            log_to_console=common.log_to_console,
            log_to_file=common.log_to_file,
            workspace_path=common.workspace_path,
            task_name=common.task_name,
            concurrency=common.concurrency,
            timeout=common.timeout,
        )

        # Store in manager
        self.presets[name] = preset

        return preset

    def save_preset(
        self,
        preset: BasePreset,
        filepath: str,
        format: str = "yaml"
    ) -> None:
        """
        Save a preset to a file.

        Args:
            preset: Preset to save
            filepath: Path to save to
            format: File format (yaml or json)
        """
        config_dict = preset.to_unified_config()
        config_dict["_meta"] = {
            "name": preset.name,
            "category": preset.category,
            "description": preset.description
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_preset(self, filepath: str) -> BasePreset:
        """
        Load a preset from a file.

        Args:
            filepath: Path to load from

        Returns:
            BasePreset instance
        """
        path = Path(filepath)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Extract metadata
        meta = data.pop("_meta", {})

        # Create custom preset
        class LoadedPreset(BasePreset):
            pass

        preset = LoadedPreset(**data, **meta)
        return preset

    def search_presets(
        self,
        keyword: str,
        search_descriptions: bool = True
    ) -> List[str]:
        """
        Search for presets by keyword.

        Args:
            keyword: Keyword to search for
            search_descriptions: Whether to search in descriptions

        Returns:
            List of matching preset names
        """
        keyword_lower = keyword.lower()
        matches = []

        for name, preset in self.presets.items():
            # Check name
            if keyword_lower in name.lower():
                matches.append(name)
                continue

            # Check description
            if search_descriptions and keyword_lower in preset.description.lower():
                matches.append(name)
                continue

        return sorted(matches)

    def get_presets_by_category(self, category: str) -> Dict[str, BasePreset]:
        """
        Get all presets in a category.

        Args:
            category: Category name

        Returns:
            Dictionary mapping preset names to presets
        """
        return {
            name: preset
            for name, preset in self.presets.items()
            if preset.category == category
        }

    def print_preset_summary(self, name: str) -> None:
        """
        Print a summary of a preset.

        Args:
            name: Preset name
        """
        info = self.get_preset_info(name)

        print(f"\n{'='*60}")
        print(f"Preset: {info.name}")
        print(f"Category: {info.category}")
        print(f"{'='*60}")
        print(f"\nDescription: {info.description}")
        print(f"\nWhen to Use:")
        print(f"  {info.when_to_use}")
        print(f"\nTrade-offs:")
        for key, value in info.trade_offs.items():
            print(f"  {key}: {value}")
        print(f"\nRelated Presets:")
        print(f"  {', '.join(info.related_presets)}")
        print(f"\nExample Usage:")
        print(info.example_usage)
        print(f"\n{'='*60}\n")

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# Global preset manager instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """
    Get or create the global preset manager.

    Returns:
        PresetManager instance
    """
    global _preset_manager

    if _preset_manager is None:
        _preset_manager = PresetManager()

    return _preset_manager
