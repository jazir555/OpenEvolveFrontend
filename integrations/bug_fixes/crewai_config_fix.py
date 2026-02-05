"""
CrewAI Configuration Override Adapter

Fixes invalid paths in CrewAI config without modifying core files.
Uses YAML override pattern to provide corrected configuration.

Bug Fixed:
- phases_folder: ./example_workflows/crackme_solving -> ./example_workflows/prd_to_software
- worktree_base: /tmp/crewai_worktrees -> ./crewai_worktrees
- project_root: /tmp/test_3gaur34 -> .
- main_repo_path: /tmp/test_3gaur34 -> .

Usage:
    from integrations.bug_fixes import CrewAIConfigOverride

    # Load config with fixes applied
    config_override = CrewAIConfigOverride()
    fixed_config = config_override.get_fixed_config()

    # Or apply to existing config
    original_config = load_crewai_config()
    fixed_config = config_override.apply_fixes(original_config)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CrewAIConfigOverride:
    """
    Provides corrected CrewAI configuration without modifying core files.

    Uses override pattern to fix path issues in crewai_config.yaml.
    """

    # Path corrections (core paths -> fixed paths)
    PATH_FIXES = {
        'phases_folder': './example_workflows/prd_to_software',
        'worktree_base': './crewai_worktrees',
        'project_root': '.',
    }

    GIT_PATH_FIXES = {
        'main_repo_path': '.',
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config override.

        Args:
            config_path: Path to crewai_config.yaml (auto-detected if None)
        """
        if config_path is None:
            # Auto-detect CrewAI config
            possible_paths = [
                './crewai/crewai_config.yaml',
                '../crewai/crewai_config.yaml',
                '../../crewai/crewai_config.yaml',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        self.config_path = config_path
        self._original_config: Optional[Dict[str, Any]] = None
        self._fixed_config: Optional[Dict[str, Any]] = None

    def load_original_config(self) -> Dict[str, Any]:
        """Load the original CrewAI config."""
        if self._original_config is not None:
            return self._original_config

        if self.config_path is None or not os.path.exists(self.config_path):
            logger.warning(f"CrewAI config not found at {self.config_path}")
            self._original_config = {}
            return self._original_config

        try:
            with open(self.config_path, 'r') as f:
                self._original_config = yaml.safe_load(f)
            logger.info(f"Loaded CrewAI config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load CrewAI config: {e}")
            self._original_config = {}

        return self._original_config

    def apply_fixes(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply path fixes to CrewAI config.

        Args:
            config: Config dict to fix (loads original if None)

        Returns:
            Fixed config dict
        """
        if config is None:
            config = self.load_original_config()

        if not config:
            return {}

        # Create deep copy to avoid modifying original
        import copy
        fixed_config = copy.deepcopy(config)

        # Fix paths section
        if 'paths' in fixed_config:
            paths = fixed_config['paths']
            for key, new_value in self.PATH_FIXES.items():
                if key in paths:
                    old_value = paths[key]
                    paths[key] = new_value
                    logger.debug(f"Fixed path: paths.{key}: {old_value} -> {new_value}")

        # Fix git section
        if 'git' in fixed_config:
            git = fixed_config['git']
            for key, new_value in self.GIT_PATH_FIXES.items():
                if key in git:
                    old_value = git[key]
                    git[key] = new_value
                    logger.debug(f"Fixed path: git.{key}: {old_value} -> {new_value}")

        # Ensure directories exist
        self._ensure_directories(fixed_config)

        self._fixed_config = fixed_config
        return fixed_config

    def _ensure_directories(self, config: Dict[str, Any]) -> None:
        """Create necessary directories if they don't exist."""
        if 'paths' not in config:
            return

        paths = config['paths']
        directories_to_create = [
            paths.get('worktree_base'),
            paths.get('phases_folder'),
        ]

        for dir_path in directories_to_create:
            if dir_path and not dir_path.startswith('/tmp'):
                # Convert relative paths
                if not os.path.isabs(dir_path):
                    dir_path = os.path.abspath(dir_path)

                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Ensured directory exists: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not create directory {dir_path}: {e}")

    def get_fixed_config(self) -> Dict[str, Any]:
        """
        Get the fixed CrewAI config.

        Loads original config, applies fixes, ensures directories exist.

        Returns:
            Fixed configuration dictionary
        """
        return self.apply_fixes()

    def save_fixed_config(self, output_path: str) -> None:
        """
        Save fixed config to a file (for debugging/validation).

        Args:
            output_path: Where to save the fixed config
        """
        fixed_config = self.get_fixed_config()

        try:
            with open(output_path, 'w') as f:
                yaml.safe_dump(fixed_config, f, default_flow_style=False)
            logger.info(f"Saved fixed config to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save fixed config: {e}")


# Convenience function for quick usage
def get_crewai_config() -> Dict[str, Any]:
    """
    Quick access to fixed CrewAI config.

    Usage:
        from integrations.bug_fixes.crewai_config_fix import get_crewai_config
        config = get_crewai_config()
    """
    override = CrewAIConfigOverride()
    return override.get_fixed_config()
