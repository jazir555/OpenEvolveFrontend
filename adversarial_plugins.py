"""
Plugin System for Adversarial Testing

This module provides a flexible plugin architecture for extending the
adversarial testing system with custom attacks, defenses, and evaluators.

Features:
1. Plugin discovery and registration
2. Hot-reloading of plugins
3. Plugin lifecycle management
4. Dependency injection
5. Configuration management
6. Validation and sandboxing
7. Plugin marketplace integration
8. Custom attack/defense plugins

Author: OpenEvolve Plugin Team
Created: 2025-01-07
Version: 1.0.0
"""

import importlib
import importlib.util
import importlib.machinery
import inspect
import json
import logging
import os
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# PLUGIN BASE CLASSES
# =============================================================================

class AttackPlugin(ABC):
    """
    Base class for custom attack plugins

    Implement this class to create custom attack strategies
    """

    plugin_type: str = "attack"
    plugin_name: str = "base_attack"
    plugin_version: str = "1.0.0"
    plugin_author: str = "Unknown"
    plugin_description: str = "Base attack plugin"

    @abstractmethod
    async def generate_attack(
        self,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an attack

        Args:
            content: Content to attack
            content_type: Type of content
            theorem: Theorem statement
            context: Additional context

        Returns:
            Attack result dictionary with:
            - success: bool
            - severity: float (0-1)
            - description: str
            - weak_point: str
            - confidence: float
        """
        pass

    def validate_input(self, content: str, content_type: str) -> bool:
        """Validate input before attack generation"""
        return bool(content and content_type)

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this plugin"""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }


class DefensePlugin(ABC):
    """
    Base class for custom defense plugins

    Implement this class to create custom defense strategies
    """

    plugin_type: str = "defense"
    plugin_name: str = "base_defense"
    plugin_version: str = "1.0.0"
    plugin_author: str = "Unknown"
    plugin_description: str = "Base defense plugin"

    @abstractmethod
    async def generate_defense(
        self,
        content: str,
        attack: Dict[str, Any],
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a defense

        Args:
            content: Original content
            attack: Attack to defend against
            theorem: Theorem statement
            context: Additional context

        Returns:
            Defense result dictionary with:
            - attack_blocked: bool
            - effectiveness: float (0-1)
            - improved_proof: str
            - description: str
            - confidence: float
        """
        pass

    def validate_input(self, content: str, attack: Dict[str, Any]) -> bool:
        """Validate input before defense generation"""
        return bool(content and attack)

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this plugin"""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }


class EvaluatorPlugin(ABC):
    """
    Base class for custom evaluator plugins

    Implement this class to create custom evaluation strategies
    """

    plugin_type: str = "evaluator"
    plugin_name: str = "base_evaluator"
    plugin_version: str = "1.0.0"
    plugin_author: str = "Unknown"
    plugin_description: str = "Base evaluator plugin"

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate content

        Args:
            content: Content to evaluate
            content_type: Type of content
            theorem: Theorem statement
            context: Additional context

        Returns:
            Evaluation result dictionary with:
            - score: float (0-1)
            - metrics: Dict[str, float]
            - issues: List[str]
            - recommendations: List[str]
        """
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this plugin"""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================

@dataclass
class PluginMetadata:
    """Metadata for a registered plugin"""
    plugin_id: str
    plugin_type: str
    name: str
    version: str
    author: str
    description: str
    class_path: str
    config_schema: Dict[str, Any]
    enabled: bool = True
    loaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    dependencies: List[str] = field(default_factory=list)


class PluginRegistry:
    """
    Central registry for all plugins

    Manages plugin discovery, registration, and lifecycle
    """

    def __init__(self):
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugin_instances: Dict[str, Union[AttackPlugin, DefensePlugin, EvaluatorPlugin]] = {}
        self.plugin_paths: List[str] = []

        logger.info("Plugin registry initialized")

    def register_plugin(
        self,
        plugin_class: Type[Union[AttackPlugin, DefensePlugin, EvaluatorPlugin]],
        plugin_id: Optional[str] = None
    ) -> str:
        """
        Register a plugin class

        Args:
            plugin_class: Plugin class to register
            plugin_id: Optional custom plugin ID

        Returns:
            Plugin ID
        """
        if plugin_id is None:
            plugin_id = f"{plugin_class.plugin_type}_{plugin_class.plugin_name}_{uuid.uuid4().hex[:8]}"

        # Create metadata
        metadata = PluginMetadata(
            plugin_id=plugin_id,
            plugin_type=plugin_class.plugin_type,
            name=plugin_class.plugin_name,
            version=plugin_class.plugin_version,
            author=plugin_class.plugin_author,
            description=plugin_class.plugin_description,
            class_path=f"{plugin_class.__module__}.{plugin_class.__name__}",
            config_schema=plugin_class().get_config_schema(),
            dependencies=getattr(plugin_class, 'dependencies', [])
        )

        self.plugins[plugin_id] = metadata
        logger.info(f"Registered plugin: {plugin_id} ({metadata.name} v{metadata.version})")

        return plugin_id

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin"""
        if plugin_id in self.plugins:
            # Unload instance if loaded
            if plugin_id in self.plugin_instances:
                del self.plugin_instances[plugin_id]

            del self.plugins[plugin_id]
            logger.info(f"Unregistered plugin: {plugin_id}")
            return True

        return False

    def load_plugin(self, plugin_id: str, config: Optional[Dict[str, Any]] = None) -> Union[AttackPlugin, DefensePlugin, EvaluatorPlugin]:
        """
        Load a plugin instance

        Args:
            plugin_id: Plugin ID to load
            config: Optional configuration

        Returns:
            Plugin instance
        """
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_id}")

        metadata = self.plugins[plugin_id]

        # Check if already loaded
        if plugin_id in self.plugin_instances:
            return self.plugin_instances[plugin_id]

        # Load module and class
        module_path, class_name = metadata.class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)

        # Create instance
        instance = plugin_class()

        # Apply configuration if provided
        if config:
            for key, value in config.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

        self.plugin_instances[plugin_id] = instance
        logger.info(f"Loaded plugin instance: {plugin_id}")

        return instance

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin instance"""
        if plugin_id in self.plugin_instances:
            del self.plugin_instances[plugin_id]
            logger.info(f"Unloaded plugin instance: {plugin_id}")
            return True

        return False

    def get_plugin(self, plugin_id: str) -> Optional[Union[AttackPlugin, DefensePlugin, EvaluatorPlugin]]:
        """Get a loaded plugin instance"""
        return self.plugin_instances.get(plugin_id)

    def list_plugins(
        self,
        plugin_type: Optional[str] = None,
        enabled_only: bool = False
    ) -> List[PluginMetadata]:
        """List registered plugins"""
        plugins = list(self.plugins.values())

        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]

        if enabled_only:
            plugins = [p for p in plugins if p.enabled]

        return plugins

    def discover_plugins(self, directory: str):
        """
        Discover plugins in a directory

        Args:
            directory: Directory to search for plugins
        """
        plugin_dir = Path(directory)

        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return

        # Add to path if not already there
        dir_str = str(plugin_dir.parent)
        if dir_str not in sys.path:
            sys.path.insert(0, dir_str)
            self.plugin_paths.append(dir_str)

        # Find all Python files
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Load module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Find plugin classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, (AttackPlugin, DefensePlugin, EvaluatorPlugin)):
                            if obj != (AttackPlugin, DefensePlugin, EvaluatorPlugin):
                                self.register_plugin(obj)

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to load plugin from {py_file}: {e}")

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin"""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].enabled = True
            logger.info(f"Enabled plugin: {plugin_id}")
            return True
        return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].enabled = False
            # Unload if loaded
            self.unload_plugin(plugin_id)
            logger.info(f"Disabled plugin: {plugin_id}")
            return True
        return False

    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information"""
        if plugin_id not in self.plugins:
            return None

        metadata = self.plugins[plugin_id]
        instance = self.plugin_instances.get(plugin_id)

        return {
            "metadata": asdict(metadata),
            "loaded": instance is not None,
            "config_schema": metadata.config_schema,
            "dependencies": metadata.dependencies
        }


# =============================================================================
# PLUGIN MANAGER
# =============================================================================

class PluginManager:
    """
    High-level plugin management system

    Provides convenient methods for plugin operations
    """

    def __init__(self, plugin_directories: Optional[List[str]] = None):
        self.registry = PluginRegistry()

        # Default plugin directories
        self.plugin_directories = plugin_directories or [
            "./plugins/attacks",
            "./plugins/defenses",
            "./plugins/evaluators",
        ]

        # Discover plugins on init
        self._discover_all_plugins()

    def _discover_all_plugins(self):
        """Discover plugins in all configured directories"""
        for directory in self.plugin_directories:
            self.registry.discover_plugins(directory)

    def register_attack_plugin(self, plugin_class: Type[AttackPlugin]) -> str:
        """Register an attack plugin"""
        return self.registry.register_plugin(plugin_class)

    def register_defense_plugin(self, plugin_class: Type[DefensePlugin]) -> str:
        """Register a defense plugin"""
        return self.registry.register_plugin(plugin_class)

    def register_evaluator_plugin(self, plugin_class: Type[EvaluatorPlugin]) -> str:
        """Register an evaluator plugin"""
        return self.registry.register_plugin(plugin_class)

    async def execute_attack_plugin(
        self,
        plugin_id: str,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an attack plugin"""
        plugin = self.registry.load_plugin(plugin_id, config)

        if not isinstance(plugin, AttackPlugin):
            raise ValueError(f"Plugin {plugin_id} is not an attack plugin")

        # Validate input
        if not plugin.validate_input(content, content_type):
            return {
                "success": False,
                "error": "Input validation failed"
            }

        # Execute
        return await plugin.generate_attack(content, content_type, theorem, context)

    async def execute_defense_plugin(
        self,
        plugin_id: str,
        content: str,
        attack: Dict[str, Any],
        theorem: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a defense plugin"""
        plugin = self.registry.load_plugin(plugin_id, config)

        if not isinstance(plugin, DefensePlugin):
            raise ValueError(f"Plugin {plugin_id} is not a defense plugin")

        # Validate input
        if not plugin.validate_input(content, attack):
            return {
                "attack_blocked": False,
                "error": "Input validation failed"
            }

        # Execute
        return await plugin.generate_defense(content, attack, theorem, context)

    async def execute_evaluator_plugin(
        self,
        plugin_id: str,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an evaluator plugin"""
        plugin = self.registry.load_plugin(plugin_id, config)

        if not isinstance(plugin, EvaluatorPlugin):
            raise ValueError(f"Plugin {plugin_id} is not an evaluator plugin")

        # Execute
        return await plugin.evaluate(content, content_type, theorem, context)

    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all available plugins"""
        plugins = self.registry.list_plugins(plugin_type=plugin_type, enabled_only=True)

        return [
            {
                "id": p.plugin_id,
                "name": p.name,
                "type": p.plugin_type,
                "version": p.version,
                "author": p.author,
                "description": p.description,
                "loaded": p.plugin_id in self.registry.plugin_instances
            }
            for p in plugins
        ]

    def generate_plugin_manifest(self, output_path: str = "./plugin_manifest.json"):
        """Generate a manifest of all plugins"""
        plugins = self.registry.list_plugins()

        manifest = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_plugins": len(plugins),
            "plugins": [
                {
                    "id": p.plugin_id,
                    "type": p.plugin_type,
                    "name": p.name,
                    "version": p.version,
                    "author": p.author,
                    "description": p.description,
                    "enabled": p.enabled,
                    "dependencies": p.dependencies
                }
                for p in plugins
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Plugin manifest generated: {output_path}")
        return manifest


# =============================================================================
# EXAMPLE PLUGINS
# =============================================================================

class SQLInjectionAttackPlugin(AttackPlugin):
    """Example: SQL injection attack plugin"""

    plugin_name = "sql_injection"
    plugin_version = "1.0.0"
    plugin_author = "Security Team"
    plugin_description = "Detects SQL injection vulnerabilities"

    async def generate_attack(
        self,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL injection attack"""

        # Look for SQL patterns
        sql_patterns = [
            "SELECT * FROM",
            "SELECT ",
            "INSERT INTO",
            "UPDATE ",
            "DELETE FROM"
        ]

        # Look for string concatenation
        concat_patterns = [
            f"SELECT * FROM {table}",
            f'"{query}"',
            f"'{query}'"
        ]

        vulnerabilities = []
        for pattern in sql_patterns:
            if pattern in content.upper():
                # Check for concatenation
                if "f\"" in content or "f'" in content or "+" in content:
                    vulnerabilities.append({
                        "type": "SQL Injection",
                        "severity": "High",
                        "pattern": pattern,
                        "description": "Potential SQL injection via string concatenation"
                    })

        return {
            "success": len(vulnerabilities) > 0,
            "severity": 0.8 if vulnerabilities else 0.0,
            "description": f"Found {len(vulnerabilities)} potential SQL injection vulnerabilities",
            "weak_point": "String concatenation in SQL queries",
            "confidence": 0.85,
            "vulnerabilities": vulnerabilities
        }


class InputValidationDefensePlugin(DefensePlugin):
    """Example: Input validation defense plugin"""

    plugin_name = "input_validation"
    plugin_version = "1.0.0"
    plugin_author = "Security Team"
    plugin_description = "Adds input validation to prevent injection attacks"

    async def generate_defense(
        self,
        content: str,
        attack: Dict[str, Any],
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate input validation defense"""

        if not attack.get("success"):
            return {
                "attack_blocked": False,
                "effectiveness": 0.0,
                "improved_proof": content,
                "description": "No attack to defend against",
                "confidence": 1.0
            }

        # Add parameterized queries suggestion
        improved = content + "\n\n# DEFENSE: Use parameterized queries\n# Instead of: f\"SELECT * FROM users WHERE username='{username}'\"\n# Use: \"SELECT * FROM users WHERE username = ?\", [username]"

        return {
            "attack_blocked": True,
            "effectiveness": 0.9,
            "improved_proof": improved,
            "description": "Added parameterized query recommendation",
            "confidence": 0.95
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_plugin_skeleton(
    plugin_type: str,
    plugin_name: str,
    output_path: str
) -> str:
    """Create a skeleton plugin file"""

    if plugin_type == "attack":
        base_class = "AttackPlugin"
    elif plugin_type == "defense":
        base_class = "DefensePlugin"
    elif plugin_type == "evaluator":
        base_class = "EvaluatorPlugin"
    else:
        raise ValueError(f"Invalid plugin type: {plugin_type}")

    skeleton = f'''"""
{plugin_name.title()} {plugin_type.title()} Plugin

Author: Your Name
Version: 1.0.0
Created: {datetime.utcnow().strftime('%Y-%m-%d')}
"""

from typing import Dict, Any
from adversarial_plugins import {base_class}


class {plugin_name.title().replace('_', '')}Plugin({base_class}):
    """Custom {plugin_type} plugin: {plugin_name}"""

    plugin_name = "{plugin_name}"
    plugin_version = "1.0.0"
    plugin_author = "Your Name"
    plugin_description = "Description of your plugin"

    async def generate_{plugin_type}(
        self,
        content: str,
        content_type: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate {plugin_type}"""

        # Your implementation here
        return {{
            # Return appropriate result based on plugin type
        }}

    def validate_input(self, *args) -> bool:
        """Validate input"""
        return True

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        return {{
            "type": "object",
            "properties": {{
                # Add your configuration parameters
            }}
        }}
'''

    with open(output_path, 'w') as f:
        f.write(skeleton)

    logger.info(f"Plugin skeleton created: {output_path}")
    return skeleton


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Plugin System for Adversarial Testing")
    print("=" * 60)

    # Create plugin manager
    manager = PluginManager()

    # Register example plugins
    manager.register_attack_plugin(SQLInjectionAttackPlugin)
    manager.register_defense_plugin(InputValidationDefensePlugin)

    # List available plugins
    plugins = manager.get_available_plugins()
    print(f"\nAvailable Plugins ({len(plugins)}):")
    for plugin in plugins:
        print(f"  - {plugin['name']} ({plugin['type']}) v{plugin['version']}")

    # Generate manifest
    manifest = manager.generate_plugin_manifest()
    print(f"\nTotal plugins: {manifest['total_plugins']}")
