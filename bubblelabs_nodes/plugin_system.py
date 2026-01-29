"""
Plugin System for OpenEvolve Gauntlet System

Provides a flexible plugin architecture that allows extending the Gauntlet
system with custom evaluators, teams, validators, and other components.

Key Features:
- Plugin lifecycle management (load, initialize, execute, cleanup)
- Plugin sandboxing for security
- Plugin validation and verification
- Plugin discovery and registration
- Plugin dependencies and versioning
"""

from typing import Dict, List, Any, Optional, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import logging
import importlib
import sys
import inspect
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class PluginMetadata:
    """Metadata for a plugin"""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        author: str,
        dependencies: List[str] = None,
        python_version: str = "3.8+",
        gauntlet_version: str = "1.0.0+"
    ):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.python_version = python_version
        self.gauntlet_version = gauntlet_version


class PluginContext:
    """Context provided to plugins during execution"""

    def __init__(
        self,
        plugin_name: str,
        config: Dict[str, Any],
        shared_state: Dict[str, Any] = None
    ):
        self.plugin_name = plugin_name
        self.config = config
        self.shared_state = shared_state or {}
        self.logger = logging.getLogger(f"plugin.{plugin_name}")


class PluginResult:
    """Result from plugin execution"""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}


# Plugin interfaces (protocols)
class CustomEvaluator(Protocol):
    """Interface for custom evaluator plugins"""

    async def evaluate(self, solution: Any, context: PluginContext) -> PluginResult:
        """Evaluate a solution and return result"""
        ...

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        ...

    def validate_input(self, solution: Any) -> bool:
        """Validate solution before evaluation"""
        ...


class CustomTeam(Protocol):
    """Interface for custom team plugins"""

    async def execute(self, problem: Dict[str, Any], context: PluginContext) -> PluginResult:
        """Execute problem and return solution"""
        ...

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        ...

    def validate_input(self, problem: Dict[str, Any]) -> bool:
        """Validate problem before execution"""
        ...


class CustomValidator(Protocol):
    """Interface for custom validator plugins"""

    async def validate(self, solution: Any, criteria: Dict[str, Any], context: PluginContext) -> PluginResult:
        """Validate solution against criteria"""
        ...

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        ...


class Plugin:
    """Base plugin class that all plugins must inherit from"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = True
        self.metadata = None

    def initialize(self, context: PluginContext) -> bool:
        """
        Initialize the plugin.

        Args:
            context: Plugin context

        Returns:
            True if initialization successful
        """
        return True

    async def execute(self, **kwargs) -> PluginResult:
        """
        Execute the plugin logic.

        Args:
            **kwargs: Plugin-specific arguments

        Returns:
            PluginResult
        """
        return PluginResult(success=True)

    def cleanup(self, context: PluginContext) -> bool:
        """
        Cleanup plugin resources.

        Args:
            context: Plugin context

        Returns:
            True if cleanup successful
        """
        return True

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return self.metadata

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        return (True, [])


class PluginValidator:
    """Validates plugins before loading"""

    def __init__(self):
        self.allowed_modules = set()
        self.blocked_modules = {
            'os', 'subprocess', 'shutil', 'sys', 'importlib',
            'eval', 'exec', 'compile', '__import__'
        }

    def validate_plugin(self, plugin_class: type) -> tuple[bool, List[str]]:
        """
        Validate a plugin class.

        Args:
            plugin_class: Plugin class to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check if plugin inherits from Plugin
        if not issubclass(plugin_class, Plugin):
            errors.append("Plugin must inherit from Plugin base class")

        # Check for required methods
        required_methods = ['initialize', 'execute', 'cleanup']
        for method_name in required_methods:
            if not hasattr(plugin_class, method_name):
                errors.append(f"Missing required method: {method_name}")

        # Check for metadata
        if not hasattr(plugin_class, 'get_metadata'):
            errors.append("Missing get_metadata method")

        return (len(errors) == 0, errors)

    def validate_code(self, code: str) -> tuple[bool, List[str]]:
        """
        Validate plugin code for security issues.

        Args:
            code: Plugin code to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check for blocked modules
        for blocked in self.blocked_modules:
            if f"import {blocked}" in code:
                errors.append(f"Blocked module import: {blocked}")

        # Check for dangerous functions
        dangerous = ['eval(', 'exec(', 'compile(']
        for danger in dangerous:
            if danger in code:
                errors.append(f"Dangerous function detected: {danger}")

        # Check for file operations
        if 'open(' in code and 'write' in code:
            errors.append("File write operations detected (use plugin API instead)")

        return (len(errors) == 0, errors)


class PluginSandbox:
    """
    Provides sandboxed execution environment for plugins.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.resource_limits = {
            'max_memory': 1024 * 1024 * 1024,  # 1GB
            'max_execution_time': timeout,
            'max_file_size': 1024 * 1024,  # 1MB
        }

    async def execute_plugin(
        self,
        plugin: Plugin,
        context: PluginContext,
        **kwargs
    ) -> PluginResult:
        """
        Execute a plugin in the sandbox.

        Args:
            plugin: Plugin instance
            context: Plugin context
            **kwargs: Additional arguments

        Returns:
            PluginResult
        """
        import asyncio

        try:
            # Initialize plugin
            init_success = plugin.initialize(context)
            if not init_success:
                return PluginResult(
                    success=False,
                    error="Plugin initialization failed"
                )

            # Execute with timeout
            result = await asyncio.wait_for(
                plugin.execute(**kwargs),
                timeout=self.timeout
            )

            # Cleanup
            plugin.cleanup(context)

            return result

        except asyncio.TimeoutError:
            return PluginResult(
                success=False,
                error=f"Plugin execution timed out after {self.timeout}s"
            )

        except Exception as e:
            return PluginResult(
                success=False,
                error=f"Plugin execution failed: {str(e)}"
            )


class PluginRegistry:
    """
    Registry for managing loaded plugins.
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}

    def register_plugin(
        self,
        name: str,
        plugin: Plugin,
        metadata: PluginMetadata
    ) -> bool:
        """
        Register a plugin.

        Args:
            name: Unique plugin name
            plugin: Plugin instance
            metadata: Plugin metadata

        Returns:
            True if registered successfully
        """
        # Validate plugin
        validator = PluginValidator()
        is_valid, errors = validator.validate_plugin(type(plugin))

        if not is_valid:
            logger.error(f"Plugin {name} validation failed: {errors}")
            return False

        self.plugins[name] = plugin
        self.plugin_metadata[name] = metadata
        logger.info(f"Registered plugin: {name} v{metadata.version}")

        return True

    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unregistered successfully
        """
        if name in self.plugins:
            del self.plugins[name]
            del self.plugin_metadata[name]
            logger.info(f"Unregistered plugin: {name}")
            return True

        return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name"""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugin names"""
        return list(self.plugins.keys())

    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata"""
        return self.plugin_metadata.get(name)

    def get_all_metadata(self) -> Dict[str, PluginMetadata]:
        """Get all plugin metadata"""
        return self.plugin_metadata.copy()


class PluginLoader:
    """
    Loads plugins from files and modules.
    """

    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.registry = PluginRegistry()
        self.validator = PluginValidator()

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directory.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        # Look for Python files
        for py_file in self.plugin_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            # Extract plugin name
            plugin_name = py_file.stem
            discovered.append(plugin_name)

        logger.info(f"Discovered {len(discovered)} plugins in {self.plugin_dir}")
        return discovered

    def load_plugin_from_file(
        self,
        name: str,
        filepath: str = None
    ) -> Optional[Plugin]:
        """
        Load a plugin from a Python file.

        Args:
            name: Plugin name
            filepath: Path to plugin file (optional)

        Returns:
            Plugin instance or None if loading failed
        """
        if filepath is None:
            filepath = self.plugin_dir / f"{name}.py"

        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"Plugin file not found: {filepath}")
            return None

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                name.replace('-', '_'),
                str(filepath)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class in module
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if inspect.isclass(item) and issubclass(item, Plugin):
                    plugin_class = item
                    break

            if not plugin_class:
                logger.error(f"No Plugin subclass found in {filepath}")
                return None

            # Instantiate plugin
            plugin = plugin_class()

            # Extract metadata
            metadata = self._extract_metadata(module, plugin)

            # Validate plugin
            is_valid, errors = self.validator.validate_plugin(plugin_class)
            if not is_valid:
                logger.error(f"Plugin validation failed for {name}: {errors}")
                return None

            logger.info(f"Loaded plugin: {name} from {filepath}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin {name} from {filepath}: {e}")
            return None

    def load_plugin_from_dict(
        self,
        name: str,
        plugin_dict: Dict[str, Any]
    ) -> Optional[Plugin]:
        """
        Load a plugin from dictionary configuration.

        Args:
            name: Plugin name
            plugin_dict: Plugin configuration

        Returns:
            Plugin instance or None
        """
        try:
            # Extract plugin class path
            class_path = plugin_dict.get('class_path')
            if not class_path:
                logger.error(f"No class_path specified for plugin {name}")
                return None

            # Import and instantiate
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            # Instantiate with config
            plugin = plugin_class(plugin_dict.get('config'))

            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin {name} from dict: {e}")
            return None

    def _extract_metadata(self, module, plugin) -> PluginMetadata:
        """Extract metadata from plugin module"""
        # Try to get metadata from module
        if hasattr(module, 'PLUGIN_METADATA'):
            metadata_dict = module.PLUGIN_METADATA
            return PluginMetadata(**metadata_dict)

        # Try to get from plugin.get_metadata()
        if hasattr(plugin, 'get_metadata'):
            return plugin.get_metadata()

        # Create default metadata
        return PluginMetadata(
            name=plugin.__class__.__name__,
            version="1.0.0",
            description="Auto-generated metadata",
            author="Unknown"
        )


class PluginManager:
    """
    Main interface for plugin system.
    """

    def __init__(
        self,
        plugin_dir: str = "./plugins",
        sandbox_enabled: bool = True,
        sandbox_timeout: float = 30.0
    ):
        self.loader = PluginLoader(plugin_dir)
        self.registry = PluginRegistry()
        self.sandbox = PluginSandbox(timeout=sandbox_timeout) if sandbox_enabled else None
        self.sandbox_enabled = sandbox_enabled

    async def load_plugin(
        self,
        name: str,
        filepath: str = None
    ) -> bool:
        """
        Load and register a plugin.

        Args:
            name: Plugin name
            filepath: Optional path to plugin file

        Returns:
            True if loaded successfully
        """
        # Discover plugins if name not specified
        if not filepath:
            discovered = self.loader.discover_plugins()
            if name in discovered:
                filepath = self.loader.plugin_dir / f"{name}.py"
            else:
                logger.error(f"Plugin not found: {name}")
                return False

        # Load plugin
        plugin = self.loader.load_plugin_from_file(name, filepath)

        if not plugin:
            return False

        # Get metadata
        metadata = plugin.get_metadata()
        if not metadata:
            logger.error(f"Plugin {name} missing metadata")
            return False

        # Register plugin
        return self.registry.register_plugin(name, plugin, metadata)

    async def execute_plugin(
        self,
        name: str,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> PluginResult:
        """
        Execute a plugin.

        Args:
            name: Plugin name
            context: Execution context
            **kwargs: Plugin-specific arguments

        Returns:
            PluginResult
        """
        plugin = self.registry.get_plugin(name)
        if not plugin:
            return PluginResult(
                success=False,
                error=f"Plugin not found: {name}"
            )

        # Create plugin context
        plugin_context = PluginContext(
            plugin_name=name,
            config=context or {},
            shared_state={}
        )

        # Execute with or without sandbox
        if self.sandbox_enabled:
            result = await self.sandbox.execute_plugin(
                plugin=plugin,
                context=plugin_context,
                **kwargs
            )
        else:
            # Initialize
            init_success = plugin.initialize(plugin_context)
            if not init_success:
                return PluginResult(
                    success=False,
                    error="Plugin initialization failed"
                )

            # Execute
            result = await plugin.execute(**kwargs)

            # Cleanup
            plugin.cleanup(plugin_context)

        return result

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin"""
        return self.registry.unregister_plugin(name)

    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return self.registry.list_plugins()

    def get_plugin_info(self, name: str) -> Dict[str, Any]:
        """Get detailed plugin information"""
        metadata = self.registry.get_plugin_metadata(name)
        plugin = self.registry.get_plugin(name)

        if not metadata or not plugin:
            return {
                'error': f"Plugin {name} not found"
            }

        return {
            'name': metadata.name,
            'version': metadata.version,
            'description': metadata.description,
            'author': metadata.author,
            'dependencies': metadata.dependencies,
            'python_version': metadata.python_version,
            'gauntlet_version': metadata.gauntlet_version,
            'enabled': plugin.enabled,
        }

    def get_all_plugins_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all loaded plugins"""
        return {
            name: self.get_plugin_info(name)
            for name in self.list_plugins()
        }

    async def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.

        Returns:
            Dict mapping plugin name to load success
        """
        discovered = self.loader.discover_plugins()
        results = {}

        for name in discovered:
            success = await self.load_plugin(name)
            results[name] = success

        logger.info(f"Loaded {sum(results.values())}/{len(results)} plugins")
        return results


# Example plugin
class ExampleEvaluatorPlugin(Plugin):
    """Example custom evaluator plugin"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_evaluator",
            version="1.0.0",
            description="Example evaluator for demonstration",
            author="OpenEvolve",
            dependencies=[],
        )

    def initialize(self, context: PluginContext) -> bool:
        """Initialize the evaluator"""
        context.logger.info("Example evaluator initialized")
        return True

    async def execute(self, solution: Any, **kwargs) -> PluginResult:
        """Evaluate a solution"""
        # Simple example: check if solution is not empty
        if solution is None or (isinstance(solution, (list, dict)) and len(solution) == 0):
            return PluginResult(
                success=False,
                error="Solution is empty"
            )

        # Check if solution has expected structure
        if isinstance(solution, dict):
            required_keys = ['result', 'confidence']
            missing = [k for k in required_keys if k not in solution]
            if missing:
                return PluginResult(
                    success=False,
                    error=f"Missing keys: {', '.join(missing)}"
                )

        return PluginResult(
            success=True,
            data={'evaluated': True, 'score': 0.85},
            execution_time=0.1
        )

    def cleanup(self, context: PluginContext) -> bool:
        """Cleanup evaluator resources"""
        context.logger.info("Example evaluator cleaned up")
        return True


# Convenience functions
def create_plugin_manager(
    plugin_dir: str = "./plugins",
    sandbox_enabled: bool = True
) -> PluginManager:
    """Create a plugin manager"""
    return PluginManager(plugin_dir=plugin_dir, sandbox_enabled=sandbox_enabled)


# Example usage
async def demo_plugin_system():
    """Demonstration of plugin system"""

    # Create plugin manager
    manager = create_plugin_manager()

    # Create example plugin file
    plugin_dir = Path("./plugins")
    plugin_dir.mkdir(exist_ok=True)

    plugin_code = '''
from bubblelabs_nodes.plugin_system import Plugin, PluginMetadata

class ExampleEvaluatorPlugin(Plugin):
    def get_metadata(self):
        return PluginMetadata(
            name="example_evaluator",
            version="1.0.0",
            description="Example evaluator",
            author="Demo"
        )

    def initialize(self, context):
        return True

    async def execute(self, solution, **kwargs):
        return PluginResult(success=True, data={"score": 0.8})

    def cleanup(self, context):
        return True
'''

    # Write plugin file
    with open(plugin_dir / "example_evaluator.py", 'w') as f:
        f.write(plugin_code)

    print("\n" + "=" * 60)
    print("Plugin System Demo")
    print("=" * 60)

    # Load all plugins
    results = await manager.load_all_plugins()
    print(f"\nLoaded plugins: {list(results.keys())}")

    # Execute plugin
    result = await manager.execute_plugin(
        "example_evaluator",
        context={},
        solution={"result": "test", "confidence": 0.9}
    )

    print(f"\nPlugin execution:")
    print(f"  Success: {result.success}")
    print(f"  Data: {result.data}")
    print(f"  Time: {result.execution_time}s")

    # List all plugins with info
    print(f"\nAll plugins:")
    all_info = manager.get_all_plugins_info()
    for name, info in all_info.items():
        print(f"\n  {name}:")
        print(f"    Version: {info.get('version', 'unknown')}")
        print(f"    Description: {info.get('description', 'no description')}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_plugin_system())
