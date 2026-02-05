"""
Optional Imports Utility for Knowledge Engine

Provides standardized handling for optional dependencies with:
- Clear error messages when dependencies are missing
- Graceful degradation with feature flags
- Consistent patterns across all integrations
"""

import logging
import warnings
from typing import Any, Optional, Callable, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""
    
    def __init__(self, package_name: str, feature_name: str, install_command: str):
        self.package_name = package_name
        self.feature_name = feature_name
        self.install_command = install_command
        super().__init__(
            f"'{package_name}' is required for {feature_name}. "
            f"Install with: {install_command}"
        )


class OptionalImportManager:
    """Manages optional imports with proper error handling."""
    
    def __init__(self):
        self._availability_cache: dict[str, bool] = {}
        self._module_cache: dict[str, Any] = {}
    
    def import_optional(
        self,
        module_name: str,
        package_name: str,
        feature_name: str,
        install_command: str,
        fail_silently: bool = False
    ) -> Optional[Any]:
        """
        Import an optional module with standardized error handling.
        
        Args:
            module_name: The module to import (e.g., 'sentence_transformers')
            package_name: The package name for error messages (e.g., 'sentence-transformers')
            feature_name: The feature that requires this dependency
            install_command: The pip install command
            fail_silently: If True, return None on failure instead of warning
            
        Returns:
            The imported module or None if not available
        """
        # Check cache
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        
        if module_name in self._availability_cache:
            if not self._availability_cache[module_name]:
                if not fail_silently:
                    warnings.warn(
                        f"{package_name} not available. {feature_name} will be disabled. "
                        f"Install with: {install_command}",
                        RuntimeWarning,
                        stacklevel=3
                    )
                return None
        
        try:
            module = __import__(module_name, fromlist=[''])
            self._module_cache[module_name] = module
            self._availability_cache[module_name] = True
            logger.debug(f"Successfully imported optional module: {module_name}")
            return module
            
        except ImportError:
            self._availability_cache[module_name] = False
            
            if fail_silently:
                return None
            
            warnings.warn(
                f"{package_name} not available. {feature_name} will be disabled. "
                f"Install with: {install_command}",
                RuntimeWarning,
                stacklevel=3
            )
            return None
    
    def require_dependency(
        self,
        module_name: str,
        package_name: str,
        feature_name: str,
        install_command: str
    ) -> Any:
        """
        Import a module and raise an error if it's not available.
        
        Args:
            module_name: The module to import
            package_name: The package name for error messages
            feature_name: The feature that requires this dependency
            install_command: The pip install command
            
        Returns:
            The imported module
            
        Raises:
            OptionalDependencyError: If the module is not available
        """
        module = self.import_optional(
            module_name, package_name, feature_name, install_command, fail_silently=True
        )
        
        if module is None:
            raise OptionalDependencyError(package_name, feature_name, install_command)
        
        return module
    
    def is_available(self, module_name: str) -> bool:
        """Check if an optional module is available."""
        if module_name not in self._availability_cache:
            # Try a quick import
            try:
                __import__(module_name)
                self._availability_cache[module_name] = True
            except ImportError:
                self._availability_cache[module_name] = False
        
        return self._availability_cache[module_name]


# Global instance
_import_manager = OptionalImportManager()


def import_optional(
    module_name: str,
    package_name: str,
    feature_name: str,
    install_command: str,
    fail_silently: bool = False
) -> Optional[Any]:
    """Convenience function for importing optional modules."""
    return _import_manager.import_optional(
        module_name, package_name, feature_name, install_command, fail_silently
    )


def require_dependency(
    module_name: str,
    package_name: str,
    feature_name: str,
    install_command: str
) -> Any:
    """Convenience function for requiring dependencies."""
    return _import_manager.require_dependency(
        module_name, package_name, feature_name, install_command
    )


def is_available(module_name: str) -> bool:
    """Check if an optional module is available."""
    return _import_manager.is_available(module_name)


class FailingMock:
    """
    A mock class that fails loudly when used.
    
    This is used as a base class for mock implementations that should
    not be used in production. When any method is called, it raises
    an informative error.
    """
    
    _package_name: str = ""
    _feature_name: str = ""
    _install_command: str = ""
    
    def __init__(self, *args, **kwargs):
        raise OptionalDependencyError(
            self._package_name,
            self._feature_name,
            self._install_command
        )
    
    def __getattr__(self, name: str) -> Any:
        raise OptionalDependencyError(
            self._package_name,
            self._feature_name,
            self._install_command
        )


def create_failing_mock(
    package_name: str,
    feature_name: str,
    install_command: str
) -> type:
    """
    Create a failing mock class.
    
    Args:
        package_name: The package that's missing
        feature_name: The feature that requires it
        install_command: How to install the package
        
    Returns:
        A class that raises an error when instantiated
    """
    return type(
        f'FailingMock_{package_name.replace("-", "_")}',
        (FailingMock,),
        {
            '_package_name': package_name,
            '_feature_name': feature_name,
            '_install_command': install_command
        }
    )


# Common optional dependencies
OPTIONAL_DEPENDENCIES = {
    'sentence_transformers': {
        'package': 'sentence-transformers',
        'feature': 'real embedding generation',
        'install': 'pip install sentence-transformers'
    },
    'psutil': {
        'package': 'psutil',
        'feature': 'system performance monitoring',
        'install': 'pip install psutil'
    },
    'boto3': {
        'package': 'boto3',
        'feature': 'AWS S3 storage',
        'install': 'pip install boto3'
    },
    'google.cloud.storage': {
        'package': 'google-cloud-storage',
        'feature': 'Google Cloud Storage',
        'install': 'pip install google-cloud-storage'
    },
    'azure.storage.blob': {
        'package': 'azure-storage-blob',
        'feature': 'Azure Blob Storage',
        'install': 'pip install azure-storage-blob'
    },
    'qdrant_client': {
        'package': 'qdrant-client',
        'feature': 'Qdrant vector database',
        'install': 'pip install qdrant-client'
    },
    'asyncpg': {
        'package': 'asyncpg',
        'feature': 'PostgreSQL async support',
        'install': 'pip install asyncpg'
    },
    'torch': {
        'package': 'torch',
        'feature': 'neural network operations',
        'install': 'pip install torch'
    },
    'networkx': {
        'package': 'networkx',
        'feature': 'graph analysis',
        'install': 'pip install networkx'
    },
    'sklearn': {
        'package': 'scikit-learn',
        'feature': 'machine learning utilities',
        'install': 'pip install scikit-learn'
    },
    'z3': {
        'package': 'z3-solver',
        'feature': 'theorem proving',
        'install': 'pip install z3-solver'
    },
    'memgraph': {
        'package': 'gqlalchemy',
        'feature': 'Memgraph graph database',
        'install': 'pip install gqlalchemy'
    }
}

# NOTE: Non-permissive licenses (BLOCKED)
# These packages are NOT included due to license restrictions:
# - Neo4j (GPL v3) - Use Memgraph (Apache 2.0) instead
# - MongoDB (SSPL) - Use PostgreSQL (PostgreSQL License) instead
# - Elasticsearch (SSPL) - Use OpenSearch (Apache 2.0) instead


def check_all_optional_dependencies() -> dict[str, bool]:
    """Check the availability of all optional dependencies."""
    results = {}
    for module_name, info in OPTIONAL_DEPENDENCIES.items():
        available = is_available(module_name.replace('.', '_'))
        results[info['package']] = available
        status = "[OK]" if available else "[FAIL]"
        print(f"{status} {info['package']:30s} - {info['feature']}")
    return results


__all__ = [
    'OptionalDependencyError',
    'OptionalImportManager',
    'FailingMock',
    'create_failing_mock',
    'import_optional',
    'require_dependency',
    'is_available',
    'check_all_optional_dependencies',
    'OPTIONAL_DEPENDENCIES'
]
