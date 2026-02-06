"""
Integration Registry for OpenEvolve

This module provides a dynamic registry for loading and managing integrations.
It implements the factory pattern for creating integration instances and
handles graceful degradation when integrations are unavailable.

Author: Agent 8 (Integration Orchestrator)
Created: 2026-01-02
Status: [OK] Complete
"""

import os
import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from integrations.base.knowledge_interface import KnowledgeGraphInterface
from integrations.base.extraction_interface import ExtractionInterface
from integrations.base.experimentation_interface import ExperimentationInterface
from integrations.base.optimization_interface import OptimizationInterface
from integrations.base.uq_interface import UncertaintyQuantificationInterface
from integrations.base.visualization_interface import VisualizationInterface
from integrations.base.domain_knowledge_interface import DomainKnowledgeInterface
from integrations.base.causal_interface import CausalDiscoveryInterface


logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of integrations available."""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    EXTRACTION = "extraction"
    EXPERIMENTATION = "experimentation"
    OPTIMIZATION = "optimization"
    UQ = "uncertainty_quantification"
    VISUALIZATION = "visualization"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CAUSAL_DISCOVERY = "causal_discovery"


class IntegrationStatus(Enum):
    """Status of an integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class IntegrationInfo:
    """
    Information about an integration.

    Attributes:
        name: Integration name
        type: Integration type
        module_path: Python module path
        class_name: Adapter class name
        config_path: Path to configuration file
        status: Current status
        dependencies: Required dependencies
        version: Integration version
        description: Human-readable description
        interface: Base interface class
        grpc_target: Optional gRPC target (host:port) for health checks
    """
    name: str
    type: IntegrationType
    module_path: str
    class_name: str
    config_path: Optional[str] = None
    status: IntegrationStatus = IntegrationStatus.UNAVAILABLE
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    interface: Optional[Type] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    grpc_target: Optional[str] = None


class IntegrationRegistry:
    """
    Registry for managing OpenEvolve integrations.

    This class provides:
    - Dynamic integration discovery and loading
    - Factory pattern for creating integration instances
    - Graceful degradation when integrations are unavailable
    - Health monitoring for all integrations
    - Dependency validation
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the integration registry.

        Args:
            config_dir: Directory containing integration configuration files
        """
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), "configs"
        )
        self._integrations: Dict[str, IntegrationInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

        # Register built-in integrations
        self._register_builtin_integrations()

    async def check_health(self, name: str) -> Dict[str, Any]:
        """
        Check the health of an integration via gRPC or fallback.
        
        Args:
            name: Integration name
            
        Returns:
            Dictionary with health status and metrics
        """
        integration = self._integrations.get(name)
        if not integration:
            return {"status": "unknown", "error": "Integration not found"}

        # Dependency check first
        deps = await self.check_dependencies(name)
        if not all(deps.values()):
             return {
                 "status": "unhealthy",
                 "details": "Missing dependencies",
                 "dependencies": deps
             }

        # gRPC Health Check
        if integration.grpc_target:
            try:
                import grpc
                # Simple connectivity check
                channel = grpc.aio.insecure_channel(integration.grpc_target)
                try:
                    # Try to connect with a short timeout
                    await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
                    return {"status": "healthy", "method": "grpc", "target": integration.grpc_target}
                except asyncio.TimeoutError:
                    return {"status": "unhealthy", "error": "gRPC connection timeout", "target": integration.grpc_target}
                except Exception as e:
                     return {"status": "unhealthy", "error": str(e), "target": integration.grpc_target}
                finally:
                    await channel.close()
            except ImportError:
                 logger.warning("grpc module not found, skipping gRPC health check")
                 return {"status": "unknown", "error": "grpc module missing"}
        
        # Fallback to instance check if loaded
        if name in self._instances:
             instance = self._instances[name]
             if hasattr(instance, 'check_health'):
                 try:
                     return await instance.check_health()
                 except Exception as e:
                     return {"status": "unhealthy", "error": str(e)}
        
        return {"status": "healthy", "method": "dependency_check_only"}

    def _register_builtin_integrations(self):
        """Register built-in integrations."""
        # Use relative paths from the integrations directory
        integrations_dir = os.path.dirname(__file__)
        
        builtin_integrations = [
            # Graphiti - Temporal Knowledge Graph
            IntegrationInfo(
                name="graphiti",
                type=IntegrationType.KNOWLEDGE_GRAPH,
                module_path="integrations.graphiti.adapter",
                class_name="GraphitiAdapter",
                config_path=os.path.join(integrations_dir, "graphiti", "config.yaml"),
                dependencies=["graphiti"],
                version="1.0.0",
                description="Temporal knowledge graph with Neo4j/FalkorDB backend",
                interface=KnowledgeGraphInterface,
                metadata={
                    "priority": "P1",
                    "gaps_filled": ["GAP-14", "GAP-10"],
                    "features": ["temporal_metadata", "hybrid_search", "graph_traversal"]
                }
            ),

            # OneKE - Schema-Guided Extraction
            IntegrationInfo(
                name="oneke",
                type=IntegrationType.EXTRACTION,
                module_path="integrations.oneke.adapter",
                class_name="OneKEAdapter",
                config_path=os.path.join(integrations_dir, "oneke", "config.yaml"),
                dependencies=["oneke"],
                version="1.0.0",
                description="Schema-guided knowledge extraction with multi-agent workflows",
                interface=ExtractionInterface,
                metadata={
                    "priority": "P2",
                    "gaps_filled": ["GAP-2", "GAP-10"],
                    "features": ["ner", "re", "ee", "triple_extraction"]
                }
            ),

            # Curie - Scientific Experimentation
            IntegrationInfo(
                name="curie",
                type=IntegrationType.EXPERIMENTATION,
                module_path="integrations.curie.adapter",
                class_name="CurieAdapter",
                config_path=os.path.join(integrations_dir, "curie", "config.yaml"),
                dependencies=["curie"],
                version="1.0.0",
                description="Automated scientific experimentation and protocol design",
                interface=ExperimentationInterface,
                metadata={
                    "priority": "P1.5",
                    "gaps_filled": ["GAP-4", "GAP-12"],
                    "features": ["hypothesis_testing", "protocol_design", "statistical_validation"]
                }
            ),

            # NeuroMANCER - Physics-Informed Optimization
            IntegrationInfo(
                name="neuromancer",
                type=IntegrationType.OPTIMIZATION,
                module_path="integrations.neuromancer.adapter",
                class_name="NeuroMANCERAdapter",
                config_path=os.path.join(integrations_dir, "neuromancer", "config.yaml"),
                dependencies=["neuromancer", "torch"],
                version="1.0.0",
                description="Physics-informed optimization and system identification",
                interface=OptimizationInterface,
                metadata={
                    "priority": "P3",
                    "gaps_filled": ["GAP-3", "GAP-1"],
                    "features": ["physics_informed", "system_identification", "constrained_optimization"],
                    "isolation": "conda"  # Requires separate PyTorch environment
                }
            ),

            # pygraphistry - Graph Visualization
            IntegrationInfo(
                name="pygraphistry",
                type=IntegrationType.VISUALIZATION,
                module_path="integrations.pygraphistry.adapter",
                class_name="PygraphistryAdapter",
                config_path=os.path.join(integrations_dir, "pygraphistry", "config.yaml"),
                dependencies=["graphistry"],
                version="1.0.0",
                description="Interactive graph visualization with GPU-accelerated ML",
                interface=VisualizationInterface,
                metadata={
                    "priority": "P2",
                    "gaps_filled": ["GAP-7", "GAP-10", "GAP-11"],
                    "features": ["gpu_acceleration", "umap", "dbscan", "interactive_dashboards"]
                }
            ),

            # uqtestfuns - Uncertainty Quantification
            IntegrationInfo(
                name="uqtestfuns",
                type=IntegrationType.UQ,
                module_path="integrations.uqtestfuns.adapter",
                class_name="UQTestFunsAdapter",
                config_path=os.path.join(integrations_dir, "uqtestfuns", "config.yaml"),
                dependencies=["uqtestfuns"],
                version="1.0.0",
                description="Uncertainty quantification test functions library",
                interface=UncertaintyQuantificationInterface,
                metadata={
                    "priority": "P3",
                    "gaps_filled": ["GAP-15"],
                    "features": ["probabilistic_inputs", "sensitivity_analysis", "validation_pipeline"]
                }
            ),

            # global-chem - Chemical Knowledge
            IntegrationInfo(
                name="global_chem",
                type=IntegrationType.DOMAIN_KNOWLEDGE,
                module_path="integrations.global_chem.adapter",
                class_name="GlobalChemAdapter",
                config_path=os.path.join(integrations_dir, "global_chem", "config.yaml"),
                dependencies=["global-chem"],
                version="1.0.0",
                description="Chemical knowledge graphs and SMILES/SMARTS support",
                interface=DomainKnowledgeInterface,
                metadata={
                    "priority": "P4",
                    "gaps_filled": ["GAP-13", "GAP-2"],
                    "features": ["smiles", "smarts", "chemical_properties", "domain_knowledge"]
                }
            ),

            # causal-learn - Causal Discovery
            IntegrationInfo(
                name="causal_learn",
                type=IntegrationType.CAUSAL_DISCOVERY,
                module_path="integrations.causal_learn.adapter",
                class_name="CausalLearnAdapter",
                config_path=os.path.join(integrations_dir, "causal_learn", "config.yaml"),
                dependencies=["causallearn"],
                version="1.0.0",
                description="Causal structure discovery and reasoning",
                interface=CausalDiscoveryInterface,
                metadata={
                    "priority": "P2",
                    "features": ["pc", "ges", "lingam", "fci", "independence_tests"]
                }
            ),
        ]

        for integration in builtin_integrations:
            self._integrations[integration.name] = integration

    def register_integration(self, integration: IntegrationInfo) -> None:
        """
        Register a custom integration.

        Args:
            integration: IntegrationInfo object describing the integration
        """
        self._integrations[integration.name] = integration
        logger.info(f"Registered integration: {integration.name}")

    def unregister_integration(self, name: str) -> None:
        """
        Unregister an integration.

        Args:
            name: Integration name
        """
        if name in self._integrations:
            del self._integrations[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Unregistered integration: {name}")

    def list_integrations(
        self,
        status_filter: Optional[IntegrationStatus] = None,
        type_filter: Optional[IntegrationType] = None
    ) -> List[IntegrationInfo]:
        """
        List all registered integrations.

        Args:
            status_filter: Optional status filter
            type_filter: Optional type filter

        Returns:
            List of IntegrationInfo objects
        """
        integrations = list(self._integrations.values())

        if status_filter:
            integrations = [i for i in integrations if i.status == status_filter]

        if type_filter:
            integrations = [i for i in integrations if i.type == type_filter]

        return integrations

    def get_integration_info(self, name: str) -> Optional[IntegrationInfo]:
        """
        Get information about an integration.

        Args:
            name: Integration name

        Returns:
            IntegrationInfo object or None if not found
        """
        return self._integrations.get(name)

    async def check_dependencies(self, name: str) -> Dict[str, bool]:
        """
        Check if integration dependencies are available.

        Args:
            name: Integration name

        Returns:
            Dictionary mapping dependency names to availability status
        """
        integration = self._integrations.get(name)
        if not integration:
            return {}

        dependency_status = {}
        integrations_root = os.path.dirname(os.path.dirname(__file__))
        
        for dep in integration.dependencies:
            try:
                importlib.import_module(dep)
                dependency_status[dep] = True
            except ImportError:
                # Check for common local module structures
                possible_paths = [
                    os.path.join(integrations_root, dep),
                    os.path.join(integrations_root, f"py{dep}"), # e.g. pygraphistry for graphistry
                    os.path.join(integrations_root, name),
                ]
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        if path not in sys.path:
                            sys.path.insert(0, path)
                        try:
                            importlib.import_module(dep)
                            dependency_status[dep] = True
                            found = True
                            break
                        except ImportError:
                            pass
                
                if not found:
                    dependency_status[dep] = False
                    logger.warning(f"Dependency {dep} for {name} is not available")

        return dependency_status

    async def is_available(self, name: str) -> bool:
        """
        Check if an integration is available for use.

        Args:
            name: Integration name

        Returns:
            True if available, False otherwise
        """
        integration = self._integrations.get(name)
        if not integration:
            return False

        if integration.status == IntegrationStatus.DISABLED:
            return False

        dependency_status = await self.check_dependencies(name)
        return all(dependency_status.values())

    async def load_integration(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        force_reload: bool = False
    ) -> Optional[Any]:
        """
        Load an integration adapter instance.

        Args:
            name: Integration name
            config: Optional configuration override
            force_reload: Force reload even if already loaded

        Returns:
            Integration instance or None if unavailable

        Raises:
            ImportError: If integration module cannot be imported
            InstantiationError: If adapter class cannot be instantiated
        """
        # Check if already loaded
        if not force_reload and name in self._instances:
            return self._instances[name]

        # Get integration info
        integration = self._integrations.get(name)
        if not integration:
            logger.error(f"Integration {name} not registered")
            return None

        # Check dependencies
        if not await self.is_available(name):
            integration.status = IntegrationStatus.UNAVAILABLE
            logger.warning(f"Integration {name} is unavailable (missing dependencies)")
            return None

        try:
            # Import module
            module = importlib.import_module(integration.module_path)

            # Get adapter class
            adapter_class = getattr(module, integration.class_name)

            # Load configuration
            if config is None and integration.config_path:
                config = self._load_config(integration.config_path)
            
            # Normalize config (camelCase to snake_case)
            if config:
                config = self._normalize_config(config)

            # Create instance
            # Try parameterless instantiation first (Standard Pattern)
            try:
                instance = adapter_class()
            except TypeError:
                # Fallback to passing config if parameterless fails
                instance = adapter_class(config or {})

            # Initialize
            if hasattr(instance, 'initialize'):
                init_success = await instance.initialize(config or {})
                if not init_success:
                    integration.status = IntegrationStatus.ERROR
                    logger.error(f"Failed to initialize integration {name}")
                    return None

            # Store instance
            self._instances[name] = instance
            integration.status = IntegrationStatus.AVAILABLE
            logger.info(f"Successfully loaded integration: {name}")

            return instance

        except ImportError as e:
            integration.status = IntegrationStatus.ERROR
            logger.error(f"Failed to import integration {name}: {e}")
            raise
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            logger.error(f"Failed to load integration {name}: {e}")
            raise

    async def get_instance(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get or create an integration instance.

        This is the main factory method for obtaining integration instances.
        Implements graceful degradation by returning None if unavailable.

        Args:
            name: Integration name
            config: Optional configuration override

        Returns:
            Integration instance or None
        """
        try:
            return await self.load_integration(name, config)
        except Exception as e:
            logger.error(f"Error getting instance for {name}: {e}")
            return None

    async def shutdown_integration(self, name: str) -> bool:
        """
        Shutdown an integration instance.

        Args:
            name: Integration name

        Returns:
            True if shutdown successful
        """
        if name not in self._instances:
            return True

        try:
            instance = self._instances[name]
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()

            del self._instances[name]
            logger.info(f"Shutdown integration: {name}")
            return True

        except Exception as e:
            logger.error(f"Error shutting down {name}: {e}")
            return False

    async def shutdown_all(self) -> Dict[str, bool]:
        """
        Shutdown all active integrations.

        Returns:
            Dictionary mapping integration names to shutdown success status
        """
        shutdown_status = {}

        for name in list(self._instances.keys()):
            shutdown_status[name] = await self.shutdown_integration(name)

        return shutdown_status

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        from integrations.config_loader import ConfigLoader

        loader = ConfigLoader()
        return loader.load(config_path)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration by converting camelCase keys to snake_case.

        Args:
            config: Configuration dictionary

        Returns:
            Normalized configuration dictionary
        """
        if not isinstance(config, dict):
            return config

        normalized = {}
        for key, value in config.items():
            # Convert camelCase to snake_case
            snake_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
            
            # Recurse for nested dictionaries
            if isinstance(value, dict):
                normalized[snake_key] = self._normalize_config(value)
            elif isinstance(value, list):
                normalized[snake_key] = [
                    self._normalize_config(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            else:
                normalized[snake_key] = value
        
        return normalized

    async def validate_all_integrations(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all registered integrations.

        Returns:
            Dictionary mapping integration names to validation results
        """
        validation_results = {}

        for name, integration in self._integrations.items():
            try:
                # Load integration
                instance = await self.get_instance(name)

                if instance is None:
                    validation_results[name] = {
                        "valid": False,
                        "status": integration.status.value,
                        "message": "Integration unavailable"
                    }
                    continue

                # Validate
                if hasattr(instance, 'validate'):
                    validation_result = await instance.validate()
                    validation_results[name] = validation_result
                else:
                    validation_results[name] = {
                        "valid": True,
                        "message": "No validate method available"
                    }

            except Exception as e:
                validation_results[name] = {
                    "valid": False,
                    "error": str(e)
                }

        return validation_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary containing registry statistics
        """
        stats = {
            "total_integrations": len(self._integrations),
            "active_instances": len(self._instances),
            "by_status": {},
            "by_type": {},
        }

        # Count by status
        for integration in self._integrations.values():
            status = integration.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Count by type
        for integration in self._integrations.values():
            int_type = integration.type.value
            stats["by_type"][int_type] = stats["by_type"].get(int_type, 0) + 1

        return stats


# Global registry instance
_global_registry: Optional[IntegrationRegistry] = None


def get_registry(config_dir: Optional[str] = None) -> IntegrationRegistry:
    """
    Get the global integration registry instance.

    Args:
        config_dir: Optional configuration directory

    Returns:
        IntegrationRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = IntegrationRegistry(config_dir)

    return _global_registry
