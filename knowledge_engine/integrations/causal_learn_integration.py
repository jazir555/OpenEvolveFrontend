"""
Causal-Learn Integration Module for OpenEvolve Knowledge Engine

This module provides causal discovery capabilities by integrating with the
primary causal-learn integration (integrations.causal_learn).

SSOT (Single Source of Truth): integrations/causal_learn/
- adapter.py: CausalLearnAdapter - Main adapter implementing CausalDiscoveryInterface
- bridge.py: CausalDiscoveryBridge - Bridge to OpenEvolve systems
- config.yaml: Configuration for causal-learn integration

This file is a thin wrapper/re-export for knowledge_engine-specific usage.
It provides backward-compatible APIs while delegating to the main implementation.

Business Logic:
    - Delegate all causal discovery operations to CausalLearnAdapter
    - Provide simplified API for knowledge engine unified hub
    - Maintain backward compatibility with existing code

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timezone

# Add integrations to path for imports
_integrations_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "integrations"
)
if _integrations_path not in sys.path:
    sys.path.insert(0, _integrations_path)

# Import from SSOT (Single Source of Truth)
try:
    from integrations.causal_learn.adapter import CausalLearnAdapter
    from integrations.causal_learn.bridge import CausalDiscoveryBridge
    from integrations.causal_learn import (
        CAUSAL_LEARN_AVAILABLE,
        get_adapter,
        get_bridge,
        validate_installation
    )
    from integrations.base.causal_interface import (
        CausalGraphResult,
        CausalEffectResult,
        CausalMethod,
        IndependenceTest,
        EdgeType,
    )
    SSOT_AVAILABLE = True
except ImportError as e:
    SSOT_AVAILABLE = False
    SSOT_IMPORT_ERROR = str(e)
    # Define fallback constants
    CAUSAL_LEARN_AVAILABLE = False

# Third-party imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__all__ = [
    # Main classes
    'CausalLearnIntegration',
    'CausalDiscoveryEngine',
    # SSOT re-exports
    'CausalLearnAdapter',
    'CausalDiscoveryBridge',
    # Constants
    'CAUSAL_LEARN_AVAILABLE',
    'SSOT_AVAILABLE',
    # Functions
    'get_adapter',
    'get_bridge',
    'validate_installation',
]


class CausalLearnIntegration:
    """
    Main Causal-Learn Integration class for the Knowledge Engine.
    
    This is a thin wrapper around the primary CausalLearnAdapter from
    integrations.causal_learn, providing a simplified API for the
    knowledge engine unified hub.
    
    SSOT: integrations/causal_learn/adapter.py -> CausalLearnAdapter
    
    Business Capabilities:
        1. Discover causal structure from observational data
        2. Support multiple algorithms (PC, GES, FCI, LiNGAM)
        3. Independence testing
        4. Graceful degradation when causal-learn unavailable
    
    Example:
        >>> from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
        >>> 
        >>> # Initialize
        >>> causal = CausalLearnIntegration()
        >>> 
        >>> # Check availability
        >>> if causal.is_available():
        ...     result = causal.discover_structure(data, algorithm='pc')
        ...     print(result['graph']['edges'])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Causal-Learn Integration.
        
        Args:
            config: Configuration dictionary for causal discovery
                   See CausalLearnAdapter for full configuration options
        """
        self.config = config or {}
        self._adapter: Optional[CausalLearnAdapter] = None
        self._bridge: Optional[CausalDiscoveryBridge] = None
        
        if SSOT_AVAILABLE:
            try:
                self._adapter = CausalLearnAdapter()
                self._bridge = CausalDiscoveryBridge()
            except Exception as e:
                logger.warning(f"Failed to initialize causal-learn components: {e}")
        else:
            logger.warning(f"SSOT not available: {SSOT_IMPORT_ERROR}")
    
    def is_available(self) -> bool:
        """
        Check if Causal-Learn is available.
        
        Returns:
            True if causal-learn and all components are available
        """
        return SSOT_AVAILABLE and CAUSAL_LEARN_AVAILABLE and self._adapter is not None
    
    async def initialize(self) -> bool:
        """
        Initialize the integration with configuration.
        
        Returns:
            True if initialization successful
        """
        if not self.is_available():
            return False
        
        try:
            # Initialize adapter with config
            await self._adapter.initialize(self.config)
            
            # Initialize bridge
            await self._bridge.initialize()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize causal-learn integration: {e}")
            return False
    
    def discover_structure(
        self,
        data: Union[np.ndarray, List[List[float]]],
        algorithm: str = 'pc',
        variable_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        independence_test: str = 'fisherz',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover causal structure from data.
        
        This method delegates to CausalLearnAdapter.discover_causal_structure
        from the SSOT implementation.
        
        Args:
            data: Data matrix (n_samples x n_features) or list of lists
            algorithm: Algorithm to use ('pc', 'fci', 'ges', 'lingam', etc.)
            variable_names: Names of variables (optional)
            alpha: Significance level for independence tests
            independence_test: Independence test to use ('fisherz', 'chisq', 'gsq', 'kci')
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Dictionary containing causal graph and metadata:
            {
                'status': 'success' | 'error',
                'graph': {
                    'nodes': List[str],
                    'edges': List[Dict],
                    'adjacency_matrix': List[List[int]]
                },
                'algorithm': str,
                'parameters': Dict[str, Any]
            }
            
        Example:
            >>> import numpy as np
            >>> data = np.random.randn(100, 3)
            >>> result = causal.discover_structure(data, algorithm='pc')
            >>> print(f"Found {len(result['graph']['edges'])} causal relationships")
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'Causal-learn integration not available',
                'graph': None
            }
        
        try:
            # Convert data to numpy array if needed
            if NUMPY_AVAILABLE and not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Use adapter from SSOT
            import asyncio
            
            # Check if adapter is initialized
            if not self._adapter.is_initialized:
                asyncio.run(self._adapter.initialize(self.config))
            
            # Run discovery
            result = asyncio.run(
                self._adapter.discover_causal_structure(
                    data=data,
                    method=algorithm,
                    **kwargs
                )
            )
            
            # Convert CausalGraphResult to dictionary format
            return {
                'status': 'success',
                'graph': {
                    'nodes': result.nodes,
                    'edges': [
                        {
                            'source': edge[0],
                            'target': edge[1],
                            'type': edge[2].value if hasattr(edge[2], 'value') else str(edge[2])
                        }
                        for edge in result.edges
                    ],
                    'adjacency_matrix': result.adjacency_matrix.tolist() if hasattr(result.adjacency_matrix, 'tolist') else result.adjacency_matrix,
                    'directed_edges': result.directed_edges,
                    'undirected_edges': result.undirected_edges,
                    'bidirected_edges': result.bidirected_edges,
                    'causal_order': result.causal_order
                },
                'algorithm': result.algorithm_used,
                'parameters': result.method_parameters,
                'timestamp': result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp)
            }
            
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            return {
                'status': 'error',
                'message': f'Causal discovery failed: {str(e)}',
                'graph': None
            }
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get list of available algorithms.
        
        Returns:
            List of algorithm names available in the SSOT
        """
        if not self.is_available():
            return []
        
        return list(self._adapter._algorithms.keys())
    
    def get_available_independence_tests(self) -> List[str]:
        """
        Get list of available independence tests.
        
        Returns:
            List of independence test names
        """
        if not self.is_available():
            return []
        
        return list(self._adapter._indep_tests.keys())
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """
        Get information about a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Dictionary with algorithm information
        """
        if not self.is_available():
            return {'name': algorithm, 'available': False, 'error': 'Causal-learn not available'}
        
        algo_key = algorithm.lower()
        available_algos = self.get_available_algorithms()
        
        # Algorithm metadata from SSOT
        ALGORITHMS = {
            'pc': {
                'description': 'Peter-Clark algorithm for causal discovery',
                'type': 'constraint-based',
                'requires_independence_test': True,
                'handles_latent': False
            },
            'pc_stable': {
                'description': 'Stable PC algorithm',
                'type': 'constraint-based',
                'requires_independence_test': True,
                'handles_latent': False
            },
            'fci': {
                'description': 'Fast Causal Inference for latent variables',
                'type': 'constraint-based',
                'requires_independence_test': True,
                'handles_latent': True
            },
            'ges': {
                'description': 'Greedy Equivalence Search',
                'type': 'score-based',
                'requires_score': True,
                'handles_latent': False
            },
            'direct_lingam': {
                'description': 'Direct LiNGAM algorithm for non-Gaussian data',
                'type': 'functional-causal-model',
                'requires_independence_test': False,
                'handles_latent': False
            },
            'ica_lingam': {
                'description': 'ICA-based LiNGAM',
                'type': 'functional-causal-model',
                'requires_independence_test': False,
                'handles_latent': False
            }
        }
        
        if algo_key in ALGORITHMS:
            return {
                'name': algorithm,
                'available': algo_key in available_algos,
                **ALGORITHMS[algo_key]
            }
        
        return {'name': algorithm, 'available': False, 'error': 'Unknown algorithm'}
    
    def run_independence_test(
        self,
        data: Union[np.ndarray, List[List[float]]],
        x: int,
        y: int,
        conditioning_set: Optional[List[int]] = None,
        test: str = 'fisherz'
    ) -> Dict[str, Any]:
        """
        Run conditional independence test.
        
        Args:
            data: Data matrix
            x: Index of first variable
            y: Index of second variable
            conditioning_set: Indices of conditioning variables
            test: Test method ('fisherz', 'chisq', 'gsq', 'kci')
            
        Returns:
            Dictionary with test results
        """
        if not self.is_available():
            return {'status': 'error', 'message': 'Causal-learn not available'}
        
        try:
            if NUMPY_AVAILABLE and not isinstance(data, np.ndarray):
                data = np.array(data)
            
            import asyncio
            result = asyncio.run(
                self._adapter.test_conditional_independence(
                    data=data,
                    x=x,
                    y=y,
                    conditioning_set=conditioning_set or [],
                    method=test
                )
            )
            
            return {
                'status': 'success',
                'is_independent': result.is_independent,
                'p_value': result.p_value,
                'test_statistic': result.test_statistic,
                'method': result.method,
                'is_significant': result.is_significant
            }
            
        except Exception as e:
            logger.error(f"Independence test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get integration status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'ssot_available': SSOT_AVAILABLE,
            'causal_learn_available': CAUSAL_LEARN_AVAILABLE,
            'adapter_initialized': self._adapter.is_initialized if self._adapter else False,
            'available_algorithms': self.get_available_algorithms(),
            'available_tests': self.get_available_independence_tests(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class CausalDiscoveryEngine:
    """
    Causal discovery engine using causal-learn algorithms.
    
    DEPRECATED: Use CausalLearnIntegration or CausalLearnAdapter directly.
    This class is kept for backward compatibility.
    
    SSOT: integrations/causal_learn/adapter.py -> CausalLearnAdapter
    """
    
    ALGORITHMS = {
        'pc': {
            'description': 'Peter-Clark algorithm for causal discovery',
            'requires_independence_test': True,
            'handles_latent': False
        },
        'fci': {
            'description': 'Fast Causal Inference for latent variables',
            'requires_independence_test': True,
            'handles_latent': True
        },
        'ges': {
            'description': 'Greedy Equivalence Search',
            'requires_score': True,
            'handles_latent': False
        },
        'lingam': {
            'description': 'Linear Non-Gaussian Acyclic Model',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'direct_lingam': {
            'description': 'Direct LiNGAM algorithm',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'ica_lingam': {
            'description': 'ICA-based LiNGAM',
            'requires_independence_test': False,
            'handles_latent': False
        },
        'granger': {
            'description': 'Granger causality for time series',
            'requires_independence_test': False,
            'handles_latent': False,
            'time_series': True
        }
    }
    
    INDEPENDENCE_TESTS = {
        'fisherz': 'Fisher\'s Z conditional independence test',
        'chisq': 'Chi-squared conditional independence test',
        'gsq': 'G-squared conditional independence test',
        'kci': 'Kernel-based conditional independence test'
    }
    
    def __init__(self):
        """Initialize causal-learn modules."""
        logger.warning(
            "CausalDiscoveryEngine is deprecated. "
            "Use CausalLearnIntegration or CausalLearnAdapter directly."
        )
        self._integration = CausalLearnIntegration()
    
    def is_available(self) -> bool:
        """Check if causal-learn integration is available."""
        return self._integration.is_available()
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available causal discovery algorithms."""
        return self._integration.get_available_algorithms()
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        return self._integration.get_algorithm_info(algorithm)
    
    def discover_causal_structure(
        self,
        data: Union[np.ndarray, List[List[float]]],
        variable_names: Optional[List[str]] = None,
        algorithm: str = 'pc',
        alpha: float = 0.05,
        independence_test: str = 'fisherz',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover causal structure from data.
        
        DEPRECATED: Use CausalLearnIntegration.discover_structure()
        """
        return self._integration.discover_structure(
            data=data,
            algorithm=algorithm,
            variable_names=variable_names,
            alpha=alpha,
            independence_test=independence_test,
            **kwargs
        )
    
    def _run_pc(self, data, variable_names, alpha, independence_test, **kwargs):
        """Run PC algorithm - delegates to integration."""
        return self._integration.discover_structure(
            data, algorithm='pc', variable_names=variable_names,
            alpha=alpha, independence_test=independence_test, **kwargs
        )
    
    def _run_fci(self, data, variable_names, alpha, independence_test, **kwargs):
        """Run FCI algorithm - delegates to integration."""
        return self._integration.discover_structure(
            data, algorithm='fci', variable_names=variable_names,
            alpha=alpha, independence_test=independence_test, **kwargs
        )
    
    def _run_ges(self, data, variable_names, **kwargs):
        """Run GES algorithm - delegates to integration."""
        return self._integration.discover_structure(
            data, algorithm='ges', variable_names=variable_names, **kwargs
        )
    
    def _run_ica_lingam(self, data, variable_names, **kwargs):
        """Run ICA-LiNGAM algorithm - delegates to integration."""
        return self._integration.discover_structure(
            data, algorithm='ica_lingam', variable_names=variable_names, **kwargs
        )
    
    def _run_direct_lingam(self, data, variable_names, **kwargs):
        """Run DirectLiNGAM algorithm - delegates to integration."""
        return self._integration.discover_structure(
            data, algorithm='direct_lingam', variable_names=variable_names, **kwargs
        )
    
    def _extract_edges_from_matrix(self, matrix, variable_names):
        """Extract edges from adjacency matrix."""
        edges = []
        if matrix is None or not NUMPY_AVAILABLE:
            return edges
        
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                if matrix[i, j] != 0:
                    edges.append({
                        'source': variable_names[i],
                        'target': variable_names[j],
                        'weight': float(matrix[i, j]),
                        'type': 'directed' if matrix[i, j] > 0 else 'undirected'
                    })
        return edges


# Convenience functions
async def discover_causal_structure(
    data: Union[np.ndarray, List[List[float]]],
    algorithm: str = 'pc',
    **kwargs
) -> Dict[str, Any]:
    """
    Quick causal discovery from data.
    
    Args:
        data: Data matrix
        algorithm: Algorithm to use
        **kwargs: Additional parameters
        
    Returns:
        Discovery results
    """
    integration = CausalLearnIntegration()
    
    if not await integration.initialize():
        return {'status': 'error', 'message': 'Failed to initialize'}
    
    return integration.discover_structure(data, algorithm=algorithm, **kwargs)


def get_ssot_info() -> Dict[str, Any]:
    """
    Get information about the SSOT (Single Source of Truth).
    
    Returns:
        Dictionary with SSOT information
    """
    return {
        'ssot_location': 'integrations/causal_learn/',
        'ssot_available': SSOT_AVAILABLE,
        'components': {
            'adapter': 'integrations/causal_learn/adapter.py',
            'bridge': 'integrations/causal_learn/bridge.py',
            'config': 'integrations/causal_learn/config.yaml',
            'interface': 'integrations/base/causal_interface.py'
        },
        'exports': [
            'CausalLearnAdapter',
            'CausalDiscoveryBridge',
            'CAUSAL_LEARN_AVAILABLE',
            'get_adapter',
            'get_bridge',
            'validate_installation'
        ],
        'version': __version__
    }
