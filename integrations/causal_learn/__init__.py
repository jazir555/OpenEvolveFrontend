"""
Causal-learn Integration Package for OpenEvolve

This package provides integration between causal-learn causal discovery library
and OpenEvolve systems including SOP Generator, Problem Analyzer, Knowledge Engine,
and ROMA/MDAP.

Components:
- CausalLearnAdapter: Adapter implementing CausalDiscoveryInterface
- CausalDiscoveryBridge: Bridge to OpenEvolve systems

Usage:
    from integrations.causal_learn import CausalLearnAdapter, CausalDiscoveryBridge

    # Use adapter
    adapter = CausalLearnAdapter()
    await adapter.initialize(config)
    result = await adapter.discover_causal_structure(data)

    # Use bridge
    bridge = CausalDiscoveryBridge()
    await bridge.initialize()
    validation = await bridge.pre_experiment_validation(workflow_data)

Author: Causal-learn Integration Specialist
Version: 1.0.0
Date: 2026-01-02
"""

from typing import Dict, Any, Optional, List
import logging
import os

# Try to import causal-learn
try:
    import sys
    causal_learn_path = os.path.join(
        os.path.dirname(__file__),
        "../../projects to analyze/causal-learn"
    )
    if causal_learn_path not in sys.path:
        sys.path.insert(0, causal_learn_path)

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False

from integrations.causal_learn.adapter import CausalLearnAdapter
from integrations.causal_learn.bridge import CausalDiscoveryBridge

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    "CausalLearnAdapter",
    "CausalDiscoveryBridge",
    "CAUSAL_LEARN_AVAILABLE",
]


def get_adapter() -> CausalLearnAdapter:
    """
    Factory function to get a configured CausalLearnAdapter.

    Returns:
        CausalLearnAdapter instance

    Raises:
        ImportError: If causal-learn is not available
    """
    if not CAUSAL_LEARN_AVAILABLE:
        raise ImportError(
            "causal-learn is not available. "
            "Install with: pip install causal-learn"
        )

    return CausalLearnAdapter()


def get_bridge(config_path: Optional[str] = None) -> CausalDiscoveryBridge:
    """
    Factory function to get a configured CausalDiscoveryBridge.

    Args:
        config_path: Optional path to config.yaml file

    Returns:
        CausalDiscoveryBridge instance

    Raises:
        ImportError: If causal-learn is not available
    """
    if not CAUSAL_LEARN_AVAILABLE:
        raise ImportError(
            "causal-learn is not available. "
            "Install with: pip install causal-learn"
        )

    return CausalDiscoveryBridge(config_path=config_path)


async def validate_installation() -> Dict[str, Any]:
    """
    Validate causal-learn installation.

    Returns:
        Dictionary with validation results:
        - available: Whether causal-learn is available
        - version: causal-learn version (if available)
        - algorithms: List of available algorithms
        - issues: List of any issues found
    """
    result = {
        'available': CAUSAL_LEARN_AVAILABLE,
        'version': None,
        'algorithms': [],
        'issues': []
    }

    if not CAUSAL_LEARN_AVAILABLE:
        result['issues'].append("causal-learn not installed")
        return result

    # Try to get version
    try:
        import causallearn
        result['version'] = getattr(causallearn, '__version__', '0.1.4.4')
    except Exception as e:
        result['issues'].append(f"Could not determine version: {e}")

    # Test algorithms
    algorithms = ['pc', 'ges', 'fci', 'direct_lingam']
    for algo in algorithms:
        try:
            # Simple test
            import numpy as np
            data = np.random.randn(100, 3)

            if algo == 'pc':
                from causallearn.search.ConstraintBased.PC import pc
                from causallearn.utils.cit import fisherz
                cg = pc(data, 0.05, fisherz)
                result['algorithms'].append('pc')
            elif algo == 'ges':
                from causallearn.search.ScoreBased.GES import ges
                res = ges(data)
                result['algorithms'].append('ges')
            elif algo == 'fci':
                from causallearn.search.ConstraintBased.FCI import fci
                from causallearn.utils.cit import fisherz
                cg = fci(data, 0.05, fisherz)
                result['algorithms'].append('fci')
            elif algo == 'direct_lingam':
                from causallearn.search.FCMBased.lingam import DirectLiNGAM
                model = DirectLiNGAM()
                model.fit(data)
                result['algorithms'].append('direct_lingam')
        except Exception as e:
            result['issues'].append(f"Algorithm {algo} failed: {e}")

    return result


# Export key classes
__all__.extend([
    "get_adapter",
    "get_bridge",
    "validate_installation",
])


# Module-level initialization
if CAUSAL_LEARN_AVAILABLE:
    logger.info("Causal-learn integration loaded successfully")
    _available_algorithms = [
        "PC",
        "GES",
        "FCI",
        "DirectLiNGAM",
        "ICA-LiNGAM",
        "VAR-LiNGAM",
    ]
    logger.info("Available algorithms: %s", ", ".join(_available_algorithms))
else:
    logger.warning("Causal-learn not available. Integration will use graceful degradation.")
