"""
Curie Integration - Automated Scientific Experimentation for OpenEvolve

This package provides integration with Curie's automated scientific experimentation
framework, filling GAP-4 (Experimental Data Integration) and GAP-12 (Scientific
Experimentation Automation).

Key Components:
- CurieAdapter: Main adapter implementing ExperimentationInterface
- CurieBridge: Integration with SOP Generator and validation systems
- Experiment templates for physics, chemistry, and biology

Usage:
    from integrations.curie import CurieAdapter, CurieConfig

    config = CurieConfig(
        openai_api_key="your-api-key",
        domain="physics"
    )

    adapter = CurieAdapter(config)
    await adapter.initialize({})

    # Design experiment
    protocol = await adapter.design_experiment(
        hypothesis="Increasing temperature increases reaction rate",
        domain=ExperimentDomain.CHEMISTRY
    )

    # Run experiment
    results = await adapter.run_experiment(protocol)

    # Analyze results
    analysis = await adapter.analyze_results(results, protocol.hypothesis)

Author: Agent 3 (Curie Integration Specialist)
Version: 1.0.0
Repository: https://github.com/Just-Curieous/curie
"""

from .adapter import CurieAdapter, CurieConfig
from .bridge import CurieBridge

__all__ = [
    "CurieAdapter",
    "CurieConfig",
    "CurieBridge"
]

__version__ = "1.0.0"
__author__ = "Agent 3 (Curie Integration Specialist)"
