"""
OpenEvolve Gauntlets module - re-exports from core-projects
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From __init__.py: gauntlets -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export all gauntlet components from core-projects
try:
    # Import the core-projects gauntlets module directly by file path to avoid circular import
    gauntlets_module_path = core_projects_path / "openevolve" / "gauntlets" / "__init__.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_gauntlets", gauntlets_module_path)
    if spec and spec.loader:
        gauntlets_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_gauntlets'] = gauntlets_module
        spec.loader.exec_module(gauntlets_module)

        # Extract all the classes and functions we need
        LoongFlowGauntletEvaluator = gauntlets_module.LoongFlowGauntletEvaluator
        LoongFlowGauntletConfig = gauntlets_module.LoongFlowGauntletConfig
        GauntletEvaluationResult = gauntlets_module.GauntletEvaluationResult
        ThreeRoundGauntletOrchestrator = gauntlets_module.ThreeRoundGauntletOrchestrator
        ThreeRoundConfig = gauntlets_module.ThreeRoundConfig
        FullGauntletResult = gauntlets_module.FullGauntletResult
        Round1Result = gauntlets_module.Round1Result
        Round2Result = gauntlets_module.Round2Result
        Round3Result = gauntlets_module.Round3Result
        MultiRoundGauntletOrchestrator = gauntlets_module.MultiRoundGauntletOrchestrator
        GauntletState = gauntlets_module.GauntletState
        FusedArtifacts = gauntlets_module.FusedArtifacts

        # Legacy aliases for backward compatibility
        GauntletOrchestrator = MultiRoundGauntletOrchestrator
        ThreeRoundOrchestrator = ThreeRoundGauntletOrchestrator
        MultiRoundOrchestrator = MultiRoundGauntletOrchestrator

        __all__ = [
            'LoongFlowGauntletEvaluator',
            'LoongFlowGauntletConfig',
            'GauntletEvaluationResult',
            'ThreeRoundGauntletOrchestrator',
            'ThreeRoundConfig',
            'FullGauntletResult',
            'Round1Result',
            'Round2Result',
            'Round3Result',
            'MultiRoundGauntletOrchestrator',
            'GauntletState',
            'FusedArtifacts',
            'GauntletOrchestrator',  # Legacy alias
            'ThreeRoundOrchestrator',  # Legacy alias
            'MultiRoundOrchestrator',  # Legacy alias
        ]
    else:
        raise ImportError("Could not load core-projects gauntlets module")
except (ImportError, AttributeError) as e:
    # If core-projects not available, provide stubs
    import warnings
    warnings.warn(f"Core projects not available: {e}")

    from typing import Any, Dict, List, Optional

    class GauntletOrchestrator:
        """Gauntlet orchestrator (stub)."""
        pass

    class ThreeRoundOrchestrator:
        """Three round orchestrator (stub)."""
        pass

    class MultiRoundOrchestrator:
        """Multi round orchestrator (stub)."""
        pass

    class LoongFlowGauntletEvaluator:
        """LoongFlow gauntlet evaluator (stub)."""
        pass

    class LoongFlowGauntletConfig:
        """LoongFlow gauntlet config (stub)."""
        pass

    class GauntletEvaluationResult:
        """Gauntlet evaluation result (stub)."""
        pass

    class ThreeRoundGauntletOrchestrator:
        """Three round gauntlet orchestrator (stub)."""
        pass

    class ThreeRoundConfig:
        """Three round config (stub)."""
        pass

    class FullGauntletResult:
        """Full gauntlet result (stub)."""
        pass

    class Round1Result:
        """Round 1 result (stub)."""
        pass

    class Round2Result:
        """Round 2 result (stub)."""
        pass

    class Round3Result:
        """Round 3 result (stub)."""
        pass

    class GauntletState:
        """Gauntlet state (stub)."""
        pass

    class FusedArtifacts:
        """Fused artifacts (stub)."""
        pass

    __all__ = [
        'GauntletOrchestrator',
        'ThreeRoundOrchestrator',
        'MultiRoundOrchestrator',
        'LoongFlowGauntletEvaluator',
        'LoongFlowGauntletConfig',
        'GauntletEvaluationResult',
        'ThreeRoundGauntletOrchestrator',
        'ThreeRoundConfig',
        'FullGauntletResult',
        'Round1Result',
        'Round2Result',
        'Round3Result',
        'GauntletState',
        'FusedArtifacts',
    ]
