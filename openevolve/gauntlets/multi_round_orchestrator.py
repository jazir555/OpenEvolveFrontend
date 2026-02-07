"""
OpenEvolve Multi Round Orchestrator - re-exports from core-projects
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From multi_round_orchestrator.py: gauntlets -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export from core-projects
try:
    # Import the core-projects module directly by file path to avoid circular import
    module_path = core_projects_path / "openevolve" / "gauntlets" / "multi_round_orchestrator.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_multi_round_orchestrator", module_path)
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_multi_round_orchestrator'] = core_module
        spec.loader.exec_module(core_module)

        # Extract all the classes we need
        MultiRoundGauntletOrchestrator = core_module.MultiRoundGauntletOrchestrator
        GauntletState = core_module.GauntletState
        FusedArtifacts = core_module.FusedArtifacts
        PerformanceMetrics = core_module.PerformanceMetrics
        MultiRoundConfig = core_module.MultiRoundConfig
        Round1Result = core_module.Round1Result
        Round2Result = core_module.Round2Result
        Round3Result = core_module.Round3Result
        RoundStatus = core_module.RoundStatus
        create_multi_round_orchestrator = core_module.create_multi_round_orchestrator

        __all__ = [
            'MultiRoundGauntletOrchestrator',
            'GauntletState',
            'FusedArtifacts',
            'PerformanceMetrics',
            'MultiRoundConfig',
            'Round1Result',
            'Round2Result',
            'Round3Result',
            'RoundStatus',
            'create_multi_round_orchestrator',
        ]
    else:
        raise ImportError("Could not load core-projects multi_round_orchestrator module")
except (ImportError, AttributeError) as e:
    # If core-projects not available, provide stubs
    import warnings
    warnings.warn(f"Core projects not available: {e}")

    from typing import Any, Dict, List, Optional
    from dataclasses import dataclass
    from datetime import datetime
    from enum import Enum

    class RoundStatus(Enum):
        """Round status enum (stub)."""
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"

    class MultiRoundGauntletOrchestrator:
        """Multi round gauntlet orchestrator (stub)."""
        pass

    class GauntletState:
        """Gauntlet state (stub)."""
        pass

    class FusedArtifacts:
        """Fused artifacts (stub)."""
        pass

    @dataclass
    class PerformanceMetrics:
        """Performance metrics (stub)."""
        total_duration_seconds: float = 0.0
        average_round_duration: float = 0.0
        peak_memory_mb: float = 0.0

    @dataclass
    class MultiRoundConfig:
        """Multi round config (stub)."""
        pass

    @dataclass
    class Round1Result:
        """Round 1 result (stub)."""
        pass

    @dataclass
    class Round2Result:
        """Round 2 result (stub)."""
        pass

    @dataclass
    class Round3Result:
        """Round 3 result (stub)."""
        pass

    def create_multi_round_orchestrator(config=None):
        """Create multi-round orchestrator (stub)."""
        return MultiRoundGauntletOrchestrator()

    __all__ = [
        'MultiRoundGauntletOrchestrator',
        'GauntletState',
        'FusedArtifacts',
        'PerformanceMetrics',
        'MultiRoundConfig',
        'Round1Result',
        'Round2Result',
        'Round3Result',
        'RoundStatus',
        'create_multi_round_orchestrator',
    ]
