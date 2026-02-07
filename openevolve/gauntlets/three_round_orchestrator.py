"""
OpenEvolve Three Round Orchestrator - re-exports from core-projects
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From three_round_orchestrator.py: gauntlets -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export from core-projects
try:
    # Import the core-projects module directly by file path to avoid circular import
    module_path = core_projects_path / "openevolve" / "gauntlets" / "three_round_orchestrator.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_three_round_orchestrator", module_path)
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_three_round_orchestrator'] = core_module
        spec.loader.exec_module(core_module)

        # Extract all the classes we need
        ThreeRoundGauntletOrchestrator = core_module.ThreeRoundGauntletOrchestrator
        ThreeRoundConfig = core_module.ThreeRoundConfig
        FullGauntletResult = core_module.FullGauntletResult
        Round1Result = core_module.Round1Result
        Round2Result = core_module.Round2Result
        Round3Result = core_module.Round3Result
        GauntletRound = core_module.GauntletRound
        create_strict_config = core_module.create_strict_config
        create_lenient_config = core_module.create_lenient_config
        create_balanced_config = core_module.create_balanced_config
        create_domain_config = core_module.create_domain_config

        __all__ = [
            'ThreeRoundGauntletOrchestrator',
            'ThreeRoundConfig',
            'FullGauntletResult',
            'Round1Result',
            'Round2Result',
            'Round3Result',
            'GauntletRound',
            'create_strict_config',
            'create_lenient_config',
            'create_balanced_config',
            'create_domain_config',
        ]
    else:
        raise ImportError("Could not load core-projects three_round_orchestrator module")
except (ImportError, AttributeError) as e:
    # If core-projects not available, provide stubs
    import warnings
    warnings.warn(f"Core projects not available: {e}")

    from typing import Any, Dict, List, Optional
    from dataclasses import dataclass
    from enum import Enum

    class GauntletRound(Enum):
        """Gauntlet round enum (stub)."""
        ROUND1 = "round1"
        ROUND2 = "round2"
        ROUND3 = "round3"

    class ThreeRoundGauntletOrchestrator:
        """Three round gauntlet orchestrator (stub)."""
        pass

    @dataclass
    class ThreeRoundConfig:
        """Three round config (stub)."""
        pass

    @dataclass
    class FullGauntletResult:
        """Full gauntlet result (stub)."""
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

    def create_strict_config():
        """Create strict config (stub)."""
        return ThreeRoundConfig()

    def create_lenient_config():
        """Create lenient config (stub)."""
        return ThreeRoundConfig()

    def create_balanced_config():
        """Create balanced config (stub)."""
        return ThreeRoundConfig()

    def create_domain_config(domain):
        """Create domain config (stub)."""
        return ThreeRoundConfig()

    __all__ = [
        'ThreeRoundGauntletOrchestrator',
        'ThreeRoundConfig',
        'FullGauntletResult',
        'Round1Result',
        'Round2Result',
        'Round3Result',
        'GauntletRound',
        'create_strict_config',
        'create_lenient_config',
        'create_balanced_config',
        'create_domain_config',
    ]
