"""
DeFi Vertical - Protocol Risk Evolution

This vertical evolves DeFi lending protocol parameters to survive historical exploits
and black swan attacks.

Components:
- DeFiProtocolEvolver: Main evolution agent
- DeFiProtocolSimulator: Attack and historical simulation
- DeFiAttackGenerator: Realistic attack scenario generation
- Historical Exploits Database: Learning from past failures
"""

from openevolve.finance.verticals.defi.defi_evolver import (
    DeFiProtocolEvolver,
    DeFiEvolutionResult,
    ProtocolParameters,
    ProtocolConstraints,
    DeFiAttackScenario,
)

from openevolve.finance.verticals.defi.defi_simulator import (
    DeFiProtocolSimulator,
    DeFiAttackResult,
    HistoricalSimulation,
    ProtocolState,
)

from openevolve.finance.verticals.defi.attack_generator import (
    DeFiAttackGenerator,
)

from openevolve.finance.verticals.defi.historical_exploits import (
    HISTORICAL_EXPLOITS,
    get_exploit_lessons,
    get_exploits_by_type,
)

__all__ = [
    # Main evolver
    "DeFiProtocolEvolver",
    "DeFiEvolutionResult",
    "ProtocolParameters",
    "ProtocolConstraints",
    "DeFiAttackScenario",
    # Simulator
    "DeFiProtocolSimulator",
    "DeFiAttackResult",
    "HistoricalSimulation",
    "ProtocolState",
    # Attack generator
    "DeFiAttackGenerator",
    # Historical data
    "HISTORICAL_EXPLOITS",
    "get_exploit_lessons",
    "get_exploits_by_type",
]

__version__ = "1.0.0"
__author__ = "OpenEvolve Finance Team"
