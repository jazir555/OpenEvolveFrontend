"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.api import (
    run_evolution,
    evolve_function,
    evolve_algorithm,
    evolve_code,
    EvolutionResult,
)
from openevolve.hybrid_maker import (
    MakerHybridStrategy,
    MCTSThenMaker,
    MakerThenEvolution,
    MakerAdversarialHybrid,
    AdaptiveMakerHybrid,
    MakerMDAPParallel,
    FullMakerHybrid,
    VerificationOracle,
    CandidateGenerator,
    DefaultCandidateGenerator,
    VerificationResult,
    MakerHybridConfig,
    MakerHybridMode,
    MakerHybridResult,
    create_maker_hybrid,
    get_maker_hybrid_capabilities,
)

__all__ = [
    "OpenEvolve",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
    "MakerHybridStrategy",
    "MCTSThenMaker",
    "MakerThenEvolution",
    "MakerAdversarialHybrid",
    "AdaptiveMakerHybrid",
    "MakerMDAPParallel",
    "FullMakerHybrid",
    "VerificationOracle",
    "CandidateGenerator",
    "DefaultCandidateGenerator",
    "VerificationResult",
    "MakerHybridConfig",
    "MakerHybridMode",
    "MakerHybridResult",
    "create_maker_hybrid",
    "get_maker_hybrid_capabilities",
]

# Ensure top-level knowledge_engine is available under openevolve.knowledge_engine
try:  # pragma: no cover - runtime alias for integration compatibility
    import importlib
    import sys

    import knowledge_engine as _knowledge_engine

    sys.modules.setdefault("openevolve.knowledge_engine", _knowledge_engine)
    try:
        sys.modules.setdefault(
            "openevolve.knowledge_engine.integrations",
            importlib.import_module("knowledge_engine.integrations"),
        )
    except Exception:
        pass
except Exception:
    pass
