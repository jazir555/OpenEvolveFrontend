"""
Insurance vertical models - re-exports from core-projects
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From insurance_models.py: verticals -> finance -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export from core-projects
try:
    # Import the core-projects models module directly by file path to avoid circular import
    models_module_path = core_projects_path / "openevolve" / "finance" / "verticals" / "insurance" / "models.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_insurance_models", models_module_path)
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_insurance_models'] = core_module
        spec.loader.exec_module(core_module)

        # Extract all the classes we need
        CreditRating = core_module.CreditRating
        Bond = core_module.Bond
        Portfolio = core_module.Portfolio
        PortfolioConstraints = core_module.PortfolioConstraints
        StressScenario = core_module.StressScenario
        StressTestResult = core_module.StressTestResult
        InsuranceEvolutionResult = core_module.InsuranceEvolutionResult
        ScenarioEvolutionResult = core_module.ScenarioEvolutionResult
        RBCCalculationResult = core_module.RBCCalculationResult

        __all__ = [
            'CreditRating',
            'Bond',
            'Portfolio',
            'PortfolioConstraints',
            'StressScenario',
            'StressTestResult',
            'InsuranceEvolutionResult',
            'ScenarioEvolutionResult',
            'RBCCalculationResult',
        ]
    else:
        raise ImportError("Could not load core-projects insurance models module")
except (ImportError, AttributeError) as e:
    # If core-projects not available, provide stubs
    import warnings
    warnings.warn(f"Core projects not available: {e}")

    from typing import Any, Dict, List, Optional
    from dataclasses import dataclass
    from datetime import datetime
    from enum import Enum

    class CreditRating(Enum):
        """Credit rating scale (stub)."""
        AAA = "AAA"
        AA = "AA"
        A = "A"
        BBB = "BBB"
        BB = "BB"
        B = "B"
        CCC = "CCC"
        CC = "CC"
        C = "C"
        D = "D"

    @dataclass
    class Bond:
        """Bond position (stub)."""
        ticker: str
        rating: CreditRating
        par_value: float
        market_value: float
        book_value: float
        duration: float
        convexity: float
        yield_to_maturity: float
        sector: str
        coupon_rate: float
        maturity_date: datetime

    @dataclass
    class Portfolio:
        """Insurance reserve portfolio (stub)."""
        bonds: List[Bond] = None
        cash: float = 0.0
        total_value: float = 0.0

    @dataclass
    class PortfolioConstraints:
        """Portfolio constraints (stub)."""
        pass

    @dataclass
    class StressScenario:
        """Stress scenario (stub)."""
        pass

    @dataclass
    class StressTestResult:
        """Stress test result (stub)."""
        pass

    @dataclass
    class InsuranceEvolutionResult:
        """Insurance evolution result (stub)."""
        pass

    @dataclass
    class ScenarioEvolutionResult:
        """Scenario evolution result (stub)."""
        pass

    @dataclass
    class RBCCalculationResult:
        """RBC calculation result (stub)."""
        pass

    __all__ = [
        'CreditRating',
        'Bond',
        'Portfolio',
        'PortfolioConstraints',
        'StressScenario',
        'StressTestResult',
        'InsuranceEvolutionResult',
        'ScenarioEvolutionResult',
        'RBCCalculationResult',
    ]
