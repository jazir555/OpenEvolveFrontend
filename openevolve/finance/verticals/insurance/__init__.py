"""
Insurance vertical - re-exports from core-projects
"""
import sys
from pathlib import Path
import importlib.util

# Add core-projects to Python path if not already there
# From __init__.py: insurance -> verticals -> finance -> openevolve -> Frontend -> core-projects/openevolve
core_projects_path = Path(__file__).parent.parent.parent.parent.parent / "core-projects" / "openevolve"
if str(core_projects_path) not in sys.path:
    sys.path.insert(0, str(core_projects_path))

# Import and re-export all insurance vertical components from core-projects
try:
    # Import the core-projects insurance module directly by file path to avoid circular import
    insurance_module_path = core_projects_path / "openevolve" / "finance" / "verticals" / "insurance" / "__init__.py"

    spec = importlib.util.spec_from_file_location("openevolve_core_finance_insurance", insurance_module_path)
    if spec and spec.loader:
        insurance_module = importlib.util.module_from_spec(spec)
        sys.modules['openevolve_core_finance_insurance'] = insurance_module
        spec.loader.exec_module(insurance_module)

        # Extract all the classes and functions we need
        RBCCalculator = insurance_module.RBCCalculator
        RBCCalculationResult = insurance_module.RBCCalculationResult
        Portfolio = insurance_module.Portfolio
        Bond = insurance_module.Bond
        CreditRating = insurance_module.CreditRating
        InsuranceReserveEvolver = insurance_module.InsuranceReserveEvolver
        InsuranceEvolutionResult = insurance_module.InsuranceEvolutionResult
        PortfolioConstraints = insurance_module.PortfolioConstraints
        StressScenario = insurance_module.StressScenario
        ScenarioEvolutionResult = insurance_module.ScenarioEvolutionResult
        StressScenarioGenerator = insurance_module.StressScenarioGenerator
        HistoricalCrises = insurance_module.HistoricalCrises
        RateShockScenario = insurance_module.RateShockScenario

        # Also re-export legacy aliases for backward compatibility
        InsuranceOptimizer = InsuranceReserveEvolver

        __all__ = [
            'RBCCalculator',
            'RBCCalculationResult',
            'Portfolio',
            'Bond',
            'CreditRating',
            'InsuranceReserveEvolver',
            'InsuranceEvolutionResult',
            'PortfolioConstraints',
            'StressScenario',
            'ScenarioEvolutionResult',
            'StressScenarioGenerator',
            'HistoricalCrises',
            'RateShockScenario',
            'InsuranceOptimizer',  # Legacy alias
        ]
    else:
        raise ImportError("Could not load core-projects insurance module")
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

    class RBCCalculator:
        """Risk-Based Capital calculator (stub)."""
        def calculate(self, portfolio_value: float, liabilities: float, portfolio: Optional[Portfolio] = None) -> float:
            return 350.0  # Minimum RBC ratio

    class InsuranceOptimizer:
        """Insurance optimizer (stub)."""
        pass

    class InsuranceReserveEvolver:
        """Insurance reserve evolver (stub)."""
        pass

    class InsuranceEvolutionResult:
        """Insurance evolution result (stub)."""
        pass

    class PortfolioConstraints:
        """Portfolio constraints (stub)."""
        pass

    class StressScenario:
        """Stress scenario (stub)."""
        pass

    class ScenarioEvolutionResult:
        """Scenario evolution result (stub)."""
        pass

    class StressScenarioGenerator:
        """Stress scenario generator (stub)."""
        pass

    class HistoricalCrises:
        """Historical crises (stub)."""
        pass

    class RateShockScenario:
        """Rate shock scenario (stub)."""
        pass

    __all__ = [
        'RBCCalculator',
        'Portfolio',
        'Bond',
        'CreditRating',
        'InsuranceOptimizer',
        'InsuranceReserveEvolver',
        'InsuranceEvolutionResult',
        'PortfolioConstraints',
        'StressScenario',
        'ScenarioEvolutionResult',
        'StressScenarioGenerator',
        'HistoricalCrises',
        'RateShockScenario',
    ]
