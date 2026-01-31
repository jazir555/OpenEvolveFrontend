"""
Domain-Specific Optimizers
Pre-configured optimizers for 6 target domains with custom best practices

Domains:
- Finance: Portfolio optimization, risk analysis
- Trading: Strategy development, signal optimization
- Science: Experimental design, data analysis
- Engineering: Structural optimization, circuit design
- Pharma: Molecular optimization, drug design
- Web Design: Landing page optimization, UX optimization

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig

# Define enums locally since they're not in unified.config
from enum import Enum

class EvolutionMode(str, Enum):
    PES = "pes"
    QD = "qd"
    MO = "mo"
    ADVERSARIAL = "adversarial"

class DomainType(str, Enum):
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB_DESIGN = "web_design"
    GENERAL = "general"

# Import base class first
from .base import DomainOptimizer

# Then import domain optimizers
from .finance_optimizer import FinanceOptimizer
from .trading_optimizer import TradingOptimizer
from .science_optimizer import ScienceOptimizer
from .engineering_optimizer import EngineeringOptimizer
from .pharma_optimizer import PharmaOptimizer
from .web_design_optimizer import WebDesignOptimizer

__all__ = [
    'FinanceOptimizer',
    'TradingOptimizer',
    'ScienceOptimizer',
    'EngineeringOptimizer',
    'PharmaOptimizer',
    'WebDesignOptimizer',
    'detect_domain',
    'get_optimizer',
    'optimize_by_domain',
    'optimize_multi_domain'
]


# ============================================================================
# DOMAIN KEYWORDS FOR AUTO-DETECTION
# ============================================================================

DOMAIN_KEYWORDS = {
    "finance": [
        "portfolio", "asset", "allocation", "sharpe", "risk", "return",
        "volatility", "var", "cvar", "drawdown", "sortino", "treynor",
        "beta", "alpha", "benchmark", "rebalance", "diversification",
        "optimization", "backtest", "asset_class", "equity", "bond",
        "derivative", "option", "future", "commodity"
    ],
    "trading": [
        "strategy", "signal", "indicator", "entry", "exit", "stop_loss",
        "take_profit", "position_sizing", "momentum", "trend", "reversal",
        "breakout", "support", "resistance", "crossover", "moving_average",
        "rsi", "macd", "bollinger", "regime", "market_condition",
        "volatility_spike", "black_swan", "profit_factor", "win_rate"
    ],
    "science": [
        "experiment", "hypothesis", "scientific", "research", "data_analysis",
        "experimental_design", "doe", "statistical", "discovery", "novelty",
        "reproducibility", "cost_efficiency", "scientific_method", "analysis",
        "laboratory", "reaction", "synthesis", "characterization", "measurement",
        "assay", "screening", "validation", "clinical_trial"
    ],
    "engineering": [
        "structural", "optimization", "fea", "simulation", "circuit", "control",
        "design", "weight", "strength", "safety", "reliability", "fatigue",
        "resonance", "load_exceedance", "constraint", "manufacturing", "tolerance",
        "cad", "cae", "mesh", "finite_element", "vibration", "thermal",
        "fluid_dynamics", "optimization_engineering", "mechanical", "civil"
    ],
    "pharma": [
        "molecule", "drug", "binding", "clinical", "admet", "synthesis",
        "pharmacokinetic", "pharmacodynamic", "toxicity", "efficacy", "solubility",
        "permeability", "bioavailability", "drug_likeness", "synthetic_accessibility",
        "lead_compound", "target", "receptor", "enzyme", "inhibitor", "agonist",
        "antagonist", "formulation", "dosage", "clinical_trial_design"
    ],
    "web_design": [
        "landing_page", "ux", "conversion", "bounce", "user", "interface",
        "ui", "web_design", "a_b_test", "variant", "click_through", "engagement",
        "time_on_page", "scroll_depth", "user_satisfaction", "navigation",
        "layout", "visual", "accessibility", "responsive", "mobile", "desktop",
        "call_to_action", "headline", "copy", "funnel", "acquisition"
    ]
}


# ============================================================================
# DOMAIN AUTO-DETECTION
# ============================================================================

def detect_domain(problem_description: str, keywords: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Auto-detect domain from problem description

    Args:
        problem_description: Problem statement
        keywords: Optional custom keyword dictionary

    Returns:
        Domain name (e.g., 'finance', 'trading', etc.) or 'general'

    Example:
        >>> detect_domain("Optimize portfolio allocation for max return")
        'finance'
        >>> detect_domain("Design trading strategy with entry/exit rules")
        'trading'
    """
    if keywords is None:
        keywords = DOMAIN_KEYWORDS

    problem_lower = problem_description.lower()

    # Score each domain by keyword matches
    scores = {}
    for domain, domain_keywords in keywords.items():
        score = sum(1 for kw in domain_keywords if kw.lower() in problem_lower)
        scores[domain] = score

    # Find domain with highest score
    best_domain = max(scores, key=scores.get)

    # Return 'general' if no keywords matched
    if scores[best_domain] == 0:
        return "general"

    return best_domain


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

def get_optimizer(domain: str, sub_domain: str = "general") -> 'DomainOptimizer':
    """
    Get optimizer for domain

    Args:
        domain: Domain name (finance, trading, science, engineering, pharma, web_design)
        sub_domain: Sub-domain specialization (e.g., 'portfolio', 'risk')

    Returns:
        Domain-specific optimizer instance

    Example:
        >>> optimizer = get_optimizer('finance', 'portfolio')
        >>> result = optimizer.optimize("Maximize return with min risk")
    """
    optimizer_classes = {
        "finance": FinanceOptimizer,
        "trading": TradingOptimizer,
        "science": ScienceOptimizer,
        "engineering": EngineeringOptimizer,
        "pharma": PharmaOptimizer,
        "web_design": WebDesignOptimizer,
        "web": WebDesignOptimizer  # Alias
    }

    OptimizerClass = optimizer_classes.get(domain, FinanceOptimizer)  # Default to finance
    return OptimizerClass(sub_domain=sub_domain)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def optimize_by_domain(
    problem: str,
    domain: Optional[str] = None,
    sub_domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize using domain-specific configuration

    Args:
        problem: Problem description
        domain: Domain (auto-detected if None)
        sub_domain: Sub-domain specialization
        constraints: Additional constraints
        **kwargs: Additional parameters

    Returns:
        Optimization result with domain-specific metrics

    Example:
        >>> result = await optimize_by_domain(
        ...     "Optimize portfolio allocation",
        ...     domain="finance",
        ...     sub_domain="portfolio"
        ... )
        >>> print(result['sharpe_ratio'])
    """
    # Auto-detect domain if not specified
    if domain is None:
        domain = detect_domain(problem)

    # Get optimizer
    optimizer = get_optimizer(domain, sub_domain)

    # Run optimization
    return await optimizer.optimize(problem, constraints=constraints, **kwargs)


async def optimize_multi_domain(
    problem: str,
    domains: List[str],
    sub_domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize for multiple domains and compare results

    Args:
        problem: Problem description
        domains: List of domains to try
        sub_domain: Sub-domain specialization
        constraints: Additional constraints
        **kwargs: Additional parameters

    Returns:
        Dictionary mapping domain to result

    Example:
        >>> results = await optimize_multi_domain(
        ...     "Optimize trading strategy",
        ...     domains=['trading', 'finance']
        ... )
        >>> print(results['trading']['sharpe_ratio'])
        >>> print(results['finance']['sharpe_ratio'])
    """
    results = {}

    for domain in domains:
        try:
            optimizer = get_optimizer(domain, sub_domain)
            results[domain] = await optimizer.optimize(
                problem, constraints=constraints, **kwargs
            )
        except Exception as e:
            results[domain] = {
                'error': str(e),
                'domain': domain,
                'success': False
            }

    return results


