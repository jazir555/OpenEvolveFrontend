"""
Configuration Presets for OpenEvolve Unified Evolution Engine

This package provides ready-to-use configuration presets for common use cases.
Presets are organized into categories:
- Performance: Speed/resource optimization presets
- Domains: Domain-specific configurations
- Use Cases: Common usage scenarios
- Systems: System mode configurations
- Problem Types: Problem-specific configurations
"""

from .base import BasePreset, PresetInfo, ValidationResult, PresetComparison
from .performance import (
    FastPreset,
    BalancedPreset,
    ThoroughPreset,
    BudgetPreset
)
from .domains import (
    # Finance
    FinanceGeneralPreset,
    FinancePortfolioPreset,
    FinanceRiskPreset,
    # Trading
    TradingGeneralPreset,
    TradingSignalPreset,
    TradingParameterPreset,
    # Science
    ScienceGeneralPreset,
    ScienceOptimizationPreset,
    ScienceDiscoveryPreset,
    # Engineering
    EngineeringGeneralPreset,
    EngineeringDesignPreset,
    EngineeringControlPreset,
    # Pharma
    PharmaGeneralPreset,
    PharmaDrugDiscoveryPreset,
    PharmaClinicalPreset,
    # Web Design
    WebDesignGeneralPreset,
    WebDesignUxPreset,
    WebDesignPerformancePreset
)
from .use_cases import (
    QuickPrototypePreset,
    ProductionPreset,
    ResearchPreset,
    ResourceConstrainedPreset,
    QualityCriticalPreset
)
from .systems import (
    PureOpenEvolvePreset,
    PureLoongFlowPreset,
    HybridAutoPreset,
    CustomPreset
)
from .problem_types import (
    SingleObjectivePreset,
    MultiObjectivePreset,
    ExpensiveEvaluationPreset,
    FastEvaluationPreset,
    SafetyCriticalPreset
)
from .manager import PresetManager

__all__ = [
    # Base classes
    "BasePreset",
    "PresetInfo",
    "ValidationResult",
    "PresetComparison",
    "PresetManager",

    # Performance presets
    "FastPreset",
    "BalancedPreset",
    "ThoroughPreset",
    "BudgetPreset",

    # Domain presets - Finance
    "FinanceGeneralPreset",
    "FinancePortfolioPreset",
    "FinanceRiskPreset",

    # Domain presets - Trading
    "TradingGeneralPreset",
    "TradingSignalPreset",
    "TradingParameterPreset",

    # Domain presets - Science
    "ScienceGeneralPreset",
    "ScienceOptimizationPreset",
    "ScienceDiscoveryPreset",

    # Domain presets - Engineering
    "EngineeringGeneralPreset",
    "EngineeringDesignPreset",
    "EngineeringControlPreset",

    # Domain presets - Pharma
    "PharmaGeneralPreset",
    "PharmaDrugDiscoveryPreset",
    "PharmaClinicalPreset",

    # Domain presets - Web Design
    "WebDesignGeneralPreset",
    "WebDesignUxPreset",
    "WebDesignPerformancePreset",

    # Use case presets
    "QuickPrototypePreset",
    "ProductionPreset",
    "ResearchPreset",
    "ResourceConstrainedPreset",
    "QualityCriticalPreset",

    # System mode presets
    "PureOpenEvolvePreset",
    "PureLoongFlowPreset",
    "HybridAutoPreset",
    "CustomPreset",

    # Problem type presets
    "SingleObjectivePreset",
    "MultiObjectivePreset",
    "ExpensiveEvaluationPreset",
    "FastEvaluationPreset",
    "SafetyCriticalPreset",
]
