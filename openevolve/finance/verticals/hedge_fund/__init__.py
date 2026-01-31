#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hedge Fund Vertical - Alpha Signal Discovery and Evolution

This vertical provides tools for discovering and evolving alpha signals
that survive market crises and avoid overfitting to delisted microcaps.

Key Features:
- Survivorship-bias-free backtesting
- Crisis period validation (2000, 2008, 2020, 2022)
- Feature importance learning
- Multi-signal combination (ensemble)
- Adversarial testing for robustness
"""

from openevolve.finance.verticals.hedge_fund.alpha_evolver import (
    AlphaSignalEvolver,
    AlphaDiscoveryResult,
    SignalConstraints,
    FeatureHypothesis,
    SignalEvolutionResult
)

from openevolve.finance.verticals.hedge_fund.feature_importance import (
    FeatureImportanceTracker
)

from openevolve.finance.verticals.hedge_fund.signal_combiner import (
    SignalCombiner,
    EnsembleSignal
)

from openevolve.finance.verticals.hedge_fund.schemas import (
    AlphaSignal,
    CrisisPerformance,
    BacktestResult,
    FeatureSet
)

__all__ = [
    # Main evolver
    "AlphaSignalEvolver",
    "AlphaDiscoveryResult",
    "SignalConstraints",
    "FeatureHypothesis",
    "SignalEvolutionResult",

    # Feature tracking
    "FeatureImportanceTracker",

    # Signal combination
    "SignalCombiner",
    "EnsembleSignal",

    # Data structures
    "AlphaSignal",
    "CrisisPerformance",
    "BacktestResult",
    "FeatureSet",
]

__version__ = "0.1.0"
__author__ = "Claude (Sonnet 4.5)"
__date__ = "January 30, 2026"
