#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Trading Strategy Evolution System

A continuous 24/7 autonomous trading strategy research and evolution platform.
Uses evolutionary algorithms, LLM reasoning, and causal learning to discover
and refine profitable trading strategies.

Main Components:
- TradingEvolver: Main orchestrator
- RLMGenerator: Strategy ideation via reasoning
- VariantManager: Parallel variant testing
- JudgePanel: Multi-perspective evaluation
- CausalModeler: Learning from outcomes
- Adversary: Red team testing
"""

from openevolve.agents.trading.schemas import (
    Strategy,
    StrategyVariant,
    StrategyPerformance,
    MarketData,
    TradeSignal,
    EvolutionState
)
from openevolve.agents.trading.trading_evolver import TradingEvolver
from openevolve.agents.trading.rlm_generator import RLMGenerator
from openevolve.agents.trading.variant_manager import VariantManager
from openevolve.agents.trading.judge_panel import JudgePanel
from openevolve.agents.trading.causal_modeler import CausalModeler
from openevolve.agents.trading.adversary import Adversary

__all__ = [
    # Schemas
    "Strategy",
    "StrategyVariant",
    "StrategyPerformance",
    "MarketData",
    "TradeSignal",
    "EvolutionState",
    # Main components
    "TradingEvolver",
    "RLMGenerator",
    "VariantManager",
    "JudgePanel",
    "CausalModeler",
    "Adversary",
]
