#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenEvolve PES (Plan-Execute-Summarize) Framework

This package provides the core PES framework extracted and adapted from LoongFlow.
It manages evolutionary processes with concurrent Plan -> Execute -> Summarize cycles.

Main Components:
- PESAgent: Main orchestrator for evolution processes
- BasePESRunner: Base class for creating PES agent runners
- EvolveDatabase: Memory and database management
- Configuration classes: Pydantic models for PES configuration

Example Usage:
    ```python
    from openevolve.pes import PESAgent
    from openevolve.pes.config import EvolveChainConfig

    # Load configuration
    config = EvolveChainConfig.model_validate(config_dict)

    # Create and run agent
    agent = PESAgent(config=config)
    result = await agent.run()
    ```
"""

from openevolve.pes.pes_agent import PESAgent
from openevolve.pes.base_runner import BasePESRunner
from openevolve.pes.config import (
    EvolveChainConfig,
    EvolveConfig,
    EvaluatorConfig,
    LLMConfig,
    LoggerConfig,
    DatabaseConfig,
    Context,
    Workspace,
    load_config,
)
from openevolve.pes.memory import EvolveDatabase
from openevolve.pes.utils import (
    Worker,
    register_worker,
    get_worker,
    Finalizer,
    PESFinalizer,
    PLANNER,
    EXECUTOR,
    SUMMARY,
)

__version__ = "0.1.0"
__author__ = "OpenEvolve Team"

__all__ = [
    # Main Agent Classes
    "PESAgent",
    "BasePESRunner",

    # Configuration
    "EvolveChainConfig",
    "EvolveConfig",
    "EvaluatorConfig",
    "LLMConfig",
    "LoggerConfig",
    "DatabaseConfig",
    "Context",
    "Workspace",
    "load_config",

    # Memory/Database
    "EvolveDatabase",

    # Workers
    "Worker",
    "register_worker",
    "get_worker",
    "PLANNER",
    "EXECUTOR",
    "SUMMARY",

    # Finalizer
    "Finalizer",
    "PESFinalizer",
]
