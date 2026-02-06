"""Gauntlet Structures module."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class GauntletConfig:
    """Configuration for gauntlet execution."""
    name: str = "default"
    rounds: int = 3

@dataclass
class GauntletResult:
    """Result from gauntlet execution."""
    success: bool = False
    score: float = 0.0
    feedback: str = ""
