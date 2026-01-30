"""
RAGBits Agents Module

Provides agent coordination and A2A protocol integration for workflow teams.
Bridges RAGBits agent framework with CREWAI LLM management.
"""

from ragbits_integration.agents.base_agent import BaseWorkflowAgent
from ragbits_integration.agents.blue_team_agent import BlueTeamAgent
from ragbits_integration.agents.red_team_agent import RedTeamAgent
from ragbits_integration.agents.gold_team_agent import GoldTeamAgent

__all__ = [
    "BaseWorkflowAgent",
    "BlueTeamAgent",
    "RedTeamAgent",
    "GoldTeamAgent",
]
