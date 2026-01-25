"""
ROMA Knowledge Graph Plugin - Commands

This package contains TUI command handlers for knowledge graph operations.
All commands use dependency injection - no direct coupling to ROMA core.
"""

from .kg_commands import KnowledgeGraphCommands

__all__ = [
    "KnowledgeGraphCommands"
]
