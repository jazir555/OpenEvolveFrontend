"""
UI/CLI Integration Module

Phase 5: User interface enhancements, CLI tools, monitoring dashboards,
and interactive knowledge exploration.
"""

from ragbits_integration.ui_cli.cli.ragbits_cli import RAGBitsCLI
from ragbits_integration.ui_cli.interfaces.review_interface import ReviewInterface
from ragbits_integration.ui_cli.monitoring.dashboard import MonitoringDashboard
from ragbits_integration.ui_cli.exploration.knowledge_explorer import KnowledgeExplorer

__all__ = [
    "RAGBitsCLI",
    "ReviewInterface",
    "MonitoringDashboard",
    "KnowledgeExplorer"
]
