"""
RAGBits Integration for Decomposition Workflow

This package integrates RAGBits components into the Sovereign-Grade Decomposition Workflow,
providing:
- Real-time intermediary storage and retrieval during workflow execution
- Semantic search over historical solutions and patterns
- Agent coordination via A2A protocol
- Enhanced evaluation framework for gauntlets
- UI/CLI tools for review, monitoring, and knowledge exploration
"""

__version__ = "0.1.0"

# Phase 1: Storage & Retrieval
from ragbits_integration.intermediary_storage.storage_manager import IntermediaryStorageManager
from ragbits_integration.intermediary_storage.context_gatherer import ContextGatherer
from ragbits_integration.intermediary_storage.artifact_lifecycle import ArtifactLifecycleManager
from ragbits_integration.document_search.knowledge_retriever import RagbitsKnowledgeRetriever

# Phase 5: UI/CLI Integration
from ragbits_integration.ui_cli.cli.ragbits_cli import RAGBitsCLI
from ragbits_integration.ui_cli.interfaces.review_interface import ReviewInterface
from ragbits_integration.ui_cli.monitoring.dashboard import MonitoringDashboard
from ragbits_integration.ui_cli.exploration.knowledge_explorer import KnowledgeExplorer

__all__ = [
    # Phase 1
    "IntermediaryStorageManager",
    "ContextGatherer",
    "ArtifactLifecycleManager",
    "RagbitsKnowledgeRetriever",
    # Phase 5
    "RAGBitsCLI",
    "ReviewInterface",
    "MonitoringDashboard",
    "KnowledgeExplorer",
]
