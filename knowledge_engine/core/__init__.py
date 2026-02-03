"""
Core Knowledge Engine Components

This package provides the core data structures and utilities for the Knowledge Engine.

Enhanced implementations following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify operations succeed
- IDEMPOTENCY: All operations safe to retry
- CONFIGURATION EXPLICITNESS: No magic defaults
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs

Version: 2.0.0
"""

# Import enhanced implementations
from .entity_knowledge_graph import EntityKnowledgeGraph, Entity, Relationship
from .knowledge_state import KnowledgeState, KnowledgeTriple, StateSnapshot

# Try to import KnowledgeEngine from orchestration module
# Use a placeholder if not available (to avoid import errors)
try:
    import importlib.util
    import sys
    from pathlib import Path
    
    # Try to load orchestration.py directly
    _orch_path = Path(__file__).parent.parent / "orchestration.py"
    if _orch_path.exists():
        _spec = importlib.util.spec_from_file_location("_orchestration_py", _orch_path)
        _orch_module = importlib.util.module_from_spec(_spec)
        # Don't add to sys.modules to avoid conflicts
        _spec.loader.exec_module(_orch_module)
        KnowledgeEngine = _orch_module.KnowledgeEngine
        del _spec, _orch_module
    else:
        raise ImportError("orchestration.py not found")
except Exception:
    # Create a placeholder base class
    class KnowledgeEngine:
        """Placeholder KnowledgeEngine base class."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "KnowledgeEngine requires orchestration.py to be available. "
                "Ensure all dependencies are installed."
            )

__all__ = [
    'KnowledgeState',
    'EntityKnowledgeGraph',
    'Entity',
    'Relationship',
    'KnowledgeTriple',
    'StateSnapshot',
    'KnowledgeEngine'
]
