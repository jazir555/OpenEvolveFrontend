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

# Import KnowledgeEngine from orchestration.py for backward compatibility
# Use importlib to avoid circular imports
try:
    import importlib.util
    import sys
    from pathlib import Path
    
    # Check if orchestration module already loaded
    if 'orchestration' in sys.modules:
        KnowledgeEngine = sys.modules['orchestration'].KnowledgeEngine
    else:
        # Load orchestration.py directly
        _orch_path = Path(__file__).parent.parent / "orchestration.py"
        if _orch_path.exists():
            _spec = importlib.util.spec_from_file_location("orchestration", _orch_path)
            _orch_module = importlib.util.module_from_spec(_spec)
            sys.modules['orchestration'] = _orch_module
            _spec.loader.exec_module(_orch_module)
            KnowledgeEngine = _orch_module.KnowledgeEngine
            del _spec
        else:
            raise ImportError("orchestration.py not found")
    
    __all__ = [
        'KnowledgeState',
        'EntityKnowledgeGraph',
        'Entity',
        'Relationship',
        'KnowledgeTriple',
        'StateSnapshot',
        'KnowledgeEngine'
    ]
except Exception:
    __all__ = [
        'KnowledgeState',
        'EntityKnowledgeGraph',
        'Entity',
        'Relationship',
        'KnowledgeTriple',
        'StateSnapshot'
    ]
