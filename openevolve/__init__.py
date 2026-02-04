"""
Shim package to expose core OpenEvolve package from core-projects/openevolve.

This keeps integration tests that import `openevolve.*` working without
requiring an editable install.
"""

from __future__ import annotations

import os
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_core_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core-projects", "openevolve", "openevolve")
)

if os.path.isdir(_core_path) and _core_path not in __path__:
    __path__.append(_core_path)

# Provide alias to root knowledge_engine for legacy imports
try:  # pragma: no cover - runtime alias for integration compatibility
    import importlib
    import sys

    import knowledge_engine as _knowledge_engine

    sys.modules.setdefault("openevolve.knowledge_engine", _knowledge_engine)
    try:
        sys.modules.setdefault(
            "openevolve.knowledge_engine.integrations",
            importlib.import_module("knowledge_engine.integrations"),
        )
    except Exception:
        pass
except Exception:
    pass
