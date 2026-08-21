"""Re-export shim for the flat ``engines/other`` sys.path layout.

The ``SteerCrewAIWorkflowBridge`` implementation lives in
``integrations/other/steer_crewai_bridge.py``, which is not on the conventional
flat ``engines/other`` path. This thin module preserves the public name used by
``engines/orchestration/tripartite_production.py`` without duplicating code.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_OTHER = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "integrations", "other")
)
if _OTHER not in sys.path:
    sys.path.insert(0, _OTHER)

_REAL = os.path.join(_OTHER, "steer_crewai_bridge.py")
_spec = importlib.util.spec_from_file_location("_steer_crewai_bridge_impl", _REAL)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

SteerCrewAIWorkflowBridge = _module.SteerCrewAIWorkflowBridge
