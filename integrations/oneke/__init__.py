"""
OneKE Integration for OpenEvolve

This package provides schema-guided information extraction capabilities
using the OneKE framework. It fills GAP-2 (Physics Domain Knowledge) and
enhances GAP-10 (Knowledge Extraction).

Components:
- adapter: OneKEAdapter implementing ExtractionInterface
- bridge: OneKEBridge for workflow knowledge extraction
- schemas: YAML schema definitions for physics, chemistry, and relations

Usage:
    from integrations.oneke import OneKEAdapter, OneKEBridge

    # Using adapter directly
    adapter = OneKEAdapter()
    await adapter.initialize()
    result = await adapter.extract_ner(text, schema)

    # Using bridge for workflows
    bridge = OneKEBridge()
    await bridge.initialize()
    knowledge = await bridge.extract_physics_knowledge(workflow)

Repository: https://github.com/zjunlp/OneKE
"""

from .adapter import OneKEAdapter
from .bridge import OneKEBridge, create_oneke_bridge, extract_domain_knowledge

__version__ = "0.1.0"
__all__ = [
    "OneKEAdapter",
    "OneKEBridge",
    "create_oneke_bridge",
    "extract_domain_knowledge",
]
