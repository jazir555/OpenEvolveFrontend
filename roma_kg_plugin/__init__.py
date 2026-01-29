"""
ROMA Knowledge Graph Plugin Entry Point

This plugin extends ROMA with knowledge graph capabilities
without modifying ROMA core files (CLAUDE.md Air Gap principle).

Features:
- Knowledge graph visualization panel
- Analytics dashboard
- Interactive exploration
- Command extensions (8 custom commands)
- Menu extensions
"""

from .plugin import ROMAKnowledgeGraphPlugin

__version__ = "1.0.0"
__author__ = "OpenEvolve"
__description__ = "Knowledge Graph Integration for ROMA"

# Plugin instance (singleton)
_plugin_instance = None

def create_plugin():
    """
    Factory function to create plugin instance.

    This function is called by ROMA's plugin system at startup.
    Returns the singleton plugin instance.
    """
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = ROMAKnowledgeGraphPlugin()
    return _plugin_instance

def get_plugin():
    """
    Get the existing plugin instance.
    Returns None if plugin hasn't been created yet.
    """
    return _plugin_instance

# Export public API
__all__ = [
    "create_plugin",
    "get_plugin",
    "ROMAKnowledgeGraphPlugin",
    "__version__"
]
