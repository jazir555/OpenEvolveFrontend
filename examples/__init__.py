"""
Examples package for OpenEvolve.
"""

# Import all example modules for easy access
try:
    from . import roma_decomposition_basic
    from . import roma_decomposition_advanced
    from . import enhanced_gauntlet_example
except ImportError:
    pass

__all__ = [
    'roma_decomposition_basic',
    'roma_decomposition_advanced', 
    'enhanced_gauntlet_example',
]
