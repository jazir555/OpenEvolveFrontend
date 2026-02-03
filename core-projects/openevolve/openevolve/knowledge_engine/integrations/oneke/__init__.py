"""
OneKE Integration Package for OpenEvolve Knowledge Engine

This package provides integration with OneKE knowledge extraction system,
enabling bilingual extraction and advanced quality enhancement.

Components:
- OneKEModelAdapter: Main adapter for OneKE models
- MultiTaskExtractionFramework: Framework for multitask extraction
- EnhancedOneKEBridge: Enhanced bridge with quality enhancement
- QualityEnhancer: Quality enhancement utilities
"""

try:
    from .model_adapter import OneKEModelAdapter
except ImportError:
    OneKEModelAdapter = None

try:
    from .multitask_framework import MultiTaskExtractionFramework
except ImportError:
    MultiTaskExtractionFramework = None

try:
    from .enhanced_bridge import EnhancedOneKEBridge
except ImportError:
    EnhancedOneKEBridge = None

try:
    from .quality_enhancer import QualityEnhancer
except ImportError:
    QualityEnhancer = None

__all__ = [
    'OneKEModelAdapter',
    'MultiTaskExtractionFramework',
    'EnhancedOneKEBridge',
    'QualityEnhancer'
]