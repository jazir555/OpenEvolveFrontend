"""
DeepKE Integration for OpenEvolve Knowledge Extraction

This module provides integration with DeepKE (Deep Learning based Knowledge Extraction)
for entity and relation extraction from text.

DeepKE Repository: https://github.com/zjunlp/DeepKE
"""

from .adapter import DeepKEAdapter
from .bridge import DeepKEBridge

__all__ = ['DeepKEAdapter', 'DeepKEBridge']
