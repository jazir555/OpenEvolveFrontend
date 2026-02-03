#!/usr/bin/env python3
"""
Case 2 Hierarchical Extraction Package
Multi-type documents with relationships extraction system
"""

from .case2_core import (
    ExtractionStrategy, ExtractionStage, RelationshipMapping, 
    RelationshipType, ExtractionStageType, StageExtractor, 
    HierarchicalExtractionResult, Case2Config, DocumentClassifier
)

from .case2_strategy_generator import Case2StrategyGenerator
from .case2_model_generator import Case2ModelGenerator
from .case2_extractor import Case2SequentialExtractor, Case2RelationshipManager
from .case2_main import Case2Orchestrator

__version__ = "1.0.0"
__author__ = "Knowledge Extraction Agent"

__all__ = [
    # Core data structures
    'ExtractionStrategy',
    'ExtractionStage', 
    'RelationshipMapping',
    'RelationshipType',
    'ExtractionStageType',
    'StageExtractor',
    'HierarchicalExtractionResult',
    'Case2Config',
    'DocumentClassifier',
    
    # Main components
    'Case2StrategyGenerator',
    'Case2ModelGenerator', 
    'Case2SequentialExtractor',
    'Case2RelationshipManager',
    'Case2Orchestrator'
]
