"""
DeepKE Bridge for OpenEvolve Workflow Knowledge Extractor

This module bridges DeepKE's entity and relation extraction capabilities with
OpenEvolve's workflow knowledge extractor.
"""

import logging
from typing import Dict, Any, List, Optional

from .adapter import DeepKEAdapter, DeepKEExtractionResult, ExtractionTask

logger = logging.getLogger(__name__)


class DeepKEBridge:
    """
    Bridge between DeepKE and OpenEvolve knowledge extraction.
    
    This class provides convenient methods for extracting knowledge from
    workflow executions using DeepKE's NER and RE capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DeepKE bridge.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.adapter = DeepKEAdapter(self.config)
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the bridge and adapter.
        
        Returns:
            True if initialization successful
        """
        try:
            self._initialized = self.adapter.initialize()
            return self._initialized
        except Exception as e:
            logger.error(f"Failed to initialize DeepKE bridge: {e}")
            self._initialized = False
            return False
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relations from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities and relations
        """
        if not self._initialized:
            self.initialize()
        
        # Extract entities and relations
        result = self.adapter.extract_entities_and_relations(text)
        
        return {
            'entities': [
                {
                    'text': e.text,
                    'type': e.entity_type,
                    'start': e.start_pos,
                    'end': e.end_pos,
                    'confidence': e.confidence
                }
                for e in result.entities
            ],
            'relations': [
                {
                    'head': r.head_entity,
                    'tail': r.tail_entity,
                    'type': r.relation_type,
                    'confidence': r.confidence
                }
                for r in result.relations
            ],
            'success': result.success,
            'error': result.error_message if not result.success else None
        }
    
    def extract_from_workflow(
        self, 
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract knowledge from workflow execution data.
        
        Args:
            workflow_data: Workflow state dictionary
            
        Returns:
            Dictionary with extracted knowledge
        """
        # Combine workflow text fields
        text_parts = []
        
        if 'problem_statement' in workflow_data:
            text_parts.append(f"Problem: {workflow_data['problem_statement']}")
        
        if 'final_solution' in workflow_data:
            text_parts.append(f"Solution: {workflow_data['final_solution']}")
        
        if 'decomposition_plan' in workflow_data:
            text_parts.append(f"Decomposition: {workflow_data['decomposition_plan']}")
        
        text = '\n\n'.join(text_parts)
        
        return self.extract_from_text(text)
    
    def extract_technical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract technical entities (algorithms, methods, systems).
        
        Args:
            text: Input text
            
        Returns:
            List of technical entities
        """
        result = self.extract_from_text(text)
        
        # Filter for technical entity types
        technical_types = {'TECH', 'CONCEPT', 'ALGORITHM', 'METHOD', 'SYSTEM'}
        
        return [
            entity for entity in result['entities']
            if entity['type'] in technical_types
        ]
    
    def extract_solution_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract solution-related patterns from text.
        
        Args:
            text: Input text
            
        Returns:
            List of solution patterns
        """
        result = self.extract_from_text(text)
        
        patterns = []
        
        # Look for solution-related relations
        solution_relations = {'USES', 'IMPLEMENTS', 'SOLVES', 'ADDRESSES'}
        
        for relation in result['relations']:
            if relation['type'] in solution_relations:
                patterns.append({
                    'pattern_type': relation['type'],
                    'components': [relation['head'], relation['tail']],
                    'confidence': relation['confidence']
                })
        
        return patterns
    
    def batch_extract(
        self, 
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch extract from multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of extraction results
        """
        results = []
        for text in texts:
            results.append(self.extract_from_text(text))
        return results
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate DeepKE integration.
        
        Returns:
            Validation results
        """
        checks = []
        
        # Check adapter availability
        is_available = self.adapter.is_available()
        checks.append({
            'name': 'deepke_available',
            'status': 'passed' if is_available else 'failed',
            'message': 'DeepKE is installed' if is_available else 'DeepKE not installed'
        })
        
        # Check initialization
        checks.append({
            'name': 'adapter_initialized',
            'status': 'passed' if self._initialized else 'failed',
            'message': 'Adapter is initialized' if self._initialized else 'Adapter not initialized'
        })
        
        # Get stats
        stats = self.adapter.get_stats()
        
        is_valid = all(c['status'] == 'passed' for c in checks)
        
        return {
            'is_valid': is_valid,
            'checks': checks,
            'stats': stats
        }
    
    def shutdown(self):
        """Shutdown the bridge."""
        self._initialized = False
        logger.info("DeepKE bridge shutdown")


# Convenience functions

def create_deepke_bridge(config: Optional[Dict[str, Any]] = None) -> DeepKEBridge:
    """
    Create and initialize DeepKE bridge.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized DeepKEBridge
    """
    bridge = DeepKEBridge(config)
    bridge.initialize()
    return bridge


def extract_knowledge(text: str) -> Dict[str, Any]:
    """
    Quick extraction function.
    
    Args:
        text: Input text
        
    Returns:
        Extracted knowledge
    """
    bridge = create_deepke_bridge()
    result = bridge.extract_from_text(text)
    bridge.shutdown()
    return result
