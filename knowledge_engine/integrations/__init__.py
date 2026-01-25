"""
OpenEvolve Knowledge Engine Integrations Package

This package provides enhanced capabilities by integrating existing AI knowledge graph projects
(DeepKE, Karate Club, kg-gen, OneKE) without modifying any core files.
"""

from .deepke_integration import DeepKEEnhancedExtractor
from .karateclub_integration import KarateClubGraphAnalyzer
from .kg_gen_integration import EnhancedKnowledgeGraphManager

__all__ = [
    'DeepKEEnhancedExtractor',
    'KarateClubGraphAnalyzer', 
    'EnhancedKnowledgeGraphManager'
]

class AIKnowledgeGraphIntegrator:
    """
    Main integrator class that combines all AI knowledge graph integrations.
    
    This class provides a unified interface to leverage DeepKE, Karate Club,
    kg-gen, and OneKE capabilities for enhanced knowledge extraction,
    graph analysis, and knowledge graph management.
    """
    
    def __init__(self):
        """Initialize all integration modules."""
        self.deepke_extractor = DeepKEEnhancedExtractor()
        self.karateclub_analyzer = KarateClubGraphAnalyzer()
        self.kg_gen_manager = EnhancedKnowledgeGraphManager()
    
    def extract_knowledge_with_deepke(self, text: str, config: dict = None) -> dict:
        """
        Extract knowledge using DeepKE integration.
        
        Args:
            text: Input text to analyze
            config: Extraction configuration
            
        Returns:
            Knowledge extraction results
        """
        return self.deepke_extractor.extract_with_deepke(text, config)
    
    def analyze_graph_with_karateclub(self, graph_data: dict, config: dict = None) -> dict:
        """
        Analyze knowledge graph using Karate Club integration.
        
        Args:
            graph_data: Knowledge graph data
            config: Analysis configuration
            
        Returns:
            Graph analysis results
        """
        return self.karateclub_analyzer.analyze_graph(graph_data, config)
    
    def manage_knowledge_graph(self, knowledge_artifacts: list, config: dict = None) -> dict:
        """
        Manage knowledge graph lifecycle using kg-gen and OneKE integration.
        
        Args:
            knowledge_artifacts: List of knowledge artifacts
            config: Management configuration
            
        Returns:
            Knowledge graph management results
        """
        return self.kg_gen_manager.generate_and_store_knowledge_graph(knowledge_artifacts, config)
    
    def complete_knowledge_pipeline(self, text: str, pipeline_config: dict = None) -> dict:
        """
        Execute complete knowledge pipeline from extraction to graph management.
        
        This method provides a comprehensive workflow that integrates all
        AI knowledge graph capabilities in a single pipeline.
        
        Args:
            text: Input text to process
            pipeline_config: Complete pipeline configuration
            
        Returns:
            Complete pipeline results
        """
        try:
            # Set default pipeline configuration
            config = pipeline_config or {
                'extraction': {
                    'enabled': True,
                    'deepke_config': {}
                },
                'analysis': {
                    'enabled': True,
                    'karateclub_config': {}
                },
                'graph_management': {
                    'enabled': True,
                    'kg_gen_config': {}
                }
            }
            
            results = {}
            
            # Step 1: Knowledge extraction
            if config['extraction']['enabled']:
                extraction_result = self.extract_knowledge_with_deepke(
                    text, config['extraction']['deepke_config']
                )
                results['extraction'] = extraction_result
                
                # Convert extraction results to knowledge artifacts
                if extraction_result.get('status') == 'success':
                    knowledge_artifacts = self._convert_to_knowledge_artifacts(extraction_result)
                else:
                    knowledge_artifacts = []
            else:
                knowledge_artifacts = []
            
            # Step 2: Graph analysis (if we have artifacts and analysis is enabled)
            if config['analysis']['enabled'] and knowledge_artifacts:
                # Convert artifacts to graph format for analysis
                graph_data = self._convert_artifacts_to_graph(knowledge_artifacts)
                analysis_result = self.analyze_graph_with_karateclub(
                    graph_data, config['analysis']['karateclub_config']
                )
                results['analysis'] = analysis_result
            
            # Step 3: Knowledge graph management
            if config['graph_management']['enabled'] and knowledge_artifacts:
                management_result = self.manage_knowledge_graph(
                    knowledge_artifacts, config['graph_management']['kg_gen_config']
                )
                results['graph_management'] = management_result
            
            return {
                'status': 'success',
                'pipeline_results': results,
                'config_used': config,
                'metadata': {
                    'pipeline_timestamp': self._get_current_timestamp(),
                    'knowledge_engine_version': '5x_enhanced_with_ai_integration'
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Complete knowledge pipeline failed: {str(e)}',
                'pipeline_results': {}
            }
    
    def _convert_to_knowledge_artifacts(self, extraction_result: dict) -> list:
        """Convert extraction results to knowledge artifacts format."""
        artifacts = []
        
        for item in extraction_result.get('extracted_knowledge', []):
            artifact = {
                'source': 'deepke',
                'extraction_method': item['extractor'],
                'knowledge_type': item['type'],
                'confidence': item['confidence'],
                'raw_data': item['knowledge_item'],
                'metadata': {
                    'extractor': item['extractor'],
                    'type': item['type'],
                    'timestamp': self._get_current_timestamp()
                }
            }
            
            # Add specific fields based on knowledge type
            if item['type'] == 'triple' and isinstance(item['knowledge_item'], dict):
                artifact.update({
                    'subject': item['knowledge_item'].get('subject'),
                    'predicate': item['knowledge_item'].get('predicate'),
                    'object': item['knowledge_item'].get('object')
                })
            
            artifacts.append(artifact)
        
        return artifacts
    
    def _convert_artifacts_to_graph(self, knowledge_artifacts: list) -> dict:
        """Convert knowledge artifacts to graph format for analysis."""
        graph_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'source': 'knowledge_artifacts',
                'conversion_timestamp': self._get_current_timestamp()
            }
        }
        
        node_set = set()
        edge_set = set()
        
        for artifact in knowledge_artifacts:
            # Add nodes
            if artifact.get('knowledge_type') == 'triple':
                subject = artifact.get('subject')
                object_val = artifact.get('object')
                
                if subject and subject not in node_set:
                    node_set.add(subject)
                    graph_data['nodes'].append({
                        'id': subject,
                        'type': 'entity',
                        'source': artifact.get('source')
                    })
                
                if object_val and object_val not in node_set:
                    node_set.add(object_val)
                    graph_data['nodes'].append({
                        'id': object_val,
                        'type': 'entity',
                        'source': artifact.get('source')
                    })
                
                # Add edge
                if subject and object_val:
                    edge_key = f"{subject}|{artifact.get('predicate')}|{object_val}"
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        graph_data['edges'].append({
                            'source': subject,
                            'target': object_val,
                            'type': artifact.get('predicate'),
                            'relationship': artifact.get('predicate')
                        })
        
        return graph_data
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_integration_status(self) -> dict:
        """Get the availability status of all integrations."""
        return {
            'deepke': self.deepke_extractor.is_available(),
            'karateclub': self.karateclub_analyzer.is_available(),
            'kg_gen': self.kg_gen_manager.is_kg_gen_available(),
            'oneke': self.kg_gen_manager.is_oneke_available(),
            'timestamp': self._get_current_timestamp()
        }
# Mock classes for contract testing
try:
    from .kggen_pipeline_mock import KnowledgeGraph as MockKnowledgeGraph, UploadResult
    _MOCK_AVAILABLE = True
except ImportError:
    _MOCK_AVAILABLE = False

# For backward compatibility with tests
if _MOCK_AVAILABLE:
    KnowledgeGraph = MockKnowledgeGraph
else:
    # Fallback if mock not available
    from dataclasses import dataclass, field
    from typing import Dict, Any, List, Optional
    
    @dataclass
    class KnowledgeGraph:
        entities: Dict[str, Any] = field(default_factory=dict)
        relationships: List[Dict[str, Any]] = field(default_factory=list)
        
        def add_entity(self, name: str, attributes: Optional[Dict[str, Any]] = None):
            if name not in self.entities:
                self.entities[name] = attributes or {}
        
        def merge(self, other: 'KnowledgeGraph'):
            self.entities.update(other.entities)
            self.relationships.extend(other.relationships)
    
    @dataclass
    class UploadResult:
        success: bool
        entities_uploaded: int
        relationships_uploaded: int
        error: Optional[str] = None
        duration_ms: float = 0.0
