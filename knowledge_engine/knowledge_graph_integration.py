"""
Knowledge Graph Integration for OpenEvolve Knowledge Engine

This module integrates the existing AI knowledge graph capabilities into the enhanced knowledge engine.
It leverages the ai-knowledge-graph project for advanced knowledge representation and reasoning.
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add the ai-knowledge-graph directory to Python path
ai_kg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai-knowledge-graph', 'src'))
if ai_kg_path not in sys.path:
    sys.path.insert(0, ai_kg_path)

try:
    from knowledge_graph.main import process_with_llm
    from knowledge_graph.entity_standardization import standardize_entities, infer_relationships
    from knowledge_graph.visualization import visualize_knowledge_graph
    from knowledge_graph.config import load_config
    from knowledge_graph.text_utils import chunk_text
    from knowledge_graph.prompts import prompt_factory
    
    # Import knowledge engine components
    from knowledge_extractor import KnowledgeArtifact
    from advanced_knowledge_extractor import AdvancedKnowledgeExtractor
    
    kg_available = True
except ImportError as e:
    logging.warning(f"Knowledge graph modules not available: {e}")
    kg_available = False
    from knowledge_extractor import KnowledgeArtifact

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraphIntegrator:
    """
    Knowledge Graph Integrator for OpenEvolve Knowledge Engine.
    
    This class integrates knowledge graph capabilities with the knowledge engine by:
    - Converting knowledge artifacts to knowledge graph triples
    - Enhancing artifacts with knowledge graph relationships
    - Providing visualization capabilities
    - Enabling semantic reasoning and inference
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize knowledge graph integrator"""
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(__name__)
        self.kg_config = self._load_kg_config()
        self.graph_data = {
            'nodes': [],
            'edges': [],
            'triples': []
        }
        self.integration_stats = {
            'artifacts_processed': 0,
            'triples_generated': 0,
            'relationships_inferred': 0,
            'graph_visualizations': 0
        }
        
        self.logger.info("Knowledge graph integrator initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'knowledge_graph': {
                'enabled': kg_available,
                'max_triples_per_artifact': 10,
                'relationship_inference': True,
                'visualization_enabled': True
            },
            'integration': {
                'auto_update': True,
                'quality_threshold': 0.75
            }
        }
    
    def _load_kg_config(self) -> Optional[Dict[str, Any]]:
        """Load knowledge graph configuration"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'ai-knowledge-graph', 'config.toml')
            return load_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load knowledge graph config: {str(e)}")
            return None
    
    def integrate_knowledge_artifacts(self, artifacts: List[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Integrate knowledge artifacts with knowledge graph capabilities.
        
        Args:
            artifacts: List of knowledge artifacts to integrate
            
        Returns:
            List of enhanced knowledge artifacts with graph integration
        """
        if not self.config['knowledge_graph']['enabled'] or not kg_available:
            logger.warning("Knowledge graph integration disabled or not available")
            return artifacts
        
        start_time = datetime.now()
        enhanced_artifacts = []
        
        for artifact in artifacts:
            try:
                # Convert artifact to knowledge graph triples
                triples = self._artifact_to_triples(artifact)
                
                if triples:
                    # Standardize entities
                    standardized_triples = standardize_entities(triples, self.kg_config)
                    
                    # Infer additional relationships
                    if self.config['knowledge_graph']['relationship_inference']:
                        inferred_triples = infer_relationships(standardized_triples, self.kg_config)
                        standardized_triples.extend(inferred_triples)
                    
                    # Enhance artifact with graph data
                    enhanced_artifact = self._enhance_artifact_with_graph(artifact, standardized_triples)
                    enhanced_artifacts.append(enhanced_artifact)
                    
                    # Update graph data
                    self._update_graph_data(standardized_triples)
                    
                    # Update statistics
                    self.integration_stats['artifacts_processed'] += 1
                    self.integration_stats['triples_generated'] += len(standardized_triples)
                    self.integration_stats['relationships_inferred'] += len(inferred_triples) if inferred_triples else 0
                else:
                    enhanced_artifacts.append(artifact)
                    
            except Exception as e:
                logger.error(f"Failed to integrate artifact {artifact.id}: {str(e)}")
                enhanced_artifacts.append(artifact)
        
        integration_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Knowledge graph integration completed for {len(artifacts)} artifacts")
        logger.info(f"  - Artifacts processed: {self.integration_stats['artifacts_processed']}")
        logger.info(f"  - Triples generated: {self.integration_stats['triples_generated']}")
        logger.info(f"  - Integration time: {integration_time:.3f}s")
        
        return enhanced_artifacts
    
    def _artifact_to_triples(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Convert knowledge artifact to knowledge graph triples"""
        triples = []
        
        try:
            # Extract basic artifact information
            artifact_id = artifact.id
            artifact_type = artifact.artifact_type
            
            # Create base triple for the artifact itself
            triples.append({
                'subject': artifact_id,
                'predicate': 'is_a',
                'object': artifact_type,
                'source': 'knowledge_artifact',
                'confidence': 0.95
            })
            
            # Add domain and problem type relationships
            if artifact.domain:
                triples.append({
                    'subject': artifact_id,
                    'predicate': 'has_domain',
                    'object': artifact.domain,
                    'source': 'knowledge_artifact',
                    'confidence': 0.90
                })
            
            if artifact.problem_type:
                triples.append({
                    'subject': artifact_id,
                    'predicate': 'addresses_problem',
                    'object': artifact.problem_type,
                    'source': 'knowledge_artifact',
                    'confidence': 0.85
                })
            
            # Extract relationships from content
            content_triples = self._extract_content_triples(artifact)
            triples.extend(content_triples)
            
            # Extract relationships from metadata
            metadata_triples = self._extract_metadata_triples(artifact)
            triples.extend(metadata_triples)
            
            # Extract relationships from NLP analysis if available
            if 'nlp_analysis' in artifact.metadata:
                nlp_triples = self._extract_nlp_triples(artifact)
                triples.extend(nlp_triples)
            
            # Extract relationships from ML analysis if available
            if 'ml_analysis' in artifact.metadata:
                ml_triples = self._extract_ml_triples(artifact)
                triples.extend(ml_triples)
            
            # Limit triples per artifact
            if len(triples) > self.config['knowledge_graph']['max_triples_per_artifact']:
                triples = triples[:self.config['knowledge_graph']['max_triples_per_artifact']]
                
        except Exception as e:
            logger.error(f"Failed to convert artifact {artifact.id} to triples: {str(e)}")
        
        return triples
    
    def _extract_content_triples(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Extract triples from artifact content"""
        triples = []
        
        try:
            content = artifact.content
            
            # Extract solution-specific relationships
            if artifact.artifact_type == 'solution_pattern':
                if 'solution_approach' in content:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'uses_approach',
                        'object': content['solution_approach'],
                        'source': 'content_analysis',
                        'confidence': 0.80
                    })
                
                if 'pattern_type' in content:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'implements_pattern',
                        'object': content['pattern_type'],
                        'source': 'content_analysis',
                        'confidence': 0.85
                    })
            
            # Extract critique-specific relationships
            elif artifact.artifact_type == 'critique_insight':
                if 'issue_type' in content:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'identifies_issue',
                        'object': content['issue_type'],
                        'source': 'content_analysis',
                        'confidence': 0.82
                    })
                
                if 'root_cause' in content:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'has_root_cause',
                        'object': content['root_cause'],
                        'source': 'content_analysis',
                        'confidence': 0.78
                    })
            
            # Extract team-specific relationships
            elif artifact.artifact_type == 'team_performance':
                if 'team_name' in content:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'evaluates_team',
                        'object': content['team_name'],
                        'source': 'content_analysis',
                        'confidence': 0.88
                    })
            
            # Extract entities as nodes
            if 'entities' in content:
                for entity in content['entities']:
                    triples.append({
                        'subject': artifact.id,
                        'predicate': 'mentions_entity',
                        'object': entity.get('text', ''),
                        'source': 'entity_extraction',
                        'confidence': entity.get('confidence', 0.75)
                    })
                    
                    # Add entity type if available
                    if entity.get('label'):
                        triples.append({
                            'subject': entity.get('text', ''),
                            'predicate': 'is_a',
                            'object': entity.get('label', ''),
                            'source': 'entity_typing',
                            'confidence': 0.80
                        })
            
        except Exception as e:
            logger.error(f"Failed to extract content triples: {str(e)}")
        
        return triples
    
    def _extract_metadata_triples(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Extract triples from artifact metadata"""
        triples = []
        
        try:
            metadata = artifact.metadata
            
            # Extract quality assessment relationships
            if 'quality_assessment' in metadata:
                quality = metadata['quality_assessment']
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'has_quality_score',
                    'object': str(quality.get('overall_quality', 0.0)),
                    'source': 'quality_assessment',
                    'confidence': 0.90
                })
                
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'has_quality_category',
                    'object': quality.get('quality_category', 'unknown'),
                    'source': 'quality_assessment',
                    'confidence': 0.85
                })
            
            # Extract validation relationships
            if 'validation' in metadata:
                validation = metadata['validation']
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'has_validation_status',
                    'object': validation.get('status', 'unknown'),
                    'source': 'validation',
                    'confidence': 0.90
                })
            
            # Extract processing metadata relationships
            if 'processing_metadata' in metadata:
                processing = metadata['processing_metadata']
                for stage, data in processing.items():
                    triples.append({
                        'subject': artifact.id,
                        'predicate': f'has_{stage}_status',
                        'object': data.get('status', 'unknown'),
                        'source': 'processing_metadata',
                        'confidence': 0.75
                    })
            
        except Exception as e:
            logger.error(f"Failed to extract metadata triples: {str(e)}")
        
        return triples
    
    def _extract_nlp_triples(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Extract triples from NLP analysis"""
        triples = []
        
        try:
            nlp_data = artifact.metadata['nlp_analysis']
            
            # Extract sentiment relationships
            sentiment = nlp_data.get('sentiment', {})
            if sentiment.get('sentiment'):
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'has_sentiment',
                    'object': sentiment['sentiment'],
                    'source': 'nlp_analysis',
                    'confidence': 0.80
                })
            
            # Extract key phrases as related concepts
            for phrase in nlp_data.get('key_phrases', []):
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'related_to_concept',
                    'object': phrase,
                    'source': 'nlp_analysis',
                    'confidence': 0.75
                })
            
        except Exception as e:
            logger.error(f"Failed to extract NLP triples: {str(e)}")
        
        return triples
    
    def _extract_ml_triples(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Extract triples from ML analysis"""
        triples = []
        
        try:
            ml_data = artifact.metadata['ml_analysis']
            
            # Extract pattern relationships
            for pattern in ml_data.get('patterns', []):
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'exhibits_pattern',
                    'object': pattern['pattern_type'],
                    'source': 'ml_analysis',
                    'confidence': pattern.get('confidence', 0.70)
                })
            
            # Extract topic relationships
            for topic in ml_data.get('topics', []):
                triples.append({
                    'subject': artifact.id,
                    'predicate': 'related_to_topic',
                    'object': topic['topic'],
                    'source': 'ml_analysis',
                    'confidence': topic.get('confidence', 0.70)
                })
            
        except Exception as e:
            logger.error(f"Failed to extract ML triples: {str(e)}")
        
        return triples
    
    def _enhance_artifact_with_graph(self, artifact: KnowledgeArtifact, triples: List[Dict[str, Any]]) -> KnowledgeArtifact:
        """Enhance artifact with knowledge graph data"""
        enhanced = KnowledgeArtifact(**artifact.to_dict())
        
        try:
            # Add knowledge graph metadata
            if 'knowledge_graph' not in enhanced.metadata:
                enhanced.metadata['knowledge_graph'] = {}
            
            enhanced.metadata['knowledge_graph']['triples'] = triples
            enhanced.metadata['knowledge_graph']['node_count'] = len(self._get_unique_nodes(triples))
            enhanced.metadata['knowledge_graph']['edge_count'] = len(triples)
            enhanced.metadata['knowledge_graph']['integration_timestamp'] = datetime.now().isoformat()
            
            # Update artifact quality based on graph integration
            graph_quality = min(0.15, len(triples) * 0.01)  # Max 0.15 boost
            enhanced.source_quality = min(1.0, enhanced.source_quality + graph_quality)
            
        except Exception as e:
            logger.error(f"Failed to enhance artifact with graph data: {str(e)}")
        
        return enhanced
    
    def _get_unique_nodes(self, triples: List[Dict[str, Any]]) -> Set[str]:
        """Get unique nodes from triples"""
        nodes = set()
        for triple in triples:
            nodes.add(triple['subject'])
            nodes.add(triple['object'])
        return nodes
    
    def _update_graph_data(self, triples: List[Dict[str, Any]]):
        """Update the overall graph data structure"""
        try:
            # Add triples to the graph
            self.graph_data['triples'].extend(triples)
            
            # Update nodes and edges for visualization
            nodes = set()
            edges = []
            
            for triple in triples:
                nodes.add(triple['subject'])
                nodes.add(triple['object'])
                edges.append({
                    'source': triple['subject'],
                    'target': triple['object'],
                    'label': triple['predicate'],
                    'confidence': triple.get('confidence', 0.7)
                })
            
            # Convert nodes to list format for visualization
            self.graph_data['nodes'] = [
                {'id': node, 'label': node, 'size': 20} 
                for node in nodes
            ]
            self.graph_data['edges'] = edges
            
        except Exception as e:
            logger.error(f"Failed to update graph data: {str(e)}")
    
    def visualize_knowledge_graph(self, output_file: str = 'knowledge_graph.html') -> bool:
        """Visualize the knowledge graph using the existing visualization module"""
        if not self.config['knowledge_graph']['visualization_enabled'] or not kg_available:
            logger.warning("Graph visualization disabled or not available")
            return False
        
        try:
            # Prepare data for visualization
            nodes = self.graph_data['nodes']
            edges = self.graph_data['edges']
            
            if not nodes or not edges:
                logger.warning("No graph data available for visualization")
                return False
            
            # Use the existing visualization function
            visualize_knowledge_graph(nodes, edges, output_file)
            
            self.integration_stats['graph_visualizations'] += 1
            logger.info(f"Knowledge graph visualized successfully: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to visualize knowledge graph: {str(e)}")
            return False
    
    def export_knowledge_graph(self, format: str = 'json', output_file: str = 'knowledge_graph.json') -> bool:
        """Export knowledge graph in various formats"""
        try:
            if format == 'json':
                with open(output_file, 'w') as f:
                    json.dump({
                        'nodes': self.graph_data['nodes'],
                        'edges': self.graph_data['edges'],
                        'triples': self.graph_data['triples'],
                        'metadata': {
                            'export_timestamp': datetime.now().isoformat(),
                            'artifact_count': self.integration_stats['artifacts_processed'],
                            'triple_count': self.integration_stats['triples_generated']
                        }
                    }, f, indent=2)
                
            elif format == 'csv':
                # Export triples as CSV
                with open(output_file, 'w') as f:
                    f.write('subject,predicate,object,source,confidence\n')
                    for triple in self.graph_data['triples']:
                        f.write(f"{triple['subject']},{triple['predicate']},{triple['object']},"
                               f"{triple.get('source', '')},{triple.get('confidence', 0.7)}\n")
            
            else:
                logger.warning(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Knowledge graph exported successfully: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export knowledge graph: {str(e)}")
            return False
    
    def query_knowledge_graph(self, query: str, query_type: str = 'simple') -> List[Dict[str, Any]]:
        """Query the knowledge graph for specific information"""
        results = []
        
        try:
            if query_type == 'simple':
                # Simple string matching query
                for triple in self.graph_data['triples']:
                    if (query.lower() in triple['subject'].lower() or
                        query.lower() in triple['predicate'].lower() or
                        query.lower() in triple['object'].lower()):
                        results.append(triple)
            
            elif query_type == 'subject':
                # Query by subject
                for triple in self.graph_data['triples']:
                    if triple['subject'].lower() == query.lower():
                        results.append(triple)
            
            elif query_type == 'object':
                # Query by object
                for triple in self.graph_data['triples']:
                    if triple['object'].lower() == query.lower():
                        results.append(triple)
            
            elif query_type == 'predicate':
                # Query by predicate
                for triple in self.graph_data['triples']:
                    if triple['predicate'].lower() == query.lower():
                        results.append(triple)
            
        except Exception as e:
            logger.error(f"Failed to query knowledge graph: {str(e)}")
        
        return results
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get knowledge graph integration statistics"""
        stats = {
            'artifacts_processed': self.integration_stats['artifacts_processed'],
            'triples_generated': self.integration_stats['triples_generated'],
            'relationships_inferred': self.integration_stats['relationships_inferred'],
            'graph_visualizations': self.integration_stats['graph_visualizations'],
            'node_count': len(self.graph_data['nodes']),
            'edge_count': len(self.graph_data['edges']),
            'triple_count': len(self.graph_data['triples']),
            'average_triples_per_artifact': (self.integration_stats['triples_generated'] / 
                                           max(1, self.integration_stats['artifacts_processed']))
        }
        
        return stats
    
    def reset_integration_stats(self):
        """Reset integration statistics"""
        self.integration_stats = {
            'artifacts_processed': 0,
            'triples_generated': 0,
            'relationships_inferred': 0,
            'graph_visualizations': 0
        }
        
        # Clear graph data
        self.graph_data = {
            'nodes': [],
            'edges': [],
            'triples': []
        }
        
        logger.info("Knowledge graph integration statistics reset")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create knowledge graph integrator
    integrator = KnowledgeGraphIntegrator()
    
    # Create example knowledge artifacts
    from knowledge_extractor import KnowledgeArtifact
    
    example_artifacts = [
        KnowledgeArtifact(
            id='test_artifact_001',
            artifact_type='solution_pattern',
            content={
                'solution_id': 'sol_001',
                'problem_type': 'neural_network_optimization',
                'solution_approach': 'Advanced neural network architecture with transfer learning',
                'success_rate': 0.95,
                'complexity': 8,
                'pattern_type': 'neural_network_architecture',
                'entities': [
                    {'text': 'neural network', 'label': 'ML_MODEL', 'confidence': 0.95},
                    {'text': 'transfer learning', 'label': 'ML_TECHNIQUE', 'confidence': 0.90}
                ]
            },
            source_workflow_id='workflow_001',
            extraction_timestamp=datetime.now().timestamp(),
            domain='machine_learning',
            problem_type='model_optimization',
            effectiveness_score=0.95,
            metadata={
                'quality_assessment': {
                    'overall_quality': 0.92,
                    'quality_category': 'excellent'
                },
                'validation': {
                    'status': 'validated'
                },
                'nlp_analysis': {
                    'entities': [
                        {'text': 'neural network', 'label': 'ML_MODEL', 'confidence': 0.95},
                        {'text': 'transfer learning', 'label': 'ML_TECHNIQUE', 'confidence': 0.90}
                    ],
                    'sentiment': {'polarity': 0.3, 'subjectivity': 0.4, 'sentiment': 'positive'},
                    'key_phrases': ['advanced architecture', 'transfer learning', 'neural network']
                },
                'ml_analysis': {
                    'patterns': [
                        {'pattern_type': 'neural_network_architecture', 'confidence': 0.92},
                        {'pattern_type': 'transfer_learning', 'confidence': 0.88}
                    ],
                    'topics': [
                        {'topic_id': 'topic_1', 'topic': 'advanced architecture', 'confidence': 0.85},
                        {'topic_id': 'topic_2', 'topic': 'transfer learning', 'confidence': 0.80}
                    ]
                }
            }
        )
    ]
    
    print("Starting knowledge graph integration...")
    
    # Integrate artifacts with knowledge graph
    enhanced_artifacts = integrator.integrate_knowledge_artifacts(example_artifacts)
    
    print(f"\nIntegration Results:")
    print(f"  - Artifacts processed: {len(enhanced_artifacts)}")
    print(f"  - Triples generated: {integrator.integration_stats['triples_generated']}")
    print(f"  - Relationships inferred: {integrator.integration_stats['relationships_inferred']}")
    
    # Show enhanced artifact details
    for i, artifact in enumerate(enhanced_artifacts, 1):
        print(f"\nEnhanced Artifact {i}:")
        print(f"  - ID: {artifact.id}")
        print(f"  - Type: {artifact.artifact_type}")
        print(f"  - Quality: {artifact.calculate_quality_score():.2f}")
        
        if 'knowledge_graph' in artifact.metadata:
            kg_data = artifact.metadata['knowledge_graph']
            print(f"  - KG Triples: {len(kg_data['triples'])}")
            print(f"  - KG Nodes: {kg_data['node_count']}")
            print(f"  - KG Edges: {kg_data['edge_count']}")
            
            # Show some sample triples
            print(f"  - Sample Triples:")
            for triple in kg_data['triples'][:3]:  # Show first 3
                print(f"    {triple['subject']} -> {triple['predicate']} -> {triple['object']}")
    
    # Get integration statistics
    stats = integrator.get_integration_stats()
    print(f"\nIntegration Statistics:")
    print(f"  - Total triples: {stats['triple_count']}")
    print(f"  - Total nodes: {stats['node_count']}")
    print(f"  - Total edges: {stats['edge_count']}")
    print(f"  - Average triples per artifact: {stats['average_triples_per_artifact']:.1f}")
    
    # Export knowledge graph
    export_success = integrator.export_knowledge_graph('json', 'test_knowledge_graph.json')
    print(f"\nKnowledge graph export: {'success' if export_success else 'failed'}")
    
    # Query knowledge graph
    query_results = integrator.query_knowledge_graph('neural network')
    print(f"\nQuery Results for 'neural network' ({len(query_results)} results):")
    for result in query_results[:3]:  # Show first 3
        print(f"  {result['subject']} -> {result['predicate']} -> {result['object']}")
    
    print(f"\nKnowledge graph integration completed successfully!")