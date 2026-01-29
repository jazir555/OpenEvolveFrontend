"""
OpenEvolve Knowledge Engine Integrations Package

This package provides enhanced capabilities by integrating existing AI knowledge graph projects
(DeepKE, Karate Club, kg-gen, OneKE, PAMI, NeuralKG, Causal-Learn, Lagrange-Mapper,
GlobalChem, Neuromancer) without modifying any core files.
"""

from .deepke_integration import DeepKEEnhancedExtractor
from .karateclub_integration import KarateClubGraphAnalyzer
from .kg_gen_integration import EnhancedKnowledgeGraphManager
try:
    from .pami_integration import PAMIPatternMiner
except ImportError:
    PAMIPatternMiner = None
try:
    from .neuralkg_integration import NeuralKGEmbedder
except ImportError:
    NeuralKGEmbedder = None
try:
    from .causal_learn_integration import CausalDiscoveryEngine
except ImportError:
    CausalDiscoveryEngine = None
try:
    from .lagrange_mapper_integration import LagrangeAttractorAnalyzer
except ImportError:
    LagrangeAttractorAnalyzer = None
try:
    from .global_chem_integration import GlobalChemKnowledgeAdapter
except ImportError:
    GlobalChemKnowledgeAdapter = None
try:
    from .neuromancer_integration import NeuromancerDynamicsModeler
except ImportError:
    NeuromancerDynamicsModeler = None

__all__ = [
    'DeepKEEnhancedExtractor',
    'KarateClubGraphAnalyzer', 
    'EnhancedKnowledgeGraphManager',
    'PAMIPatternMiner',
    'NeuralKGEmbedder',
    'CausalDiscoveryEngine',
    'LagrangeAttractorAnalyzer',
    'GlobalChemKnowledgeAdapter',
    'NeuromancerDynamicsModeler',
    'EnhancedKnowledgeIntegrator'
]


class AIKnowledgeGraphIntegrator:
    """
    Main integrator class that combines all AI knowledge graph integrations.
    
    This class provides a unified interface to leverage DeepKE, Karate Club,
    kg-gen, OneKE, PAMI, NeuralKG, Causal-Learn, Lagrange-Mapper, GlobalChem,
    and Neuromancer capabilities for enhanced knowledge extraction, graph analysis,
    and knowledge graph management.
    """
    
    def __init__(self):
        """Initialize all integration modules."""
        self.deepke_extractor = DeepKEEnhancedExtractor()
        self.karateclub_analyzer = KarateClubGraphAnalyzer()
        self.kg_gen_manager = EnhancedKnowledgeGraphManager()
        
        # Optional integrations
        self.pami_miner = PAMIPatternMiner() if PAMIPatternMiner else None
        self.neuralkg_embedder = NeuralKGEmbedder() if NeuralKGEmbedder else None
        self.causal_engine = CausalDiscoveryEngine() if CausalDiscoveryEngine else None
        self.lagrange_analyzer = LagrangeAttractorAnalyzer() if LagrangeAttractorAnalyzer else None
        self.global_chem_adapter = GlobalChemKnowledgeAdapter() if GlobalChemKnowledgeAdapter else None
        self.neuromancer_modeler = NeuromancerDynamicsModeler() if NeuromancerDynamicsModeler else None
    
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
    
    def mine_patterns_with_pami(self, data: dict, config: dict = None) -> dict:
        """
        Mine patterns using PAMI integration.
        
        Args:
            data: Data to mine patterns from
            config: Mining configuration
            
        Returns:
            Pattern mining results
        """
        if self.pami_miner is None:
            return {
                'status': 'error',
                'message': 'PAMI integration not available'
            }
        
        config = config or {}
        
        # Determine mining type
        mining_type = config.get('mining_type', 'frequent_patterns')
        
        if mining_type == 'frequent_patterns':
            return self.pami_miner.mine_frequent_patterns(
                transactions=data.get('transactions', []),
                min_support=config.get('min_support', 0.1),
                algorithm=config.get('algorithm', 'fpgrowth')
            )
        elif mining_type == 'sequences':
            return self.pami_miner.mine_sequences(
                sequences=data.get('sequences', []),
                min_support=config.get('min_support', 0.1)
            )
        elif mining_type == 'graph_patterns':
            return self.pami_miner.analyze_knowledge_graph_patterns(
                graph_data=data,
                min_support=config.get('min_support', 0.1)
            )
        elif mining_type == 'association_rules':
            return self.pami_miner.discover_association_rules(
                transactions=data.get('transactions', []),
                min_support=config.get('min_support', 0.1),
                min_confidence=config.get('min_confidence', 0.5)
            )
        else:
            return {
                'status': 'error',
                'message': f'Unknown mining type: {mining_type}'
            }
    
    def embed_knowledge_graph_with_neuralkg(
        self,
        triples: list,
        model: str = 'transe',
        config: dict = None
    ) -> dict:
        """
        Generate knowledge graph embeddings using NeuralKG.
        
        Args:
            triples: List of (head, relation, tail) triples
            model: Model to use ('transe', 'rotate', 'complex', etc.)
            config: Embedding configuration
            
        Returns:
            Embedding results
        """
        if self.neuralkg_embedder is None:
            return {
                'status': 'error',
                'message': 'NeuralKG integration not available'
            }
        
        config = config or {}
        
        return self.neuralkg_embedder.generate_embeddings(
            triples=triples,
            model_name=model,
            embedding_dim=config.get('embedding_dim', 100),
            epochs=config.get('epochs', 100)
        )
    
    def discover_causal_structure(
        self,
        data: 'np.ndarray',
        variable_names: list = None,
        algorithm: str = 'pc',
        config: dict = None
    ) -> dict:
        """
        Discover causal structure using causal-learn.
        
        Args:
            data: Data matrix (n_samples x n_variables)
            variable_names: Names of variables
            algorithm: Algorithm to use ('pc', 'fci', 'ges', 'lingam', etc.)
            config: Algorithm configuration
            
        Returns:
            Causal discovery results
        """
        if self.causal_engine is None:
            return {
                'status': 'error',
                'message': 'Causal-learn integration not available'
            }
        
        config = config or {}
        
        return self.causal_engine.discover_causal_structure(
            data=data,
            variable_names=variable_names,
            algorithm=algorithm,
            alpha=config.get('alpha', 0.05),
            independence_test=config.get('independence_test', 'fisherz'),
            **{k: v for k, v in config.items() if k not in ['alpha', 'independence_test']}
        )
    
    def analyze_attractor_landscape(
        self,
        embeddings: 'np.ndarray',
        labels: list = None,
        config: dict = None
    ) -> dict:
        """
        Analyze attractor landscape using lagrange-mapper.
        
        Args:
            embeddings: Embedding matrix
            labels: Optional labels for embeddings
            config: Analysis configuration
            
        Returns:
            Landscape analysis results
        """
        if self.lagrange_analyzer is None:
            return {
                'status': 'error',
                'message': 'Lagrange-mapper integration not available'
            }
        
        config = config or {}
        
        return self.lagrange_analyzer.analyze_embedding_landscape(
            embeddings=embeddings,
            labels=labels,
            n_clusters=config.get('n_clusters', 8),
            reduction_method=config.get('reduction_method', 'pca'),
            reduction_dims=config.get('reduction_dims', 2)
        )
    
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
                },
                'pattern_mining': {
                    'enabled': True,
                    'pami_config': {}
                },
                'embedding': {
                    'enabled': True,
                    'neuralkg_config': {}
                }
            }
            
            results = {}
            
            # Step 1: Knowledge extraction
            if config['extraction']['enabled']:
                extraction_result = self.extract_knowledge_with_deepke(
                    text, config['extraction'].get('deepke_config')
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
                graph_data = self._convert_artifacts_to_graph(knowledge_artifacts)
                analysis_result = self.analyze_graph_with_karateclub(
                    graph_data, config['analysis'].get('karateclub_config')
                )
                results['analysis'] = analysis_result
            
            # Step 3: Knowledge graph management
            if config['graph_management']['enabled'] and knowledge_artifacts:
                management_result = self.manage_knowledge_graph(
                    knowledge_artifacts, config['graph_management'].get('kg_gen_config')
                )
                results['graph_management'] = management_result
            
            # Step 4: Pattern mining (if enabled and we have artifacts)
            if config.get('pattern_mining', {}).get('enabled') and knowledge_artifacts:
                # Create transactions from knowledge artifacts
                transactions = self._artifacts_to_transactions(knowledge_artifacts)
                pattern_result = self.mine_patterns_with_pami(
                    {'transactions': transactions},
                    config['pattern_mining'].get('pami_config', {'mining_type': 'frequent_patterns'})
                )
                results['pattern_mining'] = pattern_result
            
            # Step 5: Generate embeddings (if enabled and we have artifacts)
            if config.get('embedding', {}).get('enabled') and knowledge_artifacts:
                triples = self._artifacts_to_triples(knowledge_artifacts)
                if triples:
                    embedding_result = self.embed_knowledge_graph_with_neuralkg(
                        triples,
                        config['embedding'].get('model', 'transe'),
                        config['embedding'].get('neuralkg_config', {})
                    )
                    results['embedding'] = embedding_result
            
            return {
                'status': 'success',
                'pipeline_results': results,
                'config_used': config,
                'metadata': {
                    'pipeline_timestamp': self._get_current_timestamp(),
                    'knowledge_engine_version': '5x_enhanced_with_full_ai_integration'
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
    
    def _artifacts_to_transactions(self, knowledge_artifacts: list) -> list:
        """Convert knowledge artifacts to transaction format for pattern mining."""
        transactions = []
        
        for artifact in knowledge_artifacts:
            transaction = []
            
            # Add type
            if artifact.get('knowledge_type'):
                transaction.append(f"type:{artifact['knowledge_type']}")
            
            # Add source
            if artifact.get('source'):
                transaction.append(f"source:{artifact['source']}")
            
            # Add triple components
            if artifact.get('subject'):
                transaction.append(f"subject:{artifact['subject']}")
            if artifact.get('predicate'):
                transaction.append(f"predicate:{artifact['predicate']}")
            if artifact.get('object'):
                transaction.append(f"object:{artifact['object']}")
            
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    def _artifacts_to_triples(self, knowledge_artifacts: list) -> list:
        """Convert knowledge artifacts to triple format."""
        triples = []
        
        for artifact in knowledge_artifacts:
            if (artifact.get('knowledge_type') == 'triple' and
                artifact.get('subject') and
                artifact.get('predicate') and
                artifact.get('object')):
                triples.append((
                    artifact['subject'],
                    artifact['predicate'],
                    artifact['object']
                ))
        
        return triples
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def recognize_chemical_entities(self, text: str) -> dict:
        """
        Recognize chemical entities using GlobalChem.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Chemical entity recognition results
        """
        if self.global_chem_adapter is None:
            return {
                'status': 'error',
                'message': 'GlobalChem integration not available'
            }
        
        entities = self.global_chem_adapter.recognize_chemical_entities(text)
        return {
            'status': 'success',
            'entities': entities,
            'count': len(entities)
        }
    
    def get_chemical_info(self, name: str) -> dict:
        """
        Get chemical information using GlobalChem.
        
        Args:
            name: Chemical name
            
        Returns:
            Chemical information
        """
        if self.global_chem_adapter is None:
            return {
                'status': 'error',
                'message': 'GlobalChem integration not available'
            }
        
        info = self.global_chem_adapter.get_chemical_by_name(name)
        if info:
            return {
                'status': 'success',
                'chemical': info
            }
        return {
            'status': 'error',
            'message': f'Chemical {name} not found'
        }
    
    def train_dynamics_model(
        self,
        time_series_data,
        time_points,
        config: dict = None
    ) -> dict:
        """
        Train neural dynamics model using Neuromancer.
        
        Args:
            time_series_data: Time series data
            time_points: Time points
            config: Training configuration
            
        Returns:
            Training results
        """
        if self.neuromancer_modeler is None:
            return {
                'status': 'error',
                'message': 'Neuromancer integration not available'
            }
        
        return self.neuromancer_modeler.train_neural_ode(
            time_series_data=time_series_data,
            time_points=time_points,
            config=config
        )
    
    def predict_dynamics(
        self,
        initial_state,
        time_horizon: int,
        model_id: str = None
    ) -> dict:
        """
        Predict dynamics using Neuromancer model.
        
        Args:
            initial_state: Initial state
            time_horizon: Time horizon
            model_id: Model ID
            
        Returns:
            Prediction results
        """
        if self.neuromancer_modeler is None:
            return {
                'status': 'error',
                'message': 'Neuromancer integration not available'
            }
        
        return self.neuromancer_modeler.predict_dynamics(
            initial_state=initial_state,
            time_horizon=time_horizon,
            model_id=model_id
        )
    
    def get_integration_status(self) -> dict:
        """Get the availability status of all integrations."""
        return {
            'deepke': self.deepke_extractor.is_available(),
            'karateclub': self.karateclub_analyzer.is_available(),
            'kg_gen': self.kg_gen_manager.is_kg_gen_available(),
            'oneke': self.kg_gen_manager.is_oneke_available(),
            'pami': self.pami_miner.is_available() if self.pami_miner else False,
            'neuralkg': self.neuralkg_embedder.is_available() if self.neuralkg_embedder else False,
            'causal_learn': self.causal_engine.is_available() if self.causal_engine else False,
            'lagrange_mapper': self.lagrange_analyzer.is_available() if self.lagrange_analyzer else False,
            'global_chem': self.global_chem_adapter.is_available() if self.global_chem_adapter else False,
            'neuromancer': self.neuromancer_modeler.is_available() if self.neuromancer_modeler else False,
            'timestamp': self._get_current_timestamp()
        }


# Alias for backward compatibility
EnhancedKnowledgeIntegrator = AIKnowledgeGraphIntegrator


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
