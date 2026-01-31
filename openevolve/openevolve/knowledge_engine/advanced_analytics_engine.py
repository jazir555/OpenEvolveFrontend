"""
Advanced Analytics Engine for OpenEvolve Knowledge Engine

This module integrates Karate Club, PAMI, NeuralKG, Causal-Learn, and Lagrange-Mapper
directly into the Knowledge Engine core for enhanced graph analysis, pattern mining,
embedding generation, causal discovery, and topological analysis.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Verify component availability
- IDEMPOTENCY: Safe to retry
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Import knowledge engine components
try:
    from knowledge_extractor import KnowledgeExtractor, KnowledgeArtifact
except ImportError:
    KnowledgeExtractor = None
    KnowledgeArtifact = None

try:
    from knowledge_graph_integration import KnowledgeGraphIntegrator
except ImportError:
    KnowledgeGraphIntegrator = None

# Import integration modules
try:
    from .integrations import (
        KarateClubGraphAnalyzer,
        PAMIPatternMiner,
        NeuralKGEmbedder,
        CausalDiscoveryEngine,
        LagrangeAttractorAnalyzer,
        GlobalChemKnowledgeAdapter,
        NeuromancerDynamicsModeler
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    KarateClubGraphAnalyzer = None
    PAMIPatternMiner = None
    NeuralKGEmbedder = None
    CausalDiscoveryEngine = None
    LagrangeAttractorAnalyzer = None
    GlobalChemKnowledgeAdapter = None
    NeuromancerDynamicsModeler = None

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsResult:
    """Standardized analytics result container"""
    status: str
    analysis_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: List[str] = field(default_factory=list)


class AdvancedAnalyticsEngine:
    """
    Advanced Analytics Engine for Knowledge Engine.
    
    Integrates external analysis tools:
    - Karate Club: Graph community detection and embeddings
    - PAMI: Pattern mining from knowledge artifacts
    - NeuralKG: Knowledge graph embeddings
    - Causal-Learn: Causal discovery and inference
    - Lagrange-Mapper: Topological data analysis
    
    This engine enhances the knowledge extraction and graph analysis capabilities
    of the core Knowledge Engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Advanced Analytics Engine.
        
        Args:
            config: Configuration for analytics engine
        """
        self.config = config or self._default_config()
        self._initialize_integrations()
        
        logger.info({
            "msg": "AdvancedAnalyticsEngine initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'karateclub': {'enabled': True},
            'pami': {'enabled': True, 'min_support': 0.1},
            'neuralkg': {'enabled': True, 'embedding_dim': 100},
            'causal_learn': {'enabled': True, 'alpha': 0.05},
            'lagrange_mapper': {'enabled': True, 'n_clusters': 8},
            'global_chem': {'enabled': True},
            'neuromancer': {'enabled': True, 'device': 'cpu'}
        }
    
    def _initialize_integrations(self):
        """Initialize all analytics integrations"""
        self.integrations = {}
        
        if not INTEGRATIONS_AVAILABLE:
            logger.warning("Analytics integrations not available")
            return
        
        # Initialize Karate Club
        if self.config['karateclub']['enabled'] and KarateClubGraphAnalyzer:
            try:
                self.integrations['karateclub'] = KarateClubGraphAnalyzer()
                logger.info("Karate Club integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Karate Club: {e}")
        
        # Initialize PAMI
        if self.config['pami']['enabled'] and PAMIPatternMiner:
            try:
                self.integrations['pami'] = PAMIPatternMiner()
                logger.info("PAMI integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PAMI: {e}")
        
        # Initialize NeuralKG
        if self.config['neuralkg']['enabled'] and NeuralKGEmbedder:
            try:
                self.integrations['neuralkg'] = NeuralKGEmbedder()
                logger.info("NeuralKG integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NeuralKG: {e}")
        
        # Initialize Causal-Learn
        if self.config['causal_learn']['enabled'] and CausalDiscoveryEngine:
            try:
                self.integrations['causal'] = CausalDiscoveryEngine()
                logger.info("Causal-Learn integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Causal-Learn: {e}")
        
        # Initialize Lagrange-Mapper
        if self.config['lagrange_mapper']['enabled'] and LagrangeAttractorAnalyzer:
            try:
                self.integrations['lagrange'] = LagrangeAttractorAnalyzer()
                logger.info("Lagrange-Mapper integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Lagrange-Mapper: {e}")
        
        # Initialize GlobalChem
        if GlobalChemKnowledgeAdapter:
            try:
                self.integrations['global_chem'] = GlobalChemKnowledgeAdapter()
                logger.info("GlobalChem integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GlobalChem: {e}")
        
        # Initialize Neuromancer
        if NeuromancerDynamicsModeler:
            try:
                self.integrations['neuromancer'] = NeuromancerDynamicsModeler()
                logger.info("Neuromancer integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Neuromancer: {e}")
    
    def get_available_integrations(self) -> List[str]:
        """Get list of available integrations"""
        return list(self.integrations.keys())
    
    # ==================== Graph Analysis with Karate Club ====================
    
    def analyze_knowledge_graph_communities(
        self,
        graph_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Analyze knowledge graph communities using Karate Club.
        
        Args:
            graph_data: Knowledge graph with nodes and edges
            config: Analysis configuration
            
        Returns:
            AnalyticsResult with community analysis
        """
        if 'karateclub' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='community_detection',
                errors=['Karate Club integration not available']
            )
        
        try:
            result = self.integrations['karateclub'].analyze_graph(
                graph_data,
                analysis_config=config
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='community_detection',
                data=result.get('analysis_results', {}),
                metadata={'config_used': result.get('config_used', {})}
            )
            
        except Exception as e:
            logger.error(f"Community analysis failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='community_detection',
                errors=[str(e)]
            )
    
    # ==================== Pattern Mining with PAMI ====================
    
    def mine_artifact_patterns(
        self,
        artifacts: List[KnowledgeArtifact],
        pattern_type: str = 'frequent',
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Mine patterns from knowledge artifacts using PAMI.
        
        Args:
            artifacts: List of knowledge artifacts
            pattern_type: Type of pattern ('frequent', 'sequential', 'association')
            config: Mining configuration
            
        Returns:
            AnalyticsResult with mined patterns
        """
        if 'pami' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='pattern_mining',
                errors=['PAMI integration not available']
            )
        
        try:
            # Convert artifacts to transactions
            transactions = self._artifacts_to_transactions(artifacts)
            
            cfg = config or {}
            min_support = cfg.get('min_support', self.config['pami']['min_support'])
            
            if pattern_type == 'frequent':
                result = self.integrations['pami'].mine_frequent_patterns(
                    transactions=transactions,
                    min_support=min_support
                )
            elif pattern_type == 'sequential':
                result = self.integrations['pami'].mine_sequences(
                    sequences=transactions,
                    min_support=min_support
                )
            elif pattern_type == 'association':
                result = self.integrations['pami'].discover_association_rules(
                    transactions=transactions,
                    min_support=min_support,
                    min_confidence=cfg.get('min_confidence', 0.5)
                )
            else:
                return AnalyticsResult(
                    status='error',
                    analysis_type='pattern_mining',
                    errors=[f'Unknown pattern type: {pattern_type}']
                )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='pattern_mining',
                data=result,
                metadata={'pattern_type': pattern_type, 'artifacts_processed': len(artifacts)}
            )
            
        except Exception as e:
            logger.error(f"Pattern mining failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='pattern_mining',
                errors=[str(e)]
            )
    
    def _artifacts_to_transactions(
        self,
        artifacts: List[KnowledgeArtifact]
    ) -> List[List[str]]:
        """Convert knowledge artifacts to transaction format"""
        transactions = []
        
        for artifact in artifacts:
            transaction = []
            
            # Add artifact type
            if hasattr(artifact, 'artifact_type'):
                transaction.append(f"type:{artifact.artifact_type}")
            
            # Add source
            if hasattr(artifact, 'source'):
                transaction.append(f"source:{artifact.source}")
            
            # Add domain if available
            if hasattr(artifact, 'domain') and artifact.domain:
                transaction.append(f"domain:{artifact.domain}")
            
            # Add problem type if available
            if hasattr(artifact, 'problem_type') and artifact.problem_type:
                transaction.append(f"problem:{artifact.problem_type}")
            
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    # ==================== Knowledge Graph Embeddings with NeuralKG ====================
    
    def generate_knowledge_embeddings(
        self,
        triples: List[Tuple[str, str, str]],
        model: str = 'transe',
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Generate knowledge graph embeddings using NeuralKG.
        
        Args:
            triples: List of (head, relation, tail) triples
            model: Model name ('transe', 'rotate', 'complex', etc.)
            config: Embedding configuration
            
        Returns:
            AnalyticsResult with embeddings
        """
        if 'neuralkg' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='embedding_generation',
                errors=['NeuralKG integration not available']
            )
        
        try:
            cfg = config or {}
            embedding_dim = cfg.get('embedding_dim', self.config['neuralkg']['embedding_dim'])
            
            result = self.integrations['neuralkg'].generate_embeddings(
                triples=triples,
                model_name=model,
                embedding_dim=embedding_dim,
                epochs=cfg.get('epochs', 100)
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='embedding_generation',
                data=result.get('embeddings', {}),
                metadata={
                    'model': model,
                    'embedding_dim': embedding_dim,
                    'num_triples': len(triples)
                }
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='embedding_generation',
                errors=[str(e)]
            )
    
    def predict_missing_links(
        self,
        head: str,
        relation: str,
        candidate_tails: List[str],
        embeddings: Dict[str, Any],
        top_k: int = 10
    ) -> AnalyticsResult:
        """
        Predict missing links using NeuralKG embeddings.
        
        Args:
            head: Head entity
            relation: Relation
            candidate_tails: Candidate tail entities
            embeddings: Pre-computed embeddings
            top_k: Number of top predictions
            
        Returns:
            AnalyticsResult with predictions
        """
        if 'neuralkg' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='link_prediction',
                errors=['NeuralKG integration not available']
            )
        
        try:
            result = self.integrations['neuralkg'].predict_links(
                head=head,
                relation=relation,
                candidate_tails=candidate_tails,
                embeddings=embeddings,
                top_k=top_k
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='link_prediction',
                data=result.get('predictions', []),
                metadata={'head': head, 'relation': relation, 'top_k': top_k}
            )
            
        except Exception as e:
            logger.error(f"Link prediction failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='link_prediction',
                errors=[str(e)]
            )
    
    # ==================== Causal Discovery with Causal-Learn ====================
    
    def discover_causal_relationships(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        algorithm: str = 'pc',
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Discover causal relationships using Causal-Learn.
        
        Args:
            data: Data matrix (n_samples x n_variables)
            variable_names: Names of variables
            algorithm: Algorithm ('pc', 'fci', 'ges', 'lingam')
            config: Algorithm configuration
            
        Returns:
            AnalyticsResult with causal graph
        """
        if 'causal' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='causal_discovery',
                errors=['Causal-Learn integration not available']
            )
        
        try:
            cfg = config or {}
            alpha = cfg.get('alpha', self.config['causal_learn']['alpha'])
            
            result = self.integrations['causal'].discover_causal_structure(
                data=data,
                variable_names=variable_names,
                algorithm=algorithm,
                alpha=alpha,
                independence_test=cfg.get('independence_test', 'fisherz')
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='causal_discovery',
                data=result.get('graph', {}),
                metadata={
                    'algorithm': algorithm,
                    'alpha': alpha,
                    'variables': variable_names
                }
            )
            
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='causal_discovery',
                errors=[str(e)]
            )
    
    def identify_causal_confounders(
        self,
        graph_data: Dict[str, Any],
        target_x: str,
        target_y: str
    ) -> AnalyticsResult:
        """
        Identify confounders in a causal graph.
        
        Args:
            graph_data: Causal graph
            target_x: First target variable
            target_y: Second target variable
            
        Returns:
            AnalyticsResult with confounders
        """
        if 'causal' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='confounder_identification',
                errors=['Causal-Learn integration not available']
            )
        
        try:
            result = self.integrations['causal'].identify_confounders(
                graph_data=graph_data,
                target_x=target_x,
                target_y=target_y
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='confounder_identification',
                data=result.get('confounders', {})
            )
            
        except Exception as e:
            logger.error(f"Confounder identification failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='confounder_identification',
                errors=[str(e)]
            )
    
    # ==================== Topological Analysis with Lagrange-Mapper ====================
    
    def analyze_embedding_landscape(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Analyze attractor landscape using Lagrange-Mapper.
        
        Args:
            embeddings: Embedding matrix
            labels: Optional labels
            config: Analysis configuration
            
        Returns:
            AnalyticsResult with landscape analysis
        """
        if 'lagrange' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='landscape_analysis',
                errors=['Lagrange-Mapper integration not available']
            )
        
        try:
            cfg = config or {}
            n_clusters = cfg.get('n_clusters', self.config['lagrange_mapper']['n_clusters'])
            
            result = self.integrations['lagrange'].analyze_embedding_landscape(
                embeddings=embeddings,
                labels=labels,
                n_clusters=n_clusters,
                reduction_method=cfg.get('reduction_method', 'pca'),
                reduction_dims=cfg.get('reduction_dims', 2)
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='landscape_analysis',
                data=result.get('landscape', {}),
                metadata={'n_clusters': n_clusters}
            )
            
        except Exception as e:
            logger.error(f"Landscape analysis failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='landscape_analysis',
                errors=[str(e)]
            )
    
    def analyze_knowledge_topology(
        self,
        graph_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResult:
        """
        Analyze knowledge graph topology.
        
        Args:
            graph_data: Knowledge graph
            config: Analysis configuration
            
        Returns:
            AnalyticsResult with topology analysis
        """
        if 'lagrange' not in self.integrations:
            return AnalyticsResult(
                status='error',
                analysis_type='topology_analysis',
                errors=['Lagrange-Mapper integration not available']
            )
        
        try:
            cfg = config or {}
            embedding_dim = cfg.get('embedding_dim', 50)
            
            result = self.integrations['lagrange'].analyze_knowledge_topology(
                graph_data=graph_data,
                embedding_dim=embedding_dim
            )
            
            return AnalyticsResult(
                status=result.get('status', 'error'),
                analysis_type='topology_analysis',
                data=result.get('landscape', {}),
                metadata={'embedding_dim': embedding_dim}
            )
            
        except Exception as e:
            logger.error(f"Topology analysis failed: {e}")
            return AnalyticsResult(
                status='error',
                analysis_type='topology_analysis',
                errors=[str(e)]
            )
    
    # ==================== Comprehensive Analysis ====================
    
    def comprehensive_graph_analysis(
        self,
        graph_data: Dict[str, Any],
        include_communities: bool = True,
        include_embeddings: bool = True,
        include_topology: bool = True
    ) -> Dict[str, AnalyticsResult]:
        """
        Run comprehensive analysis on a knowledge graph.
        
        Args:
            graph_data: Knowledge graph
            include_communities: Include community detection
            include_embeddings: Include embedding generation
            include_topology: Include topological analysis
            
        Returns:
            Dictionary of AnalyticsResult objects
        """
        results = {}
        
        # Community detection
        if include_communities:
            results['communities'] = self.analyze_knowledge_graph_communities(graph_data)
        
        # Embedding generation (requires triples)
        if include_embeddings and 'edges' in graph_data:
            triples = [
                (e['source'], e.get('type', 'related_to'), e['target'])
                for e in graph_data['edges']
            ]
            if triples:
                results['embeddings'] = self.generate_knowledge_embeddings(triples)
        
        # Topological analysis
        if include_topology:
            results['topology'] = self.analyze_knowledge_topology(graph_data)
        
        return results
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics capabilities and status"""
        return {
            'available_integrations': self.get_available_integrations(),
            'config': self.config,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
