"""
Knowledge Engine Integration Connector

This module connects the new analytics integrations (PAMI, NeuralKG, Causal-Learn, Lagrange-Mapper)
to the existing Knowledge Engine core components.

Usage:
    from knowledge_engine import KnowledgeEngineIntegrationConnector
    
    # Connect analytics to knowledge engine
    connector = KnowledgeEngineIntegrationConnector()
    
    # Use analytics with knowledge artifacts
    result = connector.analyze_artifacts_with_pattern_mining(artifacts)
    
    # Use graph analytics
    result = connector.analyze_graph_comprehensive(graph_data)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import knowledge engine components
try:
    from knowledge_extractor import KnowledgeArtifact
except ImportError:
    KnowledgeArtifact = None

try:
    from knowledge_graph_integration import KnowledgeGraphIntegrator
except ImportError:
    KnowledgeGraphIntegrator = None

try:
    from advanced_analytics_engine import AdvancedAnalyticsEngine
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    AdvancedAnalyticsEngine = None
    print(f"Warning: Could not import AdvancedAnalyticsEngine: {e}")

logger = logging.getLogger(__name__)


class KnowledgeEngineIntegrationConnector:
    """
    Connector that integrates advanced analytics into Knowledge Engine workflows.
    
    This connector provides seamless integration between:
    - Knowledge Extractor -> Advanced Analytics
    - Knowledge Graph -> Graph Analytics
    - Knowledge Artifacts -> Pattern Mining
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration connector.
        
        Args:
            config: Configuration for connector and analytics
        """
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Knowledge Engine components not available")
        
        self.config = config or {}
        self.analytics = AdvancedAnalyticsEngine(config)
        
        logger.info({
            "msg": "KnowledgeEngineIntegrationConnector initialized",
            "available_analytics": self.analytics.get_available_integrations(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    # ==================== Pattern Mining Integration ====================
    
    def analyze_artifacts_with_pattern_mining(
        self,
        artifacts: List[KnowledgeArtifact],
        pattern_type: str = 'frequent'
    ) -> Dict[str, Any]:
        """
        Analyze knowledge artifacts using pattern mining.
        
        Connects: KnowledgeExtractor -> PAMI
        
        Args:
            artifacts: Knowledge artifacts from extraction
            pattern_type: Type of patterns to mine
            
        Returns:
            Pattern mining results
        """
        result = self.analytics.mine_artifact_patterns(
            artifacts=artifacts,
            pattern_type=pattern_type
        )
        
        return {
            'status': result.status,
            'patterns': result.data,
            'metadata': result.metadata,
            'artifacts_processed': len(artifacts)
        }
    
    # ==================== Graph Analytics Integration ====================
    
    def analyze_graph_comprehensive(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive graph analysis using multiple tools.
        
        Connects: KnowledgeGraph -> Karate Club + NeuralKG + Lagrange-Mapper
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            Comprehensive analysis results
        """
        results = self.analytics.comprehensive_graph_analysis(
            graph_data=graph_data,
            include_communities=True,
            include_embeddings=True,
            include_topology=True
        )
        
        # Convert to simple dict format
        return {
            'status': 'success',
            'communities': results.get('communities', {}).data if 'communities' in results else None,
            'embeddings': results.get('embeddings', {}).data if 'embeddings' in results else None,
            'topology': results.get('topology', {}).data if 'topology' in results else None,
            'integrations_used': list(results.keys())
        }
    
    def detect_graph_communities(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect communities in knowledge graph.
        
        Connects: KnowledgeGraph -> Karate Club
        
        Args:
            graph_data: Knowledge graph
            
        Returns:
            Community detection results
        """
        result = self.analytics.analyze_knowledge_graph_communities(graph_data)
        
        return {
            'status': result.status,
            'communities': result.data,
            'metadata': result.metadata
        }
    
    def generate_graph_embeddings(
        self,
        triples: List[tuple],
        model: str = 'transe'
    ) -> Dict[str, Any]:
        """
        Generate embeddings for knowledge graph.
        
        Connects: KnowledgeGraph -> NeuralKG
        
        Args:
            triples: List of (head, relation, tail) triples
            model: Embedding model
            
        Returns:
            Embedding results
        """
        result = self.analytics.generate_knowledge_embeddings(
            triples=triples,
            model=model
        )
        
        return {
            'status': result.status,
            'embeddings': result.data,
            'metadata': result.metadata
        }
    
    # ==================== Causal Analysis Integration ====================
    
    def discover_causal_structure(
        self,
        data,
        variable_names: List[str],
        algorithm: str = 'pc'
    ) -> Dict[str, Any]:
        """
        Discover causal relationships in data.
        
        Connects: Knowledge Metrics -> Causal-Learn
        
        Args:
            data: Data matrix
            variable_names: Variable names
            algorithm: Causal discovery algorithm
            
        Returns:
            Causal graph
        """
        import numpy as np
        
        result = self.analytics.discover_causal_relationships(
            data=np.array(data),
            variable_names=variable_names,
            algorithm=algorithm
        )
        
        return {
            'status': result.status,
            'causal_graph': result.data,
            'metadata': result.metadata
        }
    
    # ==================== Topological Analysis Integration ====================
    
    def analyze_knowledge_landscape(
        self,
        embeddings,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze knowledge embedding landscape.
        
        Connects: Knowledge Embeddings -> Lagrange-Mapper
        
        Args:
            embeddings: Embedding matrix
            labels: Optional labels
            
        Returns:
            Landscape analysis
        """
        import numpy as np
        
        result = self.analytics.analyze_embedding_landscape(
            embeddings=np.array(embeddings),
            labels=labels
        )
        
        return {
            'status': result.status,
            'landscape': result.data,
            'metadata': result.metadata
        }
    
    # ==================== Complete Pipeline ====================
    
    def run_enhanced_extraction_pipeline(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run enhanced extraction pipeline with analytics.
        
        Full pipeline:
        1. Extract knowledge artifacts
        2. Mine patterns from artifacts
        3. Build knowledge graph
        4. Analyze graph communities
        5. Generate embeddings
        
        Args:
            workflow_data: Workflow execution data
            
        Returns:
            Complete extraction and analysis results
        """
        results = {
            'status': 'processing',
            'stages': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Stage 1: Pattern mining (if artifacts available)
        if 'artifacts' in workflow_data:
            artifacts = workflow_data['artifacts']
            pattern_result = self.analyze_artifacts_with_pattern_mining(artifacts)
            results['stages']['pattern_mining'] = pattern_result
        
        # Stage 2: Graph analysis (if graph available)
        if 'graph' in workflow_data:
            graph_data = workflow_data['graph']
            graph_result = self.analyze_graph_comprehensive(graph_data)
            results['stages']['graph_analysis'] = graph_result
        
        # Stage 3: Embedding generation (if triples available)
        if 'triples' in workflow_data:
            triples = workflow_data['triples']
            emb_result = self.generate_graph_embeddings(triples)
            results['stages']['embeddings'] = emb_result
        
        results['status'] = 'completed'
        return results
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        return {
            'analytics_available': self.analytics.get_available_integrations(),
            'config': self.config,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Convenience function
def connect_analytics_to_knowledge_engine(config: Optional[Dict[str, Any]] = None):
    """
    Create and return a KnowledgeEngineIntegrationConnector.
    
    Args:
        config: Configuration
        
    Returns:
        KnowledgeEngineIntegrationConnector instance
    """
    return KnowledgeEngineIntegrationConnector(config)
