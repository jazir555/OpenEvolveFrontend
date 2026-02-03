"""
OpenEvolve Knowledge Engine - COMPLETION MODULE

This module provides the completed implementations for the Knowledge Engine,
integrating all the missing pieces:

1. Real Embedding Service (sentence-transformers + fallbacks)
2. Cloud Storage Backends (S3, GCS, Azure)
3. Full-Featured Backends (complete CRUD operations)
4. Confidence Scoring System
5. Ensemble Strategy Recommender

Usage:
    from knowledge_engine.__complete__ import (
        create_complete_knowledge_engine,
        EmbeddingService,
        ConfidenceScorer,
        EnsembleStrategySelector
    )
    
    # Create fully-featured knowledge engine
    engine = create_complete_knowledge_engine()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure parent directory is in path for imports
_parent_dir = str(Path(__file__).parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import the new completion modules with fallback
try:
    from embedding_service import (
        EmbeddingService,
        EmbeddingConfig,
        create_embedding_service,
        get_default_embedding_service
    )
except ImportError:
    from .embedding_service import (
        EmbeddingService,
        EmbeddingConfig,
        create_embedding_service,
        get_default_embedding_service
    )

try:
    from cloud_storage_backends import (
        S3BackupStorage,
        GCSBackupStorage,
        AzureBackupStorage,
        S3Credentials,
        GCSCredentials,
        AzureCredentials,
        create_cloud_storage
    )
except ImportError:
    from .cloud_storage_backends import (
        S3BackupStorage,
        GCSBackupStorage,
        AzureBackupStorage,
        S3Credentials,
        GCSCredentials,
        AzureCredentials,
        create_cloud_storage
    )

try:
    from core.backends.full_featured_backends import (
        FullFeaturedInMemoryBackend,
        FullFeaturedPostgreSQLBackend,
        FullFeaturedQdrantBackend,
        create_full_featured_backend
    )
except ImportError:
    from .core.backends.full_featured_backends import (
        FullFeaturedInMemoryBackend,
        FullFeaturedPostgreSQLBackend,
        FullFeaturedQdrantBackend,
        create_full_featured_backend
    )

try:
    from confidence_scorer import (
        ConfidenceScorer,
        ConfidenceFactors,
        calculate_confidence,
        get_confidence_scorer
    )
except ImportError:
    from .confidence_scorer import (
        ConfidenceScorer,
        ConfidenceFactors,
        calculate_confidence,
        get_confidence_scorer
    )

from .core.strategy_recommender_complete import (
    StrategyRecommendation,
    BaseStrategyRecommender,
    KeywordBasedRecommender,
    DomainBasedRecommender,
    ComplexityBasedRecommender,
    HistoricalPerformanceRecommender,
    EnsembleStrategySelector,
    recommend_strategy
)

# Import existing components
from .master_engine import MasterKnowledgeEngine, KnowledgeDomain
from .unified_knowledge_platform import UnifiedKnowledgePlatform

logger = logging.getLogger(__name__)


class CompletedKnowledgeEngine:
    """
    Fully completed knowledge engine with all features implemented.
    
    This class wraps the existing MasterKnowledgeEngine and adds:
    - Real embedding generation
    - Confidence scoring
    - Strategy recommendations
    - Full CRUD operations
    """
    
    def __init__(
        self,
        storage_path: str = "./knowledge_data",
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_learning: bool = True,
        enable_cloud_backups: bool = False,
        cloud_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the completed knowledge engine.
        
        Args:
            storage_path: Path for local storage
            embedding_model: Name of embedding model to use
            enable_learning: Whether to enable self-learning
            enable_cloud_backups: Whether to enable cloud backups
            cloud_config: Cloud storage configuration
        """
        self.storage_path = storage_path
        
        # Initialize embedding service
        self.embedding_service = create_embedding_service(
            model_name=embedding_model
        )
        
        # Initialize confidence scorer
        self.confidence_scorer = get_confidence_scorer()
        
        # Initialize strategy selector
        self.strategy_selector = EnsembleStrategySelector()
        
        # Initialize base engine
        self.base_engine = MasterKnowledgeEngine(
            storage_path=storage_path,
            enable_learning=enable_learning
        )
        
        # Initialize cloud storage if enabled
        self.cloud_storage = None
        if enable_cloud_backups and cloud_config:
            self._init_cloud_storage(cloud_config)
        
        logger.info({
            "msg": "CompletedKnowledgeEngine initialized",
            "embedding_model": embedding_model,
            "embedding_dimensions": self.embedding_service.config.dimensions,
            "cloud_enabled": enable_cloud_backups
        })
    
    def _init_cloud_storage(self, config: Dict[str, Any]):
        """Initialize cloud storage from config."""
        try:
            storage_type = config.get('type', 's3')
            self.cloud_storage = create_cloud_storage(
                storage_type=storage_type,
                bucket_or_container=config.get('bucket'),
                prefix=config.get('prefix', 'backups/'),
                **config.get('credentials', {})
            )
            logger.info({
                "msg": "Cloud storage initialized",
                "type": storage_type
            })
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize cloud storage",
                "error": str(e)
            })
    
    async def process(
        self,
        query: str,
        domain: str = "general",
        use_confidence_scoring: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query with full feature set.
        
        Args:
            query: Query to process
            domain: Knowledge domain
            use_confidence_scoring: Whether to apply confidence scoring
            
        Returns:
            Results with confidence scores
        """
        # Get base results
        response = await self.base_engine.process(
            query=query,
            domain=getattr(KnowledgeDomain, domain.upper(), KnowledgeDomain.GENERAL)
        )
        
        # Add confidence scores if requested
        if use_confidence_scoring and response.success:
            response.results['confidence_data'] = self._score_results(
                query, response.results
            )
        
        # Add strategy recommendation
        strategy_rec = self.strategy_selector.recommend_strategy(
            problem_description=query,
            domain=domain
        )
        response.results['recommended_strategy'] = strategy_rec.to_dict()
        
        return response
    
    def _score_results(
        self,
        query: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply confidence scoring to results."""
        scored_items = []
        
        # Extract items to score
        items = results.get('results', [])
        
        for item in items:
            confidence, factors = self.confidence_scorer.calculate_confidence(
                similarity_score=item.get('similarity', 0.5),
                source=item.get('source', 'unknown'),
                metadata=item.get('metadata', {})
            )
            
            scored_items.append({
                **item,
                'confidence': confidence,
                'confidence_factors': factors.to_dict(),
                'confidence_level': self.confidence_scorer.get_confidence_level(confidence)
            })
        
        # Sort by confidence
        scored_items.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'items': scored_items,
            'avg_confidence': sum(i['confidence'] for i in scored_items) / len(scored_items) if scored_items else 0
        }
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the embedding service.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_service.embed_text(text)
        # Handle both numpy arrays and lists
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general"
    ) -> StrategyRecommendation:
        """
        Recommend a strategy for the given problem.
        
        Args:
            problem_description: Problem to analyze
            domain: Problem domain
            
        Returns:
            Strategy recommendation
        """
        return self.strategy_selector.recommend_strategy(
            problem_description=problem_description,
            domain=domain
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        base_stats = self.base_engine.get_statistics()
        embedding_stats = self.embedding_service.get_stats()
        
        return {
            **base_stats,
            "embedding_service": embedding_stats,
            "confidence_scorer_enabled": True,
            "strategy_selector": "ensemble",
            "cloud_storage_enabled": self.cloud_storage is not None
        }


def create_complete_knowledge_engine(
    storage_path: str = "./knowledge_data",
    embedding_model: str = "all-MiniLM-L6-v2",
    enable_learning: bool = True,
    enable_cloud_backups: bool = False,
    cloud_config: Optional[Dict[str, Any]] = None
) -> CompletedKnowledgeEngine:
    """
    Factory function to create a fully completed knowledge engine.
    
    Args:
        storage_path: Path for local storage
        embedding_model: Embedding model name
        enable_learning: Enable self-learning
        enable_cloud_backups: Enable cloud backup storage
        cloud_config: Cloud storage configuration
        
    Returns:
        CompletedKnowledgeEngine instance
    """
    return CompletedKnowledgeEngine(
        storage_path=storage_path,
        embedding_model=embedding_model,
        enable_learning=enable_learning,
        enable_cloud_backups=enable_cloud_backups,
        cloud_config=cloud_config
    )


# Export all completion components
__all__ = [
    # Main completed engine
    'CompletedKnowledgeEngine',
    'create_complete_knowledge_engine',
    
    # Embedding
    'EmbeddingService',
    'EmbeddingConfig',
    'create_embedding_service',
    'get_default_embedding_service',
    
    # Cloud storage
    'S3BackupStorage',
    'GCSBackupStorage',
    'AzureBackupStorage',
    'S3Credentials',
    'GCSCredentials',
    'AzureCredentials',
    'create_cloud_storage',
    
    # Full-featured backends
    'FullFeaturedInMemoryBackend',
    'FullFeaturedPostgreSQLBackend',
    'FullFeaturedQdrantBackend',
    'create_full_featured_backend',
    
    # Confidence scoring
    'ConfidenceScorer',
    'ConfidenceFactors',
    'calculate_confidence',
    'get_confidence_scorer',
    
    # Strategy recommendation
    'StrategyRecommendation',
    'BaseStrategyRecommender',
    'KeywordBasedRecommender',
    'DomainBasedRecommender',
    'ComplexityBasedRecommender',
    'HistoricalPerformanceRecommender',
    'EnsembleStrategySelector',
    'recommend_strategy'
]


# Version info
__version__ = "2.0.0-complete"
__completion_date__ = "2026-02-03"

logger.info({
    "msg": "Knowledge Engine Completion Module loaded",
    "version": __version__,
    "completion_date": __completion_date__
})
