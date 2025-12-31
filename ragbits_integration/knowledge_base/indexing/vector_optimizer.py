"""
Vector Index Optimizer

Optimizes vector indexing strategies for improved search performance
and relevance.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IndexingStrategy(Enum):
    """Vector indexing strategies"""
    BASIC = "basic"                      # Simple flat indexing
    HNSW = "hnsw"                        # Hierarchical Navigable Small World
    IVF = "ivf"                          # Inverted File Index
    HNSW_IVF = "hnsw_ivf"                # Hybrid HNSW + IVF
    ADAPTIVE = "adaptive"                # Adaptive strategy selection


@dataclass
class IndexConfiguration:
    """Vector index configuration"""
    strategy: IndexingStrategy
    dimension: int
    metric: str = "cosine"              # cosine, euclidean, dotproduct
    ef_construction: int = 200          # HNSW ef_construction parameter
    M: int = 16                         # HNSW M parameter (connections)
    nlist: int = 100                    # IVF nlist parameter (clusters)
    nprobe: int = 10                    # IVF nprobe parameter (clusters to search)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy": self.strategy.value,
            "dimension": self.dimension,
            "metric": self.metric,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "nlist": self.nlist,
            "nprobe": self.nprobe
        }


@dataclass
class OptimizationReport:
    """Report from index optimization"""
    original_strategy: IndexingStrategy
    optimized_strategy: IndexingStrategy
    performance_improvement: float  # Percentage
    index_size_reduction: float     # Percentage
    search_accuracy: float
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_strategy": self.original_strategy.value,
            "optimized_strategy": self.optimized_strategy.value,
            "performance_improvement": self.performance_improvement,
            "index_size_reduction": self.index_size_reduction,
            "search_accuracy": self.search_accuracy,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }


class VectorIndexOptimizer:
    """
    Optimizes vector indexing for search performance.

    Analyzes data characteristics and recommends optimal
    indexing strategies.

    Usage:
        optimizer = VectorIndexOptimizer()

        # Analyze and recommend strategy
        report = await optimizer.analyze_and_recommend(
            document_count=10000,
            dimension=1536,
            query_pattern="read_heavy"
        )

        print(f"Recommended strategy: {report.optimized_strategy}")
    """

    # Strategy recommendations based on data characteristics
    STRATEGY_GUIDELINES = {
        "small": {
            "max_documents": 10000,
            "recommended": IndexingStrategy.BASIC,
            "reason": "Basic indexing sufficient for small datasets"
        },
        "medium_read_heavy": {
            "max_documents": 1000000,
            "recommended": IndexingStrategy.HNSW,
            "reason": "HNSW provides excellent recall for read-heavy workloads"
        },
        "medium_write_heavy": {
            "max_documents": 1000000,
            "recommended": IndexingStrategy.IVF,
            "reason": "IVF better handles frequent updates"
        },
        "large_read_heavy": {
            "min_documents": 1000000,
            "recommended": IndexingStrategy.HNSW_IVF,
            "reason": "Hybrid strategy optimal for large read-heavy datasets"
        },
        "large_mixed": {
            "min_documents": 1000000,
            "recommended": IndexingStrategy.ADAPTIVE,
            "reason": "Adaptive strategy balances read/write performance"
        }
    }

    def __init__(self, document_search=None):
        """
        Initialize vector index optimizer.

        Args:
            document_search: Optional document search instance
        """
        self.document_search = document_search

        # Optimization history
        self.optimization_history: List[OptimizationReport] = []

        logger.info("VectorIndexOptimizer initialized")

    async def analyze_and_recommend(
        self,
        document_count: int,
        dimension: int,
        query_pattern: str = "mixed",
        current_strategy: Optional[IndexingStrategy] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationReport:
        """
        Analyze data and recommend optimal indexing strategy.

        Args:
            document_count: Number of documents in index
            dimension: Vector dimension
            query_pattern: Query pattern (read_heavy, write_heavy, mixed)
            current_strategy: Current indexing strategy
            performance_metrics: Optional current performance metrics

        Returns:
            Optimization report with recommendations
        """
        logger.info(
            f"Analyzing index: {document_count} docs, "
            f"dim={dimension}, pattern={query_pattern}"
        )

        # Determine recommended strategy
        recommended_strategy = self._recommend_strategy(
            document_count,
            query_pattern
        )

        # Generate configuration
        config = self._generate_configuration(
            recommended_strategy,
            document_count,
            dimension
        )

        # Calculate expected improvements
        improvements = self._estimate_improvements(
            current_strategy,
            recommended_strategy,
            document_count
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            recommended_strategy,
            document_count,
            query_pattern
        )

        report = OptimizationReport(
            original_strategy=current_strategy or IndexingStrategy.BASIC,
            optimized_strategy=recommended_strategy,
            performance_improvement=improvements["performance"],
            index_size_reduction=improvements["size"],
            search_accuracy=improvements["accuracy"],
            recommendations=recommendations,
            metrics={
                "document_count": document_count,
                "dimension": dimension,
                "estimated_build_time": self._estimate_build_time(
                    document_count,
                    recommended_strategy
                ),
                "estimated_memory_usage": self._estimate_memory_usage(
                    document_count,
                    dimension,
                    recommended_strategy
                )
            }
        )

        self.optimization_history.append(report)

        logger.info(
            f"Recommended strategy: {recommended_strategy.value}, "
            f"expected improvement: {improvements['performance']:.1f}%"
        )

        return report

    def _recommend_strategy(
        self,
        document_count: int,
        query_pattern: str
    ) -> IndexingStrategy:
        """Recommend indexing strategy based on characteristics"""
        # Small datasets
        if document_count < 10000:
            return IndexingStrategy.BASIC

        # Medium datasets
        if document_count < 1000000:
            if query_pattern == "read_heavy":
                return IndexingStrategy.HNSW
            elif query_pattern == "write_heavy":
                return IndexingStrategy.IVF
            else:
                return IndexingStrategy.HNSW  # Default to HNSW for mixed

        # Large datasets
        if query_pattern == "read_heavy":
            return IndexingStrategy.HNSW_IVF
        else:
            return IndexingStrategy.ADAPTIVE

    def _generate_configuration(
        self,
        strategy: IndexingStrategy,
        document_count: int,
        dimension: int
    ) -> IndexConfiguration:
        """Generate optimal configuration for strategy"""
        # Default configuration
        config = IndexConfiguration(
            strategy=strategy,
            dimension=dimension
        )

        # Strategy-specific tuning
        if strategy == IndexingStrategy.HNSW:
            # Tune HNSW parameters
            config.ef_construction = max(200, int(document_count / 1000))
            config.M = 16 if document_count < 100000 else 32

        elif strategy == IndexingStrategy.IVF:
            # Tune IVF parameters
            config.nlist = min(1000, max(100, int(document_count / 1000)))
            config.nprobe = min(100, max(10, int(config.nlist / 10)))

        elif strategy == IndexingStrategy.HNSW_IVF:
            # Hybrid configuration
            config.ef_construction = 200
            config.M = 24
            config.nlist = min(500, max(50, int(document_count / 5000)))
            config.nprobe = min(50, max(5, int(config.nlist / 10)))

        return config

    def _estimate_improvements(
        self,
        current: Optional[IndexingStrategy],
        recommended: IndexingStrategy,
        document_count: int
    ) -> Dict[str, float]:
        """Estimate performance improvements"""
        if not current or current == recommended:
            return {
                "performance": 0.0,
                "size": 0.0,
                "accuracy": 0.95
            }

        # Estimate improvements based on strategy transitions
        improvements = {
            "performance": 0.0,
            "size": 0.0,
            "accuracy": 0.95
        }

        # Basic to advanced strategies
        if current == IndexingStrategy.BASIC:
            if recommended in [IndexingStrategy.HNSW, IndexingStrategy.IVF]:
                improvements["performance"] = 30.0
                improvements["accuracy"] = 0.98
            elif recommended in [IndexingStrategy.HNSW_IVF, IndexingStrategy.ADAPTIVE]:
                improvements["performance"] = 50.0
                improvements["accuracy"] = 0.97

        # HNSW to hybrid
        elif current == IndexingStrategy.HNSW and recommended == IndexingStrategy.HNSW_IVF:
            improvements["performance"] = 20.0
            improvements["size"] = 15.0
            improvements["accuracy"] = 0.96

        # IVF to hybrid
        elif current == IndexingStrategy.IVF and recommended == IndexingStrategy.HNSW_IVF:
            improvements["performance"] = 25.0
            improvements["accuracy"] = 0.97

        return improvements

    def _generate_recommendations(
        self,
        strategy: IndexingStrategy,
        document_count: int,
        query_pattern: str
    ) -> List[str]:
        """Generate implementation recommendations"""
        recommendations = []

        if strategy == IndexingStrategy.BASIC:
            recommendations.append(
                "Basic indexing is sufficient for current dataset size"
            )
            recommendations.append(
                "Consider upgrading to HNSW when document count exceeds 10,000"
            )

        elif strategy == IndexingStrategy.HNSW:
            recommendations.append(
                "Use HNSW for best search accuracy on read-heavy workloads"
            )
            recommendations.append(
                "Monitor index build time for large datasets"
            )
            if document_count > 100000:
                recommendations.append(
                    "Consider HNSW_IVF hybrid for better memory efficiency"
                )

        elif strategy == IndexingStrategy.IVF:
            recommendations.append(
                "IVF provides good balance of search speed and update performance"
            )
            recommendations.append(
                "Tune nprobe parameter based on accuracy requirements"
            )

        elif strategy == IndexingStrategy.HNSW_IVF:
            recommendations.append(
                "Hybrid strategy optimal for large-scale read-heavy workloads"
            )
            recommendations.append(
                "Requires more memory but provides excellent performance"
            )

        elif strategy == IndexingStrategy.ADAPTIVE:
            recommendations.append(
                "Adaptive strategy will adjust based on query patterns"
            )
            recommendations.append(
                "Monitor performance metrics to fine-tune adaptation parameters"
            )

        # General recommendations
        recommendations.append(
            f"Index size estimated at {self._estimate_memory_human(document_count)}"
        )

        return recommendations

    def _estimate_build_time(
        self,
        document_count: int,
        strategy: IndexingStrategy
    ) -> float:
        """Estimate index build time in seconds"""
        base_time = document_count / 1000  # Base estimate

        multipliers = {
            IndexingStrategy.BASIC: 1.0,
            IndexingStrategy.HNSW: 2.5,
            IndexingStrategy.IVF: 1.5,
            IndexingStrategy.HNSW_IVF: 3.0,
            IndexingStrategy.ADAPTIVE: 2.0
        }

        return base_time * multipliers.get(strategy, 2.0)

    def _estimate_memory_usage(
        self,
        document_count: int,
        dimension: int,
        strategy: IndexingStrategy
    ) -> float:
        """Estimate memory usage in MB"""
        # Base memory: documents * dimension * 4 bytes (float32)
        base_memory = document_count * dimension * 4 / (1024 * 1024)

        # Strategy overhead
        overhead = {
            IndexingStrategy.BASIC: 1.0,
            IndexingStrategy.HNSW: 2.0,
            IndexingStrategy.IVF: 1.2,
            IndexingStrategy.HNSW_IVF: 2.5,
            IndexingStrategy.ADAPTIVE: 1.8
        }

        return base_memory * overhead.get(strategy, 1.5)

    def _estimate_memory_human(self, document_count: int) -> str:
        """Estimate memory in human-readable format"""
        # Assuming 1536 dimensions (OpenAI embeddings)
        memory_mb = self._estimate_memory_usage(
            document_count,
            1536,
            IndexingStrategy.HNSW
        )

        if memory_mb < 1024:
            return f"{memory_mb:.0f} MB"
        else:
            return f"{memory_mb / 1024:.1f} GB"

    async def optimize_index(
        self,
        current_config: IndexConfiguration,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[IndexConfiguration, List[str]]:
        """
        Optimize existing index configuration.

        Args:
            current_config: Current index configuration
            target_metrics: Target performance metrics

        Returns:
            Tuple of (optimized_config, changes_made)
        """
        changes = []

        # Start with current config
        optimized = IndexConfiguration(
            strategy=current_config.strategy,
            dimension=current_config.dimension,
            metric=current_config.metric,
            ef_construction=current_config.ef_construction,
            M=current_config.M,
            nlist=current_config.nlist,
            nprobe=current_config.nprobe
        )

        # Optimize based on strategy
        if optimized.strategy == IndexingStrategy.HNSW:
            # Optimize ef_construction and M
            if optimized.ef_construction < 200:
                optimized.ef_construction = 200
                changes.append("Increased ef_construction to 200 for better recall")

            if optimized.M < 16:
                optimized.M = 16
                changes.append("Increased M to 16 for better connectivity")

        elif optimized.strategy == IndexingStrategy.IVF:
            # Optimize nlist and nprobe
            if optimized.nprobe < 10:
                optimized.nprobe = 10
                changes.append("Increased nprobe to 10 for better accuracy")

        return optimized, changes

    def get_optimization_history(self) -> List[OptimizationReport]:
        """Get history of optimizations"""
        return self.optimization_history.copy()
