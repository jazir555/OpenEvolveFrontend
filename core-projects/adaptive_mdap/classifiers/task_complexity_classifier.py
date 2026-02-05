"""
Task Complexity Classifier for Adaptive MDAP.

Based on the MAKER paper, this classifier computes a complexity score for each
sub-problem using multiple features to determine the appropriate solving strategy.

Features:
- Text Length: Longer descriptions may indicate complexity
- Domain Rarity: Rare domains may need more resources
- Depth: Deeper sub-problems in decomposition tree
- Historical Error Rate: Domain-specific error history
- Dependency Complexity: Number of dependencies
"""

import math
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from adaptive_mdap.core.types import SubProblem, ComplexityScore
from adaptive_mdap.core.errors import ClassificationError
from adaptive_mdap.utils.cache import EmbeddingCache, FeatureCache
from adaptive_mdap.utils.metrics import get_metrics
from adaptive_mdap.utils.logger import get_logger

logger = get_logger("classifiers.task_complexity")


@dataclass
class ClassifierConfig:
    """Configuration for TaskComplexityClassifier."""
    embedding_model: str = "all-MiniLM-L6-v2"
    feature_weights: Dict[str, float] = None
    cache_dir: str = ".cache/adaptive_mdap"
    max_text_length: int = 5000
    max_depth: int = 10
    max_dependencies: int = 10
    
    # New granular weights
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {
                "text_length": 0.15,
                "domain_rarity": 0.20,
                "depth": 0.15,
                "historical_error": 0.20,
                "dependency": 0.10,
                "keyword_complexity": 0.10,
                "constraint_density": 0.10
            }
        # Validate weights sum to 1.0
        total = sum(self.feature_weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Feature weights must sum to 1.0, got {total}")

class TaskComplexityClassifier:
    """
    Enhanced Classifier that uses granular feature extraction to compute
    a multi-dimensional complexity score.
    """
    
    # Granular complexity keywords
    COMPLEXITY_KEYWORDS = {
        "high": [
            "optimize", "concurrency", "distributed", "security", "vulnerability",
            "cryptography", "performance", "scaling", "refactor", "architect",
            "recursive", "dynamic programming", "heuristics", "bottleneck"
        ],
        "medium": [
            "integrate", "validate", "transform", "interface", "protocol",
            "persistence", "caching", "middleware", "abstraction", "normalization"
        ]
    }

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """Initialize the enhanced classifier."""
        self.config = config or ClassifierConfig()
        self._embedding_model: Optional[Any] = None
        self._embedding_cache = EmbeddingCache(
            cache_dir=self.config.cache_dir,
            max_size=10000,
            default_ttl=7 * 24 * 3600,
        )
        self._feature_cache = FeatureCache(
            max_size=10000,
            default_ttl=3600,
        )
        self._historical_stats: Dict[str, Dict[str, Any]] = {}
        self._domain_embeddings: Dict[str, List[float]] = {}
        
        logger.info("Initialized Enhanced TaskComplexityClassifier")

    def compute_keyword_feature(self, subproblem: SubProblem) -> float:
        """Analyze description for high-complexity technical keywords."""
        text = (subproblem.description or "").lower()
        if not text:
            return 0.0
            
        high_hits = sum(1 for kw in self.COMPLEXITY_KEYWORDS["high"] if kw in text)
        med_hits = sum(1 for kw in self.COMPLEXITY_KEYWORDS["medium"] if kw in text)
        
        # Weighted score: high hits count for more
        raw_score = (high_hits * 0.25) + (med_hits * 0.1)
        return min(raw_score, 1.0)

    def compute_constraint_feature(self, subproblem: SubProblem) -> float:
        """Detect explicit constraints or success criteria density."""
        # Check metadata for explicit constraints
        constraints = subproblem.metadata.get("constraints", [])
        criteria = subproblem.metadata.get("success_criteria", [])
        
        density = len(constraints) + len(criteria)
        # Normalize: 5+ constraints/criteria is considered complex
        return min(density / 5.0, 1.0)

    def compute_text_length_feature(self, subproblem: SubProblem) -> float:
        """Compute text length feature with calibrated sigmoid."""
        description = subproblem.description or ""
        length = len(description)
        if length == 0: return 0.0
        
        length = min(length, self.config.max_text_length)
        midpoint = 800
        slope = 0.005
        return 1 / (1 + math.exp(-slope * (length - midpoint)))

    def compute_domain_rarity_feature(self, subproblem: SubProblem) -> float:
        """Compute domain rarity using embedding distances."""
        domain = subproblem.domain or ""
        if not domain: return 0.5
        
        embedding = self._get_domain_embedding(domain)
        if embedding is None: return 0.5
        
        self._domain_embeddings[domain] = embedding
        if len(self._domain_embeddings) <= 1: return 0.3 # Optimistic initial rarity
        
        similarities = []
        for other_domain, other_embedding in self._domain_embeddings.items():
            if other_domain != domain:
                try:
                    sim = 1 - cosine(embedding, other_embedding)
                    similarities.append(sim)
                except Exception: pass
        
        if not similarities: return 0.5
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity

    def compute_depth_feature(self, subproblem: SubProblem) -> float:
        """Normalize decomposition depth."""
        return min(max(0, subproblem.depth) / self.config.max_depth, 1.0)

    def compute_historical_error_feature(self, subproblem: SubProblem) -> float:
        """Compute historical error rate with Bayesian-style smoothing."""
        domain = subproblem.domain or ""
        if not domain or domain not in self._historical_stats:
            return 0.4 # Default prior: slightly better than average
            
        stats = self._historical_stats[domain]
        total = stats.get("total_count", 0)
        if total == 0: return 0.4
        
        successes = stats.get("success_count", 0)
        raw_error = 1.0 - (successes / total)
        
        # Smoothing: confidence increases with sample size
        # prior=0.4, weight=5 samples
        prior_weight = 5
        smoothed_error = (raw_error * total + 0.4 * prior_weight) / (total + prior_weight)
        return smoothed_error

    def compute_dependency_feature(self, subproblem: SubProblem) -> float:
        """Normalize dependency count."""
        deps = subproblem.dependencies or []
        return min(len(deps) / self.config.max_dependencies, 1.0)

    def compute_complexity(self, subproblem: SubProblem) -> ComplexityScore:
        """Compute granular overall complexity score."""
        start_time = time.time()
        
        try:
            # Extract granular features - ensure keys match ClassifierConfig.feature_weights
            features = {
                "text_length": self.compute_text_length_feature(subproblem),
                "domain_rarity": self.compute_domain_rarity_feature(subproblem),
                "depth": self.compute_depth_feature(subproblem),
                "historical_error": self.compute_historical_error_feature(subproblem),
                "dependency": self.compute_dependency_feature(subproblem),
                "keyword_complexity": self.compute_keyword_feature(subproblem),
                "constraint_density": self.compute_constraint_feature(subproblem)
            }
            
            # Weighted average
            weights = self.config.feature_weights
            overall = sum(features[k] * weights[k] for k in features)
            overall = max(0.0, min(1.0, overall))
            
            duration_ms = (time.time() - start_time) * 1000
            get_metrics().record_classification(duration_ms, success=True)
            
            logger.info(f"Granular Complexity for {subproblem.id}: {overall:.4f}")
            
            return ComplexityScore(
                overall_score=overall,
                text_length_score=features["text_length"],
                domain_rarity_score=features["domain_rarity"],
                depth_score=features["depth"],
                historical_error_score=features["historical_error"],
                dependency_score=features["dependency"],
                feature_weights=weights,
                keyword_score=features["keyword_complexity"],
                constraint_score=features["constraint_density"],
            )
            
        except Exception as e:
            logger.exception(f"Granular classification failed for {subproblem.id}")
            raise ClassificationError(f"Enhanced classification failed: {e}")
    
    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model."""
        if self._embedding_model is not None:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using fallback embeddings")
            self._embedding_model = None
            return
        
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._embedding_model = None

    def _get_domain_embedding(self, domain: str) -> Optional[List[float]]:
        """Get embedding for domain, using cache if available."""
        cached = self._embedding_cache.get_embedding(domain)
        if cached is not None:
            return cached
        
        self._load_embedding_model()
        
        if self._embedding_model is None:
            return self._fallback_embedding(domain)
        
        try:
            embedding = self._embedding_model.encode(domain).tolist()
            self._embedding_cache.set_embedding(domain, embedding)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding for '{domain}': {e}")
            return self._fallback_embedding(domain)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Create a simple hash-based embedding fallback."""
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [(b / 255.0) for b in hash_bytes]

    def update_historical_stats(
        self,
        domain: str,
        success: bool,
        complexity: float
    ) -> None:
        """
        Update historical statistics for a domain.
        """
        if domain not in self._historical_stats:
            self._historical_stats[domain] = {
                "success_count": 0,
                "failure_count": 0,
                "total_count": 0,
                "total_complexity": 0.0,
            }
        
        stats = self._historical_stats[domain]
        stats["total_count"] += 1
        stats["total_complexity"] += complexity
        
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        
        logger.debug(f"Updated stats for {domain}: success={success}, total={stats['total_count']}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "embedding_cache": self._embedding_cache.get_stats(),
            "feature_cache": self._feature_cache.get_stats(),
        }
