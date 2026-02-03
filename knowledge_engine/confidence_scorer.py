"""
Confidence Scoring System for Knowledge Retrieval

Provides real confidence scoring based on:
- Vector similarity scores
- Source reliability
- Result consistency
- Historical accuracy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """Factors that contribute to confidence score."""
    similarity_score: float = 0.0
    source_reliability: float = 1.0
    consistency_score: float = 1.0
    recency_score: float = 1.0
    coverage_score: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "similarity_score": self.similarity_score,
            "source_reliability": self.source_reliability,
            "consistency_score": self.consistency_score,
            "recency_score": self.recency_score,
            "coverage_score": self.coverage_score
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for knowledge retrieval results.
    Uses multiple factors to compute a holistic confidence metric.
    """
    
    # Source reliability weights
    SOURCE_RELIABILITY = {
        "verified_database": 1.0,
        "peer_reviewed": 0.95,
        "official_documentation": 0.9,
        "expert_contribution": 0.85,
        "community_wiki": 0.7,
        "user_generated": 0.6,
        "unknown": 0.5,
        "unverified": 0.3
    }
    
    def __init__(
        self,
        similarity_weight: float = 0.35,
        source_weight: float = 0.25,
        consistency_weight: float = 0.20,
        recency_weight: float = 0.10,
        coverage_weight: float = 0.10
    ):
        """
        Initialize confidence scorer.
        
        Args:
            similarity_weight: Weight for vector similarity (0-1)
            source_weight: Weight for source reliability (0-1)
            consistency_weight: Weight for result consistency (0-1)
            recency_weight: Weight for result recency (0-1)
            coverage_weight: Weight for query coverage (0-1)
        """
        self.weights = {
            'similarity': similarity_weight,
            'source': source_weight,
            'consistency': consistency_weight,
            'recency': recency_weight,
            'coverage': coverage_weight
        }
        
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total
    
    def calculate_confidence(
        self,
        similarity_score: float,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        query_terms: Optional[List[str]] = None,
        result_text: Optional[str] = None
    ) -> Tuple[float, ConfidenceFactors]:
        """
        Calculate confidence score for a retrieval result.
        
        Args:
            similarity_score: Vector similarity score (0-1)
            source: Source identifier
            metadata: Additional metadata about the result
            query_terms: Terms from the original query
            result_text: Text content of the result
            
        Returns:
            Tuple of (confidence_score, factors)
        """
        metadata = metadata or {}
        
        factors = ConfidenceFactors()
        
        # 1. Similarity score (normalized)
        factors.similarity_score = self._normalize_score(similarity_score)
        
        # 2. Source reliability
        factors.source_reliability = self._get_source_reliability(source)
        
        # 3. Consistency score
        factors.consistency_score = self._calculate_consistency(metadata)
        
        # 4. Recency score
        factors.recency_score = self._calculate_recency(metadata)
        
        # 5. Coverage score
        factors.coverage_score = self._calculate_coverage(
            query_terms, result_text
        )
        
        # Calculate weighted confidence
        confidence = (
            factors.similarity_score * self.weights['similarity'] +
            factors.source_reliability * self.weights['source'] +
            factors.consistency_score * self.weights['consistency'] +
            factors.recency_score * self.weights['recency'] +
            factors.coverage_score * self.weights['coverage']
        )
        
        # Apply sigmoid to get more realistic distribution
        confidence = self._apply_sigmoid(confidence)
        
        return confidence, factors
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range."""
        # Handle various input ranges
        if score < -1:  # Cosine similarity can be -1 to 1
            score = -1
        elif score > 1:
            score = 1
        
        # Convert to 0-1 range
        return (score + 1) / 2 if score < 0 else score
    
    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for a source."""
        source_lower = source.lower().replace(" ", "_")
        
        # Try exact match
        if source_lower in self.SOURCE_RELIABILITY:
            return self.SOURCE_RELIABILITY[source_lower]
        
        # Try partial match
        for key, value in self.SOURCE_RELIABILITY.items():
            if key in source_lower or source_lower in key:
                return value
        
        return self.SOURCE_RELIABILITY["unknown"]
    
    def _calculate_consistency(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate consistency score based on metadata.
        Checks for version consistency, verification status, etc.
        """
        scores = []
        
        # Check verification status
        if metadata.get('verified', False):
            scores.append(1.0)
        elif 'verified' in metadata:
            scores.append(0.5)
        
        # Check version stability
        if 'version' in metadata:
            version = metadata['version']
            if isinstance(version, int) and version > 1:
                scores.append(min(0.9, 0.5 + version * 0.1))
        
        # Check cross-references
        if metadata.get('cross_references'):
            ref_count = len(metadata['cross_references'])
            scores.append(min(1.0, 0.6 + ref_count * 0.1))
        
        # Check author reputation
        if 'author_reputation' in metadata:
            scores.append(metadata['author_reputation'])
        
        # Default to neutral if no data
        if not scores:
            return 0.75
        
        return np.mean(scores)
    
    def _calculate_recency(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate recency score.
        Newer results get higher scores.
        """
        try:
            # Try to get timestamp
            timestamp = metadata.get('created_at') or metadata.get('timestamp')
            
            if not timestamp:
                return 0.75  # Neutral if no timestamp
            
            # Parse timestamp
            if isinstance(timestamp, str):
                from dateutil import parser
                created_date = parser.parse(timestamp)
            else:
                created_date = timestamp
            
            # Calculate age
            age_days = (datetime.utcnow() - created_date).days
            
            # Score decays exponentially with age
            # 0 days = 1.0, 30 days = 0.9, 365 days = 0.5
            import math
            score = math.exp(-age_days / 365)
            
            # Boost for very recent (last 7 days)
            if age_days <= 7:
                score = min(1.0, score * 1.1)
            
            return score
            
        except Exception as e:
            logger.debug({"msg": "Recency calculation failed", "error": str(e)})
            return 0.75
    
    def _calculate_coverage(
        self,
        query_terms: Optional[List[str]],
        result_text: Optional[str]
    ) -> float:
        """
        Calculate query coverage score.
        Measures how many query terms appear in the result.
        """
        if not query_terms or not result_text:
            return 0.75  # Neutral if no data
        
        result_lower = result_text.lower()
        
        matches = sum(
            1 for term in query_terms
            if term.lower() in result_lower
        )
        
        coverage = matches / len(query_terms)
        
        # Boost for exact phrase matches
        query_phrase = " ".join(query_terms).lower()
        if query_phrase in result_lower:
            coverage = min(1.0, coverage * 1.2)
        
        return coverage
    
    def _apply_sigmoid(self, x: float) -> float:
        """Apply sigmoid function to get realistic distribution."""
        # Using a shifted sigmoid to center around 0.5
        import math
        return 1 / (1 + math.exp(-5 * (x - 0.5)))
    
    def batch_calculate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Tuple[float, ConfidenceFactors]]:
        """
        Calculate confidence for multiple results.
        
        Args:
            results: List of result dictionaries with keys:
                - similarity_score
                - source
                - metadata
                - query_terms
                - result_text
                
        Returns:
            List of (confidence_score, factors) tuples
        """
        return [
            self.calculate_confidence(
                similarity_score=r.get('similarity_score', 0),
                source=r.get('source', 'unknown'),
                metadata=r.get('metadata'),
                query_terms=r.get('query_terms'),
                result_text=r.get('result_text')
            )
            for r in results
        ]
    
    def get_confidence_level(self, score: float) -> str:
        """
        Get human-readable confidence level.
        
        Args:
            score: Confidence score (0-1)
            
        Returns:
            Confidence level string
        """
        if score >= 0.9:
            return "Very High"
        elif score >= 0.75:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def explain_confidence(
        self,
        confidence: float,
        factors: ConfidenceFactors
    ) -> str:
        """
        Generate human-readable explanation of confidence score.
        
        Args:
            confidence: Final confidence score
            factors: Individual confidence factors
            
        Returns:
            Explanation string
        """
        level = self.get_confidence_level(confidence)
        
        explanations = []
        
        # Identify strongest and weakest factors
        factor_dict = factors.to_dict()
        strongest = max(factor_dict.items(), key=lambda x: x[1])
        weakest = min(factor_dict.items(), key=lambda x: x[1])
        
        explanations.append(f"Overall confidence: {level} ({confidence:.2%})")
        
        if strongest[1] > 0.8:
            explanations.append(
                f"Strong {strongest[0].replace('_', ' ')} ({strongest[1]:.2%})"
            )
        
        if weakest[1] < 0.5:
            explanations.append(
                f"Low {weakest[0].replace('_', ' ')} ({weakest[1]:.2%})"
            )
        
        return "; ".join(explanations)


# Global scorer instance
_default_scorer: Optional[ConfidenceScorer] = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create the default confidence scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = ConfidenceScorer()
    return _default_scorer


def calculate_confidence(
    similarity_score: float,
    source: str = "unknown",
    **kwargs
) -> float:
    """
    Convenience function to calculate confidence.
    
    Args:
        similarity_score: Vector similarity score
        source: Source identifier
        **kwargs: Additional parameters
        
    Returns:
        Confidence score (0-1)
    """
    scorer = get_confidence_scorer()
    confidence, _ = scorer.calculate_confidence(
        similarity_score=similarity_score,
        source=source,
        **kwargs
    )
    return confidence


__all__ = [
    'ConfidenceScorer',
    'ConfidenceFactors',
    'calculate_confidence',
    'get_confidence_scorer'
]
