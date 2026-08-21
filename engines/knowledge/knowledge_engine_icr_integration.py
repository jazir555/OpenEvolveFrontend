"""
Knowledge Engine + ICR Integration Module

This module provides ICR (Iterative Contextual Refinements) integration
for the Knowledge Engine, enabling:
- Pattern learning from knowledge operations
- Prediction of knowledge retrieval quality
- Adaptive threshold adjustment for knowledge validation
- Learning from knowledge graph outcomes

ICR Integration Points:
- Knowledge extraction patterns
- Retrieval quality patterns
- Graph update patterns
- Query optimization patterns
- Validation outcome patterns
"""
from __future__ import annotations


from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

# ICR Integration
try:
    from icr_integration import get_icr_integration, ICRPatternType, ICRIntegration
    ICR_AVAILABLE = True
except ImportError:
    ICR_AVAILABLE = False
    get_icr_integration = None
    ICRPatternType = None
    ICRIntegration = None

logger = logging.getLogger(__name__)


class KnowledgeEngineICRIntegration:
    """
    ICR Integration for Knowledge Engine operations.
    
    Features:
    - Stores knowledge operation patterns for learning
    - Predicts retrieval quality and validation outcomes
    - Adapts thresholds based on historical performance
    - Learns from knowledge graph updates and queries
    """

    def __init__(self, enable_icr: bool = True):
        """
        Initialize Knowledge Engine ICR integration.
        
        Args:
            enable_icr: Enable ICR pattern learning
        """
        self.enable_icr = enable_icr and ICR_AVAILABLE
        self.icr = None
        self.operation_history: List[Dict[str, Any]] = []
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        
        if self.enable_icr:
            try:
                self.icr = get_icr_integration()
                if self.icr:
                    self.icr.enable()
            except Exception as e:
                logger.warning(f"Failed to initialize ICR for Knowledge Engine: {e}")
                self.enable_icr = False
                self.icr = None

    def record_extraction_outcome(
        self,
        source_type: str,
        entities_extracted: int,
        relationships_extracted: int,
        quality_score: float,
        duration_seconds: float
    ) -> str:
        """
        Record knowledge extraction outcome for learning.
        
        Args:
            source_type: Type of source (document, api, database, etc.)
            entities_extracted: Number of entities extracted
            relationships_extracted: Number of relationships extracted
            quality_score: Extraction quality score (0-1)
            duration_seconds: Extraction duration
            
        Returns:
            Pattern ID if stored
        """
        if not self.enable_icr or not self.icr:
            return ""
        
        # Determine success based on quality score
        success = quality_score >= 0.7
        
        pattern_id = self.icr.store_pattern(
            pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
            passed=success,
            context={
                "content_type": "knowledge_extraction",
                "source_type": source_type,
                "complexity_score": min(10, int((entities_extracted + relationships_extracted) / 10))
            },
            metrics={
                "quality_score": quality_score,
                "entities_count": entities_extracted,
                "relationships_count": relationships_extracted,
                "duration_seconds": duration_seconds,
                "extraction_rate": (entities_extracted + relationships_extracted) / max(duration_seconds, 0.1)
            }
        )
        
        # Store in local cache
        cache_key = f"extraction_{source_type}"
        if cache_key not in self.pattern_cache:
            self.pattern_cache[cache_key] = []
        
        self.pattern_cache[cache_key].append({
            "type": "extraction",
            "source_type": source_type,
            "entities": entities_extracted,
            "relationships": relationships_extracted,
            "quality": quality_score,
            "duration": duration_seconds,
            "success": success,
            "pattern_id": pattern_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update adaptive threshold
        self._update_threshold("extraction", success)
        
        return pattern_id

    def record_retrieval_outcome(
        self,
        query_type: str,
        results_count: int,
        relevance_score: float,
        latency_ms: float,
        cache_hit: bool = False
    ) -> str:
        """
        Record knowledge retrieval outcome for learning.
        
        Args:
            query_type: Type of query (semantic, keyword, graph, etc.)
            results_count: Number of results returned
            relevance_score: Average relevance score (0-1)
            latency_ms: Query latency in milliseconds
            cache_hit: Whether result was from cache
            
        Returns:
            Pattern ID if stored
        """
        if not self.enable_icr or not self.icr:
            return ""
        
        # Success = good relevance and reasonable latency
        success = relevance_score >= 0.6 and latency_ms <= 1000
        
        pattern_id = self.icr.store_pattern(
            pattern_type=ICRPatternType.RESOURCE_USAGE,
            passed=success,
            context={
                "content_type": "knowledge_retrieval",
                "query_type": query_type,
                "cache_hit": cache_hit
            },
            metrics={
                "relevance_score": relevance_score,
                "results_count": results_count,
                "latency_ms": latency_ms,
                "efficiency": relevance_score / max(latency_ms, 1) * 1000
            }
        )
        
        # Store in local cache
        cache_key = f"retrieval_{query_type}"
        if cache_key not in self.pattern_cache:
            self.pattern_cache[cache_key] = []
        
        self.pattern_cache[cache_key].append({
            "type": "retrieval",
            "query_type": query_type,
            "results": results_count,
            "relevance": relevance_score,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "success": success,
            "pattern_id": pattern_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update adaptive threshold
        self._update_threshold("retrieval", success)
        
        return pattern_id

    def record_graph_update_outcome(
        self,
        update_type: str,
        nodes_added: int,
        edges_added: int,
        validation_passed: bool,
        duration_seconds: float
    ) -> str:
        """
        Record knowledge graph update outcome for learning.
        
        Args:
            update_type: Type of update (add, update, delete, merge)
            nodes_added: Number of nodes added
            edges_added: Number of edges added
            validation_passed: Whether validation passed
            duration_seconds: Update duration
            
        Returns:
            Pattern ID if stored
        """
        if not self.enable_icr or not self.icr:
            return ""
        
        pattern_id = self.icr.store_pattern(
            pattern_type=ICRPatternType.GAUNTLET_OUTCOME,
            passed=validation_passed,
            context={
                "content_type": "graph_update",
                "update_type": update_type,
                "complexity_score": min(10, int((nodes_added + edges_added) / 5))
            },
            metrics={
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "duration_seconds": duration_seconds,
                "update_rate": (nodes_added + edges_added) / max(duration_seconds, 0.1)
            }
        )
        
        # Store in local cache
        cache_key = f"graph_update_{update_type}"
        if cache_key not in self.pattern_cache:
            self.pattern_cache[cache_key] = []
        
        self.pattern_cache[cache_key].append({
            "type": "graph_update",
            "update_type": update_type,
            "nodes": nodes_added,
            "edges": edges_added,
            "validation_passed": validation_passed,
            "duration": duration_seconds,
            "pattern_id": pattern_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update adaptive threshold
        self._update_threshold("graph_update", validation_passed)
        
        return pattern_id

    def predict_retrieval_quality(
        self,
        query_type: str,
        query_complexity: int = 5
    ) -> Dict[str, Any]:
        """
        Predict retrieval quality based on ICR patterns.
        
        Args:
            query_type: Type of query
            query_complexity: Query complexity (1-10)
            
        Returns:
            Prediction results with confidence
        """
        if not self.enable_icr or not self.icr:
            return {
                "predicted": False,
                "reason": "ICR integration not available"
            }
        
        prediction = self.icr.predict(
            pattern_type=ICRPatternType.RESOURCE_USAGE,
            context={
                "content_type": "knowledge_retrieval",
                "query_type": query_type,
                "complexity_score": query_complexity
            }
        )
        
        return {
            "predicted": True,
            "predicted_outcome": prediction.predicted_outcome,
            "probability": prediction.probability,
            "confidence": prediction.confidence,
            "reason": prediction.reason,
            "pattern_count": prediction.pattern_count,
            "recommended_action": prediction.recommended_action,
            "expected_latency_ms": self._estimate_latency(query_type)
        }

    def recommend_query_optimization(
        self,
        query_type: str,
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Recommend query optimization based on ICR patterns.
        
        Args:
            query_type: Type of query
            current_performance: Current performance metrics
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            "query_type": query_type,
            "recommendations": [],
            "priority": "normal"
        }
        
        # Analyze current performance
        latency_ms = current_performance.get("latency_ms", 0)
        relevance = current_performance.get("relevance_score", 0)
        cache_hit = current_performance.get("cache_hit", False)
        
        # Latency optimization
        if latency_ms > 500:
            recommendations["recommendations"].append({
                "type": "latency",
                "action": "Consider query caching or indexing",
                "expected_improvement": "30-50% latency reduction"
            })
            recommendations["priority"] = "high"
        
        # Relevance optimization
        if relevance < 0.6:
            recommendations["recommendations"].append({
                "type": "relevance",
                "action": "Review query semantics or embedding model",
                "expected_improvement": "20-40% relevance improvement"
            })
            recommendations["priority"] = "critical"
        
        # Cache optimization
        if not cache_hit and latency_ms > 200:
            recommendations["recommendations"].append({
                "type": "caching",
                "action": "Enable result caching for this query type",
                "expected_improvement": "80-90% latency reduction for repeated queries"
            })
        
        # Add ICR-based recommendations
        if self.enable_icr and self.icr:
            stats = self.get_statistics()
            if query_type in stats:
                avg_relevance = stats[query_type].get("avg_relevance", 0)
                if relevance < avg_relevance * 0.8:
                    recommendations["recommendations"].append({
                        "type": "pattern_based",
                        "action": f"Query performing below average ({relevance:.2f} vs {avg_relevance:.2f})",
                        "expected_improvement": "10-20% with query refinement"
                    })
        
        return recommendations

    def _update_threshold(self, operation_type: str, success: bool) -> None:
        """
        Update adaptive threshold based on outcome.
        
        Args:
            operation_type: Type of operation
            success: Whether operation succeeded
        """
        current = self.adaptive_thresholds.get(operation_type, 0.5)
        
        if success:
            # Success - slightly lower threshold
            new_threshold = max(0.3, current - 0.02)
        else:
            # Failure - raise threshold
            new_threshold = min(0.9, current + 0.05)
        
        self.adaptive_thresholds[operation_type] = new_threshold

    def _estimate_latency(self, query_type: str) -> float:
        """
        Estimate query latency based on historical patterns.
        
        Args:
            query_type: Type of query
            
        Returns:
            Estimated latency in milliseconds
        """
        cache_key = f"retrieval_{query_type}"
        if cache_key not in self.pattern_cache or not self.pattern_cache[cache_key]:
            return 200.0  # Default estimate
        
        # Calculate average latency from history
        latencies = [p.get("latency_ms", 200) for p in self.pattern_cache[cache_key] if "latency_ms" in p]
        return sum(latencies) / len(latencies) if latencies else 200.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about knowledge operations and ICR patterns.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "icr_enabled": self.enable_icr,
            "operation_types": list(self.pattern_cache.keys()),
            "adaptive_thresholds": self.adaptive_thresholds.copy()
        }
        
        # Calculate statistics per operation type
        for op_type, patterns in self.pattern_cache.items():
            if patterns:
                total = len(patterns)
                successful = sum(1 for p in patterns if p.get("success", False))
                
                # Type-specific metrics
                if "extraction" in op_type:
                    qualities = [p.get("quality", 0) for p in patterns if "quality" in p]
                    avg_quality = sum(qualities) / len(qualities) if qualities else 0
                    
                    stats[op_type] = {
                        "total_operations": total,
                        "success_rate": successful / total if total > 0 else 0.0,
                        "avg_quality_score": avg_quality,
                        "avg_entities": sum(p.get("entities", 0) for p in patterns) / total
                    }
                
                elif "retrieval" in op_type:
                    relevances = [p.get("relevance", 0) for p in patterns if "relevance" in p]
                    latencies = [p.get("latency_ms", 0) for p in patterns if "latency_ms" in p]
                    
                    stats[op_type] = {
                        "total_queries": total,
                        "success_rate": successful / total if total > 0 else 0.0,
                        "avg_relevance": sum(relevances) / len(relevances) if relevances else 0,
                        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                        "cache_hit_rate": sum(1 for p in patterns if p.get("cache_hit", False)) / total
                    }
                
                elif "graph_update" in op_type:
                    stats[op_type] = {
                        "total_updates": total,
                        "validation_pass_rate": successful / total if total > 0 else 0.0,
                        "avg_nodes": sum(p.get("nodes", 0) for p in patterns) / total,
                        "avg_edges": sum(p.get("edges", 0) for p in patterns) / total
                    }
        
        return stats


# Global instance
_knowledge_icr: Optional[KnowledgeEngineICRIntegration] = None


def get_knowledge_icr_integration() -> KnowledgeEngineICRIntegration:
    """Get or create global Knowledge Engine ICR integration instance."""
    global _knowledge_icr
    if _knowledge_icr is None:
        _knowledge_icr = KnowledgeEngineICRIntegration()
    return _knowledge_icr


def initialize_knowledge_icr_integration(enable_icr: bool = True) -> KnowledgeEngineICRIntegration:
    """Initialize Knowledge Engine ICR integration."""
    global _knowledge_icr
    _knowledge_icr = KnowledgeEngineICRIntegration(enable_icr=enable_icr)
    return _knowledge_icr
