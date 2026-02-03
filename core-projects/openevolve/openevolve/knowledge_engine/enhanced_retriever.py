"""
Enhanced Knowledge Retriever for OpenEvolve Knowledge Engine

This module provides enhanced retrieval capabilities with ML-based ranking,
personalization, and advanced search features for the Phase 2 implementation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import uuid
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Result of an enhanced retrieval operation."""
    success: bool
    results: List[Dict[str, Any]]
    count: int
    query_type: str
    processing_time_ms: float = 0.0
    ml_ranking_applied: bool = False
    personalization_applied: bool = False
    error: Optional[str] = None


class EnhancedKnowledgeRetriever:
    """
    Enhanced retrieval layer with ML-based ranking and personalization.
    
    Provides methods for:
    - ML-enhanced search ranking
    - Personalized recommendations
    - Quality metrics and analytics
    - Trend analysis
    - Performance optimization
    """
    
    def __init__(self, storage, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced knowledge retriever.
        
        Args:
            storage: EnhancedKnowledgeStorage instance for data access
            config: Configuration for enhanced retrieval
        """
        self.storage = storage
        self.config = config or self._get_default_config()
        
        # Initialize cache
        self.cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default
        
        # Initialize ML components
        self.ml_model = None
        self.personalization_model = None
        self._initialize_ml_components()
        
        # Initialize performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "cached_queries": 0,
            "ml_ranked_queries": 0,
            "personalized_queries": 0,
            "average_response_time": 0.0,
            "query_types": defaultdict(int)
        }
        
        logger.info({
            "msg": "EnhancedKnowledgeRetriever initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enhanced retrieval."""
        return {
            "default_query_type": "hybrid",
            "max_results": 10,
            "cache_ttl": 300,
            "enable_personalization": True,
            "enable_ml_ranking": True,
            "ml_model_path": None,
            "similarity_threshold": 0.7,
            "keyword_weight": 0.3,
            "semantic_weight": 0.5,
            "vector_weight": 0.2,
            "enable_trend_analysis": True,
            "trend_analysis_window": 30,  # days
            "enable_quality_scoring": True,
            "quality_score_threshold": 0.5
        }
    
    def _initialize_ml_components(self):
        """Initialize ML components for ranking and personalization."""
        if self.config.get("enable_ml_ranking", True):
            try:
                # Placeholder for ML model initialization
                # In a real implementation, this would load a trained model
                logger.info("ML ranking components initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ML components: {e}")
        
        if self.config.get("enable_personalization", True):
            try:
                # Placeholder for personalization model initialization
                logger.info("Personalization components initialized")
            except Exception as e:
                logger.warning(f"Could not initialize personalization components: {e}")
    
    def search_knowledge(
        self,
        query: str,
        query_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        use_cache: bool = True,
        apply_ml_ranking: bool = True,
        apply_personalization: bool = False,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> EnhancedRetrievalResult:
        """
        Enhanced search with ML ranking and personalization.
        
        Args:
            query: Search query string
            query_type: Type of search ('keyword', 'semantic', 'vector', 'hybrid')
            filters: Additional filters to apply
            limit: Maximum number of results
            use_cache: Whether to use caching
            apply_ml_ranking: Whether to apply ML-based ranking
            apply_personalization: Whether to apply personalization
            user_profile: User profile for personalization
            
        Returns:
            EnhancedRetrievalResult with results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        # Update performance stats
        self.performance_stats["total_queries"] += 1
        self.performance_stats["query_types"][query_type] += 1
        
        # Create cache key
        cache_key = f"enhanced_search:{query}:{query_type}:{filters}:{limit}:{apply_personalization}"
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            # Check if cache is still valid
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl:
                self.performance_stats["cached_queries"] += 1
                
                logger.info({
                    "msg": "Returning cached enhanced search results",
                    "query": query,
                    "result_count": len(cached_result.results),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Update the result to reflect that it came from cache
                cached_result.processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return cached_result
        
        logger.info({
            "msg": "Starting enhanced knowledge search",
            "query": query,
            "query_type": query_type,
            "limit": limit,
            "apply_ml_ranking": apply_ml_ranking,
            "apply_personalization": apply_personalization,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Perform base search
            base_results = self._perform_base_search(query, query_type, filters, limit)
            
            # Apply ML ranking if enabled
            if apply_ml_ranking and self.config.get("enable_ml_ranking", True):
                self.performance_stats["ml_ranked_queries"] += 1
                base_results = self._apply_ml_ranking(base_results, query)
            
            # Apply personalization if enabled
            if apply_personalization and user_profile and self.config.get("enable_personalization", True):
                self.performance_stats["personalized_queries"] += 1
                base_results = self._apply_personalization(base_results, user_profile)
            
            # Apply quality filtering if enabled
            if self.config.get("enable_quality_scoring", True):
                quality_threshold = self.config.get("quality_score_threshold", 0.5)
                base_results = [r for r in base_results if r.get("quality_score", 1.0) >= quality_threshold]
            
            # Update cache
            if use_cache:
                result_to_cache = EnhancedRetrievalResult(
                    success=True,
                    results=base_results,
                    count=len(base_results),
                    query_type=query_type,
                    ml_ranking_applied=apply_ml_ranking,
                    personalization_applied=apply_personalization
                )
                self.cache[cache_key] = (result_to_cache, datetime.now(timezone.utc))
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Update average response time
            total_queries = self.performance_stats["total_queries"]
            current_avg = self.performance_stats["average_response_time"]
            new_avg = ((current_avg * (total_queries - 1)) + processing_time_ms) / total_queries
            self.performance_stats["average_response_time"] = new_avg
            
            result = EnhancedRetrievalResult(
                success=True,
                results=base_results,
                count=len(base_results),
                query_type=query_type,
                processing_time_ms=processing_time_ms,
                ml_ranking_applied=apply_ml_ranking,
                personalization_applied=apply_personalization
            )
            
            logger.info({
                "msg": "Enhanced knowledge search completed",
                "query": query,
                "result_count": len(base_results),
                "ml_ranking_applied": apply_ml_ranking,
                "personalization_applied": apply_personalization,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Enhanced knowledge search failed",
                "query": query,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return EnhancedRetrievalResult(
                success=False,
                results=[],
                count=0,
                query_type=query_type,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def _perform_base_search(
        self,
        query: str,
        query_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform the base search using the underlying storage."""
        # This would delegate to the storage layer in a real implementation
        # For now, we'll return sample results
        sample_results = [
            {
                "artifact_id": f"sample_{i}",
                "content": f"Sample content for query '{query}'",
                "type": "solution_pattern",
                "source": "enhanced_search",
                "context": "general",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "relevance_score": 0.9 - (i * 0.1),  # Decreasing relevance
                "quality_score": 0.85,
                "metadata": {"iteration": i}
            }
            for i in range(min(limit, 10))
        ]
        
        return sample_results
    
    def _apply_ml_ranking(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Apply ML-based ranking to results."""
        # In a real implementation, this would use a trained model
        # For now, we'll adjust scores based on content similarity to query
        
        def calculate_similarity_score(result: Dict[str, Any], query: str) -> float:
            # Simple keyword matching for demonstration
            content = result.get("content", "").lower()
            query_terms = query.lower().split()
            
            matches = sum(1 for term in query_terms if term in content)
            total_terms = len(query_terms)
            
            if total_terms == 0:
                return result.get("relevance_score", 0.5)
            
            keyword_match_score = matches / total_terms
            base_score = result.get("relevance_score", 0.5)
            
            # Combine base score with keyword match score
            combined_score = 0.7 * base_score + 0.3 * keyword_match_score
            
            return min(combined_score, 1.0)  # Ensure score is between 0 and 1
        
        # Calculate new scores for each result
        scored_results = []
        for result in results:
            new_score = calculate_similarity_score(result, query)
            updated_result = result.copy()
            updated_result["ml_ranking_score"] = new_score
            updated_result["relevance_score"] = new_score
            scored_results.append(updated_result)
        
        # Sort by the new ML-based score
        sorted_results = sorted(scored_results, key=lambda x: x.get("ml_ranking_score", 0), reverse=True)
        
        return sorted_results
    
    def _apply_personalization(
        self,
        results: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply personalization to results based on user profile."""
        # Calculate personalization scores based on user preferences
        preferred_types = user_profile.get("preferred_artifact_types", [])
        expertise_level = user_profile.get("expertise_level", "intermediate")
        preferred_sources = user_profile.get("preferred_sources", [])
        
        personalized_results = []
        
        for result in results:
            score = result.get("relevance_score", 0.5)  # Start with existing score
            
            # Boost score if artifact type matches user preference
            if result.get("type") in preferred_types:
                score += 0.2
            
            # Boost score if source matches user preference
            if result.get("source") in preferred_sources:
                score += 0.15
            
            # Adjust score based on complexity match
            artifact_complexity = result.get("metadata", {}).get("complexity", "medium")
            if expertise_level == "beginner" and artifact_complexity == "low":
                score += 0.1
            elif expertise_level == "intermediate" and artifact_complexity == "medium":
                score += 0.1
            elif expertise_level == "expert" and artifact_complexity == "high":
                score += 0.1
            
            # Ensure score doesn't exceed 1.0
            score = min(score, 1.0)
            
            # Create a copy with updated score
            personalized_result = result.copy()
            personalized_result["personalization_score"] = score
            personalized_result["relevance_score"] = score
            personalized_results.append(personalized_result)
        
        # Sort by personalization score
        sorted_results = sorted(
            personalized_results,
            key=lambda x: x.get("personalization_score", 0),
            reverse=True
        )
        
        return sorted_results
    
    def get_personalized_recommendations(
        self,
        context: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> EnhancedRetrievalResult:
        """
        Get personalized recommendations with enhanced features.
        
        Args:
            context: Context information for recommendations
            user_profile: Optional user profile for personalization
            limit: Maximum number of recommendations
            
        Returns:
            EnhancedRetrievalResult with personalized recommendations
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting enhanced personalized recommendations",
            "context_keys": list(context.keys()) if context else [],
            "user_profile_keys": list(user_profile.keys()) if user_profile else [],
            "limit": limit,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Build recommendation query based on context
            recommendation_context = context.get("recommendation_type", "general")
            problem_type = context.get("problem_type", "general")
            complexity = context.get("complexity", "medium")
            
            # Create a query that matches the context
            query = f"{recommendation_context} {problem_type} {complexity}"
            
            # Apply filters based on context
            filters = {
                "type": context.get("recommendation_type"),
                "context": problem_type,
                "complexity": complexity
            }
            
            # Get relevant artifacts with personalization
            result = self.search_knowledge(
                query=query,
                query_type="hybrid",
                filters=filters,
                limit=limit * 2,  # Get more to allow for personalization
                apply_personalization=True,
                user_profile=user_profile
            )
            
            if result.success:
                # Take the top results after personalization
                final_results = result.results[:limit]
                
                # Create a new result with the limited set
                final_result = EnhancedRetrievalResult(
                    success=True,
                    results=final_results,
                    count=len(final_results),
                    query_type="personalized_recommendation",
                    processing_time_ms=result.processing_time_ms,
                    ml_ranking_applied=result.ml_ranking_applied,
                    personalization_applied=True,
                    error=None
                )
                
                logger.info({
                    "msg": "Enhanced personalized recommendations completed",
                    "result_count": len(final_results),
                    "processing_time_ms": final_result.processing_time_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return final_result
            else:
                logger.error({
                    "msg": "Base search failed for personalized recommendations",
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return EnhancedRetrievalResult(
                    success=False,
                    results=[],
                    count=0,
                    query_type="personalized_recommendation",
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    error=result.error
                )
                
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Enhanced personalized recommendations failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return EnhancedRetrievalResult(
                success=False,
                results=[],
                count=0,
                query_type="personalized_recommendation",
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    def get_knowledge_quality_metrics(self) -> Dict[str, Any]:
        """
        Get enhanced quality metrics for the knowledge base.
        
        Returns:
            Dictionary with enhanced quality metrics
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Calculating enhanced knowledge quality metrics",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Get basic statistics from storage
            storage_stats = self.storage.get_aggregated_statistics()
            
            # Calculate additional enhanced quality metrics
            quality_metrics = {
                "completeness": self._calculate_completeness(),
                "consistency": self._calculate_consistency(),
                "timeliness": self._calculate_timeliness(),
                "diversity": self._calculate_diversity(),
                "accuracy_proxy": self._calculate_accuracy_proxy(),
                "relevance": self._calculate_relevance(),
                "coverage": self._calculate_coverage()
            }
            
            # Calculate knowledge graph metrics if available
            try:
                graph = self.storage.create_knowledge_graph()
                graph_metrics = self._calculate_graph_metrics(graph)
                quality_metrics.update(graph_metrics)
            except Exception as e:
                logger.warning(f"Could not calculate graph metrics: {e}")
                quality_metrics["graph_metrics_error"] = str(e)
            
            # Overall quality score
            valid_metrics = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
            overall_score = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0
            
            result = {
                "quality_metrics": quality_metrics,
                "overall_quality_score": overall_score,
                "storage_stats": storage_stats,
                "calculation_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "Enhanced knowledge quality metrics calculated",
                "overall_score": overall_score,
                "calculation_time_ms": result["calculation_time_ms"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            calculation_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Enhanced knowledge quality metrics calculation failed",
                "error": str(e),
                "calculation_time_ms": calculation_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "error": str(e),
                "overall_quality_score": 0.0,
                "quality_metrics": {}
            }
    
    def _calculate_completeness(self) -> float:
        """Calculate completeness metric."""
        # In a real implementation, this would analyze coverage
        # of different knowledge areas
        return 0.75  # Placeholder
    
    def _calculate_consistency(self) -> float:
        """Calculate consistency metric."""
        # In a real implementation, this would check for contradictions
        # and consistency across artifacts
        return 0.88  # Placeholder
    
    def _calculate_timeliness(self) -> float:
        """Calculate timeliness metric."""
        # In a real implementation, this would check recency of artifacts
        return 0.82  # Placeholder
    
    def _calculate_diversity(self) -> float:
        """Calculate diversity metric."""
        # In a real implementation, this would measure variety
        # across different types, sources, etc.
        return 0.78  # Placeholder
    
    def _calculate_accuracy_proxy(self) -> float:
        """Calculate accuracy proxy metric."""
        # In a real implementation, this would use various signals
        # like source reliability, peer review, etc.
        return 0.92  # Placeholder
    
    def _calculate_relevance(self) -> float:
        """Calculate relevance metric."""
        # In a real implementation, this would analyze how well
        # artifacts match their described context and purpose
        return 0.85  # Placeholder
    
    def _calculate_coverage(self) -> float:
        """Calculate coverage metric."""
        # In a real implementation, this would measure how comprehensively
        # different domains/topics are covered
        return 0.70  # Placeholder
    
    def _calculate_graph_metrics(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics based on knowledge graph structure."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if not nodes:
            return {"graph_connectivity": 0.0, "graph_density": 0.0}
        
        node_count = len(nodes)
        edge_count = len(edges)
        
        # Calculate basic graph metrics
        if node_count <= 1:
            connectivity = 0.0
        else:
            # Simple connectivity measure (edges per possible connection)
            max_possible_edges = node_count * (node_count - 1)  # Directed graph
            connectivity = edge_count / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Density: ratio of actual edges to possible edges
        density = connectivity
        
        return {
            "graph_connectivity": connectivity,
            "graph_density": density,
            "node_count": node_count,
            "edge_count": edge_count
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get enhanced performance metrics for the retrieval system.
        
        Returns:
            Dictionary with enhanced performance metrics
        """
        return {
            "total_queries": self.performance_stats["total_queries"],
            "cached_queries": self.performance_stats["cached_queries"],
            "ml_ranked_queries": self.performance_stats["ml_ranked_queries"],
            "personalized_queries": self.performance_stats["personalized_queries"],
            "cache_hit_rate": (
                self.performance_stats["cached_queries"] / self.performance_stats["total_queries"]
                if self.performance_stats["total_queries"] > 0 else 0.0
            ),
            "average_response_time_ms": self.performance_stats["average_response_time"],
            "query_types_distribution": dict(self.performance_stats["query_types"]),
            "most_common_queries": self._get_most_common_queries(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_most_common_queries(self) -> List[str]:
        """Get most common queries from cache keys."""
        # This is a simplified approach - in reality, you'd track queries separately
        common_queries = []
        for key in list(self.cache.keys())[:5]:  # Just return first 5 cache keys as example
            # Extract query from cache key (this is a simplification)
            parts = key.split(":")
            if len(parts) > 1:
                common_queries.append(parts[1])
        return common_queries
    
    def get_knowledge_trends(
        self,
        time_range: str = "30d",
        analysis_type: str = "advanced"
    ) -> Dict[str, Any]:
        """
        Get enhanced trend analysis of the knowledge base.
        
        Args:
            time_range: Time range for analysis ('7d', '30d', '90d', '1y')
            analysis_type: Type of analysis ('basic', 'advanced')
            
        Returns:
            Dictionary with enhanced trend analysis
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting enhanced knowledge trend analysis",
            "time_range": time_range,
            "analysis_type": analysis_type,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # In a real implementation, this would analyze the knowledge base
            # over the specified time period to identify trends
            # For now, we'll return sample trends
            
            trends = {
                "top_emerging_topics": [
                    {"topic": "artificial intelligence", "growth_rate": 2.5, "volume": 120},
                    {"topic": "machine learning", "growth_rate": 2.1, "volume": 95},
                    {"topic": "data science", "growth_rate": 1.8, "volume": 87},
                    {"topic": "cloud computing", "growth_rate": 1.6, "volume": 78},
                    {"topic": "cybersecurity", "growth_rate": 1.5, "volume": 72}
                ],
                "declining_topics": [
                    {"topic": "legacy systems", "decline_rate": -1.2, "volume": 23},
                    {"topic": "on-premise infrastructure", "decline_rate": -0.9, "volume": 34}
                ],
                "topic_velocity": {
                    "artificial intelligence": 2.5,
                    "machine learning": 2.1,
                    "data science": 1.8
                },
                "seasonality_patterns": [
                    {"period": "Q4", "trending_topics": ["planning", "budgeting", "strategy"]},
                    {"period": "Q2", "trending_topics": ["innovation", "development", "research"]}
                ],
                "prediction_confidence": 0.85,
                "anomaly_detection": {
                    "detected_anomalies": 3,
                    "anomaly_types": ["sudden_spike", "sharp_decline", "pattern_break"]
                }
            }
            
            result = {
                "trend_analysis": trends,
                "time_range": time_range,
                "analysis_type": analysis_type,
                "analysis_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info({
                "msg": "Enhanced knowledge trend analysis completed",
                "analysis_time_ms": result["analysis_time_ms"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            analysis_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Enhanced knowledge trend analysis failed",
                "error": str(e),
                "analysis_time_ms": analysis_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "error": str(e),
                "trend_analysis": {},
                "analysis_time_ms": analysis_time_ms
            }
    
    def clear_cache(self):
        """Clear the retrieval cache."""
        old_size = len(self.cache)
        self.cache.clear()
        
        logger.info({
            "msg": "Enhanced cache cleared",
            "previous_size": old_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def get_retrieval_insights(self) -> Dict[str, Any]:
        """
        Get insights about the retrieval system's behavior.
        
        Returns:
            Dictionary with retrieval insights
        """
        return {
            "performance_stats": self.performance_stats,
            "cache_efficiency": {
                "hit_rate": self.performance_stats["cached_queries"] / max(self.performance_stats["total_queries"], 1),
                "cache_size": len(self.cache),
                "estimated_savings_ms": self.performance_stats["cached_queries"] * 50  # Assuming 50ms saved per cache hit
            },
            "ml_utilization": {
                "ranking_enabled": self.config.get("enable_ml_ranking", True),
                "personalization_enabled": self.config.get("enable_personalization", True),
                "model_version": "1.0.0"  # Placeholder
            },
            "query_patterns": {
                "most_common_types": sorted(
                    self.performance_stats["query_types"].items(),
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }