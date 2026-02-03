"""
Result Ranking and Fusion

Ranking algorithms for search results including RRF and cross-encoder reranking.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .search import SearchResult

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF)
    
    Combines results from multiple sources using the formula:
    score = sum(1 / (k + rank)) for each list
    
    Where k is a constant (typically 60) that smooths the impact of rankings.
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        result_lists: List[List[SearchResult]],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Fuse multiple result lists using RRF
        
        Args:
            result_lists: List of result lists from different sources
            top_k: Number of top results to return
        
        Returns:
            Fused and ranked results
        """
        # Map of doc_id -> aggregated score and metadata
        scores: Dict[str, Dict[str, Any]] = {}
        
        # Process each result list
        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                doc_id = result.id
                
                if doc_id not in scores:
                    scores[doc_id] = {
                        "rrf_score": 0.0,
                        "content": result.content,
                        "metadata": result.metadata,
                        "sources": [],
                        "original_scores": {}
                    }
                
                # RRF score for this rank
                rrf_score = 1.0 / (self.k + rank + 1)
                scores[doc_id]["rrf_score"] += rrf_score
                scores[doc_id]["sources"].append(result.source)
                scores[doc_id]["original_scores"][result.source] = result.score
        
        # Sort by RRF score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )
        
        # Create SearchResult objects
        fused_results = []
        for doc_id, data in sorted_results[:top_k]:
            fused_results.append(SearchResult(
                id=doc_id,
                score=data["rrf_score"],
                content=data["content"],
                metadata={
                    **data["metadata"],
                    "rrf_score": data["rrf_score"],
                    "sources": list(set(data["sources"])),
                    "original_scores": data["original_scores"]
                },
                source="hybrid"
            ))
        
        return fused_results
    
    def fuse_with_weights(
        self,
        result_lists: List[List[SearchResult]],
        weights: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Fuse with different weights for each source
        
        Args:
            result_lists: List of result lists
            weights: Weight for each result list
            top_k: Number of results to return
        
        Returns:
            Weighted fused results
        """
        if len(result_lists) != len(weights):
            raise ValueError("Number of result lists must match number of weights")
        
        scores: Dict[str, Dict[str, Any]] = {}
        
        for weight, result_list in zip(weights, result_lists):
            for rank, result in enumerate(result_list):
                doc_id = result.id
                
                if doc_id not in scores:
                    scores[doc_id] = {
                        "rrf_score": 0.0,
                        "content": result.content,
                        "metadata": result.metadata,
                        "sources": [],
                        "original_scores": {}
                    }
                
                # Weighted RRF score
                rrf_score = weight * (1.0 / (self.k + rank + 1))
                scores[doc_id]["rrf_score"] += rrf_score
                scores[doc_id]["sources"].append(result.source)
                scores[doc_id]["original_scores"][result.source] = result.score
        
        # Sort and return
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )
        
        return [
            SearchResult(
                id=doc_id,
                score=data["rrf_score"],
                content=data["content"],
                metadata=data["metadata"],
                source="hybrid"
            )
            for doc_id, data in sorted_results[:top_k]
        ]


class ResultRanker:
    """
    Result ranking with multiple strategies
    """
    
    def __init__(self):
        self.rrf = ReciprocalRankFusion()
    
    def rank(
        self,
        results: List[SearchResult],
        strategy: str = "score",
        query: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Rank results using specified strategy
        
        Args:
            results: List of search results
            strategy: Ranking strategy (score, diversity, recency)
            query: Optional query for relevance scoring
        
        Returns:
            Ranked results
        """
        if strategy == "score":
            return self._rank_by_score(results)
        elif strategy == "diversity":
            return self._rank_by_diversity(results)
        elif strategy == "recency":
            return self._rank_by_recency(results)
        elif strategy == "confidence":
            return self._rank_by_confidence(results)
        else:
            return self._rank_by_score(results)
    
    def _rank_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank by raw score"""
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def _rank_by_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rank with Maximal Marginal Relevance (MMR) for diversity
        
        MMR = λ * Relevance - (1-λ) * max(Similarity to already selected)
        """
        lambda_param = 0.5  # Balance between relevance and diversity
        selected = []
        remaining = results.copy()
        
        while remaining and len(selected) < len(results):
            if not selected:
                # First item: highest relevance
                best = max(remaining, key=lambda r: r.score)
            else:
                # MMR scoring
                best = None
                best_mmr = -float('inf')
                
                for candidate in remaining:
                    # Relevance component
                    relevance = candidate.score
                    
                    # Diversity component (max similarity to selected)
                    max_sim = 0.0
                    for sel in selected:
                        sim = self._similarity(candidate, sel)
                        max_sim = max(max_sim, sim)
                    
                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = candidate
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _rank_by_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank by recency (if timestamp available in metadata)"""
        def get_timestamp(r):
            ts = r.metadata.get('timestamp') or r.metadata.get('created_at')
            if ts:
                from datetime import datetime
                try:
                    return datetime.fromisoformat(ts)
                except:
                    pass
            return datetime.min
        
        return sorted(results, key=get_timestamp, reverse=True)
    
    def _rank_by_confidence(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank by confidence score in metadata"""
        def get_confidence(r):
            return r.metadata.get('confidence', r.score)
        
        return sorted(results, key=get_confidence, reverse=True)
    
    def _similarity(self, a: SearchResult, b: SearchResult) -> float:
        """Calculate similarity between two results"""
        # Simple Jaccard similarity on content tokens
        tokens_a = set(a.content.lower().split())
        tokens_b = set(b.content.lower().split())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        
        return len(intersection) / len(union)
    
    def deduplicate(
        self,
        results: List[SearchResult],
        threshold: float = 0.9
    ) -> List[SearchResult]:
        """
        Remove duplicate results based on content similarity
        
        Args:
            results: List of results
            threshold: Similarity threshold for deduplication
        
        Returns:
            Deduplicated results
        """
        deduplicated = []
        
        for result in results:
            is_duplicate = False
            for existing in deduplicated:
                if self._similarity(result, existing) > threshold:
                    is_duplicate = True
                    # Keep the one with higher score
                    if result.score > existing.score:
                        deduplicated[deduplicated.index(existing)] = result
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def filter_by_threshold(
        self,
        results: List[SearchResult],
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Filter results by minimum score"""
        return [r for r in results if r.score >= min_score]
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> List[SearchResult]:
        """
        Rerank results using a cross-encoder model
        
        Note: This requires the sentence-transformers package
        """
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder(model_name)
            
            # Prepare pairs
            pairs = [(query, r.content) for r in results]
            
            # Get scores
            scores = model.predict(pairs)
            
            # Update results with new scores
            for result, score in zip(results, scores):
                result.score = float(score)
                result.metadata['reranked'] = True
                result.metadata['cross_encoder_score'] = float(score)
            
            # Re-sort
            return sorted(results, key=lambda r: r.score, reverse=True)
            
        except ImportError:
            logger.warning("sentence-transformers not available for reranking")
            return results
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results


class DiversityReranker:
    """
    Rerank results to maximize diversity using MMR
    """
    
    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank with diversity
        
        Args:
            query: Original query
            results: Initial ranked results
            top_k: Number of results to return
        
        Returns:
            Diverse reranked results
        """
        if len(results) <= top_k:
            return results
        
        selected = []
        remaining = results.copy()
        
        while len(selected) < top_k and remaining:
            if not selected:
                # First: highest relevance
                best = max(remaining, key=lambda r: r.score)
            else:
                best = None
                best_mmr = -float('inf')
                
                for candidate in remaining:
                    relevance = candidate.score
                    max_sim = max(
                        self._similarity(candidate, s)
                        for s in selected
                    )
                    
                    mmr = (
                        self.lambda_param * relevance -
                        (1 - self.lambda_param) * max_sim
                    )
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = candidate
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _similarity(self, a: SearchResult, b: SearchResult) -> float:
        """Calculate cosine similarity if embeddings available, else Jaccard"""
        if a.embeddings and b.embeddings:
            # Cosine similarity
            a_vec = np.array(a.embeddings)
            b_vec = np.array(b.embeddings)
            
            norm_a = np.linalg.norm(a_vec)
            norm_b = np.linalg.norm(b_vec)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return np.dot(a_vec, b_vec) / (norm_a * norm_b)
        
        # Fallback to Jaccard
        tokens_a = set(a.content.lower().split())
        tokens_b = set(b.content.lower().split())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        
        return len(intersection) / len(union)
