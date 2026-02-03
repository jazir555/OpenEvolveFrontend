"""
Complexity Router - System 1 / System 2 Decision Engine

Implements the "System 1" Router for latency optimization.
Analyzes query complexity and routes to appropriate processing tier:
- Fast/Cheap (Haiku/Flash) for simple queries
- Deep/Expensive (Knowledge Engine) for complex queries

This creates the illusion of a living consciousness rather than a batch processor.

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

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import json
import hashlib
import re

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based model routing
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class ModelTier(Enum):
    """Model tiers from fastest/cheapest to most capable"""
    FAST = "fast"              # Haiku, Flash - <1s, $0.0001
    BALANCED = "balanced"      # GPT-4o-mini, Sonnet - 2-5s, $0.001
    CAPABLE = "capable"        # GPT-4o, Opus - 5-10s, $0.01
    DEEP = "deep"              # Full Knowledge Engine - 10-60s, $0.10+
    
    @property
    def typical_latency(self) -> float:
        latencies = {
            'fast': 0.5,
            'balanced': 3.0,
            'capable': 8.0,
            'deep': 30.0
        }
        return latencies.get(self.value, 30.0)
    
    @property
    def typical_cost(self) -> float:
        costs = {
            'fast': 0.0001,
            'balanced': 0.001,
            'capable': 0.01,
            'deep': 0.1
        }
        return costs.get(self.value, 0.1)


def _map_adaptive_mdap_score_to_level(score: float) -> 'ComplexityLevel':
    """
    Map Adaptive MDAP complexity score (0-1) to ComplexityLevel enum.
    
    Mapping:
    - score < 0.15: TRIVIAL
    - score < 0.35: SIMPLE
    - score < 0.60: MODERATE
    - score < 0.85: COMPLEX
    - score >= 0.85: DEEP
    
    Args:
        score: Adaptive MDAP complexity score (0-1)
        
    Returns:
        ComplexityLevel enum value
    """
    if score < 0.15:
        return ComplexityLevel.TRIVIAL
    elif score < 0.35:
        return ComplexityLevel.SIMPLE
    elif score < 0.60:
        return ComplexityLevel.MODERATE
    elif score < 0.85:
        return ComplexityLevel.COMPLEX
    else:
        return ComplexityLevel.DEEP


class ComplexityLevel(Enum):
    """Complexity levels for queries"""
    TRIVIAL = 1      # Simple facts, greetings, confirmations
    SIMPLE = 2       # Basic questions, definitions
    MODERATE = 3     # Multi-step reasoning, comparisons
    COMPLEX = 4      # Analysis, synthesis, problem-solving
    DEEP = 5         # Research, verification, multi-source integration


@dataclass
class RouteDecision:
    """Decision made by the router"""
    query: str
    complexity_score: float  # 0-1
    complexity_level: ComplexityLevel
    selected_tier: ModelTier
    reasoning: str
    confidence: float
    estimated_latency: float
    estimated_cost: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query[:100] + '...' if len(self.query) > 100 else self.query,
            'complexity_score': self.complexity_score,
            'complexity_level': self.complexity_level.name,
            'selected_tier': self.selected_tier.value,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'estimated_latency': self.estimated_latency,
            'estimated_cost': self.estimated_cost,
            'timestamp': self.timestamp
        }


class ComplexityAnalyzer:
    """
    Analyzes query complexity using multiple signals:
    - Length and structure
    - Keyword indicators
    - Domain complexity
    - Required reasoning steps
    
    **ACTUAL INTEGRATION**: Uses Adaptive MDAP TaskComplexityClassifier when available,
    with fallback to keyword-based analysis.
    """
    
    # Keywords indicating complexity
    COMPLEXITY_INDICATORS = {
        'trivial': [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'ok', 'yes', 'no',
            'what time', 'what day', 'what is your name'
        ],
        'simple': [
            'what is', 'define', 'explain', 'who is', 'when did', 'where is',
            'how many', 'list', 'name'
        ],
        'moderate': [
            'compare', 'difference', 'similarities', 'how to', 'steps',
            'process', 'why does', 'causes', 'effects'
        ],
        'complex': [
            'analyze', 'evaluate', 'synthesize', 'integrate', 'optimize',
            'design', 'implement', 'debug', 'fix', 'solve', 'prove'
        ],
        'deep': [
            'research', 'investigate', 'comprehensive', 'thorough',
            'verify', 'validate', 'audit', 'assess', 'review'
        ]
    }
    
    # Domain complexity multipliers
    DOMAIN_MULTIPLIERS = {
        'mathematics': 1.2,
        'physics': 1.2,
        'chemistry': 1.1,
        'biology': 1.1,
        'programming': 1.3,
        'law': 1.2,
        'medicine': 1.3,
        'finance': 1.1,
        'general': 1.0
    }
    
    def __init__(self):
        """
        Initialize ComplexityAnalyzer.
        
        **ACTUAL INTEGRATION**: Uses TaskComplexityClassifier from Adaptive MDAP
        when available for more accurate complexity computation.
        """
        self._adaptive_classifier: Optional[Any] = None
        
        if ADAPTIVE_MDAP_AVAILABLE and TaskComplexityClassifier is not None:
            try:
                self._adaptive_classifier = TaskComplexityClassifier()
                logger.info("Adaptive MDAP TaskComplexityClassifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TaskComplexityClassifier: {e}")
                self._adaptive_classifier = None
    
    def analyze(self, query: str, domain: str = 'general') -> Tuple[float, ComplexityLevel]:
        """
        Analyze query complexity.
        
        **ACTUAL INTEGRATION**: Uses Adaptive MDAP TaskComplexityClassifier.compute_complexity()
        when available, with fallback to keyword-based analysis.
        
        Args:
            query: The query to analyze
            domain: Domain context
            
        Returns:
            Tuple of (complexity_score 0-1, complexity_level)
        """
        # **ACTUAL INTEGRATION**: Use Adaptive MDAP if available
        if self._adaptive_classifier is not None:
            try:
                complexity_score = self._adaptive_classifier.compute_complexity(query)
                complexity_level = _map_adaptive_mdap_score_to_level(complexity_score)
                logger.debug({
                    'msg': 'Used Adaptive MDAP for complexity analysis',
                    'query': query[:50] + '...' if len(query) > 50 else query,
                    'score': complexity_score,
                    'level': complexity_level.name
                })
                return complexity_score, complexity_level
            except Exception as e:
                logger.warning(f"Adaptive MDAP analysis failed, falling back: {e}")
                # Fall through to keyword-based analysis
        
        # Fallback: Keyword-based analysis
        """
        Analyze query complexity.
        
        Args:
            query: The query to analyze
            domain: Domain context
            
        Returns:
            Tuple of (complexity_score 0-1, complexity_level)
        """
        query_lower = query.lower()
        
        # Base score from length (longer = more complex, but with diminishing returns)
        length_score = min(len(query) / 500, 0.3)
        
        # Score from keyword indicators
        keyword_score = self._analyze_keywords(query_lower)
        
        # Score from structure
        structure_score = self._analyze_structure(query)
        
        # Domain multiplier
        domain_mult = self.DOMAIN_MULTIPLIERS.get(domain.lower(), 1.0)
        
        # Combine scores
        raw_score = (length_score * 0.2 + keyword_score * 0.5 + structure_score * 0.3) * domain_mult
        
        # Normalize to 0-1
        complexity_score = min(raw_score, 1.0)
        
        # Map to level
        complexity_level = self._score_to_level(complexity_score)
        
        return complexity_score, complexity_level
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Get information about the classifier being used.
        
        Returns:
            Dictionary with classifier information
        """
        if self._adaptive_classifier is not None:
            return {
                'type': 'adaptive_mdap',
                'available': True,
                'classifier': 'TaskComplexityClassifier'
            }
        return {
            'type': 'keyword_based',
            'available': True,
            'classifier': 'ComplexityAnalyzer (fallback)'
        }
    
    def _analyze_keywords(self, query: str) -> float:
        """Analyze complexity based on keywords"""
        scores = []
        
        for level, keywords in self.COMPLEXITY_INDICATORS.items():
            for keyword in keywords:
                if keyword in query:
                    # Assign score based on level
                    level_scores = {
                        'trivial': 0.1,
                        'simple': 0.3,
                        'moderate': 0.5,
                        'complex': 0.8,
                        'deep': 1.0
                    }
                    scores.append(level_scores[level])
        
        return max(scores) if scores else 0.3  # Default to simple
    
    def _analyze_structure(self, query: str) -> float:
        """Analyze complexity based on query structure"""
        score = 0.0
        
        # Multiple questions
        question_count = query.count('?')
        if question_count > 1:
            score += 0.1 * min(question_count, 3)
        
        # Code blocks or technical content
        if '```' in query or '`' in query:
            score += 0.2
        
        # Multiple steps indicated
        step_indicators = ['step', 'first', 'then', 'next', 'finally', 'after']
        step_count = sum(1 for indicator in step_indicators if indicator in query.lower())
        score += 0.1 * min(step_count, 3)
        
        # Lists or enumerations
        if re.search(r'\n\d+\.', query) or re.search(r'\n-', query):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_to_level(self, score: float) -> ComplexityLevel:
        """Convert score to complexity level"""
        if score < 0.15:
            return ComplexityLevel.TRIVIAL
        elif score < 0.35:
            return ComplexityLevel.SIMPLE
        elif score < 0.6:
            return ComplexityLevel.MODERATE
        elif score < 0.85:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.DEEP


class ComplexityRouter:
    """
    Intelligent router that directs queries to appropriate processing tier.
    
    Routing Strategy:
    - TRIVIAL (0-0.15): FAST tier - instant response
    - SIMPLE (0.15-0.35): BALANCED tier - quick, cheap
    - MODERATE (0.35-0.6): CAPABLE tier - good reasoning
    - COMPLEX (0.6-0.85): DEEP tier - full analysis
    - DEEP (0.85+): DEEP tier with all tools
    
    Example:
        router = ComplexityRouter()
        decision = router.route("What time is it?")
        # -> FAST tier, <1s response
        
        decision = router.route("Analyze the causal structure of this dataset")
        # -> DEEP tier, full Knowledge Engine
    """
    
    # Routing thresholds
    THRESHOLDS = {
        ComplexityLevel.TRIVIAL: ModelTier.FAST,
        ComplexityLevel.SIMPLE: ModelTier.BALANCED,
        ComplexityLevel.MODERATE: ModelTier.CAPABLE,
        ComplexityLevel.COMPLEX: ModelTier.DEEP,
        ComplexityLevel.DEEP: ModelTier.DEEP
    }
    
    def __init__(
        self,
        fast_model_config: Optional[Dict[str, Any]] = None,
        balanced_model_config: Optional[Dict[str, Any]] = None,
        capable_model_config: Optional[Dict[str, Any]] = None,
        use_caching: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize Complexity Router.
        
        Args:
            fast_model_config: Config for fast model (Haiku, Flash)
            balanced_model_config: Config for balanced model (GPT-4o-mini)
            capable_model_config: Config for capable model (GPT-4o)
            use_caching: Cache routing decisions
            cache_size: Max cache entries
        """
        self.analyzer = ComplexityAnalyzer()
        
        # Model configurations
        self.fast_model_config = fast_model_config or {'model': 'claude-haiku'}
        self.balanced_model_config = balanced_model_config or {'model': 'gpt-4o-mini'}
        self.capable_model_config = capable_model_config or {'model': 'gpt-4o'}
        
        # Caching
        self.use_caching = use_caching
        self.cache: Dict[str, RouteDecision] = {}
        self.cache_size = cache_size
        
        # Metrics
        self.routing_history: List[Dict[str, Any]] = []
        self.tier_usage: Dict[str, int] = {tier.value: 0 for tier in ModelTier}
        
        logger.info({
            'msg': 'ComplexityRouter initialized',
            'caching': use_caching,
            'tiers': [tier.value for tier in ModelTier]
        })
    
    def route(
        self,
        query: str,
        domain: str = 'general',
        force_tier: Optional[ModelTier] = None,
        user_preference: Optional[str] = None
    ) -> RouteDecision:
        """
        Route a query to appropriate processing tier.
        
        Args:
            query: User query
            domain: Domain context
            force_tier: Override routing (for testing/debugging)
            user_preference: User's speed/quality preference
            
        Returns:
            RouteDecision with routing information
        """
        # Check cache
        cache_key = hashlib.md5(f"{query}:{domain}".encode()).hexdigest()
        if self.use_caching and cache_key in self.cache:
            decision = self.cache[cache_key]
            decision.reasoning += " [cached]"
            return decision
        
        # Analyze complexity
        complexity_score, complexity_level = self.analyzer.analyze(query, domain)
        
        # Determine tier
        if force_tier:
            selected_tier = force_tier
            reasoning = f"Tier forced to {force_tier.value}"
        else:
            selected_tier = self._select_tier(complexity_level, user_preference)
            reasoning = self._generate_reasoning(complexity_level, complexity_score, query)
        
        # Calculate estimates
        estimated_latency = selected_tier.typical_latency
        estimated_cost = selected_tier.typical_cost * (1 + complexity_score)
        
        # Create decision
        decision = RouteDecision(
            query=query,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            selected_tier=selected_tier,
            reasoning=reasoning,
            confidence=self._calculate_confidence(complexity_score),
            estimated_latency=estimated_latency,
            estimated_cost=estimated_cost,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Cache decision
        if self.use_caching:
            self._cache_decision(cache_key, decision)
        
        # Log routing
        self._log_routing(decision)
        
        return decision
    
    def _select_tier(
        self,
        level: ComplexityLevel,
        user_preference: Optional[str]
    ) -> ModelTier:
        """Select appropriate tier based on complexity and preferences"""
        base_tier = self.THRESHOLDS[level]
        
        # Adjust for user preference
        if user_preference == 'speed':
            # Downgrade one tier for speed
            tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.CAPABLE, ModelTier.DEEP]
            idx = tier_order.index(base_tier)
            return tier_order[max(0, idx - 1)]
        
        elif user_preference == 'quality':
            # Upgrade one tier for quality
            tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.CAPABLE, ModelTier.DEEP]
            idx = tier_order.index(base_tier)
            return tier_order[min(len(tier_order) - 1, idx + 1)]
        
        return base_tier
    
    def _generate_reasoning(
        self,
        level: ComplexityLevel,
        score: float,
        query: str
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasons = []
        
        if level == ComplexityLevel.TRIVIAL:
            reasons.append("Simple greeting or confirmation")
        elif level == ComplexityLevel.SIMPLE:
            reasons.append("Basic factual question")
        elif level == ComplexityLevel.MODERATE:
            reasons.append("Requires comparison or explanation")
        elif level == ComplexityLevel.COMPLEX:
            reasons.append("Requires analysis or problem-solving")
        else:
            reasons.append("Requires deep research or synthesis")
        
        # Add length-based reasoning
        if len(query) < 20:
            reasons.append("Very short query")
        elif len(query) > 500:
            reasons.append("Detailed, multi-part query")
        
        return "; ".join(reasons)
    
    def _calculate_confidence(self, complexity_score: float) -> float:
        """Calculate confidence in routing decision"""
        # Higher confidence at extremes (clearly simple or clearly complex)
        # Lower confidence in middle range (ambiguous)
        if complexity_score < 0.2 or complexity_score > 0.8:
            return 0.9
        elif complexity_score < 0.4 or complexity_score > 0.6:
            return 0.75
        else:
            return 0.6
    
    def _cache_decision(self, key: str, decision: RouteDecision):
        """Cache routing decision"""
        self.cache[key] = decision
        
        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def _log_routing(self, decision: RouteDecision):
        """Log routing decision"""
        self.routing_history.append(decision.to_dict())
        self.tier_usage[decision.selected_tier.value] += 1
        
        # Keep last 1000
        self.routing_history = self.routing_history[-1000:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total = len(self.routing_history)
        if total == 0:
            return {'total_routed': 0}
        
        tier_percentages = {
            tier: (count / total) * 100
            for tier, count in self.tier_usage.items()
        }
        
        return {
            'total_routed': total,
            'tier_usage': self.tier_usage,
            'tier_percentages': tier_percentages,
            'average_complexity': sum(
                d['complexity_score'] for d in self.routing_history
            ) / total,
            'cache_hit_rate': len(self.cache) / max(total, 1)
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get routing history"""
        return self.routing_history.copy()


# Convenience function for quick routing
def route_query(
    query: str,
    domain: str = 'general',
    **kwargs
) -> RouteDecision:
    """
    Quick route a query without creating a router instance.
    
    Example:
        decision = route_query("What is 2+2?")
        print(decision.selected_tier)  # -> ModelTier.FAST
    """
    router = ComplexityRouter()
    return router.route(query, domain, **kwargs)
