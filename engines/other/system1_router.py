"""
System 1 Router - Latency Optimization

An intelligent semantic router that analyzes request complexity
and routes to appropriate processing paths:
- Low Complexity: Fast, cheap models (Haiku/Flash) -> Direct Output
- High Complexity: Full OpenEvolve pipeline -> Graphiti -> Z3

This creates the illusion of a living consciousness - snappy for
easy things, thoughtful for hard things.

Key Features:
- BERT-based complexity classification
- RouteLLM-style intelligent routing
- Latency-aware model selection
- Cost optimization
- Feedback loop for routing accuracy
"""
from __future__ import annotations


import os
import json
import time
import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for routing decisions"""
    TRIVIAL = "trivial"       # Instant response, no LLM needed
    SIMPLE = "simple"         # Fast model (Haiku/Flash)
    MODERATE = "moderate"     # Standard model
    COMPLEX = "complex"       # Powerful model
    DEEP = "deep"            # Full OpenEvolve pipeline


class ModelTier(Enum):
    """Model tiers for routing"""
    FAST = "fast"            # Haiku, Flash, GPT-3.5-turbo
    BALANCED = "balanced"    # Sonnet, GPT-4o-mini
    POWERFUL = "powerful"    # Opus, GPT-4o
    FULL_SYSTEM = "full_system"  # Complete OpenEvolve pipeline


@dataclass
class RouteDecision:
    """Decision made by the router"""
    request_id: str
    complexity: ComplexityLevel
    model_tier: ModelTier
    selected_model: str
    estimated_latency_ms: float
    estimated_cost: float
    confidence: float
    reasoning: str
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RouteResult:
    """Result of a routed request"""
    request_id: str
    decision: RouteDecision
    response: Any
    actual_latency_ms: float
    actual_cost: float
    success: bool
    user_satisfaction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RouterConfig:
    """Configuration for the router"""
    # Complexity thresholds
    trivial_word_count: int = 5
    simple_word_count: int = 30
    moderate_word_count: int = 100
    
    # Latency targets (ms)
    target_trivial_latency: float = 50
    target_simple_latency: float = 500
    target_moderate_latency: float = 2000
    target_complex_latency: float = 10000
    
    # Cost targets (USD per 1K tokens)
    cost_fast: float = 0.00025
    cost_balanced: float = 0.003
    cost_powerful: float = 0.015
    cost_full_system: float = 0.5
    
    # Routing model
    classifier_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_embeddings: bool = True
    
    # Feedback
    enable_feedback_loop: bool = True
    min_samples_for_retrain: int = 100


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    tier: ModelTier
    avg_latency_ms: float
    cost_per_1k_tokens: float
    context_window: int
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


class ComplexityClassifier:
    """Classifies request complexity using multiple signals"""
    
    def __init__(self, config: RouterConfig = None):
        self.config = config or RouterConfig()
        self._embedding_model = None
        self._pattern_weights = {
            "code": 2.0,
            "math": 2.5,
            "proof": 3.0,
            "theorem": 3.0,
            "z3": 3.0,
            "smt": 3.0,
            "lean": 3.0,
            "decompose": 2.0,
            "optimize": 2.0,
            "evolve": 2.5,
            "generate": 1.5,
            "analyze": 1.8,
            "verify": 2.2,
            "fix": 1.6,
            "refactor": 1.7
        }
        
        # Initialize embedding model if available
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize sentence embeddings model"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.config.classifier_model)
            logger.info(f"Loaded embedding model: {self.config.classifier_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using heuristic classification only.")
            self._embedding_model = None
    
    def classify(self, request: str, context: Dict[str, Any] = None) -> Tuple[ComplexityLevel, Dict[str, float]]:
        """
        Classify request complexity
        
        Returns:
            Tuple of (complexity_level, feature_scores)
        """
        features = self._extract_features(request, context)
        
        # Calculate composite score
        score = self._calculate_complexity_score(features)
        
        # Map score to complexity level
        if score < 0.2:
            level = ComplexityLevel.TRIVIAL
        elif score < 0.4:
            level = ComplexityLevel.SIMPLE
        elif score < 0.6:
            level = ComplexityLevel.MODERATE
        elif score < 0.8:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.DEEP
        
        return level, features
    
    def _extract_features(
        self,
        request: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Extract features from request for classification"""
        context = context or {}
        features = {}
        request_lower = request.lower()
        
        # 1. Length features
        word_count = len(request.split())
        char_count = len(request)
        features["word_count"] = min(word_count / 500, 1.0)
        features["char_count"] = min(char_count / 3000, 1.0)
        
        # 2. Complexity indicators
        complexity_score = 0
        for pattern, weight in self._pattern_weights.items():
            if pattern in request_lower:
                complexity_score += weight
        features["complexity_indicators"] = min(complexity_score / 10, 1.0)
        
        # 3. Question complexity
        features["is_question"] = 1.0 if "?" in request else 0.0
        features["question_count"] = min(request.count("?") / 3, 1.0)
        
        # 4. Code detection
        code_indicators = ["def ", "class ", "import ", "function", "{}", "();", "=>"]
        features["has_code"] = 1.0 if any(ind in request for ind in code_indicators) else 0.0
        
        # 5. Multi-part requests
        bullet_points = request.count("\n-") + request.count("\n*") + request.count("\n1.")
        features["multi_part"] = min(bullet_points / 5, 1.0)
        
        # 6. Context features
        if context:
            features["has_history"] = 1.0 if context.get("conversation_history") else 0.0
            features["has_files"] = 1.0 if context.get("attached_files") else 0.0
            
            # Previous routing feedback
            if "previous_routing" in context:
                prev = context["previous_routing"]
                if prev.get("was_underserved"):
                    features["previous_underserved"] = 1.0
                if prev.get("was_overserved"):
                    features["previous_overserved"] = 1.0
        
        # 7. Domain-specific complexity
        domains = {
            "mathematics": ["theorem", "proof", "lemma", "equation", "integral", "derivative"],
            "programming": ["debug", "refactor", "optimize", "algorithm", "complexity"],
            "science": ["experiment", "hypothesis", "analysis", "simulation"],
            "creative": ["write", "create", "design", "story", "poem"]
        }
        
        for domain, keywords in domains.items():
            if any(kw in request_lower for kw in keywords):
                features[f"domain_{domain}"] = 1.0
        
        return features
    
    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate composite complexity score from features"""
        # Weighted sum of features
        weights = {
            "word_count": 0.15,
            "complexity_indicators": 0.25,
            "has_code": 0.20,
            "multi_part": 0.15,
            "has_history": 0.10,
            "domain_mathematics": 0.25,
            "domain_programming": 0.15,
            "previous_underserved": 0.30
        }
        
        score = sum(
            features.get(feature, 0) * weight
            for feature, weight in weights.items()
        )
        
        # Normalize to 0-1
        return min(score, 1.0)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text"""
        if self._embedding_model:
            return self._embedding_model.encode(text)
        return None


class ModelRegistry:
    """Registry of available models with their characteristics"""
    
    MODELS = {
        # Fast tier - for simple queries
        "claude-haiku": ModelInfo(
            name="claude-3-haiku-20240307",
            tier=ModelTier.FAST,
            avg_latency_ms=300,
            cost_per_1k_tokens=0.00025,
            context_window=200000,
            strengths=["speed", "cost", "simple_tasks"],
            weaknesses=["complex_reasoning", "creative_tasks"]
        ),
        "gemini-flash": ModelInfo(
            name="gemini-1.5-flash",
            tier=ModelTier.FAST,
            avg_latency_ms=250,
            cost_per_1k_tokens=0.00035,
            context_window=1000000,
            strengths=["speed", "multimodal", "long_context"],
            weaknesses=["complex_math", "precise_reasoning"]
        ),
        "gpt-3.5-turbo": ModelInfo(
            name="gpt-3.5-turbo",
            tier=ModelTier.FAST,
            avg_latency_ms=400,
            cost_per_1k_tokens=0.0005,
            context_window=16385,
            strengths=["general_knowledge", "conversational"],
            weaknesses=["deep_reasoning", "code_generation"]
        ),
        
        # Balanced tier - for moderate queries
        "claude-sonnet": ModelInfo(
            name="claude-3-sonnet-20240229",
            tier=ModelTier.BALANCED,
            avg_latency_ms=800,
            cost_per_1k_tokens=0.003,
            context_window=200000,
            strengths=["reasoning", "code", "analysis"],
            weaknesses=["creative_writing"]
        ),
        "gpt-4o-mini": ModelInfo(
            name="gpt-4o-mini",
            tier=ModelTier.BALANCED,
            avg_latency_ms=600,
            cost_per_1k_tokens=0.0006,
            context_window=128000,
            strengths=["multimodal", "reasoning", "cost_efficient"],
            weaknesses=["complex_math"]
        ),
        
        # Powerful tier - for complex queries
        "claude-opus": ModelInfo(
            name="claude-3-opus-20240229",
            tier=ModelTier.POWERFUL,
            avg_latency_ms=2000,
            cost_per_1k_tokens=0.015,
            context_window=200000,
            strengths=["reasoning", "code", "analysis", "math", "creativity"],
            weaknesses=["cost", "latency"]
        ),
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            tier=ModelTier.POWERFUL,
            avg_latency_ms=1500,
            cost_per_1k_tokens=0.005,
            context_window=128000,
            strengths=["multimodal", "reasoning", "code", "general"],
            weaknesses=["cost"]
        ),
        
        # Full system - for deep queries
        "openevolve-pipeline": ModelInfo(
            name="openevolve-full-pipeline",
            tier=ModelTier.FULL_SYSTEM,
            avg_latency_ms=30000,
            cost_per_1k_tokens=0.5,
            context_window=1000000,
            strengths=["optimization", "evolution", "verification", "decomposition"],
            weaknesses=["latency", "cost", "not_for_simple_queries"]
        )
    }
    
    @classmethod
    def get_model(cls, name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        return cls.MODELS.get(name)
    
    @classmethod
    def get_by_tier(cls, tier: ModelTier) -> List[ModelInfo]:
        """Get all models in a tier"""
        return [m for m in cls.MODELS.values() if m.tier == tier]
    
    @classmethod
    def select_for_tier(cls, tier: ModelTier, preference: str = None) -> ModelInfo:
        """Select best model for tier"""
        models = cls.get_by_tier(tier)
        if preference:
            for model in models:
                if preference in model.name:
                    return model
        return models[0] if models else None


class RoutingHistory:
    """Tracks routing decisions and outcomes for feedback"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.decisions: deque = deque(maxlen=max_history)
        self.results: deque = deque(maxlen=max_history)
        self._accuracy_metrics: Dict[str, List[float]] = {
            "correct_tier": [],
            "latency_prediction": [],
            "user_satisfaction": []
        }
    
    def record_decision(self, decision: RouteDecision):
        """Record a routing decision"""
        self.decisions.append(decision)
    
    def record_result(self, result: RouteResult):
        """Record the outcome of a routing decision"""
        self.results.append(result)
        
        # Calculate accuracy metrics
        if result.success:
            # Was the tier appropriate?
            latency_error = abs(
                result.decision.estimated_latency_ms - result.actual_latency_ms
            ) / result.actual_latency_ms
            self._accuracy_metrics["latency_prediction"].append(1 - latency_error)
            
            if result.user_satisfaction:
                self._accuracy_metrics["user_satisfaction"].append(
                    result.user_satisfaction
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.results:
            return {}
        
        recent_results = list(self.results)[-100:]
        
        return {
            "total_routed": len(self.results),
            "avg_latency_ms": np.mean([r.actual_latency_ms for r in recent_results]),
            "avg_cost": np.mean([r.actual_cost for r in recent_results]),
            "success_rate": np.mean([r.success for r in recent_results]),
            "complexity_distribution": self._get_complexity_distribution(),
            "avg_latency_prediction_accuracy": np.mean(
                self._accuracy_metrics["latency_prediction"][-100:]
            ) if self._accuracy_metrics["latency_prediction"] else 0
        }
    
    def _get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of complexity levels"""
        distribution = {level.value: 0 for level in ComplexityLevel}
        for decision in self.decisions:
            distribution[decision.complexity.value] += 1
        return distribution


class System1Router:
    """
    Main System 1 Router - Latency Optimization
    
    Routes requests to appropriate processing paths based on complexity.
    Creates the illusion of a living consciousness:
    - Snappy for easy things
    - Thoughtful for hard things
    """
    
    def __init__(self, config: RouterConfig = None):
        self.config = config or RouterConfig()
        self.classifier = ComplexityClassifier(self.config)
        self.history = RoutingHistory()
        self._route_handlers: Dict[ModelTier, Callable] = {}
    
    def register_handler(self, tier: ModelTier, handler: Callable):
        """Register a handler for a model tier"""
        self._route_handlers[tier] = handler
    
    async def route(
        self,
        request: str,
        context: Dict[str, Any] = None,
        preference: str = None
    ) -> RouteDecision:
        """
        Route a request to the appropriate model/tier
        
        Args:
            request: The user request
            context: Additional context (conversation history, etc.)
            preference: Optional model preference
            
        Returns:
            RouteDecision with routing information
        """
        request_id = hashlib.md5(f"{request}{time.time()}".encode()).hexdigest()[:12]
        
        # Classify complexity
        complexity, features = self.classifier.classify(request, context)
        
        # Map complexity to model tier
        tier = self._complexity_to_tier(complexity)
        
        # Select specific model
        model_info = ModelRegistry.select_for_tier(tier, preference)
        
        # Calculate estimates
        estimated_latency = self._estimate_latency(model_info, request)
        estimated_cost = self._estimate_cost(model_info, request)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(complexity, tier, model_info, features)
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, complexity)
        
        decision = RouteDecision(
            request_id=request_id,
            complexity=complexity,
            model_tier=tier,
            selected_model=model_info.name,
            estimated_latency_ms=estimated_latency,
            estimated_cost=estimated_cost,
            confidence=confidence,
            reasoning=reasoning,
            features=features
        )
        
        # Record decision
        self.history.record_decision(decision)
        
        logger.info(f"Routed request {request_id}: {complexity.value} -> {model_info.name}")
        
        return decision
    
    def _complexity_to_tier(self, complexity: ComplexityLevel) -> ModelTier:
        """Map complexity level to model tier"""
        mapping = {
            ComplexityLevel.TRIVIAL: ModelTier.FAST,
            ComplexityLevel.SIMPLE: ModelTier.FAST,
            ComplexityLevel.MODERATE: ModelTier.BALANCED,
            ComplexityLevel.COMPLEX: ModelTier.POWERFUL,
            ComplexityLevel.DEEP: ModelTier.FULL_SYSTEM
        }
        return mapping.get(complexity, ModelTier.BALANCED)
    
    def _estimate_latency(self, model: ModelInfo, request: str) -> float:
        """Estimate latency for a request"""
        base_latency = model.avg_latency_ms
        
        # Adjust for request length
        word_count = len(request.split())
        length_factor = 1 + (word_count / 500)  # Up to 2x for long requests
        
        return base_latency * length_factor
    
    def _estimate_cost(self, model: ModelInfo, request: str) -> float:
        """Estimate cost for a request"""
        # Estimate tokens (rough approximation)
        estimated_tokens = len(request.split()) * 1.5 + 500  # Input + output
        
        return (estimated_tokens / 1000) * model.cost_per_1k_tokens
    
    def _generate_reasoning(
        self,
        complexity: ComplexityLevel,
        tier: ModelTier,
        model: ModelInfo,
        features: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the routing decision"""
        reasons = []
        
        if complexity == ComplexityLevel.TRIVIAL:
            reasons.append("Request is very short and straightforward")
        elif complexity == ComplexityLevel.SIMPLE:
            reasons.append("Request is simple and can be handled by a fast model")
        elif complexity == ComplexityLevel.MODERATE:
            reasons.append("Request requires moderate reasoning capabilities")
        elif complexity == ComplexityLevel.COMPLEX:
            reasons.append("Request is complex and requires powerful reasoning")
        elif complexity == ComplexityLevel.DEEP:
            reasons.append("Request requires deep analysis with full system capabilities")
        
        if features.get("has_code", 0) > 0.5:
            reasons.append("Contains code-related content")
        
        if features.get("complexity_indicators", 0) > 0.5:
            reasons.append("Contains complexity indicators (math, theorem, proof, etc.)")
        
        if features.get("domain_mathematics", 0) > 0.5:
            reasons.append("Mathematical content detected")
        
        return "; ".join(reasons)
    
    def _calculate_confidence(
        self,
        features: Dict[str, float],
        complexity: ComplexityLevel
    ) -> float:
        """Calculate confidence in routing decision"""
        # Higher confidence for clear signals
        signal_strength = (
            features.get("complexity_indicators", 0) +
            features.get("has_code", 0) +
            features.get("domain_mathematics", 0)
        ) / 3
        
        # Confidence based on complexity clarity
        if complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.DEEP]:
            clarity_bonus = 0.2
        else:
            clarity_bonus = 0.0
        
        return min(0.95, 0.6 + signal_strength * 0.3 + clarity_bonus)
    
    async def execute_routed(
        self,
        request: str,
        context: Dict[str, Any] = None
    ) -> Tuple[Any, RouteResult]:
        """
        Route and execute a request
        
        Returns:
            Tuple of (response, route_result)
        """
        start_time = time.time()
        
        # Make routing decision
        decision = await self.route(request, context)
        
        # Execute with appropriate handler
        handler = self._route_handlers.get(decision.model_tier)
        
        if handler:
            try:
                response = await handler(request, decision)
                success = True
            except Exception as e:
                logger.error(f"Handler error: {e}")
                response = {"error": str(e)}
                success = False
        else:
            # No handler registered - return decision only
            response = {"routing_decision": decision}
            success = True
        
        actual_latency = (time.time() - start_time) * 1000
        actual_cost = decision.estimated_cost  # In production, track actual
        
        result = RouteResult(
            request_id=decision.request_id,
            decision=decision,
            response=response,
            actual_latency_ms=actual_latency,
            actual_cost=actual_cost,
            success=success
        )
        
        self.history.record_result(result)
        
        return response, result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return self.history.get_stats()
    
    def should_use_full_system(self, request: str) -> bool:
        """Quick check if request needs full OpenEvolve system"""
        complexity, _ = self.classifier.classify(request)
        return complexity == ComplexityLevel.DEEP


class RouterMiddleware:
    """Middleware for integrating router with existing systems"""
    
    def __init__(self, router: System1Router):
        self.router = router
    
    async def process_request(
        self,
        request: str,
        handlers: Dict[ModelTier, Callable]
    ) -> Dict[str, Any]:
        """
        Process a request through the routing system
        
        Args:
            request: User request
            handlers: Dict of tier -> handler function
            
        Returns:
            Response with routing metadata
        """
        # Register handlers
        for tier, handler in handlers.items():
            self.router.register_handler(tier, handler)
        
        # Route and execute
        response, result = await self.router.execute_routed(request)
        
        return {
            "response": response,
            "routing": {
                "complexity": result.decision.complexity.value,
                "model": result.decision.selected_model,
                "latency_ms": result.actual_latency_ms,
                "estimated_cost": result.decision.estimated_cost,
                "reasoning": result.decision.reasoning
            }
        }


# Convenience functions for quick usage
def classify_complexity(request: str) -> ComplexityLevel:
    """Quick complexity classification"""
    classifier = ComplexityClassifier()
    complexity, _ = classifier.classify(request)
    return complexity


async def route_request(request: str, handlers: Dict[ModelTier, Callable] = None) -> Dict[str, Any]:
    """Quick routing function"""
    router = System1Router()
    
    if handlers:
        for tier, handler in handlers.items():
            router.register_handler(tier, handler)
    
    response, result = await router.execute_routed(request)
    
    return {
        "response": response,
        "routing": result.decision
    }


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("SYSTEM 1 ROUTER DEMO - Latency Optimization")
        print("=" * 60)
        
        router = System1Router()
        
        # Example requests of varying complexity
        test_requests = [
            ("What time is it?", "TRIVIAL - Instant response"),
            ("Fix this typo: 'recieve' -> 'receive'", "SIMPLE - Fast model"),
            ("Explain Python list comprehensions with examples", "MODERATE - Balanced model"),
            ("Debug this Z3 solver error: 'model is not available'", "COMPLEX - Powerful model"),
            ("Optimize this algorithm using evolutionary computation with formal verification", "DEEP - Full system")
        ]
        
        print("\nRouting Examples:")
        print("-" * 60)
        
        for request, expected in test_requests:
            decision = await router.route(request)
            print(f"\nRequest: {request[:50]}...")
            print(f"  Expected: {expected}")
            print(f"  Routed to: {decision.model_tier.value} ({decision.selected_model})")
            print(f"  Complexity: {decision.complexity.value}")
            print(f"  Est. Latency: {decision.estimated_latency_ms:.0f}ms")
            print(f"  Est. Cost: ${decision.estimated_cost:.4f}")
            print(f"  Confidence: {decision.confidence:.2%}")
        
        print("\n" + "=" * 60)
        print("Router Statistics:")
        stats = router.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    asyncio.run(demo())
