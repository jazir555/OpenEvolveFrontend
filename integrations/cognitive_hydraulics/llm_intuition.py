"""LLM Intuition Engine for Cognitive Hydraulics.

Small LLM provides P (probability) and C (cost) estimates for ACT-R.
Generates operators when Soar has no-change impasse.

Key Functions:
    - estimate_probability(operator, goal, context): Estimate P(success)
    - estimate_cost(operator, context): Estimate C(time/effort)
    - generate_operators(state, n=3): Generate candidate operators
    - evaluate_resolution(impasse, resolution): Rate success
    - encode_chunk(impasse, resolution): Convert to SoarRule

Integration with LLM:
    - Uses qwen3:8b or similar models
    - Caches results for efficiency
    - Circuit breaker for failures
    - Structured JSON output
"""

import logging
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

from .config import LLMConfig
from .soar_engine import SoarOperator, SoarRule, Impasse

logger = logging.getLogger(__name__)


class SuccessRating(Enum):
    """Rating for resolution success."""
    COMPLETE = 1.0
    PARTIAL = 0.5
    FAILURE = 0.0


@dataclass
class LLMCacheEntry:
    """Cache entry for LLM responses."""
    key: str
    response: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age < self.ttl_seconds


class LLMCache:
    """Simple LRU cache for LLM responses."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, LLMCacheEntry] = {}
    
    def _make_key(self, prompt: str) -> str:
        """Create cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[Any]:
        """Get cached response."""
        key = self._make_key(prompt)
        entry = self.cache.get(key)
        
        if entry and entry.is_valid():
            return entry.response
        
        # Remove expired entry
        if key in self.cache:
            del self.cache[key]
        
        return None
    
    def set(self, prompt: str, response: Any):
        """Cache response."""
        key = self._make_key(prompt)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest]
        
        self.cache[key] = LLMCacheEntry(
            key=key,
            response=response,
            ttl_seconds=self.ttl_seconds
        )
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()


class CircuitBreaker:
    """Circuit breaker for LLM calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed > self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker OPENED")


class ProbabilityEstimator:
    """Estimate P(success) for operators using LLM."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def build_prompt(
        self,
        operator: Any,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for probability estimation."""
        prompt = f"""Given the following task, estimate the probability of success for the proposed operator.

Goal: {json.dumps(goal, indent=2)}

Operator: {getattr(operator, 'name', str(operator))}
Operator details: {json.dumps(getattr(operator, 'preconditions', []), indent=2)}

Context: {json.dumps(context, indent=2)}

Estimate the probability of success (0.0 to 1.0):
- 0.0-0.3: Low chance of success
- 0.3-0.7: Moderate chance of success  
- 0.7-1.0: High chance of success

Return only a JSON object: {{"probability": float, "reasoning": "brief explanation"}}
"""
        return prompt
    
    def parse_response(self, response: str) -> float:
        """Parse probability from LLM response."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            prob = float(data.get("probability", 0.5))
            return max(0.0, min(1.0, prob))
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract number from text
            import re
            numbers = re.findall(r'0?\.\d+|[01](?=\s|$)', response)
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
            return 0.5


class CostEstimator:
    """Estimate C(time/effort) for operators using LLM."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def build_prompt(self, operator: Any, context: Dict[str, Any]) -> str:
        """Build prompt for cost estimation."""
        prompt = f"""Estimate the cost (time/effort) of executing the following operator.

Operator: {getattr(operator, 'name', str(operator))}
Actions: {json.dumps(getattr(operator, 'actions', []), indent=2)}

Context: {json.dumps(context, indent=2)}

Cost scale:
- 1-3: Low cost (simple operation)
- 4-7: Medium cost (moderate complexity)
- 8-10: High cost (complex operation)

Return only a JSON object: {{"cost": float, "time_estimate_seconds": int, "reasoning": "brief explanation"}}
"""
        return prompt
    
    def parse_response(self, response: str) -> float:
        """Parse cost from LLM response."""
        try:
            data = json.loads(response)
            cost = float(data.get("cost", 1.0))
            return max(1.0, min(10.0, cost))
        except (json.JSONDecodeError, ValueError):
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                return max(1.0, min(10.0, float(numbers[0])))
            return 1.0


class OperatorGenerator:
    """Generate candidate operators when Soar has no-change impasse."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def build_prompt(self, state: Dict[str, Any], n: int = 3) -> str:
        """Build prompt for operator generation."""
        prompt = f"""Generate {n} candidate operators to transform the current state toward the goal.

Current state: {json.dumps(state.get('working_memory', {}), indent=2)}
Goal: {json.dumps(state.get('goal', {}), indent=2)}

Each operator should have:
- name: descriptive name
- preconditions: list of conditions that must be true
- actions: list of state transformations

Return only a JSON object: {{"operators": [{{"name": str, "preconditions": [...], "actions": [...]}}]}}
"""
        return prompt
    
    def parse_response(self, response: str) -> List[SoarOperator]:
        """Parse operators from LLM response."""
        try:
            data = json.loads(response)
            operators = []
            
            for op_data in data.get("operators", []):
                op = SoarOperator(
                    name=op_data.get("name", "generated_op"),
                    preconditions=op_data.get("preconditions", []),
                    actions=op_data.get("actions", [])
                )
                operators.append(op)
            
            return operators
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse operators: {e}")
            return []


class ChunkEncoder:
    """Encode successful resolutions as Soar rules (chunks)."""
    
    def build_prompt(
        self,
        impasse: Impasse,
        resolution: Dict[str, Any]
    ) -> str:
        """Build prompt for chunk encoding."""
        prompt = f"""Convert the following impasse resolution into a reusable production rule.

Impasse type: {impasse.impasse_type.name}
Impasse context: {json.dumps(impasse.context, indent=2)}
Resolution: {json.dumps(resolution, indent=2)}

Create a production rule with:
- conditions: patterns that trigger this rule
- actions: the successful resolution steps

Return only a JSON object: {{"rule": {{"conditions": [...], "actions": [...]}}}}
"""
        return prompt
    
    def parse_response(self, response: str, impasse: Impasse) -> Optional[SoarRule]:
        """Parse rule from LLM response."""
        try:
            data = json.loads(response)
            rule_data = data.get("rule", {})
            
            rule = SoarRule(
                name=f"learned_{impasse.impasse_type.name.lower()}_{int(time.time())}",
                conditions=rule_data.get("conditions", []),
                actions=rule_data.get("actions", []),
                learned=True,
                utility=0.5
            )
            
            return rule
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse chunk: {e}")
            return None


class LLMIntuitionEngine:
    """
    Main LLM Intuition Engine.
    
    Provides intuition via LLM for:
    - Probability estimation
    - Cost estimation
    - Operator generation
    - Resolution evaluation
    - Chunk encoding
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        
        # Sub-components
        self.probability_estimator = ProbabilityEstimator(self.config)
        self.cost_estimator = CostEstimator(self.config)
        self.operator_generator = OperatorGenerator(self.config)
        self.chunk_encoder = ChunkEncoder()
        
        # Cache and circuit breaker
        self.cache = LLMCache(
            ttl_seconds=self.config.cache_ttl_seconds,
            max_size=1000
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60
        )
        
        # Stats
        self.call_count = 0
        self.cache_hits = 0
        self.call_history: List[Dict] = []
        
        # Mock LLM for testing (can be replaced with real implementation)
        self.llm_callback: Optional[Callable[[str], str]] = None
    
    def set_llm_callback(self, callback: Callable[[str], str]):
        """Set callback function for LLM calls."""
        self.llm_callback = callback
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with caching and circuit breaker."""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker OPEN, skipping LLM call")
            return None
        
        # Check cache
        cached = self.cache.get(prompt)
        if cached:
            self.cache_hits += 1
            return cached
        
        # Make LLM call
        self.call_count += 1
        start_time = time.time()
        
        try:
            if self.llm_callback:
                response = self.llm_callback(prompt)
            else:
                # Default: return a reasonable default based on prompt type
                response = self._default_response(prompt)
            
            duration = time.time() - start_time
            
            # Record call
            self.call_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration * 1000,
                "prompt_length": len(prompt),
                "success": True
            })
            
            # Cache response
            self.cache.set(prompt, response)
            
            # Record success
            self.circuit_breaker.record_success()
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.call_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration * 1000,
                "prompt_length": len(prompt),
                "success": False,
                "error": str(e)
            })
            
            self.circuit_breaker.record_failure()
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _default_response(self, prompt: str) -> str:
        """Generate default response when no LLM is available."""
        # Return reasonable defaults based on prompt type
        if "probability" in prompt.lower():
            return '{"probability": 0.5, "reasoning": "default estimate"}'
        elif "cost" in prompt.lower():
            return '{"cost": 5.0, "time_estimate_seconds": 10, "reasoning": "default estimate"}'
        elif "operators" in prompt.lower():
            return '{"operators": [{"name": "explore", "preconditions": [], "actions": []}]}'
        else:
            return '{"result": "default"}'
    
    def estimate_probability(
        self,
        operator: Any,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Estimate P(success) for operator. Returns 0-1."""
        prompt = self.probability_estimator.build_prompt(operator, goal, context)
        response = self._call_llm(prompt)
        
        if response:
            return self.probability_estimator.parse_response(response)
        
        return 0.5  # Default
    
    def estimate_cost(
        self,
        operator: Any,
        context: Dict[str, Any]
    ) -> float:
        """Estimate C(time/effort) for operator. Returns 1-10."""
        prompt = self.cost_estimator.build_prompt(operator, context)
        response = self._call_llm(prompt)
        
        if response:
            return self.cost_estimator.parse_response(response)
        
        return 1.0  # Default
    
    def generate_operators(
        self,
        state: Dict[str, Any],
        n: int = 3
    ) -> List[SoarOperator]:
        """Generate candidate operators."""
        prompt = self.operator_generator.build_prompt(state, n)
        response = self._call_llm(prompt)
        
        if response:
            return self.operator_generator.parse_response(response)
        
        return []
    
    def evaluate_resolution(
        self,
        impasse: Impasse,
        resolution: Dict[str, Any]
    ) -> SuccessRating:
        """Rate the success of a resolution."""
        # Simple heuristic based on resolution content
        if "error" in resolution or "failure" in resolution:
            return SuccessRating.FAILURE
        
        if "complete" in resolution or "success" in resolution:
            return SuccessRating.COMPLETE
        
        return SuccessRating.PARTIAL
    
    def encode_chunk(
        self,
        impasse: Impasse,
        resolution: Dict[str, Any]
    ) -> Optional[SoarRule]:
        """Encode impasse resolution as Soar rule."""
        prompt = self.chunk_encoder.build_prompt(impasse, resolution)
        response = self._call_llm(prompt)
        
        if response:
            return self.chunk_encoder.parse_response(response, impasse)
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "call_count": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache.cache),
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }


# Convenience alias
IntuitionEngine = LLMIntuitionEngine
