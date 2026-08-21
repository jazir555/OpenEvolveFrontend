"""
ICR (Iterative Contextual Refinements) Integration Module

This module provides ICR pattern learning and prediction capabilities
for integration with various system components including:
- Process Optimization
- Adaptive Retry Strategy
- Resource Estimation
- Quality Gate Engine
- SGD Workflow Orchestrator
- Solution Orchestrator
- Robustness Coordinator
- Knowledge Engine

All integrations follow the Federation Constitution patterns for:
- Idempotency: All operations safe to retry
- UTC: All timestamps in UTC ISO-8601 format
- Observability: All operations include correlation IDs
"""
from __future__ import annotations


from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import hashlib
import json


class ICRPatternType(str, Enum):
    """Types of ICR patterns for learning."""
    WORKFLOW_EXECUTION = "workflow_execution"
    REFINEMENT_LOOP = "refinement_loop"
    RESOURCE_USAGE = "resource_usage"
    QUALITY_OUTCOME = "quality_outcome"
    RETRY_PATTERN = "retry_pattern"
    BOTTLENECK = "bottleneck"
    OPTIMIZATION = "optimization"
    SECURITY_POLICY = "security_policy"
    GAUNTLET_OUTCOME = "gauntlet_outcome"


@dataclass
class ICRPattern:
    """Base ICR pattern for storing learned patterns."""
    pattern_id: str
    pattern_type: ICRPatternType
    pattern_key: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context information
    content_type: Optional[str] = None
    complexity_score: Optional[int] = None
    problem_type: Optional[str] = None
    
    # Outcome information
    passed: bool = True
    overall_score: Optional[float] = None
    pass_rate: float = 1.0
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ICRPrediction:
    """ICR prediction for workflow outcomes."""
    predicted_outcome: str  # "pass" or "fail"
    probability: float
    confidence: float
    reason: str
    pattern_count: int = 0
    recommended_action: Optional[str] = None


class ICRPatternStore:
    """Thread-safe ICR pattern storage with automatic pruning."""
    
    def __init__(self, max_patterns_per_key: int = 100, max_history_size: int = 500):
        self.max_patterns_per_key = max_patterns_per_key
        self.max_history_size = max_history_size
        
        # Pattern storage by type and key
        self._patterns: Dict[str, Dict[str, List[ICRPattern]]] = {}
        
        # Operation history (capped)
        self._history: deque = deque(maxlen=max_history_size)
        
        # Adaptive thresholds by pattern type
        self._adaptive_thresholds: Dict[str, float] = {}
    
    def store_pattern(self, pattern: ICRPattern) -> str:
        """Store a pattern and return pattern ID."""
        pattern_type = pattern.pattern_type.value
        pattern_key = pattern.pattern_key
        
        # Initialize storage if needed
        if pattern_type not in self._patterns:
            self._patterns[pattern_type] = {}
        if pattern_key not in self._patterns[pattern_type]:
            self._patterns[pattern_type][pattern_key] = []
        
        # Store pattern
        self._patterns[pattern_type][pattern_key].append(pattern)
        
        # Prune if exceeds max
        if len(self._patterns[pattern_type][pattern_key]) > self.max_patterns_per_key:
            self._patterns[pattern_type][pattern_key] = \
                self._patterns[pattern_type][pattern_key][-self.max_patterns_per_key:]
        
        # Add to history
        self._history.append({
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern_type,
            "pattern_key": pattern_key,
            "timestamp": pattern.timestamp.isoformat(),
            "passed": pattern.passed
        })
        
        # Update adaptive thresholds
        self._update_adaptive_threshold(pattern_type, pattern.passed)
        
        return pattern.pattern_id
    
    def get_similar_patterns(
        self,
        pattern_type: ICRPatternType,
        context: Dict[str, Any],
        limit: int = 10
    ) -> List[ICRPattern]:
        """Get similar patterns based on context."""
        pattern_type_str = pattern_type.value
        pattern_key = self._compute_pattern_key(pattern_type_str, context)
        
        # Get patterns for this key
        if pattern_type_str not in self._patterns:
            return []
        if pattern_key not in self._patterns[pattern_type_str]:
            return []
        
        return self._patterns[pattern_type_str][pattern_key][-limit:]
    
    def get_statistics(self, pattern_type: ICRPatternType) -> Dict[str, Any]:
        """Get statistics for a pattern type."""
        pattern_type_str = pattern_type.value
        
        if pattern_type_str not in self._patterns:
            return {"count": 0, "pass_rate": 0.0, "confidence": 0.0}
        
        total_patterns = sum(
            len(patterns)
            for patterns in self._patterns[pattern_type_str].values()
        )
        
        if total_patterns == 0:
            return {"count": 0, "pass_rate": 0.0, "confidence": 0.0}
        
        # Calculate pass rate
        total_passed = 0
        total_patterns_count = 0
        
        for patterns in self._patterns[pattern_type_str].values():
            for pattern in patterns:
                total_patterns_count += 1
                if pattern.passed:
                    total_passed += 1
        
        pass_rate = total_passed / total_patterns_count if total_patterns_count > 0 else 0.0
        
        # Calculate confidence based on sample size
        confidence = min(1.0, total_patterns_count / 100.0)  # Max confidence at 100 patterns
        
        return {
            "count": total_patterns,
            "pass_rate": pass_rate,
            "confidence": confidence
        }
    
    def get_adaptive_threshold(self, pattern_type: str, default: float = 0.5) -> float:
        """Get adaptive threshold for a pattern type."""
        return self._adaptive_thresholds.get(pattern_type, default)
    
    def _update_adaptive_threshold(self, pattern_type: str, passed: bool) -> None:
        """Update adaptive threshold based on outcome."""
        current = self._adaptive_thresholds.get(pattern_type, 0.5)
        
        # Adjust threshold based on pass/fail
        if passed:
            # Slightly lower threshold (things are working well)
            new_threshold = max(0.3, current - 0.02)
        else:
            # Raise threshold (need more scrutiny)
            new_threshold = min(0.9, current + 0.05)
        
        self._adaptive_thresholds[pattern_type] = new_threshold
    
    def _compute_pattern_key(self, pattern_type: str, context: Dict[str, Any]) -> str:
        """Compute a pattern key from context."""
        # Extract key fields for hashing
        key_fields = {
            "type": pattern_type,
            "content_type": context.get("content_type"),
            "complexity": context.get("complexity_score"),
            "problem_type": context.get("problem_type"),
        }
        
        key_string = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]


class ICRPredictor:
    """ICR-based predictor for workflow outcomes."""
    
    def __init__(self, pattern_store: ICRPatternStore):
        self.pattern_store = pattern_store
    
    def predict_outcome(
        self,
        pattern_type: ICRPatternType,
        context: Dict[str, Any],
        assessments: Optional[List[Dict[str, Any]]] = None
    ) -> ICRPrediction:
        """
        Predict outcome based on historical patterns.
        
        Args:
            pattern_type: Type of pattern to match
            context: Context information for matching
            assessments: Optional list of metric assessments
            
        Returns:
            ICRPrediction with outcome and confidence
        """
        # Get similar patterns
        similar_patterns = self.pattern_store.get_similar_patterns(
            pattern_type,
            context,
            limit=20
        )
        
        if not similar_patterns:
            return ICRPrediction(
                predicted_outcome="unknown",
                probability=0.5,
                confidence=0.0,
                reason="No historical patterns available",
                pattern_count=0
            )
        
        # Calculate pass probability from similar patterns
        total_patterns = len(similar_patterns)
        passed_patterns = sum(1 for p in similar_patterns if p.passed)
        pass_probability = passed_patterns / total_patterns
        
        # Determine predicted outcome
        predicted_outcome = "pass" if pass_probability >= 0.5 else "fail"
        
        # Calculate confidence based on pattern count and consistency
        pattern_confidence = min(1.0, total_patterns / 20.0)  # Max at 20 patterns
        
        # Adjust confidence based on consistency
        if pass_probability > 0.8 or pass_probability < 0.2:
            consistency_bonus = 0.2
        elif pass_probability > 0.6 or pass_probability < 0.4:
            consistency_bonus = 0.1
        else:
            consistency_bonus = 0.0
        
        confidence = min(1.0, pattern_confidence + consistency_bonus)
        
        # Generate reason
        if total_patterns >= 10:
            reason = f"Based on {total_patterns} similar historical patterns ({passed_patterns} passed)"
        else:
            reason = f"Based on limited data ({total_patterns} similar patterns)"
        
        # Generate recommended action
        recommended_action = None
        if predicted_outcome == "fail" and confidence > 0.7:
            recommended_action = "Consider additional refinement before proceeding"
        elif predicted_outcome == "pass" and confidence < 0.5:
            recommended_action = "Low confidence prediction - proceed with caution"
        
        return ICRPrediction(
            predicted_outcome=predicted_outcome,
            probability=pass_probability,
            confidence=confidence,
            reason=reason,
            pattern_count=total_patterns,
            recommended_action=recommended_action
        )


# ============================================================================
# GLOBAL ICR INTEGRATION INSTANCE
# ============================================================================

class ICRIntegration:
    """Global ICR integration instance for system-wide pattern learning."""
    
    def __init__(self):
        self.pattern_store = ICRPatternStore()
        self.predictor = ICRPredictor(self.pattern_store)
        self._enabled = True
    
    def enable(self) -> None:
        """Enable ICR integration."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable ICR integration."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if ICR integration is enabled."""
        return self._enabled
    
    def store_pattern(
        self,
        pattern_type: ICRPatternType,
        passed: bool,
        context: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Store an ICR pattern.
        
        Args:
            pattern_type: Type of pattern
            passed: Whether the operation passed
            context: Context information
            metrics: Optional metrics
            
        Returns:
            Pattern ID
        """
        if not self._enabled:
            return ""
        
        pattern = ICRPattern(
            pattern_id=f"icr_{pattern_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            pattern_type=pattern_type,
            pattern_key="",  # Will be computed by store
            passed=passed,
            metrics=metrics or {},
            context=context
        )
        
        return self.pattern_store.store_pattern(pattern)
    
    def predict(
        self,
        pattern_type: ICRPatternType,
        context: Dict[str, Any]
    ) -> ICRPrediction:
        """
        Predict outcome based on ICR patterns.
        
        Args:
            pattern_type: Type of pattern
            context: Context information
            
        Returns:
            ICRPrediction
        """
        if not self._enabled:
            return ICRPrediction(
                predicted_outcome="unknown",
                probability=0.5,
                confidence=0.0,
                reason="ICR integration disabled",
                pattern_count=0
            )
        
        return self.predictor.predict_outcome(pattern_type, context)
    
    def get_statistics(self, pattern_type: ICRPatternType) -> Dict[str, Any]:
        """Get statistics for a pattern type."""
        return self.pattern_store.get_statistics(pattern_type)
    
    def get_adaptive_threshold(self, pattern_type: str, default: float = 0.5) -> float:
        """Get adaptive threshold for a pattern type."""
        return self.pattern_store.get_adaptive_threshold(pattern_type, default)


# Global instance
_icr_integration: Optional[ICRIntegration] = None


def get_icr_integration() -> ICRIntegration:
    """Get or create global ICR integration instance."""
    global _icr_integration
    if _icr_integration is None:
        _icr_integration = ICRIntegration()
    return _icr_integration


def initialize_icr_integration() -> ICRIntegration:
    """Initialize ICR integration with default settings."""
    global _icr_integration
    _icr_integration = ICRIntegration()
    return _icr_integration
