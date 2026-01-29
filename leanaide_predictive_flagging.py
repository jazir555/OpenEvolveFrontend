"""
LeanAide Predictive Flagging System for MCTS-MDAP-MAKER Integration

Advanced predictive flagging system that anticipates quality issues before they occur.
Uses machine learning models to predict potential problems based on:
- Historical performance data
- Pattern recognition
- Agent behavior analysis
- Contextual factors

Features:
- Predictive quality assessment
- Machine learning-based flagging
- Historical pattern analysis
- Agent behavior prediction
- Context-aware forecasting
- Early warning systems

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
import hashlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
from pathlib import Path
import statistics


logger = logging.getLogger(__name__)


# =============================================================================
# Predictive Flagging Configuration
# =============================================================================

@dataclass
class PredictiveFlagConfig:
    """
    Configuration for predictive flagging system.
    """
    # Prediction thresholds
    prediction_confidence_threshold: float = 0.7  # Minimum confidence for prediction
    prediction_accuracy_threshold: float = 0.8    # Minimum accuracy for model use
    prediction_horizon: int = 5                   # Look ahead N steps
    
    # Historical data requirements
    min_historical_samples: int = 10              # Minimum samples for prediction
    historical_window_days: int = 30              # Days of history to consider
    
    # Feature weights
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "agent_performance": 0.3,
        "confidence_trend": 0.25,
        "pattern_frequency": 0.2,
        "context_similarity": 0.15,
        "structural_indicators": 0.1
    })
    
    # Model parameters
    enable_ml_prediction: bool = True
    ml_model_type: str = "ensemble"  # ensemble, neural_network, decision_tree
    enable_feature_engineering: bool = True
    enable_context_awareness: bool = True
    
    # Prediction types
    enable_quality_prediction: bool = True
    enable_performance_prediction: bool = True
    enable_pattern_prediction: bool = True
    enable_agent_behavior_prediction: bool = True
    
    # Feedback loop
    enable_prediction_feedback: bool = True
    feedback_learning_rate: float = 0.1  # How quickly to adjust based on feedback
    
    # Integration settings
    enable_predictive_flagging: bool = True
    enable_early_warning: bool = True
    enable_preemptive_pruning: bool = False  # Only if very confident


# =============================================================================
# Predictive Flagging Enums and Data Classes
# =============================================================================

class PredictionType(Enum):
    """Types of predictions."""
    QUALITY_LOW = "quality_low"
    PERFORMANCE_POOR = "performance_poor"
    PATTERN_BLOCKED = "pattern_blocked"
    AGENT_BEHAVIOR_ANOMALOUS = "agent_behavior_anomalous"
    STRUCTURAL_ISSUE = "structural_issue"
    CONFIDENCE_DECLINING = "confidence_declining"
    VOTE_AGREEMENT_DETERIORATING = "vote_agreement_deteriorating"
    RESOURCE_EXCEEDANCE = "resource_exceedance"


@dataclass
class Prediction:
    """A single prediction with details."""
    prediction_type: PredictionType
    predicted_item: str  # ID of item being predicted
    confidence: float    # Confidence in the prediction (0.0 to 1.0)
    probability: float   # Probability of occurrence (0.0 to 1.0)
    severity: float      # Expected severity if occurs (0.0 to 1.0)
    timestamp: float = field(default_factory=time.time)
    features: Dict[str, Any] = field(default_factory=dict)  # Features used for prediction
    model_used: str = "default"  # Model that made the prediction
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["prediction_type"] = self.prediction_type.value
        return result


@dataclass
class PredictionHistory:
    """Historical record of predictions and outcomes."""
    prediction_id: str
    prediction: Prediction
    actual_outcome: Optional[bool] = None  # Whether predicted issue actually occurred
    actual_severity: Optional[float] = None  # Actual severity if occurred
    prediction_accuracy: Optional[float] = None  # Accuracy of prediction
    feedback_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.prediction:
            result["prediction"] = self.prediction.to_dict()
        return result


@dataclass
class PredictiveAnalysis:
    """Comprehensive analysis of predictive flagging."""
    total_predictions: int = 0
    accurate_predictions: int = 0
    prediction_accuracy_rate: float = 0.0
    prediction_types: Dict[str, int] = field(default_factory=dict)
    prediction_confidence_distribution: Dict[str, int] = field(default_factory=dict)
    prediction_probability_distribution: Dict[str, int] = field(default_factory=dict)
    prediction_severity_distribution: Dict[str, int] = field(default_factory=dict)
    early_warnings_issued: int = 0
    preemptive_actions_taken: int = 0
    analysis_time: float = 0.0
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Feature Engineering System
# =============================================================================

class FeatureExtractor:
    """
    Extracts features for predictive modeling.
    """
    
    def __init__(self, config: PredictiveFlagConfig):
        self.config = config
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
    
    def extract_features(
        self,
        item: Any,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from an item for prediction.
        
        Args:
            item: The item to extract features from
            context: Additional context
            history: Historical data for trend analysis
            
        Returns:
            Dictionary of extracted features
        """
        context = context or {}
        history = history or []
        
        features = {}
        
        # Extract basic features
        features.update(self._extract_basic_features(item, context))
        
        # Extract agent performance features
        features.update(self._extract_agent_performance_features(context))
        
        # Extract confidence trend features
        features.update(self._extract_confidence_trend_features(history))
        
        # Extract pattern features
        features.update(self._extract_pattern_features(item))
        
        # Extract structural features
        features.update(self._extract_structural_features(item))
        
        # Extract contextual features
        features.update(self._extract_contextual_features(context))
        
        # Normalize features if enabled
        if self.config.enable_feature_engineering:
            features = self._normalize_features(features)
        
        return features
    
    def _extract_basic_features(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract basic features."""
        features = {}
        
        # Item characteristics
        item_str = self._item_to_string(item)
        features["item_length"] = len(item_str)
        features["item_word_count"] = len(item_str.split())
        features["item_line_count"] = len(item_str.split('\n'))
        
        # Context characteristics
        features["context_size"] = len(context)
        features["has_agent_id"] = "agent_id" in context
        features["has_confidence"] = "confidence" in context
        features["has_votes"] = "votes" in context
        features["has_state"] = "state" in context
        
        return features
    
    def _extract_agent_performance_features(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract agent performance features."""
        features = {}
        
        agent_id = context.get("agent_id", "unknown")
        
        # If we have historical performance data for this agent
        # (would be populated by the main system)
        features[f"agent_{agent_id}_performance_known"] = False
        features[f"agent_{agent_id}_avg_confidence"] = 0.5
        features[f"agent_{agent_id}_success_rate"] = 0.5
        features[f"agent_{agent_id}_error_rate"] = 0.5
        
        # Add more sophisticated features if available
        if "agent_performance_history" in context:
            perf_history = context["agent_performance_history"]
            if perf_history:
                confidences = [p.get("confidence", 0.5) for p in perf_history]
                success_rates = [p.get("success_rate", 0.5) for p in perf_history]
                
                features[f"agent_{agent_id}_avg_confidence"] = statistics.mean(confidences) if confidences else 0.5
                features[f"agent_{agent_id}_avg_success_rate"] = statistics.mean(success_rates) if success_rates else 0.5
                features[f"agent_{agent_id}_confidence_std"] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                features[f"agent_{agent_id}_recent_decline"] = self._has_recent_decline(confidences)
        
        return features
    
    def _extract_confidence_trend_features(
        self,
        history: List[Any]
    ) -> Dict[str, Any]:
        """Extract confidence trend features."""
        features = {}
        
        if not history:
            features["confidence_trend"] = 0.0
            features["confidence_volatility"] = 0.0
            features["confidence_declining"] = False
            return features
        
        # Extract confidence values from history
        confidences = []
        for item in history:
            if hasattr(item, 'confidence'):
                confidences.append(getattr(item, 'confidence'))
            elif isinstance(item, dict) and 'confidence' in item:
                confidences.append(item['confidence'])
            elif hasattr(item, 'confidence_score'):
                confidences.append(getattr(item, 'confidence_score'))
        
        if not confidences:
            features["confidence_trend"] = 0.0
            features["confidence_volatility"] = 0.0
            features["confidence_declining"] = False
            return features
        
        # Calculate trend
        if len(confidences) >= 2:
            # Simple linear trend
            n = len(confidences)
            x = list(range(n))
            y = confidences
            
            # Calculate slope (trend)
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
            
            if denominator != 0:
                trend = numerator / denominator
            else:
                trend = 0.0
            
            features["confidence_trend"] = trend
            features["confidence_volatility"] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            features["confidence_declining"] = trend < -0.01  # Declining trend
        else:
            features["confidence_trend"] = 0.0
            features["confidence_volatility"] = 0.0
            features["confidence_declining"] = False
        
        return features
    
    def _extract_pattern_features(self, item: Any) -> Dict[str, Any]:
        """Extract pattern-based features."""
        features = {}
        
        item_str = self._item_to_string(item)
        
        # Count occurrences of various patterns
        blocked_patterns = ["sorry", "admit", "classical.choice", "noncomputable"]
        suspicious_patterns = ["error", "failed", "incomplete", "undefined"]
        
        for pattern in blocked_patterns:
            features[f"pattern_{pattern}_count"] = item_str.lower().count(pattern)
        
        for pattern in suspicious_patterns:
            features[f"suspicious_pattern_{pattern}_count"] = item_str.lower().count(pattern)
        
        # Calculate ratios
        total_chars = len(item_str)
        if total_chars > 0:
            for pattern in blocked_patterns + suspicious_patterns:
                pattern_key = f"pattern_{pattern}_count" if pattern in blocked_patterns else f"suspicious_pattern_{pattern}_count"
                features[f"{pattern_key}_ratio"] = features[pattern_key] / total_chars
        
        return features
    
    def _extract_structural_features(self, item: Any) -> Dict[str, Any]:
        """Extract structural features."""
        features = {}
        
        item_str = self._item_to_string(item)
        
        # Structural indicators
        features["has_nested_structures"] = item_str.count('(') > 5 or item_str.count('{') > 5
        features["has_multiple_levels"] = item_str.count('\n') > 10
        features["has_complex_expressions"] = item_str.count(':=') > 3
        features["has_quantifiers"] = any(q in item_str for q in ["forall", "exists", "∀", "∃"])
        features["has_implications"] = any(op in item_str for op in ["→", "->", "implies"])
        features["has_negations"] = "¬" in item_str or "not" in item_str.lower()
        
        return features
    
    def _extract_contextual_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features."""
        features = {}
        
        # Context-based features
        features["context_depth"] = context.get("depth", 0)
        features["context_node_count"] = context.get("node_count", 0)
        features["context_branch_factor"] = context.get("branch_factor", 1.0)
        features["context_remaining_goals"] = len(context.get("remaining_goals", []))
        features["context_proof_state_complexity"] = context.get("proof_state_complexity", 1.0)
        
        # Time-based features
        features["context_time_elapsed"] = context.get("time_elapsed", 0.0)
        features["context_iterations"] = context.get("iterations", 0)
        
        return features
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize features to common scale."""
        normalized = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Normalize numeric values to 0-1 range
                # For simplicity, assume most values are already in reasonable ranges
                # In practice, you'd use historical data to determine normalization ranges
                if abs(value) > 1000:  # Large values, normalize
                    normalized[key] = value / 1000.0
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized
    
    def _has_recent_decline(self, values: List[float], window: int = 3) -> bool:
        """Check if there's a recent decline in values."""
        if len(values) < window + 1:
            return False
        
        recent_avg = statistics.mean(values[-window:])
        earlier_avg = statistics.mean(values[-(window*2):-window]) if len(values) >= window*2 else statistics.mean(values[:-window])
        
        return recent_avg < earlier_avg * 0.9  # 10% decline
    
    def _item_to_string(self, item: Any) -> str:
        """Convert an item to string for analysis."""
        if isinstance(item, str):
            return item
        elif hasattr(item, 'to_string'):
            return getattr(item, 'to_string')()
        elif hasattr(item, 'lean_code'):
            return getattr(item, 'lean_code')
        elif hasattr(item, 'proof'):
            return str(getattr(item, 'proof'))
        elif hasattr(item, 'action'):
            return str(getattr(item, 'action'))
        elif isinstance(item, dict):
            return json.dumps(item, default=str)
        else:
            return str(item)


# =============================================================================
# Predictive Modeling System
# =============================================================================

class PredictionModel:
    """
    Base class for prediction models.
    """
    
    def __init__(self, model_id: str, config: PredictiveFlagConfig):
        self.model_id = model_id
        self.config = config
        self.is_trained = False
        self.training_data: List[Tuple[Dict[str, Any], bool]] = []
        self.model_performance: Dict[str, float] = {}
    
    def train(self, training_data: List[Tuple[Dict[str, Any], bool]]) -> None:
        """
        Train the prediction model.

        Args:
            training_data: List of (features, outcome) tuples
        """
        self.training_data = training_data
        self.is_trained = len(training_data) >= self.config.min_historical_samples
        if self.is_trained:
            self._train_model()
    
    def _train_model(self) -> None:
        """Train the specific model implementation."""
        raise NotImplementedError
    
    def predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict outcome and confidence.
        
        Args:
            features: Features to predict on
            
        Returns:
            Tuple of (probability, confidence)
        """
        if not self.is_trained:
            # Default prediction if not trained
            return 0.5, 0.1  # 50% chance, low confidence
        
        return self._predict(features)
    
    def _predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """Predict using the specific model implementation."""
        raise NotImplementedError
    
    def update_performance(self, actual_outcome: bool, predicted_probability: float) -> None:
        """Update model performance metrics."""
        if "predictions_made" not in self.model_performance:
            self.model_performance["predictions_made"] = 0
            self.model_performance["correct_predictions"] = 0
            self.model_performance["total_error"] = 0.0
        
        self.model_performance["predictions_made"] += 1
        
        # Calculate accuracy (binary classification)
        predicted_outcome = predicted_probability > 0.5
        if predicted_outcome == actual_outcome:
            self.model_performance["correct_predictions"] += 1
        
        # Calculate error (for probability calibration)
        actual_prob = 1.0 if actual_outcome else 0.0
        self.model_performance["total_error"] += abs(predicted_probability - actual_prob)
        
        # Update accuracy
        if self.model_performance["predictions_made"] > 0:
            self.model_performance["accuracy"] = (
                self.model_performance["correct_predictions"] / 
                self.model_performance["predictions_made"]
            )
        
        # Update mean absolute error
        if self.model_performance["predictions_made"] > 0:
            self.model_performance["mae"] = (
                self.model_performance["total_error"] / 
                self.model_performance["predictions_made"]
            )


class SimpleEnsembleModel(PredictionModel):
    """
    Simple ensemble model combining multiple heuristics.
    """
    
    def __init__(self, model_id: str, config: PredictiveFlagConfig):
        super().__init__(model_id, config)
        self.feature_weights = config.feature_weights
    
    def _train_model(self) -> None:
        """Training is not needed for this simple model."""
        # The simple ensemble model doesn't require training - it uses heuristics
        pass
    
    def _predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction using weighted feature combination."""
        score = 0.0
        total_weight = 0.0
        
        # Agent performance component
        if "agent_performance" in self.feature_weights:
            agent_perf_score = 0.0
            for key, value in features.items():
                if "agent_" in key and "_success_rate" in key:
                    agent_perf_score = min(1.0, max(0.0, value))  # Clamp to [0,1]
                    break
            score += self.feature_weights["agent_performance"] * (1.0 - agent_perf_score)  # Lower success = higher risk
            total_weight += self.feature_weights["agent_performance"]
        
        # Confidence trend component
        if "confidence_trend" in features:
            trend = features["confidence_trend"]
            # Negative trend increases risk
            trend_score = max(0.0, min(1.0, -trend * 10))  # Amplify negative trends
            score += self.feature_weights["confidence_trend"] * trend_score
            total_weight += self.feature_weights["confidence_trend"]
        
        # Pattern component
        pattern_score = 0.0
        for key, value in features.items():
            if "pattern_" in key and "_count" in key and isinstance(value, (int, float)):
                pattern_score += value  # Higher pattern counts = higher risk
        if pattern_score > 0:
            pattern_score = min(1.0, pattern_score / 10)  # Normalize
            score += self.feature_weights["pattern_frequency"] * pattern_score
            total_weight += self.feature_weights["pattern_frequency"]
        
        # Structural component
        structural_score = 0.0
        if features.get("has_nested_structures", False):
            structural_score += 0.3
        if features.get("has_complex_expressions", False):
            structural_score += 0.2
        structural_score = min(1.0, structural_score)
        score += self.feature_weights["structural_indicators"] * structural_score
        total_weight += self.feature_weights["structural_indicators"]
        
        # Contextual component
        context_score = 0.0
        if features.get("context_depth", 0) > 50:
            context_score += 0.5
        if features.get("context_iterations", 0) > 1000:
            context_score += 0.3
        context_score = min(1.0, context_score)
        score += self.feature_weights["context_similarity"] * context_score
        total_weight += self.feature_weights["context_similarity"]
        
        # Normalize score
        if total_weight > 0:
            probability = score / total_weight
        else:
            probability = 0.5  # Default if no features matched
        
        # Confidence based on feature coverage
        confidence = min(1.0, total_weight / sum(self.feature_weights.values()))
        
        return min(1.0, max(0.0, probability)), confidence


class PredictiveFlaggingSystem:
    """
    Main predictive flagging system.
    
    Uses machine learning models to predict potential quality issues before they occur.
    """
    
    def __init__(self, config: Optional[PredictiveFlagConfig] = None):
        """Initialize the predictive flagging system."""
        self.config = config or PredictiveFlagConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.models: Dict[str, PredictionModel] = {}
        self.prediction_history: List[PredictionHistory] = []
        self.historical_data: List[Dict[str, Any]] = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize prediction models."""
        if self.config.enable_ml_prediction:
            if self.config.ml_model_type == "ensemble":
                self.models["quality"] = SimpleEnsembleModel("quality_ensemble", self.config)
                self.models["performance"] = SimpleEnsembleModel("performance_ensemble", self.config)
                self.models["pattern"] = SimpleEnsembleModel("pattern_ensemble", self.config)
            # Could add more model types here
    
    def predict_item_quality(
        self,
        item: Any,
        item_id: str = "",
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None
    ) -> List[Prediction]:
        """
        Predict potential quality issues for an item.
        
        Args:
            item: The item to predict quality for
            item_id: Optional ID for the item
            context: Additional context
            history: Historical data for trend analysis
            
        Returns:
            List of predictions
        """
        if not self.config.enable_predictive_flagging:
            return []
        
        context = context or {}
        history = history or []
        
        # Extract features
        features = self.feature_extractor.extract_features(item, context, history)
        
        predictions = []
        
        # Predict quality issues
        if self.config.enable_quality_prediction:
            prob, conf = self._predict_with_model("quality", features)
            if conf >= self.config.prediction_confidence_threshold and prob >= 0.5:
                predictions.append(Prediction(
                    prediction_type=PredictionType.QUALITY_LOW,
                    predicted_item=item_id or str(uuid.uuid4()),
                    confidence=conf,
                    probability=prob,
                    severity=prob,  # Use probability as severity proxy
                    features=features,
                    model_used="quality"
                ))
        
        # Predict performance issues
        if self.config.enable_performance_prediction:
            prob, conf = self._predict_with_model("performance", features)
            if conf >= self.config.prediction_confidence_threshold and prob >= 0.5:
                predictions.append(Prediction(
                    prediction_type=PredictionType.PERFORMANCE_POOR,
                    predicted_item=item_id or str(uuid.uuid4()),
                    confidence=conf,
                    probability=prob,
                    severity=prob,
                    features=features,
                    model_used="performance"
                ))
        
        # Predict pattern issues
        if self.config.enable_pattern_prediction:
            prob, conf = self._predict_with_model("pattern", features)
            if conf >= self.config.prediction_confidence_threshold and prob >= 0.5:
                predictions.append(Prediction(
                    prediction_type=PredictionType.PATTERN_BLOCKED,
                    predicted_item=item_id or str(uuid.uuid4()),
                    confidence=conf,
                    probability=prob,
                    severity=prob,
                    features=features,
                    model_used="pattern"
                ))
        
        # Store predictions in history
        for pred in predictions:
            self.prediction_history.append(PredictionHistory(
                prediction_id=f"pred_{uuid.uuid4()}",
                prediction=pred
            ))
        
        return predictions
    
    def _predict_with_model(self, model_name: str, features: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction using a specific model."""
        if model_name in self.models:
            return self.models[model_name].predict(features)
        else:
            # Default prediction if model doesn't exist
            return 0.5, 0.1
    
    def predict_agent_behavior(
        self,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """
        Predict potential anomalous behavior from an agent.
        
        Args:
            agent_id: ID of the agent to predict behavior for
            context: Additional context
            
        Returns:
            List of behavior predictions
        """
        if not self.config.enable_agent_behavior_prediction:
            return []
        
        context = context or {}
        context["agent_id"] = agent_id
        
        # Create a dummy item to extract features
        dummy_item = {"agent_id": agent_id}
        
        features = self.feature_extractor.extract_features(dummy_item, context, [])
        
        # Simple prediction based on historical agent performance
        prob, conf = 0.1, 0.8  # Default: low probability, high confidence
        
        # If we have historical data for this agent, use it
        agent_history = [h for h in self.historical_data if h.get("agent_id") == agent_id]
        if agent_history:
            # Calculate historical failure rate
            failures = sum(1 for h in agent_history if h.get("outcome") == "failure")
            total = len(agent_history)
            if total > 0:
                historical_failure_rate = failures / total
                prob = min(1.0, historical_failure_rate * 2)  # Amplify slightly
                conf = min(1.0, total / 20)  # Confidence increases with more data
        
        predictions = []
        if conf >= self.config.prediction_confidence_threshold and prob >= 0.3:
            predictions.append(Prediction(
                prediction_type=PredictionType.AGENT_BEHAVIOR_ANOMALOUS,
                predicted_item=agent_id,
                confidence=conf,
                probability=prob,
                severity=prob,
                features=features,
                model_used="agent_behavior"
            ))
        
        return predictions
    
    def predict_confidence_decline(
        self,
        item: Any,
        history: List[Any]
    ) -> List[Prediction]:
        """
        Predict if confidence will decline in the near future.
        
        Args:
            item: Current item
            history: Historical items to analyze trend
            
        Returns:
            List of confidence decline predictions
        """
        # Extract confidence trend features
        features = self.feature_extractor.extract_features(item, {}, history)
        
        predictions = []
        
        if features.get("confidence_declining", False):
            # If we detect a declining trend, predict continued decline
            predictions.append(Prediction(
                prediction_type=PredictionType.CONFIDENCE_DECLINING,
                predicted_item=str(uuid.uuid4()),
                confidence=0.8,
                probability=0.7,
                severity=0.6,
                features=features,
                model_used="confidence_trend"
            ))
        
        return predictions
    
    def get_prediction_analysis(self) -> PredictiveAnalysis:
        """Get comprehensive analysis of prediction performance."""
        start_time = time.time()
        
        analysis = PredictiveAnalysis(
            total_predictions=len(self.prediction_history),
            analysis_time=time.time() - start_time
        )
        
        if not self.prediction_history:
            return analysis
        
        # Count prediction types
        for hist in self.prediction_history:
            if hist.prediction:
                type_key = hist.prediction.prediction_type.value
                analysis.prediction_types[type_key] = analysis.prediction_types.get(type_key, 0) + 1
        
        # Analyze prediction accuracy
        accurate_predictions = 0
        for hist in self.prediction_history:
            if hist.actual_outcome is not None and hist.prediction:
                # Consider prediction accurate if direction matches
                predicted_positive = hist.prediction.probability > 0.5
                actual_positive = hist.actual_outcome
                if predicted_positive == actual_positive:
                    accurate_predictions += 1
        
        analysis.accurate_predictions = accurate_predictions
        if analysis.total_predictions > 0:
            analysis.prediction_accuracy_rate = accurate_predictions / analysis.total_predictions
        
        # Analyze confidence distribution
        conf_ranges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        prob_ranges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        severity_ranges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        
        for hist in self.prediction_history:
            if hist.prediction:
                # Confidence distribution
                conf = hist.prediction.confidence
                if conf < 0.2:
                    analysis.prediction_confidence_distribution["0.0-0.2"] = analysis.prediction_confidence_distribution.get("0.0-0.2", 0) + 1
                elif conf < 0.4:
                    analysis.prediction_confidence_distribution["0.2-0.4"] = analysis.prediction_confidence_distribution.get("0.2-0.4", 0) + 1
                elif conf < 0.6:
                    analysis.prediction_confidence_distribution["0.4-0.6"] = analysis.prediction_confidence_distribution.get("0.4-0.6", 0) + 1
                elif conf < 0.8:
                    analysis.prediction_confidence_distribution["0.6-0.8"] = analysis.prediction_confidence_distribution.get("0.6-0.8", 0) + 1
                else:
                    analysis.prediction_confidence_distribution["0.8-1.0"] = analysis.prediction_confidence_distribution.get("0.8-1.0", 0) + 1
                
                # Probability distribution
                prob = hist.prediction.probability
                if prob < 0.2:
                    analysis.prediction_probability_distribution["0.0-0.2"] = analysis.prediction_probability_distribution.get("0.0-0.2", 0) + 1
                elif prob < 0.4:
                    analysis.prediction_probability_distribution["0.2-0.4"] = analysis.prediction_probability_distribution.get("0.2-0.4", 0) + 1
                elif prob < 0.6:
                    analysis.prediction_probability_distribution["0.4-0.6"] = analysis.prediction_probability_distribution.get("0.4-0.6", 0) + 1
                elif prob < 0.8:
                    analysis.prediction_probability_distribution["0.6-0.8"] = analysis.prediction_probability_distribution.get("0.6-0.8", 0) + 1
                else:
                    analysis.prediction_probability_distribution["0.8-1.0"] = analysis.prediction_probability_distribution.get("0.8-1.0", 0) + 1
                
                # Severity distribution
                sev = hist.prediction.severity
                if sev < 0.2:
                    analysis.prediction_severity_distribution["0.0-0.2"] = analysis.prediction_severity_distribution.get("0.0-0.2", 0) + 1
                elif sev < 0.4:
                    analysis.prediction_severity_distribution["0.2-0.4"] = analysis.prediction_severity_distribution.get("0.2-0.4", 0) + 1
                elif sev < 0.6:
                    analysis.prediction_severity_distribution["0.4-0.6"] = analysis.prediction_severity_distribution.get("0.4-0.6", 0) + 1
                elif sev < 0.8:
                    analysis.prediction_severity_distribution["0.6-0.8"] = analysis.prediction_severity_distribution.get("0.6-0.8", 0) + 1
                else:
                    analysis.prediction_severity_distribution["0.8-1.0"] = analysis.prediction_severity_distribution.get("0.8-1.0", 0) + 1
        
        # Model performance
        for model_name, model in self.models.items():
            if model.model_performance:
                analysis.model_performance[model_name] = dict(model.model_performance)
        
        # Feature importance (simplified)
        if self.config.enable_feature_engineering:
            # In a real system, this would come from model analysis
            analysis.feature_importance = {k: v for k, v in self.config.feature_weights.items()}
        
        # Detailed analysis
        if self.config.enable_ml_prediction:
            analysis.detailed_analysis = {
                "models_available": list(self.models.keys()),
                "historical_data_points": len(self.historical_data),
                "prediction_horizon_used": self.config.prediction_horizon,
                "feature_engineering_enabled": self.config.enable_feature_engineering,
                "average_prediction_confidence": sum(
                    hist.prediction.confidence if hist.prediction else 0 
                    for hist in self.prediction_history
                ) / len(self.prediction_history) if self.prediction_history else 0
            }
        
        return analysis
    
    def provide_early_warning(
        self,
        item: Any,
        item_id: str = "",
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None
    ) -> Tuple[bool, List[Prediction], str]:
        """
        Provide early warning if potential issues are predicted.
        
        Args:
            item: Item to analyze
            item_id: Optional ID for the item
            context: Additional context
            history: Historical data
            
        Returns:
            Tuple of (needs_attention, predictions, warning_message)
        """
        if not self.config.enable_early_warning:
            return False, [], "Early warning disabled"
        
        predictions = self.predict_item_quality(item, item_id, context, history)
        
        if not predictions:
            return False, [], "No issues predicted"
        
        # Determine if attention is needed
        attention_needed = any(
            pred.confidence >= self.config.prediction_confidence_threshold and
            pred.probability >= 0.6  # High probability threshold for warnings
            for pred in predictions
        )
        
        if attention_needed:
            warning_message = f"Early warning: {len(predictions)} potential issues predicted with high confidence"
        else:
            warning_message = f"Monitoring: {len(predictions)} potential issues predicted with lower confidence"
        
        return attention_needed, predictions, warning_message
    
    def record_outcome(
        self,
        prediction_id: str,
        actual_outcome: bool,
        actual_severity: Optional[float] = None
    ) -> bool:
        """
        Record the actual outcome of a prediction for feedback.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: Whether the predicted issue actually occurred
            actual_severity: Actual severity if occurred
            
        Returns:
            True if outcome was recorded successfully
        """
        if not self.config.enable_prediction_feedback:
            return False
        
        for hist in self.prediction_history:
            if hist.prediction_id == prediction_id:
                hist.actual_outcome = actual_outcome
                hist.actual_severity = actual_severity
                hist.feedback_timestamp = time.time()
                
                # Calculate prediction accuracy
                if hist.prediction:
                    # Binary accuracy: did prediction direction match outcome?
                    predicted_positive = hist.prediction.probability > 0.5
                    actual_positive = actual_outcome
                    hist.prediction_accuracy = 1.0 if predicted_positive == actual_positive else 0.0
                
                # Update model performance if we have the model
                if hist.prediction and hist.prediction.model_used in self.models:
                    model = self.models[hist.prediction.model_used]
                    model.update_performance(actual_outcome, hist.prediction.probability)
                
                return True
        
        return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        # In a real ML system, this would come from trained models
        # For now, return the configured weights
        return dict(self.config.feature_weights)


# =============================================================================
# Integration with MDAP-MCTS System
# =============================================================================

class MDAPPredictiveFlaggingSystem(PredictiveFlaggingSystem):
    """
    Predictive flagging system specifically for MDAP-MCTS integration.
    
    Adds MDAP-specific prediction capabilities.
    """
    
    def __init__(self, config: Optional[PredictiveFlagConfig] = None):
        super().__init__(config)
        self.agent_prediction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def predict_mdap_node_quality(
        self,
        node: Any,  # MDAPMCTSNode
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for an MDAP node."""
        context = context or {}
        context["node_type"] = "mdap_node"
        
        # Add node-specific information
        if hasattr(node, 'state'):
            context["state_hash"] = getattr(node.state, 'hash', '')
            context["state_goals"] = getattr(node.state, 'goals', [])
            context["state_depth"] = getattr(node.state, 'depth', 0)

        if hasattr(node, 'agent_votes'):
            votes = getattr(node, 'agent_votes')
            if votes and hasattr(votes, 'values'):
                context["vote_count"] = sum(len(v_list) if hasattr(v_list, '__len__') else 0 for v_list in votes.values()) if votes else 0
                context["agent_count"] = len(votes) if votes else 0
            elif votes and hasattr(votes, '__iter__'):
                context["vote_count"] = len(list(votes)) if votes else 0
                context["agent_count"] = 1
            else:
                context["vote_count"] = 0
                context["agent_count"] = 0
        
        # Get historical data for this node's context
        history = self._get_node_history(node)
        
        return self.predict_item_quality(node, getattr(node, 'hash', str(uuid.uuid4())), context, history)
    
    def predict_mdap_action_quality(
        self,
        action: str,
        agent_id: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for an MDAP action."""
        context = context or {}
        context["agent_id"] = agent_id
        context["confidence"] = confidence
        context["action"] = action
        context["node_type"] = "mdap_action"
        
        # Create a simple item to analyze
        item = {
            "action": action,
            "agent_id": agent_id,
            "confidence": confidence
        }
        
        # Get historical data for this agent
        agent_history = self.agent_prediction_history[agent_id]
        
        return self.predict_item_quality(item, f"action_{uuid.uuid4()}", context, agent_history)
    
    def predict_mdap_proof_quality(
        self,
        proof: Any,  # LeanProof
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for an MDAP proof."""
        context = context or {}
        context["node_type"] = "mdap_proof"
        
        # Add proof-specific information
        if hasattr(proof, 'lean_code'):
            context["lean_code"] = getattr(proof, 'lean_code')
        if hasattr(proof, 'confidence'):
            context["confidence"] = getattr(proof, 'confidence')
        if hasattr(proof, 'tactics'):
            tactics = getattr(proof, 'tactics', [])
            if hasattr(tactics, '__len__'):
                context["tactic_count"] = len(tactics)
            else:
                context["tactic_count"] = 1 if tactics else 0
        else:
            context["tactic_count"] = 0
        
        return self.predict_item_quality(
            proof, 
            getattr(proof, 'theorem_name', str(uuid.uuid4())), 
            context
        )
    
    def _get_node_history(self, node: Any) -> List[Any]:
        """Get historical data for a node."""
        # In a real system, this would query historical data
        # For now, return empty list
        return []
    
    def record_agent_outcome(
        self,
        agent_id: str,
        action: str,
        outcome: bool,  # True if successful, False if problematic
        confidence: float,
        prediction_successful: bool = False
    ):
        """Record outcome for an agent's action."""
        self.agent_prediction_history[agent_id].append({
            "action": action,
            "outcome": "success" if outcome else "failure",
            "confidence": confidence,
            "prediction_successful": prediction_successful,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.agent_prediction_history[agent_id]) > 100:  # Limit history size
            self.agent_prediction_history[agent_id] = self.agent_prediction_history[agent_id][-100:]


# =============================================================================
# MCTS-Specific Predictive Flagging
# =============================================================================

class MCTSPredictiveFlaggingSystem(PredictiveFlaggingSystem):
    """
    Predictive flagging system specifically for MCTS integration.
    
    Adds MCTS-specific prediction capabilities.
    """
    
    def __init__(self, config: Optional[PredictiveFlagConfig] = None):
        super().__init__(config)
        self.node_prediction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def predict_mcts_node_quality(
        self,
        node: Any,  # MCTSNode
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for an MCTS node."""
        context = context or {}
        context["node_type"] = "mcts_node"
        
        # Add node-specific information
        if hasattr(node, 'N'):  # visit count
            context["visit_count"] = getattr(node, 'N')
        if hasattr(node, 'W'):  # total reward
            context["total_reward"] = getattr(node, 'W')
        if hasattr(node, 'Q'):  # average reward
            context["avg_reward"] = getattr(node, 'Q')
        if hasattr(node, 'depth'):
            context["depth"] = getattr(node, 'depth')
        
        return self.predict_item_quality(node, getattr(node, 'hash', str(uuid.uuid4())), context)
    
    def predict_mcts_path_quality(
        self,
        path: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for an MCTS path."""
        context = context or {}
        context["node_type"] = "mcts_path"
        context["path_length"] = len(path)
        
        # Analyze the path for potential issues
        if len(path) > 100:  # Very long path might be problematic
            context["path_too_long"] = True
        
        return self.predict_item_quality(path, f"path_{uuid.uuid4()}", context)
    
    def record_node_outcome(
        self,
        node_hash: str,
        outcome: bool,  # True if node led to good result, False if problematic
        visit_count: int,
        reward: float,
        prediction_successful: bool = False
    ):
        """Record outcome for a node."""
        self.node_prediction_history[node_hash].append({
            "outcome": "success" if outcome else "failure",
            "visit_count": visit_count,
            "reward": reward,
            "prediction_successful": prediction_successful,
            "timestamp": time.time()
        })


# =============================================================================
# MAKER-Specific Predictive Flagging
# =============================================================================

class MAKERPredictiveFlaggingSystem(PredictiveFlaggingSystem):
    """
    Predictive flagging system specifically for MAKER integration.
    
    Adds MAKER-specific prediction capabilities.
    """
    
    def __init__(self, config: Optional[PredictiveFlagConfig] = None):
        super().__init__(config)
        self.voter_prediction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def predict_maker_vote_quality(
        self,
        vote: Any,  # TacticVote or ActionVote
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for a MAKER vote."""
        context = context or {}
        context["node_type"] = "maker_vote"
        
        # Add vote-specific information
        if hasattr(vote, 'confidence'):
            context["confidence"] = getattr(vote, 'confidence')
        if hasattr(vote, 'voter_id'):
            context["voter_id"] = getattr(vote, 'voter_id')
        if hasattr(vote, 'tactic') or hasattr(vote, 'action'):
            context["tactic"] = getattr(vote, 'tactic', getattr(vote, 'action', ''))
        
        return self.predict_item_quality(vote, getattr(vote, 'voter_id', str(uuid.uuid4())), context)
    
    def predict_maker_aggregation_quality(
        self,
        votes: List[Any],
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """Predict quality issues for a MAKER aggregation result."""
        context = context or {}
        context["node_type"] = "maker_aggregation"
        context["vote_count"] = len(votes)
        
        # Add vote information
        if votes:
            confidences = []
            for v in votes:
                if hasattr(v, 'confidence'):
                    confidences.append(getattr(v, 'confidence'))
                elif isinstance(v, dict) and 'confidence' in v:
                    confidences.append(v['confidence'])
            
            if confidences:
                context["avg_confidence"] = sum(confidences) / len(confidences)
                if len(confidences) > 1:
                    context["confidence_variance"] = statistics.variance(confidences)
                else:
                    context["confidence_variance"] = 0.0
        
        return self.predict_item_quality(result, f"aggregation_{uuid.uuid4()}", context)
    
    def record_voter_outcome(
        self,
        voter_id: str,
        vote_accepted: bool,
        confidence: float,
        prediction_successful: bool = False
    ):
        """Record outcome for a voter."""
        self.voter_prediction_history[voter_id].append({
            "vote_accepted": vote_accepted,
            "confidence": confidence,
            "prediction_successful": prediction_successful,
            "timestamp": time.time()
        })


# =============================================================================
# Main Integrated Predictive Flagging System
# =============================================================================

class IntegratedPredictiveFlaggingSystem:
    """
    Integrated predictive flagging system for MDAP-MCTS-MAKER.
    
    Combines all specialized predictive flagging systems.
    """
    
    def __init__(self, config: Optional[PredictiveFlagConfig] = None):
        self.config = config or PredictiveFlagConfig()
        self.mdap_system = MDAPPredictiveFlaggingSystem(self.config)
        self.mcts_system = MCTSPredictiveFlaggingSystem(self.config)
        self.maker_system = MAKERPredictiveFlaggingSystem(self.config)
    
    def predict_quality(
        self,
        item: Any,
        item_type: str,  # 'node', 'action', 'proof', 'vote', 'path', 'aggregation'
        context: Optional[Dict[str, Any]] = None
    ) -> List[Prediction]:
        """
        Predict potential quality issues for an item in the MDAP-MCTS-MAKER system.
        
        Args:
            item: The item to predict quality for
            item_type: Type of item ('node', 'action', 'proof', 'vote', 'path', 'aggregation')
            context: Additional context
            
        Returns:
            List of predictions
        """
        context = context or {}
        
        if item_type == 'node':
            if context.get('system') == 'mcts':
                return self.mcts_system.predict_mcts_node_quality(item, context)
            else:
                return self.mdap_system.predict_mdap_node_quality(item, context)
        elif item_type == 'action':
            agent_id = context.get('agent_id', 'unknown')
            confidence = context.get('confidence', 0.5)
            return self.mdap_system.predict_mdap_action_quality(item, agent_id, confidence, context)
        elif item_type == 'proof':
            return self.mdap_system.predict_mdap_proof_quality(item, context)
        elif item_type == 'vote':
            return self.maker_system.predict_maker_vote_quality(item, context)
        elif item_type == 'path':
            return self.mcts_system.predict_mcts_path_quality(item, context)
        elif item_type == 'aggregation':
            votes = context.get('votes', [])
            return self.maker_system.predict_maker_aggregation_quality(votes, item, context)
        else:
            # Generic prediction
            return self.mdap_system.predict_item_quality(item, context=context)
    
    def provide_early_warning(
        self,
        item: Any,
        item_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[Prediction], str]:
        """
        Provide early warning for potential issues.
        
        Args:
            item: Item to analyze
            item_type: Type of item
            context: Additional context
            
        Returns:
            Tuple of (needs_attention, predictions, warning_message)
        """
        predictions = self.predict_quality(item, item_type, context)
        
        if not predictions:
            return False, [], "No issues predicted"
        
        # Determine if attention is needed
        attention_needed = any(
            pred.confidence >= self.config.prediction_confidence_threshold and
            pred.probability >= 0.6  # High probability threshold for warnings
            for pred in predictions
        )
        
        if attention_needed:
            warning_message = f"Early warning: {len(predictions)} potential issues predicted with high confidence"
        else:
            warning_message = f"Monitoring: {len(predictions)} potential issues predicted with lower confidence"
        
        return attention_needed, predictions, warning_message
    
    def analyze_predictions(self) -> Dict[str, Any]:
        """Analyze prediction performance across the system."""
        return {
            "mdap_analysis": self.mdap_system.get_prediction_analysis().to_dict(),
            "mcts_analysis": self.mcts_system.get_prediction_analysis().to_dict(),
            "maker_analysis": self.maker_system.get_prediction_analysis().to_dict(),
            "total_predictions": (
                self.mdap_system.get_prediction_analysis().total_predictions +
                self.mcts_system.get_prediction_analysis().total_predictions +
                self.maker_system.get_prediction_analysis().total_predictions
            )
        }
    
    def record_outcome(
        self,
        system_type: str,  # 'mdap', 'mcts', 'maker'
        item_id: str,
        outcome: bool,
        actual_severity: Optional[float] = None
    ) -> bool:
        """
        Record the actual outcome of a prediction.

        Args:
            system_type: Type of system ('mdap', 'mcts', 'maker')
            item_id: ID of the item
            outcome: Whether the predicted issue actually occurred
            actual_severity: Actual severity if occurred

        Returns:
            True if outcome was recorded successfully
        """
        if system_type == 'mdap':
            return self.mdap_system.record_outcome(item_id, outcome, actual_severity)
        elif system_type == 'mcts':
            return self.mcts_system.record_outcome(item_id, outcome, actual_severity)
        elif system_type == 'maker':
            return self.maker_system.record_outcome(item_id, outcome, actual_severity)
        else:
            # For backward compatibility, try all systems
            success = False
            try:
                success |= self.mdap_system.record_outcome(item_id, outcome, actual_severity)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in leanaide_predictive_flagging.py: {e}", exc_info=True)
                raise
            try:
                success |= self.mcts_system.record_outcome(item_id, outcome, actual_severity)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in leanaide_predictive_flagging.py: {e}", exc_info=True)
                raise
            try:
                success |= self.maker_system.record_outcome(item_id, outcome, actual_severity)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in leanaide_predictive_flagging.py: {e}", exc_info=True)
                raise
            return success


# =============================================================================
# Convenience Functions
# =============================================================================

def create_integrated_predictive_system(
    config: Optional[PredictiveFlagConfig] = None
) -> IntegratedPredictiveFlaggingSystem:
    """
    Create an integrated predictive flagging system.
    
    Args:
        config: Optional configuration
        
    Returns:
        IntegratedPredictiveFlaggingSystem instance
    """
    return IntegratedPredictiveFlaggingSystem(config)


def predict_item_quality(
    item: Any,
    item_type: str,
    config: Optional[PredictiveFlagConfig] = None,
    context: Optional[Dict[str, Any]] = None
) -> List[Prediction]:
    """
    Convenience function to predict quality for an item.
    
    Args:
        item: The item to predict quality for
        item_type: Type of item ('node', 'action', 'proof', 'vote', 'path', 'aggregation')
        config: Optional configuration
        context: Optional context
        
    Returns:
        List of predictions
    """
    system = create_integrated_predictive_system(config)
    return system.predict_quality(item, item_type, context)


def provide_early_warning(
    item: Any,
    item_type: str,
    config: Optional[PredictiveFlagConfig] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[Prediction], str]:
    """
    Convenience function to provide early warning.
    
    Args:
        item: Item to analyze
        item_type: Type of item
        config: Optional configuration
        context: Optional context
        
    Returns:
        Tuple of (needs_attention, predictions, warning_message)
    """
    system = create_integrated_predictive_system(config)
    return system.provide_early_warning(item, item_type, context)


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example usage of the predictive flagging system."""
    print("=" * 80)
    print("Predictive Flagging System Example")
    print("=" * 80)
    
    # Create configuration
    config = PredictiveFlagConfig(
        prediction_confidence_threshold=0.6,
        min_historical_samples=5,
        enable_ml_prediction=True,
        enable_early_warning=True
    )
    
    # Create integrated system
    system = IntegratedPredictiveFlaggingSystem(config)
    
    # Example 1: Predict quality for a low-confidence action
    print("\nExample 1: Predicting quality for low-confidence action")
    predictions = system.predict_quality(
        item="simp",
        item_type="action",
        context={"agent_id": "test_agent", "confidence": 0.2}
    )
    print(f"Predictions made: {len(predictions)}")
    for pred in predictions:
        print(f"  - {pred.prediction_type.value}: {pred.probability:.3f} prob, {pred.confidence:.3f} conf")
    
    # Example 2: Early warning for problematic proof
    print("\nExample 2: Early warning for problematic proof")
    needs_attention, predictions, message = system.provide_early_warning(
        item="theorem test : True := by sorry  -- This uses sorry which is problematic",
        item_type="proof",
        context={"agent_id": "test_agent"}
    )
    print(f"Needs attention: {needs_attention}")
    print(f"Message: {message}")
    for pred in predictions:
        print(f"  - {pred.prediction_type.value}: {pred.probability:.3f} prob")
    
    # Example 3: Predict for MCTS node
    print("\nExample 3: Predicting for MCTS node")
    fake_node = type('FakeNode', (), {
        'N': 50,  # Many visits but low reward might be concerning
        'W': 2,
        'Q': 0.04,
        'hash': 'fake_node_hash'
    })()
    
    predictions = system.predict_quality(
        item=fake_node,
        item_type="node",
        context={"system": "mcts"}
    )
    print(f"MCTS node predictions: {len(predictions)}")
    for pred in predictions:
        print(f"  - {pred.prediction_type.value}: {pred.probability:.3f} prob")
    
    # Example 4: Analyze system predictions
    print("\nExample 4: System prediction analysis")
    analysis = system.analyze_predictions()
    print(f"Total predictions across system: {analysis['total_predictions']}")
    print(f"MDAP predictions: {analysis['mdap_analysis']['total_predictions']}")
    print(f"MCTS predictions: {analysis['mcts_analysis']['total_predictions']}")
    print(f"MAKER predictions: {analysis['maker_analysis']['total_predictions']}")
    
    print("\nPredictive flagging system example completed!")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())