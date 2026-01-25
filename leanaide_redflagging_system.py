"""
LeanAide Red-Flagging System for MCTS-MDAP-MAKER Integration

Comprehensive red-flagging system for quality control in:
    - MCTS (Monte Carlo Tree Search) for intelligent tree search
    - MDAP (Multi-Agent Pipeline) for multi-agent voting
    - MAKER (Multi-Agent Knowledge Enhanced Reasoning) for tactic voting

Features:
    - Multi-level quality assessment
    - Confidence-based flagging
    - Pattern-based detection
    - Performance-based flagging
    - Adaptive threshold adjustment
    - Comprehensive analysis and reporting

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


logger = logging.getLogger(__name__)


# =============================================================================
# Red-Flagging Configuration
# =============================================================================

@dataclass
class RedFlagConfig:
    """
    Configuration for red-flagging system.
    """
    # Confidence-based flagging
    confidence_threshold: float = 0.3  # Below this is flagged
    confidence_variance_threshold: float = 0.1  # High variance triggers flagging
    
    # Pattern-based flagging
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "sorry", "admit", "axiom", "classical.choice", "noncomputable"
    ])
    suspicious_patterns: List[str] = field(default_factory=lambda: [
        "undefined", "error", "failed", "incomplete"
    ])
    
    # Length-based flagging
    max_proof_length: int = 1000  # Max lines
    max_token_count: int = 4000   # Max tokens (approx)
    min_proof_length: int = 1     # Min meaningful proof
    
    # Performance-based flagging
    performance_threshold: float = 0.1  # Agent performance below this is flagged
    vote_agreement_threshold: float = 0.3  # Low agreement triggers flagging
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    threshold_adjustment_rate: float = 0.05  # How quickly thresholds adjust
    
    # Analysis and reporting
    enable_detailed_analysis: bool = True
    enable_performance_tracking: bool = True
    enable_pattern_learning: bool = True
    
    # Integration settings
    enable_flagging: bool = True
    enable_pruning: bool = True
    enable_fallback: bool = True


# =============================================================================
# Red-Flagging Enums and Data Classes
# =============================================================================

class RedFlagType(Enum):
    """Types of red flags."""
    CONFIDENCE_LOW = "confidence_low"
    CONFIDENCE_VARIANCE_HIGH = "confidence_variance_high"
    PATTERN_BLOCKED = "pattern_blocked"
    PATTERN_SUSPICIOUS = "pattern_suspicious"
    LENGTH_TOO_LONG = "length_too_long"
    LENGTH_TOO_SHORT = "length_too_short"
    TOKEN_COUNT_EXCEEDED = "token_count_exceeded"
    PERFORMANCE_POOR = "performance_poor"
    VOTE_AGREEMENT_LOW = "vote_agreement_low"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    LOGIC_ERROR = "logic_error"
    INCONSISTENCY = "inconsistency"
    RECURSION_DEPTH = "recursion_depth"
    RESOURCE_EXCEEDED = "resource_exceeded"


@dataclass
class RedFlag:
    """A single red flag with details."""
    flag_type: RedFlagType
    reason: str
    severity: float  # 0.0 to 1.0
    confidence: float  # Confidence in the flag
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["flag_type"] = self.flag_type.value
        return result


@dataclass
class RedFlagAnalysis:
    """Comprehensive analysis of red flags."""
    total_flags: int = 0
    flag_types: Dict[str, int] = field(default_factory=dict)  # RedFlagType.value -> count
    severity_distribution: Dict[str, int] = field(default_factory=dict)  # severity range -> count
    confidence_distribution: Dict[str, int] = field(default_factory=dict)  # confidence range -> count
    flagged_items: List[str] = field(default_factory=list)  # IDs of flagged items
    analysis_time: float = 0.0
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Core Red-Flagging System
# =============================================================================

class RedFlaggingSystem:
    """
    Comprehensive red-flagging system for quality control.
    
    Features:
    - Multi-level quality assessment
    - Confidence-based flagging
    - Pattern-based detection
    - Performance-based flagging
    - Adaptive threshold adjustment
    - Comprehensive analysis and reporting
    """
    
    def __init__(self, config: Optional[RedFlagConfig] = None):
        """Initialize the red-flagging system."""
        self.config = config or RedFlagConfig()
        self.flag_history: List[RedFlag] = []
        self.performance_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success": 0.0, "total": 0.0, "avg_confidence": 0.5}
        )
        self.pattern_stats: Dict[str, int] = defaultdict(int)
        self.adaptive_thresholds: Dict[str, float] = {}
        
    def flag_item(
        self,
        item: Any,
        item_id: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """
        Flag an item based on quality assessment.
        
        Args:
            item: The item to flag (could be proof, tactic, vote, etc.)
            item_id: Optional ID for the item
            context: Additional context for flagging
            
        Returns:
            Tuple of (is_flagged, list_of_flags)
        """
        if not self.config.enable_flagging:
            return False, []
        
        flags = []
        context = context or {}
        
        # Check different aspects
        flags.extend(self._check_confidence(item, context))
        flags.extend(self._check_patterns(item, context))
        flags.extend(self._check_length(item, context))
        flags.extend(self._check_performance(item, context))
        flags.extend(self._check_votes(item, context))
        flags.extend(self._check_syntax(item, context))
        flags.extend(self._check_logic(item, context))
        
        # Record flags
        self.flag_history.extend(flags)
        
        # Update adaptive thresholds
        if self.config.enable_adaptive_thresholds:
            self._update_adaptive_thresholds(flags)
        
        # Update pattern stats
        if self.config.enable_pattern_learning:
            self._update_pattern_stats(flags)
        
        is_flagged = len(flags) > 0
        
        return is_flagged, flags
    
    def _check_confidence(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check confidence-based flags."""
        flags = []
        
        # Extract confidence from item
        confidence = self._extract_confidence(item)
        
        if confidence is not None and confidence < self.config.confidence_threshold:
            flags.append(RedFlag(
                flag_type=RedFlagType.CONFIDENCE_LOW,
                reason=f"Confidence {confidence:.3f} below threshold {self.config.confidence_threshold:.3f}",
                severity=1.0 - confidence,
                confidence=0.9,
                metadata={"confidence": confidence, "threshold": self.config.confidence_threshold}
            ))
        
        # Check variance if multiple confidence values exist
        confidence_values = self._extract_confidence_values(item)
        if len(confidence_values) > 1:
            variance = sum((c - sum(confidence_values)/len(confidence_values))**2 for c in confidence_values) / len(confidence_values)
            if variance > self.config.confidence_variance_threshold:
                flags.append(RedFlag(
                    flag_type=RedFlagType.CONFIDENCE_VARIANCE_HIGH,
                    reason=f"Confidence variance {variance:.3f} above threshold {self.config.confidence_variance_threshold:.3f}",
                    severity=min(1.0, variance / self.config.confidence_variance_threshold),
                    confidence=0.8,
                    metadata={"variance": variance, "threshold": self.config.confidence_variance_threshold}
                ))
        
        return flags
    
    def _check_patterns(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check pattern-based flags."""
        flags = []
        
        # Convert item to string for pattern matching
        item_str = self._item_to_string(item)
        
        # Check blocked patterns
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in item_str.lower():
                flags.append(RedFlag(
                    flag_type=RedFlagType.PATTERN_BLOCKED,
                    reason=f"Blocked pattern '{pattern}' found",
                    severity=1.0,
                    confidence=1.0,
                    metadata={"pattern": pattern, "matched_text": item_str[:100]}
                ))
        
        # Check suspicious patterns
        for pattern in self.config.suspicious_patterns:
            if pattern.lower() in item_str.lower():
                flags.append(RedFlag(
                    flag_type=RedFlagType.PATTERN_SUSPICIOUS,
                    reason=f"Suspicious pattern '{pattern}' found",
                    severity=0.7,
                    confidence=0.8,
                    metadata={"pattern": pattern, "matched_text": item_str[:100]}
                ))
        
        return flags
    
    def _check_length(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check length-based flags."""
        flags = []
        
        # Get length metrics
        line_count = self._get_line_count(item)
        token_count = self._get_token_count(item)
        
        # Check max length
        if line_count > self.config.max_proof_length:
            flags.append(RedFlag(
                flag_type=RedFlagType.LENGTH_TOO_LONG,
                reason=f"Line count {line_count} exceeds max {self.config.max_proof_length}",
                severity=min(1.0, (line_count - self.config.max_proof_length) / self.config.max_proof_length),
                confidence=0.9,
                metadata={"line_count": line_count, "max_lines": self.config.max_proof_length}
            ))
        
        # Check min length
        if line_count < self.config.min_proof_length:
            flags.append(RedFlag(
                flag_type=RedFlagType.LENGTH_TOO_SHORT,
                reason=f"Line count {line_count} below min {self.config.min_proof_length}",
                severity=1.0,
                confidence=0.8,
                metadata={"line_count": line_count, "min_lines": self.config.min_proof_length}
            ))
        
        # Check token count
        if token_count > self.config.max_token_count:
            flags.append(RedFlag(
                flag_type=RedFlagType.TOKEN_COUNT_EXCEEDED,
                reason=f"Token count {token_count} exceeds max {self.config.max_token_count}",
                severity=min(1.0, (token_count - self.config.max_token_count) / self.config.max_token_count),
                confidence=0.9,
                metadata={"token_count": token_count, "max_tokens": self.config.max_token_count}
            ))
        
        return flags
    
    def _check_performance(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check performance-based flags."""
        flags = []
        
        # Check agent performance if available
        agent_id = context.get("agent_id", "")
        if agent_id and agent_id in self.performance_stats:
            perf = self.performance_stats[agent_id]
            if perf["total"] > 0 and perf["success"] / perf["total"] < self.config.performance_threshold:
                flags.append(RedFlag(
                    flag_type=RedFlagType.PERFORMANCE_POOR,
                    reason=f"Agent {agent_id} performance {perf['success']/perf['total']:.3f} below threshold {self.config.performance_threshold:.3f}",
                    severity=1.0 - (perf["success"] / perf["total"]),
                    confidence=0.8,
                    metadata={
                        "agent_id": agent_id,
                        "success_rate": perf["success"] / perf["total"],
                        "threshold": self.config.performance_threshold
                    }
                ))
        
        return flags
    
    def _check_votes(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check vote agreement flags."""
        flags = []
        
        # Check vote agreement if available
        votes = context.get("votes", [])
        if len(votes) > 1:
            # Calculate agreement (simplified - in real implementation would be more complex)
            unique_votes = len(set(str(v) for v in votes))
            agreement = 1.0 - (unique_votes - 1) / (len(votes) - 1) if len(votes) > 1 else 1.0
            
            if agreement < self.config.vote_agreement_threshold:
                flags.append(RedFlag(
                    flag_type=RedFlagType.VOTE_AGREEMENT_LOW,
                    reason=f"Vote agreement {agreement:.3f} below threshold {self.config.vote_agreement_threshold:.3f}",
                    severity=1.0 - agreement,
                    confidence=0.7,
                    metadata={
                        "agreement": agreement,
                        "threshold": self.config.vote_agreement_threshold,
                        "total_votes": len(votes),
                        "unique_votes": unique_votes
                    }
                ))
        
        return flags
    
    def _check_syntax(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check syntax-based flags."""
        flags = []
        
        # For now, basic syntax checking
        item_str = self._item_to_string(item)
        
        # Check for obvious syntax errors
        if "syntax error" in item_str.lower():
            flags.append(RedFlag(
                flag_type=RedFlagType.SYNTAX_ERROR,
                reason="Syntax error detected in item",
                severity=1.0,
                confidence=0.9,
                metadata={"error_type": "syntax", "matched_text": item_str[:100]}
            ))
        
        return flags
    
    def _check_logic(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> List[RedFlag]:
        """Check logic-based flags."""
        flags = []
        
        # For now, basic logic checking
        item_str = self._item_to_string(item)
        
        # Check for obvious logical inconsistencies
        if "contradiction" in item_str.lower() or "inconsistency" in item_str.lower():
            flags.append(RedFlag(
                flag_type=RedFlagType.LOGIC_ERROR,
                reason="Logical error detected in item",
                severity=1.0,
                confidence=0.8,
                metadata={"error_type": "logic", "matched_text": item_str[:100]}
            ))
        
        return flags
    
    def _extract_confidence(self, item: Any) -> Optional[float]:
        """Extract confidence from an item."""
        if hasattr(item, 'confidence'):
            conf = getattr(item, 'confidence')
            if isinstance(conf, (int, float)):
                return float(conf)
            elif hasattr(conf, '__float__'):
                return float(conf)
        elif isinstance(item, dict) and 'confidence' in item:
            conf = item['confidence']
            if isinstance(conf, (int, float)):
                return float(conf)
            elif hasattr(conf, '__float__'):
                return float(conf)
        elif hasattr(item, 'confidence_score'):
            conf = getattr(item, 'confidence_score')
            if isinstance(conf, (int, float)):
                return float(conf)
            elif hasattr(conf, '__float__'):
                return float(conf)
        return None
    
    def _extract_confidence_values(self, item: Any) -> List[float]:
        """Extract multiple confidence values from an item."""
        if hasattr(item, 'confidence_values'):
            values = getattr(item, 'confidence_values')
            if isinstance(values, list):
                return [float(v) for v in values if isinstance(v, (int, float))]
            else:
                return []
        elif isinstance(item, dict) and 'confidence_values' in item:
            values = item['confidence_values']
            if isinstance(values, list):
                return [float(v) for v in values if isinstance(v, (int, float))]
            else:
                return []
        elif hasattr(item, 'votes'):
            votes = getattr(item, 'votes')
            if votes:
                if hasattr(votes, '__iter__'):  # Check if iterable
                    return [float(v.confidence) if hasattr(v, 'confidence') else 0.5 for v in votes]
                else:
                    return []
        return []
    
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
    
    def _get_line_count(self, item: Any) -> int:
        """Get line count of an item."""
        item_str = self._item_to_string(item)
        return len(item_str.split('\n'))
    
    def _get_token_count(self, item: Any) -> int:
        """Get approximate token count of an item."""
        item_str = self._item_to_string(item)
        # Rough approximation: 4 characters per token
        return max(1, len(item_str) // 4)
    
    def _update_adaptive_thresholds(self, flags: List[RedFlag]):
        """Update adaptive thresholds based on flag patterns."""
        if not self.config.enable_adaptive_thresholds:
            return
        
        # Adjust confidence threshold based on flag patterns
        low_confidence_flags = [f for f in flags if f.flag_type == RedFlagType.CONFIDENCE_LOW]
        if low_confidence_flags:
            # If we're flagging too many items as low confidence, raise threshold
            self.adaptive_thresholds['confidence'] = self.config.confidence_threshold * (1 + self.config.threshold_adjustment_rate)
        else:
            # If we're not flagging enough, lower threshold
            self.adaptive_thresholds['confidence'] = self.config.confidence_threshold * (1 - self.config.threshold_adjustment_rate)
        
        # Keep within reasonable bounds
        self.adaptive_thresholds['confidence'] = max(0.1, min(0.9, self.adaptive_thresholds.get('confidence', self.config.confidence_threshold)))
    
    def _update_pattern_stats(self, flags: List[RedFlag]):
        """Update pattern statistics."""
        if not self.config.enable_pattern_learning:
            return
        
        for flag in flags:
            if flag.flag_type == RedFlagType.PATTERN_BLOCKED:
                pattern = flag.metadata.get('pattern', 'unknown')
                self.pattern_stats[pattern] += 1
    
    def analyze_flags(self, flags: List[RedFlag]) -> RedFlagAnalysis:
        """Perform comprehensive analysis of flags."""
        start_time = time.time()
        
        analysis = RedFlagAnalysis(
            total_flags=len(flags),
            analysis_time=time.time() - start_time
        )
        
        if not flags:
            return analysis
        
        # Count flag types
        for flag in flags:
            type_key = flag.flag_type.value
            analysis.flag_types[type_key] = analysis.flag_types.get(type_key, 0) + 1
        
        # Analyze severity distribution
        severity_ranges = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        for flag in flags:
            if flag.severity < 0.2:
                analysis.severity_distribution["0.0-0.2"] = analysis.severity_distribution.get("0.0-0.2", 0) + 1
            elif flag.severity < 0.4:
                analysis.severity_distribution["0.2-0.4"] = analysis.severity_distribution.get("0.2-0.4", 0) + 1
            elif flag.severity < 0.6:
                analysis.severity_distribution["0.4-0.6"] = analysis.severity_distribution.get("0.4-0.6", 0) + 1
            elif flag.severity < 0.8:
                analysis.severity_distribution["0.6-0.8"] = analysis.severity_distribution.get("0.6-0.8", 0) + 1
            else:
                analysis.severity_distribution["0.8-1.0"] = analysis.severity_distribution.get("0.8-1.0", 0) + 1
        
        # Analyze confidence distribution
        for flag in flags:
            if flag.confidence < 0.2:
                analysis.confidence_distribution["0.0-0.2"] = analysis.confidence_distribution.get("0.0-0.2", 0) + 1
            elif flag.confidence < 0.4:
                analysis.confidence_distribution["0.2-0.4"] = analysis.confidence_distribution.get("0.2-0.4", 0) + 1
            elif flag.confidence < 0.6:
                analysis.confidence_distribution["0.4-0.6"] = analysis.confidence_distribution.get("0.4-0.6", 0) + 1
            elif flag.confidence < 0.8:
                analysis.confidence_distribution["0.6-0.8"] = analysis.confidence_distribution.get("0.6-0.8", 0) + 1
            else:
                analysis.confidence_distribution["0.8-1.0"] = analysis.confidence_distribution.get("0.8-1.0", 0) + 1
        
        # Detailed analysis if enabled
        if self.config.enable_detailed_analysis:
            analysis.detailed_analysis = {
                "most_common_flag_type": max(analysis.flag_types.items(), key=lambda x: x[1])[0] if analysis.flag_types else None,
                "average_severity": sum(f.severity for f in flags) / len(flags),
                "average_confidence": sum(f.confidence for f in flags) / len(flags),
                "high_severity_flags": len([f for f in flags if f.severity > 0.8]),
                "low_confidence_flags": len([f for f in flags if f.confidence < 0.5])
            }
        
        return analysis
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        return dict(self.performance_stats)
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Get pattern statistics."""
        return dict(self.pattern_stats)
    
    def update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        confidence: float
    ):
        """Update agent performance statistics."""
        if not self.config.enable_performance_tracking:
            return
        
        perf = self.performance_stats[agent_id]
        perf["total"] += 1
        if success:
            perf["success"] += 1
        
        # Update average confidence with exponential moving average
        alpha = 0.1
        perf["avg_confidence"] = alpha * confidence + (1 - alpha) * perf["avg_confidence"]
    
    def should_prune(self, flags: List[RedFlag]) -> bool:
        """Determine if an item should be pruned based on flags."""
        if not self.config.enable_pruning:
            return False
        
        if not flags:
            return False
        
        # Calculate overall severity
        avg_severity = sum(f.severity for f in flags) / len(flags)
        
        # High severity items should be pruned
        return avg_severity > 0.7
    
    def get_fallback_action(self, flags: List[RedFlag]) -> str:
        """Get appropriate fallback action based on flags."""
        if not self.config.enable_fallback:
            return "none"
        
        if not flags:
            return "continue"
        
        # Check for critical flags
        critical_flags = [f for f in flags if f.severity > 0.8]
        if critical_flags:
            return "abort"
        
        # Check for confidence issues
        confidence_flags = [f for f in flags if f.flag_type in [RedFlagType.CONFIDENCE_LOW, RedFlagType.CONFIDENCE_VARIANCE_HIGH]]
        if confidence_flags:
            return "retry_with_different_agent"
        
        # Check for pattern issues
        pattern_flags = [f for f in flags if f.flag_type in [RedFlagType.PATTERN_BLOCKED, RedFlagType.PATTERN_SUSPICIOUS]]
        if pattern_flags:
            return "use_alternative_approach"
        
        return "continue_with_caution"


# =============================================================================
# Integration with MDAP-MCTS System
# =============================================================================

class MDAPRedFlaggingSystem(RedFlaggingSystem):
    """
    Red-flagging system specifically for MDAP-MCTS integration.
    
    Adds MDAP-specific flagging capabilities.
    """
    
    def __init__(self, config: Optional[RedFlagConfig] = None):
        super().__init__(config)
        self.agent_vote_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_votes": 0, "flagged_votes": 0, "avg_confidence": 0.5}
        )
    
    def flag_mdap_node(
        self,
        node: Any,  # MDAPMCTSNode
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an MDAP node specifically."""
        context = context or {}
        context["node_type"] = "mdap_node"
        
        # Add node-specific information
        if hasattr(node, 'state'):
            context["state_hash"] = getattr(node.state, 'hash', '')
            context["state_goals"] = getattr(node.state, 'goals', [])
        
        if hasattr(node, 'agent_votes'):
            votes = getattr(node, 'agent_votes')
            if votes and hasattr(votes, 'values'):
                context["votes"] = list(votes.values()) if votes else []
            elif votes and hasattr(votes, '__iter__'):
                context["votes"] = list(votes)
            else:
                context["votes"] = []
        
        return self.flag_item(node, item_id=getattr(node, 'hash', str(uuid.uuid4())), context=context)
    
    def flag_mdap_action(
        self,
        action: str,
        agent_id: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an MDAP action specifically."""
        context = context or {}
        context["agent_id"] = agent_id
        context["confidence"] = confidence
        context["action"] = action
        context["node_type"] = "mdap_action"
        
        # Track agent vote statistics
        self._update_agent_vote_stats(agent_id, confidence)
        
        return self.flag_item(action, item_id=f"action_{uuid.uuid4()}", context=context)
    
    def flag_mdap_proof(
        self,
        proof: Any,  # LeanProof
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an MDAP proof specifically."""
        context = context or {}
        context["node_type"] = "mdap_proof"
        
        # Add proof-specific information
        if hasattr(proof, 'lean_code'):
            context["lean_code"] = getattr(proof, 'lean_code')
        if hasattr(proof, 'confidence'):
            context["confidence"] = getattr(proof, 'confidence')
        
        return self.flag_item(proof, item_id=getattr(proof, 'theorem_name', str(uuid.uuid4())), context=context)
    
    def _update_agent_vote_stats(self, agent_id: str, confidence: float):
        """Update agent vote statistics."""
        stats = self.agent_vote_stats[agent_id]
        stats["total_votes"] += 1
        stats["avg_confidence"] = (stats["avg_confidence"] * (stats["total_votes"] - 1) + confidence) / stats["total_votes"]
    
    def get_agent_vote_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get agent vote statistics."""
        return dict(self.agent_vote_stats)
    
    def should_retry_agent(self, agent_id: str) -> bool:
        """Determine if an agent should be retried based on vote statistics."""
        if agent_id not in self.agent_vote_stats:
            return True
        
        stats = self.agent_vote_stats[agent_id]
        if stats["total_votes"] < 5:  # Not enough data
            return True
        
        # If average confidence is low, consider retrying
        return stats["avg_confidence"] > 0.3


# =============================================================================
# MCTS-Specific Red-Flagging
# =============================================================================

class MCTSRedFlaggingSystem(RedFlaggingSystem):
    """
    Red-flagging system specifically for MCTS integration.
    
    Adds MCTS-specific flagging capabilities.
    """
    
    def __init__(self, config: Optional[RedFlagConfig] = None):
        super().__init__(config)
        self.node_statistics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"visits": 0, "wins": 0, "avg_reward": 0.0}
        )
    
    def flag_mcts_node(
        self,
        node: Any,  # MCTSNode
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an MCTS node specifically."""
        context = context or {}
        context["node_type"] = "mcts_node"
        
        # Add node-specific information
        if hasattr(node, 'N'):  # visit count
            context["visit_count"] = getattr(node, 'N')
        if hasattr(node, 'W'):  # total reward
            context["total_reward"] = getattr(node, 'W')
        if hasattr(node, 'Q'):  # average reward
            context["avg_reward"] = getattr(node, 'Q')
        
        return self.flag_item(node, item_id=getattr(node, 'hash', str(uuid.uuid4())), context=context)
    
    def flag_mcts_path(
        self,
        path: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an MCTS path specifically."""
        context = context or {}
        context["node_type"] = "mcts_path"
        context["path_length"] = len(path)
        
        # Analyze the path for potential issues
        if len(path) > 100:  # Very long path might be problematic
            context["path_too_long"] = True
        
        return self.flag_item(path, item_id=f"path_{uuid.uuid4()}", context=context)
    
    def update_node_statistics(
        self,
        node_hash: str,
        reward: float,
        is_win: bool
    ):
        """Update node statistics for flagging decisions."""
        stats = self.node_statistics[node_hash]
        stats["visits"] += 1
        if is_win:
            stats["wins"] += 1
        stats["avg_reward"] = (stats["avg_reward"] * (stats["visits"] - 1) + reward) / stats["visits"]
    
    def should_explore_further(self, node_hash: str) -> bool:
        """Determine if a node should be explored further based on statistics."""
        if node_hash not in self.node_statistics:
            return True
        
        stats = self.node_statistics[node_hash]
        if stats["visits"] < 10:  # Not explored enough
            return True
        
        # If win rate is very low, consider stopping exploration
        win_rate = stats["wins"] / stats["visits"]
        return win_rate > 0.1


# =============================================================================
# MAKER-Specific Red-Flagging
# =============================================================================

class MAKERRedFlaggingSystem(RedFlaggingSystem):
    """
    Red-flagging system specifically for MAKER integration.
    
    Adds MAKER-specific flagging capabilities.
    """
    
    def __init__(self, config: Optional[RedFlagConfig] = None):
        super().__init__(config)
        self.voter_statistics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total_votes": 0, "accepted_votes": 0, "avg_confidence": 0.5}
        )
    
    def flag_maker_vote(
        self,
        vote: Any,  # TacticVote or ActionVote
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag a MAKER vote specifically."""
        context = context or {}
        context["node_type"] = "maker_vote"
        
        # Add vote-specific information
        if hasattr(vote, 'confidence'):
            context["confidence"] = getattr(vote, 'confidence')
        if hasattr(vote, 'voter_id'):
            context["voter_id"] = getattr(vote, 'voter_id')
        if hasattr(vote, 'tactic') or hasattr(vote, 'action'):
            context["tactic"] = getattr(vote, 'tactic', getattr(vote, 'action', ''))
        
        return self.flag_item(vote, item_id=getattr(vote, 'voter_id', str(uuid.uuid4())), context=context)
    
    def flag_maker_aggregation(
        self,
        votes: List[Any],
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag a MAKER aggregation result specifically."""
        context = context or {}
        context["node_type"] = "maker_aggregation"
        context["vote_count"] = len(votes)
        
        # Add vote information
        if votes:
            confidences = [getattr(v, 'confidence', 0.5) for v in votes]
            context["avg_confidence"] = sum(confidences) / len(confidences)
            context["confidence_variance"] = sum((c - context["avg_confidence"])**2 for c in confidences) / len(confidences)
        
        return self.flag_item(result, item_id=f"aggregation_{uuid.uuid4()}", context=context)
    
    def update_voter_statistics(
        self,
        voter_id: str,
        vote_accepted: bool,
        confidence: float
    ):
        """Update voter statistics."""
        stats = self.voter_statistics[voter_id]
        stats["total_votes"] += 1
        if vote_accepted:
            stats["accepted_votes"] += 1
        stats["avg_confidence"] = (stats["avg_confidence"] * (stats["total_votes"] - 1) + confidence) / stats["total_votes"]
    
    def get_reliable_voters(self, min_acceptance_rate: float = 0.5) -> List[str]:
        """Get list of reliable voters."""
        reliable = []
        for voter_id, stats in self.voter_statistics.items():
            if stats["total_votes"] > 0:
                acceptance_rate = stats["accepted_votes"] / stats["total_votes"]
                if acceptance_rate >= min_acceptance_rate:
                    reliable.append(voter_id)
        return reliable


# =============================================================================
# Main Integrated Red-Flagging System
# =============================================================================

class IntegratedRedFlaggingSystem:
    """
    Integrated red-flagging system for MDAP-MCTS-MAKER.
    
    Combines all specialized red-flagging systems.
    """
    
    def __init__(self, config: Optional[RedFlagConfig] = None):
        self.config = config or RedFlagConfig()
        self.mdap_system = MDAPRedFlaggingSystem(self.config)
        self.mcts_system = MCTSRedFlaggingSystem(self.config)
        self.maker_system = MAKERRedFlaggingSystem(self.config)
    
    def flag_mdap_mcts_item(
        self,
        item: Any,
        item_type: str,  # 'node', 'action', 'proof', 'vote', 'path', 'aggregation'
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """Flag an item in the MDAP-MCTS-MAKER system."""
        context = context or {}
        
        if item_type == 'node':
            if context.get('system') == 'mcts':
                return self.mcts_system.flag_mcts_node(item, context)
            else:
                return self.mdap_system.flag_mdap_node(item, context)
        elif item_type == 'action':
            agent_id = context.get('agent_id', 'unknown')
            confidence = context.get('confidence', 0.5)
            return self.mdap_system.flag_mdap_action(item, agent_id, confidence, context)
        elif item_type == 'proof':
            return self.mdap_system.flag_mdap_proof(item, context)
        elif item_type == 'vote':
            return self.maker_system.flag_maker_vote(item, context)
        elif item_type == 'path':
            return self.mcts_system.flag_mcts_path(item, context)
        elif item_type == 'aggregation':
            votes = context.get('votes', [])
            return self.maker_system.flag_maker_aggregation(votes, item, context)
        else:
            # Generic flagging
            return self.mdap_system.flag_item(item, context=context)
    
    def analyze_system_flags(self, flags: List[RedFlag]) -> Dict[str, Any]:
        """Analyze flags across the entire system."""
        mdap_flags = [f for f in flags if 'mdap' in f.reason.lower() or f.flag_type in [
            RedFlagType.CONFIDENCE_LOW, RedFlagType.PERFORMANCE_POOR
        ]]
        mcts_flags = [f for f in flags if 'mcts' in f.reason.lower() or f.flag_type in [
            RedFlagType.RECURSION_DEPTH, RedFlagType.RESOURCE_EXCEEDED
        ]]
        maker_flags = [f for f in flags if 'maker' in f.reason.lower() or f.flag_type in [
            RedFlagType.VOTE_AGREEMENT_LOW
        ]]
        
        return {
            "total_flags": len(flags),
            "mdap_flags": len(mdap_flags),
            "mcts_flags": len(mcts_flags),
            "maker_flags": len(maker_flags),
            "mdap_analysis": self.mdap_system.analyze_flags(mdap_flags),
            "mcts_analysis": self.mcts_system.analyze_flags(mcts_flags),
            "maker_analysis": self.maker_system.analyze_flags(maker_flags)
        }
    
    def get_system_recommendations(self, flags: List[RedFlag]) -> List[str]:
        """Get system-wide recommendations based on flags."""
        recommendations = []
        
        # Check for patterns that suggest system-wide issues
        flag_types = [f.flag_type for f in flags]
        
        if RedFlagType.CONFIDENCE_LOW in flag_types:
            recommendations.append("Consider lowering confidence threshold or using different agents")
        
        if RedFlagType.VOTE_AGREEMENT_LOW in flag_types:
            recommendations.append("Increase number of voters or adjust voting strategy")
        
        if RedFlagType.PERFORMANCE_POOR in flag_types:
            recommendations.append("Review agent performance and consider agent replacement")
        
        if RedFlagType.PATTERN_BLOCKED in flag_types:
            recommendations.append("Review blocked patterns and adjust as needed")
        
        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def create_integrated_red_flagging_system(
    config: Optional[RedFlagConfig] = None
) -> IntegratedRedFlaggingSystem:
    """
    Create an integrated red-flagging system.
    
    Args:
        config: Optional configuration
        
    Returns:
        IntegratedRedFlaggingSystem instance
    """
    return IntegratedRedFlaggingSystem(config)


def flag_mdap_mcts_item(
    item: Any,
    item_type: str,
    config: Optional[RedFlagConfig] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[RedFlag]]:
    """
    Convenience function to flag an item in MDAP-MCTS-MAKER system.
    
    Args:
        item: The item to flag
        item_type: Type of item ('node', 'action', 'proof', 'vote', 'path', 'aggregation')
        config: Optional configuration
        context: Optional context
        
    Returns:
        Tuple of (is_flagged, list_of_flags)
    """
    system = create_integrated_red_flagging_system(config)
    return system.flag_mdap_mcts_item(item, item_type, context)


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example usage of the red-flagging system."""
    print("=" * 80)
    print("Red-Flagging System Example")
    print("=" * 80)
    
    # Create configuration
    config = RedFlagConfig(
        confidence_threshold=0.4,
        max_proof_length=500,
        enable_detailed_analysis=True
    )
    
    # Create integrated system
    system = IntegratedRedFlaggingSystem(config)
    
    # Example 1: Flag a low-confidence action
    print("\nExample 1: Flagging low-confidence action")
    is_flagged, flags = system.flag_mdap_mcts_item(
        item="simp",
        item_type="action",
        context={"agent_id": "test_agent", "confidence": 0.2}
    )
    print(f"Flagged: {is_flagged}")
    for flag in flags:
        print(f"  - {flag.flag_type.value}: {flag.reason}")
    
    # Example 2: Flag a proof with blocked patterns
    print("\nExample 2: Flagging proof with blocked patterns")
    bad_proof = "theorem test : True := by sorry  -- This uses sorry which is blocked"
    is_flagged, flags = system.flag_mdap_mcts_item(
        item=bad_proof,
        item_type="proof"
    )
    print(f"Flagged: {is_flagged}")
    for flag in flags:
        print(f"  - {flag.flag_type.value}: {flag.reason}")
    
    # Example 3: Flag a vote with low agreement
    print("\nExample 3: Flagging vote with low agreement")
    votes = ["simp", "intros", "rw", "apply", "cases"]  # All different
    is_flagged, flags = system.flag_mdap_mcts_item(
        item="selected_tactic",
        item_type="aggregation",
        context={"votes": votes}
    )
    print(f"Flagged: {is_flagged}")
    for flag in flags:
        print(f"  - {flag.flag_type.value}: {flag.reason}")
    
    # Example 4: Analyze system flags
    print("\nExample 4: System flag analysis")
    all_flags = flags  # Use flags from previous example
    analysis = system.analyze_system_flags(all_flags)
    print(f"Total flags: {analysis['total_flags']}")
    print(f"MDAP flags: {analysis['mdap_flags']}")
    print(f"MCTS flags: {analysis['mcts_flags']}")
    print(f"MAKER flags: {analysis['maker_flags']}")
    
    # Example 5: Get recommendations
    recommendations = system.get_system_recommendations(all_flags)
    print(f"\nRecommendations: {recommendations}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())