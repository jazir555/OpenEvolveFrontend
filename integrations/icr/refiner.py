"""Refinement application for ICR.

Applies improvements to outputs based on critique feedback through
multiple refinement strategies.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum, auto
from copy import deepcopy

from integrations.icr.critic import Issue, Suggestion, IssueType, Severity

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Strategies for applying refinements."""
    INCREMENTAL = "incremental"  # Small targeted changes
    REWRITE = "rewrite"  # Full regeneration with feedback
    HYBRID = "hybrid"  # Mix of incremental and rewrite


@dataclass
class Change:
    """Record of a single change."""
    description: str
    issue_type: IssueType
    before: str
    after: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "issue_type": self.issue_type.value,
            "before": self.before,
            "after": self.after,
            "timestamp": self.timestamp,
        }


@dataclass
class RefinedOutput:
    """Result of refinement operation."""
    content: str
    original_content: str
    changes: List[Change]
    improvement_score: float = 0.0
    refinement_time: float = 0.0
    strategy_used: RefinementStrategy = RefinementStrategy.INCREMENTAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def change_count(self) -> int:
        """Number of changes made."""
        return len(self.changes)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "original_content": self.original_content,
            "changes": [c.to_dict() for c in self.changes],
            "improvement_score": self.improvement_score,
            "refinement_time": self.refinement_time,
            "strategy_used": self.strategy_used.value,
            "metadata": self.metadata,
        }


@dataclass
class ModifiedOutput:
    """Result of single suggestion application."""
    content: str
    suggestion: Suggestion
    applied: bool
    reason: str = ""


@dataclass
class CombinedOutput:
    """Result of merging multiple suggestions."""
    content: str
    applied_suggestions: List[Suggestion]
    rejected_suggestions: List[Suggestion]
    conflicts: List[str]


class RefinementTracker:
    """Track refinement changes and convergence.
    
    Attributes:
        changes: List of all changes made
        improvement_history: Quality scores over iterations
        convergence_score: Measure of convergence (0-1)
    """
    
    def __init__(self, max_history: int = 100):
        self.changes: List[Change] = []
        self.improvement_history: List[float] = []
        self.convergence_score: float = 0.0
        self._max_history = max_history
        self._start_time: Optional[str] = None
        self._iteration_count = 0
    
    def start_tracking(self) -> None:
        """Start tracking a refinement session."""
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._iteration_count = 0
        logger.debug("Refinement tracking started")
    
    def record_change(self, change: Change) -> None:
        """Record a change."""
        self.changes.append(change)
        if len(self.changes) > self._max_history:
            self.changes = self.changes[-self._max_history:]
    
    def record_score(self, score: float) -> None:
        """Record a quality score."""
        self.improvement_history.append(score)
        self._iteration_count += 1
        self._update_convergence()
    
    def _update_convergence(self) -> None:
        """Update convergence score based on recent history."""
        if len(self.improvement_history) < 3:
            self.convergence_score = 0.0
            return
        
        # Look at last 3 scores
        recent = self.improvement_history[-3:]
        variance = sum((x - sum(recent) / len(recent)) ** 2 for x in recent)
        
        # Lower variance = higher convergence
        self.convergence_score = max(0.0, 1.0 - variance * 10)
    
    @property
    def has_converged(self, threshold: float = 0.8) -> bool:
        """Check if refinement has converged."""
        return self.convergence_score >= threshold
    
    @property
    def improvement_trend(self) -> str:
        """Get trend direction of improvements."""
        if len(self.improvement_history) < 2:
            return "insufficient_data"
        
        recent = self.improvement_history[-5:]
        if len(recent) < 2:
            recent = self.improvement_history[-2:]
        
        diff = recent[-1] - recent[0]
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            "total_changes": len(self.changes),
            "iterations": self._iteration_count,
            "convergence_score": self.convergence_score,
            "has_converged": self.has_converged,
            "trend": self.improvement_trend,
            "start_time": self._start_time,
            "current_score": self.improvement_history[-1] if self.improvement_history else None,
            "score_history": self.improvement_history,
        }


class Refiner:
    """Apply improvements to outputs based on critique feedback.
    
    The Refiner takes critique results and applies the suggested improvements
    to produce refined content. Supports multiple refinement strategies.
    
    Example:
        >>> refiner = Refiner(strategy=RefinementStrategy.INCREMENTAL)
        >>> result = refiner.refine(
        ...     output="Original content...",
        ...     critique=critique_result
        ... )
        >>> print(result.content)
        >>> print(f"Made {result.change_count} improvements")
    """
    
    def __init__(
        self,
        strategy: RefinementStrategy = RefinementStrategy.HYBRID,
        max_changes_per_iteration: int = 10,
        preserve_structure: bool = True,
    ):
        """Initialize the refiner.
        
        Args:
            strategy: Default refinement strategy
            max_changes_per_iteration: Maximum changes to apply at once
            preserve_structure: Try to maintain document structure
        """
        self.strategy = strategy
        self.max_changes_per_iteration = max_changes_per_iteration
        self.preserve_structure = preserve_structure
        self.tracker = RefinementTracker()
        self._refinement_count = 0
        self._backend: Optional[Callable] = None
        
        # Auto-fix rules for common issues
        self._auto_fix_rules = self._compile_auto_fix_rules()
        
        logger.info(f"Initialized Refiner with strategy={strategy.value}")
    
    def set_backend(self, backend: Callable[[str, str, Dict[str, Any]], str]) -> None:
        """Set backend for rewrite-based refinements.
        
        Args:
            backend: Callable that takes (content, feedback, params) and returns refined content
        """
        self._backend = backend
        logger.debug("Refinement backend registered")
    
    def refine(
        self,
        output: str,
        critique: Any,  # CritiqueResult
        strategy: Optional[RefinementStrategy] = None,
    ) -> RefinedOutput:
        """Refine output based on critique.
        
        Args:
            output: Original content to refine
            critique: CritiqueResult with issues and suggestions
            strategy: Override default strategy
            
        Returns:
            RefinedOutput with improved content and change tracking
        """
        import time
        start_time = time.time()
        
        strategy = strategy or self.strategy
        self._refinement_count += 1
        
        logger.info(f"Starting refinement with strategy={strategy.value}")
        
        if strategy == RefinementStrategy.REWRITE:
            result = self._rewrite_refinement(output, critique)
        elif strategy == RefinementStrategy.INCREMENTAL:
            result = self._incremental_refinement(output, critique)
        else:  # HYBRID
            result = self._hybrid_refinement(output, critique)
        
        refinement_time = time.time() - start_time
        
        refined = RefinedOutput(
            content=result,
            original_content=output,
            changes=self.tracker.changes.copy(),
            improvement_score=self._estimate_improvement(output, result),
            refinement_time=refinement_time,
            strategy_used=strategy,
            metadata={
                "refinement_number": self._refinement_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "critique_score": getattr(critique, 'score', None),
            },
        )
        
        logger.debug(f"Refinement complete in {refinement_time:.3f}s")
        return refined
    
    def apply_suggestion(
        self,
        output: str,
        suggestion: Suggestion,
    ) -> ModifiedOutput:
        """Apply a single suggestion to output.
        
        Args:
            output: Content to modify
            suggestion: Suggestion to apply
            
        Returns:
            ModifiedOutput with result
        """
        if not suggestion.automated:
            return ModifiedOutput(
                content=output,
                suggestion=suggestion,
                applied=False,
                reason="Suggestion requires manual intervention",
            )
        
        original = output
        applied = False
        
        # Apply based on issue type
        if suggestion.issue.type == IssueType.GRAMMAR:
            output = self._fix_grammar(output, suggestion)
            applied = True
        elif suggestion.issue.type == IssueType.CONCISENESS:
            output = self._fix_conciseness(output, suggestion)
            applied = True
        elif suggestion.issue.type == IssueType.CLARITY:
            output = self._fix_clarity(output, suggestion)
            applied = True
        elif suggestion.issue.type == IssueType.STYLE:
            output = self._fix_style(output, suggestion)
            applied = True
        
        if applied:
            change = Change(
                description=suggestion.fix,
                issue_type=suggestion.issue.type,
                before=original,
                after=output,
            )
            self.tracker.record_change(change)
        
        return ModifiedOutput(
            content=output,
            suggestion=suggestion,
            applied=applied,
            reason="" if applied else "Could not auto-apply suggestion",
        )
    
    def merge_suggestions(
        self,
        output: str,
        suggestions: List[Suggestion],
    ) -> CombinedOutput:
        """Merge multiple suggestions into output.
        
        Args:
            output: Content to modify
            suggestions: List of suggestions to apply
            
        Returns:
            CombinedOutput with merged result
        """
        applied = []
        rejected = []
        conflicts = []
        
        # Sort by priority
        sorted_suggestions = sorted(suggestions, key=lambda s: s.priority)
        
        # Take top N suggestions
        to_apply = sorted_suggestions[:self.max_changes_per_iteration]
        
        for suggestion in to_apply:
            result = self.apply_suggestion(output, suggestion)
            if result.applied:
                output = result.content
                applied.append(suggestion)
            else:
                rejected.append(suggestion)
        
        return CombinedOutput(
            content=output,
            applied_suggestions=applied,
            rejected_suggestions=rejected,
            conflicts=conflicts,
        )
    
    def _incremental_refinement(self, output: str, critique: Any) -> str:
        """Apply incremental changes."""
        suggestions = getattr(critique, 'suggestions', [])
        
        # Filter to automated suggestions only
        auto_suggestions = [s for s in suggestions if s.automated]
        
        if not auto_suggestions:
            logger.debug("No automated suggestions to apply")
            return output
        
        result = self.merge_suggestions(output, auto_suggestions)
        return result.content
    
    def _rewrite_refinement(self, output: str, critique: Any) -> str:
        """Full rewrite with feedback."""
        if self._backend:
            feedback = self._critique_to_feedback(critique)
            return self._backend(output, feedback, {})
        else:
            # Fallback: just add a note
            issues_summary = "\n".join([
                f"- {i.description}" for i in getattr(critique, 'issues', [])[:3]
            ])
            return f"{output}\n\n[Refined based on feedback:\n{issues_summary}\n]"
    
    def _hybrid_refinement(self, output: str, critique: Any) -> str:
        """Combine incremental and rewrite approaches."""
        # First apply incremental fixes for simple issues
        output = self._incremental_refinement(output, critique)
        
        # Check if major issues remain
        major_issues = [
            i for i in getattr(critique, 'issues', [])
            if i.severity in (Severity.MAJOR, Severity.CRITICAL)
        ]
        
        # If major issues remain and we have a backend, do a targeted rewrite
        if major_issues and self._backend:
            feedback = "Please address these major issues:\n"
            for issue in major_issues[:3]:
                feedback += f"- {issue.description}\n"
            output = self._backend(output, feedback, {"temperature": 0.3})
        
        return output
    
    def _compile_auto_fix_rules(self) -> Dict[str, Callable]:
        """Compile rules for automatic fixes."""
        return {
            "double_spaces": lambda t: re.sub(r'  +', ' ', t),
            "trim_whitespace": lambda t: t.strip(),
        }
    
    def _fix_grammar(self, text: str, suggestion: Suggestion) -> str:
        """Apply grammar fixes."""
        for rule_name, rule_fn in self._auto_fix_rules.items():
            text = rule_fn(text)
        return text
    
    def _fix_conciseness(self, text: str, suggestion: Suggestion) -> str:
        """Apply conciseness fixes."""
        # Remove redundant phrases
        replacements = {
            r'\bclose proximity\b': 'proximity',
            r'\bend result\b': 'result',
            r'\bfree gift\b': 'gift',
            r'\bvery ([a-z]+)\b': r'\1',
            r'\breally ([a-z]+)\b': r'\1',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.I)
        return text
    
    def _fix_clarity(self, text: str, suggestion: Suggestion) -> str:
        """Apply clarity fixes."""
        # Replace vague words with more specific alternatives
        replacements = {
            r'\bmaybe\b': 'likely',
            r'\bperhaps\b': 'possibly',
            r'\bsomewhat\b': 'moderately',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.I)
        return text
    
    def _fix_style(self, text: str, suggestion: Suggestion) -> str:
        """Apply style fixes."""
        # Remove forbidden patterns
        if hasattr(suggestion.issue, 'description'):
            desc = suggestion.issue.description.lower()
            if "forbidden pattern" in desc:
                # Extract pattern from description
                match = re.search(r"'([^']+)'", desc)
                if match:
                    pattern = match.group(1)
                    text = re.sub(re.escape(pattern), '', text, flags=re.I)
        return text
    
    def _critique_to_feedback(self, critique: Any) -> str:
        """Convert critique to feedback string."""
        lines = ["Please improve the following content:"]
        
        for issue in getattr(critique, 'issues', [])[:5]:
            lines.append(f"\n- {issue.severity.value.upper()}: {issue.description}")
        
        for strength in getattr(critique, 'strengths', [])[:3]:
            lines.append(f"\n+ Keep: {strength}")
        
        return "\n".join(lines)
    
    def _estimate_improvement(self, original: str, refined: str) -> float:
        """Estimate improvement score."""
        # Simple heuristic: refined should be different but not too different
        if original == refined:
            return 0.0
        
        length_diff = abs(len(refined) - len(original)) / max(len(original), 1)
        if length_diff > 0.5:
            return 0.5  # Major changes, uncertain improvement
        
        return 0.15  # Assume moderate improvement
    
    def get_stats(self) -> Dict[str, Any]:
        """Get refiner statistics."""
        return {
            "total_refinements": self._refinement_count,
            "strategy": self.strategy.value,
            "tracker": self.tracker.get_stats(),
        }


class RefinementError(Exception):
    """Error during refinement operation."""
    pass
