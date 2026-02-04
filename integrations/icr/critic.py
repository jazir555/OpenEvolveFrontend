"""Critique generation for ICR.

Identifies issues in generated outputs and suggests improvements
through systematic analysis across multiple quality dimensions.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of issues that can be identified."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    CONCISENESS = "conciseness"
    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    COHERENCE = "coherence"
    STYLE = "style"
    STRUCTURE = "structure"
    GRAMMAR = "grammar"


class Severity(Enum):
    """Severity levels for issues."""
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    
    def weight(self) -> float:
        """Get numeric weight for scoring."""
        return {
            Severity.MINOR: 0.1,
            Severity.MAJOR: 0.3,
            Severity.CRITICAL: 0.6,
        }[self]


@dataclass
class Issue:
    """Identified problem in output.
    
    Attributes:
        type: Category of the issue
        severity: How severe the issue is
        description: Human-readable description
        location: Optional location reference (line number, section, etc.)
        suggestion_id: Optional reference to related suggestion
    """
    type: IssueType
    severity: Severity
    description: str
    location: Optional[str] = None
    suggestion_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate issue fields."""
        if isinstance(self.type, str):
            self.type = IssueType(self.type)
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "suggestion_id": self.suggestion_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Issue":
        """Create from dictionary."""
        return cls(
            type=IssueType(data["type"]),
            severity=Severity(data["severity"]),
            description=data["description"],
            location=data.get("location"),
            suggestion_id=data.get("suggestion_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Suggestion:
    """Improvement suggestion.
    
    Attributes:
        issue: The issue this suggestion addresses
        fix: Description of the fix to apply
        priority: Priority number (lower = higher priority)
        estimated_impact: Expected improvement in quality score
        automated: Whether this suggestion can be auto-applied
    """
    issue: Issue
    fix: str
    priority: int = 5
    estimated_impact: float = 0.1
    automated: bool = False
    id: str = field(default_factory=lambda: f"sugg_{datetime.now(timezone.utc).timestamp()}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "issue": self.issue.to_dict(),
            "fix": self.fix,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact,
            "automated": self.automated,
        }


@dataclass
class CritiqueResult:
    """Result of a critique operation.
    
    Attributes:
        score: Overall quality score (0-1)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        strengths: Positive aspects of the output
        metadata: Additional critique metadata
        timestamp: When the critique was performed
    """
    score: float
    issues: List[Issue]
    suggestions: List[Suggestion]
    strengths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Validate score range."""
        self.score = max(0.0, min(1.0, self.score))
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist."""
        return any(i.severity == Severity.CRITICAL for i in self.issues)
    
    @property
    def issue_count(self) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {"minor": 0, "major": 0, "critical": 0}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts
    
    @property
    def top_suggestions(self, n: int = 5) -> List[Suggestion]:
        """Get top N suggestions by priority."""
        return sorted(self.suggestions, key=lambda s: s.priority)[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "strengths": self.strengths,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class CritiqueCriteria:
    """Criteria configuration for critique."""
    
    def __init__(
        self,
        check_accuracy: bool = True,
        check_completeness: bool = True,
        check_clarity: bool = True,
        check_conciseness: bool = True,
        check_consistency: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_elements: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ):
        self.check_accuracy = check_accuracy
        self.check_completeness = check_completeness
        self.check_clarity = check_clarity
        self.check_conciseness = check_conciseness
        self.check_consistency = check_consistency
        self.min_length = min_length
        self.max_length = max_length
        self.required_elements = required_elements or []
        self.forbidden_patterns = forbidden_patterns or []


class Critic:
    """Critique generator for outputs.
    
    The Critic analyzes generated content and identifies issues
    across multiple quality dimensions, then suggests improvements.
    
    Example:
        >>> critic = Critic()
        >>> result = critic.critique(
        ...     output="This is the generated content...",
        ...     criteria=CritiqueCriteria(check_completeness=True)
        ... )
        >>> print(f"Score: {result.score}")
        >>> for issue in result.issues:
        ...     print(f"- {issue.severity.value}: {issue.description}")
    """
    
    def __init__(
        self,
        default_criteria: Optional[CritiqueCriteria] = None,
        auto_suggest: bool = True,
    ):
        """Initialize the critic.
        
        Args:
            default_criteria: Default criteria for critique
            auto_suggest: Whether to auto-generate suggestions
        """
        self.default_criteria = default_criteria or CritiqueCriteria()
        self.auto_suggest = auto_suggest
        self._critique_count = 0
        self._issue_patterns = self._compile_patterns()
        
        logger.info("Initialized Critic")
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for issue detection."""
        return {
            "vague_words": re.compile(r'\b(maybe|perhaps|somewhat|kind of|sort of)\b', re.I),
            "weak_modifiers": re.compile(r'\b(very|really|quite|rather|pretty)\s+\w+', re.I),
            "passive_voice": re.compile(r'\b(is|are|was|were|been|be|being)\s+\w+ed\b', re.I),
            "redundant_phrases": re.compile(r'\b(close proximity|end result|free gift)\b', re.I),
        }
    
    def critique(
        self,
        output: str,
        criteria: Optional[CritiqueCriteria] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CritiqueResult:
        """Generate comprehensive critique.
        
        Args:
            output: The content to critique
            criteria: Criteria to apply (uses default if not provided)
            context: Additional context for critique
            
        Returns:
            CritiqueResult with issues, suggestions, and score
        """
        criteria = criteria or self.default_criteria
        context = context or {}
        
        logger.info("Starting critique", extra={
            "correlation_id": context.get("correlation_id"),
            "output_length": len(output),
        })
        
        issues = []
        strengths = []
        
        # Run all enabled checks
        if criteria.check_accuracy:
            issues.extend(self._check_accuracy(output, context))
        if criteria.check_completeness:
            issues.extend(self._check_completeness(output, criteria))
        if criteria.check_clarity:
            issues.extend(self._check_clarity(output))
        if criteria.check_conciseness:
            issues.extend(self._check_conciseness(output))
        if criteria.check_consistency:
            issues.extend(self._check_consistency(output))
        
        # Check length constraints
        if criteria.min_length and len(output) < criteria.min_length:
            issues.append(Issue(
                type=IssueType.COMPLETENESS,
                severity=Severity.MAJOR,
                description=f"Output too short ({len(output)} < {criteria.min_length} chars)",
            ))
        
        if criteria.max_length and len(output) > criteria.max_length:
            issues.append(Issue(
                type=IssueType.CONCISENESS,
                severity=Severity.MINOR,
                description=f"Output too long ({len(output)} > {criteria.max_length} chars)",
            ))
        
        # Check required elements
        for element in criteria.required_elements:
            if element.lower() not in output.lower():
                issues.append(Issue(
                    type=IssueType.COMPLETENESS,
                    severity=Severity.MAJOR,
                    description=f"Missing required element: '{element}'",
                ))
        
        # Check forbidden patterns
        for pattern in criteria.forbidden_patterns:
            if pattern.lower() in output.lower():
                issues.append(Issue(
                    type=IssueType.STYLE,
                    severity=Severity.MINOR,
                    description=f"Contains forbidden pattern: '{pattern}'",
                ))
        
        # Identify strengths
        strengths = self._identify_strengths(output, issues)
        
        # Calculate score
        score = self._calculate_score(output, issues, strengths)
        
        # Generate suggestions
        suggestions = []
        if self.auto_suggest:
            suggestions = self.suggest_improvements(output, issues)
        
        self._critique_count += 1
        
        result = CritiqueResult(
            score=score,
            issues=issues,
            suggestions=suggestions,
            strengths=strengths,
            metadata={
                "critique_number": self._critique_count,
                "criteria_applied": {
                    "accuracy": criteria.check_accuracy,
                    "completeness": criteria.check_completeness,
                    "clarity": criteria.check_clarity,
                    "conciseness": criteria.check_conciseness,
                    "consistency": criteria.check_consistency,
                },
            },
        )
        
        logger.debug(f"Critique complete: score={score:.3f}, issues={len(issues)}")
        return result
    
    def identify_issues(
        self,
        output: str,
        issue_types: Optional[Set[IssueType]] = None,
    ) -> List[Issue]:
        """Identify specific issues in output.
        
        Args:
            output: Content to analyze
            issue_types: Specific types to check (all if None)
            
        Returns:
            List of identified issues
        """
        issues = []
        types_to_check = issue_types or set(IssueType)
        
        for issue_type in types_to_check:
            if issue_type == IssueType.CLARITY:
                issues.extend(self._check_clarity(output))
            elif issue_type == IssueType.CONCISENESS:
                issues.extend(self._check_conciseness(output))
            elif issue_type == IssueType.CORRECTNESS:
                issues.extend(self._check_accuracy(output, {}))
            elif issue_type == IssueType.GRAMMAR:
                issues.extend(self._check_grammar(output))
        
        return issues
    
    def suggest_improvements(
        self,
        output: str,
        issues: List[Issue],
    ) -> List[Suggestion]:
        """Generate improvement suggestions for issues.
        
        Args:
            output: The original content
            issues: Issues to address
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for issue in issues:
            suggestion = self._issue_to_suggestion(issue, output)
            if suggestion:
                suggestions.append(suggestion)
        
        # Sort by priority
        suggestions.sort(key=lambda s: s.priority)
        
        return suggestions
    
    def _check_accuracy(self, output: str, context: Dict[str, Any]) -> List[Issue]:
        """Check for accuracy issues."""
        issues = []
        
        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "XXX", "PLACEHOLDER", "[Generated]"]
        for placeholder in placeholders:
            if placeholder in output:
                issues.append(Issue(
                    type=IssueType.ACCURACY,
                    severity=Severity.CRITICAL,
                    description=f"Contains placeholder text: '{placeholder}'",
                ))
        
        return issues
    
    def _check_completeness(self, output: str, criteria: CritiqueCriteria) -> List[Issue]:
        """Check for completeness issues."""
        issues = []
        
        # Check for incomplete sentences
        if output.endswith(("...", "etc.", "and so on")):
            issues.append(Issue(
                type=IssueType.COMPLETENESS,
                severity=Severity.MAJOR,
                description="Content appears to be cut off or incomplete",
            ))
        
        # Check for questions without answers
        if "?" in output and output.count("?") > output.count("."):
            issues.append(Issue(
                type=IssueType.COMPLETENESS,
                severity=Severity.MINOR,
                description="Many questions but potentially few definitive statements",
            ))
        
        return issues
    
    def _check_clarity(self, output: str) -> List[Issue]:
        """Check for clarity issues."""
        issues = []
        
        # Check for vague words
        vague_matches = self._issue_patterns["vague_words"].findall(output)
        if vague_matches:
            issues.append(Issue(
                type=IssueType.CLARITY,
                severity=Severity.MINOR,
                description=f"Uses vague language: {set(vague_matches)}",
            ))
        
        # Check sentence length
        sentences = output.split(".")
        long_sentences = [s for s in sentences if len(s) > 200]
        if len(long_sentences) > len(sentences) * 0.3:
            issues.append(Issue(
                type=IssueType.CLARITY,
                severity=Severity.MAJOR,
                description=f"{len(long_sentences)} overly long sentences detected",
            ))
        
        return issues
    
    def _check_conciseness(self, output: str) -> List[Issue]:
        """Check for conciseness issues."""
        issues = []
        
        # Check for redundant phrases
        redundant = self._issue_patterns["redundant_phrases"].findall(output)
        if redundant:
            issues.append(Issue(
                type=IssueType.CONCISENESS,
                severity=Severity.MINOR,
                description=f"Redundant phrases: {set(redundant)}",
            ))
        
        # Check for weak modifiers
        weak = self._issue_patterns["weak_modifiers"].findall(output)
        if len(weak) > 3:
            issues.append(Issue(
                type=IssueType.CONCISENESS,
                severity=Severity.MINOR,
                description=f"Excessive use of weak modifiers: {len(weak)} instances",
            ))
        
        return issues
    
    def _check_consistency(self, output: str) -> List[Issue]:
        """Check for consistency issues."""
        issues = []
        
        # Check for tense consistency
        past_tense = len(re.findall(r'\b(was|were|had|did)\b', output))
        present_tense = len(re.findall(r'\b(is|are|has|does)\b', output))
        
        if past_tense > 0 and present_tense > 0:
            # This is a heuristic - mixed tenses might be intentional
            ratio = min(past_tense, present_tense) / max(past_tense, present_tense)
            if 0.3 < ratio < 0.7:  # Significant mixing
                issues.append(Issue(
                    type=IssueType.CONSISTENCY,
                    severity=Severity.MINOR,
                    description="Mixed verb tenses detected",
                ))
        
        return issues
    
    def _check_grammar(self, output: str) -> List[Issue]:
        """Check for grammar issues (basic)."""
        issues = []
        
        # Check for double spaces
        if "  " in output:
            issues.append(Issue(
                type=IssueType.GRAMMAR,
                severity=Severity.MINOR,
                description="Double spaces detected",
            ))
        
        return issues
    
    def _identify_strengths(self, output: str, issues: List[Issue]) -> List[str]:
        """Identify positive aspects of the output."""
        strengths = []
        
        # Length appropriate
        if 100 <= len(output) <= 5000:
            strengths.append("Appropriate length for detailed content")
        
        # Good structure
        paragraphs = output.split("\n\n")
        if len(paragraphs) >= 2:
            strengths.append("Well-structured with multiple paragraphs")
        
        # Specific examples
        if "example" in output.lower() or "for instance" in output.lower():
            strengths.append("Includes specific examples")
        
        # Few issues relative to length
        if len(issues) < len(output) / 500:
            strengths.append("High overall quality with minimal issues")
        
        return strengths
    
    def _calculate_score(self, output: str, issues: List[Issue], strengths: List[str]) -> float:
        """Calculate overall quality score."""
        base_score = 0.85
        
        # Deduct for issues
        for issue in issues:
            base_score -= issue.severity.weight()
        
        # Bonus for strengths
        base_score += len(strengths) * 0.02
        
        return max(0.0, min(1.0, base_score))
    
    def _issue_to_suggestion(self, issue: Issue, output: str) -> Optional[Suggestion]:
        """Convert an issue to a suggestion."""
        fix_templates = {
            IssueType.ACCURACY: "Replace placeholder text with actual content",
            IssueType.COMPLETENESS: "Expand section to provide full explanation",
            IssueType.CLARITY: "Rewrite using more specific, concrete language",
            IssueType.CONCISENESS: "Remove redundant words and phrases",
            IssueType.CORRECTNESS: "Verify facts and correct any errors",
            IssueType.CONSISTENCY: "Standardize tense and terminology throughout",
            IssueType.GRAMMAR: "Correct grammatical errors and punctuation",
            IssueType.STYLE: "Adjust tone and style to match requirements",
        }
        
        priority_map = {
            Severity.CRITICAL: 1,
            Severity.MAJOR: 3,
            Severity.MINOR: 5,
        }
        
        return Suggestion(
            issue=issue,
            fix=fix_templates.get(issue.type, "Review and improve this aspect"),
            priority=priority_map.get(issue.severity, 5),
            estimated_impact=issue.severity.weight() * 1.5,
            automated=issue.type in {IssueType.GRAMMAR, IssueType.CONCISENESS},
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get critic statistics."""
        return {
            "total_critiques": self._critique_count,
            "auto_suggest": self.auto_suggest,
        }


class CritiqueError(Exception):
    """Error during critique operation."""
    pass
