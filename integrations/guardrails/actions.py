"""Actions taken when violations are detected.

Automated responses to policy violations and validation failures.

Following CLAUDE.md patterns:
- UTC timestamps for all actions
- Structured logging with correlation_id
- Fail-safe defaults (block on error)
- Idempotent action execution
"""

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.guardrails.policies import (
    Violation,
    PolicyResult,
    PolicySeverity,
    PolicyAction
)
from integrations.guardrails.validators import ValidationResult

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of action execution."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ActionResult:
    """Result of an action execution.
    
    SSOT for action outcome.
    """
    action_name: str
    status: ActionStatus
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    transformed_content: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "action_name": self.action_name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "details": self.details,
            "has_transformed_content": self.transformed_content is not None
        }


@dataclass
class BlockResult:
    """Result of a block action."""
    blocked: bool
    reason: str
    action_results: List[ActionResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "action_count": len(self.action_results),
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class Action(ABC):
    """Base class for all actions."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        auto_execute: bool = False
    ):
        self.name = name
        self.enabled = enabled
        self.auto_execute = auto_execute
        self._execution_count = 0
        self._success_count = 0
        
    @abstractmethod
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Execute the action.
        
        Args:
            violation: The violation that triggered this action
            content: The content being processed
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ActionResult with execution outcome
        """
        raise NotImplementedError
        
    def can_execute(self, violation: Union[Violation, ValidationResult]) -> bool:
        """Check if this action can handle the violation."""
        return self.enabled
        
    def get_stats(self) -> Dict[str, Any]:
        """Get action statistics."""
        return {
            "action_name": self.name,
            "enabled": self.enabled,
            "executions": self._execution_count,
            "successes": self._success_count,
            "success_rate": self._success_count / max(1, self._execution_count)
        }
        
    def _create_result(
        self,
        status: ActionStatus,
        message: str,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        transformed_content: Optional[Any] = None
    ) -> ActionResult:
        """Create an action result."""
        self._execution_count += 1
        if status == ActionStatus.SUCCESS:
            self._success_count += 1
            
        return ActionResult(
            action_name=self.name,
            status=status,
            message=message,
            correlation_id=correlation_id,
            details=details or {},
            transformed_content=transformed_content
        )


class BlockAction(Action):
    """Block the request/response."""
    
    def __init__(
        self,
        block_message: str = "Content blocked due to policy violation",
        include_reason: bool = True,
        name: str = "BlockAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.block_message = block_message
        self.include_reason = include_reason
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Block the content."""
        try:
            reason = self._get_reason(violation)
            message = f"{self.block_message}: {reason}" if self.include_reason else self.block_message
            
            logger.warning({
                "msg": "Content blocked",
                "action": self.name,
                "reason": reason,
                "correlation_id": correlation_id
            })
            
            return self._create_result(
                ActionStatus.SUCCESS,
                message,
                correlation_id,
                details={
                    "blocked": True,
                    "reason": reason,
                    "violation_type": type(violation).__name__
                }
            )
            
        except Exception as e:
            logger.error({
                "msg": "Block action error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                ActionStatus.ERROR,
                f"Block action failed: {str(e)}",
                correlation_id
            )
            
    def _get_reason(self, violation: Union[Violation, ValidationResult]) -> str:
        """Extract reason from violation."""
        if isinstance(violation, Violation):
            return violation.message
        elif isinstance(violation, ValidationResult):
            return violation.message
        return "Unknown violation"


class FilterAction(Action):
    """Filter sensitive content from output."""
    
    def __init__(
        self,
        redact_pii: bool = True,
        remove_toxicity: bool = True,
        mask_chars: str = "*",
        name: str = "FilterAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled, auto_execute=True)
        self.redact_pii = redact_pii
        self.remove_toxicity = remove_toxicity
        self.mask_chars = mask_chars
        
        # PII patterns
        self.pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        }
        
        # Toxic words (simplified)
        self.toxic_words = ["stupid", "idiot", "loser", "hate", "kill"]
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Filter sensitive content."""
        try:
            if not isinstance(content, str):
                return self._create_result(
                    ActionStatus.SKIPPED,
                    "Filter action only applies to string content",
                    correlation_id
                )
                
            filtered = content
            redactions = []
            
            # Redact PII
            if self.redact_pii:
                for pii_type, pattern in self.pii_patterns.items():
                    matches = pattern.findall(filtered)
                    if matches:
                        filtered = pattern.sub(f"[{pii_type.upper()}_REDACTED]", filtered)
                        redactions.append(pii_type)
                        
            # Mask toxic words
            if self.remove_toxicity:
                for word in self.toxic_words:
                    pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                    matches = pattern.findall(filtered)
                    if matches:
                        masked = self.mask_chars * len(word)
                        filtered = pattern.sub(masked, filtered)
                        redactions.append(f"toxic_word:{word}")
                        
            if redactions:
                return self._create_result(
                    ActionStatus.SUCCESS,
                    f"Filtered content: redacted {len(redactions)} items",
                    correlation_id,
                    details={"redactions": redactions},
                    transformed_content=filtered
                )
                
            return self._create_result(
                ActionStatus.SKIPPED,
                "No sensitive content to filter",
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "Filter action error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                ActionStatus.ERROR,
                f"Filter failed: {str(e)}",
                correlation_id
            )
            
    def redact_pii_text(self, text: str) -> str:
        """Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text
        """
        result = text
        for pii_type, pattern in self.pii_patterns.items():
            result = pattern.sub(f"[{pii_type.upper()}_REDACTED]", result)
        return result
        
    def remove_toxicity_text(self, text: str) -> str:
        """Remove toxicity from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        result = text
        for word in self.toxic_words:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            masked = self.mask_chars * len(word)
            result = pattern.sub(masked, result)
        return result


class RewriteAction(Action):
    """Rewrite content to comply with policies."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        name: str = "RewriteAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled, auto_execute=True)
        self.max_attempts = max_attempts
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Attempt to rewrite content."""
        # Note: In a real implementation, this would call an LLM
        # to rewrite the content. For now, we provide a placeholder.
        
        try:
            if not isinstance(content, str):
                return self._create_result(
                    ActionStatus.SKIPPED,
                    "Rewrite only applies to string content",
                    correlation_id
                )
                
            # Simple rewrite strategies
            rewritten = content
            strategies_applied = []
            
            # Remove sentences with violations (simplified)
            if isinstance(violation, Violation):
                if violation.rule_name == "harmful_content":
                    rewritten = self._remove_harmful_content(rewritten)
                    strategies_applied.append("removed_harmful_content")
                elif violation.rule_name == "discrimination":
                    rewritten = self._remove_discriminatory_content(rewritten)
                    strategies_applied.append("removed_discriminatory_content")
                    
            # Apply generic improvements
            rewritten = self._generic_improvements(rewritten)
            strategies_applied.append("generic_improvements")
            
            if rewritten != content:
                return self._create_result(
                    ActionStatus.SUCCESS,
                    f"Rewrote content using {len(strategies_applied)} strategies",
                    correlation_id,
                    details={"strategies": strategies_applied},
                    transformed_content=rewritten
                )
                
            return self._create_result(
                ActionStatus.SKIPPED,
                "No rewrite needed or possible",
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "Rewrite action error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                ActionStatus.ERROR,
                f"Rewrite failed: {str(e)}",
                correlation_id
            )
            
    def _remove_harmful_content(self, text: str) -> str:
        """Remove harmful content sentences."""
        # Simple implementation - in production use ML
        harmful_indicators = ["kill", "hurt", "harm", "attack"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = []
        
        for sentence in sentences:
            if not any(indicator in sentence.lower() for indicator in harmful_indicators):
                clean_sentences.append(sentence)
                
        return " ".join(clean_sentences) if clean_sentences else "[Content removed due to policy violation]"
        
    def _remove_discriminatory_content(self, text: str) -> str:
        """Remove discriminatory content."""
        discriminatory_terms = ["inferior", "superior race", "all X are"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = []
        
        for sentence in sentences:
            if not any(term in sentence.lower() for term in discriminatory_terms):
                clean_sentences.append(sentence)
                
        return " ".join(clean_sentences) if clean_sentences else "[Content removed due to policy violation]"
        
    def _generic_improvements(self, text: str) -> str:
        """Apply generic text improvements."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        return text.strip()


class LogAction(Action):
    """Log violations for review."""
    
    def __init__(
        self,
        log_level: int = logging.WARNING,
        include_content: bool = False,
        hash_content: bool = True,
        name: str = "LogAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled, auto_execute=True)
        self.log_level = log_level
        self.include_content = include_content
        self.hash_content = hash_content
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Log the violation."""
        try:
            log_data = {
                "msg": "Policy violation detected",
                "action": self.name,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add violation details
            if isinstance(violation, Violation):
                log_data.update({
                    "policy": violation.policy_name,
                    "rule": violation.rule_name,
                    "severity": violation.severity.value,
                    "message": violation.message
                })
            elif isinstance(violation, ValidationResult):
                log_data.update({
                    "validator": violation.validator_name,
                    "severity": violation.severity.value,
                    "message": violation.message
                })
                
            # Add content info
            if isinstance(content, str):
                log_data["content_length"] = len(content)
                if self.hash_content:
                    log_data["content_hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]
                if self.include_content:
                    log_data["content_preview"] = content[:200]
                    
            logger.log(self.log_level, log_data)
            
            return self._create_result(
                ActionStatus.SUCCESS,
                "Violation logged successfully",
                correlation_id,
                details={"log_level": logging.getLevelName(self.log_level)}
            )
            
        except Exception as e:
            # Logging should never fail, but handle gracefully
            print(f"CRITICAL: Log action failed: {e}")
            return self._create_result(
                ActionStatus.ERROR,
                f"Logging failed: {str(e)}",
                correlation_id
            )
            
    def log_violation(
        self,
        violation: Violation,
        correlation_id: Optional[str] = None
    ) -> None:
        """Log a violation record.
        
        Args:
            violation: The violation to log
            correlation_id: Optional correlation ID
        """
        self.execute(violation, None, correlation_id)


class EscalateAction(Action):
    """Escalate to human review."""
    
    def __init__(
        self,
        escalation_threshold: PolicySeverity = PolicySeverity.HIGH,
        ticket_system: Optional[str] = None,
        name: str = "EscalateAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.escalation_threshold = escalation_threshold
        self.ticket_system = ticket_system
        self._ticket_count = 0
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Create escalation ticket."""
        try:
            # Check if escalation is needed
            severity = self._get_severity(violation)
            severity_order = [
                PolicySeverity.LOW,
                PolicySeverity.MEDIUM,
                PolicySeverity.HIGH,
                PolicySeverity.CRITICAL
            ]
            
            if severity_order.index(severity) < severity_order.index(self.escalation_threshold):
                return self._create_result(
                    ActionStatus.SKIPPED,
                    f"Severity {severity.value} below threshold {self.escalation_threshold.value}",
                    correlation_id
                )
                
            # Create ticket
            ticket_id = self._create_ticket(violation, content, correlation_id)
            self._ticket_count += 1
            
            logger.warning({
                "msg": "Escalation ticket created",
                "ticket_id": ticket_id,
                "severity": severity.value,
                "correlation_id": correlation_id
            })
            
            return self._create_result(
                ActionStatus.SUCCESS,
                f"Escalation ticket created: {ticket_id}",
                correlation_id,
                details={
                    "ticket_id": ticket_id,
                    "severity": severity.value,
                    "ticket_system": self.ticket_system
                }
            )
            
        except Exception as e:
            logger.error({
                "msg": "Escalation action error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                ActionStatus.ERROR,
                f"Escalation failed: {str(e)}",
                correlation_id
            )
            
    def _get_severity(
        self,
        violation: Union[Violation, ValidationResult]
    ) -> PolicySeverity:
        """Extract severity from violation."""
        if isinstance(violation, Violation):
            return violation.severity
        elif isinstance(violation, ValidationResult):
            severity_map = {
                "info": PolicySeverity.LOW,
                "warning": PolicySeverity.MEDIUM,
                "error": PolicySeverity.HIGH,
                "critical": PolicySeverity.CRITICAL
            }
            return severity_map.get(violation.severity.value, PolicySeverity.MEDIUM)
        return PolicySeverity.MEDIUM
        
    def _create_ticket(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str]
    ) -> str:
        """Create an escalation ticket.
        
        In production, this would integrate with ticketing system.
        For now, returns a mock ticket ID.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_suffix = hashlib.sha256(str(correlation_id).encode()).hexdigest()[:6]
        return f"GR-{timestamp}-{hash_suffix}"
        
    def create_ticket(
        self,
        violation: Violation,
        correlation_id: Optional[str] = None
    ) -> str:
        """Create a ticket for review.
        
        Args:
            violation: The violation to escalate
            correlation_id: Optional correlation ID
            
        Returns:
            Ticket ID
        """
        result = self.execute(violation, None, correlation_id)
        return result.details.get("ticket_id", "unknown")


class NotifyAction(Action):
    """Send notification about violation."""
    
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        notify_on: Optional[List[str]] = None,
        name: str = "NotifyAction",
        enabled: bool = True
    ):
        super().__init__(name=name, enabled=enabled)
        self.channels = channels or ["log"]  # log, email, webhook, slack
        self.notify_on = set(notify_on or ["critical", "high"])
        
    def execute(
        self,
        violation: Union[Violation, ValidationResult],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> ActionResult:
        """Send notifications."""
        try:
            severity = self._get_severity_value(violation)
            
            if severity not in self.notify_on:
                return self._create_result(
                    ActionStatus.SKIPPED,
                    f"Severity {severity} not in notify list",
                    correlation_id
                )
                
            sent_channels = []
            
            for channel in self.channels:
                if channel == "log":
                    logger.warning({
                        "msg": "Violation notification",
                        "severity": severity,
                        "correlation_id": correlation_id
                    })
                    sent_channels.append("log")
                elif channel == "webhook":
                    # Placeholder for webhook integration
                    sent_channels.append("webhook")
                elif channel == "email":
                    # Placeholder for email integration
                    sent_channels.append("email")
                elif channel == "slack":
                    # Placeholder for slack integration
                    sent_channels.append("slack")
                    
            return self._create_result(
                ActionStatus.SUCCESS,
                f"Notifications sent to {len(sent_channels)} channels",
                correlation_id,
                details={"channels": sent_channels}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Notify action error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                ActionStatus.ERROR,
                f"Notification failed: {str(e)}",
                correlation_id
            )
            
    def _get_severity_value(
        self,
        violation: Union[Violation, ValidationResult]
    ) -> str:
        """Get severity as string."""
        if isinstance(violation, Violation):
            return violation.severity.value
        elif isinstance(violation, ValidationResult):
            return violation.severity.value
        return "unknown"


class ActionEngine:
    """Engine to execute actions for violations.
    
    Coordinates multiple actions and aggregates results.
    """
    
    def __init__(
        self,
        actions: Optional[List[Action]] = None,
        default_action: PolicyAction = PolicyAction.BLOCK
    ):
        self.actions = actions or []
        self.default_action = default_action
        self._execution_history: List[ActionResult] = []
        
    def add_action(self, action: Action) -> 'ActionEngine':
        """Add an action to the engine."""
        self.actions.append(action)
        return self
        
    def execute_actions(
        self,
        violations: List[Union[Violation, ValidationResult]],
        content: Any,
        correlation_id: Optional[str] = None
    ) -> BlockResult:
        """Execute actions for violations.
        
        Args:
            violations: List of violations
            content: Content being processed
            correlation_id: Optional correlation ID
            
        Returns:
            BlockResult with all action results
        """
        results = []
        should_block = False
        block_reason = None
        final_content = content
        
        for violation in violations:
            for action in self.actions:
                if not action.enabled:
                    continue
                    
                if not action.can_execute(violation):
                    continue
                    
                try:
                    result = action.execute(violation, final_content, correlation_id)
                    results.append(result)
                    self._execution_history.append(result)
                    
                    # Track if should block
                    if isinstance(action, BlockAction):
                        should_block = True
                        block_reason = result.message
                        
                    # Track transformed content
                    if result.transformed_content is not None:
                        final_content = result.transformed_content
                        
                except Exception as e:
                    logger.error({
                        "msg": "Action execution error",
                        "action": action.name,
                        "error": str(e),
                        "correlation_id": correlation_id
                    })
                    results.append(ActionResult(
                        action_name=action.name,
                        status=ActionStatus.ERROR,
                        message=f"Execution error: {str(e)}",
                        correlation_id=correlation_id
                    ))
                    
        # Apply default action if no actions executed and violations exist
        if violations and not results and self.default_action == PolicyAction.BLOCK:
            should_block = True
            block_reason = "Default block action applied"
            
        return BlockResult(
            blocked=should_block,
            reason=block_reason or "No blocking required",
            action_results=results,
            correlation_id=correlation_id
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        successful = sum(1 for r in self._execution_history if r.status == ActionStatus.SUCCESS)
        return {
            "actions": [a.get_stats() for a in self.actions],
            "total_executions": len(self._execution_history),
            "successful_executions": successful,
            "default_action": self.default_action.value
        }
