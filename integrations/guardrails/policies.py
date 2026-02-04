"""Safety and compliance policies.

Define rules for acceptable AI behavior including safety checks,
regulatory compliance, and content guidelines.

Following CLAUDE.md patterns:
- UTC timestamps for all policy evaluations
- Structured logging with correlation_id
- Fail-safe defaults (block on policy violation)
- Configurable strictness levels
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class PolicySeverity(Enum):
    """Severity levels for policy violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(Enum):
    """Actions to take on policy violation."""
    ALLOW = "allow"  # Allow with logging
    WARN = "warn"    # Allow with warning
    BLOCK = "block"  # Block the content
    ESCALATE = "escalate"  # Escalate to human review


@dataclass
class Violation:
    """Represents a policy violation.
    
    SSOT for violation information.
    """
    policy_name: str
    rule_name: str
    message: str
    severity: PolicySeverity
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "policy_name": self.policy_name,
            "rule_name": self.rule_name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "details": self.details,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class PolicyResult:
    """Result of policy evaluation.
    
    SSOT for policy evaluation outcome.
    """
    allowed: bool
    violations: List[Violation] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_violations(self) -> bool:
        """Check if any violations exist."""
        return len(self.violations) > 0
        
    @property
    def highest_severity(self) -> Optional[PolicySeverity]:
        """Get highest severity violation."""
        if not self.violations:
            return None
        severity_order = [PolicySeverity.LOW, PolicySeverity.MEDIUM, PolicySeverity.HIGH, PolicySeverity.CRITICAL]
        highest = PolicySeverity.LOW
        for v in self.violations:
            if severity_order.index(v.severity) > severity_order.index(highest):
                highest = v.severity
        return highest
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "allowed": self.allowed,
            "violations": [v.to_dict() for v in self.violations],
            "violation_count": len(self.violations),
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


@dataclass
class Fix:
    """Suggested fix for a violation."""
    violation: Violation
    fix_type: str
    description: str
    auto_applicable: bool = False
    transformed_content: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fix to dictionary."""
        return {
            "violation": self.violation.to_dict(),
            "fix_type": self.fix_type,
            "description": self.description,
            "auto_applicable": self.auto_applicable
        }


class Policy(ABC):
    """Base class for all policies."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        default_action: PolicyAction = PolicyAction.BLOCK
    ):
        self.name = name
        self.enabled = enabled
        self.default_action = default_action
        self._evaluation_count = 0
        self._violation_count = 0
        
    @abstractmethod
    def evaluate(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Violation]:
        """Evaluate the policy against input/output.
        
        Args:
            input_data: User input
            output_data: LLM output
            context: Additional context
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            List of violations (empty if policy passes)
        """
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            "policy_name": self.name,
            "enabled": self.enabled,
            "evaluations": self._evaluation_count,
            "violations": self._violation_count,
            "violation_rate": self._violation_count / max(1, self._evaluation_count)
        }
        
    def _create_violation(
        self,
        rule_name: str,
        message: str,
        severity: PolicySeverity,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggested_fix: Optional[str] = None
    ) -> Violation:
        """Create a violation record."""
        self._violation_count += 1
        return Violation(
            policy_name=self.name,
            rule_name=rule_name,
            message=message,
            severity=severity,
            correlation_id=correlation_id,
            details=details or {},
            suggested_fix=suggested_fix
        )


class SafetyPolicy(Policy):
    """General safety policy for harmful content detection."""
    
    def __init__(
        self,
        harmful_content: bool = True,
        discrimination: bool = True,
        misinformation: bool = True,
        privacy_violation: bool = True,
        violence: bool = True,
        self_harm: bool = True,
        sexual_content: bool = True,
        enabled: bool = True,
        name: str = "SafetyPolicy"
    ):
        super().__init__(name=name, enabled=enabled)
        self.harmful_content = harmful_content
        self.discrimination = discrimination
        self.misinformation = misinformation
        self.privacy_violation = privacy_violation
        self.violence = violence
        self.self_harm = self_harm
        self.sexual_content = sexual_content
        
        # Detection patterns (simplified - in production use ML models)
        self._init_patterns()
        
    def _init_patterns(self):
        """Initialize detection patterns."""
        self.patterns = {
            "harmful_content": [
                re.compile(r'\b(kill|murder|attack|hurt|harm)\s+(someone|people|them|him|her)\b', re.IGNORECASE),
                re.compile(r'\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|poison)\b', re.IGNORECASE),
            ],
            "discrimination": [
                re.compile(r'\b(hate|inferior|superior)\s+(race|gender|religion|nationality)\b', re.IGNORECASE),
                re.compile(r'\b(all|every)\s+\w+\s+(are|is)\s+(stupid|bad|inferior)\b', re.IGNORECASE),
            ],
            "misinformation": [
                re.compile(r'\b(conspiracy|hoax|fake\s+news)\b', re.IGNORECASE),
            ],
            "privacy_violation": [
                re.compile(r'\b(ssn|social\s+security|credit\s+card)\s*:\s*\d', re.IGNORECASE),
                re.compile(r'\bpassword\s*[=:]\s*\S+', re.IGNORECASE),
            ],
            "violence": [
                re.compile(r'\b(attack|fight|violence|assault)\b', re.IGNORECASE),
            ],
            "self_harm": [
                re.compile(r'\b(suicide|self.?harm|hurt\s+myself|end\s+my\s+life)\b', re.IGNORECASE),
            ],
            "sexual_content": [
                re.compile(r'\b(sexual|explicit|adult\s+content|nsfw)\b', re.IGNORECASE),
            ],
        }
        
    def evaluate(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Violation]:
        """Evaluate safety policy."""
        if not self.enabled:
            return []
            
        self._evaluation_count += 1
        violations = []
        
        text_to_check = ""
        if isinstance(input_data, str):
            text_to_check += input_data + " "
        if isinstance(output_data, str):
            text_to_check += output_data
            
        if not text_to_check.strip():
            return violations
            
        # Check each category
        checks = [
            ("harmful_content", self.harmful_content, PolicySeverity.CRITICAL),
            ("discrimination", self.discrimination, PolicySeverity.HIGH),
            ("misinformation", self.misinformation, PolicySeverity.MEDIUM),
            ("privacy_violation", self.privacy_violation, PolicySeverity.CRITICAL),
            ("violence", self.violence, PolicySeverity.HIGH),
            ("self_harm", self.self_harm, PolicySeverity.CRITICAL),
            ("sexual_content", self.sexual_content, PolicySeverity.MEDIUM),
        ]
        
        for category, enabled, severity in checks:
            if not enabled:
                continue
                
            patterns = self.patterns.get(category, [])
            for pattern in patterns:
                matches = pattern.findall(text_to_check)
                if matches:
                    violations.append(self._create_violation(
                        rule_name=category,
                        message=f"Potential {category.replace('_', ' ')} detected",
                        severity=severity,
                        correlation_id=correlation_id,
                        details={
                            "category": category,
                            "matches": matches[:5],  # Limit matches logged
                            "pattern": pattern.pattern[:50]
                        },
                        suggested_fix=f"Review and remove {category.replace('_', ' ')} content"
                    ))
                    break  # One violation per category is enough
                    
        return violations


class CompliancePolicy(Policy):
    """Regulatory compliance policy (GDPR, HIPAA, etc.)."""
    
    def __init__(
        self,
        gdpr: bool = True,
        hipaa: bool = False,
        pci_dss: bool = False,
        sox: bool = False,
        ccpa: bool = False,
        enabled: bool = True,
        name: str = "CompliancePolicy"
    ):
        super().__init__(name=name, enabled=enabled)
        self.gdpr = gdpr
        self.hipaa = hipaa
        self.pci_dss = pci_dss
        self.sox = sox
        self.ccpa = ccpa
        
    def evaluate(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Violation]:
        """Evaluate compliance policy."""
        if not self.enabled:
            return []
            
        self._evaluation_count += 1
        violations = []
        
        # GDPR checks
        if self.gdpr:
            violations.extend(self._check_gdpr(input_data, output_data, correlation_id))
            
        # HIPAA checks
        if self.hipaa:
            violations.extend(self._check_hipaa(input_data, output_data, correlation_id))
            
        # PCI DSS checks
        if self.pci_dss:
            violations.extend(self._check_pci_dss(input_data, output_data, correlation_id))
            
        return violations
        
    def _check_gdpr(
        self,
        input_data: Any,
        output_data: Any,
        correlation_id: Optional[str]
    ) -> List[Violation]:
        """Check GDPR compliance."""
        violations = []
        
        # Check for PII without consent indication
        text = ""
        if isinstance(input_data, str):
            text += input_data
        if isinstance(output_data, str):
            text += output_data
            
        # Email pattern
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        if email_pattern.search(text):
            violations.append(self._create_violation(
                rule_name="gdpr_pii",
                message="Potential GDPR violation: Personal email detected without consent verification",
                severity=PolicySeverity.HIGH,
                correlation_id=correlation_id,
                suggested_fix="Verify user consent for processing personal data"
            ))
            
        return violations
        
    def _check_hipaa(
        self,
        input_data: Any,
        output_data: Any,
        correlation_id: Optional[str]
    ) -> List[Violation]:
        """Check HIPAA compliance."""
        violations = []
        
        # Check for PHI (Protected Health Information)
        text = ""
        if isinstance(input_data, str):
            text += input_data
        if isinstance(output_data, str):
            text += output_data
            
        # Medical record number pattern
        mrn_pattern = re.compile(r'\bMRN\s*[:#]?\s*\d+\b', re.IGNORECASE)
        if mrn_pattern.search(text):
            violations.append(self._create_violation(
                rule_name="hipaa_phi",
                message="Potential HIPAA violation: Medical record number detected",
                severity=PolicySeverity.CRITICAL,
                correlation_id=correlation_id,
                suggested_fix="Remove all PHI before processing"
            ))
            
        return violations
        
    def _check_pci_dss(
        self,
        input_data: Any,
        output_data: Any,
        correlation_id: Optional[str]
    ) -> List[Violation]:
        """Check PCI DSS compliance."""
        violations = []
        
        text = ""
        if isinstance(input_data, str):
            text += input_data
        if isinstance(output_data, str):
            text += output_data
            
        # Credit card pattern (simplified Luhn check)
        cc_pattern = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        if cc_pattern.search(text):
            violations.append(self._create_violation(
                rule_name="pci_dss_card",
                message="PCI DSS violation: Credit card number detected",
                severity=PolicySeverity.CRITICAL,
                correlation_id=correlation_id,
                suggested_fix="Remove payment card data immediately"
            ))
            
        return violations


class ContentPolicy(Policy):
    """Content guidelines policy."""
    
    def __init__(
        self,
        allowed_topics: Optional[List[str]] = None,
        blocked_topics: Optional[List[str]] = None,
        tone_requirements: Optional[Dict[str, Any]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        require_citations: bool = False,
        enabled: bool = True,
        name: str = "ContentPolicy"
    ):
        super().__init__(name=name, enabled=enabled)
        self.allowed_topics = set(allowed_topics or [])
        self.blocked_topics = set(blocked_topics or [])
        self.tone_requirements = tone_requirements or {}
        self.min_length = min_length
        self.max_length = max_length
        self.require_citations = require_citations
        
    def evaluate(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Violation]:
        """Evaluate content policy."""
        if not self.enabled:
            return []
            
        self._evaluation_count += 1
        violations = []
        
        if not isinstance(output_data, str):
            return violations
            
        text = output_data
        
        # Check blocked topics
        for topic in self.blocked_topics:
            if topic.lower() in text.lower():
                violations.append(self._create_violation(
                    rule_name="blocked_topic",
                    message=f"Blocked topic detected: {topic}",
                    severity=PolicySeverity.HIGH,
                    correlation_id=correlation_id,
                    details={"topic": topic},
                    suggested_fix=f"Remove content about {topic}"
                ))
                
        # Check allowed topics (if specified)
        if self.allowed_topics:
            has_allowed = any(topic.lower() in text.lower() for topic in self.allowed_topics)
            if not has_allowed:
                violations.append(self._create_violation(
                    rule_name="no_allowed_topic",
                    message="Content does not address any allowed topics",
                    severity=PolicySeverity.MEDIUM,
                    correlation_id=correlation_id,
                    details={"allowed_topics": list(self.allowed_topics)}
                ))
                
        # Check length
        if self.min_length and len(text) < self.min_length:
            violations.append(self._create_violation(
                rule_name="min_length",
                message=f"Content too short: {len(text)} chars (min {self.min_length})",
                severity=PolicySeverity.LOW,
                correlation_id=correlation_id
            ))
            
        if self.max_length and len(text) > self.max_length:
            violations.append(self._create_violation(
                rule_name="max_length",
                message=f"Content too long: {len(text)} chars (max {self.max_length})",
                severity=PolicySeverity.LOW,
                correlation_id=correlation_id
            ))
            
        # Check citations
        if self.require_citations:
            citation_pattern = re.compile(r'\[\d+\]|\(\w+\s+et\s+al\.?|\b(source|reference|citation)\b', re.IGNORECASE)
            if not citation_pattern.search(text):
                violations.append(self._create_violation(
                    rule_name="no_citations",
                    message="Content requires citations but none found",
                    severity=PolicySeverity.MEDIUM,
                    correlation_id=correlation_id,
                    suggested_fix="Add proper citations to claims"
                ))
                
        return violations


class PolicyEngine:
    """Engine to evaluate multiple policies.
    
    Coordinates policy evaluation and aggregates results.
    """
    
    def __init__(
        self,
        policies: Optional[List[Policy]] = None,
        strict_mode: bool = False,
        action_on_violation: PolicyAction = PolicyAction.BLOCK
    ):
        self.policies = policies or []
        self.strict_mode = strict_mode
        self.action_on_violation = action_on_violation
        self._evaluation_history: List[PolicyResult] = []
        
    def add_policy(self, policy: Policy) -> 'PolicyEngine':
        """Add a policy to the engine."""
        self.policies.append(policy)
        return self
        
    def evaluate(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> PolicyResult:
        """Evaluate all policies.
        
        Args:
            input_data: User input
            output_data: LLM output
            context: Additional context
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            PolicyResult with all violations
        """
        all_violations = []
        
        for policy in self.policies:
            if not policy.enabled:
                continue
                
            try:
                violations = policy.evaluate(
                    input_data,
                    output_data,
                    context,
                    correlation_id
                )
                all_violations.extend(violations)
                
            except Exception as e:
                logger.error({
                    "msg": "Policy evaluation error",
                    "policy": policy.name,
                    "error": str(e),
                    "correlation_id": correlation_id
                })
                if self.strict_mode:
                    all_violations.append(Violation(
                        policy_name=policy.name,
                        rule_name="evaluation_error",
                        message=f"Policy evaluation failed: {str(e)}",
                        severity=PolicySeverity.CRITICAL,
                        correlation_id=correlation_id
                    ))
                    
        # Determine if allowed
        critical_violations = [v for v in all_violations if v.severity == PolicySeverity.CRITICAL]
        allowed = len(critical_violations) == 0
        
        if self.action_on_violation == PolicyAction.BLOCK and all_violations:
            allowed = False
            
        result = PolicyResult(
            allowed=allowed,
            violations=all_violations,
            correlation_id=correlation_id,
            metadata={
                "policies_evaluated": len([p for p in self.policies if p.enabled]),
                "strict_mode": self.strict_mode
            }
        )
        
        self._evaluation_history.append(result)
        
        logger.info({
            "msg": "Policy evaluation complete",
            "allowed": allowed,
            "violations": len(all_violations),
            "correlation_id": correlation_id
        })
        
        return result
        
    def get_violations(
        self,
        severity: Optional[PolicySeverity] = None
    ) -> List[Violation]:
        """Get violations from evaluation history.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of violations
        """
        violations = []
        for result in self._evaluation_history:
            for v in result.violations:
                if severity is None or v.severity == severity:
                    violations.append(v)
        return violations
        
    def suggest_fixes(self, result: PolicyResult) -> List[Fix]:
        """Suggest fixes for violations.
        
        Args:
            result: Policy result with violations
            
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        for violation in result.violations:
            fix = Fix(
                violation=violation,
                fix_type="manual_review",
                description=violation.suggested_fix or "Manual review required",
                auto_applicable=False
            )
            fixes.append(fix)
            
        return fixes
        
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_violations = sum(len(r.violations) for r in self._evaluation_history)
        return {
            "policies": [p.get_stats() for p in self.policies],
            "total_evaluations": len(self._evaluation_history),
            "total_violations": total_violations,
            "strict_mode": self.strict_mode,
            "action_on_violation": self.action_on_violation.value
        }
