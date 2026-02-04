"""Guardrails Engine - Main orchestration.

Coordinates validation, rails, policies, and actions for comprehensive
AI safety and output quality assurance.

Following CLAUDE.md patterns:
- UTC timestamps for all engine operations
- Structured logging with correlation_id
- SSOT pattern for state management
- Fail-safe defaults (block on error)
- Circuit breaker pattern for external checks
- Configurable strictness levels
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.guardrails.validators import (
    Validator,
    ValidationResult,
    ValidationSeverity,
    TypeValidator,
    LengthValidator,
    JSONValidator,
    PIIValidator,
    ToxicityValidator
)
from integrations.guardrails.rails import (
    InputRail,
    OutputRail,
    RailSet,
    RailStatus,
    ProcessedInput,
    ProcessedOutput,
    SanitizationRail,
    JailbreakDetectionRail,
    ContextWindowRail,
    ValidationRail,
    LoggingRail
)
from integrations.guardrails.policies import (
    Policy,
    PolicyEngine,
    PolicyResult,
    PolicyAction,
    PolicySeverity,
    SafetyPolicy,
    CompliancePolicy,
    ContentPolicy,
    Violation
)
from integrations.guardrails.actions import (
    Action,
    ActionEngine,
    BlockAction,
    FilterAction,
    RewriteAction,
    LogAction,
    EscalateAction,
    BlockResult,
    ActionStatus
)

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety strictness levels."""
    STRICT = "strict"      # Block on any violation
    MODERATE = "moderate"  # Fix if possible, block if not
    PERMISSIVE = "permissive"  # Log only, don't block


@dataclass
class SafetyResult:
    """Result of safety check.
    
    SSOT for safety check outcome.
    """
    safe: bool
    violations: List[Violation] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "safe": self.safe,
            "violation_count": len(self.violations),
            "actions_taken": self.actions_taken,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


@dataclass
class EngineConfig:
    """Configuration for GuardrailsEngine.
    
    SSOT for engine configuration.
    """
    safety_level: SafetyLevel = SafetyLevel.MODERATE
    strict_mode: bool = False
    auto_fix: bool = True
    log_violations: bool = True
    escalation_threshold: float = 0.8
    max_validation_time: float = 30.0
    fail_on_error: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "safety_level": self.safety_level.value,
            "strict_mode": self.strict_mode,
            "auto_fix": self.auto_fix,
            "log_violations": self.log_violations,
            "escalation_threshold": self.escalation_threshold,
            "max_validation_time": self.max_validation_time,
            "fail_on_error": self.fail_on_error
        }


class GuardrailsEngine:
    """Main orchestrator for guardrails system.
    
    Coordinates validators, rails, policies, and actions to provide
    comprehensive AI safety and output quality assurance.
    
    Example:
        >>> engine = GuardrailsEngine(safety_level=SafetyLevel.STRICT)
        >>> engine.add_validator(PIIValidator())
        >>> result = engine.process_output("Hello world")
    """
    
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        safety_level: Optional[SafetyLevel] = None,
        validators: Optional[List[Validator]] = None,
        input_rails: Optional[List[InputRail]] = None,
        output_rails: Optional[List[OutputRail]] = None,
        policies: Optional[List[Policy]] = None,
        actions: Optional[List[Action]] = None
    ):
        """Initialize the guardrails engine.
        
        Args:
            config: Engine configuration
            safety_level: Override safety level from config
            validators: Custom validators
            input_rails: Custom input rails
            output_rails: Custom output rails
            policies: Custom policies
            actions: Custom actions
        """
        self.config = config or EngineConfig()
        if safety_level:
            self.config.safety_level = safety_level
            
        # Initialize components
        self.validators: List[Validator] = validators or self._default_validators()
        self.input_rails: List[InputRail] = input_rails or self._default_input_rails()
        self.output_rails: List[OutputRail] = output_rails or self._default_output_rails()
        self.policies: List[Policy] = policies or self._default_policies()
        self.actions: List[Action] = actions or self._default_actions()
        
        # Initialize engines
        self.rail_set = RailSet(
            input_rails=self.input_rails,
            output_rails=self.output_rails,
            fail_fast=True
        )
        self.policy_engine = PolicyEngine(
            policies=self.policies,
            strict_mode=self.config.strict_mode,
            action_on_violation=self._get_policy_action()
        )
        self.action_engine = ActionEngine(
            actions=self.actions,
            default_action=self._get_policy_action()
        )
        
        # Statistics
        self._stats = {
            "inputs_processed": 0,
            "outputs_processed": 0,
            "violations_detected": 0,
            "contents_blocked": 0,
            "contents_fixed": 0
        }
        
        logger.info({
            "msg": "GuardrailsEngine initialized",
            "safety_level": self.config.safety_level.value,
            "validators": len(self.validators),
            "policies": len(self.policies),
            "actions": len(self.actions)
        })
        
    def _default_validators(self) -> List[Validator]:
        """Create default set of validators based on safety level."""
        validators = [
            TypeValidator(str, allow_none=False, name="OutputTypeValidator"),
            LengthValidator(min_length=1, max_length=10000, name="OutputLengthValidator"),
            JSONValidator(allow_partial=True, name="OutputJSONValidator"),
        ]
        
        # Add safety validators for moderate and strict levels
        if self.config.safety_level in (SafetyLevel.MODERATE, SafetyLevel.STRICT):
            validators.extend([
                PIIValidator(detect_types=["email", "phone", "ssn"], block_on_detection=True),
                ToxicityValidator(sensitivity="medium", name="OutputToxicityValidator"),
            ])
            
        # Add stricter validators for strict level
        if self.config.safety_level == SafetyLevel.STRICT:
            validators.extend([
                PIIValidator(block_on_detection=True, name="StrictPIIValidator"),
                ToxicityValidator(sensitivity="high", name="StrictToxicityValidator"),
            ])
            
        return validators
        
    def _default_input_rails(self) -> List[InputRail]:
        """Create default input rails."""
        rails = [
            SanitizationRail(max_length=10000),
            ContextWindowRail(max_tokens=4096, reserve_tokens=500),
        ]
        
        # Add jailbreak detection for strict mode
        if self.config.safety_level in (SafetyLevel.MODERATE, SafetyLevel.STRICT):
            rails.append(JailbreakDetectionRail(block_on_detection=True))
            
        return rails
        
    def _default_output_rails(self) -> List[OutputRail]:
        """Create default output rails."""
        rails = [
            ValidationRail(
                validators=self.validators,
                auto_fix=self.config.auto_fix,
                allow_partial_fix=True
            ),
        ]
        
        # Add logging rail
        if self.config.log_violations:
            rails.append(LoggingRail(hash_output=True))
            
        return rails
        
    def _default_policies(self) -> List[Policy]:
        """Create default policies based on safety level."""
        policies = [
            SafetyPolicy(
                harmful_content=True,
                discrimination=True,
                misinformation=True,
                privacy_violation=True,
                enabled=self.config.safety_level != SafetyLevel.PERMISSIVE
            )
        ]
        
        if self.config.safety_level == SafetyLevel.STRICT:
            policies.extend([
                CompliancePolicy(gdpr=True, hipaa=False, pci_dss=True),
                ContentPolicy(
                    blocked_topics=["violence", "hate_speech"],
                    require_citations=False
                )
            ])
            
        return policies
        
    def _default_actions(self) -> List[Action]:
        """Create default actions based on safety level."""
        actions = [LogAction(enabled=self.config.log_violations)]
        
        if self.config.safety_level == SafetyLevel.MODERATE:
            actions.extend([
                FilterAction(redact_pii=True, remove_toxicity=True),
                RewriteAction(max_attempts=2),
            ])
            
        if self.config.safety_level == SafetyLevel.STRICT:
            actions.extend([
                BlockAction(),
                FilterAction(redact_pii=True, remove_toxicity=True),
                EscalateAction(escalation_threshold=PolicySeverity.HIGH),
            ])
            
        return actions
        
    def _get_policy_action(self) -> PolicyAction:
        """Map safety level to policy action."""
        mapping = {
            SafetyLevel.STRICT: PolicyAction.BLOCK,
            SafetyLevel.MODERATE: PolicyAction.WARN,
            SafetyLevel.PERMISSIVE: PolicyAction.ALLOW
        }
        return mapping.get(self.config.safety_level, PolicyAction.BLOCK)
        
    def add_validator(self, validator: Validator) -> 'GuardrailsEngine':
        """Add a validator to the engine.
        
        Args:
            validator: Validator to add
            
        Returns:
            Self for chaining
        """
        self.validators.append(validator)
        return self
        
    def add_input_rail(self, rail: InputRail) -> 'GuardrailsEngine':
        """Add an input rail.
        
        Args:
            rail: Input rail to add
            
        Returns:
            Self for chaining
        """
        self.input_rails.append(rail)
        self.rail_set = RailSet(
            input_rails=self.input_rails,
            output_rails=self.output_rails,
            fail_fast=True
        )
        return self
        
    def add_output_rail(self, rail: OutputRail) -> 'GuardrailsEngine':
        """Add an output rail.
        
        Args:
            rail: Output rail to add
            
        Returns:
            Self for chaining
        """
        self.output_rails.append(rail)
        self.rail_set = RailSet(
            input_rails=self.input_rails,
            output_rails=self.output_rails,
            fail_fast=True
        )
        return self
        
    def add_policy(self, policy: Policy) -> 'GuardrailsEngine':
        """Add a policy.
        
        Args:
            policy: Policy to add
            
        Returns:
            Self for chaining
        """
        self.policies.append(policy)
        self.policy_engine.add_policy(policy)
        return self
        
    def add_action(self, action: Action) -> 'GuardrailsEngine':
        """Add an action.
        
        Args:
            action: Action to add
            
        Returns:
            Self for chaining
        """
        self.actions.append(action)
        self.action_engine.add_action(action)
        return self
        
    def process_input(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedInput:
        """Process input through all input rails.
        
        Args:
            input_data: Raw user input
            context: Additional context
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ProcessedInput with sanitization and validation results
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        context = context or {}
        
        logger.info({
            "msg": "Processing input",
            "correlation_id": correlation_id,
            "input_length": len(input_data) if isinstance(input_data, str) else 0
        })
        
        try:
            result = self.rail_set.process_input(input_data, correlation_id)
            self._stats["inputs_processed"] += 1
            
            if result.blocked:
                self._stats["contents_blocked"] += 1
                logger.warning({
                    "msg": "Input blocked",
                    "reason": result.block_reason,
                    "correlation_id": correlation_id
                })
                
            return result
            
        except Exception as e:
            logger.error({
                "msg": "Input processing error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.config.fail_on_error:
                return ProcessedInput(
                    original_input=input_data,
                    processed_input=input_data,
                    status=RailStatus.ERROR,
                    blocked=True,
                    block_reason=f"Processing error: {str(e)}",
                    correlation_id=correlation_id
                )
            raise
            
    def process_output(
        self,
        output: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Process output through all output rails.
        
        Args:
            output: Raw LLM output
            context: Additional context
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ProcessedOutput with validation and transformation results
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        context = context or {}
        
        logger.info({
            "msg": "Processing output",
            "correlation_id": correlation_id,
            "output_type": type(output).__name__
        })
        
        try:
            result = self.rail_set.process_output(output, self.validators, correlation_id)
            self._stats["outputs_processed"] += 1
            
            if result.blocked:
                self._stats["contents_blocked"] += 1
            elif result.fixed:
                self._stats["contents_fixed"] += 1
                
            return result
            
        except Exception as e:
            logger.error({
                "msg": "Output processing error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.config.fail_on_error:
                return ProcessedOutput(
                    original_output=output,
                    processed_output=output,
                    status=RailStatus.ERROR,
                    blocked=True,
                    block_reason=f"Processing error: {str(e)}",
                    correlation_id=correlation_id
                )
            raise
            
    def validate(
        self,
        output: Any,
        schema: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ValidationResult:
        """Validate output against validators and optional schema.
        
        Args:
            output: Output to validate
            schema: Optional JSON schema
            correlation_id: Optional correlation ID
            
        Returns:
            ValidationResult
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Run all validators
        results = []
        for validator in self.validators:
            result = validator.validate(output, correlation_id)
            results.append(result)
            
        # Aggregate results
        failures = [r for r in results if not r.is_valid]
        
        if failures:
            # Return first critical failure or first failure
            critical = [r for r in failures if r.severity == ValidationSeverity.CRITICAL]
            return critical[0] if critical else failures[0]
            
        # All passed
        return ValidationResult(
            is_valid=True,
            validator_name="GuardrailsEngine",
            message="All validations passed",
            severity=ValidationSeverity.INFO,
            correlation_id=correlation_id
        )
        
    def check_safety(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> SafetyResult:
        """Check safety of input/output pair.
        
        Args:
            input_data: User input
            output_data: LLM output
            context: Additional context
            correlation_id: Optional correlation ID
            
        Returns:
            SafetyResult with safety assessment
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Run policy evaluation
        policy_result = self.policy_engine.evaluate(
            input_data,
            output_data,
            context,
            correlation_id
        )
        
        self._stats["violations_detected"] += len(policy_result.violations)
        
        # Determine if safe based on safety level
        is_safe = policy_result.allowed
        
        if self.config.safety_level == SafetyLevel.PERMISSIVE:
            is_safe = True  # Always safe in permissive mode
        elif self.config.safety_level == SafetyLevel.STRICT:
            is_safe = len(policy_result.violations) == 0  # Must have zero violations
            
        # Execute actions for violations
        actions_taken = []
        if policy_result.violations and self.config.safety_level != SafetyLevel.PERMISSIVE:
            block_result = self.action_engine.execute_actions(
                policy_result.violations,
                output_data,
                correlation_id
            )
            actions_taken = [r.action_name for r in block_result.action_results]
            
        return SafetyResult(
            safe=is_safe,
            violations=policy_result.violations,
            actions_taken=actions_taken,
            correlation_id=correlation_id
        )
        
    def enforce_policies(
        self,
        input_data: Any,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> PolicyResult:
        """Evaluate and enforce all policies.
        
        Args:
            input_data: User input
            output_data: LLM output
            context: Additional context
            correlation_id: Optional correlation ID
            
        Returns:
            PolicyResult with violations and enforcement status
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        
        result = self.policy_engine.evaluate(
            input_data,
            output_data,
            context,
            correlation_id
        )
        
        self._stats["violations_detected"] += len(result.violations)
        
        # Execute actions
        if result.violations:
            self.action_engine.execute_actions(
                result.violations,
                output_data,
                correlation_id
            )
            
        return result
        
    def guard(
        self,
        input_data: str,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete guard flow - process input, output, and check safety.
        
        Args:
            input_data: User input
            output_data: LLM output
            context: Additional context
            correlation_id: Optional correlation ID
            
        Returns:
            Complete guard result with all checks
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        context = context or {}
        
        logger.info({
            "msg": "Running complete guard flow",
            "correlation_id": correlation_id,
            "safety_level": self.config.safety_level.value
        })
        
        # Process input
        processed_input = self.process_input(input_data, context, correlation_id)
        
        # Process output
        processed_output = self.process_output(output_data, context, correlation_id)
        
        # Check safety
        safety_result = self.check_safety(
            processed_input.processed_input,
            processed_output.processed_output,
            context,
            correlation_id
        )
        
        # Determine final result
        blocked = (
            processed_input.blocked or
            processed_output.blocked or
            not safety_result.safe
        )
        
        result = {
            "blocked": blocked,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "original": input_data[:100] if isinstance(input_data, str) else str(input_data)[:100],
                "processed": processed_input.processed_input[:100] if isinstance(processed_input.processed_input, str) else str(processed_input.processed_input)[:100],
                "blocked": processed_input.blocked,
                "block_reason": processed_input.block_reason
            },
            "output": {
                "original": str(output_data)[:100],
                "processed": str(processed_output.processed_output)[:100],
                "blocked": processed_output.blocked,
                "block_reason": processed_output.block_reason,
                "fixed": processed_output.fixed
            },
            "safety": safety_result.to_dict(),
            "config": self.config.to_dict()
        }
        
        logger.info({
            "msg": "Guard flow complete",
            "blocked": blocked,
            "correlation_id": correlation_id
        })
        
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "config": self.config.to_dict(),
            "statistics": self._stats.copy(),
            "components": {
                "validators": len(self.validators),
                "input_rails": len(self.input_rails),
                "output_rails": len(self.output_rails),
                "policies": len(self.policies),
                "actions": len(self.actions)
            },
            "policy_engine": self.policy_engine.get_stats(),
            "action_engine": self.action_engine.get_stats()
        }
        
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "inputs_processed": 0,
            "outputs_processed": 0,
            "violations_detected": 0,
            "contents_blocked": 0,
            "contents_fixed": 0
        }


def create_strict_engine(
    log_violations: bool = True,
    escalation: bool = True
) -> GuardrailsEngine:
    """Create a strict guardrails engine.
    
    Args:
        log_violations: Whether to log violations
        escalation: Whether to enable escalation
        
    Returns:
        Configured GuardrailsEngine
    """
    config = EngineConfig(
        safety_level=SafetyLevel.STRICT,
        strict_mode=True,
        auto_fix=False,
        log_violations=log_violations
    )
    
    engine = GuardrailsEngine(config=config)
    
    if escalation:
        engine.add_action(EscalateAction(escalation_threshold=PolicySeverity.MEDIUM))
        
    return engine


def create_moderate_engine(
    log_violations: bool = True,
    auto_fix: bool = True
) -> GuardrailsEngine:
    """Create a moderate guardrails engine.
    
    Args:
        log_violations: Whether to log violations
        auto_fix: Whether to auto-fix violations
        
    Returns:
        Configured GuardrailsEngine
    """
    config = EngineConfig(
        safety_level=SafetyLevel.MODERATE,
        strict_mode=False,
        auto_fix=auto_fix,
        log_violations=log_violations
    )
    
    return GuardrailsEngine(config=config)


def create_permissive_engine(
    log_violations: bool = True
) -> GuardrailsEngine:
    """Create a permissive guardrails engine.
    
    Args:
        log_violations: Whether to log violations
        
    Returns:
        Configured GuardrailsEngine
    """
    config = EngineConfig(
        safety_level=SafetyLevel.PERMISSIVE,
        strict_mode=False,
        auto_fix=True,
        log_violations=log_violations
    )
    
    return GuardrailsEngine(config=config)
