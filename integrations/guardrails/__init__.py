"""Guardrails AI Integration for OpenEvolve.

Provides comprehensive AI safety and output validation including:
- Validators: Type, length, regex, PII, toxicity, JSON, schema, quality checks
- Rails: Pre/post processing pipelines for inputs and outputs
- Policies: Safety, compliance, and content guidelines
- Actions: Block, filter, rewrite, log, escalate responses to violations
- Engine: Main orchestrator coordinating all components

Following CLAUDE.md patterns:
- UTC timestamps for all operations
- Structured logging with correlation_id
- SSOT pattern for state management
- Fail-safe defaults
- Configurable strictness levels

Example:
    >>> from integrations.guardrails import GuardrailsEngine, SafetyLevel
    >>> engine = GuardrailsEngine(safety_level=SafetyLevel.STRICT)
    >>> result = engine.guard("user input", "llm output")
    >>> if result["blocked"]:
    ...     print("Content blocked for safety")
"""

# Validators
from integrations.guardrails.validators import (
    Validator,
    ValidationResult,
    ValidationSeverity,
    TypeValidator,
    RegexValidator,
    LengthValidator,
    RangeValidator,
    EnumValidator,
    PIIValidator,
    ToxicityValidator,
    JSONValidator,
    SchemaValidator,
    QualityValidator,
    CompositeValidator,
    CustomValidator,
)

# Rails
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
    KeywordFilterRail,
    ValidationRail,
    OutputSanitizationRail,
    LoggingRail,
)

# Policies
from integrations.guardrails.policies import (
    Policy,
    PolicyEngine,
    PolicyResult,
    PolicyAction,
    PolicySeverity,
    SafetyPolicy,
    CompliancePolicy,
    ContentPolicy,
    Violation,
    Fix,
)

# Actions
from integrations.guardrails.actions import (
    Action,
    ActionEngine,
    ActionStatus,
    ActionResult,
    BlockAction,
    FilterAction,
    RewriteAction,
    LogAction,
    EscalateAction,
    NotifyAction,
    BlockResult,
)

# Engine
from integrations.guardrails.guardrails_engine import (
    GuardrailsEngine,
    EngineConfig,
    SafetyLevel,
    SafetyResult,
    create_strict_engine,
    create_moderate_engine,
    create_permissive_engine,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    
    # Validators
    "Validator",
    "ValidationResult",
    "ValidationSeverity",
    "TypeValidator",
    "RegexValidator",
    "LengthValidator",
    "RangeValidator",
    "EnumValidator",
    "PIIValidator",
    "ToxicityValidator",
    "JSONValidator",
    "SchemaValidator",
    "QualityValidator",
    "CompositeValidator",
    "CustomValidator",
    
    # Rails
    "InputRail",
    "OutputRail",
    "RailSet",
    "RailStatus",
    "ProcessedInput",
    "ProcessedOutput",
    "SanitizationRail",
    "JailbreakDetectionRail",
    "ContextWindowRail",
    "KeywordFilterRail",
    "ValidationRail",
    "OutputSanitizationRail",
    "LoggingRail",
    
    # Policies
    "Policy",
    "PolicyEngine",
    "PolicyResult",
    "PolicyAction",
    "PolicySeverity",
    "SafetyPolicy",
    "CompliancePolicy",
    "ContentPolicy",
    "Violation",
    "Fix",
    
    # Actions
    "Action",
    "ActionEngine",
    "ActionStatus",
    "ActionResult",
    "BlockAction",
    "FilterAction",
    "RewriteAction",
    "LogAction",
    "EscalateAction",
    "NotifyAction",
    "BlockResult",
    
    # Engine
    "GuardrailsEngine",
    "EngineConfig",
    "SafetyLevel",
    "SafetyResult",
    "create_strict_engine",
    "create_moderate_engine",
    "create_permissive_engine",
]
