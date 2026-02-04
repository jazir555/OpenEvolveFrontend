"""Knowledge Engine Guardrails Integration.

AI safety and output validation specifically for Knowledge Graph operations.
Provides validation for KG extraction, Cypher queries, and entity management.

Following CLAUDE.md patterns:
- UTC timestamps for all operations
- Structured logging with correlation_id
- SSOT pattern for state management
- Fail-safe defaults

Example:
    >>> from knowledge_engine.integrations.guardrails import GuardrailsKGIntegration
    >>> guardrails = GuardrailsKGIntegration(safety_level=SafetyLevel.STRICT)
    >>> result = guardrails.validate_kg_output(kg_data, schema)
    >>> if not result.is_valid:
    ...     print(f"Validation failed: {result.message}")
"""

from knowledge_engine.integrations.guardrails.guardrails_integration import (
    GuardrailsKGIntegration,
    KGValidationResult,
    KGExtractionGuardResult,
)

# Re-export base guardrails types for convenience
from integrations.guardrails import (
    # Enums and Config
    SafetyLevel,
    ValidationSeverity,
    PolicySeverity,
    PolicyAction,
    RailStatus,
    ActionStatus,
    
    # Results
    ValidationResult,
    SafetyResult,
    PolicyResult,
    BlockResult,
    ActionResult,
    
    # Core Classes
    Violation,
    Fix,
    ProcessedInput,
    ProcessedOutput,
    EngineConfig,
    
    # Validators
    Validator,
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
    
    # Rails
    InputRail,
    OutputRail,
    RailSet,
    SanitizationRail,
    JailbreakDetectionRail,
    ContextWindowRail,
    KeywordFilterRail,
    ValidationRail,
    OutputSanitizationRail,
    LoggingRail,
    
    # Policies
    Policy,
    PolicyEngine,
    SafetyPolicy,
    CompliancePolicy,
    ContentPolicy,
    
    # Actions
    Action,
    ActionEngine,
    BlockAction,
    FilterAction,
    RewriteAction,
    LogAction,
    EscalateAction,
    NotifyAction,
    
    # Engine
    GuardrailsEngine,
    create_strict_engine,
    create_moderate_engine,
    create_permissive_engine,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    
    # KG Integration
    "GuardrailsKGIntegration",
    "KGValidationResult",
    "KGExtractionGuardResult",
    
    # Enums and Config (re-exported)
    "SafetyLevel",
    "ValidationSeverity",
    "PolicySeverity",
    "PolicyAction",
    "RailStatus",
    "ActionStatus",
    "EngineConfig",
    
    # Results (re-exported)
    "ValidationResult",
    "SafetyResult",
    "PolicyResult",
    "BlockResult",
    "ActionResult",
    "ProcessedInput",
    "ProcessedOutput",
    "Violation",
    "Fix",
    
    # Validators (re-exported)
    "Validator",
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
    
    # Rails (re-exported)
    "InputRail",
    "OutputRail",
    "RailSet",
    "SanitizationRail",
    "JailbreakDetectionRail",
    "ContextWindowRail",
    "KeywordFilterRail",
    "ValidationRail",
    "OutputSanitizationRail",
    "LoggingRail",
    
    # Policies (re-exported)
    "Policy",
    "PolicyEngine",
    "SafetyPolicy",
    "CompliancePolicy",
    "ContentPolicy",
    
    # Actions (re-exported)
    "Action",
    "ActionEngine",
    "BlockAction",
    "FilterAction",
    "RewriteAction",
    "LogAction",
    "EscalateAction",
    "NotifyAction",
    
    # Engine (re-exported)
    "GuardrailsEngine",
    "create_strict_engine",
    "create_moderate_engine",
    "create_permissive_engine",
]
