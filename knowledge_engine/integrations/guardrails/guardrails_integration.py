"""Knowledge Engine Guardrails Integration.

AI safety and output validation for Knowledge Graph operations.
Provides validation, sanitization, and safety checking specifically
for KG extraction, query generation, and data operations.

Following CLAUDE.md patterns:
- UTC timestamps for all operations
- Structured logging with correlation_id
- SSOT pattern for state management
- Fail-safe defaults (block on error)
- Circuit breaker pattern for external checks
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from integrations.guardrails import (
    GuardrailsEngine,
    SafetyLevel,
    ValidationResult,
    ValidationSeverity,
    SafetyResult,
    Violation,
    PolicyResult,
    PIIValidator,
    ToxicityValidator,
    JSONValidator,
    SchemaValidator,
    TypeValidator,
    LengthValidator,
    PolicySeverity
)

logger = logging.getLogger(__name__)


@dataclass
class KGValidationResult:
    """Result of KG-specific validation.
    
    SSOT for KG validation outcome.
    """
    is_valid: bool
    validation_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    sanitized_output: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "validation_type": self.validation_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "details": self.details
        }


@dataclass
class KGExtractionGuardResult:
    """Result of guarding KG extraction.
    
    SSOT for extraction guard outcome.
    """
    allowed: bool
    original_text: str
    sanitized_text: Optional[str] = None
    extraction_allowed: bool = True
    violations: List[Violation] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "extraction_allowed": self.extraction_allowed,
            "violation_count": len(self.violations),
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class GuardrailsKGIntegration:
    """Integration between Guardrails and Knowledge Engine.
    
    Provides AI safety and validation specifically for KG operations:
    - KG extraction validation
    - Cypher query safety checks
    - Entity validation
    - PII detection in KG data
    
    Example:
        >>> guardrails = GuardrailsKGIntegration()
        >>> result = guardrails.validate_kg_output(
        ...     {"entities": [{"name": "John", "type": "PERSON"}]},
        ...     {"type": "object", "properties": {"entities": {"type": "array"}}}
        ... )
    """
    
    # Common Cypher injection patterns
    CYPHER_INJECTION_PATTERNS = [
        re.compile(r'\bDROP\s+(?:NODE|RELATIONSHIP|CONSTRAINT|INDEX)\b', re.IGNORECASE),
        re.compile(r'\bDELETE\s+(?:ALL|NODE|RELATIONSHIP)\b', re.IGNORECASE),
        re.compile(r'\bREMOVE\s+ALL\b', re.IGNORECASE),
        re.compile(r'\bCALL\s+dbms\b', re.IGNORECASE),
        re.compile(r'\bapoc\.', re.IGNORECASE),
        re.compile(r'\bLOAD\s+CSV\b', re.IGNORECASE),
    ]
    
    # Allowed entity types (whitelist)
    DEFAULT_ALLOWED_ENTITY_TYPES = {
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
        "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "GPE", "FAC"
    }
    
    # Sensitive entity types that may contain PII
    SENSITIVE_ENTITY_TYPES = {"PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD"}
    
    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.MODERATE,
        allowed_entity_types: Optional[Set[str]] = None,
        block_cypher_modification: bool = True,
        redact_pii_in_kg: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize KG Guardrails integration.
        
        Args:
            safety_level: Safety strictness level
            allowed_entity_types: Set of allowed entity type labels
            block_cypher_modification: Block queries that modify graph structure
            redact_pii_in_kg: Automatically redact PII in KG data
            config: Additional configuration
        """
        self.safety_level = safety_level
        self.allowed_entity_types = allowed_entity_types or self.DEFAULT_ALLOWED_ENTITY_TYPES.copy()
        self.block_cypher_modification = block_cypher_modification
        self.redact_pii_in_kg = redact_pii_in_kg
        self.config = config or {}
        
        # Initialize base guardrails engine
        self.engine = GuardrailsEngine(safety_level=safety_level)
        
        # Add KG-specific validators
        self._setup_kg_validators()
        
        # Statistics
        self._stats = {
            "extractions_guarded": 0,
            "queries_validated": 0,
            "pii_detected": 0,
            "queries_blocked": 0,
            "entities_redacted": 0
        }
        
        logger.info({
            "msg": "GuardrailsKGIntegration initialized",
            "safety_level": safety_level.value,
            "allowed_entity_types": len(self.allowed_entity_types)
        })
        
    def _setup_kg_validators(self) -> None:
        """Set up validators specific to KG operations."""
        # Add specialized validators for KG
        self.engine.add_validator(JSONValidator(allow_partial=True, name="KGStructureValidator"))
        self.engine.add_validator(PIIValidator(name="KGPIIValidator", block_on_detection=self.safety_level == SafetyLevel.STRICT))
        
    def validate_kg_output(
        self,
        output: Any,
        schema: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Validate KG extraction output against schema.
        
        Args:
            output: KG extraction result (entities, relations, etc.)
            schema: Expected JSON schema
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult with validation status
        """
        correlation_id = correlation_id or f"kg_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        try:
            # First, run through standard guardrails
            validation_result = self.engine.validate(output, schema, correlation_id)
            
            if not validation_result.is_valid:
                return KGValidationResult(
                    is_valid=False,
                    validation_type="schema",
                    message=validation_result.message,
                    correlation_id=correlation_id,
                    details={"validator": validation_result.validator_name}
                )
                
            # Validate entity types if present
            if isinstance(output, dict) and "entities" in output:
                entity_result = self._validate_entity_types(output["entities"], correlation_id)
                if not entity_result.is_valid:
                    return entity_result
                    
            # Check for PII in KG data
            if self.redact_pii_in_kg:
                sanitized = self._sanitize_kg_data(output, correlation_id)
                if sanitized != output:
                    return KGValidationResult(
                        is_valid=True,
                        validation_type="pii_sanitization",
                        message="KG output sanitized for PII",
                        correlation_id=correlation_id,
                        sanitized_output=sanitized,
                        details={"sanitized": True}
                    )
                    
            return KGValidationResult(
                is_valid=True,
                validation_type="complete",
                message="KG output validation passed",
                correlation_id=correlation_id,
                details={"schema_validated": schema is not None}
            )
            
        except Exception as e:
            logger.error({
                "msg": "KG validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return KGValidationResult(
                is_valid=False,
                validation_type="error",
                message=f"Validation error: {str(e)}",
                correlation_id=correlation_id
            )
            
    def _validate_entity_types(
        self,
        entities: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Validate entity types are in allowed set.
        
        Args:
            entities: List of entity dictionaries
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult
        """
        invalid_entities = []
        
        for entity in entities:
            entity_type = entity.get("type", entity.get("label", ""))
            if entity_type and entity_type.upper() not in self.allowed_entity_types:
                invalid_entities.append({
                    "name": entity.get("name", entity.get("text", "unknown")),
                    "type": entity_type
                })
                
        if invalid_entities:
            return KGValidationResult(
                is_valid=False,
                validation_type="entity_types",
                message=f"Invalid entity types found: {[e['type'] for e in invalid_entities]}",
                correlation_id=correlation_id,
                details={"invalid_entities": invalid_entities}
            )
            
        return KGValidationResult(
            is_valid=True,
            validation_type="entity_types",
            message="All entity types valid",
            correlation_id=correlation_id,
            details={"entity_count": len(entities)}
        )
        
    def validate_entity_types(
        self,
        entities: List[Dict[str, Any]],
        allowed_types: Optional[Set[str]] = None,
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Validate entities against allowed types.
        
        Args:
            entities: List of entity dictionaries
            allowed_types: Optional override for allowed types
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult
        """
        types = allowed_types or self.allowed_entity_types
        invalid = []
        
        for entity in entities:
            entity_type = entity.get("type", entity.get("label", "")).upper()
            if entity_type not in types:
                invalid.append({
                    "entity": entity.get("name", entity.get("text", "unknown")),
                    "type": entity_type
                })
                
        if invalid:
            return KGValidationResult(
                is_valid=False,
                validation_type="entity_types",
                message=f"{len(invalid)} entities have disallowed types",
                correlation_id=correlation_id,
                details={"invalid": invalid, "allowed": list(types)}
            )
            
        return KGValidationResult(
            is_valid=True,
            validation_type="entity_types",
            message=f"All {len(entities)} entities have valid types",
            correlation_id=correlation_id
        )
        
    def sanitize_kg_input(
        self,
        query: str,
        correlation_id: Optional[str] = None
    ) -> str:
        """Sanitize KG input/query.
        
        Args:
            query: Input text or query
            correlation_id: Optional correlation ID
            
        Returns:
            Sanitized query
        """
        if not isinstance(query, str):
            return str(query)
            
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', query)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Limit length
        max_len = self.config.get("max_input_length", 10000)
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
            logger.warning({
                "msg": "KG input truncated",
                "original_length": len(query),
                "max_length": max_len,
                "correlation_id": correlation_id
            })
            
        return sanitized
        
    def check_kg_safety(
        self,
        kg_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> SafetyResult:
        """Check safety of KG data.
        
        Args:
            kg_data: Knowledge graph data (entities, relations)
            correlation_id: Optional correlation ID
            
        Returns:
            SafetyResult
        """
        correlation_id = correlation_id or f"kg_safe_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Convert KG data to text for safety check
        text_repr = self._kg_to_text(kg_data)
        
        # Use base engine for safety check
        safety_result = self.engine.check_safety(
            input_data="",
            output_data=text_repr,
            context={"kg_operation": True},
            correlation_id=correlation_id
        )
        
        self._stats["extractions_guarded"] += 1
        if safety_result.violations:
            self._stats["pii_detected"] += len([v for v in safety_result.violations if "pii" in v.rule_name.lower()])
            
        return safety_result
        
    def _kg_to_text(self, kg_data: Dict[str, Any]) -> str:
        """Convert KG data to text representation for safety checking."""
        parts = []
        
        if "entities" in kg_data:
            for entity in kg_data["entities"]:
                name = entity.get("name", entity.get("text", ""))
                entity_type = entity.get("type", entity.get("label", ""))
                parts.append(f"{name} ({entity_type})")
                
        if "relations" in kg_data:
            for relation in kg_data["relations"]:
                source = relation.get("source", relation.get("from", ""))
                target = relation.get("target", relation.get("to", ""))
                rel_type = relation.get("type", relation.get("relation", ""))
                parts.append(f"{source} -[{rel_type}]-> {target}")
                
        return " ".join(parts)
        
    def enforce_extraction_policies(
        self,
        extraction: Dict[str, Any],
        source_text: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> PolicyResult:
        """Enforce policies on KG extraction.
        
        Args:
            extraction: KG extraction result
            source_text: Original source text
            correlation_id: Optional correlation ID
            
        Returns:
            PolicyResult
        """
        correlation_id = correlation_id or f"kg_pol_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Convert extraction to text for policy check
        extraction_text = self._kg_to_text(extraction)
        
        policy_result = self.engine.enforce_policies(
            input_data=source_text or "",
            output_data=extraction_text,
            context={"kg_extraction": True},
            correlation_id=correlation_id
        )
        
        return policy_result
        
    def redact_sensitive_kg(
        self,
        kg_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Redact sensitive information from KG data.
        
        Args:
            kg_data: Knowledge graph data
            correlation_id: Optional correlation ID
            
        Returns:
            Redacted KG data
        """
        import copy
        redacted = copy.deepcopy(kg_data)
        redaction_count = 0
        
        if "entities" in redacted:
            for entity in redacted["entities"]:
                entity_type = entity.get("type", entity.get("label", "")).upper()
                
                if entity_type in self.SENSITIVE_ENTITY_TYPES:
                    # Redact sensitive entities
                    if "name" in entity:
                        entity["name"] = "[REDACTED]"
                        redaction_count += 1
                    if "text" in entity:
                        entity["text"] = "[REDACTED]"
                        redaction_count += 1
                    entity["redacted"] = True
                    entity["original_type"] = entity_type
                    
        self._stats["entities_redacted"] += redaction_count
        
        if redaction_count > 0:
            logger.info({
                "msg": "KG data redacted",
                "redaction_count": redaction_count,
                "correlation_id": correlation_id
            })
            
        return redacted
        
    def _sanitize_kg_data(
        self,
        kg_data: Any,
        correlation_id: Optional[str] = None
    ) -> Any:
        """Sanitize KG data by redacting PII."""
        if isinstance(kg_data, dict):
            # Check if this looks like a KG structure
            if "entities" in kg_data or "relations" in kg_data:
                return self.redact_sensitive_kg(kg_data, correlation_id)
            else:
                return {k: self._sanitize_kg_data(v, correlation_id) for k, v in kg_data.items()}
        elif isinstance(kg_data, list):
            return [self._sanitize_kg_data(item, correlation_id) for item in kg_data]
        elif isinstance(kg_data, str):
            # Check for PII in string values
            pii_validator = PIIValidator(block_on_detection=False)
            result = pii_validator.validate(kg_data, correlation_id)
            if not result.is_valid:
                fixed = pii_validator.fix(kg_data, result)
                return fixed if fixed else kg_data
        return kg_data
        
    def validate_cypher_query(
        self,
        query: str,
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Validate Cypher query for safety.
        
        Args:
            query: Cypher query string
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult
        """
        correlation_id = correlation_id or f"cypher_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        if not isinstance(query, str):
            return KGValidationResult(
                is_valid=False,
                validation_type="type",
                message="Query must be a string",
                correlation_id=correlation_id
            )
            
        self._stats["queries_validated"] += 1
        
        # Check for injection patterns
        if self.block_cypher_modification:
            for pattern in self.CYPHER_INJECTION_PATTERNS:
                if pattern.search(query):
                    self._stats["queries_blocked"] += 1
                    return KGValidationResult(
                        is_valid=False,
                        validation_type="injection",
                        message=f"Potentially dangerous query pattern detected: {pattern.pattern[:30]}",
                        correlation_id=correlation_id,
                        details={"pattern": pattern.pattern[:50]}
                    )
                    
        # Basic Cypher syntax validation
        syntax_result = self._validate_cypher_syntax(query, correlation_id)
        if not syntax_result.is_valid:
            return syntax_result
            
        return KGValidationResult(
            is_valid=True,
            validation_type="cypher",
            message="Cypher query validation passed",
            correlation_id=correlation_id,
            details={"query_length": len(query)}
        )
        
    def _validate_cypher_syntax(
        self,
        query: str,
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Basic Cypher syntax validation."""
        # Check for balanced braces
        open_count = query.count("(")
        close_count = query.count(")")
        if open_count != close_count:
            return KGValidationResult(
                is_valid=False,
                validation_type="syntax",
                message=f"Unbalanced parentheses: {open_count} open, {close_count} close",
                correlation_id=correlation_id
            )
            
        # Check for balanced brackets
        open_bracket = query.count("[")
        close_bracket = query.count("]")
        if open_bracket != close_bracket:
            return KGValidationResult(
                is_valid=False,
                validation_type="syntax",
                message=f"Unbalanced brackets: {open_bracket} open, {close_bracket} close",
                correlation_id=correlation_id
            )
            
        # Check for balanced braces
        open_brace = query.count("{")
        close_brace = query.count("}")
        if open_brace != close_brace:
            return KGValidationResult(
                is_valid=False,
                validation_type="syntax",
                message=f"Unbalanced braces: {open_brace} open, {close_brace} close",
                correlation_id=correlation_id
            )
            
        return KGValidationResult(
            is_valid=True,
            validation_type="syntax",
            message="Syntax validation passed",
            correlation_id=correlation_id
        )
        
    def check_query_safety(
        self,
        query: str,
        correlation_id: Optional[str] = None
    ) -> SafetyResult:
        """Check safety of Cypher query.
        
        Args:
            query: Cypher query string
            correlation_id: Optional correlation ID
            
        Returns:
            SafetyResult
        """
        correlation_id = correlation_id or f"cypher_safe_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Use base engine to check query safety
        safety_result = self.engine.check_safety(
            input_data=query,
            output_data=query,
            context={"cypher_query": True},
            correlation_id=correlation_id
        )
        
        # Additional Cypher-specific safety checks
        validation = self.validate_cypher_query(query, correlation_id)
        if not validation.is_valid:
            # Add violation for unsafe query
            violation = Violation(
                policy_name="CypherSafety",
                rule_name="query_validation",
                message=validation.message,
                severity=PolicySeverity.HIGH,
                correlation_id=correlation_id
            )
            safety_result.violations.append(violation)
            safety_result.safe = False
            
        return safety_result
        
    def guard_kg_extraction(
        self,
        text: str,
        extraction_result: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> KGExtractionGuardResult:
        """Complete guard flow for KG extraction.
        
        Args:
            text: Original source text
            extraction_result: KG extraction output
            correlation_id: Optional correlation ID
            
        Returns:
            KGExtractionGuardResult
        """
        correlation_id = correlation_id or f"kg_guard_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Check safety of source text
        text_safety = self.engine.check_safety(
            input_data="",
            output_data=text,
            correlation_id=correlation_id
        )
        
        if not text_safety.safe and self.safety_level == SafetyLevel.STRICT:
            return KGExtractionGuardResult(
                allowed=False,
                original_text=text,
                extraction_allowed=False,
                violations=text_safety.violations,
                correlation_id=correlation_id
            )
            
        # Check safety of extraction
        kg_safety = self.check_kg_safety(extraction_result, correlation_id)
        
        # Validate entity types
        if "entities" in extraction_result:
            entity_validation = self.validate_entity_types(
                extraction_result["entities"],
                correlation_id=correlation_id
            )
            if not entity_validation.is_valid and self.safety_level == SafetyLevel.STRICT:
                violation = Violation(
                    policy_name="KGValidation",
                    rule_name="invalid_entity_types",
                    message=entity_validation.message,
                    severity=PolicySeverity.MEDIUM,
                    correlation_id=correlation_id
                )
                kg_safety.violations.append(violation)
                
        # Redact if needed
        sanitized_text = None
        if self.redact_pii_in_kg and kg_safety.violations:
            sanitized_text = self._sanitize_kg_text(text)
            
        allowed = len(kg_safety.violations) == 0 or self.safety_level != SafetyLevel.STRICT
        
        return KGExtractionGuardResult(
            allowed=allowed,
            original_text=text,
            sanitized_text=sanitized_text,
            extraction_allowed=allowed,
            violations=kg_safety.violations,
            correlation_id=correlation_id
        )
        
    def _sanitize_kg_text(self, text: str) -> str:
        """Sanitize text for KG extraction."""
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
        # Redact phone numbers
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE_REDACTED]', text)
        # Redact SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        return text
        
    def guard_cypher_generation(
        self,
        natural_query: str,
        cypher: str,
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Guard Cypher query generation.
        
        Args:
            natural_query: Original natural language query
            cypher: Generated Cypher query
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult
        """
        correlation_id = correlation_id or f"cypher_gen_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Validate the Cypher query
        validation = self.validate_cypher_query(cypher, correlation_id)
        if not validation.is_valid:
            return validation
            
        # Check query safety
        safety = self.check_query_safety(cypher, correlation_id)
        if not safety.safe:
            return KGValidationResult(
                is_valid=False,
                validation_type="safety",
                message=f"Cypher query failed safety check: {len(safety.violations)} violations",
                correlation_id=correlation_id,
                details={"violations": [v.rule_name for v in safety.violations]}
            )
            
        return KGValidationResult(
            is_valid=True,
            validation_type="cypher_generation",
            message="Cypher query generation validated",
            correlation_id=correlation_id,
            details={
                "natural_query_length": len(natural_query),
                "cypher_length": len(cypher)
            }
        )
        
    def check_for_pii_in_kg(
        self,
        kg_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> KGValidationResult:
        """Check for PII in KG data.
        
        Args:
            kg_data: Knowledge graph data
            correlation_id: Optional correlation ID
            
        Returns:
            KGValidationResult with PII detection results
        """
        correlation_id = correlation_id or f"kg_pii_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        # Convert to text
        text = self._kg_to_text(kg_data)
        
        # Use PII validator
        pii_validator = PIIValidator(block_on_detection=False)
        result = pii_validator.validate(text, correlation_id)
        
        if not result.is_valid:
            self._stats["pii_detected"] += 1
            return KGValidationResult(
                is_valid=False,
                validation_type="pii_detection",
                message=f"PII detected in KG data: {result.message}",
                correlation_id=correlation_id,
                details=result.details
            )
            
        return KGValidationResult(
            is_valid=True,
            validation_type="pii_detection",
            message="No PII detected in KG data",
            correlation_id=correlation_id
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "kg_integration": self._stats.copy(),
            "base_engine": self.engine.get_stats(),
            "safety_level": self.safety_level.value,
            "allowed_entity_types": len(self.allowed_entity_types),
            "redact_pii": self.redact_pii_in_kg,
            "block_cypher_modification": self.block_cypher_modification
        }
