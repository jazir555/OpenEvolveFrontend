"""
RESE Phase IV: Architecture Assembly Canonical Schemas

This module defines the canonical data models for RESE Phase IV (Architecture Assembly),
including paradigm shifts, synthesized knowledge, and architecture validation.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config values via env vars
- Law of Idempotency: UPSERT logic with deduplication by assembly_id
- Structured Logging: JSON with correlation_id
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import uuid
import json

# Import from base RESE schemas
from rese_schemas import Hypothesis, Pattern, MCTSSearchResult, HypothesisStatus


# ============================================================================
# PHASE IV ENUMS
# ============================================================================

class AssemblyStatus(Enum):
    """Status of an architecture assembly operation."""
    PENDING = "pending"
    ASSEMBLING = "assembling"
    VALIDATED = "validated"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class ParadigmShiftType(Enum):
    """Types of paradigm shifts that can be assembled."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    CROSS_DOMAIN = "cross_domain"


class ValidationLevel(Enum):
    """Levels of validation for architecture assemblies."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    FORMAL = "formal"


class IntegrationStrategy(Enum):
    """Strategies for integrating knowledge from multiple phases."""
    MERGE = "merge"
    SYNTHESIZE = "synthesize"
    ABSTRACT = "abstract"
    HIERARCHICAL = "hierarchical"
    CROSS_VALIDATION = "cross_validation"


# ============================================================================
# PHASE I OUTPUT SCHEMAS (for integration)
# ============================================================================

@dataclass
class EpistemicAuditResult:
    """
    Result from Phase I: Epistemic Audit.

    Attributes:
        audit_id: Unique identifier
        constraints: List of formalized constraints
        contradictions: Detected contradictions
        tacit_assumptions: Mined tacit assumptions
        cognitive_biases: Detected cognitive biases
        validation_status: Overall validation status
        confidence: Audit confidence score [0.0, 1.0]
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
    """
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    tacit_assumptions: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_biases: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: str = "pending"
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audit_id": self.audit_id,
            "constraints": self.constraints,
            "contradictions": self.contradictions,
            "tacit_assumptions": self.tacit_assumptions,
            "cognitive_biases": self.cognitive_biases,
            "validation_status": self.validation_status,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpistemicAuditResult":
        """Create from dictionary."""
        data = data.copy()

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# PHASE II OUTPUT SCHEMAS (for integration)
# ============================================================================

@dataclass
class IsomorphicMappingResult:
    """
    Result from Phase II: Isomorphic Resonance.

    Attributes:
        mapping_id: Unique identifier
        isomorphisms: Discovered isomorphisms
        functional_dependencies: Functional dependency graphs
        domain_mappings: Cross-domain mappings
        similarity_scores: Similarity scores for mappings
        validation_results: Validation results for mappings
        confidence: Mapping confidence score [0.0, 1.0]
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
    """
    mapping_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    isomorphisms: List[Dict[str, Any]] = field(default_factory=list)
    functional_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    domain_mappings: List[Dict[str, Any]] = field(default_factory=list)
    similarity_scores: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mapping_id": self.mapping_id,
            "isomorphisms": self.isomorphisms,
            "functional_dependencies": self.functional_dependencies,
            "domain_mappings": self.domain_mappings,
            "similarity_scores": self.similarity_scores,
            "validation_results": self.validation_results,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsomorphicMappingResult":
        """Create from dictionary."""
        data = data.copy()

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# PHASE III OUTPUT SCHEMAS (for integration)
# ============================================================================

@dataclass
class MCTSRefinementResult:
    """
    Result from Phase III: MCTS Refinement.

    Attributes:
        refinement_id: Unique identifier
        search_results: MCTS search results
        validated_hypotheses: Validated hypotheses
        convergence_metrics: Convergence metrics
        aci_reduction: ACI (Algorithmic Complexity Index) reduction achieved
        statistical_validation: Statistical validation results
        confidence: Refinement confidence score [0.0, 1.0]
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
    """
    refinement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    search_results: List[MCTSSearchResult] = field(default_factory=list)
    validated_hypotheses: List[Hypothesis] = field(default_factory=list)
    convergence_metrics: Dict[str, Any] = field(default_factory=dict)
    aci_reduction: float = 0.0
    statistical_validation: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence and ACI reduction."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.aci_reduction = max(0.0, min(1.0, float(self.aci_reduction)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "refinement_id": self.refinement_id,
            "search_results": [sr.to_dict() for sr in self.search_results],
            "validated_hypotheses": [vh.to_dict() for vh in self.validated_hypotheses],
            "convergence_metrics": self.convergence_metrics,
            "aci_reduction": self.aci_reduction,
            "statistical_validation": self.statistical_validation,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSRefinementResult":
        """Create from dictionary."""
        data = data.copy()

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        if "search_results" in data:
            data["search_results"] = [
                MCTSSearchResult.from_dict(sr) if isinstance(sr, dict) else sr
                for sr in data["search_results"]
            ]

        if "validated_hypotheses" in data:
            data["validated_hypotheses"] = [
                Hypothesis.from_dict(vh) if isinstance(vh, dict) else vh
                for vh in data["validated_hypotheses"]
            ]

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# PARADIGM SHIFT SCHEMAS
# ============================================================================

@dataclass
class ParadigmShift:
    """
    A paradigm shift assembled from validated patterns across phases.

    Attributes:
        shift_id: Unique identifier
        shift_type: Type of paradigm shift
        description: Human-readable description
        source_patterns: List of patterns that contributed to this shift
        phase1_contributions: Contributions from Phase I (epistemic audit)
        phase2_contributions: Contributions from Phase II (isomorphic mappings)
        phase3_contributions: Contributions from Phase III (MCTS refinement)
        transformation_rules: Rules for applying this paradigm shift
        confidence: Confidence in this paradigm shift [0.0, 1.0]
        validation_status: Validation status
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
    """
    shift_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    shift_type: ParadigmShiftType = ParadigmShiftType.STRUCTURAL
    description: str = ""
    source_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    phase1_contributions: List[Dict[str, Any]] = field(default_factory=list)
    phase2_contributions: List[Dict[str, Any]] = field(default_factory=list)
    phase3_contributions: List[Dict[str, Any]] = field(default_factory=list)
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    validation_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for shift_id."""
        return self.shift_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shift_id": self.shift_id,
            "id": self.shift_id,
            "shift_type": self.shift_type.value if isinstance(self.shift_type, Enum) else self.shift_type,
            "description": self.description,
            "source_patterns": self.source_patterns,
            "phase1_contributions": self.phase1_contributions,
            "phase2_contributions": self.phase2_contributions,
            "phase3_contributions": self.phase3_contributions,
            "transformation_rules": self.transformation_rules,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParadigmShift":
        """Create from dictionary."""
        data = data.copy()

        if "shift_id" not in data and "id" in data:
            data["shift_id"] = data["id"]

        if "shift_type" in data and isinstance(data["shift_type"], str):
            try:
                data["shift_type"] = ParadigmShiftType(data["shift_type"])
            except ValueError:
                pass

        for field_name in ["created_at", "updated_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# SYNTHESIZED KNOWLEDGE SCHEMAS
# ============================================================================

@dataclass
class SynthesizedKnowledge:
    """
    Knowledge synthesized from multiple RESE phases.

    Attributes:
        knowledge_id: Unique identifier
        knowledge_type: Type of synthesized knowledge
        description: Human-readable description
        source_phase1: Phase I audit results (if available)
        source_phase2: Phase II mapping results (if available)
        source_phase3: Phase III refinement results (if available)
        paradigm_shifts: Paradigm shifts incorporated
        integration_strategy: Strategy used for integration
        synthesis_rules: Rules used for synthesis
        confidence: Overall confidence [0.0, 1.0]
        completeness: Completeness estimate [0.0, 1.0]
        consistency: Consistency estimate [0.0, 1.0]
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
    """
    knowledge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_type: str = "general"
    description: str = ""
    source_phase1: Optional[EpistemicAuditResult] = None
    source_phase2: Optional[IsomorphicMappingResult] = None
    source_phase3: Optional[MCTSRefinementResult] = None
    paradigm_shifts: List[ParadigmShift] = field(default_factory=list)
    integration_strategy: IntegrationStrategy = IntegrationStrategy.SYNTHESIZE
    synthesis_rules: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    completeness: float = 0.5
    consistency: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate numeric fields."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.completeness = max(0.0, min(1.0, float(self.completeness)))
        self.consistency = max(0.0, min(1.0, float(self.consistency)))

    @property
    def id(self) -> str:
        """Alias for knowledge_id."""
        return self.knowledge_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "knowledge_id": self.knowledge_id,
            "id": self.knowledge_id,
            "knowledge_type": self.knowledge_type,
            "description": self.description,
            "source_phase1": self.source_phase1.to_dict() if self.source_phase1 else None,
            "source_phase2": self.source_phase2.to_dict() if self.source_phase2 else None,
            "source_phase3": self.source_phase3.to_dict() if self.source_phase3 else None,
            "paradigm_shifts": [ps.to_dict() for ps in self.paradigm_shifts],
            "integration_strategy": self.integration_strategy.value if isinstance(self.integration_strategy, Enum) else self.integration_strategy,
            "synthesis_rules": self.synthesis_rules,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "consistency": self.consistency,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesizedKnowledge":
        """Create from dictionary."""
        data = data.copy()

        if "knowledge_id" not in data and "id" in data:
            data["knowledge_id"] = data["id"]

        if "source_phase1" in data and isinstance(data["source_phase1"], dict):
            data["source_phase1"] = EpistemicAuditResult.from_dict(data["source_phase1"])

        if "source_phase2" in data and isinstance(data["source_phase2"], dict):
            data["source_phase2"] = IsomorphicMappingResult.from_dict(data["source_phase2"])

        if "source_phase3" in data and isinstance(data["source_phase3"], dict):
            data["source_phase3"] = MCTSRefinementResult.from_dict(data["source_phase3"])

        if "paradigm_shifts" in data:
            data["paradigm_shifts"] = [
                ParadigmShift.from_dict(ps) if isinstance(ps, dict) else ps
                for ps in data["paradigm_shifts"]
            ]

        if "integration_strategy" in data and isinstance(data["integration_strategy"], str):
            try:
                data["integration_strategy"] = IntegrationStrategy(data["integration_strategy"])
            except ValueError:
                pass

        for field_name in ["created_at", "updated_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# ARCHITECTURE ASSEMBLY SCHEMAS
# ============================================================================

@dataclass
class ArchitectureAssembly:
    """
    Final architecture assembly from all RESE phases.

    Attributes:
        assembly_id: Unique identifier
        synthesized_knowledge: Synthesized knowledge object
        paradigm_shifts: List of paradigm shifts
        validation_results: Validation results
        final_architecture: Final architecture specification
        aci_reduction_achieved: Total ACI reduction achieved [0.0, 1.0]
        confidence: Overall confidence [0.0, 1.0]
        validation_level: Level of validation performed
        status: Assembly status
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
    """
    assembly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    synthesized_knowledge: Optional[SynthesizedKnowledge] = None
    paradigm_shifts: List[ParadigmShift] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    final_architecture: Dict[str, Any] = field(default_factory=dict)
    aci_reduction_achieved: float = 0.0
    confidence: float = 0.5
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    status: AssemblyStatus = AssemblyStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate numeric fields."""
        self.aci_reduction_achieved = max(0.0, min(1.0, float(self.aci_reduction_achieved)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for assembly_id."""
        return self.assembly_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "id": self.assembly_id,
            "synthesized_knowledge": self.synthesized_knowledge.to_dict() if self.synthesized_knowledge else None,
            "paradigm_shifts": [ps.to_dict() for ps in self.paradigm_shifts],
            "validation_results": self.validation_results,
            "final_architecture": self.final_architecture,
            "aci_reduction_achieved": self.aci_reduction_achieved,
            "confidence": self.confidence,
            "validation_level": self.validation_level.value if isinstance(self.validation_level, Enum) else self.validation_level,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureAssembly":
        """Create from dictionary."""
        data = data.copy()

        if "assembly_id" not in data and "id" in data:
            data["assembly_id"] = data["id"]

        if "synthesized_knowledge" in data and isinstance(data["synthesized_knowledge"], dict):
            data["synthesized_knowledge"] = SynthesizedKnowledge.from_dict(data["synthesized_knowledge"])

        if "paradigm_shifts" in data:
            data["paradigm_shifts"] = [
                ParadigmShift.from_dict(ps) if isinstance(ps, dict) else ps
                for ps in data["paradigm_shifts"]
            ]

        if "validation_level" in data and isinstance(data["validation_level"], str):
            try:
                data["validation_level"] = ValidationLevel(data["validation_level"])
            except ValueError:
                pass

        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = AssemblyStatus(data["status"])
            except ValueError:
                pass

        for field_name in ["created_at", "updated_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# PHASE IV CONFIGURATION
# ============================================================================

@dataclass
class Phase4Config:
    """
    Configuration for Phase IV: Architecture Assembly.

    All values must come from environment variables (Law of Configuration Explicitness).

    Attributes:
        assembly_timeout_ms: Timeout for assembly operations in ms
        validation_level: Level of validation to perform
        integration_strategy: Default integration strategy
        max_paradigm_shifts: Maximum paradigm shifts to assemble
        min_confidence_threshold: Minimum confidence threshold
        enable_cross_validation: Enable cross-phase validation
        enable_formal_verification: Enable formal verification (Lean 4)
        correlation_id: For tracing (optional)
    """
    assembly_timeout_ms: int = 25000
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    integration_strategy: IntegrationStrategy = IntegrationStrategy.SYNTHESIZE
    max_paradigm_shifts: int = 50
    min_confidence_threshold: float = 0.7
    enable_cross_validation: bool = True
    enable_formal_verification: bool = False
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Phase4Config":
        """
        Create configuration from environment variables.

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        import os

        env_vars = {
            "PHASE4_ASSEMBLY_TIMEOUT_MS": ("assembly_timeout_ms", 25000, int),
            "PHASE4_MAX_PARADIGM_SHIFTS": ("max_paradigm_shifts", 50, int),
            "PHASE4_MIN_CONFIDENCE_THRESHOLD": ("min_confidence_threshold", 0.7, float),
        }

        config = {}
        for env_name, (field_name, default, field_type) in env_vars.items():
            value = os.getenv(env_name)
            if value is None:
                config[field_name] = default
            else:
                try:
                    config[field_name] = field_type(value)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid value for {env_name}: {value}. "
                        f"Expected {field_type.__name__}."
                    )

        # Parse validation level
        validation_level_str = os.getenv("PHASE4_VALIDATION_LEVEL", "standard")
        try:
            config["validation_level"] = ValidationLevel(validation_level_str)
        except ValueError:
            config["validation_level"] = ValidationLevel.STANDARD

        # Parse integration strategy
        strategy_str = os.getenv("PHASE4_INTEGRATION_STRATEGY", "synthesize")
        try:
            config["integration_strategy"] = IntegrationStrategy(strategy_str)
        except ValueError:
            config["integration_strategy"] = IntegrationStrategy.SYNTHESIZE

        # Parse boolean flags
        config["enable_cross_validation"] = os.getenv("PHASE4_ENABLE_CROSS_VALIDATION", "true").lower() == "true"
        config["enable_formal_verification"] = os.getenv("PHASE4_ENABLE_FORMAL_VERIFICATION", "false").lower() == "true"

        # Optional correlation ID
        config["correlation_id"] = os.getenv("CORRELATION_ID")

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assembly_timeout_ms": self.assembly_timeout_ms,
            "validation_level": self.validation_level.value if isinstance(self.validation_level, Enum) else self.validation_level,
            "integration_strategy": self.integration_strategy.value if isinstance(self.integration_strategy, Enum) else self.integration_strategy,
            "max_paradigm_shifts": self.max_paradigm_shifts,
            "min_confidence_threshold": self.min_confidence_threshold,
            "enable_cross_validation": self.enable_cross_validation,
            "enable_formal_verification": self.enable_formal_verification,
            "correlation_id": self.correlation_id,
        }


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Enums
    "AssemblyStatus",
    "ParadigmShiftType",
    "ValidationLevel",
    "IntegrationStrategy",

    # Phase outputs (for integration)
    "EpistemicAuditResult",
    "IsomorphicMappingResult",
    "MCTSRefinementResult",

    # Phase IV schemas
    "ParadigmShift",
    "SynthesizedKnowledge",
    "ArchitectureAssembly",

    # Configuration
    "Phase4Config",
]
