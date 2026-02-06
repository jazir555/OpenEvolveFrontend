"""
Base schema classes for the Knowledge Engine Schema System.

Consolidated unified data models for:
- Knowledge artifacts (from artifact_taxonomy.py and data/storage.py)
- Entities and Relationships (from graph/models.py and core/entity_knowledge_graph.py)
- Validation results (from schemas/entity_schema_manager.py and sovereign_data_models.py)
- Property types (unified from graph/schema.py and schemas/base.py)

All timestamps use timezone-aware UTC (datetime.now(timezone.utc)).
All enums serialize using .value.
All datetime objects serialize to ISO format strings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum, auto
from datetime import datetime, timezone
import re
import uuid


# ============================================================================
# PROPERTY TYPE ENUM (Consolidated from schemas/base.py and graph/schema.py)
# ============================================================================

class PropertyType(Enum):
    """Supported property types for entity attributes."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    # Additional types from graph/schema.py
    LIST = "list"
    DICT = "dict"
    EMBEDDING = "embedding"


# ============================================================================
# ARTIFACT ENUMS (from artifact_taxonomy.py)
# ============================================================================

class ArtifactCategory(Enum):
    """High-level artifact categories."""
    SOLUTION = "solution"
    ANTI_PATTERN = "anti_pattern"
    PROCESS = "process"
    DOMAIN = "domain"
    PERFORMANCE = "performance"
    TEAM = "team"
    SYSTEM = "system"
    QUALITY = "quality"
    LEARNING = "learning"
    OPERATIONAL = "operational"


class ArtifactType(Enum):
    """Ultra-comprehensive artifact type taxonomy. 30+ distinct artifact types."""
    
    # === SOLUTION PATTERNS (8 types) ===
    SOLUTION_PATTERN = "solution_pattern"
    CODE_PATTERN = "code_pattern"
    ARCHITECTURE_PATTERN = "architecture_pattern"
    INTEGRATION_PATTERN = "integration_pattern"
    DEPLOYMENT_PATTERN = "deployment_pattern"
    SCALING_PATTERN = "scaling_pattern"
    MIGRATION_PATTERN = "migration_pattern"
    RECOVERY_PATTERN = "recovery_pattern"
    
    # === ANTI-PATTERNS (5 types) ===
    ANTI_PATTERN = "anti_pattern"
    SECURITY_ANTI_PATTERN = "security_anti_pattern"
    PERFORMANCE_ANTI_PATTERN = "performance_anti_pattern"
    DESIGN_ANTI_PATTERN = "design_anti_pattern"
    OPERATIONAL_ANTI_PATTERN = "operational_anti_pattern"
    
    # === PROCESS ARTIFACTS (6 types) ===
    DECOMPOSITION_STRATEGY = "decomposition_strategy"
    WORKFLOW_TEMPLATE = "workflow_template"
    DECISION_FRAMEWORK = "decision_framework"
    COMMUNICATION_TEMPLATE = "communication_template"
    COLLABORATION_PATTERN = "collaboration_pattern"
    FACILITATION_GUIDE = "facilitation_guide"
    
    # === DOMAIN KNOWLEDGE (4 types) ===
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    REGULATORY_GUIDANCE = "regulatory_guidance"
    INDUSTRY_STANDARD = "industry_standard"
    TECHNOLOGY_GUIDE = "technology_guide"
    
    # === PERFORMANCE ARTIFACTS (5 types) ===
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    OPTIMIZATION_RECORD = "optimization_record"
    CAPACITY_PLANNING_GUIDE = "capacity_planning_guide"
    COST_OPTIMIZATION = "cost_optimization"
    RESOURCE_ESTIMATION = "resource_estimation"
    
    # === TEAM ARTIFACTS (5 types) ===
    TEAM_PERFORMANCE_DATA = "team_performance_data"
    SKILL_MATRIX = "skill_matrix"
    ASSIGNMENT_STRATEGY = "assignment_strategy"
    ONBOARDING_GUIDE = "onboarding_guide"
    RETENTION_PATTERN = "retention_pattern"
    
    # === SYSTEM ARTIFACTS (5 types) ===
    SYSTEM_BLUEPRINT = "system_blueprint"
    COMPONENT_SPECIFICATION = "component_specification"
    API_DESIGN_PATTERN = "api_design_pattern"
    DATA_MODEL = "data_model"
    CONFIGURATION_TEMPLATE = "configuration_template"
    
    # === QUALITY ARTIFACTS (6 types) ===
    QUALITY_CRITERIA = "quality_criteria"
    TESTING_STRATEGY = "testing_strategy"
    VALIDATION_RULESET = "validation_ruleset"
    GAUNTLET_EFFECTIVENESS = "gauntlet_effectiveness"
    DEFECT_PATTERN = "defect_pattern"
    REVIEW_CHECKLIST = "review_checklist"
    
    # === LEARNING ARTIFACTS (5 types) ===
    LEARNING_PATH = "learning_path"
    TUTORIAL_TEMPLATE = "tutorial_template"
    EXPLANATION_PATTERN = "explanation_pattern"
    KNOWLEDGE_TRANSFER_GUIDE = "knowledge_transfer_guide"
    MENTORING_FRAMEWORK = "mentoring_framework"
    
    # === OPERATIONAL ARTIFACTS (6 types) ===
    INCIDENT_RESPONSE_PLAYBOOK = "incident_response_playbook"
    MONITORING_CONFIGURATION = "monitoring_configuration"
    MAINTENANCE_PROCEDURE = "maintenance_procedure"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    ROLLBACK_PROCEDURE = "rollback_procedure"
    SECURITY_PROCEDURE = "security_procedure"
    
    # === SPECIALIZED ARTIFACTS (5 types) ===
    CREATIVE_PATTERN = "creative_pattern"
    PROMPT_PATTERN = "prompt_pattern"
    ESTIMATION_HEURISTIC = "estimation_heuristic"
    RISK_PATTERN = "risk_pattern"
    DEPENDENCY_MAP = "dependency_map"


# Total artifact types count
TOTAL_ARTIFACT_TYPES = len(ArtifactType)


# ============================================================================
# ENTITY AND RELATIONSHIP TYPE ENUMS (from graph/schema.py)
# ============================================================================

class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    # Core entities
    CONCEPT = "Concept"
    ENTITY = "Entity"
    EVENT = "Event"
    DOCUMENT = "Document"
    CODE = "Code"
    
    # Project-specific
    PROJECT = "Project"
    TASK = "Task"
    TEAM = "Team"
    WORKFLOW = "Workflow"
    
    # Knowledge-specific
    FACT = "Fact"
    RULE = "Rule"
    PATTERN = "Pattern"
    STRATEGY = "Strategy"
    
    # Agent-specific
    AGENT = "Agent"
    ACTION = "Action"
    DECISION = "Decision"
    
    # Semantic
    TOPIC = "Topic"
    CATEGORY = "Category"
    TAG = "Tag"


class RelationshipType(Enum):
    """Types of relationships between entities."""
    # Hierarchical
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    
    # Causal
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    
    # Semantic
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    
    # Temporal
    FOLLOWS = "FOLLOWS"
    PRECEDES = "PRECEDES"
    
    # Project
    ASSIGNED_TO = "ASSIGNED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    BLOCKS = "BLOCKS"
    
    # Knowledge
    EVIDENCE_FOR = "EVIDENCE_FOR"
    REFUTES = "REFUTES"
    IMPLEMENTS = "IMPLEMENTS"
    
    # Agent
    PERFORMED_BY = "PERFORMED_BY"
    DECIDED_BY = "DECIDED_BY"
    TRIGGERED = "TRIGGERED"


# ============================================================================
# PROPERTY DEFINITION
# ============================================================================

@dataclass
class PropertyDefinition:
    """
    Defines a property on an entity type.
    
    Attributes:
        name: Property name
        type: Property data type
        required: Whether this property is required
        description: Human-readable description
        default_value: Default value if not provided
        allowed_values: For enum types, list of allowed values
        validation_pattern: Regex pattern for string validation
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        min_length: Minimum length for strings/arrays
        max_length: Maximum length for strings/arrays
    """
    name: str
    type: PropertyType
    required: bool = False
    description: str = ""
    default_value: Any = None
    allowed_values: Optional[List[Any]] = None
    validation_pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this property definition."""
        if value is None:
            if self.required:
                return False, f"Required property '{self.name}' is missing"
            return True, None

        if self.type == PropertyType.STRING:
            if not isinstance(value, str):
                return False, f"Property '{self.name}' must be a string"
            if self.validation_pattern and not re.match(self.validation_pattern, value):
                return False, f"Property '{self.name}' does not match required pattern"
            if self.min_length and len(value) < self.min_length:
                return False, f"Property '{self.name}' length below minimum ({self.min_length})"
            if self.max_length and len(value) > self.max_length:
                return False, f"Property '{self.name}' length exceeds maximum ({self.max_length})"

        elif self.type == PropertyType.INTEGER:
            if not isinstance(value, int):
                return False, f"Property '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' below minimum value"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' exceeds maximum value"

        elif self.type == PropertyType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Property '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' below minimum value"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' exceeds maximum value"

        elif self.type == PropertyType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Property '{self.name}' must be a boolean"

        elif self.type in (PropertyType.ARRAY, PropertyType.LIST):
            if not isinstance(value, list):
                return False, f"Property '{self.name}' must be an array"
            if self.min_length and len(value) < self.min_length:
                return False, f"Property '{self.name}' length below minimum"
            if self.max_length and len(value) > self.max_length:
                return False, f"Property '{self.name}' length exceeds maximum"

        elif self.type in (PropertyType.OBJECT, PropertyType.DICT):
            if not isinstance(value, dict):
                return False, f"Property '{self.name}' must be an object"

        elif self.type == PropertyType.ENUM:
            if self.allowed_values and value not in self.allowed_values:
                return False, f"Property '{self.name}' value not in allowed values: {self.allowed_values}"

        elif self.type == PropertyType.EMBEDDING:
            if not isinstance(value, (list, tuple)):
                return False, f"Property '{self.name}' must be an embedding (list/tuple)"

        return True, None


# ============================================================================
# VALIDATION RULE
# ============================================================================

@dataclass
class ValidationRule:
    """
    A custom validation rule for an entity type.
    
    Attributes:
        name: Rule name
        description: Human-readable description
        validator: Function that takes an entity and returns (is_valid, error_message)
        severity: 'error', 'warning', or 'info'
    """
    name: str
    description: str
    validator: Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]
    severity: str = "error"


# ============================================================================
# UNIFIED KNOWLEDGE ARTIFACT (Consolidated from artifact_taxonomy.py and data/storage.py)
# ============================================================================

@dataclass
class KnowledgeArtifact:
    """
    Universal knowledge artifact structure.
    Consolidates versions from artifact_taxonomy.py and data/storage.py.
    Supports all 30+ artifact types with extensible metadata.
    
    Backward compatibility:
    - id field maps to artifact_id
    - content can be str or Dict
    - artifact_type can be ArtifactType enum or str
    """
    
    # Core Identification (artifact_taxonomy style with storage.py alias)
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: Union[ArtifactType, str] = ArtifactType.SOLUTION_PATTERN
    category: ArtifactCategory = ArtifactCategory.SOLUTION
    
    # Content (supports both str from storage.py and Dict from artifact_taxonomy.py)
    title: str = ""
    description: str = ""
    summary: str = ""
    content: Union[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Categorization
    domain: str = ""
    subdomain: str = ""
    problem_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Source & Provenance
    source_type: str = "extraction"
    source_id: str = ""
    source: str = ""  # Alias for source_type compatibility
    author: str = "system"
    organization: str = ""
    
    # Quality Metrics (with validation)
    confidence: float = 0.5
    quality_score: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0
    
    # Versioning
    version: Union[int, str] = 1
    parent_artifact_id: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)
    
    # Status & Lifecycle
    status: str = "draft"
    review_status: str = "pending"
    
    # Relationships
    related_artifacts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    
    # Context
    applicable_contexts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    # Performance Data
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps (UTC timezone-aware)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    
    # Validation
    validation_results: List[Dict] = field(default_factory=list)
    review_comments: List[Dict] = field(default_factory=list)
    
    # Search & Discovery
    search_vectors: Optional[List[float]] = None
    semantic_signature: Optional[str] = None
    embedding: Optional[List[float]] = None  # Alias for search_vectors
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Validate confidence is between 0.0 and 1.0
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        
        # Normalize timestamp to UTC
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
    
    @property
    def id(self) -> str:
        """Backward compatibility alias for artifact_id."""
        return self.artifact_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        artifact_type_value = self.artifact_type.value if isinstance(self.artifact_type, ArtifactType) else self.artifact_type
        
        return {
            "artifact_id": self.artifact_id,
            "id": self.artifact_id,  # Backward compatibility
            "artifact_type": artifact_type_value,
            "category": self.category.value if isinstance(self.category, ArtifactCategory) else self.category,
            "title": self.title,
            "description": self.description,
            "summary": self.summary,
            "content": self.content,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "problem_types": self.problem_types,
            "tags": self.tags,
            "keywords": self.keywords,
            "source_type": self.source_type,
            "source": self.source or self.source_type,
            "source_id": self.source_id,
            "author": self.author,
            "organization": self.organization,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "version": self.version,
            "parent_artifact_id": self.parent_artifact_id,
            "derived_from": self.derived_from,
            "status": self.status,
            "review_status": self.review_status,
            "related_artifacts": self.related_artifacts,
            "prerequisites": self.prerequisites,
            "supersedes": self.supersedes,
            "applicable_contexts": self.applicable_contexts,
            "constraints": self.constraints,
            "assumptions": self.assumptions,
            "performance_metrics": self.performance_metrics,
            "resource_requirements": self.resource_requirements,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            "last_used_at": self.last_used_at.isoformat() if isinstance(self.last_used_at, datetime) else self.last_used_at,
            "reviewed_at": self.reviewed_at.isoformat() if isinstance(self.reviewed_at, datetime) else self.reviewed_at,
            "validation_results": self.validation_results,
            "review_comments": self.review_comments,
            "search_vectors": self.search_vectors,
            "embedding": self.embedding or self.search_vectors,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """Create from dictionary with backward compatibility."""
        data = data.copy()
        
        # Handle id -> artifact_id mapping
        if "artifact_id" not in data and "id" in data:
            data["artifact_id"] = data["id"]
        
        # Parse timestamps
        for field_name in ["created_at", "updated_at", "last_used_at", "reviewed_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))
        
        # Parse enums
        if "artifact_type" in data and isinstance(data["artifact_type"], str):
            try:
                data["artifact_type"] = ArtifactType(data["artifact_type"])
            except ValueError:
                pass  # Keep as string if not a valid enum value
        
        if "category" in data and isinstance(data["category"], str):
            try:
                data["category"] = ArtifactCategory(data["category"])
            except ValueError:
                pass
        
        # Handle embedding/search_vectors alias
        if "search_vectors" not in data and "embedding" in data:
            data["search_vectors"] = data["embedding"]
        
        # Filter to only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        
        return cls(**valid_fields)
    
    def update_success(self, success: bool):
        """Update success rate with new application result."""
        self.usage_count += 1
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            current_successes = self.success_rate * (self.usage_count - 1)
            new_successes = current_successes + (1 if success else 0)
            self.success_rate = new_successes / self.usage_count
        self.updated_at = datetime.now(timezone.utc)
    
    def calculate_relevance_score(self, query_tags: set, query_domain: str) -> float:
        """Calculate relevance score for a given query."""
        score = 0.0
        
        tag_overlap = len(set(self.tags) & query_tags)
        score += tag_overlap * 0.2
        
        if self.domain == query_domain:
            score += 0.3
        
        score += self.confidence * 0.2
        score += self.success_rate * 0.2
        
        if self.usage_count > 10:
            score += 0.1
        
        return min(1.0, score)
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if artifact is applicable to given context."""
        for constraint in self.constraints:
            if not self._check_constraint(constraint, context):
                return False
        
        if "domain" in context and self.domain:
            if context["domain"] != self.domain and self.domain != "general":
                return False
        
        return True
    
    def _check_constraint(self, constraint: str, context: Dict[str, Any]) -> bool:
        """Check if a single constraint is satisfied."""
        if constraint.startswith("min_confidence:"):
            min_conf = float(constraint.split(":")[1])
            return context.get("confidence", 0) >= min_conf
        return True


# ============================================================================
# UNIFIED ENTITY (Consolidated from graph/models.py and core/entity_knowledge_graph.py)
# ============================================================================

@dataclass
class Entity:
    """
    Unified Entity model.
    Consolidates versions from graph/models.py (entity_id, entity_type, properties)
    and core/entity_knowledge_graph.py (name, entity_type, attributes).

    Backward compatibility:
    - name property maps to entity_id
    - attributes parameter maps to properties (for initialization)
    """
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: Union[EntityType, str] = EntityType.ENTITY
    name: str = ""  # Human-readable name
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Post-initialization normalization."""
        # Validate confidence
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        # If name not set but entity_id is, use entity_id as name
        if not self.name and self.entity_id:
            self.name = self.entity_id

    def __init__(
        self,
        entity_id: Optional[str] = None,
        entity_type: Union[EntityType, str] = EntityType.ENTITY,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        attributes: Optional[Dict[str, Any]] = None,  # Backward compatibility
        **kwargs  # Catch any other deprecated parameters
    ):
        """
        Initialize Entity with backward compatibility support.

        Args:
            entity_id: Unique entity identifier
            entity_type: Type/category of entity
            name: Human-readable name
            properties: Entity properties (new name for attributes)
            metadata: Additional metadata
            source: Source of entity
            confidence: Confidence score
            created_at: Creation timestamp
            updated_at: Last update timestamp
            attributes: DEPRECATED - use properties instead (mapped to properties for compatibility)
            **kwargs: Additional deprecated parameters
        """
        # Handle entity_id generation
        if entity_id is None:
            entity_id = str(uuid.uuid4())

        # Handle name defaults
        if name is None:
            name = entity_id

        # Handle attributes -> properties mapping for backward compatibility
        if attributes is not None:
            if properties is None:
                properties = attributes
            else:
                # Merge both, with properties taking precedence
                properties = {**attributes, **properties}

        # Set defaults
        if properties is None:
            properties = {}
        if metadata is None:
            metadata = {}
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        if updated_at is None:
            updated_at = datetime.now(timezone.utc)

        # Validate confidence
        confidence = max(0.0, min(1.0, float(confidence)))

        # Set all attributes
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.name = name
        self.properties = properties
        self.metadata = metadata
        self.source = source
        self.confidence = confidence
        self.created_at = created_at
        self.updated_at = updated_at
    
    @property
    def attributes(self) -> Dict[str, Any]:
        """Backward compatibility alias for properties."""
        return self.properties
    
    @attributes.setter
    def attributes(self, value: Dict[str, Any]):
        self.properties = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        entity_type_value = self.entity_type.value if isinstance(self.entity_type, Enum) else self.entity_type
        
        return {
            "entity_id": self.entity_id,
            "name": self.name or self.entity_id,
            "entity_type": entity_type_value,
            "properties": self.properties,
            "attributes": self.properties,  # Backward compatibility
            "metadata": self.metadata,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary with backward compatibility."""
        data = data.copy()

        # Handle name -> entity_id mapping
        if "entity_id" not in data and "name" in data:
            data["entity_id"] = data["name"]
        elif "name" not in data and "entity_id" in data:
            data["name"] = data["entity_id"]

        # Handle attributes -> properties mapping
        if "properties" not in data and "attributes" in data:
            data["properties"] = data["attributes"]
        elif "attributes" in data and "properties" in data:
            # Merge if both exist
            data["properties"] = {**data["attributes"], **data["properties"]}

        # Parse timestamps
        for field_name in ["created_at", "updated_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))

        # Parse entity_type enum
        if "entity_type" in data and isinstance(data["entity_type"], str):
            try:
                data["entity_type"] = EntityType(data["entity_type"])
            except ValueError:
                pass

        # Filter to only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# UNIFIED RELATIONSHIP (Consolidated from graph/models.py and core/entity_knowledge_graph.py)
# ============================================================================

@dataclass
class Relationship:
    """
    Unified Relationship model.
    Consolidates versions from:
    - graph/models.py (relationship_id, source_entity_id, target_entity_id, relationship_type)
    - core/entity_knowledge_graph.py (source, target, relation_type, attributes)
    - graph/models.py (edge_type variant)
    
    Backward compatibility:
    - source/target map to source_entity_id/target_entity_id
    - relation_type/edge_type map to relationship_type
    - attributes map to properties
    """
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: Union[RelationshipType, str] = RelationshipType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-initialization normalization."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
    
    @property
    def source(self) -> str:
        """Backward compatibility alias for source_entity_id."""
        return self.source_entity_id
    
    @source.setter
    def source(self, value: str):
        self.source_entity_id = value
    
    @property
    def target(self) -> str:
        """Backward compatibility alias for target_entity_id."""
        return self.target_entity_id
    
    @target.setter
    def target(self, value: str):
        self.target_entity_id = value
    
    @property
    def relation_type(self) -> str:
        """Backward compatibility alias for relationship_type."""
        return self.relationship_type.value if isinstance(self.relationship_type, Enum) else self.relationship_type
    
    @relation_type.setter
    def relation_type(self, value: str):
        self.relationship_type = value
    
    @property
    def edge_type(self) -> str:
        """Backward compatibility alias for relationship_type."""
        return self.relationship_type.value if isinstance(self.relationship_type, Enum) else self.relationship_type
    
    @edge_type.setter
    def edge_type(self, value: str):
        self.relationship_type = value
    
    @property
    def attributes(self) -> Dict[str, Any]:
        """Backward compatibility alias for properties."""
        return self.properties
    
    @attributes.setter
    def attributes(self, value: Dict[str, Any]):
        self.properties = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        rel_type_value = self.relationship_type.value if isinstance(self.relationship_type, Enum) else self.relationship_type
        
        return {
            "relationship_id": self.relationship_id,
            "id": self.relationship_id,  # Backward compatibility
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "source": self.source_entity_id,  # Backward compatibility
            "target": self.target_entity_id,  # Backward compatibility
            "relationship_type": rel_type_value,
            "relation_type": rel_type_value,  # Backward compatibility
            "edge_type": rel_type_value,  # Backward compatibility
            "properties": self.properties,
            "attributes": self.properties,  # Backward compatibility
            "metadata": self.metadata,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary with backward compatibility."""
        data = data.copy()
        
        # Handle various field name mappings
        if "relationship_id" not in data and "id" in data:
            data["relationship_id"] = data["id"]
        if not data.get("relationship_id"):
            data["relationship_id"] = str(uuid.uuid4())
        
        if "source_entity_id" not in data and "source" in data:
            data["source_entity_id"] = data["source"]
        if "target_entity_id" not in data and "target" in data:
            data["target_entity_id"] = data["target"]
        
        # Handle relationship_type variants
        rel_type = None
        for key in ["relationship_type", "relation_type", "edge_type", "type"]:
            if key in data:
                rel_type = data[key]
                break
        if rel_type:
            data["relationship_type"] = rel_type
        
        # Handle properties/attributes
        if "properties" not in data and "attributes" in data:
            data["properties"] = data["attributes"]
        
        # Parse timestamp
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        
        # Parse relationship_type enum
        if "relationship_type" in data and isinstance(data["relationship_type"], str):
            try:
                data["relationship_type"] = RelationshipType(data["relationship_type"])
            except ValueError:
                pass
        
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    def __hash__(self):
        """Make relationship hashable for deduplication."""
        return hash((self.source_entity_id, self.target_entity_id, 
                     self.relationship_type.value if isinstance(self.relationship_type, Enum) else self.relationship_type))
    
    def __eq__(self, other):
        """Check relationship equality."""
        if not isinstance(other, Relationship):
            return False
        self_type = self.relationship_type.value if isinstance(self.relationship_type, Enum) else self.relationship_type
        other_type = other.relationship_type.value if isinstance(other.relationship_type, Enum) else other.relationship_type
        return (self.source_entity_id == other.source_entity_id and
                self.target_entity_id == other.target_entity_id and
                self_type == other_type)


# ============================================================================
# UNIFIED VALIDATION RESULT (Consolidated from entity_schema_manager.py and sovereign_data_models.py)
# ============================================================================

@dataclass
class ValidationResult:
    """
    Unified ValidationResult.
    Consolidates versions from:
    - schemas/base.py (is_valid, errors, warnings, entity_id, schema_name, timestamp)
    - schemas/entity_schema_manager.py (entity_count, valid_count, invalid_count)
    - sovereign_data_models.py (validator, passed, score, feedback, improvements)
    
    All fields are optional for maximum flexibility across use cases.
    """
    # Core validation fields (from schemas/base.py)
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entity_id: Optional[str] = None
    schema_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregate fields (from entity_schema_manager.py)
    entity_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    
    # Sovereign data model fields
    validator: Optional[str] = None
    passed: Optional[bool] = None  # Alias for is_valid
    score: Optional[float] = None
    feedback: Optional[str] = None
    improvements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization normalization."""
        # Sync passed with is_valid
        if self.passed is None:
            self.passed = self.is_valid
        if self.is_valid and self.passed is not None:
            self.is_valid = self.passed
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        self.passed = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.improvements.extend(other.improvements)
        self.entity_count += other.entity_count
        self.valid_count += other.valid_count
        self.invalid_count += other.invalid_count
        if not other.is_valid:
            self.is_valid = False
            self.passed = False
        # Merge score (use other if we don't have one, or average them)
        if other.score is not None:
            if self.score is None:
                self.score = other.score
            else:
                # Average the scores weighted by entity count
                total = self.entity_count + other.entity_count
                if total > 0:
                    self.score = (self.score * self.entity_count + other.score * other.entity_count) / total
        # Merge feedback
        if other.feedback:
            if self.feedback:
                self.feedback = f"{self.feedback}; {other.feedback}"
            else:
                self.feedback = other.feedback
        # Merge validator info
        if other.validator and not self.validator:
            self.validator = other.validator
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'entity_id': self.entity_id,
            'schema_name': self.schema_name,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'entity_count': self.entity_count,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'validator': self.validator,
            'score': self.score,
            'feedback': self.feedback,
            'improvements': self.improvements,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary."""
        # Handle passed/is_valid alias
        is_valid = data.get('is_valid')
        passed = data.get('passed')
        if is_valid is None and passed is not None:
            is_valid = passed
        elif passed is None and is_valid is not None:
            passed = is_valid
        
        return cls(
            is_valid=is_valid if is_valid is not None else True,
            passed=passed if passed is not None else True,
            errors=data.get('errors', []),
            warnings=data.get('warnings', []),
            entity_id=data.get('entity_id'),
            schema_name=data.get('schema_name'),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            metadata=data.get('metadata', {}),
            entity_count=data.get('entity_count', 0),
            valid_count=data.get('valid_count', 0),
            invalid_count=data.get('invalid_count', 0),
            validator=data.get('validator'),
            score=data.get('score'),
            feedback=data.get('feedback'),
            improvements=data.get('improvements', []),
        )


# ============================================================================
# ENTITY TYPE AND RELATIONSHIP TYPE DEFINITIONS
# ============================================================================

@dataclass
class EntityTypeDefinition:
    """
    Defines an entity type in a schema.
    
    Attributes:
        name: Type name
        description: Human-readable description
        properties: Property definitions
        validation_rules: Custom validation rules
        examples: Example entities
        base_type: Optional parent type for inheritance
    """
    name: str
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    base_type: Optional[str] = None

    def get_all_properties(self) -> Dict[str, PropertyDefinition]:
        """Get all properties including inherited ones."""
        return self.properties.copy()

    def validate(self, entity_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate entity data against this type definition."""
        errors = []
        all_properties = self.get_all_properties()
        
        for prop_name, prop_def in all_properties.items():
            value = entity_data.get(prop_name)
            is_valid, error_msg = prop_def.validate(value)
            if not is_valid:
                errors.append(f"  - {error_msg}")
        
        for rule in self.validation_rules:
            is_valid, error_msg = rule.validator(entity_data)
            if not is_valid:
                errors.append(f"  - [{rule.severity.upper()}] {error_msg}")
        
        return len(errors) == 0, errors


@dataclass
class RelationshipTypeDefinition:
    """
    Defines a relationship type in a schema.
    
    Attributes:
        name: Relationship type name
        description: Human-readable description
        source_types: Allowed source entity types
        target_types: Allowed target entity types
        properties: Property definitions
        inverse_relationship: Optional inverse relationship name
        directed: Whether the relationship is directed
    """
    name: str
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    inverse_relationship: Optional[str] = None
    directed: bool = True
    description: str = ""

    def validate(
        self,
        relationship_data: Dict[str, Any],
        source_entity_type: str,
        target_entity_type: str
    ) -> tuple[bool, List[str]]:
        """Validate relationship data against this type definition."""
        errors = []
        
        if source_entity_type not in self.source_types:
            errors.append(f"Source entity type '{source_entity_type}' not in allowed types: {self.source_types}")
        
        if target_entity_type not in self.target_types:
            errors.append(f"Target entity type '{target_entity_type}' not in allowed types: {self.target_types}")
        
        for prop_name, prop_def in self.properties.items():
            value = relationship_data.get(prop_name)
            is_valid, error_msg = prop_def.validate(value)
            if not is_valid:
                errors.append(f"  - {error_msg}")
        
        return len(errors) == 0, errors


# ============================================================================
# ENTITY SCHEMA
# ============================================================================

@dataclass
class EntitySchema:
    """
    Complete schema definition for a domain.
    
    Attributes:
        domain: Domain name
        description: Schema description
        entity_types: Entity type definitions
        relationship_types: Relationship type definitions
        metadata: Additional metadata
        version: Schema version
    """
    domain: str
    entity_types: Dict[str, EntityTypeDefinition] = field(default_factory=dict)
    relationship_types: Dict[str, RelationshipTypeDefinition] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0.0"

    def get_entity_type(self, type_name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type by name."""
        return self.entity_types.get(type_name)

    def get_relationship_type(self, type_name: str) -> Optional[RelationshipTypeDefinition]:
        """Get relationship type by name."""
        return self.relationship_types.get(type_name)

    def list_entity_types(self) -> List[str]:
        """List all entity type names."""
        return list(self.entity_types.keys())

    def list_relationship_types(self) -> List[str]:
        """List all relationship type names."""
        return list(self.relationship_types.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "domain": self.domain,
            "description": self.description,
            "version": self.version,
            "entity_types": {
                name: {
                    "description": et.description,
                    "properties": {
                        pname: {
                            "type": pdef.type.value,
                            "required": pdef.required,
                            "description": pdef.description
                        }
                        for pname, pdef in et.properties.items()
                    }
                }
                for name, et in self.entity_types.items()
            },
            "relationship_types": {
                name: {
                    "description": rt.description,
                    "source_types": rt.source_types,
                    "target_types": rt.target_types,
                    "directed": rt.directed
                }
                for name, rt in self.relationship_types.items()
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntitySchema':
        """Create schema from dictionary representation."""
        schema = cls(
            domain=data["domain"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )
        
        for name, et_data in data.get("entity_types", {}).items():
            properties = {}
            for pname, pdata in et_data.get("properties", {}).items():
                prop_type = PropertyType(pdata["type"])
                properties[pname] = PropertyDefinition(
                    name=pname,
                    type=prop_type,
                    required=pdata.get("required", False),
                    description=pdata.get("description", "")
                )
            
            schema.entity_types[name] = EntityTypeDefinition(
                name=name,
                properties=properties,
                description=et_data.get("description", "")
            )
        
        for name, rt_data in data.get("relationship_types", {}).items():
            schema.relationship_types[name] = RelationshipTypeDefinition(
                name=name,
                source_types=rt_data.get("source_types", []),
                target_types=rt_data.get("target_types", []),
                directed=rt_data.get("directed", True),
                description=rt_data.get("description", "")
            )
        
        return schema


# ============================================================================
# ARTIFACT TAXONOMY MANAGER
# ============================================================================

class ArtifactTaxonomy:
    """
    Manager for the ultra-comprehensive artifact taxonomy.
    Provides utilities for categorization, validation, and discovery.
    """
    
    def __init__(self):
        self.type_to_category = self._build_type_mapping()
        self.category_types = self._build_category_index()
    
    def _build_type_mapping(self) -> Dict[ArtifactType, ArtifactCategory]:
        """Build mapping from artifact type to category."""
        return {
            # Solution patterns
            ArtifactType.SOLUTION_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.CODE_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.ARCHITECTURE_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.INTEGRATION_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.DEPLOYMENT_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.SCALING_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.MIGRATION_PATTERN: ArtifactCategory.SOLUTION,
            ArtifactType.RECOVERY_PATTERN: ArtifactCategory.SOLUTION,
            
            # Anti-patterns
            ArtifactType.ANTI_PATTERN: ArtifactCategory.ANTI_PATTERN,
            ArtifactType.SECURITY_ANTI_PATTERN: ArtifactCategory.ANTI_PATTERN,
            ArtifactType.PERFORMANCE_ANTI_PATTERN: ArtifactCategory.ANTI_PATTERN,
            ArtifactType.DESIGN_ANTI_PATTERN: ArtifactCategory.ANTI_PATTERN,
            ArtifactType.OPERATIONAL_ANTI_PATTERN: ArtifactCategory.ANTI_PATTERN,
            
            # Process artifacts
            ArtifactType.DECOMPOSITION_STRATEGY: ArtifactCategory.PROCESS,
            ArtifactType.WORKFLOW_TEMPLATE: ArtifactCategory.PROCESS,
            ArtifactType.DECISION_FRAMEWORK: ArtifactCategory.PROCESS,
            ArtifactType.COMMUNICATION_TEMPLATE: ArtifactCategory.PROCESS,
            ArtifactType.COLLABORATION_PATTERN: ArtifactCategory.PROCESS,
            ArtifactType.FACILITATION_GUIDE: ArtifactCategory.PROCESS,
            
            # Domain knowledge
            ArtifactType.DOMAIN_KNOWLEDGE: ArtifactCategory.DOMAIN,
            ArtifactType.REGULATORY_GUIDANCE: ArtifactCategory.DOMAIN,
            ArtifactType.INDUSTRY_STANDARD: ArtifactCategory.DOMAIN,
            ArtifactType.TECHNOLOGY_GUIDE: ArtifactCategory.DOMAIN,
            
            # Performance
            ArtifactType.PERFORMANCE_BENCHMARK: ArtifactCategory.PERFORMANCE,
            ArtifactType.OPTIMIZATION_RECORD: ArtifactCategory.PERFORMANCE,
            ArtifactType.CAPACITY_PLANNING_GUIDE: ArtifactCategory.PERFORMANCE,
            ArtifactType.COST_OPTIMIZATION: ArtifactCategory.PERFORMANCE,
            ArtifactType.RESOURCE_ESTIMATION: ArtifactCategory.PERFORMANCE,
            
            # Team
            ArtifactType.TEAM_PERFORMANCE_DATA: ArtifactCategory.TEAM,
            ArtifactType.SKILL_MATRIX: ArtifactCategory.TEAM,
            ArtifactType.ASSIGNMENT_STRATEGY: ArtifactCategory.TEAM,
            ArtifactType.ONBOARDING_GUIDE: ArtifactCategory.TEAM,
            ArtifactType.RETENTION_PATTERN: ArtifactCategory.TEAM,
            
            # System
            ArtifactType.SYSTEM_BLUEPRINT: ArtifactCategory.SYSTEM,
            ArtifactType.COMPONENT_SPECIFICATION: ArtifactCategory.SYSTEM,
            ArtifactType.API_DESIGN_PATTERN: ArtifactCategory.SYSTEM,
            ArtifactType.DATA_MODEL: ArtifactCategory.SYSTEM,
            ArtifactType.CONFIGURATION_TEMPLATE: ArtifactCategory.SYSTEM,
            
            # Quality
            ArtifactType.QUALITY_CRITERIA: ArtifactCategory.QUALITY,
            ArtifactType.TESTING_STRATEGY: ArtifactCategory.QUALITY,
            ArtifactType.VALIDATION_RULESET: ArtifactCategory.QUALITY,
            ArtifactType.GAUNTLET_EFFECTIVENESS: ArtifactCategory.QUALITY,
            ArtifactType.DEFECT_PATTERN: ArtifactCategory.QUALITY,
            ArtifactType.REVIEW_CHECKLIST: ArtifactCategory.QUALITY,
            
            # Learning
            ArtifactType.LEARNING_PATH: ArtifactCategory.LEARNING,
            ArtifactType.TUTORIAL_TEMPLATE: ArtifactCategory.LEARNING,
            ArtifactType.EXPLANATION_PATTERN: ArtifactCategory.LEARNING,
            ArtifactType.KNOWLEDGE_TRANSFER_GUIDE: ArtifactCategory.LEARNING,
            ArtifactType.MENTORING_FRAMEWORK: ArtifactCategory.LEARNING,
            
            # Operational
            ArtifactType.INCIDENT_RESPONSE_PLAYBOOK: ArtifactCategory.OPERATIONAL,
            ArtifactType.MONITORING_CONFIGURATION: ArtifactCategory.OPERATIONAL,
            ArtifactType.MAINTENANCE_PROCEDURE: ArtifactCategory.OPERATIONAL,
            ArtifactType.TROUBLESHOOTING_GUIDE: ArtifactCategory.OPERATIONAL,
            ArtifactType.ROLLBACK_PROCEDURE: ArtifactCategory.OPERATIONAL,
            ArtifactType.SECURITY_PROCEDURE: ArtifactCategory.OPERATIONAL,
        }
    
    def _build_category_index(self) -> Dict[ArtifactCategory, List[ArtifactType]]:
        """Build index of types by category."""
        index = {cat: [] for cat in ArtifactCategory}
        for art_type, cat in self.type_to_category.items():
            index[cat].append(art_type)
        return index
    
    def get_category(self, art_type: ArtifactType) -> ArtifactCategory:
        """Get category for artifact type."""
        return self.type_to_category.get(art_type, ArtifactCategory.SOLUTION)
    
    def get_types_in_category(self, category: ArtifactCategory) -> List[ArtifactType]:
        """Get all artifact types in a category."""
        return self.category_types.get(category, [])
    
    def get_type_description(self, art_type: ArtifactType) -> str:
        """Get description for artifact type."""
        return art_type.__doc__ or "No description available"
    
    def get_all_types(self) -> List[ArtifactType]:
        """Get all artifact types."""
        return list(ArtifactType)
    
    def validate_artifact(self, artifact: KnowledgeArtifact) -> List[str]:
        """Validate artifact against taxonomy."""
        errors = []
        
        art_type = artifact.artifact_type
        if isinstance(art_type, str):
            try:
                art_type = ArtifactType(art_type)
            except ValueError:
                errors.append(f"Invalid artifact type: {art_type}")
                return errors
        
        expected_category = self.get_category(art_type)
        if artifact.category != expected_category:
            errors.append(f"Category mismatch: {artifact.category} != {expected_category}")
        
        if not artifact.title:
            errors.append("Title is required")
        
        if artifact.confidence < 0 or artifact.confidence > 1:
            errors.append("Confidence must be between 0 and 1")
        
        return errors
    
    def suggest_types_for_problem(self, problem_description: str) -> List[ArtifactType]:
        """Suggest artifact types based on problem description."""
        description_lower = problem_description.lower()
        suggestions = []
        
        keywords = {
            ArtifactType.SOLUTION_PATTERN: ["solution", "solve", "approach", "how to"],
            ArtifactType.CODE_PATTERN: ["code", "function", "class", "implementation"],
            ArtifactType.ARCHITECTURE_PATTERN: ["architecture", "design", "system"],
            ArtifactType.DECOMPOSITION_STRATEGY: ["break down", "decompose", "split"],
            ArtifactType.ANTI_PATTERN: ["avoid", "don't", "mistake", "wrong"],
            ArtifactType.WORKFLOW_TEMPLATE: ["workflow", "process", "procedure"],
            ArtifactType.DECISION_FRAMEWORK: ["decide", "choose", "select", "criteria"],
            ArtifactType.DOMAIN_KNOWLEDGE: ["domain", "field", "industry"],
            ArtifactType.PERFORMANCE_BENCHMARK: ["performance", "speed", "latency"],
            ArtifactType.OPTIMIZATION_RECORD: ["optimize", "improve", "faster"],
            ArtifactType.TEAM_PERFORMANCE_DATA: ["team", "performance", "assignment"],
            ArtifactType.QUALITY_CRITERIA: ["quality", "validate", "test"],
            ArtifactType.TESTING_STRATEGY: ["test", "verify", "validate"],
            ArtifactType.INCIDENT_RESPONSE_PLAYBOOK: ["incident", "outage", "emergency"],
            ArtifactType.TROUBLESHOOTING_GUIDE: ["troubleshoot", "debug", "fix"],
            ArtifactType.CREATIVE_PATTERN: ["creative", "write", "story", "content"],
            ArtifactType.PROMPT_PATTERN: ["prompt", "llm", "ai"],
        }
        
        for art_type, type_keywords in keywords.items():
            if any(kw in description_lower for kw in type_keywords):
                suggestions.append(art_type)
        
        return suggestions[:5]


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Enums
    "PropertyType",
    "ArtifactType",
    "ArtifactCategory",
    "EntityType",
    "RelationshipType",
    "TOTAL_ARTIFACT_TYPES",
    
    # Core classes
    "PropertyDefinition",
    "ValidationRule",
    "KnowledgeArtifact",
    "Entity",
    "Relationship",
    "ValidationResult",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "EntitySchema",
    "ArtifactTaxonomy",
]
