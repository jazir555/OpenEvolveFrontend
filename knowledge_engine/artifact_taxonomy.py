"""
Ultra-Comprehensive Knowledge Artifact Taxonomy

Defines 30+ artifact types covering all aspects of problem-solving,
system design, team dynamics, and operational excellence.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid


class ArtifactCategory(Enum):
    """High-level artifact categories."""
    SOLUTION = "solution"           # Things that work
    ANTI_PATTERN = "anti_pattern"   # Things to avoid
    PROCESS = "process"             # How to do things
    DOMAIN = "domain"               # Field knowledge
    PERFORMANCE = "performance"     # Metrics and benchmarks
    TEAM = "team"                   # Team dynamics
    SYSTEM = "system"               # System architecture
    QUALITY = "quality"             # Quality assurance
    LEARNING = "learning"           # Knowledge acquisition
    OPERATIONAL = "operational"     # Operations and maintenance


class ArtifactType(Enum):
    """
    Ultra-comprehensive artifact type taxonomy.
    30+ distinct artifact types for complete coverage.
    """
    
    # === SOLUTION PATTERNS (8 types) ===
    SOLUTION_PATTERN = "solution_pattern"
    """Reusable solution approaches for specific problem types."""
    
    CODE_PATTERN = "code_pattern"
    """Reusable code structures, algorithms, or implementations."""
    
    ARCHITECTURE_PATTERN = "architecture_pattern"
    """System architecture templates and design patterns."""
    
    INTEGRATION_PATTERN = "integration_pattern"
    """System integration approaches and connection strategies."""
    
    DEPLOYMENT_PATTERN = "deployment_pattern"
    """Deployment strategies and release patterns."""
    
    SCALING_PATTERN = "scaling_pattern"
    """Horizontal/vertical scaling strategies for different loads."""
    
    MIGRATION_PATTERN = "migration_pattern"
    """Data/system migration approaches and strategies."""
    
    RECOVERY_PATTERN = "recovery_pattern"
    """Disaster recovery and business continuity patterns."""
    
    # === ANTI-PATTERNS (5 types) ===
    ANTI_PATTERN = "anti_pattern"
    """Common approaches that consistently fail."""
    
    SECURITY_ANTI_PATTERN = "security_anti_pattern"
    """Security mistakes and vulnerable approaches to avoid."""
    
    PERFORMANCE_ANTI_PATTERN = "performance_anti_pattern"
    """Performance-killing approaches and bottlenecks."""
    
    DESIGN_ANTI_PATTERN = "design_anti_pattern"
    """Poor design choices and architectural mistakes."""
    
    OPERATIONAL_ANTI_PATTERN = "operational_anti_pattern"
    """Operational mistakes and process failures."""
    
    # === PROCESS ARTIFACTS (6 types) ===
    DECOMPOSITION_STRATEGY = "decomposition_strategy"
    """Problem breakdown strategies and approaches."""
    
    WORKFLOW_TEMPLATE = "workflow_template"
    """Reusable workflow structures and process templates."""
    
    DECISION_FRAMEWORK = "decision_framework"
    """Structured decision-making approaches and criteria."""
    
    COMMUNICATION_TEMPLATE = "communication_template"
    """Effective communication patterns and templates."""
    
    COLLABORATION_PATTERN = "collaboration_pattern"
    """Team collaboration and coordination strategies."""
    
    FACILITATION_GUIDE = "facilitation_guide"
    """Meeting/workshop facilitation techniques."""
    
    # === DOMAIN KNOWLEDGE (4 types) ===
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    """Field-specific insights, constraints, and best practices."""
    
    REGULATORY_GUIDANCE = "regulatory_guidance"
    """Compliance requirements and regulatory considerations."""
    
    INDUSTRY_STANDARD = "industry_standard"
    """Industry standards and common practices."""
    
    TECHNOLOGY_GUIDE = "technology_guide"
    """Technology-specific knowledge and capabilities."""
    
    # === PERFORMANCE ARTIFACTS (5 types) ===
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    """Performance metrics and baseline measurements."""
    
    OPTIMIZATION_RECORD = "optimization_record"
    """Documented optimizations with before/after metrics."""
    
    CAPACITY_PLANNING_GUIDE = "capacity_planning_guide"
    """Capacity planning strategies and thresholds."""
    
    COST_OPTIMIZATION = "cost_optimization"
    """Cost reduction strategies and efficiency improvements."""
    
    RESOURCE_ESTIMATION = "resource_estimation"
    """Resource requirement estimation heuristics."""
    
    # === TEAM ARTIFACTS (5 types) ===
    TEAM_PERFORMANCE_DATA = "team_performance_data"
    """Team effectiveness metrics and performance profiles."""
    
    SKILL_MATRIX = "skill_matrix"
    """Skill requirements and competency mappings."""
    
    ASSIGNMENT_STRATEGY = "assignment_strategy"
    """Task-to-team assignment strategies and heuristics."""
    
    ONBOARDING_GUIDE = "onboarding_guide"
    """Team member onboarding and ramp-up strategies."""
    
    RETENTION_PATTERN = "retention_pattern"
    """Knowledge retention and succession planning."""
    
    # === SYSTEM ARTIFACTS (5 types) ===
    SYSTEM_BLUEPRINT = "system_blueprint"
    """System architecture documentation and diagrams."""
    
    COMPONENT_SPECIFICATION = "component_specification"
    """Component interfaces and specifications."""
    
    API_DESIGN_PATTERN = "api_design_pattern"
    """API design best practices and patterns."""
    
    DATA_MODEL = "data_model"
    """Data models and schema patterns."""
    
    CONFIGURATION_TEMPLATE = "configuration_template"
    """System configuration templates and presets."""
    
    # === QUALITY ARTIFACTS (6 types) ===
    QUALITY_CRITERIA = "quality_criteria"
    """Output quality validation criteria and thresholds."""
    
    TESTING_STRATEGY = "testing_strategy"
    """Testing approaches and coverage strategies."""
    
    VALIDATION_RULESET = "validation_ruleset"
    """Validation rules and acceptance criteria."""
    
    GAUNTLET_EFFECTIVENESS = "gauntlet_effectiveness"
    """Quality gate effectiveness and catch rates."""
    
    DEFECT_PATTERN = "defect_pattern"
    """Common defect types and prevention strategies."""
    
    REVIEW_CHECKLIST = "review_checklist"
    """Code/design review checklists and guidelines."""
    
    # === LEARNING ARTIFACTS (5 types) ===
    LEARNING_PATH = "learning_path"
    """Structured learning progression for skills/topics."""
    
    TUTORIAL_TEMPLATE = "tutorial_template"
    """Educational content templates and structures."""
    
    EXPLANATION_PATTERN = "explanation_pattern"
    """Effective explanation techniques and approaches."""
    
    KNOWLEDGE_TRANSFER_GUIDE = "knowledge_transfer_guide"
    """Knowledge sharing and transfer strategies."""
    
    MENTORING_FRAMEWORK = "mentoring_framework"
    """Mentoring and coaching approaches."""
    
    # === OPERATIONAL ARTIFACTS (6 types) ===
    INCIDENT_RESPONSE_PLAYBOOK = "incident_response_playbook"
    """Incident response procedures and runbooks."""
    
    MONITORING_CONFIGURATION = "monitoring_configuration"
    """Monitoring setup and alerting rules."""
    
    MAINTENANCE_PROCEDURE = "maintenance_procedure"
    """Routine maintenance tasks and procedures."""
    
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    """Problem diagnosis and resolution guides."""
    
    ROLLBACK_PROCEDURE = "rollback_procedure"
    """Rollback and recovery procedures."""
    
    SECURITY_PROCEDURE = "security_procedure"
    """Security operations and hardening procedures."""
    
    # === SPECIALIZED ARTIFACTS (5 types) ===
    CREATIVE_PATTERN = "creative_pattern"
    """Creative writing and content generation patterns."""
    
    PROMPT_PATTERN = "prompt_pattern"
    """Effective LLM prompting techniques and templates."""
    
    ESTIMATION_HEURISTIC = "estimation_heuristic"
    """Effort and impact estimation techniques."""
    
    RISK_PATTERN = "risk_pattern"
    """Risk identification and mitigation patterns."""
    
    DEPENDENCY_MAP = "dependency_map"
    """Dependency relationships and management strategies."""


# Total artifact types count
TOTAL_ARTIFACT_TYPES = len(ArtifactType)


@dataclass
class KnowledgeArtifact:
    """
    Universal knowledge artifact structure.
    Supports all 30+ artifact types with extensible metadata.
    """
    
    # Core Identification
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = ArtifactType.SOLUTION_PATTERN
    category: ArtifactCategory = ArtifactCategory.SOLUTION
    
    # Content
    title: str = ""
    description: str = ""
    summary: str = ""  # Brief executive summary
    
    # Categorization
    domain: str = ""  # Business domain (e.g., "fintech", "healthcare")
    subdomain: str = ""  # Specific subdomain
    problem_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Source & Provenance
    source_type: str = "extraction"  # extraction, manual, import
    source_id: str = ""  # Workflow ID, problem ID, etc.
    author: str = "system"
    organization: str = ""
    
    # Quality Metrics
    confidence: float = 0.5  # 0.0 - 1.0
    quality_score: float = 0.0  # Validation score
    success_rate: float = 0.0  # Historical success rate
    usage_count: int = 0  # Number of times applied
    
    # Versioning
    version: int = 1
    parent_artifact_id: Optional[str] = None  # For evolved artifacts
    derived_from: List[str] = field(default_factory=list)
    
    # Status & Lifecycle
    status: str = "draft"  # draft, reviewed, approved, deprecated, archived
    review_status: str = "pending"  # pending, in_review, approved, rejected
    
    # Content Payload (type-specific)
    content: Dict[str, Any] = field(default_factory=dict)
    
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
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    
    # Validation
    validation_results: List[Dict] = field(default_factory=list)
    review_comments: List[Dict] = field(default_factory=list)
    
    # Search & Discovery
    search_vectors: Optional[List[float]] = None  # Embedding vectors
    semantic_signature: Optional[str] = None  # Semantic hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "summary": self.summary,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "problem_types": self.problem_types,
            "tags": self.tags,
            "keywords": self.keywords,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "author": self.author,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "version": self.version,
            "status": self.status,
            "content": self.content,
            "related_artifacts": self.related_artifacts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """Create from dictionary."""
        # Parse timestamps
        for field_name in ["created_at", "updated_at", "last_used_at", "reviewed_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))
        
        # Parse enums
        if "artifact_type" in data:
            data["artifact_type"] = ArtifactType(data["artifact_type"])
        if "category" in data:
            data["category"] = ArtifactCategory(data["category"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def update_success(self, success: bool):
        """Update success rate with new application result."""
        self.usage_count += 1
        # Bayesian update of success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            current_successes = self.success_rate * (self.usage_count - 1)
            new_successes = current_successes + (1 if success else 0)
            self.success_rate = new_successes / self.usage_count
        self.updated_at = datetime.utcnow()
    
    def calculate_relevance_score(self, query_tags: Set[str], query_domain: str) -> float:
        """Calculate relevance score for a given query."""
        score = 0.0
        
        # Tag overlap
        tag_overlap = len(set(self.tags) & query_tags)
        score += tag_overlap * 0.2
        
        # Domain match
        if self.domain == query_domain:
            score += 0.3
        
        # Quality metrics
        score += self.confidence * 0.2
        score += self.success_rate * 0.2
        
        # Usage popularity
        if self.usage_count > 10:
            score += 0.1
        
        return min(1.0, score)
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if artifact is applicable to given context."""
        # Check constraints
        for constraint in self.constraints:
            if not self._check_constraint(constraint, context):
                return False
        
        # Check domain compatibility
        if "domain" in context and self.domain:
            if context["domain"] != self.domain and self.domain != "general":
                return False
        
        return True
    
    def _check_constraint(self, constraint: str, context: Dict[str, Any]) -> bool:
        """Check if a single constraint is satisfied."""
        # Simple constraint checking - can be extended
        if constraint.startswith("min_confidence:"):
            min_conf = float(constraint.split(":")[1])
            return context.get("confidence", 0) >= min_conf
        return True


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
        
        if artifact.artifact_type not in ArtifactType:
            errors.append(f"Invalid artifact type: {artifact.artifact_type}")
        
        expected_category = self.get_category(artifact.artifact_type)
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
        
        # Keywords to type mapping
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
        
        return suggestions[:5]  # Return top 5


# Export key components
__all__ = [
    "ArtifactType",
    "ArtifactCategory",
    "KnowledgeArtifact",
    "ArtifactTaxonomy",
    "TOTAL_ARTIFACT_TYPES",
]
