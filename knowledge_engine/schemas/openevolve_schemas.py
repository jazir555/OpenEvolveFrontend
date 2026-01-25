"""
OpenEvolve-Specific Entity Schemas

Defines schema definitions for OpenEvolve domains:
- Software Engineering
- Mathematical Reasoning
- Workflow/Provenance
"""

from .base import (
    EntitySchema,
    EntityType,
    RelationshipType,
    PropertyDefinition,
    PropertyType,
    ValidationRule
)


# =============================================================================
# SOFTWARE ENGINEERING SCHEMA
# =============================================================================

SOFTWARE_ENGINEERING_SCHEMA = EntitySchema(
    domain="software_engineering",
    description="Schema for software engineering entities including code, dependencies, APIs, and bug patterns",
    version="1.0.0",
    metadata={
        "author": "OpenEvolve",
        "created": "2025-01-07",
        "use_cases": ["code analysis", "dependency tracking", "API documentation", "bug detection"]
    }
)

# CodeEntity - Represents code elements (functions, classes, modules, packages)
SOFTWARE_ENGINEERING_SCHEMA.entity_types["CodeEntity"] = EntityType(
    name="CodeEntity",
    description="Represents a code element such as function, class, module, or package",
    properties={
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Name of the code element",
            validation_pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        ),
        "code_type": PropertyDefinition(
            name="code_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of code element",
            allowed_values=["function", "class", "method", "module", "package", "interface"]
        ),
        "signature": PropertyDefinition(
            name="signature",
            type=PropertyType.STRING,
            required=False,
            description="Function or method signature including parameters"
        ),
        "file_path": PropertyDefinition(
            name="file_path",
            type=PropertyType.STRING,
            required=False,
            description="Path to the file containing this code element"
        ),
        "line_start": PropertyDefinition(
            name="line_start",
            type=PropertyType.INTEGER,
            required=False,
            description="Starting line number",
            min_value=1
        ),
        "line_end": PropertyDefinition(
            name="line_end",
            type=PropertyType.INTEGER,
            required=False,
            description="Ending line number",
            min_value=1
        ),
        "language": PropertyDefinition(
            name="language",
            type=PropertyType.STRING,
            required=False,
            description="Programming language",
            allowed_values=["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "Other"]
        ),
        "complexity": PropertyDefinition(
            name="complexity",
            type=PropertyType.INTEGER,
            required=False,
            description="Cyclomatic complexity",
            min_value=1
        ),
        "documentation": PropertyDefinition(
            name="documentation",
            type=PropertyType.STRING,
            required=False,
            description="Docstring or comment documentation"
        )
    },
    examples=[
        {
            "name": "calculate_hash",
            "code_type": "function",
            "signature": "calculate_hash(data: bytes, algorithm: str) -> str",
            "file_path": "src/utils/crypto.py",
            "line_start": 42,
            "line_end": 58,
            "language": "Python",
            "complexity": 3
        },
        {
            "name": "UserService",
            "code_type": "class",
            "file_path": "src/services/user_service.py",
            "language": "Python"
        }
    ]
)

# DependencyEntity - Represents dependencies between code elements
SOFTWARE_ENGINEERING_SCHEMA.entity_types["DependencyEntity"] = EntityType(
    name="DependencyEntity",
    description="Represents import or dependency relationships between code elements",
    properties={
        "import_type": PropertyDefinition(
            name="import_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of import/dependency",
            allowed_values=["direct", "indirect", "dynamic", "conditional"]
        ),
        "import_statement": PropertyDefinition(
            name="import_statement",
            type=PropertyType.STRING,
            required=False,
            description="Actual import statement"
        ),
        "is_external": PropertyDefinition(
            name="is_external",
            type=PropertyType.BOOLEAN,
            required=False,
            description="Whether the dependency is external to the project"
        ),
        "version": PropertyDefinition(
            name="version",
            type=PropertyType.STRING,
            required=False,
            description="Version constraint for external dependencies"
        )
    },
    examples=[
        {
            "import_type": "direct",
            "import_statement": "from typing import List, Dict",
            "is_external": False
        }
    ]
)

# APISchema - Represents API endpoints and their specifications
SOFTWARE_ENGINEERING_SCHEMA.entity_types["APISchema"] = EntityType(
    name="APISchema",
    description="Represents an API endpoint or interface",
    properties={
        "endpoint": PropertyDefinition(
            name="endpoint",
            type=PropertyType.STRING,
            required=True,
            description="API endpoint path"
        ),
        "method": PropertyDefinition(
            name="method",
            type=PropertyType.ENUM,
            required=True,
            description="HTTP method",
            allowed_values=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        ),
        "parameters": PropertyDefinition(
            name="parameters",
            type=PropertyType.ARRAY,
            required=False,
            description="List of parameter definitions"
        ),
        "response_type": PropertyDefinition(
            name="response_type",
            type=PropertyType.STRING,
            required=False,
            description="Expected response type or schema"
        ),
        "authentication": PropertyDefinition(
            name="authentication",
            type=PropertyType.BOOLEAN,
            required=False,
            description="Whether authentication is required"
        ),
        "rate_limit": PropertyDefinition(
            name="rate_limit",
            type=PropertyType.INTEGER,
            required=False,
            description="Rate limit in requests per minute"
        )
    },
    examples=[
        {
            "endpoint": "/api/users/{user_id}",
            "method": "GET",
            "response_type": "User",
            "authentication": True,
            "rate_limit": 60
        }
    ]
)

# BugPattern - Represents known bug patterns and their fixes
SOFTWARE_ENGINEERING_SCHEMA.entity_types["BugPattern"] = EntityType(
    name="BugPattern",
    description="Represents a known bug pattern with symptoms, causes, and fixes",
    properties={
        "pattern_name": PropertyDefinition(
            name="pattern_name",
            type=PropertyType.STRING,
            required=True,
            description="Name of the bug pattern"
        ),
        "symptom": PropertyDefinition(
            name="symptom",
            type=PropertyType.STRING,
            required=True,
            description="Observable symptoms of the bug"
        ),
        "root_cause": PropertyDefinition(
            name="root_cause",
            type=PropertyType.STRING,
            required=True,
            description="Underlying cause of the bug"
        ),
        "fix": PropertyDefinition(
            name="fix",
            type=PropertyType.STRING,
            required=True,
            description="Recommended fix or solution"
        ),
        "severity": PropertyDefinition(
            name="severity",
            type=PropertyType.ENUM,
            required=False,
            description="Bug severity level",
            allowed_values=["critical", "high", "medium", "low"]
        ),
        "language_context": PropertyDefinition(
            name="language_context",
            type=PropertyType.STRING,
            required=False,
            description="Programming language context"
        ),
        "code_example": PropertyDefinition(
            name="code_example",
            type=PropertyType.STRING,
            required=False,
            description="Example code exhibiting the bug"
        )
    },
    examples=[
        {
            "pattern_name": "Off-by-one error",
            "symptom": "Index out of bounds or incorrect iteration count",
            "root_cause": "Incorrect loop boundary or array indexing",
            "fix": "Adjust loop bounds or use inclusive/exclusive ranges consistently",
            "severity": "high"
        }
    ]
)

# Software Engineering Relationships
SOFTWARE_ENGINEERING_SCHEMA.relationship_types["calls"] = RelationshipType(
    name="calls",
    description="Function or method call relationship",
    source_types=["CodeEntity"],
    target_types=["CodeEntity"],
    directed=True
)

SOFTWARE_ENGINEERING_SCHEMA.relationship_types["imports"] = RelationshipType(
    name="imports",
    description="Import or dependency relationship",
    source_types=["CodeEntity"],
    target_types=["CodeEntity", "DependencyEntity"],
    directed=True
)

SOFTWARE_ENGINEERING_SCHEMA.relationship_types["implements"] = RelationshipType(
    name="implements",
    description="Implementation relationship (class implements interface)",
    source_types=["CodeEntity"],
    target_types=["CodeEntity"],
    directed=True
)

SOFTWARE_ENGINEERING_SCHEMA.relationship_types["exposes"] = RelationshipType(
    name="exposes",
    description="API exposure relationship",
    source_types=["CodeEntity"],
    target_types=["APISchema"],
    directed=True
)

SOFTWARE_ENGINEERING_SCHEMA.relationship_types["has_bug_pattern"] = RelationshipType(
    name="has_bug_pattern",
    description="Code element exhibits a bug pattern",
    source_types=["CodeEntity"],
    target_types=["BugPattern"],
    directed=True
)


# =============================================================================
# MATHEMATICAL REASONING SCHEMA
# =============================================================================

MATHEMATICAL_REASONING_SCHEMA = EntitySchema(
    domain="mathematical_reasoning",
    description="Schema for mathematical reasoning entities including theorems, concepts, techniques, and proof steps",
    version="1.0.0",
    metadata={
        "author": "OpenEvolve",
        "created": "2025-01-07",
        "use_cases": ["formal verification", "proof assistants", "mathematical knowledge representation"]
    }
)

# TheoremEntity - Represents mathematical theorems and propositions
MATHEMATICAL_REASONING_SCHEMA.entity_types["TheoremEntity"] = EntityType(
    name="TheoremEntity",
    description="Represents a mathematical theorem, lemma, or proposition",
    properties={
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Name of the theorem"
        ),
        "statement": PropertyDefinition(
            name="statement",
            type=PropertyType.STRING,
            required=True,
            description="Formal statement of the theorem"
        ),
        "theorem_type": PropertyDefinition(
            name="theorem_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of theorem",
            allowed_values=["theorem", "lemma", "proposition", "corollary", "axiom", "conjecture"]
        ),
        "proof": PropertyDefinition(
            name="proof",
            type=PropertyType.STRING,
            required=False,
            description="Proof or sketch of the theorem"
        ),
        "proof_status": PropertyDefinition(
            name="proof_status",
            type=PropertyType.ENUM,
            required=False,
            description="Status of the proof",
            allowed_values=["proven", "disproven", "unproven", "conjectured"]
        ),
        "dependencies": PropertyDefinition(
            name="dependencies",
            type=PropertyType.ARRAY,
            required=False,
            description="Theorems or concepts this theorem depends on"
        ),
        "domain": PropertyDefinition(
            name="domain",
            type=PropertyType.STRING,
            required=False,
            description="Mathematical domain (e.g., algebra, topology)"
        )
    },
    examples=[
        {
            "name": "Pythagorean Theorem",
            "theorem_type": "theorem",
            "statement": "In a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides",
            "proof_status": "proven",
            "domain": "geometry"
        }
    ]
)

# ConceptEntity - Represents mathematical concepts and definitions
MATHEMATICAL_REASONING_SCHEMA.entity_types["ConceptEntity"] = EntityType(
    name="ConceptEntity",
    description="Represents a mathematical concept, definition, or term",
    properties={
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Name of the concept"
        ),
        "definition": PropertyDefinition(
            name="definition",
            type=PropertyType.STRING,
            required=True,
            description="Formal definition"
        ),
        "concept_type": PropertyDefinition(
            name="concept_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of concept",
            allowed_values=["definition", "axiom", "property", "notation", "structure"]
        ),
        "related_concepts": PropertyDefinition(
            name="related_concepts",
            type=PropertyType.ARRAY,
            required=False,
            description="Related mathematical concepts"
        ),
        "examples": PropertyDefinition(
            name="examples",
            type=PropertyType.ARRAY,
            required=False,
            description="Examples illustrating the concept"
        )
    },
    examples=[
        {
            "name": "Group",
            "concept_type": "structure",
            "definition": "A set equipped with an operation satisfying closure, associativity, identity, and invertibility"
        }
    ]
)

# TechniqueEntity - Represents mathematical techniques and methods
MATHEMATICAL_REASONING_SCHEMA.entity_types["TechniqueEntity"] = EntityType(
    name="TechniqueEntity",
    description="Represents a mathematical technique or method",
    properties={
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Name of the technique"
        ),
        "description": PropertyDefinition(
            name="description",
            type=PropertyType.STRING,
            required=True,
            description="Description of the technique"
        ),
        "technique_type": PropertyDefinition(
            name="technique_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of technique",
            allowed_values=["proof_method", "calculation", "construction", "reduction", "transformation"]
        ),
        "application_area": PropertyDefinition(
            name="application_area",
            type=PropertyType.STRING,
            required=False,
            description="Area where this technique is applicable"
        ),
        "limitations": PropertyDefinition(
            name="limitations",
            type=PropertyType.STRING,
            required=False,
            description="Known limitations of the technique"
        )
    },
    examples=[
        {
            "name": "Mathematical Induction",
            "technique_type": "proof_method",
            "description": "A proof technique that establishes the truth of an infinite sequence of statements"
        }
    ]
)

# ProofStepEntity - Represents individual steps in a proof
MATHEMATICAL_REASONING_SCHEMA.entity_types["ProofStepEntity"] = EntityType(
    name="ProofStepEntity",
    description="Represents a single step in a mathematical proof",
    properties={
        "step_number": PropertyDefinition(
            name="step_number",
            type=PropertyType.INTEGER,
            required=True,
            description="Step number in the proof sequence",
            min_value=1
        ),
        "statement": PropertyDefinition(
            name="statement",
            type=PropertyType.STRING,
            required=True,
            description="Statement being proven in this step"
        ),
        "justification": PropertyDefinition(
            name="justification",
            type=PropertyType.STRING,
            required=True,
            description="Justification for this step (e.g., axiom, previous step, theorem)"
        ),
        "inference_type": PropertyDefinition(
            name="inference_type",
            type=PropertyType.ENUM,
            required=False,
            description="Type of logical inference",
            allowed_values=["deduction", "induction", "abduction", "direct", "contradiction"]
        )
    },
    examples=[
        {
            "step_number": 1,
            "statement": "Assume P(n) is true",
            "justification": "Inductive hypothesis",
            "inference_type": "deduction"
        }
    ]
)

# Mathematical Reasoning Relationships
MATHEMATICAL_REASONING_SCHEMA.relationship_types["uses"] = RelationshipType(
    name="uses",
    description="Theorem or technique uses another theorem/concept",
    source_types=["TheoremEntity", "TechniqueEntity"],
    target_types=["TheoremEntity", "ConceptEntity"],
    directed=True
)

MATHEMATICAL_REASONING_SCHEMA.relationship_types["generalizes"] = RelationshipType(
    name="generalizes",
    description="One theorem generalizes another",
    source_types=["TheoremEntity"],
    target_types=["TheoremEntity"],
    directed=True
)

MATHEMATICAL_REASONING_SCHEMA.relationship_types["defines"] = RelationshipType(
    name="defines",
    description="Concept defines another concept",
    source_types=["ConceptEntity"],
    target_types=["ConceptEntity"],
    directed=True
)

MATHEMATICAL_REASONING_SCHEMA.relationship_types["has_proof_step"] = RelationshipType(
    name="has_proof_step",
    description="Theorem has a proof step",
    source_types=["TheoremEntity"],
    target_types=["ProofStepEntity"],
    directed=True
)


# =============================================================================
# WORKFLOW/PROVENANCE SCHEMA
# =============================================================================

WORKFLOW_PROVENANCE_SCHEMA = EntitySchema(
    domain="workflow_provenance",
    description="Schema for workflow and provenance tracking including workflows, tasks, agents, and executions",
    version="1.0.0",
    metadata={
        "author": "OpenEvolve",
        "created": "2025-01-07",
        "use_cases": ["workflow orchestration", "provenance tracking", "agent coordination"]
    }
)

# WorkflowEntity - Represents a workflow definition
WORKFLOW_PROVENANCE_SCHEMA.entity_types["WorkflowEntity"] = EntityType(
    name="WorkflowEntity",
    description="Represents a workflow or pipeline definition",
    properties={
        "workflow_id": PropertyDefinition(
            name="workflow_id",
            type=PropertyType.STRING,
            required=True,
            description="Unique workflow identifier"
        ),
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Human-readable workflow name"
        ),
        "description": PropertyDefinition(
            name="description",
            type=PropertyType.STRING,
            required=False,
            description="Workflow description"
        ),
        "stages": PropertyDefinition(
            name="stages",
            type=PropertyType.ARRAY,
            required=True,
            description="List of workflow stages"
        ),
        "parameters": PropertyDefinition(
            name="parameters",
            type=PropertyType.ARRAY,
            required=False,
            description="Workflow input parameters"
        ),
        "status": PropertyDefinition(
            name="status",
            type=PropertyType.ENUM,
            required=False,
            description="Current workflow status",
            allowed_values=["draft", "active", "paused", "completed", "failed"]
        )
    },
    examples=[
        {
            "workflow_id": "code-analysis-pipeline-001",
            "name": "Code Analysis Pipeline",
            "stages": ["parse", "analyze", "report"],
            "status": "active"
        }
    ]
)

# TaskEntity - Represents a task within a workflow
WORKFLOW_PROVENANCE_SCHEMA.entity_types["TaskEntity"] = EntityType(
    name="TaskEntity",
    description="Represents a task or step within a workflow",
    properties={
        "task_id": PropertyDefinition(
            name="task_id",
            type=PropertyType.STRING,
            required=True,
            description="Unique task identifier"
        ),
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Task name"
        ),
        "task_type": PropertyDefinition(
            name="task_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of task",
            allowed_values=["analysis", "extraction", "validation", "transformation", "aggregation"]
        ),
        "status": PropertyDefinition(
            name="status",
            type=PropertyType.ENUM,
            required=True,
            description="Task status",
            allowed_values=["pending", "running", "completed", "failed", "cancelled"]
        ),
        "result": PropertyDefinition(
            name="result",
            type=PropertyType.OBJECT,
            required=False,
            description="Task result or output"
        ),
        "error": PropertyDefinition(
            name="error",
            type=PropertyType.STRING,
            required=False,
            description="Error message if task failed"
        )
    },
    examples=[
        {
            "task_id": "task-001",
            "name": "Extract dependencies",
            "task_type": "extraction",
            "status": "completed"
        }
    ]
)

# AgentEntity - Represents an agent or service
WORKFLOW_PROVENANCE_SCHEMA.entity_types["AgentEntity"] = EntityType(
    name="AgentEntity",
    description="Represents an agent, service, or worker",
    properties={
        "agent_id": PropertyDefinition(
            name="agent_id",
            type=PropertyType.STRING,
            required=True,
            description="Unique agent identifier"
        ),
        "name": PropertyDefinition(
            name="name",
            type=PropertyType.STRING,
            required=True,
            description="Agent name"
        ),
        "agent_type": PropertyDefinition(
            name="agent_type",
            type=PropertyType.ENUM,
            required=True,
            description="Type of agent",
            allowed_values=["human", "automated", "hybrid", "service"]
        ),
        "capabilities": PropertyDefinition(
            name="capabilities",
            type=PropertyType.ARRAY,
            required=False,
            description="List of agent capabilities"
        ),
        "status": PropertyDefinition(
            name="status",
            type=PropertyType.ENUM,
            required=False,
            description="Agent status",
            allowed_values=["idle", "busy", "offline", "error"]
        )
    },
    examples=[
        {
            "agent_id": "agent-code-analyzer",
            "name": "Code Analyzer Agent",
            "agent_type": "automated",
            "capabilities": ["parse_python", "parse_javascript", "extract_dependencies"],
            "status": "idle"
        }
    ]
)

# ExecutionEntity - Represents a workflow execution instance
WORKFLOW_PROVENANCE_SCHEMA.entity_types["ExecutionEntity"] = EntityType(
    name="ExecutionEntity",
    description="Represents a specific execution of a workflow or task",
    properties={
        "execution_id": PropertyDefinition(
            name="execution_id",
            type=PropertyType.STRING,
            required=True,
            description="Unique execution identifier"
        ),
        "timestamp": PropertyDefinition(
            name="timestamp",
            type=PropertyType.DATETIME,
            required=True,
            description="Execution start timestamp"
        ),
        "duration": PropertyDefinition(
            name="duration",
            type=PropertyType.FLOAT,
            required=False,
            description="Execution duration in seconds",
            min_value=0
        ),
        "status": PropertyDefinition(
            name="status",
            type=PropertyType.ENUM,
            required=True,
            description="Execution status",
            allowed_values=["started", "running", "completed", "failed", "cancelled"]
        ),
        "outcome": PropertyDefinition(
            name="outcome",
            type=PropertyType.STRING,
            required=False,
            description="Execution outcome or result summary"
        ),
        "input_data": PropertyDefinition(
            name="input_data",
            type=PropertyType.OBJECT,
            required=False,
            description="Input data for this execution"
        ),
        "output_data": PropertyDefinition(
            name="output_data",
            type=PropertyType.OBJECT,
            required=False,
            description="Output data from this execution"
        )
    },
    examples=[
        {
            "execution_id": "exec-20250107-001",
            "timestamp": "2025-01-07T10:30:00Z",
            "duration": 45.2,
            "status": "completed",
            "outcome": "Successfully analyzed 15 files"
        }
    ]
)

# Workflow/Provenance Relationships
WORKFLOW_PROVENANCE_SCHEMA.relationship_types["contains"] = RelationshipType(
    name="contains",
    description="Workflow contains a task",
    source_types=["WorkflowEntity"],
    target_types=["TaskEntity"],
    directed=True
)

WORKFLOW_PROVENANCE_SCHEMA.relationship_types["executed_by"] = RelationshipType(
    name="executed_by",
    description="Task executed by agent",
    source_types=["TaskEntity"],
    target_types=["AgentEntity"],
    directed=True
)

WORKFLOW_PROVENANCE_SCHEMA.relationship_types["has_execution"] = RelationshipType(
    name="has_execution",
    description="Workflow or task has an execution record",
    source_types=["WorkflowEntity", "TaskEntity"],
    target_types=["ExecutionEntity"],
    directed=True
)

WORKFLOW_PROVENANCE_SCHEMA.relationship_types["preceded_by"] = RelationshipType(
    name="preceded_by",
    description="Task preceded by another task (temporal ordering)",
    source_types=["TaskEntity"],
    target_types=["TaskEntity"],
    directed=True
)
