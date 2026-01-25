"""
Schema Mappings

Defines cross-project entity type mappings to enable knowledge integration
between different knowledge graph systems (Knowledge Engine, Graphiti, OneKE, etc.).
"""

# =============================================================================
# KNOWLEDGE ENGINE TO GRAPHITI MAPPINGS
# =============================================================================

KNOWLEDGE_ENGINE_TO_GRAPHITI = {
    # Software Engineering mappings
    'CodeEntity': 'Activity',  # Code elements as activities
    'DependencyEntity': 'Relation',  # Dependencies as relationships
    'APISchema': 'Activity',  # API endpoints as activities

    # Mathematical Reasoning mappings
    'TheoremEntity': 'Requirement',  # Theorems as requirements to be met
    'ConceptEntity': 'Requirement',  # Concepts as requirements
    'TechniqueEntity': 'Procedure',  # Techniques as procedures
    'ProofStepEntity': 'Activity',  # Proof steps as activities

    # Workflow/Provenance mappings
    'WorkflowEntity': 'Document',  # Workflows as documents
    'TaskEntity': 'Activity',  # Tasks as activities
    'AgentEntity': 'Organization',  # Agents as organizations
    'ExecutionEntity': 'Event',  # Executions as events

    # Generic mappings
    'solution_pattern': 'Procedure',
    'critique_insight': 'Requirement',
    'team_performance': 'Preference',
    'problem': 'Event',
    'workflow': 'Document',
    'agent': 'Organization',
    'tool': 'Technology',
    'technique': 'Methodology',
}


# =============================================================================
# GRAPHITI TO KNOWLEDGE ENGINE MAPPINGS
# =============================================================================

GRAPHITI_TO_KNOWLEDGE_ENGINE = {
    # Reverse mappings from Graphiti to Knowledge Engine
    'Activity': 'CodeEntity',  # Map to code entities or tasks
    'Relation': 'DependencyEntity',  # Map to dependencies
    'Requirement': 'TheoremEntity',  # Map to theorems or requirements
    'Procedure': 'TechniqueEntity',  # Map to techniques
    'Document': 'WorkflowEntity',  # Map to workflows
    'Event': 'ExecutionEntity',  # Map to executions
    'Organization': 'AgentEntity',  # Map to agents
    'Technology': 'CodeEntity',  # Map to code entities
    'Methodology': 'TechniqueEntity',  # Map to techniques
}


# =============================================================================
# KNOWLEDGE ENGINE TO ONEKE MAPPINGS
# =============================================================================

KNOWLEDGE_ENGINE_TO_ONEKE = {
    # Software Engineering mappings
    'CodeEntity': 'Class',  # Code elements map to class types
    'DependencyEntity': 'Relation',  # Dependencies as relations
    'APISchema': 'Method',  # API endpoints as methods

    # Mathematical Reasoning mappings
    'TheoremEntity': 'Concept',  # Theorems as concepts
    'ConceptEntity': 'Concept',  # Concepts as concepts
    'TechniqueEntity': 'Method',  # Techniques as methods

    # Workflow/Provenance mappings
    'WorkflowEntity': 'Process',  # Workflows as processes
    'TaskEntity': 'Activity',  # Tasks as activities
    'AgentEntity': 'Actor',  # Agents as actors
    'ExecutionEntity': 'Event',  # Executions as events
}


# =============================================================================
# ONEKE TO KNOWLEDGE ENGINE MAPPINGS
# =============================================================================

ONEKE_TO_KNOWLEDGE_ENGINE = {
    # Reverse mappings from OneKE to Knowledge Engine
    'Class': 'CodeEntity',
    'Relation': 'DependencyEntity',
    'Method': 'CodeEntity',
    'Concept': 'ConceptEntity',
    'Process': 'WorkflowEntity',
    'Activity': 'TaskEntity',
    'Actor': 'AgentEntity',
    'Event': 'ExecutionEntity',
}


# =============================================================================
# OPENEVOLVE TO EXTERNAL SYSTEMS MAPPINGS
# =============================================================================

OPENEVOLVE_TO_NEO4J = {
    # Generic mappings for Neo4j property graph model
    'CodeEntity': 'Node',
    'DependencyEntity': 'Relationship',
    'APISchema': 'Node',
    'BugPattern': 'Node',
    'TheoremEntity': 'Node',
    'ConceptEntity': 'Node',
    'TechniqueEntity': 'Node',
    'ProofStepEntity': 'Node',
    'WorkflowEntity': 'Node',
    'TaskEntity': 'Node',
    'AgentEntity': 'Node',
    'ExecutionEntity': 'Node',
}


# =============================================================================
# DOMAIN-SPECIFIC MAPPINGS
# =============================================================================

SOFTWARE_ENGINEERING_TO_GENERIC = {
    'CodeEntity': 'Artifact',
    'DependencyEntity': 'Dependency',
    'APISchema': 'Interface',
    'BugPattern': 'Issue',
}


MATHEMATICAL_REASONING_TO_GENERIC = {
    'TheoremEntity': 'Proposition',
    'ConceptEntity': 'Definition',
    'TechniqueEntity': 'Method',
    'ProofStepEntity': 'Step',
}


WORKFLOW_PROVENANCE_TO_GENERIC = {
    'WorkflowEntity': 'Process',
    'TaskEntity': 'Task',
    'AgentEntity': 'Actor',
    'ExecutionEntity': 'Run',
}


# =============================================================================
# CONSOLIDATED MAPPING COLLECTION
# =============================================================================

ENTITY_MAPPINGS = {
    'knowledge_engine_to_graphiti': KNOWLEDGE_ENGINE_TO_GRAPHITI,
    'graphiti_to_knowledge_engine': GRAPHITI_TO_KNOWLEDGE_ENGINE,
    'knowledge_engine_to_oneke': KNOWLEDGE_ENGINE_TO_ONEKE,
    'oneke_to_knowledge_engine': ONEKE_TO_KNOWLEDGE_ENGINE,
    'openevolve_to_neo4j': OPENEVOLVE_TO_NEO4J,
    'software_engineering_to_generic': SOFTWARE_ENGINEERING_TO_GENERIC,
    'mathematical_reasoning_to_generic': MATHEMATICAL_REASONING_TO_GENERIC,
    'workflow_provenance_to_generic': WORKFLOW_PROVENANCE_TO_GENERIC,
}


def get_mapping(mapping_name: str) -> dict:
    """
    Get a mapping by name.

    Args:
        mapping_name: Name of the mapping

    Returns:
        Mapping dictionary, or empty dict if not found
    """
    return ENTITY_MAPPINGS.get(mapping_name, {})


def list_mappings() -> list:
    """
    List all available mapping names.

    Returns:
        List of mapping names
    """
    return list(ENTITY_MAPPINGS.keys())


def apply_mapping(entity_type: str, mapping: dict) -> str:
    """
    Apply a mapping to an entity type.

    Args:
        entity_type: Source entity type
        mapping: Mapping dictionary

    Returns:
        Mapped entity type, or original type if no mapping exists
    """
    return mapping.get(entity_type, entity_type)


def create_composite_mapping(*mapping_names: str) -> dict:
    """
    Create a composite mapping by combining multiple mappings.

    Args:
        *mapping_names: Names of mappings to combine

    Returns:
        Composite mapping dictionary
    """
    composite = {}
    for name in mapping_names:
        mapping = get_mapping(name)
        composite.update(mapping)
    return composite


def get_bidirectional_mapping(mapping_name: str) -> dict:
    """
    Get a bidirectional mapping (includes both forward and reverse).

    Args:
        mapping_name: Name of the base mapping

    Returns:
        Bidirectional mapping dictionary
    """
    forward = get_mapping(mapping_name)
    reverse_name = mapping_name.split('_to_')[0] + '_to_' + mapping_name.split('_to_')[1]
    # Try to get reverse mapping
    parts = mapping_name.split('_to_')
    if len(parts) == 2:
        reverse_name = f"{parts[1]}_to_{parts[0]}"
        reverse = get_mapping(reverse_name)
    else:
        reverse = {}

    # Create bidirectional mapping
    bidirectional = {}
    bidirectional.update(forward)
    bidirectional.update(reverse)

    return bidirectional
