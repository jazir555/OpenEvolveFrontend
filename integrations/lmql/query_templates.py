"""LMQL query templates for Knowledge Graph operations.

Pre-built declarative queries for common KG tasks.
These templates use valid LMQL syntax for constrained generation.

Architecture: SSOT (Single Source of Truth)
- Primary templates in integrations/lmql/
- Knowledge Engine uses these via wrapper

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# TEMPLATE CATEGORIES
# =============================================================================


class TemplateCategory(Enum):
    """Categories of LMQL query templates."""
    ENTITY = "entity"
    RELATION = "relation"
    SCHEMA = "schema"
    QUERY = "query"
    REASONING = "reasoning"
    DIALOG = "dialog"


# =============================================================================
# ENTITY EXTRACTION TEMPLATES
# =============================================================================


ENTITY_EXTRACTION_LMQL = """
"Extract named entities from the following text.

Text: {text}

Return a JSON list of entities with this exact format:
[{{"entity": "Entity Name", "type": "ENTITY_TYPE", "confidence": 0.95}}]

Entity types to extract: {entity_types}

Entities:"

entities: list[dict] = "..." 
    WHERE len(entities) > 0 
    AND all("entity" in e AND "type" in e AND "confidence" in e for e in entities)
    AND all(len(e["entity"]) > 0 for e in entities)
    AND all(e["confidence"] >= {min_confidence} for e in entities)
    AND len(entities) <= {max_entities}

RETURN entities
"""

ENTITY_EXTRACTION_WITH_POSITION_LMQL = """
"Extract named entities from the text with their positions.

Text: {text}

Return JSON:
[{{
    "entity": "Entity Name",
    "type": "ENTITY_TYPE", 
    "confidence": 0.95,
    "start_pos": 0,
    "end_pos": 10
}}]

Entities:"

entities: list[dict] = "..."
    WHERE len(entities) >= {min_entities}
    AND len(entities) <= {max_entities}
    AND all("start_pos" in e and "end_pos" in e for e in entities)
    AND all(e["start_pos"] < e["end_pos"] for e in entities)
    AND all(e["confidence"] >= {min_confidence} for e in entities)

RETURN entities
"""

ENTITY_LINKING_LMQL = """
"Link the extracted entity to a knowledge base entry.

Entity: {entity_text}
Context: {context}
Candidate KB entries: {candidates}

Return JSON:
{{
    "linked_entity": "KB Entity Name",
    "kb_id": "entity_id",
    "confidence": 0.95,
    "match_type": "exact|partial|contextual"
}}"

link_result: dict = "..."
    WHERE "linked_entity" in link_result
    AND "kb_id" in link_result
    AND link_result["confidence"] >= {min_confidence}
    AND link_result["match_type"] in ["exact", "partial", "contextual"]

RETURN link_result
"""

ENTITY_DISAMBIGUATION_LMQL = """
"Disambiguate the entity mention to the correct KB entry.

Mention: {mention}
Context: {context}
Candidates:
{candidates}

Which candidate is correct? Return JSON:
{{
    "selected_id": "candidate_id",
    "confidence": 0.9,
    "reasoning": "brief explanation"
}}"

disambiguation: dict = "..."
    WHERE "selected_id" in disambiguation
    AND "confidence" in disambiguation
    AND disambiguation["confidence"] >= {min_confidence}
    AND len(disambiguation["reasoning"]) > 10

RETURN disambiguation
"""


# =============================================================================
# RELATION EXTRACTION TEMPLATES
# =============================================================================


RELATION_EXTRACTION_LMQL = """
"Extract relations between entities from the text.

Text: {text}
Entities: {entities}
Relation types: {relation_types}

Return JSON list:
[{{
    "subject": "Entity1",
    "predicate": "relation_type",
    "object": "Entity2",
    "confidence": 0.9
}}]"

relations: list[dict] = "..."
    WHERE len(relations) >= {min_relations}
    AND len(relations) <= {max_relations}
    AND all("subject" in r and "predicate" in r and "object" in r for r in relations)
    AND all(r["confidence"] >= {min_confidence} for r in relations)
    AND all(len(r["subject"]) > 0 and len(r["object"]) > 0 for r in relations)

RETURN relations
"""

RELATION_EXTRACTION_WITH_TEMPORAL_LMQL = """
"Extract relations with temporal information.

Text: {text}

Return JSON:
[{{
    "subject": "Entity1",
    "predicate": "relation",
    "object": "Entity2",
    "valid_from": "2020-01-01",
    "valid_to": null,
    "confidence": 0.9
}}]"

relations: list[dict] = "..."
    WHERE all("valid_from" in r for r in relations)
    AND all(r["confidence"] >= {min_confidence} for r in relations)

RETURN relations
"""

TRIPLET_EXTRACTION_LMQL = """
"Extract subject-predicate-object triplets from text.

Text: {text}

Return JSON:
{{
    "triplets": [
        ["subject", "predicate", "object"]
    ],
    "confidence_scores": [0.9]
}}"

result: dict = "..."
    WHERE "triplets" in result
    AND len(result["triplets"]) == len(result.get("confidence_scores", []))
    AND all(len(t) == 3 for t in result["triplets"])
    AND all(all(len(item) > 0 for item in t) for t in result["triplets"])

RETURN result
"""


# =============================================================================
# SCHEMA INFERENCE TEMPLATES
# =============================================================================


SCHEMA_INFERENCE_LMQL = """
"Infer the knowledge graph schema from the sample data.

Sample Data:
{sample_data}

Return JSON schema:
{{
    "entity_types": [
        {{
            "name": "EntityType",
            "properties": ["prop1", "prop2"],
            "description": "what this entity represents"
        }}
    ],
    "relation_types": [
        {{
            "name": "RELATION_TYPE",
            "domain": ["SourceEntity"],
            "range": ["TargetEntity"],
            "description": "what this relation means"
        }}
    ],
    "constraints": [
        "entity-level constraints"
    ]
}}"

schema: dict = "..."
    WHERE "entity_types" in schema
    AND "relation_types" in schema
    AND len(schema["entity_types"]) > 0
    AND all("name" in et for et in schema["entity_types"])
    AND all("name" in rt for rt in schema["relation_types"])

RETURN schema
"""

SCHEMA_INFERENCE_FROM_QUERIES_LMQL = """
"Infer schema from example queries.

Example Queries:
{example_queries}

Return inferred schema as JSON:
{{
    "entity_types": [...],
    "relation_types": [...],
    "query_patterns": [...]
}}"

schema: dict = "..."
    WHERE "entity_types" in schema and "relation_types" in schema
    AND "query_patterns" in schema

RETURN schema
"""

SCHEMA_VALIDATION_LMQL = """
"Validate if data conforms to the given schema.

Schema:
{schema}

Data to validate:
{data}

Return validation result:
{{
    "valid": true,
    "errors": ["list of validation errors"],
    "warnings": ["list of warnings"]
}}"

validation: dict = "..."
    WHERE "valid" in validation
    AND isinstance(validation["valid"], bool)
    AND "errors" in validation
    AND "warnings" in validation

RETURN validation
"""


# =============================================================================
# CYPHER GENERATION TEMPLATES (Memgraph Compatible)
# =============================================================================


CYPHER_GENERATION_LMQL = """
"Convert the natural language query to Cypher for Memgraph.

Query: {natural_language_query}
Parameters: {params}

Schema context:
{schema_context}

Generate a Cypher query that:
1. Uses Memgraph-compatible syntax
2. Uses parameterized values ($param_name)
3. Is optimized for performance
4. Returns only the Cypher query, no explanation

Cypher:"

cypher: str = "..."
    WHERE STOPS_BEFORE(cypher, "\n\n")
    AND len(cypher) > 10
    AND cypher.upper().startswith(("MATCH", "CREATE", "MERGE", "RETURN", "CALL"))
    AND "$" in cypher or "{{" not in cypher  # Use parameters, not string interpolation

RETURN cypher
"""

CYPHER_GENERATION_FOR_PATH_LMQL = """
"Generate Cypher for path finding query.

Start node: {start_node}
End node: {end_node}
Relationship types: {rel_types}
Max path length: {max_length}

Generate Memgraph Cypher path query."

cypher: str = "..."
    WHERE "MATCH" in cypher.upper()
    AND "path" in cypher.lower() or "-" in cypher
    AND "*1.." in cypher or "*" in cypher

RETURN cypher
"""

CYPHER_GENERATION_FOR_TEMPORAL_LMQL = """
"Generate Cypher for temporal query.

Query: {natural_language_query}
Temporal constraint: {temporal_constraint}
Current time: {current_time}

Generate Cypher using valid_from/valid_to properties:
- valid_from <= timestamp
- valid_to IS NULL OR valid_to > timestamp"

cypher: str = "..."
    WHERE "valid_from" in cypher or "validTo" in cypher or "valid_to" in cypher
    AND "$timestamp" in cypher or "$" in cypher

RETURN cypher
"""

CYPHER_GENERATION_FOR_AGGREGATION_LMQL = """
"Generate Cypher for aggregation query.

Query: {natural_language_query}
Aggregation type: {agg_type}
Group by: {group_by}

Generate optimized Memgraph Cypher."

cypher: str = "..."
    WHERE any(keyword in cypher.upper() for keyword in ["COUNT", "SUM", "AVG", "MAX", "MIN", "COLLECT"])
    AND "RETURN" in cypher.upper()

RETURN cypher
"""


# =============================================================================
# MULTI-HOP REASONING TEMPLATES
# =============================================================================


MULTI_HOP_REASONING_LMQL = """
"Answer the question using multi-hop reasoning over the knowledge graph.

Question: {question}
Starting entity: {start_entity}
Available relations: {relations}

Think step by step:
1. What is the starting point?
2. What relations should I follow?
3. What intermediate entities are needed?
4. What is the final answer?

Return reasoning trace:
{{
    "reasoning_steps": [
        {{
            "step": 1,
            "action": "follow relation X",
            "result": "found entity Y"
        }}
    ],
    "answer": "final answer",
    "confidence": 0.9,
    "entities_visited": ["entity1", "entity2"]
}}"

reasoning: dict = "..."
    WHERE "reasoning_steps" in reasoning
    AND "answer" in reasoning
    AND len(reasoning["reasoning_steps"]) >= {min_hops}
    AND len(reasoning["reasoning_steps"]) <= {max_hops}
    AND reasoning["confidence"] >= {min_confidence}
    AND all("step" in s and "action" in s for s in reasoning["reasoning_steps"])

RETURN reasoning
"""

CHAIN_OF_THOUGHT_LMQL = """
"Answer the complex question using chain-of-thought reasoning.

Question: {question}
Context: {context}

Let's think through this step by step:"

reasoning: str = "..."
    WHERE len(reasoning) > 50
    AND STOPS_AT(reasoning, "Final Answer:")

answer: str = "..."
    WHERE len(answer) > 0
    AND STOPS_AT(answer, "\n\n")

RETURN {{
    "reasoning": reasoning,
    "answer": answer
}}
"""

PATH_REASONING_LMQL = """
"Find reasoning path from start to answer.

Start: {start_entity}
Question: {question}
KG Entities: {available_entities}
KG Relations: {available_relations}

Return the reasoning path:
{{
    "path": [
        {{"entity": "A", "relation": "r1", "next_entity": "B"}},
        {{"entity": "B", "relation": "r2", "next_entity": "C"}}
    ],
    "answer": "C",
    "confidence": 0.9
}}"

path_result: dict = "..."
    WHERE "path" in path_result
    AND "answer" in path_result
    AND len(path_result["path"]) >= {min_path_length}
    AND len(path_result["path"]) <= {max_path_length}
    AND all("entity" in p and "relation" in p for p in path_result["path"])

RETURN path_result
"""


# =============================================================================
# DIALOG AND CONVERSATION TEMPLATES
# =============================================================================


MULTI_TURN_DIALOG_LMQL = """
"Engage in a multi-turn conversation to gather information.

Goal: {conversation_goal}
History: {history}
Current turn: {turn}
Max turns: {max_turns}

Respond appropriately and check if goal is achieved."

response: str = "..."
    WHERE len(response) > 0
    AND len(response) <= {max_response_length}

goal_achieved: bool = "..."

RETURN {{
    "response": response,
    "goal_achieved": goal_achieved,
    "turn": {turn}
}}
"""

CONSTRAINED_DIALOG_LMQL = """
"Respond within specified constraints.

User: {user_message}
Constraints:
{constraints}

Your response:"

response: str = "..."
    WHERE len(response) >= {min_length}
    AND len(response) <= {max_length}
    AND {additional_constraints}

RETURN response
"""

INFORMATION_GATHERING_LMQL = """
"Gather specific information from the user.

Target information: {target_info}
What we know: {known_info}
What's missing: {missing_info}

Ask a focused question to get the missing information."

question: str = "..."
    WHERE len(question) > 10
    AND len(question) < 200
    AND "?" in question

RETURN question
"""


# =============================================================================
# KNOWLEDGE VERIFICATION TEMPLATES
# =============================================================================


FACT_VERIFICATION_LMQL = """
"Verify if the statement is supported by the knowledge graph.

Statement: {statement}
KG Evidence: {kg_evidence}
External Evidence: {external_evidence}

Return verification result:
{{
    "verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_INFO",
    "confidence": 0.9,
    "supporting_evidence": ["..."],
    "contradicting_evidence": ["..."],
    "reasoning": "explanation"
}}"

verification: dict = "..."
    WHERE "verdict" in verification
    AND verification["verdict"] in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    AND "confidence" in verification
    AND verification["confidence"] >= {min_confidence}
    AND "reasoning" in verification

RETURN verification
"""

CONSISTENCY_CHECK_LMQL = """
"Check if new fact is consistent with existing KG.

New Fact: {new_fact}
Related KG Facts: {existing_facts}

Return consistency check:
{{
    "consistent": true,
    "confidence": 0.9,
    "conflicts": ["any conflicting facts"],
    "suggestions": ["how to resolve conflicts"]
}}"

check: dict = "..."
    WHERE "consistent" in check
    AND isinstance(check["consistent"], bool)
    AND "confidence" in check

RETURN check
"""


# =============================================================================
# TEMPLATE MANAGER
# =============================================================================


@dataclass
class QueryTemplate:
    """Represents a query template with metadata."""
    name: str
    template: str
    category: TemplateCategory
    description: str
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with provided parameters."""
        # Validate required params
        missing = [p for p in self.required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        # Apply defaults for optional params
        params = self.optional_params.copy()
        params.update(kwargs)
        
        return self.template.format(**params)


class TemplateRegistry:
    """Registry for LMQL query templates."""
    
    def __init__(self):
        self._templates: Dict[str, QueryTemplate] = {}
        self._register_default_templates()
        
    def _register_default_templates(self) -> None:
        """Register all default templates."""
        # Entity extraction
        self.register(QueryTemplate(
            name="entity_extraction",
            template=ENTITY_EXTRACTION_LMQL,
            category=TemplateCategory.ENTITY,
            description="Extract named entities from text",
            required_params=["text", "entity_types"],
            optional_params={"min_confidence": 0.5, "max_entities": 50}
        ))
        
        self.register(QueryTemplate(
            name="entity_extraction_with_position",
            template=ENTITY_EXTRACTION_WITH_POSITION_LMQL,
            category=TemplateCategory.ENTITY,
            description="Extract entities with position information",
            required_params=["text"],
            optional_params={"min_entities": 0, "max_entities": 100, "min_confidence": 0.5}
        ))
        
        self.register(QueryTemplate(
            name="entity_linking",
            template=ENTITY_LINKING_LMQL,
            category=TemplateCategory.ENTITY,
            description="Link entity to knowledge base",
            required_params=["entity_text", "context", "candidates"],
            optional_params={"min_confidence": 0.7}
        ))
        
        self.register(QueryTemplate(
            name="entity_disambiguation",
            template=ENTITY_DISAMBIGUATION_LMQL,
            category=TemplateCategory.ENTITY,
            description="Disambiguate entity mention",
            required_params=["mention", "context", "candidates"],
            optional_params={"min_confidence": 0.6}
        ))
        
        # Relation extraction
        self.register(QueryTemplate(
            name="relation_extraction",
            template=RELATION_EXTRACTION_LMQL,
            category=TemplateCategory.RELATION,
            description="Extract relations between entities",
            required_params=["text", "entities", "relation_types"],
            optional_params={"min_relations": 0, "max_relations": 50, "min_confidence": 0.5}
        ))
        
        self.register(QueryTemplate(
            name="relation_extraction_temporal",
            template=RELATION_EXTRACTION_WITH_TEMPORAL_LMQL,
            category=TemplateCategory.RELATION,
            description="Extract relations with temporal information",
            required_params=["text"],
            optional_params={"min_confidence": 0.5}
        ))
        
        self.register(QueryTemplate(
            name="triplet_extraction",
            template=TRIPLET_EXTRACTION_LMQL,
            category=TemplateCategory.RELATION,
            description="Extract S-P-O triplets",
            required_params=["text"],
            optional_params={}
        ))
        
        # Schema inference
        self.register(QueryTemplate(
            name="schema_inference",
            template=SCHEMA_INFERENCE_LMQL,
            category=TemplateCategory.SCHEMA,
            description="Infer KG schema from sample data",
            required_params=["sample_data"],
            optional_params={}
        ))
        
        self.register(QueryTemplate(
            name="schema_inference_from_queries",
            template=SCHEMA_INFERENCE_FROM_QUERIES_LMQL,
            category=TemplateCategory.SCHEMA,
            description="Infer schema from example queries",
            required_params=["example_queries"],
            optional_params={}
        ))
        
        self.register(QueryTemplate(
            name="schema_validation",
            template=SCHEMA_VALIDATION_LMQL,
            category=TemplateCategory.SCHEMA,
            description="Validate data against schema",
            required_params=["schema", "data"],
            optional_params={}
        ))
        
        # Cypher generation
        self.register(QueryTemplate(
            name="cypher_generation",
            template=CYPHER_GENERATION_LMQL,
            category=TemplateCategory.QUERY,
            description="Generate Cypher from natural language",
            required_params=["natural_language_query"],
            optional_params={"params": "{}", "schema_context": ""}
        ))
        
        self.register(QueryTemplate(
            name="cypher_path_query",
            template=CYPHER_GENERATION_FOR_PATH_LMQL,
            category=TemplateCategory.QUERY,
            description="Generate path finding Cypher",
            required_params=["start_node"],
            optional_params={"end_node": "", "rel_types": "[]", "max_length": 5}
        ))
        
        self.register(QueryTemplate(
            name="cypher_temporal",
            template=CYPHER_GENERATION_FOR_TEMPORAL_LMQL,
            category=TemplateCategory.QUERY,
            description="Generate temporal Cypher query",
            required_params=["natural_language_query"],
            optional_params={"temporal_constraint": "", "current_time": "now()"}
        ))
        
        self.register(QueryTemplate(
            name="cypher_aggregation",
            template=CYPHER_GENERATION_FOR_AGGREGATION_LMQL,
            category=TemplateCategory.QUERY,
            description="Generate aggregation Cypher",
            required_params=["natural_language_query"],
            optional_params={"agg_type": "count", "group_by": ""}
        ))
        
        # Multi-hop reasoning
        self.register(QueryTemplate(
            name="multi_hop_reasoning",
            template=MULTI_HOP_REASONING_LMQL,
            category=TemplateCategory.REASONING,
            description="Multi-hop reasoning over KG",
            required_params=["question", "start_entity", "relations"],
            optional_params={"min_hops": 1, "max_hops": 5, "min_confidence": 0.6}
        ))
        
        self.register(QueryTemplate(
            name="chain_of_thought",
            template=CHAIN_OF_THOUGHT_LMQL,
            category=TemplateCategory.REASONING,
            description="Chain-of-thought reasoning",
            required_params=["question"],
            optional_params={"context": ""}
        ))
        
        self.register(QueryTemplate(
            name="path_reasoning",
            template=PATH_REASONING_LMQL,
            category=TemplateCategory.REASONING,
            description="Path-based reasoning",
            required_params=["start_entity", "question"],
            optional_params={
                "available_entities": "[]",
                "available_relations": "[]",
                "min_path_length": 1,
                "max_path_length": 5
            }
        ))
        
        # Dialog
        self.register(QueryTemplate(
            name="multi_turn_dialog",
            template=MULTI_TURN_DIALOG_LMQL,
            category=TemplateCategory.DIALOG,
            description="Multi-turn conversation",
            required_params=["conversation_goal", "history", "turn"],
            optional_params={"max_turns": 5, "max_response_length": 500}
        ))
        
        self.register(QueryTemplate(
            name="constrained_dialog",
            template=CONSTRAINED_DIALOG_LMQL,
            category=TemplateCategory.DIALOG,
            description="Dialog with constraints",
            required_params=["user_message", "constraints"],
            optional_params={"min_length": 10, "max_length": 500, "additional_constraints": "True"}
        ))
        
        self.register(QueryTemplate(
            name="information_gathering",
            template=INFORMATION_GATHERING_LMQL,
            category=TemplateCategory.DIALOG,
            description="Ask for missing information",
            required_params=["target_info", "known_info", "missing_info"],
            optional_params={}
        ))
        
        # Verification
        self.register(QueryTemplate(
            name="fact_verification",
            template=FACT_VERIFICATION_LMQL,
            category=TemplateCategory.QUERY,
            description="Verify facts against KG",
            required_params=["statement", "kg_evidence"],
            optional_params={"external_evidence": "", "min_confidence": 0.7}
        ))
        
        self.register(QueryTemplate(
            name="consistency_check",
            template=CONSISTENCY_CHECK_LMQL,
            category=TemplateCategory.QUERY,
            description="Check fact consistency",
            required_params=["new_fact", "existing_facts"],
            optional_params={}
        ))
        
    def register(self, template: QueryTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template
        
    def get(self, name: str) -> Optional[QueryTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
        
    def list_templates(self, category: Optional[TemplateCategory] = None) -> List[str]:
        """List available templates, optionally filtered by category."""
        if category:
            return [name for name, t in self._templates.items() if t.category == category]
        return list(self._templates.keys())
        
    def render(self, name: str, **kwargs) -> str:
        """Render a template by name."""
        template = self.get(name)
        if not template:
            raise KeyError(f"Template not found: {name}")
        return template.render(**kwargs)
        
    def get_by_category(self, category: TemplateCategory) -> Dict[str, QueryTemplate]:
        """Get all templates in a category."""
        return {name: t for name, t in self._templates.items() if t.category == category}


# =============================================================================
# DEFAULT REGISTRY INSTANCE
# =============================================================================

_default_registry: Optional[TemplateRegistry] = None


def get_default_registry() -> TemplateRegistry:
    """Get the default template registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TemplateRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the default registry."""
    global _default_registry
    _default_registry = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_template(name: str) -> Optional[QueryTemplate]:
    """Get a template by name using default registry."""
    return get_default_registry().get(name)


def render_template(name: str, **kwargs) -> str:
    """Render a template by name using default registry."""
    return get_default_registry().render(name, **kwargs)


def list_templates(category: Optional[str] = None) -> List[str]:
    """List available templates."""
    cat = TemplateCategory(category) if category else None
    return get_default_registry().list_templates(cat)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Template strings
    "ENTITY_EXTRACTION_LMQL",
    "ENTITY_EXTRACTION_WITH_POSITION_LMQL",
    "ENTITY_LINKING_LMQL",
    "ENTITY_DISAMBIGUATION_LMQL",
    "RELATION_EXTRACTION_LMQL",
    "RELATION_EXTRACTION_WITH_TEMPORAL_LMQL",
    "TRIPLET_EXTRACTION_LMQL",
    "SCHEMA_INFERENCE_LMQL",
    "SCHEMA_INFERENCE_FROM_QUERIES_LMQL",
    "SCHEMA_VALIDATION_LMQL",
    "CYPHER_GENERATION_LMQL",
    "CYPHER_GENERATION_FOR_PATH_LMQL",
    "CYPHER_GENERATION_FOR_TEMPORAL_LMQL",
    "CYPHER_GENERATION_FOR_AGGREGATION_LMQL",
    "MULTI_HOP_REASONING_LMQL",
    "CHAIN_OF_THOUGHT_LMQL",
    "PATH_REASONING_LMQL",
    "MULTI_TURN_DIALOG_LMQL",
    "CONSTRAINED_DIALOG_LMQL",
    "INFORMATION_GATHERING_LMQL",
    "FACT_VERIFICATION_LMQL",
    "CONSISTENCY_CHECK_LMQL",
    # Classes
    "TemplateCategory",
    "QueryTemplate",
    "TemplateRegistry",
    # Functions
    "get_default_registry",
    "reset_registry",
    "get_template",
    "render_template",
    "list_templates",
]


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    # Demo usage
    registry = get_default_registry()
    
    print("Available templates:")
    for category in TemplateCategory:
        templates = registry.list_templates(category)
        print(f"\n{category.value.upper()}:")
        for name in templates:
            template = registry.get(name)
            print(f"  - {name}: {template.description}")
            
    # Example rendering
    print("\n" + "="*50)
    print("Example: Entity Extraction Template")
    print("="*50)
    
    rendered = registry.render(
        "entity_extraction",
        text="Apple Inc. was founded by Steve Jobs in Cupertino.",
        entity_types="ORG, PERSON, GPE",
        min_confidence=0.7,
        max_entities=10
    )
    print(rendered[:500] + "...")
