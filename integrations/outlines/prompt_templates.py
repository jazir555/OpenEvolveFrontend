"""
Prompt templates optimized for constrained generation with Outlines.

Templates that work with Outlines constraints for KG tasks:
- Entity extraction with type constraints
- Relationship extraction from text
- Schema validation prompts
- Cypher query generation

All templates are designed to work with Outlines JSON/regex constraints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =============================================================================
# ENTITY EXTRACTION TEMPLATES
# =============================================================================

ENTITY_EXTRACTION_TEMPLATE = """You are a precise entity extraction system. Extract all entities from the following text.

TEXT TO ANALYZE:
{text}

ENTITY TYPES TO EXTRACT:
{entity_types}

EXTRACTION RULES:
1. Extract ALL entities that match the specified types
2. Be precise with entity boundaries (don't include extra words)
3. Provide a confidence score (0.0-1.0) for each extraction
4. Extract relevant properties for each entity (e.g., age for persons, location for organizations)
5. Use the exact entity type names provided
6. If no entities are found, return an empty entities array

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "name": "exact entity name",
            "type": "ONE_OF_THE_SPECIFIED_TYPES",
            "confidence": 0.95,
            "properties": [
                {{
                    "name": "property_name",
                    "value": "property_value",
                    "confidence": 0.90
                }}
            ]
        }}
    ],
    "extraction_timestamp": "{timestamp}",
    "model_used": "{model}"
}}

IMPORTANT:
- Output ONLY valid JSON
- Do not include any explanatory text
- Ensure all confidence scores are between 0.0 and 1.0
- Use UTC timestamp format
"""

ENTITY_EXTRACTION_FEW_SHOT_TEMPLATE = """You are a precise entity extraction system. Learn from the examples and extract entities from the input text.

EXAMPLES:
{examples}

TEXT TO ANALYZE:
{text}

ENTITY TYPES TO EXTRACT:
{entity_types}

EXTRACTION RULES:
1. Follow the pattern shown in the examples
2. Extract ALL entities that match the specified types
3. Be precise with entity boundaries
4. Provide confidence scores (0.0-1.0) for each extraction
5. Extract relevant properties for each entity

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "name": "exact entity name",
            "type": "ENTITY_TYPE",
            "confidence": 0.95,
            "properties": []
        }}
    ],
    "extraction_timestamp": "{timestamp}"
}}

Output ONLY valid JSON. No explanatory text.
"""

# =============================================================================
# RELATIONSHIP EXTRACTION TEMPLATES
# =============================================================================

RELATION_EXTRACTION_TEMPLATE = """You are a precise relationship extraction system. Extract all relationships between entities from the following text.

TEXT TO ANALYZE:
{text}

KNOWN ENTITIES (if any):
{entities}

RELATIONSHIP TYPES TO EXTRACT:
{relation_types}

EXTRACTION RULES:
1. Extract ALL relationships between entities mentioned in the text
2. Only use relationship types from the provided list
3. Ensure source and target entities match the known entities list
4. Provide confidence scores (0.0-1.0) for each extraction
5. Extract relevant properties for each relationship (e.g., dates, locations)
6. All relationships are directed (source → target)
7. If no relationships are found, return an empty relationships array

OUTPUT FORMAT (JSON):
{{
    "relationships": [
        {{
            "source": "source entity name",
            "target": "target entity name",
            "type": "RELATIONSHIP_TYPE",
            "confidence": 0.92,
            "properties": [
                {{
                    "name": "property_name",
                    "value": "property_value",
                    "confidence": 0.85
                }}
            ],
            "directed": true
        }}
    ],
    "extraction_timestamp": "{timestamp}",
    "model_used": "{model}"
}}

IMPORTANT:
- Output ONLY valid JSON
- Do not include any explanatory text
- Source and target must exactly match entity names
- All confidence scores must be between 0.0 and 1.0
"""

RELATION_EXTRACTION_WITH_ENTITIES_TEMPLATE = """Extract relationships between the provided entities from the text.

TEXT:
{text}

ENTITIES TO CONNECT:
{entities_list}

VALID RELATIONSHIP TYPES:
{relation_types}

INSTRUCTIONS:
1. Find relationships between the provided entities
2. Use ONLY the specified relationship types
3. Provide confidence scores (0.0-1.0)
4. Include relevant temporal or descriptive properties

OUTPUT (JSON):
{{
    "relationships": [
        {{
            "source": "entity1",
            "target": "entity2",
            "type": "RELATION_TYPE",
            "confidence": 0.90,
            "properties": []
        }}
    ]
}}

Output ONLY JSON. No other text.
"""

# =============================================================================
# SCHEMA VALIDATION TEMPLATES
# =============================================================================

SCHEMA_VALIDATION_TEMPLATE = """You are a data validation system. Validate the provided data against the schema.

DATA TO VALIDATE:
```json
{data}
```

SCHEMA REQUIREMENTS:
{schema_description}

REQUIRED FIELDS:
{required_fields}

VALIDATION RULES:
1. Check all required fields are present
2. Validate data types match schema
3. Check value ranges and formats
4. Identify any missing or invalid data
5. Provide specific error messages for each issue
6. Suggest fixes for each error
7. Calculate overall validation confidence

OUTPUT FORMAT (JSON):
{{
    "is_valid": true/false,
    "errors": ["specific error messages"],
    "warnings": ["warning messages"],
    "confidence": 0.95,
    "suggestions": ["suggested fixes"],
    "issues": [
        {{
            "severity": "ERROR/WARNING/INFO",
            "message": "description",
            "field": "field_name",
            "suggestion": "how to fix"
        }}
    ],
    "validation_timestamp": "{timestamp}",
    "validator_version": "1.0.0"
}}

IMPORTANT:
- Output ONLY valid JSON
- Be specific about what's wrong and how to fix it
- Confidence score must be between 0.0 and 1.0
- Include all identified issues, not just the first one
"""

# =============================================================================
# CYPHER QUERY GENERATION TEMPLATES
# =============================================================================

CYPHER_GENERATION_TEMPLATE = """You are a Memgraph Cypher query generator. Create efficient, correct Cypher queries.

SCHEMA DESCRIPTION:
{schema_description}

QUERY INTENT:
{query_intent}

AVAILABLE NODE LABELS:
{node_labels}

AVAILABLE RELATIONSHIP TYPES:
{relationship_types}

CYPHER GENERATION RULES:
1. Generate Memgraph-compatible Cypher (not Neo4j-specific)
2. Use parameterized queries with $param syntax
3. Prefer MERGE for idempotent operations
4. Use meaningful variable names
5. Add appropriate indexes hints when beneficial
6. LIMIT results when appropriate
7. Avoid APOC procedures (not available in Memgraph)
8. Use CASE statements instead of conditional subqueries

QUERY TYPE GUIDELINES:
- READ queries: Use MATCH, RETURN, optional WHERE
- WRITE queries: Use CREATE or MERGE
- UPDATE queries: Use MATCH + SET
- DELETE queries: Use MATCH + DELETE (prefer DETACH DELETE)

OUTPUT FORMAT (JSON):
{{
    "query": "MATCH (n:Label {{property: $param}}) RETURN n",
    "parameters": {{"param": "value"}},
    "explanation": "Clear explanation of what the query does",
    "query_type": "READ/WRITE/UPDATE/DELETE",
    "estimated_complexity": "LOW/MEDIUM/HIGH",
    "requires_index": true/false,
    "idempotent": true/false
}}

EXAMPLES:

Example 1 - Simple read:
{{
    "query": "MATCH (p:PERSON {{name: $name}}) RETURN p",
    "parameters": {{"name": "John"}},
    "explanation": "Find person by name",
    "query_type": "READ",
    "estimated_complexity": "LOW",
    "requires_index": true,
    "idempotent": true
}}

Example 2 - Create with relationship:
{{
    "query": "MATCH (p:PERSON {{name: $person_name}}), (c:COMPANY {{name: $company_name}}) MERGE (p)-[r:WORKS_FOR]->(c) SET r.since = $since RETURN r",
    "parameters": {{"person_name": "John", "company_name": "Acme", "since": "2020"}},
    "explanation": "Create works-for relationship between person and company",
    "query_type": "WRITE",
    "estimated_complexity": "MEDIUM",
    "requires_index": true,
    "idempotent": true
}}

IMPORTANT:
- Output ONLY valid JSON
- Query must be valid Memgraph Cypher
- Always use parameters, never inline values
- Ensure proper escaping of special characters
"""

CYPHER_GENERATION_SIMPLE_TEMPLATE = """Generate a Memgraph Cypher query for the following request.

SCHEMA: {schema_description}
REQUEST: {query_intent}

RULES:
- Use Memgraph-compatible syntax only
- Use parameters ($name) not inline values
- Prefer MERGE over CREATE for idempotency

OUTPUT (JSON):
{{
    "query": "YOUR_CYPHER_QUERY",
    "parameters": {{}},
    "explanation": "What this query does"
}}

Output ONLY JSON. No other text.
"""

# =============================================================================
# ADVANCED TEMPLATES
# =============================================================================

BATCH_ENTITY_EXTRACTION_TEMPLATE = """Extract entities from multiple documents in a single operation.

DOCUMENTS:
{documents}

ENTITY TYPES:
{entity_types}

OUTPUT FORMAT:
{{
    "results": [
        {{
            "document_id": "doc1",
            "entities": [
                {{"name": "...", "type": "...", "confidence": 0.95}}
            ]
        }}
    ]
}}

Output ONLY JSON.
"""

MULTI_HOP_REASONING_TEMPLATE = """Perform multi-hop reasoning to extract complex relationships.

CONTEXT:
{context}

QUESTION:
{question}

REASONING STEPS:
1. Identify relevant entities
2. Find direct relationships
3. Find indirect (multi-hop) connections
4. Synthesize the answer

OUTPUT FORMAT:
{{
    "reasoning_steps": ["step1", "step2", ...],
    "entities": [...],
    "relationships": [...],
    "answer": "final answer",
    "confidence": 0.85
}}

Output ONLY JSON.
"""

# =============================================================================
# TEMPLATE MANAGER
# =============================================================================

class PromptTemplateManager:
    """
    Manager for prompt templates with variable substitution.
    
    Provides:
    - Template selection based on task type
    - Variable substitution with defaults
    - Template composition
    - Few-shot example formatting
    """
    
    TEMPLATES = {
        # Entity extraction
        "entity_extraction": ENTITY_EXTRACTION_TEMPLATE,
        "entity_extraction_few_shot": ENTITY_EXTRACTION_FEW_SHOT_TEMPLATE,
        
        # Relationship extraction
        "relation_extraction": RELATION_EXTRACTION_TEMPLATE,
        "relation_extraction_with_entities": RELATION_EXTRACTION_WITH_ENTITIES_TEMPLATE,
        
        # Schema validation
        "schema_validation": SCHEMA_VALIDATION_TEMPLATE,
        
        # Cypher generation
        "cypher_generation": CYPHER_GENERATION_TEMPLATE,
        "cypher_generation_simple": CYPHER_GENERATION_SIMPLE_TEMPLATE,
        
        # Advanced
        "batch_entity_extraction": BATCH_ENTITY_EXTRACTION_TEMPLATE,
        "multi_hop_reasoning": MULTI_HOP_REASONING_TEMPLATE,
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template string
            
        Raises:
            ValueError: If template not found
        """
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(cls.TEMPLATES.keys())}")
        return cls.TEMPLATES[template_name]
    
    @classmethod
    def format_template(
        cls,
        template_name: str,
        **kwargs
    ) -> str:
        """Format a template with variables.
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to substitute
            
        Returns:
            Formatted prompt string
        """
        template = cls.get_template(template_name)
        
        # Add default timestamp
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Format entity types as list
        if 'entity_types' in kwargs and isinstance(kwargs['entity_types'], list):
            kwargs['entity_types'] = '\n'.join(f'- {t}' for t in kwargs['entity_types'])
        
        # Format relation types as list
        if 'relation_types' in kwargs and isinstance(kwargs['relation_types'], list):
            kwargs['relation_types'] = '\n'.join(f'- {t}' for t in kwargs['relation_types'])
        
        # Format entities for relation extraction
        if 'entities' in kwargs and isinstance(kwargs['entities'], list):
            if kwargs['entities']:
                kwargs['entities_list'] = '\n'.join(f'- {e}' for e in kwargs['entities'])
            else:
                kwargs['entities_list'] = 'No pre-defined entities. Extract from text.'
        
        # Format examples for few-shot
        if 'examples' in kwargs and isinstance(kwargs['examples'], list):
            kwargs['examples'] = '\n\n'.join(kwargs['examples'])
        
        # Format documents for batch processing
        if 'documents' in kwargs and isinstance(kwargs['documents'], list):
            formatted_docs = []
            for i, doc in enumerate(kwargs['documents']):
                doc_text = doc if isinstance(doc, str) else doc.get('text', '')
                doc_id = f"doc_{i}" if isinstance(doc, str) else doc.get('id', f'doc_{i}')
                formatted_docs.append(f"[{doc_id}]\n{doc_text}")
            kwargs['documents'] = '\n\n---\n\n'.join(formatted_docs)
        
        return template.format(**kwargs)
    
    @classmethod
    def create_entity_extraction_prompt(
        cls,
        text: str,
        entity_types: List[str],
        model: str = "unknown",
        examples: Optional[List[str]] = None,
    ) -> str:
        """Create entity extraction prompt.
        
        Args:
            text: Text to extract entities from
            entity_types: List of allowed entity types
            model: Model name for metadata
            examples: Optional few-shot examples
            
        Returns:
            Formatted prompt
        """
        if examples:
            return cls.format_template(
                "entity_extraction_few_shot",
                text=text,
                entity_types=entity_types,
                model=model,
                examples=examples,
            )
        return cls.format_template(
            "entity_extraction",
            text=text,
            entity_types=entity_types,
            model=model,
        )
    
    @classmethod
    def create_relation_extraction_prompt(
        cls,
        text: str,
        relation_types: List[str],
        entities: Optional[List[str]] = None,
    ) -> str:
        """Create relationship extraction prompt.
        
        Args:
            text: Text to extract relationships from
            relation_types: List of allowed relation types
            entities: Optional list of known entity names
            
        Returns:
            Formatted prompt
        """
        if entities:
            return cls.format_template(
                "relation_extraction_with_entities",
                text=text,
                relation_types=relation_types,
                entities=entities,
            )
        return cls.format_template(
            "relation_extraction",
            text=text,
            relation_types=relation_types,
            entities=entities or [],
        )
    
    @classmethod
    def create_cypher_generation_prompt(
        cls,
        query_intent: str,
        schema_description: str,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        simple: bool = False,
    ) -> str:
        """Create Cypher query generation prompt.
        
        Args:
            query_intent: Natural language description of the query
            schema_description: Description of the graph schema
            node_labels: Available node labels
            relationship_types: Available relationship types
            simple: Use simplified template
            
        Returns:
            Formatted prompt
        """
        template = "cypher_generation_simple" if simple else "cypher_generation"
        
        return cls.format_template(
            template,
            query_intent=query_intent,
            schema_description=schema_description,
            node_labels=node_labels or [],
            relationship_types=relationship_types or [],
        )
    
    @classmethod
    def create_schema_validation_prompt(
        cls,
        data: Dict[str, Any],
        schema_description: str,
        required_fields: List[str],
    ) -> str:
        """Create schema validation prompt.
        
        Args:
            data: Data to validate (will be JSON serialized)
            schema_description: Description of expected schema
            required_fields: List of required field names
            
        Returns:
            Formatted prompt
        """
        import json
        
        return cls.format_template(
            "schema_validation",
            data=json.dumps(data, indent=2),
            schema_description=schema_description,
            required_fields=', '.join(required_fields),
        )
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available template names."""
        return list(cls.TEMPLATES.keys())


# Export templates and manager
__all__ = [
    # Templates
    "ENTITY_EXTRACTION_TEMPLATE",
    "ENTITY_EXTRACTION_FEW_SHOT_TEMPLATE",
    "RELATION_EXTRACTION_TEMPLATE",
    "RELATION_EXTRACTION_WITH_ENTITIES_TEMPLATE",
    "SCHEMA_VALIDATION_TEMPLATE",
    "CYPHER_GENERATION_TEMPLATE",
    "CYPHER_GENERATION_SIMPLE_TEMPLATE",
    "BATCH_ENTITY_EXTRACTION_TEMPLATE",
    "MULTI_HOP_REASONING_TEMPLATE",
    # Manager
    "PromptTemplateManager",
]
