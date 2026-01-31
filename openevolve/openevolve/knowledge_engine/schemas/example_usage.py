"""
Example Usage of the Knowledge Engine Schema System

Demonstrates common operations with the schema management system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.schemas import (
    EntitySchemaManager,
    Entity,
    SOFTWARE_ENGINEERING_SCHEMA,
    MATHEMATICAL_REASONING_SCHEMA,
    WORKFLOW_PROVENANCE_SCHEMA
)


def example_1_basic_validation():
    """Example 1: Basic entity validation."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Entity Validation")
    print("="*70)

    # Initialize manager
    manager = EntitySchemaManager()

    # Register schema
    manager.register_schema(
        'software_engineering',
        SOFTWARE_ENGINEERING_SCHEMA.to_dict()
    )

    # Create a valid entity
    entity = Entity(
        entity_id="func-001",
        entity_type="CodeEntity",
        properties={
            "name": "calculate_hash",
            "code_type": "function",
            "signature": "calculate_hash(data: bytes, algorithm: str) -> str",
            "file_path": "src/utils/crypto.py",
            "line_start": 42,
            "line_end": 58,
            "language": "Python",
            "complexity": 3
        }
    )

    # Validate
    result = manager.validate_entity(entity, 'software_engineering')

    print(f"\nEntity: {entity.entity_id}")
    print(f"Type: {entity.entity_type}")
    print(f"Valid: {result.is_valid}")
    print(f"Valid count: {result.valid_count}")
    print(f"Invalid count: {result.invalid_count}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")


def example_2_batch_validation():
    """Example 2: Batch entity validation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Entity Validation")
    print("="*70)

    manager = EntitySchemaManager()
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())

    # Create multiple entities
    entities = [
        Entity(
            entity_id=f"func-{i:03d}",
            entity_type="CodeEntity",
            properties={
                "name": f"function_{i}",
                "code_type": "function",
                "language": "Python"
            }
        )
        for i in range(1, 11)
    ]

    # Validate batch
    result = manager.validate_entities(entities, 'software_engineering')

    print(f"\nTotal entities: {result.entity_count}")
    print(f"Valid: {result.valid_count}")
    print(f"Invalid: {result.invalid_count}")
    print(f"Is valid: {result.is_valid}")


def example_3_cross_domain_integration():
    """Example 3: Cross-domain entity merging."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Cross-Domain Integration")
    print("="*70)

    manager = EntitySchemaManager()

    # Register multiple schemas
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
    manager.register_schema('workflow_provenance', WORKFLOW_PROVENANCE_SCHEMA.to_dict())

    # Create entities from different domains
    software_entities = [
        Entity(
            entity_id="calc-001",
            entity_type="CodeEntity",
            properties={
                "name": "calculate_hash",
                "code_type": "function",
                "language": "Python"
            },
            source="software_engineering"
        ),
        Entity(
            entity_id="utils-001",
            entity_type="CodeEntity",
            properties={
                "name": "format_bytes",
                "code_type": "function",
                "language": "Python"
            },
            source="software_engineering"
        )
    ]

    workflow_entities = [
        Entity(
            entity_id="task-001",
            entity_type="TaskEntity",
            properties={
                "name": "Analyze code",
                "task_type": "analysis",
                "status": "completed"
            },
            source="workflow_provenance"
        ),
        Entity(
            entity_id="agent-001",
            entity_type="AgentEntity",
            properties={
                "name": "Code Analyzer",
                "agent_type": "automated",
                "status": "idle"
            },
            source="workflow_provenance"
        )
    ]

    # Merge cross-domain
    merged = manager.merge_cross_domain([
        (software_entities, 'software_engineering'),
        (workflow_entities, 'workflow_provenance')
    ])

    print(f"\nOriginal software entities: {len(software_entities)}")
    print(f"Original workflow entities: {len(workflow_entities)}")
    print(f"Merged entities: {len(merged)}")
    print(f"\nMerged entity sources:")
    for entity in merged:
        sources = entity.metadata.get('sources', [])
        print(f"  - {entity.entity_id}: {sources}")


def example_4_entity_mapping():
    """Example 4: Entity mapping between schemas."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Entity Mapping")
    print("="*70)

    from knowledge_engine.schemas.schema_mappings import (
        KNOWLEDGE_ENGINE_TO_GRAPHITI,
        apply_mapping
    )

    # Show available mappings
    print("\nAvailable mappings:")
    for entity_type in ['CodeEntity', 'TheoremEntity', 'WorkflowEntity']:
        mapped = apply_mapping(entity_type, KNOWLEDGE_ENGINE_TO_GRAPHITI)
        print(f"  {entity_type} -> {mapped}")

    # Create manager with mapping
    manager = EntitySchemaManager()
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
    manager.register_mapping('knowledge_engine_to_graphiti', KNOWLEDGE_ENGINE_TO_GRAPHITI)

    print(f"\nRegistered mappings: {manager.list_mappings()}")


def example_5_llm_prompt_generation():
    """Example 5: Generate LLM extraction prompt."""
    print("\n" + "="*70)
    print("EXAMPLE 5: LLM Prompt Generation")
    print("="*70)

    manager = EntitySchemaManager()
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())

    # Generate prompt
    prompt = manager.generate_schema_prompt('software_engineering', include_examples=False)

    # Show first 500 characters of prompt
    print(f"\nGenerated prompt (first 500 chars):\n")
    print(prompt[:500] + "...")


def example_6_schema_statistics():
    """Example 6: Get schema statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Schema Statistics")
    print("="*70)

    manager = EntitySchemaManager()

    # Register all schemas
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
    manager.register_schema('mathematical_reasoning', MATHEMATICAL_REASONING_SCHEMA.to_dict())
    manager.register_schema('workflow_provenance', WORKFLOW_PROVENANCE_SCHEMA.to_dict())

    # Get statistics
    stats = manager.get_statistics()

    print(f"\nTotal schemas: {stats['total_schemas']}")
    print("\nSchema details:")
    for domain, domain_stats in stats['schemas'].items():
        print(f"\n  {domain}:")
        print(f"    Entity types: {domain_stats['entity_types']}")
        print(f"    Relationship types: {domain_stats['relationship_types']}")
        print(f"    Total properties: {domain_stats['total_properties']}")
        print(f"    Version: {domain_stats['version']}")


def example_7_invalid_entity():
    """Example 7: Handling invalid entities."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Handling Invalid Entities")
    print("="*70)

    manager = EntitySchemaManager()
    manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())

    # Create entity with missing required property
    invalid_entity = Entity(
        entity_id="func-invalid",
        entity_type="CodeEntity",
        properties={
            # Missing "name" which is required
            "code_type": "function",
            "language": "Python"
        }
    )

    # Validate
    result = manager.validate_entity(invalid_entity, 'software_engineering')

    print(f"\nEntity: {invalid_entity.entity_id}")
    print(f"Valid: {result.is_valid}")
    print(f"\nValidation errors:")
    for error in result.errors:
        print(f"  - {error}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("KNOWLEDGE ENGINE SCHEMA SYSTEM - USAGE EXAMPLES")
    print("="*70)

    try:
        example_1_basic_validation()
        example_2_batch_validation()
        example_3_cross_domain_integration()
        example_4_entity_mapping()
        example_5_llm_prompt_generation()
        example_6_schema_statistics()
        example_7_invalid_entity()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
