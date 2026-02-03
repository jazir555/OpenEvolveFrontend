"""
Test Suite for Knowledge Engine Schema System

Tests schema registration, entity validation, mapping, and cross-domain integration.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.base import (
    Entity,
    Relationship,
    PropertyDefinition,
    PropertyType,
    EntityType,
    EntitySchema,
    RelationshipType
)
from schemas.entity_schema_manager import EntitySchemaManager, ValidationResult
from schemas.validators import SchemaValidator
from schemas.openevolve_schemas import (
    SOFTWARE_ENGINEERING_SCHEMA,
    MATHEMATICAL_REASONING_SCHEMA,
    WORKFLOW_PROVENANCE_SCHEMA
)
from schemas.schema_mappings import (
    KNOWLEDGE_ENGINE_TO_GRAPHITI,
    ENTITY_MAPPINGS,
    get_mapping,
    apply_mapping
)


# =============================================================================
# TEST: BASE CLASSES
# =============================================================================

class TestPropertyDefinition:
    """Test PropertyDefinition validation."""

    def test_string_property_validation(self):
        """Test validation of string properties."""
        prop = PropertyDefinition(
            name="test_prop",
            type=PropertyType.STRING,
            required=True
        )

        # Valid string
        is_valid, error = prop.validate("test_value")
        assert is_valid
        assert error is None

        # Missing required value
        is_valid, error = prop.validate(None)
        assert not is_valid
        assert "required" in error.lower()

    def test_integer_property_validation(self):
        """Test validation of integer properties."""
        prop = PropertyDefinition(
            name="age",
            type=PropertyType.INTEGER,
            required=True,
            min_value=0,
            max_value=150
        )

        # Valid integer
        is_valid, error = prop.validate(25)
        assert is_valid

        # Below minimum
        is_valid, error = prop.validate(-1)
        assert not is_valid
        assert "minimum" in error.lower()

        # Above maximum
        is_valid, error = prop.validate(200)
        assert not is_valid
        assert "maximum" in error.lower()

    def test_enum_property_validation(self):
        """Test validation of enum properties."""
        prop = PropertyDefinition(
            name="status",
            type=PropertyType.ENUM,
            required=True,
            allowed_values=["active", "inactive", "pending"]
        )

        # Valid value
        is_valid, error = prop.validate("active")
        assert is_valid

        # Invalid value
        is_valid, error = prop.validate("unknown")
        assert not is_valid
        assert "allowed" in error.lower()


class TestEntityType:
    """Test EntityType validation."""

    def test_entity_type_validation(self):
        """Test entity validation against type definition."""
        # Define a simple entity type
        user_type = EntityType(
            name="User",
            properties={
                "username": PropertyDefinition(
                    name="username",
                    type=PropertyType.STRING,
                    required=True
                ),
                "age": PropertyDefinition(
                    name="age",
                    type=PropertyType.INTEGER,
                    required=False
                )
            }
        )

        # Valid entity
        is_valid, errors = user_type.validate({
            "username": "john_doe",
            "age": 30
        })
        assert is_valid
        assert len(errors) == 0

        # Missing required property
        is_valid, errors = user_type.validate({
            "age": 30
        })
        assert not is_valid
        assert len(errors) > 0

    def test_entity_type_with_validation_rules(self):
        """Test entity type with custom validation rules."""
        from schemas.base import ValidationRule

        # Define custom rule
        def username_not_admin(entity_data):
            username = entity_data.get("username", "")
            if username.lower() == "admin":
                return False, "Username cannot be 'admin'"
            return True, None

        user_type = EntityType(
            name="User",
            properties={
                "username": PropertyDefinition(
                    name="username",
                    type=PropertyType.STRING,
                    required=True
                )
            },
            validation_rules=[
                ValidationRule(
                    name="username_not_admin",
                    description="Username cannot be admin",
                    validator=username_not_admin,
                    severity="error"
                )
            ]
        )

        # Valid username
        is_valid, errors = user_type.validate({"username": "john_doe"})
        assert is_valid

        # Invalid username
        is_valid, errors = user_type.validate({"username": "admin"})
        assert not is_valid
        assert any("admin" in error.lower() for error in errors)


# =============================================================================
# TEST: SCHEMA MANAGER
# =============================================================================

class TestEntitySchemaManager:
    """Test EntitySchemaManager functionality."""

    def test_schema_registration(self):
        """Test registering schemas."""
        manager = EntitySchemaManager()

        # Register software engineering schema
        manager.register_schema(
            'software_engineering',
            SOFTWARE_ENGINEERING_SCHEMA.to_dict()
        )

        # Verify schema was registered
        assert 'software_engineering' in manager.list_schemas()

        # Retrieve schema
        schema = manager.get_schema('software_engineering')
        assert schema is not None
        assert schema.domain == 'software_engineering'

    def test_entity_validation(self):
        """Test entity validation."""
        manager = EntitySchemaManager()

        # Register schema
        manager.register_schema(
            'software_engineering',
            SOFTWARE_ENGINEERING_SCHEMA.to_dict()
        )

        # Create valid entity
        entity = Entity(
            entity_id="func-001",
            entity_type="CodeEntity",
            properties={
                "name": "calculate_hash",
                "code_type": "function",
                "file_path": "src/utils/crypto.py",
                "language": "Python"
            }
        )

        # Validate
        result = manager.validate_entity(entity, 'software_engineering')
        assert result.is_valid
        assert result.valid_count == 1

    def test_invalid_entity_validation(self):
        """Test validation of invalid entity."""
        manager = EntitySchemaManager()

        # Register schema
        manager.register_schema(
            'software_engineering',
            SOFTWARE_ENGINEERING_SCHEMA.to_dict()
        )

        # Create entity with missing required property
        entity = Entity(
            entity_id="func-002",
            entity_type="CodeEntity",
            properties={
                # Missing "name" (required)
                "code_type": "function"
            }
        )

        # Validate
        result = manager.validate_entity(entity, 'software_engineering')
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_batch_validation(self):
        """Test batch entity validation."""
        manager = EntitySchemaManager()

        # Register schema
        manager.register_schema(
            'software_engineering',
            SOFTWARE_ENGINEERING_SCHEMA.to_dict()
        )

        # Create entities
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
        assert result.is_valid
        assert result.entity_count == 10
        assert result.valid_count == 10

    def test_entity_mapping(self):
        """Test entity mapping between schemas."""
        manager = EntitySchemaManager()

        # Register schemas
        manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
        manager.register_mapping('knowledge_engine_to_graphiti', KNOWLEDGE_ENGINE_TO_GRAPHITI)

        # Create source entities
        source_entities = [
            Entity(
                entity_id="code-001",
                entity_type="CodeEntity",
                properties={"name": "test_function", "code_type": "function"},
                source="software_engineering"
            )
        ]

        # Map entities (this would require Graphiti schema to be registered)
        # For now, just test that the mapping exists
        mapping = get_mapping('knowledge_engine_to_graphiti')
        assert 'CodeEntity' in mapping
        assert mapping['CodeEntity'] == 'Activity'

    def test_cross_domain_merge(self):
        """Test merging entities from different domains."""
        manager = EntitySchemaManager()

        # Register schemas
        manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
        manager.register_schema('workflow_provenance', WORKFLOW_PROVENANCE_SCHEMA.to_dict())

        # Create entities from different domains
        software_entities = [
            Entity(
                entity_id="entity-001",
                entity_type="CodeEntity",
                properties={"name": "test", "code_type": "function"},
                source="software_engineering"
            )
        ]

        workflow_entities = [
            Entity(
                entity_id="entity-001",  # Same ID
                entity_type="TaskEntity",
                properties={"name": "test_task", "task_type": "analysis"},
                source="workflow_provenance"
            )
        ]

        # Merge entities
        merged = manager.merge_cross_domain([
            (software_entities, 'software_engineering'),
            (workflow_entities, 'workflow_provenance')
        ])

        # Should have 1 entity (merged by ID)
        assert len(merged) == 1
        assert merged[0].entity_id == "entity-001"
        assert 'software_engineering' in merged[0].metadata.get('sources', [])

    def test_schema_prompt_generation(self):
        """Test generating schema prompts for LLM."""
        manager = EntitySchemaManager()

        # Register schema
        manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())

        # Generate prompt
        prompt = manager.generate_schema_prompt('software_engineering')

        # Verify prompt contains expected content
        assert 'CodeEntity' in prompt
        assert 'function' in prompt
        assert 'properties' in prompt.lower()


# =============================================================================
# TEST: SCHEMA VALIDATORS
# =============================================================================

class TestSchemaValidator:
    """Test SchemaValidator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = SchemaValidator(SOFTWARE_ENGINEERING_SCHEMA)
        assert validator.schema.domain == "software_engineering"

    def test_entity_validation(self):
        """Test entity validation with validator."""
        validator = SchemaValidator(SOFTWARE_ENGINEERING_SCHEMA)

        entity = Entity(
            entity_id="code-001",
            entity_type="CodeEntity",
            properties={
                "name": "test_function",
                "code_type": "function",
                "language": "Python"
            }
        )

        result = validator.validate_entity(entity)
        assert result.is_valid

    def test_relationship_validation(self):
        """Test relationship validation."""
        validator = SchemaValidator(SOFTWARE_ENGINEERING_SCHEMA)

        relationship = Relationship(
            relationship_id="rel-001",
            source_entity_id="func-001",
            target_entity_id="func-002",
            relationship_type="calls"
        )

        result = validator.validate_relationship(
            relationship,
            source_entity_type="CodeEntity",
            target_entity_type="CodeEntity"
        )
        assert result.is_valid

    def test_invalid_relationship_validation(self):
        """Test validation of invalid relationship."""
        validator = SchemaValidator(SOFTWARE_ENGINEERING_SCHEMA)

        # Invalid: calls relationship requires CodeEntity source and target
        relationship = Relationship(
            relationship_id="rel-002",
            source_entity_id="api-001",
            target_entity_id="func-001",
            relationship_type="calls"
        )

        result = validator.validate_relationship(
            relationship,
            source_entity_type="APISchema",  # Wrong type
            target_entity_type="CodeEntity"
        )
        assert not result.is_valid

    def test_batch_validation(self):
        """Test batch validation."""
        validator = SchemaValidator(SOFTWARE_ENGINEERING_SCHEMA)

        entities = [
            Entity(
                entity_id=f"code-{i:03d}",
                entity_type="CodeEntity",
                properties={
                    "name": f"function_{i}",
                    "code_type": "function",
                    "language": "Python"
                }
            )
            for i in range(1, 6)
        ]

        result = validator.validate_batch(entities)
        assert result.is_valid
        assert result.entity_count == 5
        assert result.valid_count == 5


# =============================================================================
# TEST: OPENEVOLVE SCHEMAS
# =============================================================================

class TestOpenEvolveSchemas:
    """Test OpenEvolve-specific schemas."""

    def test_software_engineering_schema(self):
        """Test software engineering schema."""
        schema = SOFTWARE_ENGINEERING_SCHEMA

        assert schema.domain == "software_engineering"
        assert "CodeEntity" in schema.entity_types
        assert "DependencyEntity" in schema.entity_types
        assert "APISchema" in schema.entity_types
        assert "BugPattern" in schema.entity_types

        # Check relationship types
        assert "calls" in schema.relationship_types
        assert "imports" in schema.relationship_types

    def test_mathematical_reasoning_schema(self):
        """Test mathematical reasoning schema."""
        schema = MATHEMATICAL_REASONING_SCHEMA

        assert schema.domain == "mathematical_reasoning"
        assert "TheoremEntity" in schema.entity_types
        assert "ConceptEntity" in schema.entity_types
        assert "TechniqueEntity" in schema.entity_types
        assert "ProofStepEntity" in schema.entity_types

        # Check relationship types
        assert "uses" in schema.relationship_types
        assert "generalizes" in schema.relationship_types

    def test_workflow_provenance_schema(self):
        """Test workflow provenance schema."""
        schema = WORKFLOW_PROVENANCE_SCHEMA

        assert schema.domain == "workflow_provenance"
        assert "WorkflowEntity" in schema.entity_types
        assert "TaskEntity" in schema.entity_types
        assert "AgentEntity" in schema.entity_types
        assert "ExecutionEntity" in schema.entity_types

        # Check relationship types
        assert "contains" in schema.relationship_types
        assert "executed_by" in schema.relationship_types


# =============================================================================
# TEST: SCHEMA MAPPINGS
# =============================================================================

class TestSchemaMappings:
    """Test schema mapping functionality."""

    def test_get_mapping(self):
        """Test retrieving mappings."""
        mapping = get_mapping('knowledge_engine_to_graphiti')

        assert isinstance(mapping, dict)
        assert 'CodeEntity' in mapping
        assert mapping['CodeEntity'] == 'Activity'

    def test_apply_mapping(self):
        """Test applying mapping to entity type."""
        mapping = KNOWLEDGE_ENGINE_TO_GRAPHITI

        mapped_type = apply_mapping('CodeEntity', mapping)
        assert mapped_type == 'Activity'

        # Test unknown type (should return original)
        mapped_type = apply_mapping('UnknownType', mapping)
        assert mapped_type == 'UnknownType'

    def test_entity_mappings_collection(self):
        """Test ENTITY_MAPPINGS collection."""
        assert 'knowledge_engine_to_graphiti' in ENTITY_MAPPINGS
        assert 'knowledge_engine_to_oneke' in ENTITY_MAPPINGS
        assert 'openevolve_to_neo4j' in ENTITY_MAPPINGS


# =============================================================================
# TEST: INTEGRATION TESTS
# =============================================================================

class TestSchemaIntegration:
    """Integration tests for schema system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: register, validate, map, merge."""
        manager = EntitySchemaManager()

        # 1. Register schemas
        manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA.to_dict())
        manager.register_schema('workflow_provenance', WORKFLOW_PROVENANCE_SCHEMA.to_dict())

        # 2. Create and validate entities
        entities = [
            Entity(
                entity_id="func-001",
                entity_type="CodeEntity",
                properties={
                    "name": "main",
                    "code_type": "function",
                    "language": "Python"
                }
            ),
            Entity(
                entity_id="task-001",
                entity_type="TaskEntity",
                properties={
                    "name": "Analyze code",
                    "task_type": "analysis",
                    "status": "completed"
                }
            )
        ]

        # 3. Validate
        se_result = manager.validate_entity(entities[0], 'software_engineering')
        assert se_result.is_valid

        wf_result = manager.validate_entity(entities[1], 'workflow_provenance')
        assert wf_result.is_valid

        # 4. Merge cross-domain
        # (Entities have different IDs, so both should be present)
        merged = manager.merge_cross_domain([
            ([entities[0]], 'software_engineering'),
            ([entities[1]], 'workflow_provenance')
        ])
        assert len(merged) == 2

    def test_schema_export_import(self):
        """Test exporting and importing schemas."""
        import tempfile
        import os

        manager = EntitySchemaManager()

        # Register schema
        manager.register_schema('test_schema', SOFTWARE_ENGINEERING_SCHEMA.to_dict())

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            manager.export_schema('test_schema', temp_path)
            assert os.path.exists(temp_path)

            # Import into new manager
            new_manager = EntitySchemaManager()
            new_manager.import_schema('imported_schema', temp_path)

            # Verify imported schema
            schema = new_manager.get_schema('imported_schema')
            assert schema is not None
            assert schema.domain == 'imported_schema'
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("KNOWLEDGE ENGINE SCHEMA SYSTEM TEST SUITE")
    print("="*70 + "\n")

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
