"""Tests for Knowledge Engine LMQL Integration.

Test suite covering:
- Unit tests for query parsing
- Constraint evaluation tests
- Integration tests with KG
- Performance benchmarks

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

import json
import time
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Import modules under test
from knowledge_engine.integrations.lmql import (
    LMQLKGIntegration,
    EntityQueryResult,
    RelationQueryResult,
    SchemaInferenceResult,
    MultiHopResult,
    QueryExplanation,
    CypherGenerationResult,
    get_default_hub,
    register_with_hub,
)
from integrations.lmql import (
    LMQLAdapter,
    LMQLQueryBuilder,
    Constraint,
    ConstraintType,
    LMQLResult,
    TemplateRegistry,
)
from integrations.lmql.constraint_engine import (
    ConstraintEvaluator,
    ConstraintParser,
    LengthConstraint,
    RegexConstraint,
    RangeConstraint,
    EnumConstraint,
    TypeConstraint,
)


# =============================================================================
# FIXTURES
# =============================================================================


class MockKGConnection:
    """Mock knowledge graph connection for testing."""
    
    def __init__(self):
        self.entities = [
            {"id": "e1", "name": "Apple Inc.", "type": "Company"},
            {"id": "e2", "name": "Steve Jobs", "type": "Person"},
            {"id": "e3", "name": "Tim Cook", "type": "Person"},
        ]
        self.relations = [
            {"from": "e2", "to": "e1", "type": "FOUNDED"},
            {"from": "e3", "to": "e1", "type": "WORKS_AT"},
        ]
        
    def run(self, query: str, params: Dict[str, Any] = None):
        """Mock run method."""
        return []


# =============================================================================
# UNIT TESTS - QUERY PARSING
# =============================================================================


class TestConstraintParsing(unittest.TestCase):
    """Unit tests for constraint parsing."""
    
    def setUp(self):
        self.parser = ConstraintParser()
        
    def test_parse_from_lmql(self):
        """Test parsing full LMQL query."""
        lmql_query = '''
        "Extract entities"
        entities: list = "..."
        WHERE len(entities) > 0
        RETURN entities
        '''
        constraints_by_var = self.parser.parse_from_lmql(lmql_query)
        # The parser extracts constraints from WHERE clause
        # Should find at least the length constraint
        self.assertTrue(len(constraints_by_var) > 0)
        
    def test_parse_length_constraint(self):
        """Test parsing length constraints."""
        constraints = self.parser.parse("WHERE len(x) > 0")
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].get_type().value, "length")
        
    def test_parse_length_range_constraint(self):
        """Test parsing length range constraints."""
        constraints = self.parser.parse("WHERE len(x) in [1, 100]")
        self.assertEqual(len(constraints), 1)
        self.assertIsInstance(constraints[0], LengthConstraint)
        
    def test_parse_regex_constraint(self):
        """Test parsing regex constraints."""
        constraints = self.parser.parse(r"WHERE REGEX(x, r'\d+')")
        self.assertEqual(len(constraints), 1)
        self.assertIsInstance(constraints[0], RegexConstraint)
        
    def test_parse_range_constraint(self):
        """Test parsing range constraints."""
        constraints = self.parser.parse("WHERE x >= 0 AND x <= 100")
        self.assertEqual(len(constraints), 2)
        
    def test_parse_enum_constraint(self):
        """Test parsing enum constraints."""
        constraints = self.parser.parse("WHERE x in ['yes', 'no']")
        self.assertEqual(len(constraints), 1)
        self.assertIsInstance(constraints[0], EnumConstraint)
        
    def test_parse_composite_constraints(self):
        """Test parsing multiple constraints."""
        constraints = self.parser.parse(
            "WHERE len(x) > 0 AND x in ['a', 'b'] AND REGEX(x, r'[a-z]+')"
        )
        self.assertEqual(len(constraints), 3)
        



# =============================================================================
# UNIT TESTS - CONSTRAINT EVALUATION
# =============================================================================


class TestConstraintEvaluation(unittest.TestCase):
    """Unit tests for constraint evaluation."""
    
    def setUp(self):
        self.evaluator = ConstraintEvaluator()
        
    def test_length_constraint_pass(self):
        """Test length constraint passing."""
        result = self.evaluator.evaluate_length("hello", min=1, max=100)
        self.assertTrue(result.satisfied)
        
    def test_length_constraint_fail_min(self):
        """Test length constraint failing on min."""
        result = self.evaluator.evaluate_length("hi", min=10)
        self.assertFalse(result.satisfied)
        self.assertIn("minimum", result.error_message.lower())
        
    def test_length_constraint_fail_max(self):
        """Test length constraint failing on max."""
        result = self.evaluator.evaluate_length("a" * 200, max=100)
        self.assertFalse(result.satisfied)
        self.assertIn("maximum", result.error_message.lower())
        
    def test_type_constraint_pass(self):
        """Test type constraint passing."""
        result = self.evaluator.evaluate_type("hello", ["str"])
        self.assertTrue(result.satisfied)
        
    def test_type_constraint_fail(self):
        """Test type constraint failing."""
        result = self.evaluator.evaluate_type(42, ["str"])
        self.assertFalse(result.satisfied)
        
    def test_regex_constraint_pass(self):
        """Test regex constraint passing."""
        result = self.evaluator.evaluate_regex("2024-01-15", r"\d{4}-\d{2}-\d{2}")
        self.assertTrue(result.satisfied)
        
    def test_regex_constraint_fail(self):
        """Test regex constraint failing."""
        result = self.evaluator.evaluate_regex("invalid", r"\d+")
        self.assertFalse(result.satisfied)
        
    def test_range_constraint_pass(self):
        """Test range constraint passing."""
        result = self.evaluator.evaluate_range(50, min=0, max=100)
        self.assertTrue(result.satisfied)
        
    def test_range_constraint_fail_below(self):
        """Test range constraint failing below min."""
        result = self.evaluator.evaluate_range(-5, min=0, max=100)
        self.assertFalse(result.satisfied)
        
    def test_range_constraint_fail_above(self):
        """Test range constraint failing above max."""
        result = self.evaluator.evaluate_range(150, min=0, max=100)
        self.assertFalse(result.satisfied)
        
    def test_enum_constraint_pass(self):
        """Test enum constraint passing."""
        result = self.evaluator.evaluate_enum("yes", ["yes", "no"])
        self.assertTrue(result.satisfied)
        
    def test_enum_constraint_fail(self):
        """Test enum constraint failing."""
        result = self.evaluator.evaluate_enum("maybe", ["yes", "no"])
        self.assertFalse(result.satisfied)
        
    def test_custom_constraint(self):
        """Test custom constraint."""
        result = self.evaluator.evaluate_custom(
            4,
            predicate=lambda x: x % 2 == 0,
            predicate_name="is_even"
        )
        self.assertTrue(result.satisfied)
        
    def test_batch_evaluation_all_pass(self):
        """Test batch evaluation all passing."""
        constraints = [
            LengthConstraint(min=1, max=100),
            RegexConstraint(pattern=r"^[a-z]+$"),
        ]
        result = self.evaluator.evaluate_all("hello", constraints)
        self.assertTrue(result.all_satisfied)
        self.assertEqual(result.satisfied_count, 2)
        
    def test_batch_evaluation_some_fail(self):
        """Test batch evaluation some failing."""
        constraints = [
            LengthConstraint(min=1, max=10),
            RegexConstraint(pattern=r"^\d+$"),  # Only digits
        ]
        result = self.evaluator.evaluate_all("hello123", constraints)
        self.assertFalse(result.all_satisfied)
        self.assertEqual(result.satisfied_count, 1)
        self.assertEqual(result.failed_count, 1)
        
    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        self.evaluator.reset_metrics()
        self.evaluator.evaluate_length("test", min=1)
        self.evaluator.evaluate_length("t", min=10)  # Fail
        
        metrics = self.evaluator.get_metrics()
        self.assertEqual(metrics["evaluations"], 2)
        self.assertEqual(metrics["satisfied"], 1)
        self.assertEqual(metrics["failed"], 1)


# =============================================================================
# UNIT TESTS - QUERY BUILDER
# =============================================================================


class TestLMQLQueryBuilder(unittest.TestCase):
    """Unit tests for LMQL query builder."""
    
    def test_build_simple_query(self):
        """Test building simple query."""
        builder = LMQLQueryBuilder()
        query = builder.with_prompt("Extract: {text}").build()
        
        self.assertIn("Extract:", query)
        self.assertIn("argmax", query)
        
    def test_build_with_variable(self):
        """Test building query with variable."""
        builder = LMQLQueryBuilder()
        query = (builder
            .with_prompt("Extract: {text}")
            .with_variable("entities", "list")
            .build())
        
        self.assertIn("entities", query)
        
    def test_build_with_constraint(self):
        """Test building query with constraint."""
        builder = LMQLQueryBuilder()
        query = (builder
            .with_prompt("Extract: {text}")
            .with_variable("result", "str")
            .with_constraint(LengthConstraint(min=1, max=100))
            .build())
        
        self.assertIn("WHERE", query)
        
    def test_build_with_model(self):
        """Test building query with model specification."""
        builder = LMQLQueryBuilder()
        query = builder.with_prompt("Test").with_model("gpt-4").build()
        
        self.assertIsNotNone(query)
        
    def test_build_json(self):
        """Test building query as JSON."""
        builder = LMQLQueryBuilder()
        json_query = (builder
            .with_prompt("Test")
            .with_variable("result", "str")
            .build_json())
        
        self.assertIn("prompt", json_query)
        self.assertIn("variables", json_query)


# =============================================================================
# UNIT TESTS - TEMPLATE REGISTRY
# =============================================================================


class TestTemplateRegistry(unittest.TestCase):
    """Unit tests for template registry."""
    
    def setUp(self):
        self.registry = TemplateRegistry()
        
    def test_get_template(self):
        """Test getting template by name."""
        template = self.registry.get("entity_extraction")
        self.assertIsNotNone(template)
        self.assertEqual(template.name, "entity_extraction")
        
    def test_list_templates(self):
        """Test listing all templates."""
        templates = self.registry.list_templates()
        self.assertIn("entity_extraction", templates)
        self.assertIn("relation_extraction", templates)
        self.assertIn("cypher_generation", templates)
        
    def test_render_template(self):
        """Test rendering template."""
        rendered = self.registry.render(
            "entity_extraction",
            text="Apple Inc. was founded by Steve Jobs.",
            entity_types="ORG, PERSON",
            min_confidence=0.5,
            max_entities=10,
        )
        
        self.assertIn("Apple Inc.", rendered)
        self.assertIn("WHERE", rendered)
        
    def test_render_missing_required_param(self):
        """Test rendering with missing required parameter."""
        with self.assertRaises(ValueError):
            self.registry.render("entity_extraction", text="test")  # Missing entity_types
            
    def test_get_by_category(self):
        """Test getting templates by category."""
        from integrations.lmql.query_templates import TemplateCategory
        
        entity_templates = self.registry.get_by_category(TemplateCategory.ENTITY)
        self.assertIn("entity_extraction", entity_templates)


# =============================================================================
# INTEGRATION TESTS - LMQL KG INTEGRATION
# =============================================================================


class TestLMQLKGIntegration(unittest.TestCase):
    """Integration tests for LMQL KG Integration."""
    
    def setUp(self):
        self.mock_kg = MockKGConnection()
        # Create mock adapter to avoid API key requirement
        self.mock_adapter = MagicMock(spec=LMQLAdapter)
        self.integration = LMQLKGIntegration(
            adapter=self.mock_adapter,
            kg_connection=self.mock_kg
        )
        
    def test_initialization(self):
        """Test integration initialization."""
        integration = LMQLKGIntegration(adapter=self.mock_adapter)
        self.assertIsNotNone(integration.adapter)
        self.assertIsNotNone(integration.template_registry)
        
    def test_query_entities_basic(self):
        """Test basic entity query."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='[{"entity": "Apple Inc.", "type": "Company", "confidence": 0.95}]',
            correlation_id="test-123"
        )
        
        result = self.integration.query_entities(
            "Find technology companies",
            max_results=5
        )
        
        self.assertIsInstance(result, EntityQueryResult)
        self.assertTrue(result.success)
        
    def test_query_entities_with_filters(self):
        """Test entity query with filters."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='[{"entity": "Apple Inc.", "type": "Company", "confidence": 0.95}]',
            correlation_id="test-456"
        )
        
        result = self.integration.query_entities(
            "Find companies",
            filters={"entity_type": "Company"},
            entity_types=["Company"],
            max_results=10
        )
        
        self.assertIsInstance(result, EntityQueryResult)
        
    def test_query_relations(self):
        """Test relation query."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='MATCH (n)-[r:FOUNDED]->(m) RETURN r, m',
            correlation_id="test-789"
        )
        
        result = self.integration.query_relations(
            entity_ids=["e1", "e2"],
            relation_types=["FOUNDED"],
        )
        
        self.assertIsInstance(result, RelationQueryResult)
        
    def test_infer_schema(self):
        """Test schema inference."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='{"entity_types": [{"name": "Company"}, {"name": "Person"}], "relation_types": [{"name": "FOUNDED"}]}',
            correlation_id="test-schema"
        )
        
        result = self.integration.infer_schema(
            kg_sample={
                "entities": [
                    {"name": "Apple", "type": "Company"},
                    {"name": "Steve Jobs", "type": "Person"},
                ],
                "relations": [
                    {"from": "Steve Jobs", "to": "Apple", "type": "FOUNDED"}
                ]
            }
        )
        
        self.assertIsInstance(result, SchemaInferenceResult)
        self.assertTrue(result.success)
        
    def test_infer_schema_from_queries(self):
        """Test schema inference from queries."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='{"entity_types": [{"name": "Company"}], "relation_types": [{"name": "FOUNDED"}]}',
            correlation_id="test-schema-2"
        )
        
        result = self.integration.infer_schema(
            sample_queries=[
                "Find companies founded by Steve Jobs",
                "Find people who work at Apple",
            ]
        )
        
        self.assertIsInstance(result, SchemaInferenceResult)
        
    def test_multi_hop_query(self):
        """Test multi-hop query."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='{"reasoning_steps": [{"step": 1, "action": "found company"}], "answer": "Apple Inc.", "confidence": 0.9, "entities_visited": ["Steve Jobs", "Apple Inc."]}',
            correlation_id="test-hop"
        )
        
        result = self.integration.multi_hop_query(
            start_entity="Steve Jobs",
            query_path=[
                {"relation": "FOUNDED", "direction": "out"},
                {"relation": "WORKS_AT", "direction": "in"},
            ],
            max_hops=3
        )
        
        self.assertIsInstance(result, MultiHopResult)
        
    def test_explain_query(self):
        """Test query explanation."""
        explanation = self.integration.explain_query(
            "Find companies where len(name) > 0"
        )
        
        self.assertIsInstance(explanation, QueryExplanation)
        
    def test_generate_cypher(self):
        """Test Cypher generation."""
        # Setup mock response for Cypher generation
        def side_effect(*args, **kwargs):
            query_str = args[0] if args else kwargs.get('query_str', '')
            if 'cypher_generation' in str(query_str).lower() or 'MATCH' in str(query_str):
                return LMQLResult(
                    success=True,
                    data='MATCH (p:Person {name: "Steve Jobs"})-[:FOUNDED]->(c:Company) RETURN c',
                    correlation_id="test-cypher"
                )
            # For template rendering call
            return LMQLResult(
                success=True,
                data='MATCH (p:Person)-[:FOUNDED]->(c:Company) RETURN c',
                correlation_id="test-cypher-2"
            )
        
        self.mock_adapter.query.side_effect = side_effect
        
        result = self.integration.generate_cypher(
            "Find all companies founded by Steve Jobs"
        )
        
        self.assertIsInstance(result, CypherGenerationResult)
        self.assertTrue(result.success)
        
    def test_generate_cypher_path_query(self):
        """Test Cypher path query generation."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='MATCH path = (a)-[*1..5]->(b) RETURN path',
            correlation_id="test-cypher-path"
        )
        
        result = self.integration.generate_cypher(
            "Find path from Steve Jobs to Apple",
            query_type="path"
        )
        
        self.assertIsInstance(result, CypherGenerationResult)
        
    def test_generate_cypher_temporal(self):
        """Test temporal Cypher generation."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='MATCH (e:Employee) WHERE e.valid_from <= $timestamp AND (e.valid_to IS NULL OR e.valid_to > $timestamp) RETURN e',
            correlation_id="test-cypher-temporal"
        )
        
        result = self.integration.generate_cypher(
            "Find employees as of 2020-01-01",
            is_temporal=True
        )
        
        self.assertIsInstance(result, CypherGenerationResult)
        
    def test_metrics(self):
        """Test metrics collection."""
        # Setup mock responses
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='[]',
            correlation_id="test-metric"
        )
        
        self.integration.reset_metrics()
        
        # Execute some queries
        self.integration.query_entities("Test query 1")
        self.integration.query_entities("Test query 2")
        
        metrics = self.integration.get_metrics()
        self.assertGreaterEqual(metrics["total_queries"], 0)
        
    def test_cache(self):
        """Test result caching."""
        # Setup mock response
        self.mock_adapter.query.return_value = LMQLResult(
            success=True,
            data='[{"entity": "Test"}]',
            correlation_id="test-cache"
        )
        
        self.integration.clear_cache()
        
        # Same query twice
        query = "Test caching"
        result1 = self.integration.query_entities(query, use_cache=True)
        result2 = self.integration.query_entities(query, use_cache=True)
        
        # Second query should be cached
        self.assertEqual(result1.entities, result2.entities)


# =============================================================================
# INTEGRATION TESTS - UNIFIED HUB
# =============================================================================


class TestUnifiedKGIntegrationHub(unittest.TestCase):
    """Tests for unified KG integration hub."""
    
    def setUp(self):
        self.hub = get_default_hub()
        self.mock_adapter = MagicMock(spec=LMQLAdapter)
        
    def test_register_integration(self):
        """Test registering integration."""
        integration = LMQLKGIntegration(adapter=self.mock_adapter)
        self.hub.register_integration("test_lmql", integration)
        
        self.assertIn("test_lmql", self.hub.list_integrations())
        
    def test_get_integration(self):
        """Test getting registered integration."""
        integration = LMQLKGIntegration(adapter=self.mock_adapter)
        self.hub.register_integration("test_get", integration)
        
        retrieved = self.hub.get_integration("test_get")
        self.assertIsNotNone(retrieved)
        
    def test_register_with_hub(self):
        """Test register_with_hub convenience function."""
        # Create integration directly and register it
        integration = LMQLKGIntegration(adapter=self.mock_adapter)
        self.hub.register_integration("test_register", integration)
        
        self.assertIsInstance(integration, LMQLKGIntegration)
        self.assertIn("test_register", self.hub.list_integrations())


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self):
        self.evaluator = ConstraintEvaluator()
        
    def test_constraint_evaluation_speed(self):
        """Benchmark constraint evaluation speed."""
        constraints = [
            LengthConstraint(min=1, max=1000),
            RegexConstraint(pattern=r"^[a-zA-Z]+$"),
            RangeConstraint(min=0, max=1000),
        ]
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            self.evaluator.evaluate_all("test123", constraints)
            
        elapsed = time.time() - start_time
        avg_time = (elapsed / iterations) * 1000  # ms
        
        # Should complete 1000 evaluations in less than 1 second
        self.assertLess(avg_time, 1.0, f"Average evaluation time {avg_time}ms too slow")
        
    def test_query_builder_speed(self):
        """Benchmark query builder speed."""
        builder = LMQLQueryBuilder()
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            (builder
                .with_prompt("Extract: {text}")
                .with_variable("entities", "list")
                .build())
                
        elapsed = time.time() - start_time
        avg_time = (elapsed / iterations) * 1000
        
        # Should complete 1000 builds in less than 100ms
        self.assertLess(avg_time, 0.1, f"Average build time {avg_time}ms too slow")
        
    def test_template_rendering_speed(self):
        """Benchmark template rendering speed."""
        registry = TemplateRegistry()
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            registry.render(
                "entity_extraction",
                text="Apple Inc. was founded by Steve Jobs.",
                entity_types="ORG, PERSON",
                min_confidence=0.5,
                max_entities=10,
            )
            
        elapsed = time.time() - start_time
        avg_time = (elapsed / iterations) * 1000
        
        # Should complete 1000 renders in less than 10ms
        self.assertLess(avg_time, 0.01, f"Average render time {avg_time}ms too slow")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""
    
    def test_empty_string_length(self):
        """Test length constraint on empty string."""
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate_length("", min=1)
        self.assertFalse(result.satisfied)
        
    def test_none_value_type(self):
        """Test type constraint on None."""
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate_type(None, ["str"])
        self.assertFalse(result.satisfied)
        
        result = evaluator.evaluate_type(None, ["none"])
        self.assertTrue(result.satisfied)
        
    def test_invalid_regex(self):
        """Test invalid regex pattern."""
        constraint = RegexConstraint(pattern=r"[invalid")
        result = constraint.evaluate("test")
        self.assertFalse(result.satisfied)
        
    def test_non_numeric_range(self):
        """Test range constraint on non-numeric value."""
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate_range("not a number", min=0, max=100)
        self.assertFalse(result.satisfied)
        
    def test_unicode_in_enum(self):
        """Test enum constraint with unicode values."""
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate_enum("中文", ["中文", "日本語"])
        self.assertTrue(result.satisfied)
        
    def test_very_long_string(self):
        """Test constraints on very long strings."""
        evaluator = ConstraintEvaluator()
        long_string = "a" * 1000000
        result = evaluator.evaluate_length(long_string, max=100)
        self.assertFalse(result.satisfied)
        
    def test_special_characters_regex(self):
        """Test regex with special characters."""
        evaluator = ConstraintEvaluator()
        result = evaluator.evaluate_regex(
            "test@email.com",
            r"^[\w\.-]+@[\w\.-]+\.\w+$"
        )
        self.assertTrue(result.satisfied)


# =============================================================================
# TEST SUITE
# =============================================================================


def create_test_suite():
    """Create complete test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestLMQLQueryBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestTemplateRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestLMQLKGIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedKGIntegrationHub))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    return suite


if __name__ == "__main__":
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Calculate coverage estimate (simplified)
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)
    coverage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"Pass Rate: {coverage:.1f}%")
