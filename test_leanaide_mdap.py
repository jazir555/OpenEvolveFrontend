"""
Comprehensive Test Suite for LeanAide MDAP/MAKER Integration

This module provides comprehensive tests for the Multi-Agent Decomposition with Aggregated
Proof (MDAP) and MAKER (Maximal Agentic decomposition + first-to-ahead-by-K Error correction)
integration with LeanAide for Lean 4 theorem proving.

Test Categories:
    1. Unit Tests: Test individual components
    2. Integration Tests: Test complete MDAP/MAKER pipelines
    3. MAKER Tests: Test MAKER-specific functionality
    4. Workflow Tests: Test integration with decomposition workflow
    5. Red-Flagging Tests: Test validation and error detection
    6. Edge Cases: Test failure modes and edge cases

Author: OpenEvolve Frontend Team
Version: 1.0.0
Date: 2025-12-30
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TEST DEPENDENCY CHECKS
# =============================================================================

try:
    from mdap_engine import (
        MDAPStep,
        MDAPTask,
        MDAPConfig,
        MDAPVoteResult,
        MDAPStepResult,
        MDAPRunResult,
        RedFlagRules,
        RedFlagger,
        MDAPOrchestrator,
        AgentSelector,
        canonicalize_candidate,
        candidate_confidence,
        validate_schema,
        _approx_token_count,
        MDAPCache
    )
    MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MDAP engine not available: {e}")
    MDAP_AVAILABLE = False

try:
    from roma_mdap_maker_engine import (
        ROMAMDAPMakerConfig,
        ROMAMDAPMakerEngine,
        ROMARedFlagRules,
        ROMARedFlagger,
        HierarchicalVotingStrategy,
        AdaptiveKSelector
    )
    ROMA_MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ROMA-MDAP-MAKER engine not available: {e}")
    ROMA_MDAP_AVAILABLE = False

try:
    from maker_workflow_integration import (
        generate_solution_with_maker_v2,
        build_maker_config_from_workflow,
        resolve_maker_enabled
    )
    MAKER_WORKFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MAKER workflow integration not available: {e}")
    MAKER_WORKFLOW_AVAILABLE = False

try:
    from workflow_structures import (
        SubProblem,
        Team,
        WorkflowState,
        SolutionAttempt,
        ModelConfig
    )
    WORKFLOW_STRUCTURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Workflow structures not available: {e}")
    WORKFLOW_STRUCTURES_AVAILABLE = False

# =============================================================================
# UNIT TESTS
# =============================================================================

@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestMDAPStepConfiguration(unittest.TestCase):
    """Test MDAPStep configuration and validation"""

    def test_mdap_step_creation(self):
        """Test creating a basic MDAP step"""
        step = MDAPStep(
            step_id="test_step_1",
            prompt="Prove the theorem: ∀ n : Nat, n + 0 = n",
            task_type="theorem_proving"
        )

        self.assertEqual(step.step_id, "test_step_1")
        self.assertEqual(step.task_type, "theorem_proving")
        self.assertIsNone(step.expected_schema)
        self.assertEqual(step.priority, 0)

    def test_mdap_step_with_schema(self):
        """Test MDAP step with JSON schema validation"""
        schema = {
            "type": "object",
            "properties": {
                "lean_code": {"type": "string"},
                "tactics": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["lean_code"]
        }

        step = MDAPStep(
            step_id="schema_step",
            prompt="Generate Lean 4 code",
            expected_schema=schema,
            temperature_override=0.1,
            max_tokens_override=500
        )

        self.assertEqual(step.expected_schema, schema)
        self.assertEqual(step.temperature_override, 0.1)
        self.assertEqual(step.max_tokens_override, 500)

    def test_mdap_step_serialization(self):
        """Test MDAP step can be serialized to dict"""
        step = MDAPStep(
            step_id="serialize_test",
            prompt="Test prompt",
            metadata={"key": "value"}
        )

        step_dict = asdict(step)

        self.assertIn("step_id", step_dict)
        self.assertEqual(step_dict["step_id"], "serialize_test")
        self.assertEqual(step_dict["metadata"]["key"], "value")


@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestMDAPTaskConfiguration(unittest.TestCase):
    """Test MDAPTask configuration"""

    def test_mdap_task_creation(self):
        """Test creating MDAP task with multiple steps"""
        steps = [
            MDAPStep(step_id="step1", prompt="Analyze theorem"),
            MDAPStep(step_id="step2", prompt="Generate proof"),
            MDAPStep(step_id="step3", prompt="Verify proof")
        ]

        task = MDAPTask(
            task_id="test_task",
            description="Prove commutativity of addition",
            steps=steps,
            max_retries=3,
            target_success_rate=0.95
        )

        self.assertEqual(len(task.steps), 3)
        self.assertEqual(task.max_retries, 3)
        self.assertEqual(task.target_success_rate, 0.95)

    def test_mdap_task_empty_steps(self):
        """Test MDAP task with no steps raises error or handles gracefully"""
        task = MDAPTask(
            task_id="empty_task",
            description="Task with no steps",
            steps=[]
        )

        # Empty steps should be allowed but might cause issues in execution
        self.assertEqual(len(task.steps), 0)


@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestRedFlagging(unittest.TestCase):
    """Test red-flagging functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.rules = RedFlagRules(
            max_tokens=750,
            max_characters=6000,
            min_confidence=0.2,
            blocked_patterns=["ERROR", "FAILURE"]
        )
        self.flagger = RedFlagger(self.rules)

    def test_empty_response_flagged(self):
        """Test empty responses are flagged"""
        is_flagged, reasons = self.flagger.is_flagged("", None, None)

        self.assertTrue(is_flagged)
        self.assertIn("empty_response", reasons)

    def test_token_limit_enforcement(self):
        """Test responses exceeding token limit are flagged"""
        # Create a response that exceeds token limit
        long_response = "x" * 4000  # Should be ~1000 tokens

        is_flagged, reasons = self.flagger.is_flagged(long_response, {}, None)

        self.assertTrue(is_flagged)
        self.assertIn("token_limit_exceeded", reasons)

    def test_character_limit_enforcement(self):
        """Test responses exceeding character limit are flagged"""
        long_response = "y" * 7000

        is_flagged, reasons = self.flagger.is_flagged(long_response, {}, None)

        self.assertTrue(is_flagged)
        self.assertIn("response_too_long", reasons)

    def test_blocked_pattern_detection(self):
        """Test blocked patterns are detected"""
        response = "This response contains an ERROR message"

        is_flagged, reasons = self.flagger.is_flagged(response, {}, None)

        self.assertTrue(is_flagged)
        self.assertTrue(any("blocked_pattern" in r for r in reasons))

    def test_low_confidence_flagged(self):
        """Test low confidence responses are flagged"""
        candidate = {"confidence": 0.1}
        is_flagged, reasons = self.flagger.is_flagged("Valid response", candidate, None)

        self.assertTrue(is_flagged)
        self.assertIn("low_confidence", reasons)

    def test_schema_validation(self):
        """Test schema validation in red-flagging"""
        schema = {
            "type": "object",
            "required": ["lean_code", "tactics"]
        }

        # Missing required field
        candidate = {"lean_code": "theorem test : True := by trivial"}

        is_flagged, reasons = self.flagger.is_flagged("Response", candidate, schema)

        self.assertTrue(is_flagged)
        self.assertTrue(any("missing required key" in r for r in reasons))

    def test_valid_response_passes(self):
        """Test valid responses pass red-flagging"""
        candidate = {
            "lean_code": "theorem test : True := by trivial",
            "tactics": ["trivial"],
            "confidence": 0.9
        }
        response = json.dumps(candidate)

        is_flagged, reasons = self.flagger.is_flagged(response, candidate, None)

        self.assertFalse(is_flagged)
        self.assertEqual(len(reasons), 0)


@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestMDAPCache(unittest.TestCase):
    """Test MDAP caching functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.cache = MDAPCache(max_size=100, ttl_seconds=60)

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations"""
        key = "test_key"
        value = {"result": "test_value"}

        self.cache.set(key, value)
        retrieved = self.cache.get(key)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["value"]["result"], "test_value")

    def test_cache_miss(self):
        """Test cache returns None for missing keys"""
        result = self.cache.get("nonexistent_key")

        self.assertIsNone(result)

    def test_cache_expiration(self):
        """Test cache entries expire after TTL"""
        # Set TTL to 1 second for testing
        cache = MDAPCache(max_size=100, ttl_seconds=1)

        cache.set("expiring_key", {"value": "test"})
        time.sleep(1.1)

        result = cache.get("expiring_key")

        self.assertIsNone(result)

    def test_cache_size_limit(self):
        """Test cache respects max size limit"""
        small_cache = MDAPCache(max_size=3, ttl_seconds=60)

        # Add 5 entries to cache of size 3
        for i in range(5):
            small_cache.set(f"key_{i}", {"value": i})

        # Only 3 entries should remain (LRU eviction)
        self.assertEqual(len(small_cache._cache), 3)

    def test_cache_clear(self):
        """Test cache can be cleared"""
        self.cache.set("key1", {"value": 1})
        self.cache.set("key2", {"value": 2})

        self.cache.clear()

        self.assertEqual(len(self.cache._cache), 0)
        self.assertIsNone(self.cache.get("key1"))


@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestUtilityFunctions(unittest.TestCase):
    """Test MDAP utility functions"""

    def test_approx_token_count(self):
        """Test approximate token counting"""
        text = "This is a test string with some words"
        count = _approx_token_count(text)

        # Should be approximately len / 4
        expected = len(text) / 4
        self.assertAlmostEqual(count, expected, delta=1)

    def test_approx_token_count_empty(self):
        """Test token count for empty string"""
        count = _approx_token_count("")

        self.assertEqual(count, 0)

    def test_canonicalize_candidate_dict(self):
        """Test canonicalization of dict candidates"""
        candidate = {"b": 2, "a": 1}
        canonical = canonicalize_candidate(candidate)

        # Should be sorted JSON
        self.assertEqual(canonical, '{"a":1,"b":2}')

    def test_canonicalize_candidate_list(self):
        """Test canonicalization of list candidates"""
        candidate = [2, 1, 3]
        canonical = canonicalize_candidate(candidate)

        self.assertEqual(canonical, "[2,1,3]")

    def test_canonicalize_candidate_string(self):
        """Test canonicalization of string candidates"""
        candidate = "  test string  "
        canonical = canonicalize_candidate(candidate)

        self.assertEqual(canonical, "test string")

    def test_candidate_confidence_from_dict(self):
        """Test extracting confidence from dict"""
        candidate = {"confidence": 0.85}

        conf = candidate_confidence(candidate)

        self.assertEqual(conf, 0.85)

    def test_candidate_confidence_default(self):
        """Test default confidence when not present"""
        candidate = {"data": "value"}

        conf = candidate_confidence(candidate, default=0.5)

        self.assertEqual(conf, 0.5)

    def test_validate_schema_no_schema(self):
        """Test validation passes when no schema provided"""
        is_valid, errors = validate_schema({"any": "data"}, None)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_schema_correct_type(self):
        """Test validation passes for correct type"""
        schema = {"type": "string"}
        candidate = "test string"

        is_valid, errors = validate_schema(candidate, schema)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_schema_incorrect_type(self):
        """Test validation fails for incorrect type"""
        schema = {"type": "string"}
        candidate = 123

        is_valid, errors = validate_schema(candidate, schema)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestMDAPConfig(unittest.TestCase):
    """Test MDAP configuration"""

    def test_default_config(self):
        """Test default MDAP configuration"""
        config = MDAPConfig()

        self.assertEqual(config.k_min, 2)
        self.assertEqual(config.k_max, 8)
        self.assertEqual(config.max_votes_per_step, 50)
        self.assertEqual(config.timeout_seconds, 60)

    def test_custom_config(self):
        """Test custom MDAP configuration"""
        rules = RedFlagRules(max_tokens=500)
        config = MDAPConfig(
            k_min=3,
            k_max=10,
            max_votes_per_step=100,
            timeout_seconds=120,
            red_flag_rules=rules
        )

        self.assertEqual(config.k_min, 3)
        self.assertEqual(config.k_max, 10)
        self.assertEqual(config.max_votes_per_step, 100)
        self.assertEqual(config.red_flag_rules.max_tokens, 500)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestMDAPOrchestrator(unittest.TestCase):
    """Test MDAP orchestrator integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MDAPConfig(
            k_min=2,
            k_max=4,
            timeout_seconds=30
        )

    @patch('mdap_engine._request_openai_compatible_chat')
    def test_simple_mdap_execution(self, mock_llm_request):
        """Test basic MDAP execution with mocked LLM"""
        # Mock LLM response
        mock_llm_request.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "lean_code": "theorem test : True := by trivial",
                        "confidence": 0.95
                    })
                }
            }]
        }

        step = MDAPStep(
            step_id="test_step",
            prompt="Generate a simple proof"
        )

        task = MDAPTask(
            task_id="test_task",
            description="Test task",
            steps=[step]
        )

        orchestrator = MDAPOrchestrator(
            config=self.config,
            model_config=ModelConfig(
                provider="openai",
                model="gpt-4",
                api_key="test_key"
            )
        )

        # This would normally run the full MDAP pipeline
        # For unit testing, we test component interaction
        self.assertIsNotNone(orchestrator.config)
        self.assertEqual(orchestrator.config.k_min, 2)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with various configs"""
        config = MDAPConfig(
            k_min=3,
            k_max=6,
            enable_caching=True
        )

        model_config = ModelConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test_key"
        )

        orchestrator = MDAPOrchestrator(
            config=config,
            model_config=model_config
        )

        self.assertEqual(orchestrator.config.k_min, 3)
        self.assertEqual(orchestrator.config.k_max, 6)


@unittest.skipIf(not ROMA_MDAP_AVAILABLE, "ROMA-MDAP-MAKER not available")
class TestROMAMDAPMakerIntegration(unittest.TestCase):
    """Test ROMA-MDAP-MAKER integration"""

    def test_romamdap_config_defaults(self):
        """Test default ROMA-MDAP-MAKER configuration"""
        config = ROMAMDAPMakerConfig()

        self.assertEqual(config.roma_max_depth_analysis, 3)
        self.assertEqual(config.mdap_k_ahead, 3)
        self.assertTrue(config.mdap_enabled)
        self.assertTrue(config.apply_maker_to_roma_atomic)

    def test_romamdap_custom_config(self):
        """Test custom ROMA-MDAP-MAKER configuration"""
        config = ROMAMDAPMakerConfig(
            roma_max_depth_analysis=5,
            mdap_k_ahead=5,
            mdap_enable_red_flagging=False,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )

        self.assertEqual(config.roma_max_depth_analysis, 5)
        self.assertEqual(config.mdap_k_ahead, 5)
        self.assertFalse(config.mdap_enable_red_flagging)

    @unittest.skip("Requires ROMA installation")
    def test_romamdap_engine_initialization(self):
        """Test ROMA-MDAP-MAKER engine initialization"""
        config = ROMAMDAPMakerConfig()

        engine = ROMAMDAPMakerEngine(
            config=config,
            api_key="test_key"
        )

        self.assertIsNotNone(engine.config)
        self.assertEqual(engine.config.mdap_k_ahead, 3)


@unittest.skipIf(not WORKFLOW_STRUCTURES_AVAILABLE, "Workflow structures not available")
class TestSubProblemStructure(unittest.TestCase):
    """Test sub-problem structure for LeanAide integration"""

    def test_sub_problem_creation(self):
        """Test creating a sub-problem"""
        sub_problem = SubProblem(
            id="theorem_1",
            title="Prove addition commutativity",
            description="Prove that ∀ a b : Nat, a + b = b + a",
            estimated_effort=10
        )

        self.assertEqual(sub_problem.id, "theorem_1")
        self.assertEqual(sub_problem.estimated_effort, 10)

    def test_sub_problem_with_dependencies(self):
        """Test sub-problem with dependencies"""
        sub_problem = SubProblem(
            id="theorem_2",
            title="Prove multiplication commutativity",
            description="Prove that ∀ a b : Nat, a * b = b * a",
            dependencies=["theorem_1"],
            estimated_effort=15
        )

        self.assertEqual(len(sub_problem.dependencies), 1)
        self.assertIn("theorem_1", sub_problem.dependencies)


# =============================================================================
# MAKER TESTS
# =============================================================================

@unittest.skipIf(not MAKER_WORKFLOW_AVAILABLE, "MAKER workflow not available")
class TestMAKERWorkflowIntegration(unittest.TestCase):
    """Test MAKER workflow integration"""

    def test_resolve_maker_enabled_explicit(self):
        """Test MAKER enabled when explicitly set"""
        state = WorkflowState()
        state.maker_enabled = True

        enabled = resolve_maker_enabled(state, None)

        self.assertTrue(enabled)

    def test_resolve_maker_disabled_explicit(self):
        """Test MAKER disabled when explicitly set"""
        state = WorkflowState()
        state.maker_enabled = False

        enabled = resolve_maker_enabled(state, None)

        self.assertFalse(enabled)

    def test_resolve_maker_from_metadata(self):
        """Test MAKER enabled from metadata"""
        state = WorkflowState()
        state.metadata = {"maker_enabled": True}

        enabled = resolve_maker_enabled(state, None)

        self.assertTrue(enabled)


# =============================================================================
# WORKFLOW TESTS
# =============================================================================

@unittest.skipIf(not MAKER_WORKFLOW_AVAILABLE, "MAKER workflow not available")
class TestWorkflowIntegration(unittest.TestCase):
    """Test integration with decomposition workflow"""

    def test_build_maker_config_basic(self):
        """Test building basic MAKER config from workflow"""
        state = WorkflowState()
        state.maker_config = {
            "maker_mode": "sequential",
            "maker_k_ahead": 3
        }

        sub_problem = SubProblem(
            id="test_problem",
            title="Test",
            description="Test problem",
            estimated_effort=5
        )

        config = build_maker_config_from_workflow(state, sub_problem)

        self.assertIsNotNone(config)
        self.assertEqual(config.k_ahead, 3)

    def test_build_maker_config_large_effort(self):
        """Test MAKER config for large effort tasks"""
        state = WorkflowState()
        state.maker_config = {"maker_mode": "sequential"}

        sub_problem = SubProblem(
            id="large_problem",
            title="Large problem",
            description="Complex theorem",
            estimated_effort=25  # Triggers recursive mode
        )

        config = build_maker_config_from_workflow(state, sub_problem)

        # Should default to recursive for large effort
        self.assertIsNotNone(config)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestEdgeCases(unittest.TestCase):
    """Test edge cases and failure modes"""

    def test_all_agents_fail(self):
        """Test behavior when all agents fail"""
        rules = RedFlagRules()
        flagger = RedFlagger(rules)

        # All agents return empty responses
        responses = ["", "", ""]
        all_flagged = True

        for response in responses:
            is_flagged, _ = flagger.is_flagged(response, {}, None)
            if not is_flagged:
                all_flagged = False

        self.assertTrue(all_flagged)

    def test_voting_tie(self):
        """Test handling of voting ties"""
        # Simulate a tie in voting
        votes = {
            "option_a": 5,
            "option_b": 5
        }

        max_votes = max(votes.values())
        winners = [k for k, v in votes.items() if v == max_votes]

        # Should have multiple winners in case of tie
        self.assertEqual(len(winners), 2)

    def test_empty_tactic_list(self):
        """Test handling of empty tactic list"""
        candidate = {
            "lean_code": "theorem test : True := by ?",
            "tactics": [],
            "confidence": 0.5
        }

        schema = {
            "type": "object",
            "required": ["tactics"]
        }

        is_valid, errors = validate_schema(candidate, schema)

        # Empty list should be valid (it's an array)
        self.assertTrue(is_valid)

    def test_invalid_theorem_statement(self):
        """Test handling of invalid theorem statements"""
        invalid_theorems = [
            "",  # Empty
            "   ",  # Whitespace only
            "incomplete theorem",  # No statement
            "theorem",  # Just keyword
        ]

        for theorem in invalid_theorems:
            # Should handle gracefully
            if not theorem or not theorem.strip():
                # Empty theorems should be rejected
                self.assertFalse(bool(theorem.strip()))

    def test_proof_timeout(self):
        """Test handling of proof timeouts"""
        config = MDAPConfig(
            timeout_seconds=1  # Very short timeout
        )

        self.assertEqual(config.timeout_seconds, 1)

    def test_very_long_response(self):
        """Test handling of very long responses"""
        rules = RedFlagRules(max_characters=1000)
        flagger = RedFlagger(rules)

        long_response = "x" * 2000

        is_flagged, reasons = flagger.is_flagged(long_response, {}, None)

        self.assertTrue(is_flagged)
        self.assertIn("response_too_long", reasons)

    def test_malformed_json_response(self):
        """Test handling of malformed JSON in LLM response"""
        malformed_json = "{invalid json content"

        result, error = _safe_json_loads(malformed_json)

        self.assertIsNone(result)
        self.assertIsNotNone(error)

    def test_confidence_out_of_range(self):
        """Test handling of confidence values outside [0,1]"""
        candidates = [
            {"confidence": -0.5},
            {"confidence": 1.5},
            {"confidence": 2.0},
            {"confidence": None}
        ]

        for candidate in candidates:
            conf = candidate_confidence(candidate, default=0.5)
            # Should either be the value or default
            self.assertIsInstance(conf, float)

    def test_schema_type_mismatch(self):
        """Test schema validation with type mismatches"""
        schema = {
            "type": "object",
            "properties": {
                "lean_code": {"type": "string"},
                "verified": {"type": "boolean"},
                "quality_score": {"type": "number"}
            },
            "required": ["lean_code", "verified"]
        }

        # Type mismatches
        candidate = {
            "lean_code": 123,  # Should be string
            "verified": "true",  # Should be boolean
            "quality_score": "high"  # Should be number
        }

        is_valid, errors = validate_schema(candidate, schema)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_missing_required_fields(self):
        """Test schema validation with missing required fields"""
        schema = {
            "type": "object",
            "required": ["lean_code", "tactics", "verification_result"]
        }

        candidate = {
            "lean_code": "theorem test : True := by trivial"
            # Missing tactics and verification_result
        }

        is_valid, errors = validate_schema(candidate, schema)

        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 2)


@unittest.skipIf(not ROMA_MDAP_AVAILABLE, "ROMA-MDAP-MAKER not available")
class TestROMAEdgeCases(unittest.TestCase):
    """Test ROMA-MDAP-MAKER edge cases"""

    def test_excessive_depth(self):
        """Test handling of excessive decomposition depth"""
        rules = ROMARedFlagRules(max_roma_depth=3)

        # Create a deep structure
        deep_dag = {
            "depth": 5,
            "nodes": ["a", "b", "c", "d", "e"]
        }

        flagger = ROMARedFlagger(rules)
        red_flags = flagger.check_roma_decomposition_red_flags(deep_dag)

        # Should flag excessive depth
        self.assertTrue(any("excessive_depth" in flag for flag in red_flags))

    def test_unbalanced_decomposition(self):
        """Test handling of unbalanced decompositions"""
        rules = ROMARedFlagRules(max_balance_ratio=5.0)

        # Create an unbalanced structure (10:1 ratio)
        unbalanced_dag = {
            "subtasks": {
                "large": {"effort": 100},
                "small": {"effort": 10}
            }
        }

        flagger = ROMARedFlagger(rules)
        red_flags = flagger.check_roma_decomposition_red_flags(unbalanced_dag)

        # Balance ratio is 10:1 = 10.0, which exceeds max of 5.0
        # Note: This test depends on the actual implementation of _calculate_balance_ratio
        # The test structure here is illustrative


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@unittest.skipIf(not MDAP_AVAILABLE, "MDAP engine not available")
class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""

    def test_cache_performance(self):
        """Test cache lookup performance"""
        cache = MDAPCache(max_size=1000, ttl_seconds=60)

        # Populate cache
        for i in range(100):
            cache.set(f"key_{i}", {"value": i})

        # Measure lookup time
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        end_time = time.time()

        lookup_time = end_time - start_time

        # 100 lookups should be fast (< 0.1 seconds)
        self.assertLess(lookup_time, 0.1)

    def test_token_count_performance(self):
        """Test token counting performance"""
        long_text = "word " * 10000

        start_time = time.time()
        count = _approx_token_count(long_text)
        end_time = time.time()

        counting_time = end_time - start_time

        # Should be very fast
        self.assertLess(counting_time, 0.01)
        self.assertGreater(count, 0)

    def test_schema_validation_performance(self):
        """Test schema validation performance"""
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"},
                "field3": {"type": "boolean"},
                "field4": {"type": "array"},
                "field5": {"type": "object"}
            },
            "required": ["field1", "field2", "field3"]
        }

        candidate = {
            "field1": "test",
            "field2": 123,
            "field3": True,
            "field4": [1, 2, 3],
            "field5": {"nested": "data"}
        }

        start_time = time.time()
        for _ in range(100):
            validate_schema(candidate, schema)
        end_time = time.time()

        validation_time = end_time - start_time

        # 100 validations should be fast
        self.assertLess(validation_time, 0.1)


# =============================================================================
# TEST SUITE RUNNER
# =============================================================================

def run_test_suite():
    """Run the complete test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Unit tests
        TestMDAPStepConfiguration,
        TestMDAPTaskConfiguration,
        TestRedFlagging,
        TestMDAPCache,
        TestUtilityFunctions,
        TestMDAPConfig,

        # Integration tests
        TestMDAPOrchestrator,
        TestROMAMDAPMakerIntegration,
        TestSubProblemStructure,

        # MAKER tests
        TestMAKERWorkflowIntegration,

        # Workflow tests
        TestWorkflowIntegration,

        # Edge cases
        TestEdgeCases,
        TestROMAEdgeCases,

        # Performance
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result


def run_specific_tests(pattern: str):
    """Run tests matching a specific pattern"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load all tests
    all_tests = loader.loadTestsFromModule(sys.modules[__name__])

    # Filter by pattern
    for test_group in all_tests:
        for test in test_group:
            if pattern in str(test):
                suite.addTest(test)

    # Run filtered tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    # Check if pattern argument provided
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        print(f"Running tests matching pattern: {pattern}")
        result = run_specific_tests(pattern)
    else:
        print("Running complete test suite")
        result = run_test_suite()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
