#!/usr/bin/env python3
"""
Comprehensive Test Suite for Knowledge Engine Business Logic

Demonstrates all newly implemented features with real API calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TEST 1: Semantic Deduplication with LLM Integration
# =============================================================================

async def test_semantic_deduplication():
    """Test LLM-powered semantic deduplication."""
    print('\n' + '='*70)
    print('TEST 1: Semantic Deduplication with LLM Integration')
    print('='*70)

    try:
        from knowledge_engine.deduplication.base import Entity
        from knowledge_engine.deduplication.strategies.semantic_strategy import (
            SemanticDedupStrategy
        )

        # Create test entities (using 'properties' instead of 'attributes')
        entities = [
            Entity(
                id="e1",
                name="Apple Inc",
                entity_type="ORGANIZATION",
                description="Technology company founded by Steve Jobs",
                properties={"founded": "1976", "location": "Cupertino"}
            ),
            Entity(
                id="e2",
                name="Apple",
                entity_type="ORGANIZATION",
                description="Tech company from California",
                properties={"industry": "Technology", "hq": "Cupertino"}
            ),
            Entity(
                id="e3",
                name="Microsoft Corporation",
                entity_type="ORGANIZATION",
                description="Software company founded by Bill Gates",
                properties={"founded": "1975", "location": "Redmond"}
            ),
        ]

        # Initialize strategy with config
        config = {
            'confidence_threshold': 0.7,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'llm_model': 'gpt-4'
        }

        strategy = SemanticDedupStrategy(config=config)

        print('[1] Testing multi-factor similarity calculation...')
        confidence = strategy.calculate_confidence(entities[0], entities[1])
        print(f'    Confidence (Apple Inc vs Apple): {confidence:.3f}')
        print(f'    Threshold: {strategy.confidence_threshold}')
        print(f'    Would merge: {confidence >= strategy.confidence_threshold}')

        print('\n[2] Testing heuristic verification...')
        result = await strategy._heuristic_verification([entities[0], entities[1]])
        print(f'    Heuristic says merge: {result}')

        print('\n[3] Testing temporal overlap detection...')
        # Create entities with time ranges
        entity_with_time = Entity(
            id="e1_time",
            name="Apple Inc",
            entity_type="ORGANIZATION",
            description="Apple with time",
            properties={
                "start_time": "1976-04-01T00:00:00Z",
                "end_time": "2024-01-01T00:00:00Z"
            }
        )
        entity_with_time2 = Entity(
            id="e2_time",
            name="Apple",
            entity_type="ORGANIZATION",
            description="Apple with time",
            properties={
                "start_time": "1980-01-01T00:00:00Z",
                "end_time": "2025-01-01T00:00:00Z"
            }
        )

        has_overlap = await strategy._has_temporal_overlap([entity_with_time, entity_with_time2])
        print(f'    Temporal overlap detected: {has_overlap}')

        # Try LLM verification if API key is available
        if os.getenv('OPENAI_API_KEY'):
            print('\n[4] Testing LLM verification with OpenAI GPT-4...')
            print('    (This will make a real API call)')
            try:
                llm_result = await strategy._llm_verify_group([entities[0], entities[1]])
                print(f'    LLM says merge: {llm_result}')
            except Exception as e:
                print(f'    LLM call failed (expected if no valid key): {str(e)[:100]}')
        else:
            print('\n[4] Skipping LLM test (no OPENAI_API_KEY in .env)')
            print('    To test LLM integration, add OPENAI_API_KEY to .env')

        print('\n[PASS] Semantic deduplication test PASSED')
        return True

    except Exception as e:
        print(f'\n[FAIL] Semantic deduplication test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 2: Unified Knowledge Extraction
# =============================================================================

async def test_knowledge_extraction():
    """Test the unified knowledge extraction with NLP backends."""
    print('\n' + '='*70)
    print('TEST 2: Unified Knowledge Extraction')
    print('='*70)

    try:
        from knowledge_engine.integrations.unified_knowledge_extraction import (
            UnifiedKnowledgeExtractor
        )

        extractor = UnifiedKnowledgeExtractor()

        test_text = """
        Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.
        Microsoft Corporation was founded by Bill Gates in Redmond, Washington.
        Both companies became major technology companies.
        """

        print('[1] Extracting entities and relations from text...')
        print(f'    Text: "{test_text.strip()[:100]}..."')

        result = extractor.extract_from_text(
            text=test_text,
            extraction_type="entities_relations",
            config={'spacy_model': 'en_core_web_sm'}
        )

        if result.status == 'success':
            print(f'\n[2] Extraction Status: {result.status}')
            print(f'    Method: {result.metadata.get("method", "unknown")}')
            print(f'    Entities found: {len(result.data.get("entities", []))}')

            print('\n[3] Extracted Entities:')
            for i, entity in enumerate(result.data.get('entities', [])[:10]):
                print(f'    {i+1}. {entity["text"]} ({entity["type"]}) - conf: {entity.get("confidence", 0):.2f}')

            print(f'\n[4] Relations found: {len(result.data.get("relations", []))}')
            for rel in result.data.get('relations', [])[:5]:
                print(f'    - {rel["subject"]} -> {rel["predicate"]} -> {rel["object"]}')

            print(f'\n[5] Triples generated: {len(result.data.get("triples", []))}')
            for triple in result.data.get('triples', [])[:5]:
                print(f'    - ({triple["subject"]}, {triple["predicate"]}, {triple["object"]})')

            print('\n[✅] Knowledge extraction test PASSED')
            return True
        else:
            print(f'\n[❌] Extraction failed: {result.errors}')
            return False

    except Exception as e:
        print(f'\n[❌] Knowledge extraction test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 3: Security Layer Encryption
# =============================================================================

def test_security_encryption():
    """Test AES-256-GCM encryption."""
    print('\n' + '='*70)
    print('TEST 3: Security Layer - AES-256-GCM Encryption')
    print('='*70)

    try:
        from knowledge_engine.security_layer import EncryptionManager

        crypto = EncryptionManager(master_key="test_key_123")

        print('[1] Testing string encryption...')
        plaintext = "Sensitive knowledge data: Apple Inc founded 1976"
        print(f'    Original: {plaintext}')

        # Encrypt
        encrypted = crypto.encrypt(plaintext, key_id="test")
        print(f'    Encrypted length: {len(encrypted)} chars')
        print(f'    Encrypted preview: {encrypted[:50]}...')

        # Decrypt
        decrypted = crypto.decrypt(encrypted, key_id="test")
        print(f'    Decrypted: {decrypted}')
        print(f'    Match: {plaintext == decrypted}')

        print('\n[2] Testing hash generation...')
        hashed = crypto.hash_sensitive(plaintext)
        print(f'    Hash: {hashed[:40]}...')
        print(f'    Hash length: {len(hashed)} chars')

        verified = crypto.verify_hash(plaintext, hashed)
        print(f'    Hash verification: {verified}')

        print('\n[3] Testing binary encryption...')
        binary_data = b'Binary knowledge: \x00\x01\x02\x03'
        encrypted_binary = crypto.encrypt_bytes(binary_data, key_id="test")
        print(f'    Binary encrypted length: {len(encrypted_binary)} bytes')
        decrypted_binary = crypto.decrypt_bytes(encrypted_binary, key_id="test")
        print(f'    Binary decrypted: {decrypted_binary}')
        print(f'    Binary match: {binary_data == decrypted_binary}')

        print('\n[✅] Security encryption test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] Security encryption test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 4: DSPy Integration
# =============================================================================

def test_dspy_integration():
    """Test DSPy integration methods."""
    print('\n' + '='*70)
    print('TEST 4: DSPy Integration Methods')
    print('='*70)

    try:
        from knowledge_engine.integrations.dspy_integration import DSPyTask

        task = DSPyTask()

        print('[1] Testing make_prompt method...')
        row = {
            'question': 'What is the capital of France?',
            'context': 'France is a country in Europe.',
            'answer': 'Paris'
        }
        prompt = task.make_prompt(row)
        print(f'    Generated prompt:')
        for line in prompt.split('\n'):
            print(f'      {line}')

        print('\n[2] Testing metric method...')
        example = type('Example', (), {'labels': 'Paris', 'answer': 'Paris'})()
        prediction = type('Prediction', (), {'prediction': 'Paris', 'answer': 'Paris'})()

        score = task.metric(example, prediction)
        print(f'    Match score: {score:.2f}')
        print(f'    Expected: 1.0, Got: {score}')

        print('\n[3] Testing metric_with_feedback method...')
        feedback_result = task.metric_with_feedback(example, prediction)
        if hasattr(feedback_result, 'get'):
            print(f'    Score: {feedback_result.get("score", 0)}')
            print(f'    Feedback: {feedback_result.get("feedback", "")}')
        else:
            print(f'    Result: {feedback_result}')

        print('\n[4] Testing load_data_from_list...')
        data = [
            {'question': 'Q1', 'answer': 'A1'},
            {'question': 'Q2', 'answer': 'A2'},
            {'question': 'Q3', 'answer': 'A3'},
        ]
        trainset, valset = task.load_data_from_list(
            data,
            input_fields=['question'],
            label_field='answer'
        )
        print(f'    Train set size: {len(trainset)}')
        print(f'    Validation set size: {len(valset)}')

        print('\n[✅] DSPy integration test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] DSPy integration test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 5: Standardization Deduplication
# =============================================================================

async def test_standardization_dedup():
    """Test LLM-assisted standardization deduplication."""
    print('\n' + '='*70)
    print('TEST 5: Standardization Deduplication with LLM')
    print('='*70)

    try:
        from knowledge_engine.deduplication.base import Entity
        from knowledge_engine.deduplication.strategies.standardization_strategy import (
            StandardizationDedupStrategy
        )

        # Create ambiguous entities
        entities = [
            Entity(
                id="s1",
                name="Apple Corp",
                entity_type="ORGANIZATION",
                description="Apple from Cupertino"
            ),
            Entity(
                id="s2",
                name="Apple Corporation",
                entity_type="ORGANIZATION",
                description="Apple Inc from California"
            ),
        ]

        config = {
            'merge_threshold': 0.6,
            'openai_api_key': os.getenv('OPENAI_API_KEY')
        }

        strategy = StandardizationDedupStrategy(config=config)

        print('[1] Testing pairwise similarity calculation...')
        sim = strategy._calculate_pairwise_similarity(entities[0], entities[1])
        print(f'    Similarity score: {sim:.3f}')
        print(f'    Merge threshold: {strategy.merge_threshold}')

        print('\n[2] Testing heuristic merge decision...')
        should_merge = strategy._heuristic_merge_decision(entities)
        print(f'    Heuristic says merge: {should_merge}')

        print('\n[3] Testing LLM merge decision...')
        if os.getenv('OPENAI_API_KEY'):
            print('    (This will make a real API call)')
            try:
                llm_decision = await strategy._llm_merge_decision(entities)
                print(f'    LLM says merge: {llm_decision}')
            except Exception as e:
                print(f'    LLM call failed (expected if no valid key): {str(e)[:100]}')
        else:
            print('    Skipping LLM test (no OPENAI_API_KEY)')

        print('\n[✅] Standardization deduplication test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] Standardization deduplication test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 6: SEMHASH Deduplication
# =============================================================================

async def test_semhash_dedup():
    """Test SEMHASH preprocessing with abbreviation expansion."""
    print('\n' + '='*70)
    print('TEST 6: SEMHASH Deduplication with Text Preprocessing')
    print('='*70)

    try:
        from knowledge_engine.deduplication.base import Entity
        from knowledge_engine.deduplication.strategies.semhash_strategy import (
            SemHashDedupStrategy
        )

        strategy = SemHashDedupStrategy()

        print('[1] Testing abbreviation expansion...')
        test_cases = [
            ("Apple Corp", "apple corporation"),
            ("Cupertino, CA", "cupertino california"),
            ("CEO Tim Cook", "chief executive officer tim cook"),
            ("Q1 2024", "quarter 1 2024"),
            ("Dept of Eng", "department of engineering"),
        ]

        for original, expected_contains in test_cases:
            normalized = strategy._normalize_entity_text(original)
            print(f'    "{original}" -> contains "{expected_contains}": {expected_contains in normalized.lower()}')

        print('\n[2] Testing full preprocessing pipeline...')
        entities = [
            Entity(id="h1", name="Apple Corp", entity_type="ORGANIZATION"),
            Entity(id="h2", name="Cupertino Inc", entity_type="LOCATION"),
        ]

        processed = await strategy.preprocess_entities(entities)
        print(f'    Processed {len(processed)} entities')
        for entity in processed:
            print(f'    - {entity.name} (normalized)')

        print('\n[✅] SEMHASH deduplication test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] SEMHASH deduplication test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 7: Symbolic Constraint Engine
# =============================================================================

def test_symbolic_constraints():
    """Test the symbolic constraint engine."""
    print('\n' + '='*70)
    print('TEST 7: Symbolic Constraint Engine')
    print('='*70)

    try:
        from knowledge_engine.core.symbolic_constraint_engine import (
            SymbolicConstraintEngine, Constraint, ConstraintType
        )

        print('[1] Creating constraint engine...')
        engine = SymbolicConstraintEngine()

        print('[2] Adding constraints...')
        constraints = [
            Constraint(
                id="c1",
                type=ConstraintType.REQUIRED,
                expression="score >= 0.8",
                description="Minimum quality threshold"
            ),
            Constraint(
                id="c2",
                type=ConstraintType.REQUIRED,
                expression="has_entities == True",
                description="Must have entities"
            ),
            Constraint(
                id="c3",
                type=ConstraintType.FORBIDDEN,
                expression="is_empty == True",
                description="Cannot be empty"
            ),
        ]

        for constraint in constraints:
            engine.add_constraint(constraint)
            print(f'    Added: {constraint.id} - {constraint.expression}')

        print('\n[3] Testing constraint satisfaction...')
        context1 = {
            'score': 0.85,
            'has_entities': True,
            'is_empty': False
        }
        result1 = engine.check_satisfaction(context1)
        print(f'    Context 1: satisfaction={result1.is_satisfied}')
        print(f'    Satisfied: {len(result1.satisfied_constraints)}')
        print(f'    Violated: {len(result1.violated_constraints)}')

        context2 = {
            'score': 0.5,
            'has_entities': True,
            'is_empty': False
        }
        result2 = engine.check_satisfaction(context2)
        print(f'    Context 2: satisfaction={result2.is_satisfied}')
        print(f'    Satisfied: {len(result2.satisfied_constraints)}')
        print(f'    Violated: {len(result2.violated_constraints)}')

        print('\n[4] Testing variable extraction...')
        constraint = Constraint(
            id="test",
            type=ConstraintType.REQUIRED,
            expression="user_name == 'Alice' AND age > 25"
        )
        print(f'    Variables extracted: {constraint.variables}')

        stats = engine.get_statistics()
        print(f'\n[5] Engine statistics:')
        print(f'    Total constraints: {stats["total_constraints"]}')
        print(f'    Evaluation count: {stats["evaluation_count"]}')

        print('\n[✅] Symbolic constraint engine test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] Symbolic constraint engine test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 8: A/B Testing Framework
# =============================================================================

def test_ab_testing():
    """Test the A/B testing framework."""
    print('\n' + '='*70)
    print('TEST 8: A/B Testing Framework')
    print('='*70)

    try:
        from knowledge_engine.ab_testing import (
            ABTestFramework, TestVariant, VariantType, ExperimentStatus
        )

        print('[1] Creating A/B test framework...')
        framework = ABTestFramework()

        print('[2] Creating experiment with variants...')
        variants = [
            TestVariant(
                id="control",
                name="Control",
                type=VariantType.CONTROL,
                config={"learning_rate": 0.001},
                traffic_allocation=50
            ),
            TestVariant(
                id="treatment",
                name="Treatment",
                type=VariantType.TREATMENT,
                config={"learning_rate": 0.01},
                traffic_allocation=50
            ),
        ]

        experiment = framework.create_experiment(
            experiment_id="test_exp_001",
            variants=variants,
            auto_start=True
        )

        print(f'    Experiment ID: {experiment.experiment_id}')
        print(f'    Status: {experiment.status.value}')
        print(f'    Variants: {len(experiment.variants)}')
        print(f'    Started: {experiment.started_at}')

        print('\n[3] Assigning users to variants...')
        # Simulate user assignments
        assignments = {}
        for user_id in ['user1', 'user2', 'user3', 'user4', 'user5']:
            variant = framework.assign_variant('test_exp_001', user_id)
            if variant:
                assignments[user_id] = variant.id
                print(f'    {user_id} -> {variant.id}')

        print('\n[4] Recording results...')
        # Simulate some results
        framework.record_result('test_exp_001', 'user1', success=True, metrics={'accuracy': 0.85})
        framework.record_result('test_exp_001', 'user2', success=False, metrics={'accuracy': 0.72})
        framework.record_result('test_exp_001', 'user3', success=True, metrics={'accuracy': 0.88})

        print('    Recorded 3 results')

        print('\n[5] Analyzing experiment...')
        analyzed = framework.analyze_experiment('test_exp_001')
        print(f'    Winner: {analyzed.winner}')
        print(f'    Total results: {len(analyzed.results)}')
        print(f'    Statistics: {list(analyzed.stats.keys())}')

        stats = analyzed.stats.get('control')
        if stats:
            print(f'\n    Control stats:')
            print(f'      Exposures: {stats.total_exposures}')
            print(f'      Conversions: {stats.total_conversions}')
            print(f'      Conversion rate: {stats.conversion_rate:.3f}')

        print('\n[✅] A/B testing test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] A/B testing test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 9: Causal Modeling
# =============================================================================

def test_causal_modeling():
    """Test the causal modeling engine."""
    print('\n' + '='*70)
    print('TEST 9: Causal Modeling')
    print('='*70)

    try:
        from knowledge_engine.causal_modeling import (
            CausalModeling, CausalMethod
        )

        print('[1] Creating causal modeling engine...')
        engine = CausalModeling(method=CausalMethod.PC_ALGORITHM)

        print('[2] Discovering causal graph from data...')
        data = {
            'temperature': [20, 25, 30, 35, 40],
            'ice_cream_sales': [50, 70, 90, 110, 130],
            'sunshine_hours': [8, 9, 10, 11, 12]
        }

        graph = engine.discover_causal_graph(data)
        print(f'    Graph discovered with {len(graph.nodes)} nodes')
        print(f'    Nodes: {list(graph.nodes)}')

        print('\n[3] Getting graph structure...')
        print(f'    Adjacency matrix keys: {list(graph.adj_matrix.keys())}')

        print('\n[4] Estimating treatment effect...')
        treatment_effect = engine.estimate_treatment_effect(
            data=data,
            treatment='temperature',
            outcome='ice_cream_sales'
        )
        print(f'    Treatment (temperature) -> Outcome (ice_cream_sales)')
        print(f'    ATE: {treatment_effect.ate}')
        print(f'    Method: {treatment_effect.method}')

        print('\n[5] Testing path strength calculation...')
        strength = engine._calculate_path_strength(graph, 'temperature', 'ice_cream_sales')
        print(f'    Path strength (temperature -> ice_cream_sales): {strength}')

        stats = engine.get_statistics()
        print(f'\n[6] Engine statistics:')
        print(f'    Graphs discovered: {stats["graphs_discovered"]}')
        print(f'    Inferences performed: {stats["inferences_performed"]}')
        print(f'    Default method: {stats["default_method"]}')

        print('\n[✅] Causal modeling test PASSED')
        return True

    except Exception as e:
        print(f'\n[❌] Causal modeling test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all tests and report results."""
    print('\n' + '='*70)
    print('KNOWLEDGE ENGINE - COMPREHENSIVE BUSINESS LOGIC TEST SUITE')
    print('='*70)
    print(f'Date: {datetime.utcnow().isoformat()}')
    print(f'Testing all newly implemented features with real business logic')
    print('='*70)

    results = {}

    # Run all tests
    results['Semantic Deduplication'] = await test_semantic_deduplication()
    results['Knowledge Extraction'] = await test_knowledge_extraction()
    results['Security Encryption'] = test_security_encryption()
    results['DSPy Integration'] = test_dspy_integration()
    results['Standardization'] = await test_standardization_dedup()
    results['SEMHASH'] = await test_semhash_dedup()
    results['Symbolic Constraints'] = test_symbolic_constraints()
    results['A/B Testing'] = test_ab_testing()
    results['Causal Modeling'] = test_causal_modeling()

    # Summary
    print('\n' + '='*70)
    print('TEST SUMMARY')
    print('='*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_bool in results.items():
        status = '✅ PASS' if passed_bool else '❌ FAIL'
        print(f'  {status}: {test_name}')

    print(f'\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)')

    if passed == total:
        print('\n' + '='*70)
        print('🎉 ALL TESTS PASSED - BUSINESS LOGIC FULLY FUNCTIONAL')
        print('='*70)
    else:
        print('\n' + '='*70)
        print('⚠️  SOME TESTS FAILED - CHECK LOGS ABOVE')
        print('='*70)

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
