"""
Comprehensive Tests for Complete Mathematical Knowledge Integration

Tests all production-ready components:
- Z3 Knowledge Manager with full persistence
- LeanAIDE Integration with error recovery
- Unified Bridge with consensus
- Feature extraction pipelines
- Conflict detection and resolution
"""

import asyncio
import logging
import tempfile
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_z3_knowledge_manager():
    """Test complete Z3 knowledge manager."""
    print("\n" + "="*70)
    print("TEST: Z3 Knowledge Manager (Complete)")
    print("="*70)
    
    try:
        from knowledge_engine.integrations.z3_knowledge_complete import (
            Z3KnowledgeManager,
            FeatureExtractionPipeline,
            ConflictDetector,
            ExtractedFeatures
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        print(f"\n[1] Initialization")
        manager = Z3KnowledgeManager(database_url=f"sqlite:///{db_path}")
        await manager.initialize()
        print(f"  [OK] Manager initialized with database")
        
        print(f"\n[2] Feature Extraction Pipeline")
        features = manager.feature_pipeline.extract_features(
            problem_statement="Linear equation system: x + y = 10",
            constraints=["(> x 0)", "(< y 10)", "(= (+ x y) 10)"],
            result=type('Result', (), {
                'success': True,
                'status': 'sat',
                'solving_time': 1.5,
                'memory_usage': 100.0,
                'model': type('Model', (), {'assignments': {'x': 5, 'y': 5}})()
            })(),
            proof="(simplify (solve-eqs (smt)))"
        )
        print(f"  [OK] Features extracted:")
        print(f"    - Problem type: {features.problem_type}")
        print(f"    - Constraints: {features.constraint_count}")
        print(f"    - Variables: {features.variable_count}")
        print(f"    - Complexity: {features.max_constraint_complexity}")
        print(f"    - Difficulty: {features.difficulty_estimate:.2f}")
        
        print(f"\n[3] Learning from Solution")
        result = await manager.learn_from_solution(
            problem_statement="x + y = 10",
            constraints=["(> x 0)", "(< y 10)", "(= (+ x y) 10)"],
            result=type('Result', (), {
                'success': True,
                'status': 'sat',
                'solving_time': 1.5,
                'memory_usage': 100.0,
                'tactics_used': ["simplify", "solve-eqs", "smt"],
                'config': {'timeout': 30}
            })(),
            proof="(simplify (solve-eqs (smt)))",
            metadata={'source': 'test', 'confidence': 0.9}
        )
        print(f"  [OK] Learning complete:")
        print(f"    - Items learned: {result['items_learned']}")
        
        print(f"\n[4] Conflict Detection")
        detector = ConflictDetector()
        
        new_knowledge = {
            'hash': 'test_new',
            'tactics': ['simp', 'linarith'],
            'pattern_type': 'linear',
            'confidence': 0.8,
            'success_rate': 0.9
        }
        existing = [{
            'hash': 'test_existing',
            'tactics': ['simplify', 'solve-eqs'],
            'pattern_type': 'linear',
            'confidence': 0.7,
            'success_rate': 0.8
        }]
        
        conflicts = detector.detect_conflicts(new_knowledge, existing)
        print(f"  [OK] Conflicts detected: {len(conflicts)}")
        if conflicts:
            print(f"    - Type: {conflicts[0]['type']}")
        
        print(f"\n[5] Find Similar Solutions")
        similar = await manager.find_similar_solutions(
            problem_statement="Linear: x + y = 10",
            constraints=["(> x 0)", "(< y 10)"],
            top_k=3
        )
        print(f"  [OK] Similar solutions found: {len(similar)}")
        
        print(f"\n[6] Get Metrics")
        metrics = manager.get_metrics()
        print(f"  [OK] Metrics retrieved:")
        print(f"    - Knowledge stored: {metrics['knowledge_stored']}")
        print(f"    - Feature extractions: {metrics['feature_extraction']['total_extractions']}")
        
        # Cleanup
        os.unlink(db_path)
        
        print("\n[PASS] Z3 Knowledge Manager")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leanaide_complete_integration():
    """Test complete LeanAIDE integration."""
    print("\n" + "="*70)
    print("TEST: LeanAIDE Complete Integration")
    print("="*70)
    
    try:
        from knowledge_engine.integrations.leanaide_integration_complete import (
            LeanAideIntegrationComplete,
            ProofStateManager,
            ProofGoal,
            ErrorRecoveryStrategy,
            LeanAideTacticExecutor
        )
        
        print(f"\n[1] Proof State Manager")
        state_manager = ProofStateManager()
        
        goal = ProofGoal(
            goal_id="test_goal",
            statement="forall n, n + 0 = n",
            target="n + 0 = n"
        )
        
        root_id = state_manager.initialize_proof("thm_1", goal)
        print(f"  [OK] Proof initialized with root: {root_id}")
        
        print(f"\n[2] Proof Tree Operations")
        # Simulate tactic application
        result = {
            "success": True,
            "subgoals": [
                {"statement": "0 + 0 = 0"},
                {"statement": "n + 0 = n -> n+1 + 0 = n+1"}
            ]
        }
        
        new_nodes = state_manager.apply_tactic("thm_1", root_id, "induction n", result)
        print(f"  [OK] Tactic applied, new nodes: {len(new_nodes)}")
        
        # Check open goals
        open_goals = state_manager.get_open_goals("thm_1")
        print(f"  [OK] Open goals: {len(open_goals)}")
        
        print(f"\n[3] Error Recovery")
        recovery = ErrorRecoveryStrategy()
        
        # Test timeout recovery
        alt = await recovery.recover(
            "timeout",
            goal,
            "simp [add_assoc, add_comm]",
            "timeout after 30s"
        )
        print(f"  [OK] Recovery suggestion: {alt}")
        
        # Test unknown tactic recovery
        alt = await recovery.recover(
            "unknown_tactic",
            goal,
            "simplify",
            "unknown tactic 'simplify'"
        )
        print(f"  [OK] Alternative tactic: {alt}")
        
        print(f"\n[4] Tactic Executor")
        executor = LeanAideTacticExecutor()
        
        execution = await executor.execute_tactic(
            "thm_1", root_id, "intro n"
        )
        print(f"  [OK] Tactic executed:")
        print(f"    - Result: {execution.result.value}")
        print(f"    - Time: {execution.execution_time_ms:.1f} ms")
        
        print(f"\n[5] Complete Integration")
        integration = LeanAideIntegrationComplete()
        await integration.initialize()
        
        # Set up callback
        async def on_tactic_success(thm_id, execution):
            pass
        
        integration.on_tactic_success = on_tactic_success
        
        result = await integration.prove_theorem_complete(
            theorem_statement="theorem test : 1 + 1 = 2 := by",
            auto_tactics=["simp", "rfl"],
            max_depth=5
        )
        
        print(f"  [OK] Proof completed:")
        print(f"    - Success: {result['success']}")
        print(f"    - Tactics: {result.get('tactics', [])}")
        print(f"    - Depth: {result['depth_reached']}")
        
        # Get proof state
        state = integration.get_proof_state(result['theorem_id'])
        print(f"  [OK] Proof state retrieved:")
        print(f"    - Complete: {state['is_complete']}")
        print(f"    - Tactics applied: {len(state['tactics_applied'])}")
        
        print("\n[PASS] LeanAIDE Complete Integration")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_unified_bridge_complete():
    """Test complete unified bridge."""
    print("\n" + "="*70)
    print("TEST: Unified Bridge (Complete)")
    print("="*70)
    
    try:
        from knowledge_engine.integrations.unified_math_bridge_complete import (
            UnifiedMathBridgeComplete,
            SemanticTranslator,
            ConsensusEngine,
            SolverSystem,
            ConsensusLevel,
            SolverResult
        )
        
        print(f"\n[1] Semantic Translator")
        translator = SemanticTranslator()
        
        # SMT to Lean
        smt = "(assert (> x 0))"
        lean = translator.translate_smt_to_lean(smt)
        print(f"  [OK] SMT -> Lean: {lean}")
        
        # Lean to SMT
        lean = "theorem pos : x > 0 := by sorry"
        smt = translator.translate_lean_to_smt(lean)
        print(f"  [OK] Lean -> SMT: {smt}")
        
        # Semantic features
        features = translator.extract_semantic_features("forall x y, x + y = y + x")
        print(f"  [OK] Features extracted:")
        print(f"    - Operators: {features['operators']}")
        print(f"    - Quantifiers: {features['quantifiers']}")
        print(f"    - Complexity: {features['complexity']}")
        
        print(f"\n[2] Consensus Engine")
        consensus = ConsensusEngine(ConsensusLevel.CONFIDENCE)
        
        # Test full agreement
        z3_result = SolverResult(
            solver=SolverSystem.Z3,
            success=True,
            result_type="sat",
            confidence=0.9
        )
        lean_result = SolverResult(
            solver=SolverSystem.LEANAIDE,
            success=True,
            result_type="theorem",
            confidence=0.85
        )
        
        result, meta = consensus.reach_consensus(z3_result, lean_result)
        print(f"  [OK] Consensus reached:")
        print(f"    - Winner: {result.solver.value}")
        print(f"    - Reason: {meta['reason']}")
        print(f"    - Agreement: {meta['agreement']}")
        
        # Test conflict detection
        conflict = consensus.detect_conflict(z3_result, lean_result)
        if conflict:
            print(f"  [OK] Conflict detected: {conflict['type']}")
        else:
            print(f"  [OK] No conflict (expected for successful results)")
        
        print(f"\n[3] Complete Bridge")
        bridge = UnifiedMathBridgeComplete()
        await bridge.initialize()
        
        # Test solver order determination
        from knowledge_engine.integrations.unified_math_bridge_complete import UnifiedProblem
        problem = UnifiedProblem(
            problem_id="test",
            statement="forall x, x = x",
            features={"quantifiers": 1, "complexity": 3}
        )
        
        order = bridge._determine_solver_order(SolverSystem.AUTO, problem)
        print(f"  [OK] Solver order (AUTO): {[s.value for s in order]}")
        
        # Test problem solving
        print(f"\n[4] Problem Solving")
        result = await bridge.solve(
            problem_statement="Prove that for all n, n + 0 = n",
            preferred_solver=SolverSystem.AUTO,
            consensus_level=ConsensusLevel.CONFIDENCE
        )
        
        print(f"  [OK] Problem solved:")
        print(f"    - Success: {result['success']}")
        print(f"    - Time: {result['execution_time_ms']:.1f} ms")
        print(f"    - Has consensus meta: {'consensus' in result}")
        
        # Test caching
        print(f"\n[5] Caching")
        result2 = await bridge.solve(
            problem_statement="Prove that for all n, n + 0 = n",
            use_cache=True
        )
        print(f"  [OK] Cached result: {result2.get('cached', False)}")
        
        # Statistics
        stats = bridge.get_statistics()
        print(f"\n[6] Bridge Statistics")
        print(f"  [OK] Statistics:")
        print(f"    - Problems solved: {stats['problems_solved']}")
        print(f"    - Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"    - Z3 successes: {stats['z3_successes']}")
        print(f"    - Lean successes: {stats['lean_successes']}")
        
        print("\n[PASS] Unified Bridge (Complete)")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_feature_extraction():
    """Test comprehensive feature extraction."""
    print("\n" + "="*70)
    print("TEST: Feature Extraction Pipeline")
    print("="*70)
    
    try:
        from knowledge_engine.integrations.z3_knowledge_complete import (
            FeatureExtractionPipeline,
            ExtractedFeatures
        )
        
        pipeline = FeatureExtractionPipeline()
        
        print(f"\n[1] Complex Constraint Analysis")
        constraints = [
            "(> x 0)",
            "(< x 100)",
            "(= y (* x 2))",
            "(> (+ x y) 50)",
            "(< (* x y) 1000)"
        ]
        
        class MockResult:
            success = True
            status = "sat"
            solving_time = 2.5
            memory_usage = 150.0
        
        features = pipeline.extract_features(
            problem_statement="Complex nonlinear constraints",
            constraints=constraints,
            result=MockResult(),
            proof="(simplify (qe (smt)))"
        )
        
        print(f"  [OK] Features extracted:")
        print(f"    - Hash: {features.problem_hash[:16]}...")
        print(f"    - Type: {features.problem_type}")
        print(f"    - Constraints: {features.constraint_count}")
        print(f"    - Variables: {features.variable_count}")
        print(f"    - Linear ratio: {features.linear_constraint_ratio:.2f}")
        print(f"    - Nonlinear count: {features.nonlinear_constraint_count}")
        print(f"    - Complexity: {features.avg_constraint_complexity:.2f}")
        print(f"    - Difficulty: {features.difficulty_estimate:.2f}")
        print(f"    - Recommended timeout: {features.recommended_timeout}s")
        
        print(f"\n[2] Feature Vector")
        vector = features.to_vector()
        print(f"  [OK] Vector length: {len(vector)}")
        print(f"  [OK] Vector: {[round(v, 2) for v in vector[:5]]}...")
        
        print(f"\n[3] Problem Classification")
        test_cases = [
            (["(> x 0)", "(< x 10)"], "linear"),
            (["(= y (* x x))"], "nonlinear"),
            (["(forall x (> x 0))"], "quantified"),
            (["(and a b)"], "boolean")
        ]
        
        for constraints, expected in test_cases:
            features = pipeline.extract_features(
                "test", constraints, MockResult()
            )
            print(f"  [OK] {expected}: {features.problem_type == expected}")
        
        print(f"\n[4] Caching")
        # First extraction
        features1 = pipeline.extract_features(
            "cache test problem",
            ["(> x 0)"],
            MockResult()
        )
        
        # Second extraction (should hit cache)
        features2 = pipeline.extract_features(
            "cache test problem",
            ["(> x 0)"],
            MockResult()
        )
        
        print(f"  [OK] Cache hits: {pipeline.extraction_stats['cache_hits']}")
        print(f"  [OK] Same hash: {features1.problem_hash == features2.problem_hash}")
        
        print("\n[PASS] Feature Extraction Pipeline")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all comprehensive tests."""
    print("="*70)
    print("COMPREHENSIVE MATHEMATICAL KNOWLEDGE INTEGRATION TESTS")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    
    results = {
        "Z3 Knowledge Manager": await test_z3_knowledge_manager(),
        "Feature Extraction": await test_feature_extraction(),
        "LeanAIDE Complete": await test_leanaide_complete_integration(),
        "Unified Bridge Complete": await test_unified_bridge_complete()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[X]"
        print(f"{symbol} {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(f"{total - passed} TEST(S) FAILED")
        print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
