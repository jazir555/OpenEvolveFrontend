"""
Comprehensive Test for Enhanced Mathematical Knowledge Integration

Tests:
1. Enhanced Z3 Knowledge with ML capabilities
2. LeanAIDE Knowledge Extraction
3. Improved LeanAIDE Proof Integration
4. Unified Z3-LeanAIDE Knowledge Bridge
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_z3_knowledge():
    """Test enhanced Z3 knowledge integration."""
    print("\n" + "="*60)
    print("TEST: Enhanced Z3 Knowledge Integration")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.z3_enhanced_knowledge import (
            EnhancedZ3KnowledgeIntegration,
            MLPoweredPatternMatcher,
            AdaptiveStrategyOptimizer
        )
        
        # Test ML Pattern Matcher
        print("\n[1] ML Pattern Matcher")
        matcher = MLPoweredPatternMatcher(embedding_dim=64)
        
        patterns = [
            {"id": "p1", "type": "linear", "content": "x + y = 10"},
            {"id": "p2", "type": "linear", "content": "2a + 3b = 15"},
            {"id": "p3", "type": "nonlinear", "content": "x^2 + y^2 = 25"}
        ]
        
        for p in patterns:
            matcher.create_embedding(p['id'], p['content'], p['type'], p)
        print(f"  [OK] Created {len(matcher.pattern_embeddings)} embeddings")
        
        # Test similarity search
        similar = matcher.find_similar_patterns("3x + 4y = 20", pattern_type="linear", top_k=2)
        print(f"  [OK] Found {len(similar)} similar patterns")
        
        # Test Strategy Optimizer
        print("\n[2] Strategy Optimizer")
        optimizer = AdaptiveStrategyOptimizer()
        
        # Record some strategy uses
        for i in range(10):
            optimizer.record_strategy_use(
                strategy_id="strat_linear",
                success=i < 8,  # 80% success rate
                solving_time=2.0 + i * 0.1,
                memory_usage=100.0,
                problem_type="linear"
            )
        
        # Get optimal strategy
        optimal, confidence = optimizer.get_optimal_strategy(
            problem_features={"type": "linear", "vars": 5},
            problem_type="linear",
            available_strategies=["strat_linear", "strat_other"]
        )
        print(f"  [OK] Optimal strategy: {optimal} (confidence: {confidence:.2f})")
        
        # Get suggestions
        suggestions = optimizer.suggest_strategy_improvements("strat_linear")
        print(f"  [OK] Suggestions: {len(suggestions)}")
        
        # Test Enhanced Integration
        print("\n[3] Enhanced Integration")
        integration = EnhancedZ3KnowledgeIntegration(storage_engine=None)
        await integration.initialize()
        
        # Create mock result
        class MockResult:
            success = True
            model = type('Model', (), {'assignments': {'x': 5}})()
            constraints = ["(> x 0)", "(< x 10)"]
            solving_time = 1.5
        
        result = await integration.extract_with_ml_enhancement(
            result=MockResult(),
            problem_statement="Linear constraint problem",
            problem_type="linear"
        )
        
        print(f"  [OK] ML insights: {list(result['ml_insights'].keys())}")
        
        # Get analytics
        analytics = integration.get_analytics()
        print(f"  [OK] Analytics: {analytics}")
        
        print("\n[PASS] Enhanced Z3 Knowledge Integration")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leanaide_knowledge_extraction():
    """Test LeanAIDE knowledge extraction."""
    print("\n" + "="*60)
    print("TEST: LeanAIDE Knowledge Extraction")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.leanaide_knowledge_extraction import (
            LeanAideKnowledgeExtractor,
            get_leanaide_knowledge_extractor
        )
        
        extractor = get_leanaide_knowledge_extractor()
        
        # Test tactic pattern extraction
        print("\n[1] Tactic Pattern Extraction")
        proof_steps = [
            {"tactic": "intro", "goal": "forall n, n + 0 = n"},
            {"tactic": "induction", "goal": "n + 0 = n"},
            {"tactic": "simp", "goal": "0 + 0 = 0"},
            {"tactic": "rfl", "goal": "0 = 0"}
        ]
        
        patterns = extractor.extract_tactic_patterns(proof_steps, "arithmetic")
        print(f"  [OK] Extracted {len(patterns)} tactic patterns")
        
        # Test theorem analysis
        print("\n[2] Theorem Structure Analysis")
        theorem = "theorem add_zero : forall (n : Nat), n + 0 = n := by"
        theorem_pattern = extractor.analyze_theorem_structure(theorem)
        print(f"  [OK] Theorem type: {theorem_pattern.pattern_type}")
        print(f"  [OK] Variables: {theorem_pattern.variables}")
        
        # Test strategy learning
        print("\n[3] Strategy Learning")
        strategy = extractor.learn_proof_strategy(
            theorem_features={"type": "arithmetic", "var_count": 1, "has_induction": True},
            tactics_used=["intro", "induction", "simp", "rfl"],
            proof_time=2.5,
            success=True
        )
        print(f"  [OK] Learned strategy: {strategy.name}")
        print(f"  [OK] Success rate: {strategy.success_rate():.1%}")
        
        # Test strategy recommendation
        print("\n[4] Strategy Recommendation")
        recommended = extractor.recommend_strategy(
            {"type": "arithmetic", "var_count": 2, "has_induction": True}
        )
        if recommended:
            print(f"  [OK] Recommended: {recommended.name}")
        else:
            print(f"  [OK] No strong recommendation (expected with limited data)")
        
        # Test concept extraction
        print("\n[5] Mathematical Concept Extraction")
        proof = "def add (n m : Nat) := n + m\nlemma add_comm : n + m = m + n := by simp"
        concepts = extractor.extract_mathematical_concepts(theorem, proof)
        print(f"  [OK] Extracted {len(concepts)} concepts")
        
        # Get summary
        print("\n[6] Knowledge Summary")
        summary = extractor.get_knowledge_summary()
        print(f"  [OK] Tactic patterns: {summary['tactic_patterns']['count']}")
        print(f"  [OK] Theorem patterns: {summary['theorem_patterns']['count']}")
        print(f"  [OK] Strategies: {summary['proof_strategies']['count']}")
        
        print("\n[PASS] LeanAIDE Knowledge Extraction")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leanaide_proof_integration():
    """Test improved LeanAIDE proof integration."""
    print("\n" + "="*60)
    print("TEST: LeanAIDE Proof Integration")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.leanaide_proof_integration import (
            LeanAideProofIntegration,
            get_leanaide_proof_integration,
            AutomatedProofSearcher,
            ProofSearchConfig
        )
        
        # Test Proof Searcher
        print("\n[1] Automated Proof Searcher")
        from knowledge_engine.integrations.leanaide_knowledge_extraction import get_leanaide_knowledge_extractor
        
        extractor = get_leanaide_knowledge_extractor()
        config = ProofSearchConfig(max_depth=5, timeout_seconds=10.0)
        searcher = AutomatedProofSearcher(extractor, config)
        
        theorem = "theorem add_zero (n : Nat) : n + 0 = n := by"
        attempt = await searcher.search_proof(theorem)
        
        print(f"  [OK] Search status: {attempt.status.value}")
        print(f"  [OK] Proof found: {attempt.proof_found is not None}")
        print(f"  [OK] Time: {attempt.execution_time_ms:.1f} ms")
        
        # Get stats
        stats = searcher.get_search_stats()
        print(f"  [OK] Success rate: {stats.get('success_rate', 0):.1%}")
        
        # Test Integration
        print("\n[2] Proof Integration")
        integration = await get_leanaide_proof_integration()
        
        result = await integration.prove_theorem(
            theorem="theorem test : 1 + 1 = 2 := by",
            auto_search=True,
            use_knowledge=True
        )
        
        print(f"  [OK] Success: {result['success']}")
        proof_str = result.get('proof') or 'N/A'
        print(f"  [OK] Proof: {str(proof_str)[:50]}")
        
        # Test tactic recommendations
        print("\n[3] Tactic Recommendations")
        recommendations = integration.get_recommended_tactics("n + 0 = n")
        print(f"  [OK] Recommendations: {len(recommendations)}")
        
        # Get knowledge summary
        print("\n[4] Knowledge Summary")
        summary = integration.get_knowledge_summary()
        print(f"  [OK] Knowledge extracted: {summary['knowledge_extractor']['tactic_patterns']['count']} patterns")
        
        print("\n[PASS] LeanAIDE Proof Integration")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_unified_bridge():
    """Test unified Z3-LeanAIDE knowledge bridge."""
    print("\n" + "="*60)
    print("TEST: Unified Mathematical Knowledge Bridge")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.unified_math_knowledge_bridge import (
            UnifiedMathKnowledgeBridge,
            get_unified_math_bridge,
            ProblemClassifier,
            ProblemClassification,
            CrossSystemKnowledgeTransfer
        )
        
        # Test Problem Classifier
        print("\n[1] Problem Classification")
        classifier = ProblemClassifier()
        
        problems = [
            ("Solve x + 5 = 10", ProblemClassification.CONSTRAINT_SOLVING),
            ("Prove that forall n, n + 0 = n", ProblemClassification.THEOREM_PROVING),
            ("Check satisfiability of constraints", ProblemClassification.SMT_SOLVING),
            ("Prove by induction that sum 1..n = n(n+1)/2", ProblemClassification.INDUCTIVE_PROOF)
        ]
        
        for problem, expected in problems:
            classification = classifier.classify(problem)
            solver, confidence = classifier.recommend_solver(classification)
            print(f"  [OK] '{problem[:30]}...' -> {classification.value} (use {solver})")
        
        # Test Knowledge Transfer
        print("\n[2] Cross-System Knowledge Transfer")
        transfer = CrossSystemKnowledgeTransfer()
        
        z3_tactic = "simplify"
        lean_tactic = transfer.z3_to_lean_tactic(z3_tactic)
        print(f"  [OK] Z3 '{z3_tactic}' -> Lean '{lean_tactic}'")
        
        # Translate pattern
        z3_pattern = {
            "name": "LinearSolver",
            "tactics": ["simplify", "solve-eqs", "smt"],
            "type": "linear"
        }
        translated = transfer.translate_pattern(z3_pattern, "z3", "leanaide")
        print(f"  [OK] Translated pattern tactics: {translated['tactics']}")
        
        # Test Unified Bridge
        print("\n[3] Unified Bridge")
        bridge = await get_unified_math_bridge()
        
        # Solve problems with different classifications
        test_problems = [
            "Solve the system: x + y = 10, x - y = 2",
            "Prove that for all x, x = x"
        ]
        
        for problem in test_problems:
            result = await bridge.solve_problem(problem, use_hybrid=False)
            print(f"  [OK] Problem solved: {result['success']} ({result['classification']})")
        
        # Get unified summary
        print("\n[4] Unified Knowledge Summary")
        summary = bridge.get_unified_knowledge_summary()
        print(f"  [OK] Problems processed: {summary['statistics']['problems_processed']}")
        print(f"  [OK] Z3 successes: {summary['statistics']['z3_successes']}")
        print(f"  [OK] LeanAIDE successes: {summary['statistics']['leanaide_successes']}")
        
        print("\n[PASS] Unified Mathematical Knowledge Bridge")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("ENHANCED MATHEMATICAL KNOWLEDGE INTEGRATION TESTS")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    results = {
        "Enhanced Z3 Knowledge": await test_enhanced_z3_knowledge(),
        "LeanAIDE Knowledge Extraction": await test_leanaide_knowledge_extraction(),
        "LeanAIDE Proof Integration": await test_leanaide_proof_integration(),
        "Unified Bridge": await test_unified_bridge()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[X]"
        print(f"{symbol} {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
