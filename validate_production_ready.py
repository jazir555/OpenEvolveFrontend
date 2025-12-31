"""
Simple validation script for production-ready implementations
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    try:
        from problem_analyzer import ProblemAnalyzer
        from decomposition_engine import DecompositionEngine
        from sovereign_gauntlets import GauntletSystem
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_problem_analyzer():
    """Test Problem Analyzer with LLM enhancements."""
    logger.info("\nTesting Problem Analyzer...")
    try:
        from problem_analyzer import ProblemAnalyzer
        
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            "Build a web application with user authentication and data storage",
            "Web Application"
        )
        
        assert problem is not None
        assert problem.domain_context is not None
        assert problem.complexity_score is not None
        assert len(problem.constraints) >= 0
        
        logger.info(f"✓ Problem analyzed: {problem.title}")
        logger.info(f"  Domain: {problem.domain_context.domain}")
        logger.info(f"  Complexity: {problem.complexity_score.overall_complexity}/10")
        logger.info(f"  Constraints: {len(problem.constraints)}")
        return True
    except Exception as e:
        logger.error(f"✗ Problem Analyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decomposition_engine():
    """Test Decomposition Engine with intelligent strategies."""
    logger.info("\nTesting Decomposition Engine...")
    try:
        from problem_analyzer import ProblemAnalyzer
        from decomposition_engine import DecompositionEngine
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Create a REST API with authentication, data validation, and error handling",
            "REST API"
        )
        
        plan = engine.decompose(problem)
        
        assert plan is not None
        assert len(plan.sub_problems) >= 2
        assert plan.dependency_graph is not None
        
        logger.info(f"✓ Decomposition created: {len(plan.sub_problems)} sub-problems")
        logger.info(f"  Strategy: {plan.strategy.value}")
        for i, sp in enumerate(plan.sub_problems[:3], 1):
            logger.info(f"  {i}. {sp.title}")
        return True
    except Exception as e:
        logger.error(f"✗ Decomposition Engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gauntlets():
    """Test Gauntlets with LLM validation."""
    logger.info("\nTesting Gauntlets...")
    try:
        from problem_analyzer import ProblemAnalyzer
        from decomposition_engine import DecompositionEngine
        from sovereign_gauntlets import GauntletSystem
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        gauntlet_system = GauntletSystem()
        
        problem = analyzer.analyze_problem(
            "Implement a caching system with TTL and eviction policies",
            "Caching System"
        )
        
        plan = engine.decompose(problem)
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        
        assert len(results) > 0
        
        logger.info(f"✓ Gauntlets executed: {len(results)} gauntlets")
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            logger.info(f"  {name}: {status} ({result.score:.2f})")
        
        overall = gauntlet_system.get_overall_quality(results)
        logger.info(f"  Overall Quality: {overall:.2f}")
        return True
    except Exception as e:
        logger.error(f"✗ Gauntlets failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    logger.info("="*60)
    logger.info("PRODUCTION-READY VALIDATION")
    logger.info("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Problem Analyzer", test_problem_analyzer()))
    results.append(("Decomposition Engine", test_decomposition_engine()))
    results.append(("Gauntlets", test_gauntlets()))
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    logger.info(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        logger.info("\n🎉 ALL VALIDATIONS PASSED - PRODUCTION READY!")
        return 0
    else:
        logger.info("\n⚠️  SOME VALIDATIONS FAILED - NEEDS WORK")
        return 1

if __name__ == "__main__":
    sys.exit(main())
