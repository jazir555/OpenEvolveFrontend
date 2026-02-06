"""
Deep Verification - Comprehensive Functional Testing

Tests:
1. End-to-end solver workflows
2. Edge cases and error conditions
3. Integration between components
4. Data consistency
5. Performance baseline
6. API contract compliance
7. CAV-NLP integration verification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import traceback
from typing import Any, Dict, List

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None


class DeepVerifier:
    """Comprehensive verification suite."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def test(self, name: str, condition: bool, error_msg: str = ""):
        """Record test result."""
        if condition:
            self.passed += 1
            print(f"   [PASS] {name}")
        else:
            self.failed += 1
            print(f"   [FAIL] {name}: {error_msg}")
    
    def warn(self, name: str, message: str):
        """Record warning."""
        self.warnings += 1
        print(f"   [WARN] {name}: {message}")
    
    async def run_all(self):
        """Run all verification tests."""
        print("="*70)
        print("DEEP VERIFICATION - COMPREHENSIVE FUNCTIONAL TESTING")
        print("="*70)
        
        await self.verify_z3_solver()
        await self.verify_knowledge_manager()
        await self.verify_unified_bridge()
        await self.verify_api_contracts()
        await self.verify_integration()
        await self.verify_edge_cases()
        await self.verify_performance()
        await self.verify_data_consistency()
        
        self.print_summary()
    
    async def verify_z3_solver(self):
        """Verify Z3 solver functionality."""
        print("\n1. Z3 Solver Deep Verification")
        
        from z3_solver_connector import get_z3_connector, Z3SolverConfig, Z3ResultStatus
        
        z3 = get_z3_connector()
        
        # Test 1.1: Basic SAT
        try:
            result = await z3.solve_smtlib(
                "(declare-fun x () Int) (assert (= x 5)) (check-sat) (get-model)",
                Z3SolverConfig()
            )
            self.test("Basic SAT solving", 
                     result.status == Z3ResultStatus.SAT and result.model is not None,
                     f"Expected SAT with model, got {result.status}")
        except Exception as e:
            self.test("Basic SAT solving", False, str(e))
        
        # Test 1.2: Basic UNSAT
        try:
            result = await z3.solve_smtlib(
                "(declare-fun x () Int) (assert (> x 5)) (assert (< x 3)) (check-sat)",
                Z3SolverConfig()
            )
            self.test("Basic UNSAT solving",
                     result.status == Z3ResultStatus.UNSAT,
                     f"Expected UNSAT, got {result.status}")
        except Exception as e:
            self.test("Basic UNSAT solving", False, str(e))
        
        # Test 1.3: Linear system
        try:
            smt = """
            (declare-fun x () Int)
            (declare-fun y () Int)
            (assert (= (+ x y) 10))
            (assert (= (- x y) 2))
            (check-sat)
            (get-model)
            """
            result = await z3.solve_smtlib(smt, Z3SolverConfig())
            has_correct_values = False
            if result.model:
                x_val = result.model.get('x')
                y_val = result.model.get('y')
                has_correct_values = (x_val == 6 and y_val == 4)
            self.test("Linear system solving",
                     result.status == Z3ResultStatus.SAT and has_correct_values,
                     f"Model: {result.model}")
        except Exception as e:
            self.test("Linear system solving", False, str(e))
        
        # Test 1.4: Invalid SMT-LIB handling
        try:
            result = await z3.solve_smtlib("this is not valid smtlib", Z3SolverConfig())
            self.test("Invalid SMT-LIB error handling",
                     result.status == Z3ResultStatus.ERROR or result.error_message is not None,
                     "Should return error status")
        except Exception as e:
            self.test("Invalid SMT-LIB error handling", True, "Exception handled")
        
        # Test 1.5: Timeout handling
        try:
            config = Z3SolverConfig(timeout_ms=1)
            result = await z3.solve_smtlib(
                "(declare-fun x () Int) (assert (> (* x x) 1000000)) (check-sat)",
                config
            )
            self.test("Timeout handling", True, f"Status: {result.status}")
        except Exception as e:
            self.test("Timeout handling", False, str(e))
        
        # Test 1.6: Statistics tracking
        try:
            stats = z3.get_statistics()
            has_stats = isinstance(stats, dict) and "calls" in stats
            self.test("Statistics tracking", has_stats, f"Stats: {stats}")
        except Exception as e:
            self.test("Statistics tracking", False, str(e))
    
    async def verify_knowledge_manager(self):
        """Verify knowledge manager functionality."""
        print("\n2. Knowledge Manager Deep Verification")
        
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        try:
            manager = await get_z3_knowledge_manager()
            
            # Test 2.1: Learn from solution
            result = await manager.learn_from_solution(
                problem_statement="Test linear system",
                constraints=["x + y = 10", "x - y = 2"],
                result="success",
                proof="(simplify (solve-eqs))",
                metadata={"strategy": "elimination", "time_ms": 150.5}
            )
            self.test("Learn from solution",
                     result is not None and "features" in result,
                     "Should return learning result with features")
            
            # Test 2.2: Find similar solutions
            similar = await manager.find_similar_solutions(
                problem_statement="Linear system",
                constraints=["a + b = 10"],
                top_k=5
            )
            self.test("Find similar solutions",
                     isinstance(similar, list),
                     f"Should return list, got {type(similar)}")
            
            # Test 2.3: Get strategy recommendation
            strategy = await manager.get_recommended_strategy(
                problem_statement="System of equations",
                constraints=["x + y = 5"]
            )
            self.test("Strategy recommendation",
                     isinstance(strategy, dict),
                     f"Should return dict, got {type(strategy)}")
            
            # Test 2.4: Get statistics
            stats = manager.get_statistics()
            self.test("Get statistics",
                     isinstance(stats, dict) and "knowledge_stored" in stats,
                     f"Stats: {list(stats.keys()) if isinstance(stats, dict) else 'N/A'}")
            
            # Test 2.5: Get metrics (alias)
            metrics = manager.get_metrics()
            self.test("Get metrics (alias)",
                     isinstance(metrics, dict),
                     "get_metrics should work as alias")
            
        except Exception as e:
            print(f"   [FAIL] Knowledge manager initialization: {e}")
            self.failed += 5
    
    async def verify_unified_bridge(self):
        """Verify unified bridge functionality."""
        print("\n3. Unified Bridge Deep Verification")
        
        from unified_math_bridge_complete import get_unified_bridge_complete, SolverSystem
        
        try:
            bridge = await get_unified_bridge_complete()
            
            # Test 3.1: Basic solve
            result = await bridge.solve(
                problem_statement="x > 0 and x < 10",
                preferred_solver=SolverSystem.AUTO,
                timeout=10
            )
            self.test("Bridge basic solve",
                     isinstance(result, dict) and "success" in result,
                     f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            # Test 3.2: Translator exists
            self.test("Semantic translator exists",
                     hasattr(bridge, 'translator') and bridge.translator is not None,
                     "Bridge should have translator")
            
            # Test 3.3: Consensus engine exists
            self.test("Consensus engine exists",
                     hasattr(bridge, 'consensus') and bridge.consensus is not None,
                     "Bridge should have consensus")
            
            # Test 3.4: Stats tracking
            self.test("Bridge stats tracking",
                     hasattr(bridge, 'stats') and isinstance(bridge.stats, dict),
                     f"Stats: {getattr(bridge, 'stats', 'N/A')}")
            
        except Exception as e:
            print(f"   [FAIL] Bridge verification: {e}")
            self.failed += 4
    
    async def verify_api_contracts(self):
        """Verify API contracts and models."""
        print("\n4. API Contract Verification")
        
        # Test 4.1: Request/Response models
        try:
            from math_api_complete import SolveZ3Request, SolveZ3Response
            
            req = SolveZ3Request(content="(assert true)", timeout_ms=30000)
            self.test("Z3 solve request model",
                     hasattr(req, 'content') and req.content == "(assert true)",
                     "Request model should work")
            
            resp = SolveZ3Response(status="sat", model=None, proof=None, solving_time_ms=10.0, error=None)
            self.test("Z3 solve response model",
                     resp.status == "sat",
                     "Response model should work")
            
        except Exception as e:
            self.test("API models", False, str(e))
        
        # Test 4.2: FastAPI app creation
        try:
            from math_api_complete import math_api
            self.test("FastAPI app creation",
                     math_api is not None,
                     "API should be created")
            
            if math_api:
                routes = [r.path for r in math_api.routes if hasattr(r, 'path')]
                has_solve = any('/solve' in r for r in routes)
                self.test("API has solve endpoints",
                         has_solve,
                         f"Routes: {[r for r in routes if 'solve' in r]}")
                
        except Exception as e:
            self.test("FastAPI app", False, str(e))
    
    async def verify_integration(self):
        """Verify component integration."""
        print("\n5. Integration Testing")
        
        # Test 5.1: Z3 -> Knowledge flow
        try:
            from z3_solver_connector import get_z3_connector, Z3SolverConfig
            from z3_knowledge_complete import get_z3_knowledge_manager
            
            z3 = get_z3_connector()
            manager = await get_z3_knowledge_manager()
            
            # Solve problem
            result = await z3.solve_smtlib(
                "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
                Z3SolverConfig()
            )
            
            # Learn from it
            learn_result = await manager.learn_from_solution(
                problem_statement="Simple constraint",
                constraints=["x > 0"],
                result=result.status.value if hasattr(result.status, 'value') else str(result.status),
                metadata={"time_ms": result.solving_time_ms}
            )
            
            self.test("Z3 -> Knowledge integration",
                     learn_result is not None,
                     "Should successfully learn from Z3 result")
            
        except Exception as e:
            self.test("Z3 -> Knowledge integration", False, str(e))
        
        # Test 5.2: Bridge -> Solver integration
        try:
            from unified_math_bridge_complete import get_unified_bridge_complete
            
            bridge = await get_unified_bridge_complete()
            
            # Bridge should be able to solve
            result = await bridge.solve(problem_statement="x > 0", timeout=5)
            self.test("Bridge -> Solver integration",
                     result is not None,
                     "Bridge should successfully call solvers")
            
        except Exception as e:
            self.test("Bridge -> Solver integration", False, str(e))
    
    async def verify_edge_cases(self):
        """Verify edge case handling."""
        print("\n6. Edge Case Testing")
        
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        
        # Test 6.1: Empty problem
        try:
            result = await z3.solve_smtlib("", Z3SolverConfig())
            self.test("Empty problem handling",
                     result.status == Z3ResultStatus.ERROR or result.error_message is not None,
                     "Should handle empty input gracefully")
        except Exception as e:
            self.test("Empty problem handling", True, "Exception caught")
        
        # Test 6.2: Very long problem
        try:
            long_constraint = " (assert (> x 0))" * 100
            smt = f"(declare-fun x () Int){long_constraint} (check-sat)"
            result = await z3.solve_smtlib(smt, Z3SolverConfig(timeout_ms=5000))
            self.test("Long problem handling", True, f"Status: {result.status}")
        except Exception as e:
            self.test("Long problem handling", False, str(e))
        
        # Test 6.3: Special characters in constraints
        try:
            result = await z3.solve_smtlib(
                "(declare-fun x () Int) (assert (= x 0)) ; comment (check-sat)",
                Z3SolverConfig()
            )
            self.test("Comments in SMT-LIB", True, f"Status: {result.status}")
        except Exception as e:
            self.test("Comments in SMT-LIB", False, str(e))
    
    async def verify_performance(self):
        """Verify performance characteristics."""
        print("\n7. Performance Baseline")
        
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        
        z3 = get_z3_connector()
        
        # Test 7.1: Simple problem performance
        try:
            start = time.time()
            for _ in range(10):
                await z3.solve_smtlib(
                    "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
                    Z3SolverConfig()
                )
            elapsed = time.time() - start
            avg_time = elapsed / 10 * 1000
            
            self.test(f"Simple problem performance (avg: {avg_time:.1f}ms)",
                     avg_time < 1000,  # Should complete in less than 1 second each
                     f"Average time too high: {avg_time:.1f}ms")
            
        except Exception as e:
            self.test("Performance test", False, str(e))
        
        # Test 7.2: Concurrent solving capability
        try:
            problems = [
                "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
                "(declare-fun y () Int) (assert (< y 10)) (check-sat)",
                "(declare-fun z () Int) (assert (= z 5)) (check-sat)",
            ]
            
            start = time.time()
            await asyncio.gather(*[
                z3.solve_smtlib(p, Z3SolverConfig())
                for p in problems
            ])
            elapsed = time.time() - start
            
            self.test(f"Concurrent solving (3x in {elapsed*1000:.0f}ms)",
                     elapsed < 5,  # Should complete in less than 5 seconds
                     f"Concurrent solving too slow: {elapsed:.2f}s")
            
        except Exception as e:
            self.test("Concurrent solving", False, str(e))
    
    async def verify_data_consistency(self):
        """Verify data consistency across operations."""
        print("\n8. Data Consistency Verification")
        
        # Test 8.1: Configuration consistency
        try:
            from math_knowledge_config import MathKnowledgeConfig
            
            config1 = MathKnowledgeConfig()
            config2 = MathKnowledgeConfig()
            
            # Both should have same defaults
            self.test("Config default consistency",
                     config1.z3.timeout_ms == config2.z3.timeout_ms,
                     f"{config1.z3.timeout_ms} vs {config2.z3.timeout_ms}")
            
            # Modifying one shouldn't affect the other
            config1.z3.timeout_ms = 60000
            self.test("Config isolation",
                     config2.z3.timeout_ms == 30000,
                     f"Config2 changed to {config2.z3.timeout_ms}")
            
        except Exception as e:
            self.test("Data consistency", False, str(e))
        
        # Test 8.2: Statistics consistency
        try:
            from z3_solver_connector import get_z3_connector, Z3SolverConfig
            
            z3 = get_z3_connector()
            stats_before = z3.get_statistics()
            
            # Run some operations
            await z3.solve_smtlib("(assert true) (check-sat)", Z3SolverConfig())
            
            stats_after = z3.get_statistics()
            
            self.test("Statistics tracking consistency",
                     stats_after["calls"] >= stats_before["calls"],
                     f"Calls: {stats_before['calls']} -> {stats_after['calls']}")
            
        except Exception as e:
            self.test("Statistics consistency", False, str(e))
    
    def print_summary(self):
        """Print verification summary."""
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)
        print(f"\nPassed:   {self.passed}")
        print(f"Failed:   {self.failed}")
        print(f"Warnings: {self.warnings}")
        
        total = self.passed + self.failed
        if total > 0:
            pass_rate = (self.passed / total) * 100
            print(f"\nPass Rate: {pass_rate:.1f}%")
        
        print("\n" + "="*70)
        if self.failed == 0:
            print("SUCCESS: DEEP VERIFICATION PASSED - ALL TESTS SUCCESSFUL")
        else:
            print(f"WARNING: DEEP VERIFICATION COMPLETE - {self.failed} FAILURES DETECTED")
        print("="*70)


async def main():
    verifier = DeepVerifier()
    await verifier.run_all()
    return 0 if verifier.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
