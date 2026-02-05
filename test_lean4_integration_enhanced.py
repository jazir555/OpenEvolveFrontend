"""
Enhanced Lean 4 Integration Test Suite

Demonstrates the full capabilities of the enhanced lean4_integration module:
- Real LeanAide server integration
- Autoformalization pipeline
- Proof search and retrieval
- Batch verification
- Dependency graph analysis
- Comprehensive caching
- Fallback to simulation when server unavailable
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import lean4_integration
sys.path.insert(0, str(Path(__file__).parent))

from lean4_integration import (
    Lean4VerificationEngine,
    LeanAideClient,
    AutoformalizationEngine,
    ProofSearchEngine,
    DependencyGraphAnalyzer,
    MathematicalProblemProcessor,
    MathematicalProblemDetector,
    Lean4ServerConfig,
    Lean4VerificationConfig,
    VerificationResult,
    SimilaritySearchResult,
    AutoformalizationResult,
    create_lean4_verification_engine
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Lean4IntegrationTester:
    """Test suite for enhanced Lean 4 integration"""

    def __init__(self, server_url: str = "http://localhost:7654"):
        self.server_url = server_url
        self.server_config = Lean4ServerConfig(
            host="localhost",
            port=7654,
            timeout=600,
            enable_simulation_fallback=True
        )
        self.verification_config = Lean4VerificationConfig(
            enable_caching=True,
            cache_ttl_seconds=3600
        )

    async def test_server_connection(self) -> bool:
        """Test 1: Check if LeanAide server is available"""
        print("\n" + "="*80)
        print("TEST 1: Server Connection Check")
        print("="*80)

        client = LeanAideClient(self.server_url, self.server_config)

        try:
            is_available = await client.check_server_health()
            print(f"Server Status: {'[OK] Available' if is_available else '[FAIL] Unavailable'}")

            if is_available:
                print(f"Server URL: {self.server_url}")
                print("[OK] Test passed: Server is reachable")
            else:
                print("[WARN] Test warning: Server not available, will use fallback mode")

            await client.close()
            return is_available

        except Exception as e:
            print(f"[FAIL] Test failed: {e}")
            await client.close()
            return False

    async def test_autoformalization(self):
        """Test 2: Autoformalization - Natural Language to Lean Code"""
        print("\n" + "="*80)
        print("TEST 2: Autoformalization (Natural Language -> Lean Code)")
        print("="*80)

        client = LeanAideClient(self.server_url, self.server_config)
        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )
        autoformalization = AutoformalizationEngine(client, engine.cache)

        # Test cases
        test_cases = [
            {
                "name": "Simple Theorem",
                "text": "For all natural numbers n, n + 0 = n",
                "type": "theorem",
                "expected_lean_keywords": ["theorem", "Nat", "add_zero"]
            },
            {
                "name": "Lemma",
                "text": "If a function is injective, then f(x) = f(y) implies x = y",
                "type": "lemma",
                "expected_lean_keywords": ["lemma", "Function", "Injective"]
            },
            {
                "name": "Definition",
                "text": "A group is a set with an associative binary operation, identity element, and inverses",
                "type": "definition",
                "expected_lean_keywords": ["def", "Group", "structure"]
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['name']}")
            print(f"Input: {test_case['text']}")
            print("-" * 80)

            try:
                result = await autoformalization.autoformalize(
                    test_case['text'],
                    test_case['type'],
                    f"test_{test_case['name'].lower().replace(' ', '_')}"
                )

                print(f"Success: {'[OK]' if result.success else '[FAIL]'}")
                print(f"Server Available: {'[OK]' if result.server_available else '[FAIL]'}")

                if result.success:
                    print(f"Generated Lean Code:")
                    print("-" * 40)
                    print(result.lean_code)
                    print("-" * 40)

                    # Check for expected keywords
                    has_keywords = all(
                        kw.lower() in result.lean_code.lower()
                        for kw in test_case['expected_lean_keywords']
                    )
                    print(f"Expected keywords found: {'[OK]' if has_keywords else '[FAIL]'}")
                else:
                    print(f"Errors: {result.errors}")

                results.append(result)

            except Exception as e:
                print(f"[FAIL] Test case failed: {e}")
                results.append(None)

        await engine.close()
        return results

    async def test_similarity_search(self):
        """Test 3: Similarity Search - Find Related Theorems"""
        print("\n" + "="*80)
        print("TEST 3: Similarity Search (Find Related Theorems)")
        print("="*80)

        client = LeanAideClient(self.server_url, self.server_config)
        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )
        proof_search = ProofSearchEngine(client, engine.cache)

        # Test queries
        queries = [
            "additive identity",
            "group theory",
            "natural number",
            "injective function"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            print("-" * 80)

            try:
                results = await proof_search.search_related_theorems(
                    query,
                    num_results=3,
                    search_field="docString"
                )

                print(f"Found {len(results)} similar theorems:")

                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.name}")
                    print(f"   Type: {result.type}")
                    print(f"   Distance: {result.distance:.4f}")
                    print(f"   Module: {result.module}")
                    if result.doc_string:
                        preview = result.doc_string[:100] + "..." if len(result.doc_string) > 100 else result.doc_string
                        print(f"   Documentation: {preview}")

            except Exception as e:
                print(f"[FAIL] Search failed: {e}")

        await engine.close()

    async def test_verification(self):
        """Test 4: Verification - Verify Lean Code"""
        print("\n" + "="*80)
        print("TEST 4: Verification (Verify Lean Code)")
        print("="*80)

        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )

        # Test cases
        test_cases = [
            {
                "name": "Valid Theorem (with sorry)",
                "code": """
theorem test_add_zero (n : Nat) : n + 0 = n := by
  sorry
"""
            },
            {
                "name": "Valid Lemma",
                "code": """
lemma test_mul_one (n : Nat) : n * 1 = n := by
  sorry
"""
            },
            {
                "name": "Invalid Syntax",
                "code": "theorem invalid : this is not valid Lean code"
            }
        ]

        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print("-" * 80)
            print("Lean Code:")
            print(test_case['code'])
            print("-" * 80)

            try:
                result = await engine.verify_mathematical_solution(test_case['code'])

                print(f"Verification Result: {'[OK] Success' if result.success else '[FAIL] Failed'}")
                print(f"Server Available: {'[OK]' if result.server_available else '[FAIL]'}")
                print(f"Used Fallback: {'[OK]' if result.used_fallback else '[FAIL]'}")
                print(f"Verification Time: {result.verification_time:.2f}s")

                if result.errors:
                    print("Errors:")
                    for error in result.errors:
                        print(f"  - {error}")

                if result.proof_steps:
                    print("Proof Steps:")
                    for step in result.proof_steps:
                        print(f"  - {step}")

            except Exception as e:
                print(f"[FAIL] Verification failed: {e}")

        await engine.close()

    async def test_batch_verification(self):
        """Test 5: Batch Verification"""
        print("\n" + "="*80)
        print("TEST 5: Batch Verification (Multiple Theorems)")
        print("="*80)

        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )

        # Batch of code to verify
        batch_code = [
            "theorem batch1 (n : Nat) : n + 0 = n := by sorry",
            "theorem batch2 (n m : Nat) : n + m = m + n := by sorry",
            "theorem batch3 (n : Nat) : n * 1 = n := by sorry",
            "theorem batch4 (a b : Nat) : a ≤ b -> a + 1 ≤ b + 1 := by sorry",
            "theorem batch5 : ∀ n, Nat.succ n > 0 := by sorry"
        ]

        print(f"Verifying {len(batch_code)} theorems in parallel...")
        print("-" * 80)

        try:
            results = await engine.batch_verify(batch_code)

            for i, result in enumerate(results, 1):
                status = "[OK] Success" if result.success else "[FAIL] Failed"
                fallback = " (fallback)" if result.used_fallback else ""
                print(f"{i}. {status}{fallback} - {batch_code[i-1][:50]}...")

            success_count = sum(1 for r in results if r.success)
            print(f"\nBatch Summary: {success_count}/{len(results)} successful")

        except Exception as e:
            print(f"[FAIL] Batch verification failed: {e}")

        await engine.close()

    async def test_full_pipeline(self):
        """Test 6: Full Pipeline - End-to-End Problem Processing"""
        print("\n" + "="*80)
        print("TEST 6: Full Pipeline (End-to-End)")
        print("="*80)

        # Create engines
        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )
        client = engine.client
        autoformalization = AutoformalizationEngine(client, engine.cache)
        proof_search = ProofSearchEngine(client, engine.cache)
        dependency_analyzer = DependencyGraphAnalyzer(str(Path(__file__).parent / "LeanAide"))

        processor = MathematicalProblemProcessor(
            engine,
            autoformalization,
            proof_search,
            dependency_analyzer
        )

        # Test problem
        problem_description = """
        Mathematical Problem: Additive Identity

        Theorem: For all natural numbers n, we have n + 0 = n.

        This theorem states that adding zero to any natural number
        returns the same natural number.
        """

        print("Problem Description:")
        print("-" * 80)
        print(problem_description.strip())
        print("-" * 80)

        try:
            result = await processor.process_mathematical_problem(
                problem_description,
                enable_proof_search=True,
                enable_dependency_analysis=True
            )

            print("\nProcessing Results:")
            print(f"Mathematical Content Detected: {'[OK]' if result['has_mathematical_content'] else '[FAIL]'}")
            print(f"Components Extracted: {result['components_extracted']}")

            if result['components_extracted'] > 0:
                print("\nComponents:")
                for component in result['components']:
                    print(f"  - {component['type']}: {component['name']}")
                    print(f"    Statement: {component['statement'][:80]}...")

            if result.get('autoformalization_results'):
                print("\nAutoformalization Results:")
                for i, af_result in enumerate(result['autoformalization_results'], 1):
                    print(f"  {i}. Success: {'[OK]' if af_result['success'] else '[FAIL]'}")
                    if af_result['success']:
                        print(f"     Server Available: {'[OK]' if af_result['server_available'] else '[FAIL]'}")

            print("\nVerification Result:")
            vr = result['verification_result']
            print(f"  Success: {'[OK]' if vr['success'] else '[FAIL]'}")
            print(f"  Server Available: {'[OK]' if vr['server_available'] else '[FAIL]'}")
            print(f"  Verification Time: {vr['verification_time']:.2f}s")

            if result.get('proof_search_results'):
                print("\nProof Search Results:")
                ps = result['proof_search_results']
                print(f"  Confidence: {ps['confidence']:.2f}")
                print(f"  Suggested Strategies: {', '.join(ps['suggested_strategies'])}")
                if ps['similar_theorems']:
                    print(f"  Similar Theorems Found: {len(ps['similar_theorems'])}")

            if result.get('dependency_analysis'):
                print("\nDependency Analysis:")
                deps = result['dependency_analysis']
                if deps.get('imports'):
                    print(f"  Imports: {', '.join(deps['imports'])}")

        except Exception as e:
            print(f"[FAIL] Pipeline processing failed: {e}")
            import traceback
            traceback.print_exc()

        await engine.close()

    async def test_caching(self):
        """Test 7: Caching - Verify Cache Performance"""
        print("\n" + "="*80)
        print("TEST 7: Caching (Performance Test)")
        print("="*80)

        engine = create_lean4_verification_engine(
            self.server_url,
            self.server_config,
            self.verification_config
        )

        test_code = "theorem cache_test (n : Nat) : n + 0 = n := by sorry"

        print(f"Testing cache with code: {test_code[:60]}...")
        print("-" * 80)

        # First call - should hit server
        print("\nFirst verification (should hit server):")
        import time
        start = time.time()
        result1 = await engine.verify_mathematical_solution(test_code)
        time1 = time.time() - start
        print(f"  Time: {time1:.2f}s")
        print(f"  Success: {'[OK]' if result1.success else '[FAIL]'}")

        # Second call - should hit cache
        print("\nSecond verification (should hit cache):")
        start = time.time()
        result2 = await engine.verify_mathematical_solution(test_code)
        time2 = time.time() - start
        print(f"  Time: {time2:.2f}s")
        print(f"  Success: {'[OK]' if result2.success else '[FAIL]'}")

        speedup = time1 / time2 if time2 > 0 else 0
        print(f"\nCache speedup: {speedup:.1f}x")

        if speedup > 2:
            print("[OK] Cache is working effectively!")
        else:
            print("[WARN] Cache may not be optimally configured")

        await engine.close()

    async def run_all_tests(self):
        """Run all tests"""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "Lean 4 Integration Test Suite" + " " * 28 + "║")
        print("╚" + "=" * 78 + "╝")

        tests = [
            ("Server Connection", self.test_server_connection),
            ("Autoformalization", self.test_autoformalization),
            ("Similarity Search", self.test_similarity_search),
            ("Verification", self.test_verification),
            ("Batch Verification", self.test_batch_verification),
            ("Full Pipeline", self.test_full_pipeline),
            ("Caching", self.test_caching)
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                await test_func()
                results[test_name] = "[OK] Completed"
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                results[test_name] = f"[FAIL] Failed: {e}"

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for test_name, status in results.items():
            print(f"{test_name:.<50} {status}")
        print("=" * 80)


async def main():
    """Main entry point"""
    # Check for command line arguments
    server_url = "http://localhost:7654"
    if len(sys.argv) > 1:
        server_url = sys.argv[1]

    print(f"Testing with LeanAide server at: {server_url}")
    print("Note: If server is unavailable, tests will use fallback simulation mode")

    tester = Lean4IntegrationTester(server_url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
