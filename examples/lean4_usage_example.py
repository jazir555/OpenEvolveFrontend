"""
Simple Lean 4 Integration Usage Example

This example demonstrates the basic usage of the enhanced Lean 4 integration
with LeanAide server.
"""

import asyncio
from lean4_integration import (
    create_lean4_verification_engine,
    AutoformalizationEngine,
    ProofSearchEngine,
    Lean4ServerConfig,
    Lean4VerificationConfig
)


async def main():
    """Main example demonstrating Lean 4 integration features"""

    print("="*80)
    print("Lean 4 Integration - Usage Example")
    print("="*80)

    # 1. Configure the engine
    print("\n1. Setting up Lean 4 verification engine...")
    server_config = Lean4ServerConfig(
        host="localhost",
        port=7654,
        enable_simulation_fallback=True  # Use simulation if server unavailable
    )

    verification_config = Lean4VerificationConfig(
        enable_caching=True,
        cache_ttl_seconds=3600
    )

    engine = create_lean4_verification_engine(
        server_url="http://localhost:7654",
        server_config=server_config,
        config=verification_config
    )

    print("   ✓ Engine created")
    print(f"   Server: http://localhost:7654")
    print(f"   Fallback enabled: {server_config.enable_simulation_fallback}")
    print(f"   Caching enabled: {verification_config.enable_caching}")

    # 2. Autoformalization - Natural Language to Lean Code
    print("\n2. Autoformalizing natural language to Lean code...")
    print("-" * 80)

    auto = AutoformalizationEngine(engine.client, engine.cache)

    # Example: Convert natural language theorem to Lean
    natural_theorem = "For all natural numbers n, n + 0 = n"
    print(f"Input: {natural_theorem}")

    result = await auto.autoformalize(
        natural_language=natural_theorem,
        statement_type="theorem",
        name="add_zero"
    )

    print(f"\nSuccess: {'✓' if result.success else '✗'}")
    print(f"Server Available: {'✓' if result.server_available else '✗'}")

    if result.success:
        print("\nGenerated Lean Code:")
        print(result.lean_code)
    else:
        print(f"Errors: {result.errors}")

    # 3. Verification - Verify Lean Code
    print("\n3. Verifying Lean code...")
    print("-" * 80)

    lean_code = """
theorem mul_one (n : Nat) : n * 1 = n := by
  sorry
    """.strip()

    print(f"Code: {lean_code[:60]}...")

    verification_result = await engine.verify_mathematical_solution(lean_code)

    print(f"\nSuccess: {'✓' if verification_result.success else '✗'}")
    print(f"Verification Time: {verification_result.verification_time:.2f}s")
    print(f"Server Available: {'✓' if verification_result.server_available else '✗'}")
    print(f"Used Fallback: {'✓' if verification_result.used_fallback else '✗'}")

    if verification_result.errors:
        print(f"Errors: {verification_result.errors}")

    # 4. Similarity Search - Find Related Theorems
    print("\n4. Searching for related theorems...")
    print("-" * 80)

    search = ProofSearchEngine(engine.client, engine.cache)

    query = "additive identity"
    print(f"Query: '{query}'")

    search_results = await search.search_related_theorems(
        query=query,
        num_results=3,
        search_field="docString"
    )

    print(f"\nFound {len(search_results)} similar theorems:")

    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. {result.name}")
        print(f"   Type: {result.type}")
        print(f"   Distance: {result.distance:.4f}")
        if result.doc_string:
            preview = result.doc_string[:80] + "..." if len(result.doc_string) > 80 else result.doc_string
            print(f"   Documentation: {preview}")

    # 5. Batch Verification - Verify Multiple Theorems
    print("\n5. Batch verification...")
    print("-" * 80)

    theorems = [
        "theorem thm1 (n : Nat) : n + 0 = n := by sorry",
        "theorem thm2 (n m : Nat) : n + m = m + n := by sorry",
        "theorem thm3 (n : Nat) : n * 1 = n := by sorry"
    ]

    print(f"Verifying {len(theorems)} theorems concurrently...")

    batch_results = await engine.batch_verify(theorems)

    print("\nResults:")
    for i, (code, result) in enumerate(zip(theorems, batch_results), 1):
        status = "✓" if result.success else "✗"
        fallback = " (fallback)" if result.used_fallback else ""
        print(f"{i}. {status}{fallback} - {code[:50]}...")

    success_count = sum(1 for r in batch_results if r.success)
    print(f"\nBatch Summary: {success_count}/{len(theorems)} successful")

    # 6. Find Proof Strategy
    print("\n6. Finding proof strategy...")
    print("-" * 80)

    theorem_statement = "For all natural numbers n, n * 1 = n"
    print(f"Theorem: {theorem_statement}")

    strategy = await search.find_proof_strategy(theorem_statement)

    print(f"\nConfidence: {strategy['confidence']:.2f}")
    print(f"Suggested Strategies: {', '.join(strategy['suggested_strategies'])}")
    print(f"Similar Theorems: {len(strategy['similar_theorems'])}")

    if strategy['similar_theorems']:
        print("\nTop similar theorem:")
        top = strategy['similar_theorems'][0]
        print(f"  Name: {top['name']}")
        print(f"  Distance: {top['distance']:.4f}")

    # Cleanup
    print("\n7. Cleaning up...")
    await engine.close()
    print("   ✓ Connection closed")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\n\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()
