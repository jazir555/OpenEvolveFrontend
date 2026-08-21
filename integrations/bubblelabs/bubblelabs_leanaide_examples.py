"""
BubbleLabs-LeanAide Integration Examples

This module provides example workflows demonstrating the integration
between BubbleLabs and LeanAide components.

Examples include:
    - Basic theorem proving workflow
    - MCTS search visualization
    - Interactive proof verification
    - Mathematical query workflow
    - Batch theorem processing

Author: OpenEvolve
Created: 2025-01-03
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        LeanAideTaskType,
        initialize_leanaide_integration,
        LEANAIDE_AVAILABLE,
        MCTS_AVAILABLE,
        MDAP_AVAILABLE
    )
    LEANAIDE_INTEGRATION_AVAILABLE = True
except ImportError:
    LEANAIDE_INTEGRATION_AVAILABLE = False
    logger.error("LeanAide integration not available")


# =============================================================================
# Example 1: Basic Theorem Proving Workflow
# =============================================================================

def example_basic_theorem_proving():
    """
    Example 1: Basic workflow for proving a simple theorem.

    Workflow:
        1. Translate natural language theorem to Lean
        2. Generate proof sketch
        3. Verify generated code

    This demonstrates the basic LeanAide pipeline.
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Theorem Proving Workflow")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE:
        print("[FAIL] LeanAide integration not available")
        return

    # Initialize bridge
    bridge = get_leanaide_bridge()

    # Theorem to prove
    theorem = "There are infinitely many prime numbers"
    theorem_name = "infinitely_many_primes"

    print(f"\n📝 Theorem: {theorem}")
    print(f"📝 Name: {theorem_name}")

    # Step 1: Translate to Lean
    print("\n🔄 Step 1: Translating to Lean...")
    translation_result = bridge.execute_task(
        LeanAideTaskType.TRANSLATE_THEOREM,
        theorem_text=theorem,
        theorem_name=theorem_name
    )

    if translation_result.success:
        print(f"[OK] Translation successful ({translation_result.execution_time:.2f}s)")
        print(f"\nGenerated Lean code:")
        print(translation_result.data.get("lean_code", "N/A"))
    else:
        print(f"[FAIL] Translation failed: {translation_result.error}")
        return

    # Step 2: Generate proof
    print("\n📐 Step 2: Generating proof...")
    proof_result = bridge.execute_task(
        LeanAideTaskType.GENERATE_PROOF,
        theorem_text=theorem
    )

    if proof_result.success:
        print(f"[OK] Proof generated ({proof_result.execution_time:.2f}s)")
        print(f"\nProof sketch:")
        proof_doc = proof_result.data.get("proof_document", "")
        print(proof_doc[:500] + "..." if len(proof_doc) > 500 else proof_doc)

        if proof_result.data.get("lean_proof"):
            print(f"\nLean proof code:")
            print(proof_result.data["lean_proof"][:500] + "..." if len(proof_result.data["lean_proof"]) > 500 else proof_result.data["lean_proof"])
    else:
        print(f"[FAIL] Proof generation failed: {proof_result.error}")

    # Step 3: Verify if proof code exists
    if proof_result.success and proof_result.data.get("lean_proof"):
        print("\n[OK] Step 3: Verifying proof...")
        verify_result = bridge.execute_task(
            LeanAideTaskType.VERIFY_SOLUTION,
            code=proof_result.data["lean_proof"]
        )

        if verify_result.success:
            is_valid = verify_result.data.get("is_valid", False)
            unproven = verify_result.data.get("unproven_count", 0)
            print(f"[OK] Verification complete ({verify_result.execution_time:.2f}s)")
            print(f"   Valid: {is_valid}")
            print(f"   Unproven obligations: {unproven}")
        else:
            print(f"[FAIL] Verification failed: {verify_result.error}")

    print("\n" + "=" * 80)


# =============================================================================
# Example 2: MCTS Search with Visualization
# =============================================================================

def example_mcts_search():
    """
    Example 2: MCTS-based proof search with tree visualization.

    Workflow:
        1. Configure MCTS parameters
        2. Run MCTS search
        3. Visualize search tree
        4. Analyze agent performance

    This demonstrates MCTS-MDAP integration for automated proof search.
    """
    print("\n" + "=" * 80)
    print("Example 2: MCTS Search with Visualization")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE or not MCTS_AVAILABLE:
        print("[FAIL] MCTS not available")
        return

    bridge = get_leanaide_bridge()

    # Simple theorem for quick search
    theorem = "forall (n m : Nat), n + m = m + n"
    theorem_name = "add_comm"

    print(f"\n📝 Theorem: {theorem}")

    # Configure MCTS
    print("\n⚙️  MCTS Configuration:")
    print("   Max iterations: 500")
    print("   Time budget: 60s")
    print("   Expansion agents: 3")
    print("   Simulation voters: 5")

    # Run MCTS search
    print("\n🌳 Running MCTS search...")
    result = bridge.execute_task(
        LeanAideTaskType.MCTS_SEARCH,
        theorem=theorem,
        theorem_name=theorem_name,
        max_iterations=500,
        time_budget=60.0,
        c_param=1.414,
        expansion_agents=3,
        simulation_voters=5
    )

    if result.success and result.visualization_data:
        print(f"[OK] MCTS search complete ({result.execution_time:.2f}s)")

        # Get tree visualization
        tree_id = result.visualization_data["tree_id"]
        tree = bridge.get_tree(tree_id)

        print(f"\n📊 Tree Statistics:")
        print(f"   Total nodes: {len(tree.nodes)}")
        print(f"   Iterations: {tree.iterations}")
        print(f"   Max depth: {tree.statistics['max_depth']}")
        print(f"   Win rate: {tree.statistics['win_rate']:.3f}")
        print(f"   Confidence: {tree.statistics['confidence']:.3f}")

        # Best path
        print(f"\n🎯 Best Path ({len(tree.best_path)} steps):")
        for node_id in tree.best_path:
            node = tree.nodes.get(node_id)
            if node:
                print(f"   {node.depth}. {node.action} (visits={node.visits}, value={node.value:.3f})")

        # Agent performance
        if tree.statistics.get("agent_statistics"):
            print(f"\n🤖 Agent Performance:")
            for agent_id, stats in sorted(
                tree.statistics["agent_statistics"].items(),
                key=lambda x: x[1].get("votes_cast", 0),
                reverse=True
            ):
                votes = stats.get("votes_cast", 0)
                accepted = stats.get("votes_accepted", 0)
                success_rate = stats.get("success_rate", 0)
                print(f"   {agent_id}:")
                print(f"      Votes: {votes}")
                print(f"      Accepted: {accepted}")
                print(f"      Success rate: {success_rate:.2%}")

        # Red flag analysis
        if tree.statistics.get("red_flag_analysis"):
            red_flags = tree.statistics["red_flag_analysis"]
            print(f"\n🚩 Red Flag Analysis:")
            print(f"   Flagged nodes: {red_flags.get('red_flagged_nodes', 0)}")
            print(f"   Flag rate: {red_flags.get('red_flag_rate', 0):.2%}")

        # Export tree
        print(f"\n💾 Tree exported with ID: {tree_id}")
        print(f"   Access with: bridge.get_tree('{tree_id}')")

    else:
        print(f"[FAIL] MCTS search failed: {result.error}")

    print("\n" + "=" * 80)


# =============================================================================
# Example 3: Interactive Proof Verification
# =============================================================================

def example_interactive_verification():
    """
    Example 3: Interactive Lean4 code verification workflow.

    Workflow:
        1. User provides Lean code
        2. Elaborate and check for errors
        3. Display proof state
        4. Show verification results

    This demonstrates Lean4 integration for code verification.
    """
    print("\n" + "=" * 80)
    print("Example 3: Interactive Proof Verification")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE:
        print("[FAIL] LeanAide integration not available")
        return

    bridge = get_leanaide_bridge()

    # Example Lean code
    lean_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]

theorem mul_comm (a b : Nat) : a * b = b * a := by
  simp [Nat.mul_comm]

theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  simp [Nat.add_assoc]
"""

    print("\n📝 Lean Code to Verify:")
    print(lean_code)

    # Step 1: Elaborate code
    print("\n🔍 Step 1: Elaborating code...")
    elaborate_result = bridge.execute_task(
        LeanAideTaskType.ELABORATE_CODE,
        code=lean_code
    )

    if elaborate_result.success:
        print(f"[OK] Elaboration successful ({elaborate_result.execution_time:.2f}s)")

        declarations = elaborate_result.data.get("declarations", [])
        logs = elaborate_result.data.get("logs", [])
        sorries = elaborate_result.data.get("sorries_after_purge", [])

        print(f"\n📋 Declarations found: {len(declarations)}")
        for decl in declarations:
            print(f"   - {decl}")

        if sorries:
            print(f"\n[WARN]  Unsolved goals: {len(sorries)}")
            for sorry in sorries[:3]:  # Show first 3
                print(f"   - {sorry}")
        else:
            print(f"\n[OK] All goals solved!")

    else:
        print(f"[FAIL] Elaboration failed: {elaborate_result.error}")
        return

    # Step 2: Full verification
    print("\n[OK] Step 2: Verifying correctness...")
    verify_result = bridge.execute_task(
        LeanAideTaskType.VERIFY_SOLUTION,
        code=lean_code
    )

    if verify_result.success:
        is_valid = verify_result.data.get("is_valid", False)
        unproven = verify_result.data.get("unproven_count", 0)

        print(f"[OK] Verification complete ({verify_result.execution_time:.2f}s)")
        print(f"   Valid: {is_valid}")
        print(f"   Unproven obligations: {unproven}")

        if verify_result.data.get("errors"):
            print(f"\n[FAIL] Errors:")
            for error in verify_result.data["errors"][:3]:
                print(f"   - {error}")

    else:
        print(f"[FAIL] Verification failed: {verify_result.error}")

    print("\n" + "=" * 80)


# =============================================================================
# Example 4: Mathematical Query Workflow
# =============================================================================

def example_math_queries():
    """
    Example 4: Mathematical query and Q&A workflow.

    Workflow:
        1. Ask multiple math questions
        2. Get multiple answers per question
        3. Display and compare answers
        4. Analyze response quality

    This demonstrates the math query capabilities.
    """
    print("\n" + "=" * 80)
    print("Example 4: Mathematical Query Workflow")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE:
        print("[FAIL] LeanAide integration not available")
        return

    bridge = get_leanaide_bridge()

    # Questions to ask
    questions = [
        "What is the fundamental theorem of calculus?",
        "Exproof the Pythagorean theorem",
        "What is a prime number?",
        "Define a continuous function",
        "What is the difference between a theorem and a lemma?"
    ]

    print(f"\n❓ Asking {len(questions)} questions...")

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")

        result = bridge.execute_task(
            LeanAideTaskType.MATH_QUERY,
            query=question,
            n=3  # Get 3 answers
        )

        if result.success:
            print(f"[OK] Received {result.data.get('num_answers', 0)} answers ({result.execution_time:.2f}s)")

            answers = result.data.get("answers", [])
            for j, answer in enumerate(answers, 1):
                # Truncate long answers
                display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"\n   Answer {j}:")
                print(f"   {display_answer}")

            results.append({
                "question": question,
                "success": True,
                "num_answers": result.data.get("num_answers", 0),
                "execution_time": result.execution_time
            })
        else:
            print(f"[FAIL] Query failed: {result.error}")
            results.append({
                "question": question,
                "success": False,
                "error": result.error
            })

    # Summary
    print("\n" + "=" * 80)
    print("📊 Query Summary")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    total_time = sum(r.get("execution_time", 0) for r in results if r["success"])

    print(f"\nTotal queries: {len(questions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(questions) - successful}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / len(questions):.2f}s")

    print("\n" + "=" * 80)


# =============================================================================
# Example 5: Batch Theorem Processing
# =============================================================================

def example_batch_processing():
    """
    Example 5: Batch processing multiple theorems.

    Workflow:
        1. Define list of theorems
        2. Process in parallel
        3. Collect results
        4. Generate summary report

    This demonstrates batch processing capabilities.
    """
    print("\n" + "=" * 80)
    print("Example 5: Batch Theorem Processing")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE:
        print("[FAIL] LeanAide integration not available")
        return

    bridge = get_leanaide_bridge()

    # Theorems to process
    theorems = [
        ("The product of two even numbers is even", "even_product_even"),
        ("The square root of 2 is irrational", "sqrt2_irrational"),
        ("There are infinitely many primes", "inf_primes"),
        ("Every natural number has a unique prime factorization", "prime_factor_unique"),
        ("The sum of two even numbers is even", "even_sum_even")
    ]

    print(f"\n📝 Processing {len(theorems)} theorems...")

    results = []
    for theorem, name in theorems:
        print(f"\n🔄 Processing: {name}")
        print(f"   Theorem: {theorem}")

        result = bridge.execute_task(
            LeanAideTaskType.TRANSLATE_THEOREM,
            theorem_text=theorem,
            theorem_name=name
        )

        status = "[OK]" if result.success else "[FAIL]"
        print(f"   {status} {result.execution_time:.2f}s")

        if result.success:
            lean_code = result.data.get("lean_code", "")
            code_preview = lean_code[:100] + "..." if len(lean_code) > 100 else lean_code
            print(f"   Code: {code_preview}")

        results.append({
            "name": name,
            "theorem": theorem,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error if not result.success else None
        })

    # Generate report
    print("\n" + "=" * 80)
    print("📊 Batch Processing Report")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_time = sum(r["execution_time"] for r in results)

    print(f"\nTotal theorems: {len(theorems)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / len(theorems):.2f}s")

    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"\n{status} {result['name']}")
        print(f"   Theorem: {result['theorem']}")
        print(f"   Time: {result['execution_time']:.2f}s")
        if result["error"]:
            print(f"   Error: {result['error']}")

    print("\n" + "=" * 80)


# =============================================================================
# Example 6: Complete Workflow with MCTS + MDAP
# =============================================================================

def example_complete_workflow():
    """
    Example 6: Complete theorem proving workflow with MCTS and MDAP.

    Workflow:
        1. Translate theorem
        2. Run MCTS search
        3. Analyze voting patterns
        4. Verify best proof
        5. Generate report

    This demonstrates a full integrated workflow.
    """
    print("\n" + "=" * 80)
    print("Example 6: Complete Workflow with MCTS + MDAP")
    print("=" * 80)

    if not LEANAIDE_INTEGRATION_AVAILABLE or not MCTS_AVAILABLE:
        print("[FAIL] Required components not available")
        return

    bridge = get_leanaide_bridge()

    # Theorem
    theorem = "forall (n m : Nat), n * m = m * n"
    theorem_name = "mul_comm"

    print(f"\n📝 Theorem: {theorem}")
    print(f"📝 Name: {theorem_name}")

    # Step 1: Translate
    print("\n🔄 Step 1: Translating theorem...")
    translation = bridge.execute_task(
        LeanAideTaskType.TRANSLATE_THEOREM,
        theorem_text=theorem,
        theorem_name=theorem_name
    )

    if translation.success:
        print(f"[OK] Translation complete ({translation.execution_time:.2f}s)")
        print(f"   Code: {translation.data.get('lean_code', '')[:100]}...")
    else:
        print(f"[FAIL] Translation failed: {translation.error}")
        return

    # Step 2: MCTS Search
    print("\n🌳 Step 2: Running MCTS search...")
    mcts_result = bridge.execute_task(
        LeanAideTaskType.MCTS_SEARCH,
        theorem=theorem,
        theorem_name=theorem_name,
        max_iterations=1000,
        time_budget=120.0,
        expansion_agents=5,
        simulation_voters=7
    )

    if not mcts_result.success:
        print(f"[FAIL] MCTS search failed: {mcts_result.error}")
        return

    print(f"[OK] MCTS search complete ({mcts_result.execution_time:.2f}s)")

    # Step 3: Analyze results
    tree_id = mcts_result.visualization_data["tree_id"]
    tree = bridge.get_tree(tree_id)

    print(f"\n📊 Search Statistics:")
    print(f"   Iterations: {tree.iterations}")
    print(f"   Tree nodes: {len(tree.nodes)}")
    print(f"   Max depth: {tree.statistics['max_depth']}")
    print(f"   Win rate: {tree.statistics['win_rate']:.3f}")
    print(f"   Confidence: {tree.statistics['confidence']:.3f}")

    # Step 4: Agent analysis
    if tree.statistics.get("agent_statistics"):
        print(f"\n🤖 Agent Voting Analysis:")
        agent_stats = tree.statistics["agent_statistics"]

        # Sort by performance
        sorted_agents = sorted(
            agent_stats.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True
        )

        for i, (agent_id, stats) in enumerate(sorted_agents[:5], 1):
            success_rate = stats.get("success_rate", 0)
            votes_cast = stats.get("votes_cast", 0)
            votes_accepted = stats.get("votes_accepted", 0)
            print(f"   {i}. {agent_id}:")
            print(f"      Success rate: {success_rate:.2%}")
            print(f"      Votes: {votes_cast} cast, {votes_accepted} accepted")

    # Step 5: Best proof
    print(f"\n🎯 Best Proof Path:")
    for node_id in tree.best_path:
        node = tree.nodes.get(node_id)
        if node and node.action:
            print(f"   {node.depth}. {node.action}")

    # Step 6: Generate report
    print(f"\n📋 Final Report:")
    print(f"   Theorem: {theorem}")
    print(f"   Translation: [OK] Success")
    print(f"   MCTS Search: [OK] Success")
    print(f"   Search iterations: {tree.iterations}")
    print(f"   Best win rate: {tree.statistics['win_rate']:.3f}")
    print(f"   Tree ID: {tree_id}")

    print("\n" + "=" * 80)


# =============================================================================
# Main Runner
# =============================================================================

def run_all_examples():
    """Run all examples sequentially."""
    examples = [
        ("Basic Theorem Proving", example_basic_theorem_proving),
        ("MCTS Search", example_mcts_search),
        ("Interactive Verification", example_interactive_verification),
        ("Math Queries", example_math_queries),
        ("Batch Processing", example_batch_processing),
        ("Complete Workflow", example_complete_workflow)
    ]

    print("\n" + "=" * 80)
    print("BubbleLabs-LeanAide Integration Examples")
    print("=" * 80)
    print(f"\nAvailable components:")
    print(f"   LeanAide Client: {'[OK]' if LEANAIDE_AVAILABLE else '[FAIL]'}")
    print(f"   MCTS-MDAP: {'[OK]' if MCTS_AVAILABLE else '[FAIL]'}")
    print(f"   MCP Tools: {'[OK]' if MDAP_AVAILABLE else '[FAIL]'}")

    # Initialize
    if LEANAIDE_INTEGRATION_AVAILABLE:
        print("\n🔧 Initializing LeanAide integration...")
        status = initialize_leanaide_integration()
        print(f"[OK] Initialization complete")
    else:
        print("\n[FAIL] Cannot run examples - integration not available")
        return

    # Run examples
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"\n[FAIL] Example '{name}' failed with error: {e}")
            logger.error(f"Example failed", exc_info=True)

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_map = {
            "basic": example_basic_theorem_proving,
            "mcts": example_mcts_search,
            "verify": example_interactive_verification,
            "math": example_math_queries,
            "batch": example_batch_processing,
            "complete": example_complete_workflow
        }

        example_name = sys.argv[1].lower()
        if example_name in example_map:
            example_map[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(example_map.keys())}")
    else:
        # Run all examples
        run_all_examples()
