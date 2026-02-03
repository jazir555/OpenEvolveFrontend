"""
MDAP/MAKER Demo Script

Demonstrates LeanAide MDAP/MAKER integration capabilities with real examples.

Usage:
    python demo_mdap_maker.py                           # Run all demos
    python demo_mdap_maker.py basic                     # Basic MDAP demo
    python demo_mdap_maker.py maker                     # MAKER demo
    python demo_mdap_maker.py hybrid                    # Hybrid demo
    python demo_mdap_maker.py custom                    # Custom agent demo
    python demo_mdap_maker.py workflow                  # Workflow integration demo

Author: OpenEvolve Frontend Team
Version: 1.0.0
Date: 2025-12-30
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from mdap_engine import (
        MDAPOrchestrator, MDAPConfig, MDAPTask, MDAPStep,
        RedFlagRules, RedFlagger, MDAPCache
    )
    MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MDAP engine not available: {e}")
    MDAP_AVAILABLE = False

try:
    from roma_mdap_maker_engine import (
        ROMAMDAPMakerEngine, ROMAMDAPMakerConfig
    )
    ROMA_MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ROMA-MDAP-MAKER not available: {e}")
    ROMA_MDAP_AVAILABLE = False

try:
    from workflow_structures import ModelConfig, SubProblem, WorkflowState
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Workflow structures not available: {e}")
    WORKFLOW_AVAILABLE = False

try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem as AdaptiveSubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive MDAP not available: {e}")
    ADAPTIVE_MDAP_AVAILABLE = False

# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

class DemoRunner:
    """Demo runner for MDAP/MAKER examples"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "test-key")
        self.model = "gpt-4o-mini"

    async def run_basic_mdap_demo(self):
        """Demonstrate basic MDAP usage"""

        print("\n" + "=" * 80)
        print("DEMO 1: Basic MDAP Proof Generation")
        print("=" * 80 + "\n")

        if not MDAP_AVAILABLE:
            print("❌ MDAP not available. Skipping demo.")
            return

        try:
            # Configuration
            config = MDAPConfig(
                k_min=2,
                k_max=4,
                timeout_seconds=30
            )

            model_config = ModelConfig(
                provider="openai",
                model=self.model,
                api_key=self.api_key
            )

            # Create orchestrator
            orchestrator = MDAPOrchestrator(
                config=config,
                model_config=model_config
            )

            print("✓ MDAP Orchestrator created")
            print(f"  - k_min: {config.k_min}")
            print(f"  - k_max: {config.k_max}")
            print(f"  - timeout: {config.timeout_seconds}s")

            # Define proof task
            theorem = "∀ n : Nat, n + 0 = n"

            step = MDAPStep(
                step_id="add_zero_proof",
                prompt=f"Prove: {theorem}",
                task_type="theorem_proving",
                temperature_override=0.1
            )

            task = MDAPTask(
                task_id="add_zero",
                description="Prove addition with zero",
                steps=[step]
            )

            print(f"\n✓ Task defined:")
            print(f"  - Theorem: {theorem}")
            print(f"  - Steps: {len(task.steps)}")

            print("\n⏳ Executing MDAP task...")
            print("   (This would normally call the LLM API)")
            print("   For demo purposes, showing structure only...")

            # Show what would happen
            print("\n📊 Execution flow:")
            print("   1. Select k agents (2-4)")
            print("   2. Execute agents in parallel")
            print("   3. Aggregate votes")
            print("   4. Apply red-flagging")
            print("   5. Return best proof")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Basic MDAP demo error: {e}", exc_info=True)

    async def run_maker_demo(self):
        """Demonstrate MAKER workflow integration"""

        print("\n" + "=" * 80)
        print("DEMO 2: MAKER Workflow Integration")
        print("=" * 80 + "\n")

        if not WORKFLOW_AVAILABLE:
            print("❌ Workflow structures not available. Skipping demo.")
            return

        try:
            # Configure workflow
            state = WorkflowState()
            state.maker_enabled = True
            state.maker_config = {
                "maker_mode": "sequential",
                "maker_k_ahead": 3,
                "maker_max_depth": 5
            }

            print("✓ Workflow state configured")
            print(f"  - MAKER enabled: {state.maker_enabled}")
            print(f"  - Mode: {state.maker_config['maker_mode']}")
            print(f"  - K-ahead: {state.maker_config['maker_k_ahead']}")

            # Create sub-problem
            sub_problem = SubProblem(
                id="mul_one",
                title="Prove multiplication by one",
                description="∀ n : Nat, n * 1 = n",
                estimated_effort=5
            )

            print(f"\n✓ Sub-problem created:")
            print(f"  - ID: {sub_problem.id}")
            print(f"  - Title: {sub_problem.title}")
            print(f"  - Effort: {sub_problem.estimated_effort}")

            print("\n📊 MAKER execution flow:")
            print("   1. Decompose problem (if recursive mode)")
            print("   2. Generate solutions")
            print("   3. Apply first-K-ahead voting")
            print("   4. Red-flag invalid results")
            print("   5. Return best solution")

            # Show different MAKER modes
            print(f"\n📋 MAKER modes:")
            print("   - sequential: Step-by-step execution")
            print("   - parallel: Parallel execution")
            print("   - recursive: Recursive decomposition")
            print("   - hybrid: Combination of strategies")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"MAKER demo error: {e}", exc_info=True)

    async def run_hybrid_demo(self):
        """Demonstrate hybrid ROMA-MDAP-MAKER"""

        print("\n" + "=" * 80)
        print("DEMO 3: Hybrid ROMA-MDAP-MAKER Integration")
        print("=" * 80 + "\n")

        if not ROMA_MDAP_AVAILABLE:
            print("❌ ROMA-MDAP-MAKER not available. Skipping demo.")
            return

        try:
            # Configuration
            config = ROMAMDAPMakerConfig(
                roma_max_depth_solving=2,
                mdap_enabled=True,
                mdap_k_ahead=3,
                apply_maker_to_roma_atomic=True,
                enable_hierarchical_voting=True,
                provider="openai",
                model=self.model,
                api_key=self.api_key
            )

            print("✓ ROMA-MDAP-MAKER configured")
            print(f"  - ROMA max depth: {config.roma_max_depth_solving}")
            print(f"  - MDAP enabled: {config.mdap_enabled}")
            print(f"  - K-ahead: {config.mdap_k_ahead}")
            print(f"  - Hierarchical voting: {config.enable_hierarchical_voting}")

            theorem = "∀ a b c : Nat, (a + b) + c = a + (b + c)"

            print(f"\n✓ Theorem: {theorem}")

            print("\n📊 Hybrid execution flow:")
            print("   1. ROMA decomposes theorem into sub-goals")
            print("   2. MDAP generates proofs for each sub-goal")
            print("   3. MAKER applies error correction")
            print("   4. Hierarchical voting across levels")
            print("   5. Aggregate final proof")

            print("\n🌳 Example decomposition:")
            print("   Main theorem: (a + b) + c = a + (b + c)")
            print("   ├─ Sub-goal 1: Base case (c = 0)")
            print("   ├─ Sub-goal 2: Inductive step (succ c)")
            print("   └─ Sub-goal 3: Reassemble proof")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Hybrid demo error: {e}", exc_info=True)

    async def run_custom_agent_demo(self):
        """Demonstrate custom agent configuration"""

        print("\n" + "=" * 80)
        print("DEMO 4: Custom Agent Configuration")
        print("=" * 80 + "\n")

        if not MDAP_AVAILABLE:
            print("❌ MDAP not available. Skipping demo.")
            return

        try:
            # Red-flagging rules
            red_flag_rules = RedFlagRules(
                max_tokens=750,
                min_confidence=0.3,
                blocked_patterns=["sorry", "admit", "TODO"]
            )

            print("✓ Red-flagging rules configured")
            print(f"  - Max tokens: {red_flag_rules.max_tokens}")
            print(f"  - Min confidence: {red_flag_rules.min_confidence}")
            print(f"  - Blocked patterns: {red_flag_rules.blocked_patterns}")

            # Create flagger
            flagger = RedFlagger(red_flag_rules)

            # Test red-flagging
            test_responses = [
                ("Valid response", '{"proof": "theorem test : True := by trivial"}'),
                ("Blocked (sorry)", "theorem test : True := by sorry"),
                ("Low confidence", '{"proof": "...", "confidence": 0.1}')
            ]

            print("\n📊 Red-flagging tests:")
            for name, response in test_responses:
                is_flagged, reasons = flagger.is_flagged(response, {}, None)
                status = "🚩 FLAGGED" if is_flagged else "✓ PASS"
                print(f"   {status}: {name}")
                if reasons:
                    print(f"      Reasons: {reasons}")

            # Cache demo
            print("\n📦 Cache demonstration:")
            cache = MDAPCache(max_size=100, ttl_seconds=60)

            cache.set("key1", {"value": "proof1"})
            cache.set("key2", {"value": "proof2"})

            retrieved = cache.get("key1")
            print(f"   ✓ Set key1, retrieved: {retrieved}")

            cache_miss = cache.get("key3")
            print(f"   ✓ Get key3 (miss): {cache_miss}")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Custom agent demo error: {e}", exc_info=True)

    async def run_workflow_demo(self):
        """Demonstrate workflow integration"""

        print("\n" + "=" * 80)
        print("DEMO 5: Workflow Integration")
        print("=" * 80 + "\n")

        if not WORKFLOW_AVAILABLE:
            print("❌ Workflow structures not available. Skipping demo.")
            return

        try:
            # Create decomposition
            sub_problems = [
                SubProblem(
                    id="lemma_1",
                    title="Base case",
                    description="Prove: 0 + n = n",
                    estimated_effort=3
                ),
                SubProblem(
                    id="lemma_2",
                    title="Inductive step",
                    description="Prove: succ m + n = succ (m + n)",
                    estimated_effort=5
                ),
                SubProblem(
                    id="main",
                    title="Main theorem",
                    description="∀ m n : Nat, m + n = n + m",
                    dependencies=["lemma_1", "lemma_2"],
                    estimated_effort=8
                )
            ]

            print("✓ Decomposition created")
            print(f"  - Total sub-problems: {len(sub_problems)}")

            for sp in sub_problems:
                print(f"\n  📋 {sp.id}: {sp.title}")
                print(f"     Effort: {sp.estimated_effort}")
                print(f"     Dependencies: {sp.dependencies}")

            # Show solving order
            print("\n📊 Solving order (topological sort):")
            print("   1. lemma_1 (no dependencies)")
            print("   2. lemma_2 (no dependencies)")
            print("   3. main (depends on lemma_1, lemma_2)")

            print("\n⚙️ Workflow stages:")
            print("   Stage 3A: Initial MDAP proof generation")
            print("   Stage 3B: MDAP refinement (if needed)")
            print("   Stage 4: Solution reassembly")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Workflow demo error: {e}", exc_info=True)

    async def run_voting_demo(self):
        """Demonstrate voting strategies"""

        print("\n" + "=" * 80)
        print("DEMO 6: Voting Strategies")
        print("=" * 80 + "\n")

        try:
            # Simulate voting
            print("📊 Majority voting:")
            votes_majority = {
                "proof_a": 5,
                "proof_b": 3,
                "proof_c": 2
            }
            winner_majority = max(votes_majority, key=votes_majority.get)
            print(f"   Votes: {votes_majority}")
            print(f"   Winner: {winner_majority} (5 votes)")

            print("\n📊 First-K-ahead voting (K=3):")
            votes_sequence = ["proof_a", "proof_a", "proof_b", "proof_a"]
            k_ahead = 3
            from collections import Counter
            counts = Counter(votes_sequence)
            if counts["proof_a"] >= k_ahead:
                print(f"   Sequence: {votes_sequence}")
                print(f"   proof_a reached {counts['proof_a']} votes (≥ {k_ahead})")
                print(f"   STOP early, winner: proof_a")

            print("\n📊 Confidence-weighted voting:")
            candidates = [
                {"proof": "proof_a", "confidence": 0.9},
                {"proof": "proof_a", "confidence": 0.85},
                {"proof": "proof_b", "confidence": 0.7},
                {"proof": "proof_c", "confidence": 0.6}
            ]
            weighted_votes = {}
            for c in candidates:
                weighted_votes[c["proof"]] = weighted_votes.get(c["proof"], 0) + c["confidence"]
            winner_weighted = max(weighted_votes, key=weighted_votes.get)
            print(f"   Weighted votes: {weighted_votes}")
            print(f"   Winner: {winner_weighted} (1.75 total confidence)")

            print("\n✅ Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Voting demo error: {e}", exc_info=True)

    async def run_all_demos(self):
        """Run all demos"""

        print("\n" + "=" * 80)
        print("MDAP/MAKER COMPREHENSIVE DEMO")
        print("=" * 80)

        demos = [
            ("Basic MDAP", self.run_basic_mdap_demo),
            ("MAKER Workflow", self.run_maker_demo),
            ("Hybrid Integration", self.run_hybrid_demo),
            ("Custom Agents", self.run_custom_agent_demo),
            ("Workflow Integration", self.run_workflow_demo),
            ("Voting Strategies", self.run_voting_demo)
        ]

        for name, demo_func in demos:
            try:
                await demo_func()
                await asyncio.sleep(0.5)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Demo '{name}' failed: {e}")

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED")
        print("=" * 80 + "\n")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

async def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(
        description="MDAP/MAKER Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 Run all demos
  %(prog)s basic           Run basic MDAP demo
  %(prog)s maker           Run MAKER demo
  %(prog)s hybrid          Run hybrid demo
  %(prog)s custom          Run custom agent demo
  %(prog)s workflow        Run workflow demo
  %(prog)s voting          Run voting demo

Available demos:
  basic        Basic MDAP proof generation
  maker        MAKER workflow integration
  hybrid       Hybrid ROMA-MDAP-MAKER
  custom       Custom agent configuration
  workflow     Workflow integration
  voting       Voting strategies
        """
    )

    parser.add_argument(
        "demo",
        nargs="?",
        choices=["all", "basic", "maker", "hybrid", "custom", "workflow", "voting"],
        default="all",
        help="Demo to run (default: all)"
    )

    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: from OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    # Create runner
    runner = DemoRunner()

    # Override API key if provided
    if args.api_key:
        runner.api_key = args.api_key
    runner.model = args.model

    # Run selected demo
    if args.demo == "all":
        await runner.run_all_demos()
    elif args.demo == "basic":
        await runner.run_basic_mdap_demo()
    elif args.demo == "maker":
        await runner.run_maker_demo()
    elif args.demo == "hybrid":
        await runner.run_hybrid_demo()
    elif args.demo == "custom":
        await runner.run_custom_agent_demo()
    elif args.demo == "workflow":
        await runner.run_workflow_demo()
    elif args.demo == "voting":
        await runner.run_voting_demo()


if __name__ == "__main__":
    asyncio.run(main())
