"""
Example: Finding and fixing a bug using Cognitive Hydraulics.

This example demonstrates the complete Cognitive Hydraulics decision cycle:
1. Soar Phase: Read file -> Execute code -> Detect IndexError
2. Impasse Detection: Multiple valid fix options create Tie impasse
3. ACT-R Fallback: LLM evaluates options using utility equation (U = P×G - C)
4. Fix Application: Apply selected fix
5. Verification: Re-run code to confirm fix works
"""

import asyncio
from pathlib import Path
from cognitive_hydraulics.engine import CognitiveAgent
from cognitive_hydraulics.core.state import EditorState, Goal
from cognitive_hydraulics.safety import SafetyConfig


async def main():
    print("=" * 70)
    print("🐛 COGNITIVE HYDRAULICS - BUG FIX EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates the full Cognitive Hydraulics flow:")
    print("  Turn 1: Soar Phase - Read file -> Execute code -> Detect IndexError")
    print("  Turn 2: Tie Impasse - Multiple fix options")
    print("  Turn 2: ACT-R Fallback - Evaluating options with utility equation")
    print("  Turn 3: Applying fix - Selected fix is applied")
    print("  Turn 3: Verification - Code runs successfully")
    print("=" * 70)

    # Get the example directory
    example_dir = Path(__file__).parent

    # Create agent (dry-run=False to allow actual execution and fixes)
    print("\n📦 Creating cognitive agent...")
    from cognitive_hydraulics.config import load_config

    # Load application config
    app_config = load_config()

    safety_config = SafetyConfig(
        dry_run=False,  # Allow actual execution and file modifications
        require_approval_for_destructive=False,  # Auto-approve for demo
    )

    agent = CognitiveAgent(
        safety_config=safety_config,
        enable_learning=True,  # Enable learning for better suggestions
        max_cycles=50,  # Override config for complex analysis
        config=app_config,  # Use loaded config
    )
    print("[OK] Agent created")

    # Create initial state (file will be opened by rules)
    print("\n📄 Creating initial state...")
    initial_state = EditorState(
        working_directory=str(example_dir),
        open_files={},  # Start with no files open - let rules handle opening
    )
    print(f"[OK] State created (file will be opened by agent)")

    # Define goal to fix the bug
    print("\n🎯 Setting goal...")
    goal = Goal(
        description=(
            "Fix the bug in sort.py so that it runs without errors and sorts the list correctly."
        )
    )
    print(f"[OK] Goal: {goal.description}")

    # Run agent
    print("\n🚀 Running cognitive agent to fix the bug...")
    print("-" * 70)

    try:
        # Verbosity levels: 0=silent, 1=basic, 2=thinking (default), 3=debug
        success, final_state = await agent.solve(
            goal=goal,
            initial_state=initial_state,
            verbose=2,  # Thinking mode - shows reasoning
        )

        print("-" * 70)

        if success:
            print("\n[OK] SUCCESS: Bug fixed and verified!")

            # Show the fixed file
            if final_state.open_files.get("sort.py"):
                print("\n📝 Fixed file content:")
                print(final_state.open_files["sort.py"].content)

            # Check if goal was achieved
            if goal.status == "success":
                print("\n🎯 Goal achieved: Code runs without errors and sorts correctly!")
            else:
                print(f"\n[WARN]  Goal status: {goal.status}")

            if final_state.last_output:
                print("\n💡 Final output:")
                print(final_state.last_output)
        else:
            print("\n[WARN]  Fix incomplete (max cycles reached or impasse)")

        # Show statistics
        print(f"\n📊 Statistics:")
        stats = agent.get_statistics()
        for key, value in stats.items():
            print(f"   - {key}: {value}")

        if agent.memory:
            try:
                chunk_stats = agent.memory.get_stats()
                print(f"   - Chunks learned: {chunk_stats.get('total_chunks', 0)}")
                print(f"   - Contexts stored: {chunk_stats.get('total_contexts', 0)}")
            except Exception as e:
                print(f"   - Memory stats unavailable: {e}")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "=" * 70)
    print("[OK] Example complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
