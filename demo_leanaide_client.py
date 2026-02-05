"""
LeanAide Client Demo

Demonstrates the key features of the LeanAide async client.
Run this with a LeanAide server running on localhost:7654
"""

import asyncio
import json
from leanaide_client import LeanAideClient, LeanAideConfig


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, result):
    """Print a formatted result."""
    print(f"\n{name}:")
    print(f"  Success: {result.success}")
    print(f"  Response Time: {result.response_time:.2f}s")

    if result.success:
        if result.data:
            # Truncate long output
            data_str = json.dumps(result.data, indent=2)
            if len(data_str) > 300:
                data_str = data_str[:300] + "..."
            print(f"  Data: {data_str}")
        if result.logs:
            logs_preview = result.logs[:200] + "..." if len(result.logs) > 200 else result.logs
            print(f"  Logs: {logs_preview}")
    else:
        print(f"  Error: {result.error}")


async def demo_basic_translation():
    """Demonstrate basic theorem translation."""
    print_section("Demo 1: Basic Theorem Translation")

    async with LeanAideClient() as client:
        # Check server health
        is_healthy = await client.health_check()
        print(f"\nServer Health: {'[OK] Healthy' if is_healthy else '[FAIL] Unhealthy'}")

        if not is_healthy:
            print("\n[WARN] Server is not responding. Please start the LeanAide server:")
            print("  cd LeanAide && python3 leanaide_server.py")
            return

        # Translate a simple theorem
        result = await client.translate_thm(
            "There are infinitely many prime numbers"
        )
        print_result("Translation", result)


async def demo_detailed_translation():
    """Demonstrate detailed translation with naming."""
    print_section("Demo 2: Detailed Translation with Naming")

    async with LeanAideClient() as client:
        result = await client.translate_thm_detailed(
            "There are infinitely many prime numbers",
            theorem_name="infinitely_many_primes"
        )
        print_result("Detailed Translation", result)


async def demo_definition_translation():
    """Demonstrate definition translation."""
    print_section("Demo 3: Definition Translation")

    async with LeanAideClient() as client:
        result = await client.translate_def(
            "A number is cube-free if it is not divisible by the cube of any prime number"
        )
        print_result("Definition Translation", result)


async def demo_documentation_generation():
    """Demonstrate documentation generation."""
    print_section("Demo 4: Documentation Generation")

    async with LeanAideClient() as client:
        result = await client.theorem_doc(
            theorem_name="infinitely_many_primes",
            theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
        )
        print_result("Documentation Generation", result)


async def demo_math_query():
    """Demonstrate math query functionality."""
    print_section("Demo 5: Math Query")

    async with LeanAideClient() as client:
        result = await client.math_query(
            "What is the fundamental theorem of algebra?",
            n=2
        )
        print_result("Math Query", result)


async def demo_elaboration():
    """Demonstrate Lean code elaboration."""
    print_section("Demo 6: Lean Code Elaboration")

    async with LeanAideClient() as client:
        lean_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]
"""
        result = await client.elaborate(lean_code)
        print_result("Elaboration", result)


async def demo_batch_operations():
    """Demonstrate batch processing."""
    print_section("Demo 7: Batch Translation")

    async with LeanAideClient() as client:
        theorems = [
            "There are infinitely many prime numbers",
            "The square root of 2 is irrational",
            "Every natural number has a unique prime factorization"
        ]

        print(f"\nTranslating {len(theorems)} theorems in parallel...")
        results = await client.batch_translate_theorems(theorems)

        print(f"\nResults:")
        for i, result in enumerate(results, 1):
            status = "[OK]" if result.success else "[FAIL]"
            print(f"  {status} Theorem {i}: {result.response_time:.2f}s")
            if not result.success:
                print(f"      Error: {result.error}")


async def demo_parallel_different_tasks():
    """Demonstrate parallel execution of different tasks."""
    print_section("Demo 8: Parallel Mixed Tasks")

    async with LeanAideClient() as client:
        tasks = [
            {
                "task": "translate_thm",
                "theorem_text": "There are infinitely many primes"
            },
            {
                "task": "translate_def",
                "definition_text": "A prime number has exactly two divisors"
            },
            {
                "task": "math_query",
                "query": "What is a group?",
                "n": 1
            }
        ]

        print(f"\nExecuting {len(tasks)} different tasks in parallel...")
        results = await client.execute_parallel_tasks(tasks)

        print(f"\nResults:")
        task_names = ["Theorem Translation", "Definition Translation", "Math Query"]
        for i, (name, result) in enumerate(zip(task_names, results), 1):
            status = "[OK]" if result.success else "[FAIL]"
            print(f"  {status} {name}: {result.response_time:.2f}s")


async def demo_custom_config():
    """Demonstrate custom configuration."""
    print_section("Demo 9: Custom Configuration")

    config = LeanAideConfig(
        host="localhost",
        port=7654,
        timeout=300.0,  # 5 minutes
        max_retries=5,
        retry_delay=2.0,
        enable_logging=True
    )

    async with LeanAideClient(config=config) as client:
        print(f"\nConfiguration:")
        print(f"  Server: {config.base_url}")
        print(f"  Timeout: {config.timeout}s")
        print(f"  Max Retries: {config.max_retries}")
        print(f"  Retry Delay: {config.retry_delay}s")

        result = await client.translate_thm("Test theorem")
        print_result("Custom Config Test", result)


async def demo_error_handling():
    """Demonstrate error handling."""
    print_section("Demo 10: Error Handling")

    async with LeanAideClient() as client:
        # Try with invalid input
        result = await client.translate_thm("")
        print_result("Empty Input (should fail)", result)

        # Try with malformed data
        result2 = await client.elaborate("invalid lean code {")
        print_result("Invalid Lean Code (should fail)", result2)


async def run_all_demos():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print(" LeanAide Async Client - Feature Demonstration")
    print("=" * 60)
    print("\nThis demo showcases the key features of the LeanAide client.")
    print("Make sure the LeanAide server is running on localhost:7654")
    print("\nStart server with: cd LeanAide && python3 leanaide_server.py")

    demos = [
        ("Basic Translation", demo_basic_translation),
        ("Detailed Translation", demo_detailed_translation),
        ("Definition Translation", demo_definition_translation),
        ("Documentation Generation", demo_documentation_generation),
        ("Math Query", demo_math_query),
        ("Elaboration", demo_elaboration),
        ("Batch Operations", demo_batch_operations),
        ("Parallel Mixed Tasks", demo_parallel_different_tasks),
        ("Custom Configuration", demo_custom_config),
        ("Error Handling", demo_error_handling),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            await demo_func()
            await asyncio.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            print(f"\n[FAIL] Demo failed: {e}")

    print_section("Demo Complete")
    print("\nAll demonstrations finished!")
    print("\nFor more information, see LEANAIDE_CLIENT_README.md")
    print("To run tests: pytest test_leanaide_client.py -v")


async def run_interactive_demo():
    """Run an interactive demo."""
    print("\n" + "=" * 60)
    print(" LeanAide Client - Interactive Demo")
    print("=" * 60)

    async with LeanAideClient() as client:
        # Check health
        is_healthy = await client.health_check()
        if not is_healthy:
            print("\n[WARN] Server is not responding. Please start the LeanAide server.")
            return

        print("\n[OK] Server is healthy!")

        while True:
            print("\n" + "-" * 60)
            print("Choose an option:")
            print("  1. Translate a theorem")
            print("  2. Translate a definition")
            print("  3. Generate documentation")
            print("  4. Ask a math question")
            print("  5. Elaborate Lean code")
            print("  6. Batch translate theorems")
            print("  0. Exit")

            choice = input("\nYour choice: ").strip()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                theorem = input("Enter theorem: ")
                result = await client.translate_thm(theorem)
                print_result("Translation Result", result)
            elif choice == "2":
                definition = input("Enter definition: ")
                result = await client.translate_def(definition)
                print_result("Definition Result", result)
            elif choice == "3":
                name = input("Theorem name: ")
                statement = input("Theorem statement: ")
                result = await client.theorem_doc(name, statement)
                print_result("Documentation Result", result)
            elif choice == "4":
                question = input("Your question: ")
                result = await client.math_query(question, n=2)
                print_result("Math Query Result", result)
            elif choice == "5":
                print("Enter Lean code (empty line to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                code = "\n".join(lines)
                result = await client.elaborate(code)
                print_result("Elaboration Result", result)
            elif choice == "6":
                print("Enter theorems (one per line, empty line to finish):")
                theorems = []
                while True:
                    theorem = input(f"Theorem {len(theorems) + 1}: ")
                    if not theorem:
                        break
                    theorems.append(theorem)
                results = await client.batch_translate_theorems(theorems)
                for i, result in enumerate(results, 1):
                    status = "[OK]" if result.success else "[FAIL]"
                    print(f"{status} Theorem {i}: {result.response_time:.2f}s")
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(run_interactive_demo())
    else:
        asyncio.run(run_all_demos())
