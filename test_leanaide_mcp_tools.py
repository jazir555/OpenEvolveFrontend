"""
Test script for LeanAide MCP Tools

This script demonstrates and tests the LeanAide MCP tools functionality.
Note: Most tests require a running LeanAide server.
"""

import sys
import json
from leanaide_mcp_tools import (
    list_mcp_tools,
    get_mcp_tool,
    get_leanaide_status,
    leanaide_translate_theorem,
    leanaide_translate_definition,
    leanaide_generate_proof,
    leanaide_verify_solution,
    leanaide_math_query,
    leanaide_generate_documentation,
    leanaide_elaborate_code,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(result):
    """Print formatted result."""
    print(json.dumps(result, indent=2, default=str))


def test_tool_registry():
    """Test 1: MCP Tool Registry."""
    print_section("Test 1: MCP Tool Registry")

    tools = list_mcp_tools()
    print(f"Total tools registered: {len(tools)}")
    print("\nAvailable tools:")
    for tool in sorted(tools):
        print(f"  - {tool}")

    # Test getting a specific tool
    translate_tool = get_mcp_tool("leanaide_translate_theorem")
    print(f"\nRetrieved tool: {translate_tool.__name__ if translate_tool else 'None'}")


def test_server_status():
    """Test 2: Server Status Check."""
    print_section("Test 2: LeanAide Server Status")

    status = get_leanaide_status()
    print_result(status)

    return status['available']


def test_translate_theorem():
    """Test 3: Translate Theorem."""
    print_section("Test 3: Translate Theorem")

    result = leanaide_translate_theorem(
        theorem_text="There are infinitely many prime numbers",
        theorem_name="infinitely_many_primes",
        timeout=30
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nGenerated Lean Code:")
        print(result['lean_code'])
        print(f"\nExecution time: {result['execution_time']:.2f}s")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_translate_definition():
    """Test 4: Translate Definition."""
    print_section("Test 4: Translate Definition")

    result = leanaide_translate_definition(
        definition_text="A natural number n is prime if it has exactly two positive divisors",
        timeout=30
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nGenerated Lean Code:")
        print(result['lean_code'])
        print(f"\nExecution time: {result['execution_time']:.2f}s")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_verify_solution():
    """Test 5: Verify Solution."""
    print_section("Test 5: Verify Solution")

    # Simple Lean code that should be valid
    code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [add_comm]
"""

    result = leanaide_verify_solution(code, timeout=30)

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nIs valid: {result['is_valid']}")
        print(f"Declarations: {result['declarations']}")
        print(f"Unproven goals: {result['unproven_count']}")
        if result['logs']:
            print(f"\nLogs:")
            for log in result['logs'][:5]:  # Show first 5 logs
                print(f"  {log}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_elaborate_code():
    """Test 6: Elaborate Code (with errors)."""
    print_section("Test 6: Elaborate Code (Error Checking)")

    # Code with intentional error
    code = """
theorem bad_theorem (n : Nat) : n = n + 1 := by
  rfl
"""

    result = leanaide_elaborate_code(code, timeout=30)

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nHas errors: {result['has_errors']}")
        print(f"Declarations: {result['declarations']}")
        print(f"Unsolved goals: {result['unsolved_goal_count']}")

        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")

        if result['warnings']:
            print(f"\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_math_query():
    """Test 7: Math Query."""
    print_section("Test 7: Math Query")

    result = leanaide_math_query(
        query="What is the fundamental theorem of algebra?",
        n=2,
        timeout=30
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nQuery: {result['query']}")
        print(f"Number of answers: {result['num_answers']}")
        print("\nAnswers:")
        for i, answer in enumerate(result['answers'], 1):
            print(f"\n{i}. {answer}")
        print(f"\nExecution time: {result['execution_time']:.2f}s")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_generate_documentation():
    """Test 8: Generate Documentation."""
    print_section("Test 8: Generate Documentation")

    result = leanaide_generate_documentation(
        name="infinitely_many_primes",
        code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
        doc_type="theorem",
        timeout=30
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nName: {result['name']}")
        print(f"Type: {result['doc_type']}")
        print(f"\nGenerated Documentation:")
        print(result['documentation'])
        print(f"\nExecution time: {result['execution_time']:.2f}s")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def test_generate_proof():
    """Test 9: Generate Proof."""
    print_section("Test 9: Generate Proof")

    result = leanaide_generate_proof(
        theorem_text="The square root of 2 is irrational",
        timeout=60
    )

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"\nTheorem: {result['theorem_text']}")
        print(f"\nProof Document:")
        print(result['proof_document'][:500] + "..." if len(result['proof_document']) > 500 else result['proof_document'])
        print(f"\nExecution time: {result['execution_time']:.2f}s")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    return result['success']


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("#  LeanAide MCP Tools Test Suite")
    print("#" * 70)

    results = {}

    # Test 1: Tool Registry (always works)
    test_tool_registry()
    results['tool_registry'] = True

    # Test 2: Server Status
    server_available = test_server_status()
    results['server_status'] = server_available

    if not server_available:
        print("\n" + "!" * 70)
        print("!  WARNING: LeanAide server is not available!")
        print("!  Most tests will be skipped.")
        print("!  Start the server with: cd LeanAide && python3 leanaide_server.py")
        print("!" * 70)
        return results

    # Run remaining tests
    print("\nRunning remaining tests (requires server)...")

    try:
        results['translate_theorem'] = test_translate_theorem()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['translate_theorem'] = False

    try:
        results['translate_definition'] = test_translate_definition()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['translate_definition'] = False

    try:
        results['verify_solution'] = test_verify_solution()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['verify_solution'] = False

    try:
        results['elaborate_code'] = test_elaborate_code()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['elaborate_code'] = False

    try:
        results['math_query'] = test_math_query()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['math_query'] = False

    try:
        results['generate_documentation'] = test_generate_documentation()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['generate_documentation'] = False

    try:
        results['generate_proof'] = test_generate_proof()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results['generate_proof'] = False

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test}")

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    if all(results.values()):
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed")
        sys.exit(1)
