"""
Test Suite for ACE MCP Tools Security Fixes

This test suite verifies all security fixes applied to ace_mcp_tools.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all security utilities can be imported"""
    print("Testing imports...")
    try:
        from ace_security_utils import (
            validate_and_resolve_path,
            validate_file_path_safe,
            validate_numeric_range,
            validate_list_size,
            validate_string_length,
            validate_model_name,
            create_safe_error,
            sanitize_for_logging,
            get_global_lock,
            DEFAULT_SKILLBOOK_DIR,
        )
        print("  [PASS] All security utilities imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_module_imports():
    """Test that ace_mcp_tools module can be imported"""
    print("\nTesting ace_mcp_tools module import...")
    try:
        # This will fail if ACE is not installed, but should not fail due to security imports
        import ace_mcp_tools
        print(f"  [PASS] Module imported successfully")
        print(f"  [PASS] ACE Available: {ace_mcp_tools.ACE_AVAILABLE}")
        return True
    except Exception as e:
        print(f"  [FAIL] Module import failed: {e}")
        return False


def test_validation_functions():
    """Test validation functions work correctly"""
    print("\nTesting validation functions...")
    from ace_security_utils import (
        validate_model_name,
        validate_string_length,
        validate_numeric_range,
        validate_file_path_safe,
        create_safe_error,
    )

    tests_passed = 0
    tests_total = 0

    # Test validate_model_name
    tests_total += 1
    try:
        result = validate_model_name("gpt-4o-mini")
        print(f"  [PASS] Valid model name accepted: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Valid model name rejected: {e}")

    # Test validate_model_name with injection attempt
    tests_total += 1
    try:
        validate_model_name("gpt-4o; rm -rf /")
        print(f"  [FAIL] Command injection NOT prevented!")
    except ValueError:
        print(f"  [PASS] Command injection prevented")
        tests_passed += 1

    # Test validate_string_length
    tests_total += 1
    try:
        result = validate_string_length("test", "test_param", max_length=100)
        print(f"  [PASS] Valid string length accepted")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Valid string length rejected: {e}")

    # Test validate_string_length with too long string
    tests_total += 1
    try:
        validate_string_length("x" * 1000, "test_param", max_length=100)
        print(f"  [FAIL] String length limit NOT enforced!")
    except ValueError:
        print(f"  [PASS] String length limit enforced")
        tests_passed += 1

    # Test validate_numeric_range
    tests_total += 1
    try:
        result = validate_numeric_range(0.5, "test", min_val=0.0, max_val=1.0)
        print(f"  [PASS] Valid numeric range accepted")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Valid numeric range rejected: {e}")

    # Test validate_numeric_range with NaN
    tests_total += 1
    try:
        import math
        validate_numeric_range(float('nan'), "test", min_val=0.0, max_val=1.0, allow_nan=False)
        print(f"  [FAIL] NaN NOT prevented!")
    except ValueError:
        print(f"  [PASS] NaN prevented")
        tests_passed += 1

    # Test validate_file_path_safe
    tests_total += 1
    try:
        result = validate_file_path_safe("test.json", base_dir=".")
        print(f"  [PASS] Valid file path accepted")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Valid file path rejected: {e}")

    # Test validate_file_path_safe with path traversal
    tests_total += 1
    try:
        validate_file_path_safe("../../etc/passwd", base_dir=".")
        print(f"  [FAIL] Path traversal NOT prevented!")
    except ValueError:
        print(f"  [PASS] Path traversal prevented")
        tests_passed += 1

    # Test create_safe_error
    tests_total += 1
    try:
        result = create_safe_error("Test error", Exception("Internal details"))
        if result["success"] == False and "error" in result:
            print(f"  [PASS] Safe error created")
            tests_passed += 1
        else:
            print(f"  [FAIL] Safe error format incorrect")
    except Exception as e:
        print(f"  [FAIL] Safe error creation failed: {e}")

    print(f"\n  Validation tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_mcp_tool_signatures():
    """Test that MCP tools have correct signatures after fixes"""
    print("\nTesting MCP tool signatures...")
    import ace_mcp_tools

    # Check that tools are registered
    tools = ace_mcp_tools.get_registered_tools()
    print(f"  [PASS] {len(tools)} MCP tools registered")

    # Check for required tools
    required_tools = [
        "initialize_ace_agent",
        "execute_task_with_ace",
        "learn_from_samples_with_ace",
        "learn_from_execution_with_ace",
        "manage_ace_skillbook",
        "get_ace_status",
        "inject_ace_skills_into_context",
    ]

    all_present = True
    for tool_name in required_tools:
        if tool_name in tools:
            print(f"  [PASS] Tool '{tool_name}' registered")
        else:
            print(f"  [FAIL] Tool '{tool_name}' NOT registered")
            all_present = False

    return all_present


def test_thread_safety():
    """Test thread safety utilities"""
    print("\nTesting thread safety...")
    from ace_security_utils import get_global_lock, synchronized
    import threading

    # Test get_global_lock
    lock1 = get_global_lock("test_lock")
    lock2 = get_global_lock("test_lock")
    if lock1 is lock2:
        print(f"  [PASS] Global lock returns same instance")
    else:
        print(f"  [FAIL] Global lock NOT returning same instance")
        return False

    # Test synchronized decorator
    @synchronized("test_sync")
    def test_function(x):
        return x * 2

    result = test_function(5)
    if result == 10:
        print(f"  [PASS] Synchronized decorator works")
    else:
        print(f"  [FAIL] Synchronized decorator NOT working")
        return False

    return True


def test_sanitize_logging():
    """Test log sanitization"""
    print("\nTesting log sanitization...")
    from ace_security_utils import sanitize_for_logging

    # Test string sanitization
    result = sanitize_for_logging("test string")
    if result == "test string":
        print(f"  [PASS] String sanitization works")
    else:
        print(f"  [FAIL] String sanitization NOT working")
        return False

    # Test truncation
    long_string = "x" * 1000
    result = sanitize_for_logging(long_string, max_length=100)
    if len(result) < 1000 and "truncated" in result:
        print(f"  [PASS] Long string truncation works")
    else:
        print(f"  [FAIL] Long string truncation NOT working")
        return False

    # Test dictionary sanitization (should redact sensitive keys)
    sensitive_dict = {
        "username": "john",
        "password": "secret123",
        "api_key": "key123"
    }
    result = sanitize_for_logging(sensitive_dict)
    if "secret123" not in result and "key123" not in result and "***REDACTED***" in result:
        print(f"  [PASS] Sensitive data redaction works")
    else:
        print(f"  [FAIL] Sensitive data redaction NOT working")
        return False

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("ACE MCP Tools Security Fixes Test Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Module Imports", test_module_imports()))
    results.append(("Validation Functions", test_validation_functions()))
    results.append(("MCP Tool Signatures", test_mcp_tool_signatures()))
    results.append(("Thread Safety", test_thread_safety()))
    results.append(("Log Sanitization", test_sanitize_logging()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n[SUCCESS] All security fixes verified successfully!")
        return 0
    else:
        print(f"\n[WARNING]  {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
