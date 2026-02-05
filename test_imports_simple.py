"""
Simple test to verify circular import fix.
"""
import sys

def test_import(module_name):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"[PASS] {module_name}")
        return True
    except Exception as e:
        print(f"[FAIL] {module_name}: {type(e).__name__}: {e}")
        return False

print("Testing circular import fix...")
print("=" * 60)

# Test each module individually
results = []
results.append(("bubblelabs_integration", test_import("bubblelabs_integration")))

# The key test: can we import z3_api_server without circular import errors?
results.append(("z3_api_server", test_import("z3_api_server")))

print("=" * 60)
print("Summary:")
for name, passed in results:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")

if all(r[1] for r in results):
    print("\nCircular import fix verified!")
    sys.exit(0)
else:
    print("\nSome tests failed.")
    sys.exit(1)
