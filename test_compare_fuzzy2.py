#!/usr/bin/env python3
"""Test the fuzzy comparison function."""

def _compare_fuzzy(a, b, tolerance=0.01):
    """Compare two values with tolerance for floats - handles partial dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        # Check that all expected keys in b are present in a and match
        for key, expected_val in b.items():
            if key not in a:
                return False
            if not _compare_fuzzy(a[key], expected_val, tolerance):
                return False
        return True
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_compare_fuzzy(av, bv, tolerance) for av, bv in zip(a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    else:
        return a == b

# Test case 1: Basic payment - full result
result1 = {'subtotal': 100, 'discount': 0, 'tax': 8.5, 'fee': 2.9, 'total': 111.4}
expected1 = {'total': 111.4}
print("Test 1 (Basic payment):", _compare_fuzzy(result1, expected1))

# Test case 2: 10% discount - result with SAVE10 discount applied
result2 = {'subtotal': 100, 'discount': 10, 'tax': 7.65, 'fee': 2.9, 'total': 100.55}
expected2 = {'discount': 10}
print("Test 2 (10% discount):", _compare_fuzzy(result2, expected2))

# Test case 3: PayPal fee - result with paypal fee
result3 = {'subtotal': 150, 'discount': 0, 'tax': 12.75, 'fee': 5.25, 'total': 168}
expected3 = {'fee': 5.25}
print("Test 3 (PayPal fee):", _compare_fuzzy(result3, expected3))
