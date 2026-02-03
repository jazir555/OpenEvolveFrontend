#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated test wrapper

import json

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

def calculate_payment(amount, discount_code=None, payment_method="credit_card"):
    """Calculate payment with tax, discounts, and fees."""
    subtotal = amount
    discount = 0
    
    # Apply discount
    if discount_code == "SAVE10":
        discount = subtotal * 0.10
    elif discount_code == "SAVE20":
        discount = subtotal * 0.20
    
    # Calculate tax
    taxable = subtotal - discount
    tax = taxable * 0.085
    
    # Payment fee
    fee = 0
    if payment_method == "credit_card":
        fee = subtotal * 0.029
    elif payment_method == "debit_card":
        fee = subtotal * 0.015
    elif payment_method == "paypal":
        fee = subtotal * 0.035
    
    total = taxable + tax + fee
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}

def test_0():
    """Test: Basic payment"""
    input_data = {'amount': 100}
    expected = {'total': 111.4}
    try:
        result = calculate_payment(**input_data)
        # Compare with fuzzy tolerance for floats
        passed = _compare_fuzzy(result, expected)
        return {"test": "Basic payment", "passed": passed}
    except Exception as e:
        return {"test": "Basic payment", "passed": False, "error": str(e)}

def test_1():
    """Test: 10% discount"""
    input_data = {'amount': 100, 'discount_code': 'SAVE10'}
    expected = {'discount': 10}
    try:
        result = calculate_payment(**input_data)
        # Compare with fuzzy tolerance for floats
        passed = _compare_fuzzy(result, expected)
        return {"test": "10% discount", "passed": passed}
    except Exception as e:
        return {"test": "10% discount", "passed": False, "error": str(e)}

def test_2():
    """Test: PayPal fee"""
    input_data = {'amount': 150, 'payment_method': 'paypal'}
    expected = {'fee': 5.25}
    try:
        result = calculate_payment(**input_data)
        # Compare with fuzzy tolerance for floats
        passed = _compare_fuzzy(result, expected)
        return {"test": "PayPal fee", "passed": passed}
    except Exception as e:
        return {"test": "PayPal fee", "passed": False, "error": str(e)}


if __name__ == "__main__":
    import sys
    tests = {"test_0": test_0, "test_1": test_1, "test_2": test_2}
    
    results = []
    for test_name, test_func in tests.items():
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            results.append({"test": test_name, "passed": False, "error": str(e)})
    
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    
    print(f"TESTS_PASSED:{passed}")
    print(f"TESTS_TOTAL:{total}")
    
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[{status}] {r['test']}")
    
    sys.exit(0 if passed == total else 1)
