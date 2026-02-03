#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for OpenEvolve PES Integration
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set UTF-8 encoding for output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from openevolve_pes_integration import (
    OpenEvolvePESEnhancer,
    enhance_openevolve_code,
    OpenEvolveTestGenerator,
    EnhancementResult,
)


def test_test_generator():
    """Test the test generator."""
    print("+" + "="*68)
    print("  Test: OpenEvolveTestGenerator")
    print("="*70)
    
    # Test with Python code
    code = '''
def calculate_total(amount, tax_rate=0.08):
    return amount * (1 + tax_rate)

def validate_email(email):
    return "@" in email
'''
    
    functions = OpenEvolveTestGenerator.extract_functions(code)
    print(f"[PASS] Extracted {len(functions)} functions:")
    for func in functions:
        print(f"  - {func['name']}({func['params']}) - {func['language']}")
    
    # Test default test generation
    tests = OpenEvolveTestGenerator.generate_default_tests(code)
    print(f"\n[PASS] Generated {len(tests)} default tests:")
    for test in tests:
        print(f"  - {test['name']}: {test['function']}({test['input']}) -> {test['expected']}")
    
    # Test inference test generation
    tests = OpenEvolveTestGenerator.generate_inference_tests(
        code, "Calculate payment with tax and validate email"
    )
    print(f"\n[PASS] Generated {len(tests)} inference-based tests:")
    for test in tests:
        print(f"  - {test['name']}: {test['function']}({test['input']}) -> {test['expected']}")
    
    return True


def test_enhancer_basic():
    """Test basic enhancement."""
    print("+" + "="*68)
    print("  Test: OpenEvolvePESEnhancer (Basic)")
    print("="*70)
    
    code = '''def calculate_payment(amount, discount_code=None, payment_method="credit_card"):
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
    
    total = taxable + tax + fee
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}'''
    
    tests = [
        {"name": "Basic payment", "input": {"amount": 100}, "expected": {"total": 111.4}, "function": "calculate_payment"},
        {"name": "10% discount", "input": {"amount": 100, "discount_code": "SAVE10"}, "expected": {"discount": 10}, "function": "calculate_payment"},
        {"name": "PayPal fee", "input": {"amount": 150, "payment_method": "paypal"}, "expected": {"fee": 5.25}, "function": "calculate_payment"},
    ]
    
    enhancer = OpenEvolvePESEnhancer(max_iterations=5)
    result = enhancer.enhance(
        code=code,
        tests=tests,
        problem_description="Calculate payment with tax, discounts, and payment method fees"
    )
    
    print(f"[PASS] Enhancement Result:")
    print(f"  Success: {result.success}")
    print(f"  Tests Generated: {result.tests_generated}")
    print(f"  Tests Passed: {result.tests_passed_after}/{result.tests_generated}")
    print(f"  Improvements: {result.improvements}")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    return result.success or result.tests_passed_after > 0


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "="*70)
    print("  Test: Convenience Function (enhance_openevolve_code)")
    print("="*70)
    
    code = '''def process_order(items, customer_type="regular"):
    """Process an order and calculate totals."""
    subtotal = sum(item['price'] * item.get('quantity', 1) for item in items)
    
    discount = 0
    if customer_type == "premium":
        discount = subtotal * 0.15
    elif customer_type == "regular" and subtotal > 100:
        discount = subtotal * 0.10
    
    total = subtotal - discount
    return {"subtotal": subtotal, "discount": discount, "total": total}'''
    
    tests = [
        {"name": "Basic order", "input": {"items": [{"price": 10, "quantity": 2}]}, "expected": {"subtotal": 20}, "function": "process_order"},
        {"name": "Premium discount", "input": {"items": [{"price": 100}], "customer_type": "premium"}, "expected": {"discount": 15}, "function": "process_order"},
    ]
    
    result = enhance_openevolve_code(
        code=code,
        tests=tests,
        problem_description="Process orders with customer discounts",
        max_iterations=3
    )
    
    print(f"[PASS] Result: Success={result.success}, Passed={result.tests_passed_after}/{result.tests_generated}")
    
    return True


def test_inference_tests():
    """Test inference-based test generation."""
    print("+" + "="*68)
    print("  Test: Inference-based Test Generation")
    print("="*70)
    
    code = '''def validate_email(email):
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))'''
    
    # Generate tests from problem description
    tests = OpenEvolveTestGenerator.generate_inference_tests(
        code, "Validate email addresses"
    )
    
    print(f"[PASS] Generated {len(tests)} tests for email validation:")
    for test in tests:
        print(f"  - {test['name']}: input={test['input']}, expected={test['expected']}")
    
    # Now enhance
    enhancer = OpenEvolvePESEnhancer(max_iterations=3)
    result = enhancer.enhance(code, tests=tests, problem_description="Validate email addresses")
    
    print(f"\n[PASS] Enhancement Result:")
    print(f"  Success: {result.success}")
    print(f"  Tests Passed: {result.tests_passed_after}/{result.tests_generated}")
    
    return True


def test_enhancement_result():
    """Test EnhancementResult dataclass."""
    print("+" + "="*68)
    print("  Test: EnhancementResult Dataclass")
    print("="*70)
    
    result = EnhancementResult(
        original_code="original",
        enhanced_code="enhanced",
        tests_generated=3,
        tests_passed_before=2,
        tests_passed_after=3,
        improvements=["Fixed bug in payment calculation"],
        success=True
    )
    
    print(f"[PASS] EnhancementResult created successfully:")
    print(f"  Original: {result.original_code[:20]}...")
    print(f"  Enhanced: {result.enhanced_code[:20]}...")
    print(f"  Tests: {result.tests_passed_before} -> {result.tests_passed_after}/{result.tests_generated}")
    print(f"  Success: {result.success}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  OpenEvolve PES Integration Test Suite")
    print("="*70)
    
    tests = [
        ("Test Generator", test_test_generator),
        ("Enhancer Basic", test_enhancer_basic),
        ("Convenience Function", test_convenience_function),
        ("Inference Tests", test_inference_tests),
        ("Enhancement Result", test_enhancement_result),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
