#!/usr/bin/env python3
"""Debug test wrapper generation."""

import json
from openevolve_pes_integration import PythonHandler

handler = PythonHandler()

code = '''def calculate_payment(amount, discount_code=None, payment_method="credit_card"):
    """Calculate payment with tax, discounts, and fees."""
    subtotal = amount
    discount = 0
    
    if discount_code == "SAVE10":
        discount = subtotal * 0.10
    elif discount_code == "SAVE20":
        discount = subtotal * 0.20
    
    taxable = subtotal - discount
    tax = taxable * 0.085
    
    fee = 0
    if payment_method == "credit_card":
        fee = subtotal * 0.029
    elif payment_method == "debit_card":
        fee = subtotal * 0.015
    
    total = taxable + tax + fee
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}'''

tests = [
    {"name": "Basic payment", "input": {"amount": 100}, "expected": {"total": 111.4}, "function": "calculate_payment"},
    {"name": "PayPal fee", "input": {"amount": 150, "payment_method": "paypal"}, "expected": {"fee": 5.25}, "function": "calculate_payment"},
]

test_wrapper = handler.generate_test_wrapper(code, tests)

print("="*70)
print("Generated Test Wrapper:")
print("="*70)
print(test_wrapper)
print("="*70)

# Run the tests
print("\nRunning tests...")
passed, total, failing = handler.execute_tests(test_wrapper)
print(f"Passed: {passed}/{total}")
print(f"Failing: {failing}")
