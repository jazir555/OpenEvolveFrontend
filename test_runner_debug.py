#!/usr/bin/env python3
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')

from openevolve_agnostic_pes import UniversalTestRunner

# Test the test runner
code = '''def calculate_payment(amount, payment_method="credit_card"):
    """Calculate payment with tax, discounts, and fees."""
    subtotal = amount
    fee = 0
    if payment_method == "credit_card":
        fee = subtotal * 0.029
    elif payment_method == "debit_card":
        fee = subtotal * 0.015
    elif payment_method == "paypal":
        fee = subtotal * 0.035
    return {"fee": fee}'''

tests = [
    {"name": "Basic payment", "input": {"amount": 100}, "expected": {"fee": 2.9}, "function": "calculate_payment"},
    {"name": "PayPal fee", "input": {"amount": 150, "payment_method": "paypal"}, "expected": {"fee": 5.25}, "function": "calculate_payment"},
]

print("Generating test wrapper...", file=sys.stderr)
wrapper = UniversalTestRunner.generate_test_wrapper(code, tests, 'python')

print("Test wrapper generated.", file=sys.stderr)
print("Running tests...", file=sys.stderr)

passed, total, failing = UniversalTestRunner.execute(wrapper, 'python')

print(f"Results: {passed}/{total}", file=sys.stderr)
print(f"Failing: {failing}", file=sys.stderr)
