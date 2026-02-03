#!/usr/bin/env python3
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')

from openevolve_agnostic_pes import UniversalFixGenerator, UniversalCodeAnalyzer

code = '''def calculate_payment(amount, payment_method="credit_card"):
    """Calculate payment with tax, discounts, and fees."""
    subtotal = amount
    fee = 0
    if payment_method == "credit_card":
        fee = subtotal * 0.029
    elif payment_method == "debit_card":
        fee = subtotal * 0.015
    return fee'''

print("Original code:", file=sys.stderr)
print(code, file=sys.stderr)
print(file=sys.stderr)

analysis = UniversalCodeAnalyzer.analyze(code)
print("Analysis:", file=sys.stderr)
print(analysis, file=sys.stderr)
print(file=sys.stderr)

# Check what patterns we're looking for
print("Looking for pattern...", file=sys.stderr)
import re

# Python pattern
pattern = r'elif payment_method == "debit_card":\n        fee = subtotal \* 0\.015'
print(f"Pattern: {pattern}", file=sys.stderr)

match = re.search(pattern, code)
print(f"Match found: {match}", file=sys.stderr)

if match:
    print(f"Match group: {repr(match.group())}", file=sys.stderr)

# Try the fix
fix_request = {
    'issue': 'Missing branch for payment_method: paypal',
    'strategy': 'missing_branch',
    'fix_type': 'add_branch',
    'context': {
        'branch_type': 'payment_method',
        'value': 'paypal',
        'expected_fee': 0.035
    }
}

print(file=sys.stderr)
print("Generating fix...", file=sys.stderr)
new_code = UniversalFixGenerator.generate_fix(code, analysis, fix_request)

print("New code:", file=sys.stderr)
print(new_code, file=sys.stderr)
