#!/usr/bin/env python3
import asyncio
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')

from openevolve_agnostic_pes import (
    AgnosticPESEngine, UniversalTestRunner, UniversalFixGenerator, 
    UniversalCodeAnalyzer, LanguageDetector
)

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

async def main():
    language = LanguageDetector.detect(code)
    print(f"Detected language: {language}", file=sys.stderr)
    
    analysis = UniversalCodeAnalyzer.analyze(code)
    print(f"Analysis: {analysis}", file=sys.stderr)
    
    # Run tests on original
    wrapper = UniversalTestRunner.generate_test_wrapper(code, tests, language)
    passed, total, failing = UniversalTestRunner.execute(wrapper, language)
    print(f"Original: {passed}/{total}, failing: {failing}", file=sys.stderr)
    
    # Try to fix
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
    
    new_code = UniversalFixGenerator.generate_fix(code, analysis, fix_request)
    print(f"\\nOriginal code:", file=sys.stderr)
    print(code, file=sys.stderr)
    print(f"\\nFixed code:", file=sys.stderr)
    print(new_code, file=sys.stderr)
    print(f"\\nCodes are different: {code != new_code}", file=sys.stderr)
    
    # Test the fixed code
    wrapper2 = UniversalTestRunner.generate_test_wrapper(new_code, tests, language)
    passed2, total2, failing2 = UniversalTestRunner.execute(wrapper2, language)
    print(f"\\nFixed: {passed2}/{total2}, failing: {failing2}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
