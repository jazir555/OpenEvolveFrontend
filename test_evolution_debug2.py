#!/usr/bin/env python3
"""Debug the evolution loop."""
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
    analysis = UniversalCodeAnalyzer.analyze(code)
    
    current_code = code
    best_code = code
    best_score = 0.0
    
    print(f"Initial current_code == code: {current_code == code}", file=sys.stderr)
    print(f"Initial best_code == code: {best_code == code}", file=sys.stderr)
    
    # Iteration 1
    print(f"\\n=== ITERATION 1 ===", file=sys.stderr)
    wrapper = UniversalTestRunner.generate_test_wrapper(current_code, tests, language)
    passed, total, failing = UniversalTestRunner.execute(wrapper, language)
    score = passed / total if total > 0 else 0.0
    print(f"Tests on current_code: {passed}/{total}, failing: {failing}", file=sys.stderr)
    print(f"Score: {score:.1%}", file=sys.stderr)
    
    if score > best_score:
        best_score = score
        best_code = current_code
        print(f"Updated best_code (score > best_score)", file=sys.stderr)
    else:
        print(f"Did NOT update best_code (score <= best_score)", file=sys.stderr)
    
    # Apply fix
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
    
    new_code = UniversalFixGenerator.generate_fix(current_code, analysis, fix_request)
    print(f"\\nFix generated, new_code == current_code: {new_code == current_code}", file=sys.stderr)
    
    current_code = new_code
    print(f"After assignment, current_code == new_code: {current_code == new_code}", file=sys.stderr)
    print(f"After assignment, current_code == code: {current_code == code}", file=sys.stderr)
    print(f"After assignment, current_code == best_code: {current_code == best_code}", file=sys.stderr)
    
    # Test current_code after fix
    wrapper2 = UniversalTestRunner.generate_test_wrapper(current_code, tests, language)
    passed2, total2, failing2 = UniversalTestRunner.execute(wrapper2, language)
    print(f"\\nTests on current_code AFTER fix: {passed2}/{total2}, failing: {failing2}", file=sys.stderr)
    
    print(f"\\n=== FINAL STATE ===", file=sys.stderr)
    print(f"best_code has paypal: {'paypal' in best_code}", file=sys.stderr)
    print(f"current_code has paypal: {'paypal' in current_code}", file=sys.stderr)
    
    print(f"\\n=== BEST CODE ===", file=sys.stderr)
    print(best_code, file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
