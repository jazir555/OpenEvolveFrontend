#!/usr/bin/env python3
import asyncio
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')

from openevolve_agnostic_pes import evolve_code, AgnosticPESEngine

# Test code
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
    print("Starting evolution...", file=sys.stderr)
    result = await evolve_code(code, tests, max_iterations=5)
    
    print("="*60, file=sys.stderr)
    print("EVOLUTION RESULT", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Iterations: {result.iterations}", file=sys.stderr)
    print(f"Fixes Applied: {result.fixes_applied}", file=sys.stderr)
    print(f"Improvement: +{result.improvement:.1%}", file=sys.stderr)
    print(f"Final Score: {result.final_score:.1%} ({result.tests_passed}/{result.tests_total})", file=sys.stderr)
    print(file=sys.stderr)
    print("Evolved Code:", file=sys.stderr)
    print(result.evolved_code, file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
