#!/usr/bin/env python3
"""
Demo: OpenEvolve PES Integration - Content-Type Agnostic

This demo uses the actual openevolve_pes_integration module to evolve code
in multiple languages (Python, PHP, JavaScript).
"""

import asyncio
from openevolve_pes_integration import (
    evolve_code, quick_evolve, PESEvolutionEngine,
    PythonHandler, PHPHandler, JavaScriptHandler
)


def demo_python():
    """Demo with Python code."""
    print("\n" + "="*70)
    print("  PYTHON EVOLUTION DEMO")
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
    
    print("\nOriginal Code:")
    print(code)
    
    result = asyncio.run(evolve_code(code, tests, language="python", max_iterations=5))
    
    print(f"\nEvolution Result:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Fixes Applied: {result.fixes_applied}")
    print(f"  Improvement: +{result.improvement:.1%}")
    print(f"  Final Score: {result.final_score:.1%} ({result.tests_passed}/{result.tests_total})")
    
    print(f"\nEvolved Code:")
    print(result.evolved_code)
    
    return result


def demo_php():
    """Demo with PHP code."""
    print("\n" + "="*70)
    print("  PHP EVOLUTION DEMO")
    print("="*70)
    
    code = '''function calculate_payment($amount, $discount_code = null, $payment_method = "credit_card") {
    $subtotal = $amount;
    $discount = 0;
    
    // Apply discount
    if ($discount_code == "SAVE10") {
        $discount = $subtotal * 0.10;
    } else if ($discount_code == "SAVE20") {
        $discount = $subtotal * 0.20;
    }
    
    // Calculate tax
    $taxable = $subtotal - $discount;
    $tax = $taxable * 0.085;
    
    // Payment fee
    $fee = 0;
    if ($payment_method == "credit_card") {
        $fee = $subtotal * 0.029;
    } else if ($payment_method == "debit_card") {
        $fee = $subtotal * 0.015;
    }
    
    $total = $taxable + $tax + $fee;
    return [
        "subtotal" => $subtotal,
        "discount" => $discount,
        "tax" => $tax,
        "fee" => $fee,
        "total" => $total
    ];
}'''
    
    tests = [
        {"name": "Basic payment", "input": {"amount": 100}, "expected": {"total": 111.4}, "function": "calculate_payment"},
        {"name": "PayPal fee", "input": {"amount": 150, "payment_method": "paypal"}, "expected": {"fee": 5.25}, "function": "calculate_payment"},
    ]
    
    print("\nOriginal Code:")
    print(code)
    
    result = asyncio.run(evolve_code(code, tests, language="php", max_iterations=5))
    
    print(f"\nEvolution Result:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Fixes Applied: {result.fixes_applied}")
    print(f"  Improvement: +{result.improvement:.1%}")
    print(f"  Final Score: {result.final_score:.1%} ({result.tests_passed}/{result.tests_total})")
    
    print(f"\nEvolved Code:")
    print(result.evolved_code)
    
    return result


def demo_handler_info():
    """Show available content type handlers."""
    print("\n" + "="*70)
    print("  AVAILABLE CONTENT TYPE HANDLERS")
    print("="*70)
    
    handlers = [
        PythonHandler(),
        PHPHandler(),
        JavaScriptHandler(),
    ]
    
    for handler in handlers:
        print(f"\n  {handler.name}:")
        print(f"    Extension: {handler.extension}")
        print(f"    Functions: {len(handler.extract_functions('def test(a, b): pass'))}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  OpenEvolve PES Integration Demo")
    print("  Content-Type Agnostic Code Evolution")
    print("="*70)
    
    # Show available handlers
    demo_handler_info()
    
    # Run Python demo
    python_result = demo_python()
    
    # Run PHP demo
    php_result = demo_php()
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    print(f"\n  Python Evolution:")
    print(f"    Score: {python_result.final_score:.1%}")
    print(f"    Fixes: {python_result.fixes_applied}")
    
    print(f"\n  PHP Evolution:")
    print(f"    Score: {php_result.final_score:.1%}")
    print(f"    Fixes: {php_result.fixes_applied}")
    
    print("\n  The openevolve_pes_integration module is now ready to use!")
    print("  It supports multiple content types via the handler pattern.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
