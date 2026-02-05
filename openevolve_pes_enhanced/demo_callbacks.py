"""Demo of the Evolution Callback System.

This file demonstrates how to use the callback system to:
1. Monitor evolution iterations in real-time
2. Enforce budget constraints and stop when exceeded
3. Detect convergence and stop early
4. Log progress for debugging

Usage:
    python -m openevolve_pes_enhanced.demo_callbacks
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example code to evolve
EXAMPLE_CODE = '''
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
    
    total = taxable + tax + fee
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}
'''

# Test cases
TESTS = [
    {
        "name": "Basic payment",
        "input": {"amount": 100},
        "expected": {"total": 111.4},
        "function": "calculate_payment"
    },
    {
        "name": "10% discount",
        "input": {"amount": 100, "discount_code": "SAVE10"},
        "expected": {"discount": 10},
        "function": "calculate_payment"
    },
    {
        "name": "PayPal fee",
        "input": {"amount": 150, "payment_method": "paypal"},
        "expected": {"fee": 5.25},
        "function": "calculate_payment"
    },
]


async def demo_basic_callbacks():
    """Demo basic callback usage with budget and monitoring."""
    print("\n" + "="*70)
    print("Demo 1: Basic Callbacks - Budget and Monitoring")
    print("="*70)
    
    from openevolve_pes_enhanced import (
        MonitoredAgnosticPES,
        BudgetAwareCallback,
        MonitoringCallback,
        LoggingCallback,
    )
    
    # Create callbacks
    callbacks = [
        # Stop if cost exceeds $1.00
        BudgetAwareCallback(max_cost_usd=1.0, name="BudgetGuard"),
        
        # Stop if no improvement for 2 iterations
        MonitoringCallback(patience=2, min_improvement=0.01, name="ConvergenceWatch"),
        
        # Log progress
        LoggingCallback(log_every_n_iterations=1),
    ]
    
    # Create monitored engine
    engine = MonitoredAgnosticPES(
        max_iterations=5,
        callbacks=callbacks
    )
    
    # Run evolution
    result = await engine.evolve(
        code=EXAMPLE_CODE,
        tests=TESTS,
        problem_type="payment"
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Iterations run: {result.actual_iterations}/{result.max_iterations}")
    print(f"  Stopped early: {result.stopped_early}")
    if result.stopped_early:
        print(f"  Stop reason: {result.stop_reason}")
    print(f"  Final score: {result.final_score:.1%}")
    print(f"  Tests passed: {result.tests_passed}/{result.tests_total}")
    print(f"  Fixes applied: {len(result.fixes_applied)}")
    for fix in result.fixes_applied:
        print(f"    - {fix}")
    
    # Print metrics history
    print(f"\nIteration History:")
    for metrics in result.metrics_history:
        print(f"  Iteration {metrics.iteration + 1}: "
              f"fitness={metrics.best_fitness:.2f}, "
              f"tests={metrics.tests_passed}/{metrics.tests_total}, "
              f"fixes={len(metrics.fixes_applied_this_iteration)}")


async def demo_budget_stop():
    """Demo stopping evolution when budget is exceeded."""
    print("\n" + "="*70)
    print("Demo 2: Budget Enforcement - Very Low Budget")
    print("="*70)
    
    from openevolve_pes_enhanced import (
        MonitoredAgnosticPES,
        BudgetAwareCallback,
    )
    
    # Create callback with very low budget to trigger early stop
    budget_callback = BudgetAwareCallback(
        max_cost_usd=0.001,  # Very low budget
        name="StrictBudget"
    )
    
    # Manually simulate cost accumulation
    budget_callback.total_cost = 0.0005  # Start with some cost used
    
    engine = MonitoredAgnosticPES(
        max_iterations=5,
        callbacks=[budget_callback]
    )
    
    result = await engine.evolve(
        code=EXAMPLE_CODE,
        tests=TESTS,
        problem_type="payment"
    )
    
    print(f"\nResults:")
    print(f"  Iterations run: {result.actual_iterations}")
    print(f"  Stopped early: {result.stopped_early}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Budget status: ${budget_callback.total_cost:.4f} / ${budget_callback.max_cost_usd:.4f}")


async def demo_custom_callback():
    """Demo creating a custom callback."""
    print("\n" + "="*70)
    print("Demo 3: Custom Callback - Progress Reporter")
    print("="*70)
    
    from openevolve_pes_enhanced import (
        MonitoredAgnosticPES,
        EvolutionCallback,
        EvolutionContext,
        IterationMetrics,
    )
    from typing import Dict, Tuple
    
    class ProgressReporter(EvolutionCallback):
        """Custom callback that reports progress to user."""
        
        def __init__(self):
            super().__init__("ProgressReporter")
            self.milestones = [0.25, 0.5, 0.75, 0.9]
            self.reported = set()
        
        async def on_iteration_start(self, iteration: int, context: EvolutionContext):
            if iteration == 0:
                print(f"\n🚀 Evolution started (max {context.max_iterations} iterations)")
        
        async def on_iteration_end(self, iteration: int, metrics: IterationMetrics, context: EvolutionContext):
            # Calculate progress
            progress = (iteration + 1) / context.max_iterations
            
            # Report milestones
            for milestone in self.milestones:
                if progress >= milestone and milestone not in self.reported:
                    self.reported.add(milestone)
                    print(f"  📊 Progress: {milestone*100:.0f}% complete "
                          f"(fitness={metrics.best_fitness:.1%})")
            
            # Report individual fixes
            for fix in metrics.fixes_applied_this_iteration:
                print(f"  🔧 Applied fix: {fix}")
        
        async def on_evolution_end(self, context: EvolutionContext, 
                                   final_metrics: IterationMetrics, result):
            if final_metrics:
                print(f"\n[OK] Evolution complete: "
                      f"fitness={final_metrics.best_fitness:.1%}, "
                      f"iterations={final_metrics.iteration + 1}")
            else:
                print(f"\n[OK] Evolution complete")
    
    # Create engine with custom callback
    engine = MonitoredAgnosticPES(
        max_iterations=5,
        callbacks=[ProgressReporter()]
    )
    
    result = await engine.evolve(
        code=EXAMPLE_CODE,
        tests=TESTS,
        problem_type="payment"
    )
    
    print(f"\nFinal Results:")
    print(f"  Tests passed: {result.tests_passed}/{result.tests_total}")


async def demo_composite_callbacks():
    """Demo using multiple callbacks together."""
    print("\n" + "="*70)
    print("Demo 4: Composite Callbacks - Multiple Controls")
    print("="*70)
    
    from openevolve_pes_enhanced import (
        MonitoredAgnosticPES,
        create_standard_callbacks,
    )
    
    # Use the factory function to create standard callbacks
    composite = create_standard_callbacks(
        max_cost_usd=10.0,
        patience=2,
        enable_logging=False  # We'll use our own logging
    )
    
    print("Using standard callbacks:")
    for callback in composite.callbacks:
        print(f"  - {callback.name}")
    
    engine = MonitoredAgnosticPES(
        max_iterations=5,
        callbacks=[composite]
    )
    
    result = await engine.evolve(
        code=EXAMPLE_CODE,
        tests=TESTS,
        problem_type="payment"
    )
    
    print(f"\nResults:")
    print(f"  Stopped early: {result.stopped_early}")
    if result.stopped_early:
        print(f"  Reason: {result.stop_reason}")
    print(f"  Final fitness: {result.final_score:.1%}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("OpenEvolve PES Enhanced - Evolution Callback System Demo")
    print("="*70)
    
    await demo_basic_callbacks()
    await demo_budget_stop()
    await demo_custom_callback()
    await demo_composite_callbacks()
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
