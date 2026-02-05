"""Demo showing how to use the PES Enhanced layer.

This demonstrates how the enhancement layer wraps around existing
OpenEvolve PES without modifying any existing code.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_usage():
    """Demo: Basic usage with cost awareness."""
    print("=" * 60)
    print("Demo 1: Basic Cost-Aware Enhancement")
    print("=" * 60)
    
    from openevolve_pes_enhanced import create_cost_aware_enhancer
    
    # Create enhancer with $5 budget
    enhancer = create_cost_aware_enhancer(max_cost_usd=5.0)
    
    # Sample code to evolve
    code = """
def calculate_payment(amount, payment_method="credit"):
    tax = amount * 0.08
    subtotal = amount + tax
    
    if payment_method == "credit":
        fee = subtotal * 0.025
    else:
        fee = 0
    
    return {"subtotal": subtotal, "fee": fee, "total": subtotal + fee}
"""
    
    # Test cases
    tests = [
        {
            "name": "Basic payment",
            "input": {"amount": 100},
            "expected": {"total": 110.25},
        },
        {
            "name": "PayPal payment",
            "input": {"amount": 150, "payment_method": "paypal"},
            "expected": {"fee": 5.25},
        }
    ]
    
    # Run enhancement
    result = await enhancer.enhance_with_planning(
        code=code,
        problem_description="Add PayPal payment method support",
        tests=tests,
        language="python"
    )
    
    print(f"\nResults:")
    print(f"  Success: {result.original_result.success if result.original_result else False}")
    print(f"  Cost: ${result.total_cost_usd:.4f}")
    print(f"  Efficiency Gain: {result.efficiency_gain:.1%}")
    print(f"  Evaluations Saved: {result.evaluations_saved}")
    print(f"  Converged: {result.converged}")
    print(f"  Stopped Early: {result.stopped_early}")
    if result.stop_reason:
        print(f"  Stop Reason: {result.stop_reason}")


async def demo_strategy_selection():
    """Demo: Strategy selection based on problem characteristics."""
    print("\n" + "=" * 60)
    print("Demo 2: Strategy Selection")
    print("=" * 60)
    
    from openevolve_pes_enhanced import create_fully_enhanced
    
    enhancer = create_fully_enhanced()
    
    # Different problem types
    problems = [
        ("Simple function optimization", "low"),
        ("Complex multi-objective optimization with constraints", "high"),
        ("Lean 4 theorem proving", "theorem"),
        ("Multi-language code generation", "multi"),
    ]
    
    for problem_desc, complexity in problems:
        recommendation = enhancer.recommend_parameters(
            problem_description=problem_desc,
            max_cost_usd=10.0
        )
        
        print(f"\nProblem: {problem_desc}")
        print(f"  Strategy: {recommendation['strategy']}")
        print(f"  Estimated Cost: ${recommendation['estimated_cost']:.2f}")
        print(f"  Evaluations: {recommendation['estimated_evaluations']}")
        print(f"  Confidence: {recommendation['confidence']:.0%}")
        print(f"  Reasoning: {recommendation['reasoning']}")


async def demo_cost_estimation():
    """Demo: Cost estimation before running."""
    print("\n" + "=" * 60)
    print("Demo 3: Cost Estimation")
    print("=" * 60)
    
    from openevolve_pes_enhanced import create_fully_enhanced
    
    enhancer = create_fully_enhanced()
    
    # Estimate costs for different configurations
    configs = [
        ("Small", 20, 10),
        ("Medium", 50, 20),
        ("Large", 100, 50),
        ("Extra Large", 200, 100),
    ]
    
    print("\nCost Estimates:")
    print(f"{'Config':<15} {'Iters':<8} {'Pop':<8} {'Evals':<10} {'Cost':<12} {'Tokens':<10}")
    print("-" * 65)
    
    for name, iters, pop in configs:
        estimate = enhancer.get_cost_estimate(iters, pop)
        print(f"{name:<15} {iters:<8} {pop:<8} "
              f"{estimate['total_evaluations']:<10} "
              f"${estimate['total_cost_usd']:<11.4f} "
              f"{estimate['total_tokens']:<10}")


async def demo_comparison():
    """Demo: Compare enhanced vs standard evolution."""
    print("\n" + "=" * 60)
    print("Demo 4: Enhanced vs Standard Comparison")
    print("=" * 60)
    
    from openevolve_pes_enhanced import (
        PESIntegrationWrapper,
        PESEnhancedConfig,
        EnhancedAgnosticPES
    )
    
    # Standard (no enhancements)
    print("\nStandard Evolution (enhancements disabled):")
    standard_config = PESEnhancedConfig()  # All enhancements default to False
    standard = PESIntegrationWrapper(standard_config)
    print(f"  Cost optimization: {standard_config.enable_cost_optimization}")
    print(f"  Early stopping: {standard_config.enable_early_stopping}")
    print(f"  Planning: {standard_config.enable_planning}")
    
    # Enhanced (all enhancements)
    print("\nEnhanced Evolution (all enhancements enabled):")
    enhanced_config = PESEnhancedConfig.enable_all()
    enhanced = PESIntegrationWrapper(enhanced_config)
    print(f"  Cost optimization: {enhanced_config.enable_cost_optimization}")
    print(f"  Early stopping: {enhanced_config.enable_early_stopping}")
    print(f"  Planning: {enhanced_config.enable_planning}")
    print(f"  Summarization: {enhanced_config.enable_summarization}")
    print(f"  Adaptive parameters: {enhanced_config.enable_adaptive_parameters}")


async def demo_backward_compatibility():
    """Demo: Backward compatibility with existing code."""
    print("\n" + "=" * 60)
    print("Demo 5: Backward Compatibility")
    print("=" * 60)
    
    # Show that existing API still works
    print("\nExisting API (no changes needed):")
    print("  from openevolve_pes_integration import enhance_code")
    print("  result = enhance_code(code, problem, tests)")
    
    print("\nEnhanced API (additive only):")
    print("  from openevolve_pes_enhanced import create_cost_aware_enhancer")
    print("  enhancer = create_cost_aware_enhancer(max_cost_usd=5.0)")
    print("  result = await enhancer.enhance_with_planning(code, problem, tests)")
    print("  print(f'Cost: ${result.total_cost_usd}')  # New info available")
    
    print("\nDrop-in replacement:")
    print("  from openevolve_pes_enhanced import EnhancedAgnosticPES")
    print("  engine = EnhancedAgnosticPES(max_iterations=10)")
    print("  result = await engine.evolve(code, tests)  # Same API!")


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "OpenEvolve PES Enhanced" + " " * 20 + "║")
    print("║" + " " * 10 + "Pure Enhancement Layer Demo" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        await demo_basic_usage()
    except Exception as e:
        print(f"Demo 1 skipped: {e}")
    
    try:
        await demo_strategy_selection()
    except Exception as e:
        print(f"Demo 2 error: {e}")
    
    try:
        await demo_cost_estimation()
    except Exception as e:
        print(f"Demo 3 error: {e}")
    
    try:
        await demo_comparison()
    except Exception as e:
        print(f"Demo 4 error: {e}")
    
    try:
        await demo_backward_compatibility()
    except Exception as e:
        print(f"Demo 5 error: {e}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  * All existing code continues to work unchanged")
    print("  * Enhancements are purely additive")
    print("  * Cost awareness helps control spending")
    print("  * Early stopping saves evaluations")
    print("  * Strategy selection optimizes for problem type")


if __name__ == "__main__":
    asyncio.run(main())
