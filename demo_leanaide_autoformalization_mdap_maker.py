"""
Demo for LeanAide Autoformalization System with MDAP/MAKER Integration

This demo shows how to use the autoformalization system that combines:
- Natural language to Lean 4 code translation
- MDAP (Multi-Agent Decomposition) for multi-agent generation
- MAKER (Multi-Agent Voting) for voting-based refinement
"""

import asyncio
from unittest.mock import Mock

from leanaide_autoformalization_mdap_maker import (
    LeanAideAutoformalizationEngine,
    AutoformalizationStrategy,
    create_leanaide_autoformalization_engine,
    autoformalize_with_mdap_maker
)


async def demo_basic_autoformalization():
    """Demo basic autoformalization functionality."""
    print("=== Demo: Basic Autoformalization ===\n")
    
    # Create mock LeanAide client
    mock_leanaide_client = Mock()
    mock_leanaide_client.cache = Mock()
    
    # Create the autoformalization engine
    engine = create_leanaide_autoformalization_engine(
        leanaide_client=mock_leanaide_client,
        enable_caching=True
    )
    
    # Test natural language statements
    test_statements = [
        ("Prove that for all natural numbers n, n + 0 = n", "add_zero"),
        ("Prove that the sum of two even numbers is even", "sum_even"),
        ("Prove that for any real number x, x^2 >= 0", "square_nonneg")
    ]
    
    for statement, name in test_statements:
        print(f"Natural Language: {statement}")
        
        result = await engine.autoformalize(
            natural_language=statement,
            statement_type="theorem",
            name=name,
            strategy=AutoformalizationStrategy.DIRECT
        )
        
        print(f"Success: {result.success}")
        print(f"Strategy Used: {result.strategy_used}")
        print(f"Lean Code:\n{result.lean_code}")
        print(f"Confidence: {result.confidence}")
        print("-" * 50)


async def demo_strategy_comparison():
    """Demo different strategies for autoformalization."""
    print("\n=== Demo: Strategy Comparison ===\n")
    
    # Create mock client
    mock_leanaide_client = Mock()
    mock_leanaide_client.cache = Mock()
    
    engine = create_leanaide_autoformalization_engine(
        leanaide_client=mock_leanaide_client
    )
    
    statement = "Prove by induction that the sum of first n natural numbers is n*(n+1)/2"
    
    strategies = [
        AutoformalizationStrategy.DIRECT,
        AutoformalizationStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        print(f"Testing strategy: {strategy.value}")
        
        result = await engine.autoformalize(
            natural_language=statement,
            statement_type="theorem",
            name="induction_sum",
            strategy=strategy
        )
        
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Execution Time: {result.execution_time:.3f}s")
        print()


async def demo_system_status():
    """Demo system status and capabilities."""
    print("\n=== Demo: System Status ===\n")
    
    # Create engine
    mock_leanaide_client = Mock()
    mock_leanaide_client.cache = Mock()
    
    engine = LeanAideAutoformalizationEngine(
        leanaide_client=mock_leanaide_client
    )
    
    status = engine.get_system_status()
    
    print("System Capabilities:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nAvailable Strategies:")
    for strategy in AutoformalizationStrategy:
        print(f"  - {strategy.value}")


async def demo_convenience_function():
    """Demo the convenience function."""
    print("\n=== Demo: Convenience Function ===\n")
    
    # Create mock client
    mock_leanaide_client = Mock()
    mock_leanaide_client.cache = Mock()
    
    # Use the convenience function
    result = await autoformalize_with_mdap_maker(
        natural_language="Prove that for any natural number n, n equals itself",
        leanaide_client=mock_leanaide_client,
        statement_type="theorem",
        name="reflexivity"
    )
    
    print(f"Result Success: {result.success}")
    print(f"Strategy Used: {result.strategy_used}")
    print(f"Confidence: {result.confidence}")
    print(f"Lean Code Preview: {result.lean_code[:100]}...")


async def main():
    """Run all demos."""
    print("LeanAide Autoformalization System with MDAP/MAKER Integration Demo")
    print("=" * 70)
    
    await demo_basic_autoformalization()
    await demo_strategy_comparison()
    await demo_system_status()
    await demo_convenience_function()
    
    print("\nDemo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Natural language to Lean 4 code translation")
    print("- Multiple autoformalization strategies")
    print("- Caching for performance optimization")
    print("- System status and capability reporting")
    print("- Convenience functions for easy integration")


if __name__ == "__main__":
    asyncio.run(main())