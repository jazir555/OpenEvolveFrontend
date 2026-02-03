"""
Example: Using LoongFlow adapter from OpenEvolve

This example demonstrates how to use the LoongFlow adapter to integrate
LoongFlow's PES (Plan-Execute-Summarize) system with OpenEvolve.
"""

import asyncio
import logging
import sys
from openevolve.integrations import LoongFlowAdapter

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_usage():
    """
    Example 1: Basic usage with minimal configuration
    """
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60 + "\n")

    # Initialize adapter with OpenEvolve-style config
    config = {
        "max_iterations": 10,
        "population_size": 5,
        "enable_planning": True,
        "enable_memory": True,
        "timeout": 60,
    }

    adapter = LoongFlowAdapter(config)

    # Check if available
    if adapter.is_available():
        print("[OK] LoongFlow is available")

        # Run evolution
        result = await adapter.evolve(
            problem="Optimize function: f(x) = x^2 for x in range 0-100",
            domain="math"
        )

        print(f"\nResults:")
        print(f"  Best fitness: {result['best_fitness']}")
        print(f"  Evaluations: {result['total_evaluations']}")
        print(f"  Iterations: {result['iterations_performed']}")
        print(f"  Improvement rate: {result['improvement_rate']}")

        if 'error' in result:
            print(f"  Error: {result['error']}")
    else:
        print("[WARNING] LoongFlow not available, using fallback mode")
        print("   This is expected if LoongFlow is not installed.")


async def example_with_llm_config():
    """
    Example 2: With LLM configuration
    """
    print("\n" + "="*60)
    print("Example 2: With LLM Configuration")
    print("="*60 + "\n")

    # Initialize adapter with LLM config
    config = {
        "max_iterations": 20,
        "population_size": 10,
        "enable_planning": True,
        "enable_memory": True,
        "llm_config": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
        }
    }

    adapter = LoongFlowAdapter(config)

    # Get capabilities
    capabilities = adapter.get_capabilities()
    print(f"Adapter capabilities: {capabilities}")

    if adapter.is_available():
        # Run evolution with code optimization
        result = await adapter.evolve(
            problem="Optimize this sorting algorithm for performance",
            domain="code",
            initial_code="""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
        )

        print(f"\nOptimization Results:")
        print(f"  Best fitness: {result['best_fitness']}")
        print(f"  Evaluations: {result['total_evaluations']}")


async def example_math_problem():
    """
    Example 3: Math optimization problem
    """
    print("\n" + "="*60)
    print("Example 3: Math Problem Optimization")
    print("="*60 + "\n")

    config = {
        "max_iterations": 15,
        "population_size": 8,
        "enable_planning": True,
        "enable_memory": True,
        "timeout": 120,
    }

    adapter = LoongFlowAdapter(config)

    if adapter.is_available():
        # Solve a math optimization problem
        result = await adapter.evolve(
            problem="Find the minimum of f(x,y) = x^2 + y^2 - 4x - 4y",
            domain="math"
        )

        print(f"\nOptimization Results:")
        print(f"  Best solution: {result['best_solution']}")
        print(f"  Best fitness: {result['best_fitness']}")
        print(f"  Total evaluations: {result['total_evaluations']}")
        print(f"  Iterations performed: {result['iterations_performed']}")


async def example_error_handling():
    """
    Example 4: Error handling and fallback
    """
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60 + "\n")

    # This should work even if LoongFlow is not installed
    config = {
        "max_iterations": 10,
        "population_size": 5,
    }

    adapter = LoongFlowAdapter(config)

    # Always get capabilities, even if unavailable
    capabilities = adapter.get_capabilities()
    print(f"Available: {capabilities['available']}")
    print(f"Supported domains: {capabilities['supported_domains']}")

    # Try to run evolution (will use fallback if unavailable)
    result = await adapter.evolve(
        problem="Test problem",
        domain="general"
    )

    print(f"\nFallback result:")
    print(f"  Best fitness: {result['best_fitness']}")
    if 'error' in result:
        print(f"  Error message: {result['error']}")


async def main():
    """
    Run all examples
    """
    print("\n" + "="*60)
    print("LoongFlow Adapter Usage Examples")
    print("="*60)

    # Run examples
    await example_basic_usage()
    await example_with_llm_config()
    await example_math_problem()
    await example_error_handling()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
