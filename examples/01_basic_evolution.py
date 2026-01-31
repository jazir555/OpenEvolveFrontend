"""
Basic Evolution - Your First Optimization

This example shows the simplest way to use OpenEvolve.
We'll evolve a simple mathematical function to find the best solution.

Problem: Maximize f(x) = x^2 where x in [0, 10]
Expected: The best solution should approach x = 10
"""

# EVOLVE-BLOCK-START
def solve():
    """Initial solution - starting point for evolution"""
    x = 5  # Starting guess
    return x ** 2
# EVOLVE-BLOCK-END


"""
EXPECTED OUTPUT:
---------------
After running this example with OpenEvolve:

Best solution:
    def solve():
        x = 10
        return x ** 2

Final score: 100.0 (10^2 = 100)

HOW TO RUN:
----------
1. Save this file as initial_program.py
2. Create an evaluator file (see evaluator.py)
3. Run: openevolve initial_program.py evaluator.py
   OR
   Use Python API:
   ```python
   from openevolve import run_evolution
   result = run_evolution('initial_program.py', 'evaluator.py', iterations=10)
   print(f"Best score: {result.best_score}")
   ```
"""
