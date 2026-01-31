"""
Function Evolution Example - Evolve a Sorting Algorithm

This example demonstrates how to evolve a Python function to improve
its performance. We'll start with a slow bubble sort and evolve it
into a faster sorting algorithm.

Problem: Sort an array of numbers as fast as possible
"""

# EVOLVE-BLOCK-START
def sort_array(arr):
    """
    Initial implementation - Slow bubble sort
    This is our starting point for evolution
    """
    # Make a copy to avoid modifying original
    result = arr.copy()

    # Bubble sort - O(n^2) complexity
    for i in range(len(result)):
        for j in range(len(result) - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]

    return result
# EVOLVE-BLOCK-END


"""
EXPECTED OUTPUT:
---------------
Evolution should discover more efficient sorting algorithms like:
- Built-in sorted() function
- Quick sort
- Merge sort

Performance improvements:
- Bubble sort: O(n^2) - Very slow for large arrays
- Quick sort: O(n log n) - Much faster
- Built-in sorted: O(n log n) - Fastest (optimized C implementation)

HOW TO RUN:
----------
Using Python API:
```python
from openevolve import run_evolution

result = run_evolution(
    'function_evolution.py',
    'sort_evaluator.py',
    iterations=20,
    config={
        'llm': {
            'models': [
                {'name': 'gpt-4', 'api_key': 'your-key'}
            ]
        }
    }
)

print(f"Best score: {result.best_score:.4f}")
print(f"Best code:\\n{result.best_code}")
```

Or using CLI:
```bash
openevolve function_evolution.py sort_evaluator.py -i 20
```
"""
