"""
Test utility functions by executing the file directly
"""

# Execute the problem_decomposition.py file
exec(open('problem_decomposition.py').read())

# Now test the functions
def test_utility_functions():
    print("Testing utility functions...")
    
    test_content = """
# Header 1
Some content here.

```python
def function_one():
    pass

class MyClass:
    pass
```

import numpy as np
from sklearn import datasets
"""
    
    # Test analyze_content_patterns
    patterns = analyze_content_patterns(test_content)
    print(f"✅ Content patterns: {patterns}")
    
    # Test suggest_optimal_strategy
    strategy = suggest_optimal_strategy(test_content)
    print(f"✅ Suggested strategy: {strategy.value}")
    
    # Test with decomposer
    decomposer = ProblemDecomposer()
    result = decomposer.decompose_content(test_content, strategy=strategy)
    
    # Test create_decomposition_report
    report = create_decomposition_report(result)
    print(f"✅ Report generated: {len(report)} characters")
    print("First 200 chars of report:")
    print(report[:200] + "...")
    
    return True

if __name__ == "__main__":
    try:
        test_utility_functions()
        print("\n🎉 All utility function tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()