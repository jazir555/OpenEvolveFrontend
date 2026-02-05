"""
Debug the ProblemDecomposer class to find where it's failing
"""

import problem_decomposition
import inspect

# Get the class
cls = problem_decomposition.ProblemDecomposer

# Get the source code
try:
    source = inspect.getsource(cls)
    print("Class source length:", len(source))
    
    # Count methods in source
    method_count = source.count('def ')
    print("Methods found in source:", method_count)
    
    # Get actual methods
    methods = [name for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)]
    print("Actual methods available:", len(methods))
    print("Method names:", methods)
    
    # Check if reassemble_components is in source
    if 'reassemble_components' in source:
        print("[OK] reassemble_components found in source")
        # Find its position
        pos = source.find('def reassemble_components')
        print(f"Position in source: {pos}")
    else:
        print("[FAIL] reassemble_components NOT found in source")
        
except Exception as e:
    print(f"Error getting source: {e}")

# Try to create an instance
try:
    instance = cls()
    print("[OK] Instance created successfully")
    
    # Check available methods on instance
    instance_methods = [attr for attr in dir(instance) if not attr.startswith('_') and callable(getattr(instance, attr))]
    print("Instance methods:", instance_methods)
    
except Exception as e:
    print(f"[FAIL] Error creating instance: {e}")