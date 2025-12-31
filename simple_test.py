try:
    import problem_decomposition
    print("Module imported successfully")
    print("Available attributes:", [attr for attr in dir(problem_decomposition) if not attr.startswith('_')])
    
    # Try to access the class
    if hasattr(problem_decomposition, 'ProblemDecomposer'):
        print("ProblemDecomposer class found")
        decomposer = problem_decomposition.ProblemDecomposer()
        print("ProblemDecomposer instantiated successfully")
    else:
        print("ProblemDecomposer class not found")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()