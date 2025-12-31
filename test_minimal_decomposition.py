"""
Minimal test for problem decomposition
"""

# Test if we can import the basic components
try:
    exec(open('problem_decomposition.py').read())
    print("✅ File executed successfully")
    
    # Test if classes are available in local scope
    if 'ProblemDecomposer' in locals():
        print("✅ ProblemDecomposer class found")
        decomposer = ProblemDecomposer()
        print("✅ ProblemDecomposer instantiated")
        
        # Test basic functionality
        test_content = "This is a simple test content for decomposition."
        result = decomposer.decompose_content(test_content)
        
        if result:
            print("✅ Decomposition successful")
            print(f"   - Components: {len(result.components)}")
            print(f"   - Quality score: {result.quality_score}")
            print(f"   - Strategy: {result.decomposition_strategy}")
        else:
            print("❌ Decomposition failed")
    else:
        print("❌ ProblemDecomposer class not found in local scope")
        print("Available classes:", [name for name in locals() if name[0].isupper()])
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()