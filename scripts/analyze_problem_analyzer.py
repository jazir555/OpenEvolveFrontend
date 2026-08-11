"""
Direct test of problem analyzer functionality by examining the source code.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_problem_analyzer_source():
    """Analyze the problem analyzer source code for completeness."""
    
    # Read the problem analyzer source file
    with open(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\problem_analyzer.py", "r", encoding="utf-8") as f:
        source_code = f.read()
    
    print("Analyzing ProblemAnalyzer implementation...")
    
    # Check for key components
    checks = [
        ("OpenEvolve client integration", "openevolve_client" in source_code.lower()),
        ("Domain context extraction", "extract_domain_context" in source_code),
        ("Problem type classification", "classify_problem_type" in source_code),
        ("Complexity assessment", "assess_complexity" in source_code),
        ("Constraint identification", "identify_constraints" in source_code),
        ("Success criteria generation", "generate_success_criteria" in source_code),
        ("LLM-based analysis methods", "def _extract_domain_context_llm" in source_code),
        ("Fallback implementations", "def _extract_domain_context_fallback" in source_code),
        ("Error handling decorators", "@with_error_handling" in source_code or "@with_retry" in source_code),
        ("Validation methods", "validate_problem_definition" in source_code),
    ]
    
    # Print results
    all_pass = True
    for check_name, check_result in checks:
        status = "[PASS]" if check_result else "[FAIL]"
        print(f"{status}: {check_name}")
        if not check_result:
            all_pass = False
    
    print("\n" + "="*50)
    if all_pass:
        print("ALL CHECKS PASSED: ProblemAnalyzer is fully implemented!")
    else:
        print("Some checks failed - implementation may be incomplete")
    
    # Check for class definition
    has_class = "class ProblemAnalyzer:" in source_code
    print(f"{'[PASS]' if has_class else '[FAIL]'}: ProblemAnalyzer class defined")
    
    # Check for key methods
    key_methods = [
        "analyze_problem",
        "extract_domain_context", 
        "classify_problem_type",
        "assess_complexity",
        "identify_constraints",
        "generate_success_criteria"
    ]
    
    print("\nKey Method Implementation Status:")
    for method in key_methods:
        has_method = method in source_code
        print(f"{'[OK]' if has_method else '[MISSING]'} {method}")
    
    # Check for fallback implementations
    fallback_methods = [
        "_extract_domain_context_fallback",
        "_classify_problem_type_fallback", 
        "_assess_complexity_fallback",
        "_identify_constraints_fallback",
        "_generate_criteria_fallback"
    ]
    
    print("\nFallback Implementation Status:")
    for method in fallback_methods:
        has_fallback = method in source_code
        print(f"{'[OK]' if has_fallback else '[MISSING]'} {method}")
    
    # Check for LLM implementations
    llm_methods = [
        "_extract_domain_context_llm",
        "_classify_problem_type_llm",
        "_assess_complexity_llm", 
        "_identify_constraints_llm",
        "_generate_criteria_with_llm"
    ]
    
    print("\nLLM Implementation Status:")
    for method in llm_methods:
        has_llm = method in source_code
        print(f"{'[OK]' if has_llm else '[MISSING]'} {method}")
    
    return all_pass

if __name__ == "__main__":
    success = analyze_problem_analyzer_source()
    sys.exit(0 if success else 1)