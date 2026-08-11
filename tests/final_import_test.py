#!/usr/bin/env python3
"""Final import test for 100% success rate verification."""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Test all previously failing imports."""
    
    # CAV-NLP files
    cav_nlp_files = [
        'openevolve.cav_nlp_integration',
        'openevolve.cav_nlp_integration.advanced_compositional_rules',
        'openevolve.cav_nlp_integration.arxiv_corpus_learner',
        'openevolve.cav_nlp_integration.canonical_lean_generator',
        'openevolve.cav_nlp_integration.cegis_learner',
        'openevolve.cav_nlp_integration.compositional_semantics',
        'openevolve.cav_nlp_integration.z3_canonicalizer',
        'openevolve.cav_nlp_integration.z3_semantic_synthesis',
        'openevolve.cav_nlp_integration.z3_validated_ir',
    ]
    
    # Core fixed files
    core_files = [
        'input_validation',
        'crewai_zero_error_workflow',
        'roma_config',
        'sovereign_data_models',
        'decomposition_recomposition_integration',
        'bubblelabs_integration',
        'openevolve.api',
        'openevolve.config',
    ]
    
    results = {'ok': [], 'fail': []}
    
    print("Testing CAV-NLP integration files...")
    for f in cav_nlp_files:
        try:
            __import__(f)
            results['ok'].append(f)
            print(f"  OK: {f}")
        except Exception as e:
            results['fail'].append((f, type(e).__name__, str(e)))
            print(f"  FAIL: {f} - {type(e).__name__}")
    
    print("\nTesting core fixed files...")
    for f in core_files:
        try:
            __import__(f)
            results['ok'].append(f)
            print(f"  OK: {f}")
        except Exception as e:
            results['fail'].append((f, type(e).__name__, str(e)))
            print(f"  FAIL: {f} - {type(e).__name__}")
    
    print("\n" + "="*50)
    print(f"Results: {len(results['ok'])} OK, {len(results['fail'])} FAIL")
    
    if results['fail']:
        print("\nFailed imports:")
        for f, err_type, err_msg in results['fail']:
            print(f"  {f}: {err_type}")
    
    return len(results['fail']) == 0

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
