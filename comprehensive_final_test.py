#!/usr/bin/env python3
"""Comprehensive final import test for 100% success rate verification."""

import sys
import os
sys.path.insert(0, '.')

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

def test_comprehensive():
    """Test comprehensive imports."""
    
    # All the files that were previously failing
    test_modules = [
        # Root level
        'input_validation',
        'crewai_zero_error_workflow',
        'roma_config',
        'sovereign_data_models',
        'decomposition_recomposition_integration',
        'bubblelabs_integration',
        
        # OpenEvolve
        'openevolve.api',
        'openevolve.config',
        'openevolve.cav_nlp_integration',
        'openevolve.cav_nlp_integration.advanced_compositional_rules',
        'openevolve.cav_nlp_integration.arxiv_corpus_learner',
        'openevolve.cav_nlp_integration.canonical_lean_generator',
        'openevolve.cav_nlp_integration.cegis_learner',
        'openevolve.cav_nlp_integration.compositional_semantics',
        'openevolve.cav_nlp_integration.z3_canonicalizer',
        'openevolve.cav_nlp_integration.z3_semantic_synthesis',
        'openevolve.cav_nlp_integration.z3_validated_ir',
        
        # Core modules
        'workflow_structures',
        'mdap_maker_complete',
        'openevolve_client',
        'llm_utils',
        'roma_mcp_tools',
    ]
    
    results = {'ok': [], 'fail': []}
    
    print("="*60)
    print("COMPREHENSIVE IMPORT TEST")
    print("="*60)
    print()
    
    for module in test_modules:
        try:
            __import__(module)
            results['ok'].append(module)
            print(f"OK: {module}")
        except Exception as e:
            results['fail'].append((module, type(e).__name__))
            print(f"FAIL: {module}: {type(e).__name__}")
    
    print()
    print("="*60)
    print(f"Results: {len(results['ok'])} OK, {len(results['fail'])} FAIL")
    success_rate = len(results['ok']) / len(test_modules) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*60)
    
    if results['fail']:
        print("\nFailed imports:")
        for f, err_type in results['fail']:
            print(f"  - {f}: {err_type}")
    
    return len(results['fail']) == 0

if __name__ == '__main__':
    success = test_comprehensive()
    sys.exit(0 if success else 1)
