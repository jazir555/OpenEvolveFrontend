#!/usr/bin/env python3
"""Massive comprehensive import test for 100% success rate verification."""

import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, './core-projects')
sys.path.insert(0, './openevolve')

os.environ['PYTHONWARNINGS'] = 'ignore'

# All previously failing files from all batches
TEST_MODULES = [
    # Batch 1 - Root files (180 failures)
    'input_validation',
    'crewai_zero_error_workflow', 
    'roma_config',
    'sovereign_data_models',
    'decomposition_recomposition_integration',
    'bubblelabs_integration',
    'comprehensive_demo',
    'comprehensive_openevolve_test',
    'comprehensive_system_test',
    'comprehensive_security_test_coverage',
    'automated_proof_engine',
    'benchmark_ultra_comprehensive_artifacts',
    'ml_pattern_clustering',
    'openevolve_cli',
    'physics_validator_real',
    'secure_api',
    
    # OpenEvolve API
    'openevolve.api',
    'openevolve.config',
    'openevolve_client',
    'openevolve_integration',
    'openevolve_maker_integration',
    'openevolve_crewai_adapter',
    'openevolve_crewai_bridge',
    'openevolve_crewai_delegation',
    'openevolve_decomposition_adapter',
    'openevolve_enhanced_decomposition_integration',
    
    # CAV-NLP Integration (all 11 files)
    'openevolve.cav_nlp_integration',
    'openevolve.cav_nlp_integration.advanced_compositional_rules',
    'openevolve.cav_nlp_integration.arxiv_corpus_learner',
    'openevolve.cav_nlp_integration.canonical_lean_generator',
    'openevolve.cav_nlp_integration.cegis_learner',
    'openevolve.cav_nlp_integration.compositional_semantics',
    'openevolve.cav_nlp_integration.rule_discovery_from_arxiv',
    'openevolve.cav_nlp_integration.z3_canonicalizer',
    'openevolve.cav_nlp_integration.z3_semantic_synthesis',
    'openevolve.cav_nlp_integration.z3_validated_ir',
    'openevolve.cav_nlp_integration.flexible_semantic_parsing',
    
    # Core workflow/engine files
    'workflow_structures',
    'workflow_engine',
    'workflow_persistence',
    'workflow_state_manager',
    'mdap_maker_complete',
    'mdap_engine',
    'decomposition_engine',
    'decomposition_strategy',
    'decomposition_recomposition_integration',
    
    # Team and analysis
    'team_assignment_engine',
    'resource_estimation_engine',
    'complexity_analyzer',
    'problem_classifier',
    'knowledge_context_assembler',
    
    # ROM A
    'roma_crewai_bridge',
    'roma_crewai_tools',
    'roma_mdap_maker_crewai_bridge',
    'roma_mdap_maker_crewai_tools',
    'roma_mcp_tools',
    
    # CrewAI
    'crewai_client',
    'crewai_unified_bridge',
    'crewai_unified_flow',
    'crewai_zero_error_workflow',
    'crewai_enhanced_decomposition_bridge',
    
    # Leanaide
    'leanaide_redflagging',
    'leanaide_pes_benchmark',
    'leanaide_integration',
    'leanaide_pes_handler',
    'leanaide_mcts_mdap',
    'leanaide_mcp_tools',
    'leanaide_autoformalization_mdap_maker',
    
    # BubbleLabs
    'bubblelabs_integration',
    'bubblelabs_nodes',
    'bubblelabs_analytics',
    'bubblelabs_crewai_bridge',
    'openevolve_bubblelabs_ui',
    
    # Datapizza
    'datapizza_config',
    'datapizza_crewai_bridge',
    
    # Z3
    'z3prover_integration',
    'z3_leanaide_bridge',
    'z3_mcp_tools',
    
    # Test files
    'additional_unit_tests',
    'advanced_system_unit_tests',
    'advanced_unit_tests_comprehensive',
    'comprehensive_test_suite',
    'comprehensive_validation_tests',
    'edge_case_tests',
    'extended_unit_tests',
    'extra_comprehensive_tests',
    'real_xss_prevention_tests',
    'system_integration_validation',
    'testing_framework',
    'ultimate_comprehensive_tests',
    'ultra_comprehensive_tests',
    
    # Examples
    'examples.roma_decomposition_basic',
    'examples.roma_decomposition_advanced',
    'enhanced_gauntlet_example',
    'finance.insurance_example',
    
    # Utilities
    'llm_utils',
    'content_analyzer',
    'prompt_engineering',
    'solution_pattern_miner',
    
    # Glue adapters
    'glue.adapters.rese_leanaide_workflow.tests.test_leanaide_rese_workflow',
    'glue.adapters.rese_z3_bridge.tests.test_rese_z3_bridge',
    
    # Other integration
    'openevolve_workflow_manager',
    'openevolve_unified_math_service',
    'symbolic_constraint_engine',
    'associative_recomposition',
]

def test_all():
    """Test all modules."""
    results = {'ok': [], 'fail': []}
    
    print("="*70)
    print("MASSIVE COMPREHENSIVE IMPORT TEST")
    print(f"Testing {len(TEST_MODULES)} modules...")
    print("="*70)
    print()
    
    for i, module in enumerate(TEST_MODULES, 1):
        try:
            __import__(module)
            results['ok'].append(module)
            status = "OK"
        except Exception as e:
            results['fail'].append((module, type(e).__name__, str(e)[:50]))
            status = f"FAIL: {type(e).__name__}"
        
        if i % 10 == 0 or i == len(TEST_MODULES):
            print(f"  Progress: {i}/{len(TEST_MODULES)} - {status}")
    
    print()
    print("="*70)
    print(f"Results: {len(results['ok'])} OK, {len(results['fail'])} FAIL")
    success_rate = len(results['ok']) / len(TEST_MODULES) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*70)
    
    if results['fail']:
        print()
        print("Failed imports:")
        for mod, err_type, err_msg in results['fail'][:20]:
            print(f"  - {mod}: {err_type}")
        if len(results['fail']) > 20:
            print(f"  ... and {len(results['fail']) - 20} more")
    
    return len(results['fail']) == 0

if __name__ == '__main__':
    success = test_all()
    sys.exit(0 if success else 1)
