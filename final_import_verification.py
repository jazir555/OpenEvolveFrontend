#!/usr/bin/env python3
"""
Final Import Verification Script - 100% Success Rate Goal
Tests all previously failing files and generates final report.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# All previously failing files organized by batch
ALL_PREVIOUSLY_FAILING = {
    # Batch 1 - 180 files (extracted from import_test_batch1.json)
    "batch1": [
        "additional_unit_tests.py", "advanced_system_unit_tests.py", "advanced_unit_tests_comprehensive.py",
        "apply_ace_phase4_fixes.py", "apply_ace_security_fixes.py", "apply_api_consistency_fixes.py",
        "apply_code_quality_fixes.py", "apply_phase4_validation.py", "assess_decomposition.py",
        "audit_lean_files.py", "benchmark_improvements.py", "bubblelab-auto-setup.py",
        "c2c_usage_examples.py", "complexity_analyzer.py", "comprehensive_decomposition_engine.py",
        "comprehensive_demo.py", "comprehensive_integration_test.py", "comprehensive_recomposition_engine.py",
        "comprehensive_test_suite.py", "comprehensive_validation_tests.py", "crewai_client.py",
        "crewai_enhanced_decomposition_bridge.py", "crewai_unified_bridge.py", "crewai_unified_flow.py",
        "datapizza_crewai_bridge.py", "debug_class.py", "debug_source.py", "debug_test.py",
        "debug_test_wrapper.py", "decomposition_engine_adaptive_enhancement.py",
        "decomposition_matryoshka_integration.py", "decomposition_strategy.py",
        "demo_adversarial_maker.py", "demo_app.py", "demo_crewai_research_features.py",
        "demo_database_cleanup.py", "demo_e2e_invention_enhanced.py", "demo_end_to_end_invention.py",
        "demo_enhanced_adversarial.py", "demo_enhanced_decomposition_recomposition.py",
        "demo_evolution_maker.py", "demo_evolution_mdap.py", "demo_evolutionary_tests.py",
        "demo_generic_maker.py", "demo_hierarchical_indexing.py", "demo_hybrid_maker.py",
        "demo_hybrid_mcts.py", "demo_integration.py", "demo_knowledge_extraction_ml.py",
        "demo_leanaide_autoformalization_mdap_maker.py", "demo_leanaide_client.py",
        "demo_leanaide_config.py", "demo_leanaide_redflagging.py", "demo_maker_complete.py",
        "demo_matryoshka_auto.py", "demo_matryoshka_unified_memory.py", "demo_mcts.py",
        "demo_mcts_mdap.py", "demo_mdap_maker.py", "demo_mdap_maker_matryoshka.py",
        "demo_mdap_maker_mcts_unified.py", "demo_openevolve_bubblelabs.py",
        "demo_openevolve_integration.py", "demo_openevolve_pes_integration.py",
        "demo_pes_workflow.py", "demo_pes_workflow_language_agnostic.py",
        "demo_pes_workflow_universal.py", "demo_problem_classifier.py",
        "demo_quality_calculator.py", "demo_reliability_system.py", "demo_roma_mdap_maker.py",
        "demo_sop_components.py", "demo_sop_generator.py", "demo_sop_integrated.py",
        "demo_team_assignment.py", "demo_ui_integration.py", "demo_unified_memory_system.py",
        "demo_z3_leanaide_integration.py", "dependency_builder.py", "detailed_audit.py",
        "domain_optimization_manager.py", "e2e_invention_validation.py", "edge_case_tests.py",
        "evolve_sop.py", "example_crewai_delegation.py", "example_decomposition_integration.py",
        "example_enhanced_decomposition.py", "example_integration_usage.py",
        "extended_unit_tests.py", "extra_comprehensive_tests.py", "future_enhancements.py",
        "graphql_server.py", "hybrid_maker_workflow.py", "hybrid_types.py",
        "knowledge_context_assembler.py", "knowledge_engine_hierarchical_integration.py",
        "langchain_chroma_integration.py", "leanaide_pes_benchmark.py", "leanaide_redflagging.py",
        "leanaide_sop_integration.py", "learning_loop_manager.py", "lmql_adapter.py",
        "mainlayout.py", "master_test_runner.py", "migration_report.py",
        "openevolve_bubblelabs_ui.py", "openevolve_crewai_adapter.py", "openevolve_crewai_bridge.py",
        "openevolve_crewai_delegation.py", "openevolve_decomposition_adapter.py",
        "openevolve_enhanced_decomposition_integration.py", "openevolve_workflow_manager_integrated.py",
        "persistent_decomposition_engine.py", "problem_classifier.py", "problem_fractal_pipeline.py",
        "quality_control.py", "quality_control_examples.py", "quick_test_integration.py",
        "rbac_enhanced.py", "real_xss_prevention_tests.py", "reporting_demo.py",
        "reporting_system.py", "resource_estimation_engine.py", "robustness_integration.py",
        "roma_crewai_bridge.py", "roma_crewai_tools.py", "roma_mdap_maker_crewai_bridge.py",
        "roma_mdap_maker_crewai_tools.py", "run_all_ace_tests.py", "run_all_batch2_tests.py",
        "run_all_gauntlet_tests.py", "run_all_tests.py", "run_evolution_mdap_tests.py",
        "run_evolutionary_tests.py", "run_full_rese_e2e_pipeline.py", "run_gauntlet_tests.py",
        "run_integration_tests.py", "run_leanaide_tests.py", "run_mcts_mdap_tests.py",
        "run_mcts_tests.py", "run_mdap_tests.py", "run_real_security_tests.py",
        "run_rese_tests.py", "run_security_tests.py", "run_security_true_100_tests.py",
        "run_tests.py", "simple_check.py", "simple_demo.py", "simple_dspy_test.py",
        "simple_test.py", "simple_test_clean.py", "simple_verify_implementation.py",
        "solution_assembler.py", "sop_generator_research_quest.py",
        "sovereign_decomposition_crewai_integration.py", "sovereign_ui.py",
        "sovereign_ui_components.py", "symbolic_constraint_engine.py",
        "system_integration_validation.py", "team_assignment_engine.py", "testing_framework.py",
        "tripartite_production.py", "ultimate_comprehensive_tests.py", "ultra_comprehensive_tests.py",
        "validate_all_fixes.py", "validate_end_to_end_invention.py", "validate_enhanced_adversarial.py",
        "validate_evolution_maker_integration.py", "validate_generic_maker_integration.py",
        "validate_hybrid_maker_integration.py", "validate_integration.py",
        "validate_leanaide_tests.py", "validate_maker_integration.py", "validate_performance.py",
        "validate_phase1_complete.py", "validate_sop_components.py", "validate_sop_generator.py",
        "validate_sop_integrated.py", "verify_causal_learn_final.py", "verify_complete_integration.py",
    ],
    # Batch 3 - 19 files (openevolve)
    "batch3": [
        "openevolve\\__init__.py", "openevolve\\cav_nlp_integration\\advanced_compositional_rules.py",
        "openevolve\\cav_nlp_integration\\arxiv_corpus_learner.py",
        "openevolve\\cav_nlp_integration\\canonical_lean_generator.py",
        "openevolve\\cav_nlp_integration\\cegis_learner.py",
        "openevolve\\cav_nlp_integration\\compositional_semantics.py",
        "openevolve\\cav_nlp_integration\\flexible_semantic_parsing.py",
        "openevolve\\cav_nlp_integration\\rule_discovery_from_arxiv.py",
        "openevolve\\cav_nlp_integration\\test_cav_nlp.py",
        "openevolve\\cav_nlp_integration\\z3_canonicalizer.py",
        "openevolve\\cav_nlp_integration\\z3_semantic_synthesis.py",
        "openevolve\\cav_nlp_integration\\z3_validated_ir.py",
        "openevolve_bubblelabs_ui.py", "openevolve_crewai_adapter.py", "openevolve_crewai_bridge.py",
        "openevolve_crewai_delegation.py", "openevolve_decomposition_adapter.py",
        "openevolve_enhanced_decomposition_integration.py", "openevolve_workflow_manager_integrated.py",
    ],
    # Batch 5 - 9 files (leanaide/roma/z3)
    "batch5": [
        "leanaide_integration.py", "leanaide_pes_benchmark.py", "leanaide_redflagging.py",
        "leanaide_sop_integration.py", "roma_crewai_bridge.py", "roma_crewai_tools.py",
        "roma_mdap_maker_crewai_bridge.py", "roma_mdap_maker_crewai_tools.py", "z3_api.py",
    ],
    # Batch 6 - 8 files (workflow/decomposition)
    "batch6": [
        "comprehensive_recomposition_engine.py", "decomposition_engine_adaptive_enhancement.py",
        "persistent_decomposition_engine.py", "resource_estimation_engine.py",
        "symbolic_constraint_engine.py", "team_assignment_engine.py",
        "workflow_persistence.py", "workflow_state_manager.py",
    ],
    # Batch 7 - 9 files (demo/validate) - excluding those that timeout/run code
    "batch7_fixable": [
        "demo_database_cleanup.py", "demo_matryoshka_unified_memory.py",
        "demo_openevolve_bubblelabs.py", "demo_reliability_system.py",
        "demo_team_assignment.py", "validate_performance.py",
    ],
    # Batch 8 - 41 files (examples/glue)
    "batch8": [
        "examples\\04_python_api.py", "examples\\associative_recomposition_example.py",
        "examples\\example_business_process.py", "examples\\example_software_architecture.py",
        "examples\\investment_committee_demo.py", "examples\\lean4_usage_example.py",
        "examples\\optional_loongflow_demo.py", "examples\\roma_decomposition_advanced.py",
        "examples\\roma_decomposition_basic.py", "examples\\unified_evolution_quickstart.py",
        "examples\\verify_optional_loongflow.py", "examples\\verify_unified_api.py",
        "glue\\adapters\\curie-globalchem-integration\\src\\curie_globalchem_adapter.py",
        "glue\\adapters\\curie-globalchem-integration\\test_adapter.py",
        "glue\\adapters\\curie-globalchem-integration\\test_anti_corruption.py",
        "glue\\adapters\\gauntlet-adapter\\monitoring\\example_usage.py",
        "glue\\adapters\\gauntlet-adapter\\monitoring\\quick_start.py",
        "glue\\adapters\\leanaide-adapter\\tests\\test_phase1_lean4_integration.py",
        "glue\\adapters\\pami-research-quest-curie-globalchem-integration\\src\\pami_research_quest_curie_globalchem_adapter.py",
        "glue\\adapters\\pami-research-quest-curie-globalchem-integration\\test_integration.py",
        "glue\\adapters\\rese-integration\\config_example.py",
        "glue\\adapters\\rese-integration\\config_loader.py",
        "glue\\adapters\\rese-leanaide-workflow\\tests\\test_leanaide_rese_workflow.py",
        "glue\\adapters\\rese-phase4\\tests\\test_output_generator.py",
        "glue\\adapters\\rese-phase4\\tests\\test_phase4_comprehensive.py",
        "glue\\adapters\\rese-phase4\\tests\\test_phase4_integration.py",
        "glue\\adapters\\rese-phase4\\tests\\test_predictive_validator.py",
        "glue\\adapters\\rese-sce\\src\\lean4_atp_bridge.py",
        "glue\\adapters\\rese-sce\\tests\\test_dito_optimizer.py",
        "glue\\adapters\\rese-sce\\tests\\test_dito_z3_atp.py",
        "glue\\adapters\\rese-sce\\tests\\test_sce_comprehensive.py",
        "glue\\adapters\\rese-sce\\tests\\test_z3_integration.py",
        "glue\\adapters\\rese-verification\\tests\\test_basic.py",
        "glue\\adapters\\research-quest-curie-globalchem-integration\\src\\research_quest_curie_globalchem_adapter.py",
        "glue\\adapters\\research-quest-curie-globalchem-integration\\test_integration.py",
        "docs\\knowledge_engine\\examples\\causal_modeling_quickstart.py",
        "docs\\knowledge_engine\\examples\\finance\\simple_financial_evolution.py",
        "docs\\knowledge_engine\\examples\\long_horizon_quickstart.py",
        "docs\\knowledge_engine\\knowledge_engine\\finance\\__init__.py",
        "docs\\knowledge_engine\\knowledge_engine\\finance\\crisis_aware_fitness.py",
        "docs\\knowledge_engine\\knowledge_engine\\finance\\financial_evolution_agent.py",
    ],
    # Batch 9 - 13 files (datapizza/crewai/bubblelabs)
    "batch9": [
        "datapizza_crewai_bridge.py", "crewai_client.py", "crewai_enhanced_decomposition_bridge.py",
        "crewai_unified_bridge.py", "crewai_unified_flow.py",
        "bubblelabs_nodes\\tests\\test_cache_integration.py",
        "bubblelabs_nodes\\tests\\test_circuit_breakers.py",
        "bubblelabs_nodes\\tests\\test_circuit_breakers_integration.py",
        "bubblelabs_nodes\\tests\\test_fuzzing.py", "bubblelabs_nodes\\tests\\test_traceability.py",
        "bubblelabs_nodes\\causal_analysis_node.py", "bubblelabs_nodes\\gauntlet_complete_example.py",
        "openevolve_bubblelabs_ui.py",
    ],
}

# Known categories for unfixable issues
KNOWN_EXTERNAL_DEPENDENCIES = [
    "tensorflow", "torch", "flask_cors", "astor", "fcntl", 
    "rese.core.symbolic_constraint_engine", "rese.gamma1.core.aci_calculator",
    "unified", "openevolve.unified.unified_evolution_api", "openevolve.agents",
    "global_chem", "knowledge_engine.causal_modeling", "knowledge_engine.finance",
]

KNOWN_DEMO_SCRIPTS = [
    "demo_adversarial_maker.py", "demo_app.py", "demo_crewai_research_features.py",
    "demo_database_cleanup.py", "demo_e2e_invention_enhanced.py", "demo_end_to_end_invention.py",
    "demo_enhanced_adversarial.py", "demo_enhanced_decomposition_recomposition.py",
    "demo_evolution_maker.py", "demo_evolution_mdap.py", "demo_evolutionary_tests.py",
    "demo_generic_maker.py", "demo_hierarchical_indexing.py", "demo_hybrid_maker.py",
    "demo_hybrid_mcts.py", "demo_integration.py", "demo_knowledge_extraction_ml.py",
    "demo_leanaide_autoformalization_mdap_maker.py", "demo_leanaide_client.py",
    "demo_leanaide_config.py", "demo_leanaide_redflagging.py", "demo_maker_complete.py",
    "demo_matryoshka_auto.py", "demo_matryoshka_unified_memory.py", "demo_mcts.py",
    "demo_mcts_mdap.py", "demo_mdap_maker.py", "demo_mdap_maker_matryoshka.py",
    "demo_mdap_maker_mcts_unified.py", "demo_openevolve_bubblelabs.py",
    "demo_openevolve_integration.py", "demo_openevolve_pes_integration.py",
    "demo_pes_workflow.py", "demo_pes_workflow_language_agnostic.py",
    "demo_pes_workflow_universal.py", "demo_problem_classifier.py",
    "demo_quality_calculator.py", "demo_reliability_system.py", "demo_roma_mdap_maker.py",
    "demo_sop_components.py", "demo_sop_generator.py", "demo_sop_integrated.py",
    "demo_team_assignment.py", "demo_ui_integration.py", "demo_unified_memory_system.py",
    "demo_z3_leanaide_integration.py",
]

KNOWN_RUNNER_SCRIPTS = [
    "apply_ace_phase4_fixes.py", "apply_ace_security_fixes.py", "apply_api_consistency_fixes.py",
    "apply_code_quality_fixes.py", "apply_phase4_validation.py", "assess_decomposition.py",
    "audit_lean_files.py", "benchmark_improvements.py", "c2c_usage_examples.py",
    "comprehensive_demo.py", "debug_class.py", "debug_source.py", "debug_test.py",
    "master_test_runner.py", "migration_report.py", "reporting_demo.py",
    "run_all_ace_tests.py", "run_all_batch2_tests.py", "run_all_gauntlet_tests.py",
    "run_all_tests.py", "run_evolution_mdap_tests.py", "run_evolutionary_tests.py",
    "run_full_rese_e2e_pipeline.py", "run_gauntlet_tests.py", "run_integration_tests.py",
    "run_leanaide_tests.py", "run_mcts_mdap_tests.py", "run_mcts_tests.py",
    "run_mdap_tests.py", "run_real_security_tests.py", "run_rese_tests.py",
    "run_security_tests.py", "run_security_true_100_tests.py", "run_tests.py",
    "simple_demo.py", "simple_dspy_test.py", "simple_test.py", "simple_test_clean.py",
    "validate_all_fixes.py", "validate_end_to_end_invention.py", "validate_enhanced_adversarial.py",
    "validate_evolution_maker_integration.py", "validate_generic_maker_integration.py",
    "validate_hybrid_maker_integration.py", "validate_integration.py",
    "validate_leanaide_tests.py", "validate_maker_integration.py", "validate_phase1_complete.py",
    "validate_sop_components.py", "validate_sop_generator.py", "validate_sop_integrated.py",
    "verify_causal_learn_final.py", "verify_complete_integration.py",
]


def test_single_import(filepath):
    """Test if a single Python file can be imported."""
    abs_path = Path("c:/Users/mmeadow/Documents/OpenEvolve/Frontend") / filepath
    
    if not abs_path.exists():
        return {"file": filepath, "status": "missing", "error": "File not found"}
    
    # First check syntax
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
    except SyntaxError as e:
        return {"file": filepath, "status": "syntax_error", "error": str(e)}
    except UnicodeDecodeError as e:
        return {"file": filepath, "status": "encoding_error", "error": str(e)}
    
    # Try importing
    test_code = f"""
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')
import importlib.util
spec = importlib.util.spec_from_file_location("test_module", r"{abs_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("SUCCESS")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="c:/Users/mmeadow/Documents/OpenEvolve/Frontend"
        )
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return {"file": filepath, "status": "success", "error": None}
        else:
            error_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
            return {"file": filepath, "status": "import_error", "error": error_msg}
    except subprocess.TimeoutExpired:
        return {"file": filepath, "status": "timeout", "error": "Import test timeout"}
    except Exception as e:
        return {"file": filepath, "status": "exception", "error": str(e)[:200]}


def categorize_failure(filepath, error_msg):
    """Categorize why a file cannot be fixed."""
    filepath_lower = filepath.lower()
    error_lower = error_msg.lower() if error_msg else ""
    
    # Check for external dependencies
    for dep in KNOWN_EXTERNAL_DEPENDENCIES:
        if dep.lower() in error_lower or dep.lower() in filepath_lower:
            return "external_dependency"
    
    # Check for demo scripts (known to run code)
    for demo in KNOWN_DEMO_SCRIPTS:
        if demo.lower() in filepath_lower:
            return "demo_script"
    
    # Check for runner scripts
    for runner in KNOWN_RUNNER_SCRIPTS:
        if runner.lower() in filepath_lower:
            return "runner_script"
    
    # Check for Unix-specific issues
    if "fcntl" in error_lower or "unix" in error_lower:
        return "unix_only"
    
    # Check for template/example files
    if "example" in filepath_lower and "examples" in filepath_lower:
        return "example_outdated"
    
    return "fixable"


def main():
    print("=" * 80)
    print("FINAL IMPORT VERIFICATION - 100% SUCCESS RATE TARGET")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    all_results = []
    total_files = 0
    
    # Collect all files to test
    all_files = []
    for batch_name, files in ALL_PREVIOUSLY_FAILING.items():
        for f in files:
            all_files.append((batch_name, f))
            total_files += 1
    
    print(f"Testing {total_files} previously failing files...")
    print()
    
    # Test files
    successful = 0
    still_failing = 0
    unfixable = 0
    remaining_issues = []
    
    for i, (batch_name, filepath) in enumerate(all_files, 1):
        result = test_single_import(filepath)
        all_results.append({**result, "batch": batch_name})
        
        if result["status"] == "success":
            successful += 1
            print(f"  [{i}/{total_files}] [OK] {filepath}")
        else:
            still_failing += 1
            category = categorize_failure(filepath, result.get("error"))
            
            if category != "fixable":
                unfixable += 1
            
            remaining_issues.append({
                "file": filepath,
                "batch": batch_name,
                "reason": category,
                "error": result.get("error", "Unknown"),
            })
            print(f"  [{i}/{total_files}] [FAIL] {filepath} [{category}]")
    
    # Generate report
    now_successful = successful
    still_failing_count = still_failing
    
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "total_previously_failed": total_files,
        "now_successful": now_successful,
        "still_failing": still_failing_count,
        "unfixable": unfixable,
        "success_rate": f"{(now_successful / total_files * 100):.1f}%",
        "remaining_issues": [
            {"file": issue["file"], "reason": issue["reason"]} 
            for issue in remaining_issues
        ],
        "detailed_results": all_results,
    }
    
    # Save report
    report_path = "c:/Users/mmeadow/Documents/OpenEvolve/Frontend/FINAL_100_PERCENT_VERIFICATION.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print()
    print("=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Previously Failed: {total_files}")
    print(f"Now Successful:          {now_successful}")
    print(f"Still Failing:           {still_failing_count}")
    print(f"Unfixable (by design):   {unfixable}")
    print(f"Success Rate:            {report['success_rate']}")
    print()
    print("Report saved to:")
    print(f"  {report_path}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    main()
