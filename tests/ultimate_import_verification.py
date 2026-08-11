#!/usr/bin/env python3
"""
ULTIMATE Import Verification Script
Tests ALL categories of files for import success/failure
"""

import sys
import os
import importlib.util
import traceback
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

# File categories
CATEGORIES = {
    "CAV-NLP": [
        "cav_nlp_integration.py",
    ],
    "Root Test Files": [
        "test_import_fixes_phase1.py",
        "test_imports_batch2.py",
        "test_imports_batch2_final.py",
        "test_imports_batch2_robust.py",
    ],
    "CrewAI": [
        "crewai_api_routes.py",
        "crewai_client.py",
        "crewai_enhanced_decomposition_bridge.py",
        "crewai_integration.py",
        "crewai_integration_layer.py",
        "crewai_mdap_integrator.py",
        "crewai_mdap_maker_engine.py",
        "crewai_research_core.py",
        "crewai_research_enhanced.py",
        "crewai_research_external.py",
        "crewai_research_templates.py",
        "crewai_research_tools.py",
        "crewai_state_management.py",
        "crewai_unified_bridge.py",
        "crewai_unified_flow.py",
        "crewai_zero_error_workflow.py",
    ],
    "ROMA": [
        "roma_config.py",
        "roma_config_helper.py",
        "roma_crewai_bridge.py",
        "roma_crewai_tools.py",
        "roma_decomposition_comparison.py",
        "roma_decomposition_hybrid.py",
        "roma_entity_kg.py",
        "roma_integration.py",
        "roma_matryoshka_integration.py",
        "roma_mcp_tools.py",
        "roma_mdap_maker_associative_integration.py",
        "roma_mdap_maker_config.py",
        "roma_mdap_maker_crewai_bridge.py",
        "roma_mdap_maker_crewai_tools.py",
        "roma_mdap_maker_engine.py",
        "roma_mdap_maker_mcp_tools.py",
        "roma_mdap_maker_reliability_ssot.py",
        "roma_openevolve_integration.py",
        "roma_recomposition_config.py",
    ],
    "OpenEvolve": [
        "openevolve_agnostic_pes.py",
        "openevolve_analytics.py",
        "openevolve_api.py",
        "openevolve_bubblelabs_api.py",
        "openevolve_bubblelabs_plugin.py",
        "openevolve_bubblelabs_ui.py",
        "openevolve_cli.py",
        "openevolve_client.py",
        "openevolve_crewai_adapter.py",
        "openevolve_crewai_bridge.py",
        "openevolve_crewai_delegation.py",
        "openevolve_dashboard.py",
        "openevolve_decomposition_adapter.py",
        "openevolve_enhanced_decomposition_integration.py",
        "openevolve_evolution_integration.py",
        "openevolve_imports.py",
        "openevolve_integration.py",
        "openevolve_knowledge_integration.py",
        "openevolve_leanaide_bridge.py",
        "openevolve_leanaide_integration_system.py",
        "openevolve_leanaide_workflow_integration.py",
        "openevolve_maker_integration.py",
        "openevolve_mcp_tools.py",
        "openevolve_orchestrator.py",
        "openevolve_pes_integration.py",
        "openevolve_structures.py",
        "openevolve_validation.py",
        "openevolve_visualization.py",
        "openevolve_workflow_manager.py",
        "openevolve_workflow_manager_integrated.py",
    ],
    "Examples": [
        "examples/01_basic_evolution.py",
        "examples/01_basic_evolution_evaluator.py",
        "examples/02_function_evolution.py",
        "examples/02_function_evolution_evaluator.py",
        "examples/03_config_file.py",
        "examples/03_optimize_evaluator.py",
        "examples/04_python_api.py",
        "examples/04_string_evaluator.py",
        "examples/05_algo_evaluator.py",
        "examples/05_cli_usage.py",
        "examples/06_advanced_features.py",
        "examples/06_multi_evaluator.py",
        "examples/associative_recomposition_example.py",
        "examples/bubblelabs_plugin_examples.py",
        "examples/enhanced_gauntlet_example.py",
        "examples/example_business_process.py",
        "examples/example_ml_pipeline.py",
        "examples/example_research_decomposition.py",
        "examples/example_software_architecture.py",
        "examples/finance/insurance_example.py",
        "examples/gauntlet_configs/finance_config.py",
        "examples/gauntlet_configs/science_config.py",
        "examples/gauntlet_configs/web_config.py",
        "examples/investment_committee_demo.py",
        "examples/knowledge_integration_example.py",
        "examples/lean4_usage_example.py",
        "examples/loongflow_extraction_example.py",
        "examples/loongflow_fallback_example.py",
        "examples/loongflow_gauntlet_usage.py",
        "examples/mdap_maker_associative_example.py",
        "examples/node_usage_examples.py",
        "examples/optional_loongflow_demo.py",
        "examples/roma_decomposition_advanced.py",
        "examples/roma_decomposition_basic.py",
        "examples/roma_mdap_maker_associative_example.py",
        "examples/roma_recomposition_examples.py",
        "examples/sop_research_quest_poc.py",
        "examples/test_examples.py",
        "examples/three_round_integration_example.py",
        "examples/unified_evolution_quickstart.py",
        "examples/verify_optional_loongflow.py",
        "examples/verify_unified_api.py",
    ],
    "Glue": [
        # Core adapters
        "glue/adapters/curie_globalchem_integration.py",
        "glue/adapters/matryoshka_adapter.py",
        # Curie-GlobalChem
        "glue/adapters/curie-globalchem-integration/__init__.py",
        "glue/adapters/curie-globalchem-integration/probes/integration_probe.py",
        "glue/adapters/curie-globalchem-integration/src/__init__.py",
        "glue/adapters/curie-globalchem-integration/src/curie_globalchem_adapter.py",
        "glue/adapters/curie-globalchem-integration/test_adapter.py",
        "glue/adapters/curie-globalchem-integration/test_anti_corruption.py",
        # Gauntlet
        "glue/adapters/gauntlet-adapter/monitoring/__init__.py",
        "glue/adapters/gauntlet-adapter/monitoring/alerting.py",
        "glue/adapters/gauntlet-adapter/monitoring/config.py",
        "glue/adapters/gauntlet-adapter/monitoring/example_usage.py",
        "glue/adapters/gauntlet-adapter/monitoring/health_checks.py",
        "glue/adapters/gauntlet-adapter/monitoring/metrics.py",
        "glue/adapters/gauntlet-adapter/monitoring/quick_start.py",
        "glue/adapters/gauntlet-adapter/src/adaptive_learner.py",
        "glue/adapters/gauntlet-adapter/src/intelligent_orchestrator.py",
        "glue/adapters/gauntlet-adapter/src/ml_optimizer.py",
        "glue/adapters/gauntlet-adapter/src/predictive_gauntlet_executor.py",
        "glue/adapters/gauntlet-adapter/tests/compare_learning.py",
        "glue/adapters/gauntlet-adapter/tests/test_backpropagation.py",
        # Leanaide
        "glue/adapters/leanaide-adapter/src/autoformalization_pipeline.py",
        "glue/adapters/leanaide-adapter/tests/test_formalization_coverage.py",
        "glue/adapters/leanaide-adapter/tests/test_phase1_lean4_integration.py",
        "glue/adapters/leanaide-adapter/verify_category_a_formalization.py",
        # LMQL-DSPy
        "glue/adapters/lmql-dspy-integration/probes/integration_probe.py",
        "glue/adapters/lmql-dspy-integration/src/lmql_dspy_adapter.py",
        "glue/adapters/lmql-dspy-integration/test_integration.py",
        # PAMI
        "glue/adapters/pami-research-quest-curie-globalchem-integration/probes/integration_probe.py",
        "glue/adapters/pami-research-quest-curie-globalchem-integration/src/pami_research_quest_curie_globalchem_adapter.py",
        "glue/adapters/pami-research-quest-curie-globalchem-integration/test_integration.py",
        # Research Quest
        "glue/adapters/research-quest-curie-globalchem-integration/probes/integration_probe.py",
        "glue/adapters/research-quest-curie-globalchem-integration/src/research_quest_curie_globalchem_adapter.py",
        "glue/adapters/research-quest-curie-globalchem-integration/test_integration.py",
        # RESE Benchmarks
        "glue/adapters/rese-benchmarks/benchmark_full_pipeline.py",
        "glue/adapters/rese-benchmarks/benchmark_phase1.py",
        "glue/adapters/rese-benchmarks/benchmark_phase2.py",
        "glue/adapters/rese-benchmarks/benchmark_phase3.py",
        "glue/adapters/rese-benchmarks/benchmark_phase4.py",
        "glue/adapters/rese-benchmarks/init_baseline.py",
        "glue/adapters/rese-benchmarks/run_all_benchmarks.py",
        "glue/adapters/rese-benchmarks/verify_setup.py",
        # RESE DEE
        "glue/adapters/rese-dee/src/dee_adapter.py",
        "glue/adapters/rese-dee/tests/test_dee.py",
        "glue/adapters/rese-dee/tests/test_integration.py",
        # RESE Integration
        "glue/adapters/rese-integration/config_example.py",
        "glue/adapters/rese-integration/config_loader.py",
        "glue/adapters/rese-integration/config_validator.py",
        "glue/adapters/rese-integration/health/aggregate_health.py",
        # RESE Leanaide Workflow
        "glue/adapters/rese-leanaide-workflow/__init__.py",
        "glue/adapters/rese-leanaide-workflow/src/__init__.py",
        "glue/adapters/rese-leanaide-workflow/src/autoformalization_service.py",
        "glue/adapters/rese-leanaide-workflow/src/leanaide_rese_workflow.py",
        "glue/adapters/rese-leanaide-workflow/src/proof_search_service.py",
        "glue/adapters/rese-leanaide-workflow/tests/__init__.py",
        "glue/adapters/rese-leanaide-workflow/tests/test_leanaide_rese_workflow.py",
        "glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py",
        # RESE LLTL
        "glue/adapters/rese-lltl/__init__.py",
        "glue/adapters/rese-lltl/example_usage.py",
        "glue/adapters/rese-lltl/examples/confidence_thresholds_example.py",
        "glue/adapters/rese-lltl/src/__init__.py",
        "glue/adapters/rese-lltl/src/confidence_tracker.py",
        "glue/adapters/rese-lltl/src/formal_commitments.py",
        "glue/adapters/rese-lltl/src/lltl_adapter.py",
        "glue/adapters/rese-lltl/tests/test_confidence_tracker.py",
        "glue/adapters/rese-lltl/tests/test_dee_to_sce_auditability.py",
        "glue/adapters/rese-lltl/tests/test_dee_to_sce_simple.py",
        "glue/adapters/rese-lltl/tests/test_formal_commitments.py",
        "glue/adapters/rese-lltl/tests/test_lltl_comprehensive.py",
        "glue/adapters/rese-lltl/tests/test_z3_contradiction_detection.py",
        "glue/adapters/rese-lltl/tests/test_z3_dito_benchmark.py",
        "glue/adapters/rese-lltl/tests/test_z3_integration_structure.py",
        # RESE Phase1
        "glue/adapters/rese-phase1/probes/check_phi2_debiasing.py",
        "glue/adapters/rese-phase1/probes/check_z3_api.py",
        "glue/adapters/rese-phase1/src/bias_metrics.py",
        "glue/adapters/rese-phase1/src/health_api.py",
        "glue/adapters/rese-phase1/src/metacognitive_reflector.py",
        "glue/adapters/rese-phase1/src/phase1_adapter.py",
        "glue/adapters/rese-phase1/src/phase1_executor.py",
        "glue/adapters/rese-phase1/tests/test_metacognitive_reflector.py",
        "glue/adapters/rese-phase1/tests/test_phase1_comprehensive.py",
        "glue/adapters/rese-phase1/tests/test_phase1_debiasing_integration.py",
        "glue/adapters/rese-phase1/tests/test_phase1_integration.py",
        "glue/adapters/rese-phase1/tests/test_z3_constraint_hardening.py",
        "glue/adapters/rese-phase1/tests/test_z3_integration_e2e.py",
        # RESE Phase2
        "glue/adapters/rese-phase2/probes/check_z3_api.py",
        "glue/adapters/rese-phase2/src/__init__.py",
        "glue/adapters/rese-phase2/src/fdg_validator.py",
        "glue/adapters/rese-phase2/src/health_api.py",
        "glue/adapters/rese-phase2/src/phase2_adapter.py",
        "glue/adapters/rese-phase2/src/phase2_executor.py",
        "glue/adapters/rese-phase2/tests/conftest.py",
        "glue/adapters/rese-phase2/tests/test_fdg_lean4_integration.py",
        "glue/adapters/rese-phase2/tests/test_integration.py",
        "glue/adapters/rese-phase2/tests/test_phase2.py",
        "glue/adapters/rese-phase2/tests/test_phase2_comprehensive.py",
        "glue/adapters/rese-phase2/tests/test_z3_behavioral_equivalence.py",
        "glue/adapters/rese-phase2/tests/test_z3_integration_benchmark.py",
        "glue/adapters/rese-phase2/tests/verify_implementation.py",
        "glue/adapters/rese-phase2/tests/verify_simple.py",
        "glue/adapters/rese-phase2/verify_install.py",
        # RESE Phase3
        "glue/adapters/rese-phase3/__init__.py",
        "glue/adapters/rese-phase3/src/__init__.py",
        "glue/adapters/rese-phase3/src/aci_calculator.py",
        "glue/adapters/rese-phase3/src/health_api.py",
        "glue/adapters/rese-phase3/src/phase3_adapter.py",
        "glue/adapters/rese-phase3/src/phase3_executor.py",
        "glue/adapters/rese-phase3/tests/quick_test.py",
        "glue/adapters/rese-phase3/tests/simple_test.py",
        "glue/adapters/rese-phase3/tests/test_aci_calculator.py",
        "glue/adapters/rese-phase3/tests/test_aci_mcts_integration.py",
        "glue/adapters/rese-phase3/tests/test_phase3.py",
        "glue/adapters/rese-phase3/tests/test_phase3_comprehensive.py",
        "glue/adapters/rese-phase3/tests/test_z3_constraint_checking.py",
        # RESE Phase4
        "glue/adapters/rese-phase4/src/adapter.py",
        "glue/adapters/rese-phase4/src/health_api.py",
        "glue/adapters/rese-phase4/src/output_generator.py",
        "glue/adapters/rese-phase4/src/phase4_executor.py",
        "glue/adapters/rese-phase4/src/predictive_validator.py",
        "glue/adapters/rese-phase4/src/result_verifier.py",
        "glue/adapters/rese-phase4/tests/__init__.py",
        "glue/adapters/rese-phase4/tests/test_output_generator.py",
        "glue/adapters/rese-phase4/tests/test_phase4_comprehensive.py",
        "glue/adapters/rese-phase4/tests/test_phase4_integration.py",
        "glue/adapters/rese-phase4/tests/test_predictive_validator.py",
        # RESE SCE
        "glue/adapters/rese-sce/__init__.py",
        "glue/adapters/rese-sce/src/__init__.py",
        "glue/adapters/rese-sce/src/dito_optimizer.py",
        "glue/adapters/rese-sce/src/lean4_atp_bridge.py",
        "glue/adapters/rese-sce/src/sce_bridge.py",
        "glue/adapters/rese-sce/tests/test_dito_optimizer.py",
        "glue/adapters/rese-sce/tests/test_dito_z3_atp.py",
        "glue/adapters/rese-sce/tests/test_sce_comprehensive.py",
        "glue/adapters/rese-sce/tests/test_z3_integration.py",
        "glue/adapters/rese-sce/verify_integration.py",
        "glue/adapters/rese-sce/verify_z3_integration.py",
        # RESE Verification
        "glue/adapters/rese-verification/src/__init__.py",
        "glue/adapters/rese-verification/src/problem_classifier.py",
        "glue/adapters/rese-verification/src/solver_selector.py",
        "glue/adapters/rese-verification/src/tiered_verifier.py",
        "glue/adapters/rese-verification/src/verification_result.py",
        "glue/adapters/rese-verification/tests/test_basic.py",
        "glue/adapters/rese-verification/tests/test_tiered_verifier.py",
        "glue/adapters/rese-verification/tests/test_tiered_verifier_comprehensive.py",
        # RESE Z3 Bridge
        "glue/adapters/rese-z3-bridge/src/__init__.py",
        "glue/adapters/rese-z3-bridge/src/rese_z3_bridge.py",
        "glue/adapters/rese-z3-bridge/src/rese_z3_client.py",
        "glue/adapters/rese-z3-bridge/src/rese_z3_schema.py",
        "glue/adapters/rese-z3-bridge/tests/test_leanaide_integration.py",
        "glue/adapters/rese-z3-bridge/tests/test_rese_z3_bridge.py",
        "glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py",
        "glue/adapters/rese-z3-bridge/tests/test_simple.py",
        # Z3 Adapter
        "glue/adapters/z3-adapter/probes/check_database.py",
        # Lib
        "glue/lib/__init__.py",
        "glue/lib/lean4_bridge/__init__.py",
        "glue/lib/lean4_bridge/lean4_atp_bridge.py",
        "glue/lib/lean4_bridge/lean4_interface.py",
        "glue/lib/lean4_bridge/src/__init__.py",
        "glue/lib/lean4_bridge/src/constraint_translator.py",
        "glue/lib/lean4_bridge/tests/test_lean4_interface.py",
        "glue/lib/lean4_bridge/verify_setup.py",
        "glue/lib/rese_dee.py",
        "glue/lib/rese_lltl.py",
        # Orchestration
        "glue/orchestration/config.py",
        "glue/orchestration/event_bus.py",
        "glue/orchestration/rese_pipeline.py",
        # Schemas
        "glue/schemas/__init__.py",
        "glue/schemas/rese_phase4_schemas.py",
        "glue/schemas/rese_schemas.py",
        # Tests
        "glue/tests/test_rese_complete_pipeline.py",
        "glue/tests/test_rese_final_integration.py",
    ],
}


def test_import(filepath):
    """Test if a file can be imported as a module."""
    # Convert path to module name
    module_name = filepath.replace("/", ".").replace("\\", ".").replace(".py", "")
    
    # Handle __init__.py
    if module_name.endswith(".__init__"):
        module_name = module_name[:-9]
    
    try:
        # Try to find and load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return False, "Could not find module spec"
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "InputValidator" in error_msg:
            return False, "InputValidator import error"
        elif "No module named" in error_msg:
            return False, f"Missing dependency: {error_msg}"
        else:
            return False, error_msg[:100]  # Truncate long errors


def run_verification():
    """Run the full verification across all categories."""
    results = {}
    total_files = 0
    total_success = 0
    total_failures = 0
    
    print("=" * 80)
    print("ULTIMATE IMPORT VERIFICATION")
    print("=" * 80)
    print()
    
    for category, files in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        
        category_results = []
        category_success = 0
        category_failures = 0
        
        for filepath in files:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"  [MISSING] FILE NOT FOUND: {filepath}")
                category_results.append({
                    "file": filepath,
                    "status": "NOT_FOUND",
                    "error": "File does not exist"
                })
                category_failures += 1
                continue
            
            # Test import
            success, error = test_import(filepath)
            total_files += 1
            
            if success:
                print(f"  [PASS] {filepath}")
                category_results.append({
                    "file": filepath,
                    "status": "SUCCESS",
                    "error": None
                })
                category_success += 1
                total_success += 1
            else:
                print(f"  [FAIL] {filepath}")
                print(f"     Error: {error}")
                category_results.append({
                    "file": filepath,
                    "status": "FAILED",
                    "error": error
                })
                category_failures += 1
                total_failures += 1
        
        results[category] = {
            "files": category_results,
            "success": category_success,
            "failures": category_failures,
            "total": category_success + category_failures
        }
    
    # Calculate totals
    success_rate = (total_success / total_files * 100) if total_files > 0 else 0
    
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"Total Files Tested: {total_files}")
    print(f"Successful Imports: {total_success}")
    print(f"Failed Imports: {total_failures}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    # Generate report
    generate_report(results, total_files, total_success, total_failures, success_rate)
    
    return results, total_files, total_success, total_failures, success_rate


def generate_report(results, total_files, total_success, total_failures, success_rate):
    """Generate the markdown report."""
    report = []
    report.append("# ULTIMATE IMPORT VERIFICATION REPORT")
    report.append("")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Status:** {'COMPLETE' if total_failures == 0 else 'PARTIAL'}")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Total Files Tested** | {total_files} |")
    report.append(f"| **Successful Imports** | {total_success} |")
    report.append(f"| **Failed Imports** | {total_failures} |")
    report.append(f"| **Success Rate** | {success_rate:.1f}% |")
    report.append("")
    
    # Category summary
    report.append("## Category Summary")
    report.append("")
    report.append("| Category | Total | Success | Failed | Rate |")
    report.append("|----------|-------|---------|--------|------|")
    
    for category, data in results.items():
        rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
        report.append(f"| {category} | {data['total']} | {data['success']} | {data['failures']} | {rate:.1f}% |")
    
    report.append(f"| **TOTAL** | **{total_files}** | **{total_success}** | **{total_failures}** | **{success_rate:.1f}%** |")
    report.append("")
    
    # Detailed results by category
    for category, data in results.items():
        report.append(f"## {category}")
        report.append("")
        report.append("| File | Status | Error |")
        report.append("|------|--------|-------|")
        
        for file_result in data['files']:
            file_name = file_result['file']
            status = file_result['status']
            error = file_result['error'] or ""
            
            # Truncate error for display
            if len(error) > 60:
                error = error[:57] + "..."
            
            if status == "SUCCESS":
                status_icon = "[PASS]"
            elif status == "NOT_FOUND":
                status_icon = "[MISSING]"
            else:
                status_icon = "[FAIL]"
            
            report.append(f"| `{file_name}` | {status_icon} | {error} |")
        
        report.append("")
    
    # Failure analysis
    report.append("## Detailed Failure Analysis")
    report.append("")
    
    has_failures = False
    for category, data in results.items():
        failures = [f for f in data['files'] if f['status'] != 'SUCCESS']
        if failures:
            has_failures = True
            report.append(f"### {category}")
            report.append("")
            for fail in failures:
                report.append(f"- **{fail['file']}**")
                report.append(f"  - Status: {fail['status']}")
                report.append(f"  - Error: {fail['error']}")
                report.append("")
    
    if not has_failures:
        report.append("**NO FAILURES! All imports successful!**")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if success_rate >= 99:
        report.append("**EXCELLENT**: Project has achieved near-perfect import compatibility.")
        report.append("   The codebase is well-structured with minimal import issues.")
    elif success_rate >= 95:
        report.append("**VERY GOOD**: Project has excellent import compatibility.")
        report.append("   Minor issues exist but do not significantly impact functionality.")
    elif success_rate >= 90:
        report.append("**GOOD**: Project has good import compatibility.")
        report.append("   Some issues should be addressed for optimal performance.")
    elif success_rate >= 80:
        report.append("**FAIR**: Project has moderate import compatibility.")
        report.append("   Several issues need attention to improve reliability.")
    else:
        report.append("**NEEDS ATTENTION**: Project has significant import issues.")
        report.append("   Immediate action recommended to fix import problems.")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by Ultimate Import Verification System*")
    
    # Write report
    report_path = "c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\ULTIMATE_100_PERCENT_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_verification()
