#!/usr/bin/env python3
"""Test imports for Python files in examples/, glue/, and docs/ directories."""

import json
import sys
import os
import importlib.util
import contextlib
from pathlib import Path

# Add project root to path
project_root = Path(r"c:\Users\mmeadow\Documents\OpenEvolve\Frontend")
sys.path.insert(0, str(project_root))

# Suppress stdout/stderr during imports to prevent execution output
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# List all files to test
files_to_test = [
    # examples/ directory (42 files)
    r"examples\01_basic_evolution.py",
    r"examples\01_basic_evolution_evaluator.py",
    r"examples\02_function_evolution.py",
    r"examples\02_function_evolution_evaluator.py",
    r"examples\03_config_file.py",
    r"examples\03_optimize_evaluator.py",
    r"examples\04_python_api.py",
    r"examples\04_string_evaluator.py",
    r"examples\05_algo_evaluator.py",
    r"examples\05_cli_usage.py",
    r"examples\06_advanced_features.py",
    r"examples\06_multi_evaluator.py",
    r"examples\associative_recomposition_example.py",
    r"examples\bubblelabs_plugin_examples.py",
    r"examples\enhanced_gauntlet_example.py",
    r"examples\example_business_process.py",
    r"examples\example_ml_pipeline.py",
    r"examples\example_research_decomposition.py",
    r"examples\example_software_architecture.py",
    r"examples\finance\insurance_example.py",
    r"examples\gauntlet_configs\finance_config.py",
    r"examples\gauntlet_configs\science_config.py",
    r"examples\gauntlet_configs\web_config.py",
    r"examples\investment_committee_demo.py",
    r"examples\knowledge_integration_example.py",
    r"examples\lean4_usage_example.py",
    r"examples\loongflow_extraction_example.py",
    r"examples\loongflow_fallback_example.py",
    r"examples\loongflow_gauntlet_usage.py",
    r"examples\mdap_maker_associative_example.py",
    r"examples\node_usage_examples.py",
    r"examples\optional_loongflow_demo.py",
    r"examples\roma_decomposition_advanced.py",
    r"examples\roma_decomposition_basic.py",
    r"examples\roma_mdap_maker_associative_example.py",
    r"examples\roma_recomposition_examples.py",
    r"examples\sop_research_quest_poc.py",
    r"examples\test_examples.py",
    r"examples\three_round_integration_example.py",
    r"examples\unified_evolution_quickstart.py",
    r"examples\verify_optional_loongflow.py",
    r"examples\verify_unified_api.py",
    
    # glue/ directory (172 files)
    r"glue\__init__.py",
    r"glue\adapters\__init__.py",
    r"glue\adapters\curie-globalchem-integration\probes\integration_probe.py",
    r"glue\adapters\curie-globalchem-integration\src\curie_globalchem_adapter.py",
    r"glue\adapters\curie-globalchem-integration\test_adapter.py",
    r"glue\adapters\curie-globalchem-integration\test_anti_corruption.py",
    r"glue\adapters\gauntlet-adapter\monitoring\__init__.py",
    r"glue\adapters\gauntlet-adapter\monitoring\alerting.py",
    r"glue\adapters\gauntlet-adapter\monitoring\config.py",
    r"glue\adapters\gauntlet-adapter\monitoring\example_usage.py",
    r"glue\adapters\gauntlet-adapter\monitoring\health_checks.py",
    r"glue\adapters\gauntlet-adapter\monitoring\metrics.py",
    r"glue\adapters\gauntlet-adapter\monitoring\quick_start.py",
    r"glue\adapters\gauntlet-adapter\src\adaptive_learner.py",
    r"glue\adapters\gauntlet-adapter\src\intelligent_orchestrator.py",
    r"glue\adapters\gauntlet-adapter\src\ml_optimizer.py",
    r"glue\adapters\gauntlet-adapter\src\predictive_gauntlet_executor.py",
    r"glue\adapters\gauntlet-adapter\tests\compare_learning.py",
    r"glue\adapters\gauntlet-adapter\tests\test_backpropagation.py",
    r"glue\adapters\leanaide-adapter\src\autoformalization_pipeline.py",
    r"glue\adapters\leanaide-adapter\tests\test_formalization_coverage.py",
    r"glue\adapters\leanaide-adapter\tests\test_phase1_lean4_integration.py",
    r"glue\adapters\leanaide-adapter\verify_category_a_formalization.py",
    r"glue\adapters\lmql-dspy-integration\probes\integration_probe.py",
    r"glue\adapters\lmql-dspy-integration\src\lmql_dspy_adapter.py",
    r"glue\adapters\lmql-dspy-integration\test_integration.py",
    r"glue\adapters\matryoshka_adapter.py",
    r"glue\adapters\pami-research-quest-curie-globalchem-integration\probes\integration_probe.py",
    r"glue\adapters\pami-research-quest-curie-globalchem-integration\src\pami_research_quest_curie_globalchem_adapter.py",
    r"glue\adapters\pami-research-quest-curie-globalchem-integration\test_integration.py",
    r"glue\adapters\rese-benchmarks\benchmark_full_pipeline.py",
    r"glue\adapters\rese-benchmarks\benchmark_phase1.py",
    r"glue\adapters\rese-benchmarks\benchmark_phase2.py",
    r"glue\adapters\rese-benchmarks\benchmark_phase3.py",
    r"glue\adapters\rese-benchmarks\benchmark_phase4.py",
    r"glue\adapters\rese-benchmarks\init_baseline.py",
    r"glue\adapters\rese-benchmarks\run_all_benchmarks.py",
    r"glue\adapters\rese-benchmarks\verify_setup.py",
    r"glue\adapters\rese-dee\src\dee_adapter.py",
    r"glue\adapters\rese-dee\tests\test_dee.py",
    r"glue\adapters\rese-dee\tests\test_integration.py",
    r"glue\adapters\rese-integration\config_example.py",
    r"glue\adapters\rese-integration\config_loader.py",
    r"glue\adapters\rese-integration\config_validator.py",
    r"glue\adapters\rese-integration\health\aggregate_health.py",
    r"glue\adapters\rese-leanaide-workflow\__init__.py",
    r"glue\adapters\rese-leanaide-workflow\src\__init__.py",
    r"glue\adapters\rese-leanaide-workflow\src\autoformalization_service.py",
    r"glue\adapters\rese-leanaide-workflow\src\leanaide_rese_workflow.py",
    r"glue\adapters\rese-leanaide-workflow\src\proof_search_service.py",
    r"glue\adapters\rese-leanaide-workflow\tests\__init__.py",
    r"glue\adapters\rese-leanaide-workflow\tests\test_leanaide_rese_workflow.py",
    r"glue\adapters\rese-leanaide-workflow\tests\test_leanaide_workflow_comprehensive.py",
    r"glue\adapters\rese-lltl\__init__.py",
    r"glue\adapters\rese-lltl\example_usage.py",
    r"glue\adapters\rese-lltl\examples\confidence_thresholds_example.py",
    r"glue\adapters\rese-lltl\src\__init__.py",
    r"glue\adapters\rese-lltl\src\confidence_tracker.py",
    r"glue\adapters\rese-lltl\src\formal_commitments.py",
    r"glue\adapters\rese-lltl\src\lltl_adapter.py",
    r"glue\adapters\rese-lltl\tests\test_confidence_tracker.py",
    r"glue\adapters\rese-lltl\tests\test_dee_to_sce_auditability.py",
    r"glue\adapters\rese-lltl\tests\test_dee_to_sce_simple.py",
    r"glue\adapters\rese-lltl\tests\test_formal_commitments.py",
    r"glue\adapters\rese-lltl\tests\test_lltl_comprehensive.py",
    r"glue\adapters\rese-lltl\tests\test_z3_contradiction_detection.py",
    r"glue\adapters\rese-lltl\tests\test_z3_dito_benchmark.py",
    r"glue\adapters\rese-lltl\tests\test_z3_integration_structure.py",
    r"glue\adapters\rese-phase1\probes\check_phi2_debiasing.py",
    r"glue\adapters\rese-phase1\probes\check_z3_api.py",
    r"glue\adapters\rese-phase1\src\bias_metrics.py",
    r"glue\adapters\rese-phase1\src\health_api.py",
    r"glue\adapters\rese-phase1\src\metacognitive_reflector.py",
    r"glue\adapters\rese-phase1\src\phase1_adapter.py",
    r"glue\adapters\rese-phase1\src\phase1_executor.py",
    r"glue\adapters\rese-phase1\tests\test_metacognitive_reflector.py",
    r"glue\adapters\rese-phase1\tests\test_phase1_comprehensive.py",
    r"glue\adapters\rese-phase1\tests\test_phase1_debiasing_integration.py",
    r"glue\adapters\rese-phase1\tests\test_phase1_integration.py",
    r"glue\adapters\rese-phase1\tests\test_z3_constraint_hardening.py",
    r"glue\adapters\rese-phase1\tests\test_z3_integration_e2e.py",
    r"glue\adapters\rese-phase2\probes\check_z3_api.py",
    r"glue\adapters\rese-phase2\src\__init__.py",
    r"glue\adapters\rese-phase2\src\fdg_validator.py",
    r"glue\adapters\rese-phase2\src\health_api.py",
    r"glue\adapters\rese-phase2\src\phase2_adapter.py",
    r"glue\adapters\rese-phase2\src\phase2_executor.py",
    r"glue\adapters\rese-phase2\tests\conftest.py",
    r"glue\adapters\rese-phase2\tests\test_fdg_lean4_integration.py",
    r"glue\adapters\rese-phase2\tests\test_integration.py",
    r"glue\adapters\rese-phase2\tests\test_phase2.py",
    r"glue\adapters\rese-phase2\tests\test_phase2_comprehensive.py",
    r"glue\adapters\rese-phase2\tests\test_z3_behavioral_equivalence.py",
    r"glue\adapters\rese-phase2\tests\test_z3_integration_benchmark.py",
    r"glue\adapters\rese-phase2\tests\verify_implementation.py",
    r"glue\adapters\rese-phase2\tests\verify_simple.py",
    r"glue\adapters\rese-phase2\verify_install.py",
    r"glue\adapters\rese-phase3\__init__.py",
    r"glue\adapters\rese-phase3\src\__init__.py",
    r"glue\adapters\rese-phase3\src\aci_calculator.py",
    r"glue\adapters\rese-phase3\src\health_api.py",
    r"glue\adapters\rese-phase3\src\phase3_adapter.py",
    r"glue\adapters\rese-phase3\src\phase3_executor.py",
    r"glue\adapters\rese-phase3\tests\quick_test.py",
    r"glue\adapters\rese-phase3\tests\simple_test.py",
    r"glue\adapters\rese-phase3\tests\test_aci_calculator.py",
    r"glue\adapters\rese-phase3\tests\test_aci_mcts_integration.py",
    r"glue\adapters\rese-phase3\tests\test_phase3.py",
    r"glue\adapters\rese-phase3\tests\test_phase3_comprehensive.py",
    r"glue\adapters\rese-phase3\tests\test_z3_constraint_checking.py",
    r"glue\adapters\rese-phase4\src\adapter.py",
    r"glue\adapters\rese-phase4\src\health_api.py",
    r"glue\adapters\rese-phase4\src\output_generator.py",
    r"glue\adapters\rese-phase4\src\phase4_executor.py",
    r"glue\adapters\rese-phase4\src\predictive_validator.py",
    r"glue\adapters\rese-phase4\src\result_verifier.py",
    r"glue\adapters\rese-phase4\tests\__init__.py",
    r"glue\adapters\rese-phase4\tests\test_output_generator.py",
    r"glue\adapters\rese-phase4\tests\test_phase4_comprehensive.py",
    r"glue\adapters\rese-phase4\tests\test_phase4_integration.py",
    r"glue\adapters\rese-phase4\tests\test_predictive_validator.py",
    r"glue\adapters\rese-sce\__init__.py",
    r"glue\adapters\rese-sce\node_modules\flatted\python\flatted.py",
    r"glue\adapters\rese-sce\src\__init__.py",
    r"glue\adapters\rese-sce\src\dito_optimizer.py",
    r"glue\adapters\rese-sce\src\lean4_atp_bridge.py",
    r"glue\adapters\rese-sce\src\sce_bridge.py",
    r"glue\adapters\rese-sce\tests\test_dito_optimizer.py",
    r"glue\adapters\rese-sce\tests\test_dito_z3_atp.py",
    r"glue\adapters\rese-sce\tests\test_sce_comprehensive.py",
    r"glue\adapters\rese-sce\tests\test_z3_integration.py",
    r"glue\adapters\rese-sce\verify_integration.py",
    r"glue\adapters\rese-sce\verify_z3_integration.py",
    r"glue\adapters\rese-verification\src\__init__.py",
    r"glue\adapters\rese-verification\src\problem_classifier.py",
    r"glue\adapters\rese-verification\src\solver_selector.py",
    r"glue\adapters\rese-verification\src\tiered_verifier.py",
    r"glue\adapters\rese-verification\src\verification_result.py",
    r"glue\adapters\rese-verification\tests\test_basic.py",
    r"glue\adapters\rese-verification\tests\test_tiered_verifier.py",
    r"glue\adapters\rese-verification\tests\test_tiered_verifier_comprehensive.py",
    r"glue\adapters\rese-z3-bridge\src\__init__.py",
    r"glue\adapters\rese-z3-bridge\src\rese_z3_bridge.py",
    r"glue\adapters\rese-z3-bridge\src\rese_z3_client.py",
    r"glue\adapters\rese-z3-bridge\src\rese_z3_schema.py",
    r"glue\adapters\rese-z3-bridge\tests\test_leanaide_integration.py",
    r"glue\adapters\rese-z3-bridge\tests\test_rese_z3_bridge.py",
    r"glue\adapters\rese-z3-bridge\tests\test_rese_z3_comprehensive.py",
    r"glue\adapters\rese-z3-bridge\tests\test_simple.py",
    r"glue\adapters\research-quest-curie-globalchem-integration\probes\integration_probe.py",
    r"glue\adapters\research-quest-curie-globalchem-integration\src\research_quest_curie_globalchem_adapter.py",
    r"glue\adapters\research-quest-curie-globalchem-integration\test_integration.py",
    r"glue\adapters\z3-adapter\probes\check_database.py",
    r"glue\lib\__init__.py",
    r"glue\lib\lean4_bridge\__init__.py",
    r"glue\lib\lean4_bridge\lean4_atp_bridge.py",
    r"glue\lib\lean4_bridge\lean4_interface.py",
    r"glue\lib\lean4_bridge\src\__init__.py",
    r"glue\lib\lean4_bridge\src\constraint_translator.py",
    r"glue\lib\lean4_bridge\tests\test_lean4_interface.py",
    r"glue\lib\lean4_bridge\verify_setup.py",
    r"glue\lib\rese_dee.py",
    r"glue\lib\rese_lltl.py",
    r"glue\orchestration\config.py",
    r"glue\orchestration\event_bus.py",
    r"glue\orchestration\node_modules\flatted\python\flatted.py",
    r"glue\orchestration\rese_pipeline.py",
    r"glue\schemas\__init__.py",
    r"glue\schemas\rese_phase4_schemas.py",
    r"glue\schemas\rese_schemas.py",
    r"glue\tests\test_rese_complete_pipeline.py",
    r"glue\tests\test_rese_final_integration.py",
    
    # docs/ directory (31 files)
    r"docs\agents\__init__.py",
    r"docs\knowledge_engine\examples\causal_modeling_quickstart.py",
    r"docs\knowledge_engine\examples\finance\simple_financial_evolution.py",
    r"docs\knowledge_engine\examples\long_horizon_quickstart.py",
    r"docs\knowledge_engine\examples\unified_evolution_example.py",
    r"docs\knowledge_engine\knowledge_engine\__init__.py",
    r"docs\knowledge_engine\knowledge_engine\ab_testing.py",
    r"docs\knowledge_engine\knowledge_engine\causal_modeling.py",
    r"docs\knowledge_engine\knowledge_engine\config\__init__.py",
    r"docs\knowledge_engine\knowledge_engine\finance\__init__.py",
    r"docs\knowledge_engine\knowledge_engine\finance\crisis_aware_fitness.py",
    r"docs\knowledge_engine\knowledge_engine\finance\financial_evolution_agent.py",
    r"docs\knowledge_engine\knowledge_engine\finance\financial_memory.py",
    r"docs\knowledge_engine\knowledge_engine\finance\schemas.py",
    r"docs\knowledge_engine\knowledge_engine\finance\survivorship_backtester.py",
    r"docs\knowledge_engine\knowledge_engine\integrations\__init__.py",
    r"docs\knowledge_engine\knowledge_engine\integrations\unified_evolution_integration.py",
    r"docs\knowledge_engine\knowledge_engine\meta_learning.py",
    r"docs\knowledge_engine\knowledge_engine\online_learning.py",
    r"docs\knowledge_engine\knowledge_engine\schemas\__init__.py",
    r"docs\knowledge_engine\knowledge_engine\schemas\comparison_results.py",
    r"docs\knowledge_engine\knowledge_engine\schemas\evolutionary_artifacts.py",
    r"docs\knowledge_engine\knowledge_engine\schemas\long_horizon.py",
    r"docs\knowledge_engine\knowledge_engine\tests\test_causal_modeling_integration.py",
    r"docs\knowledge_engine\tests\finance\test_financial_evolution.py",
    r"docs\knowledge_engine\tests\integration\test_optional_loongflow.py",
    r"docs\knowledge_engine\tests\knowledge_engine\test_unified_evolution_integration.py",
    r"docs\knowledge_engine\tests\test_long_horizon_learning.py",
    r"docs\knowledge_engine\verify_loongflow.py",
    r"docs\knowledge_engine\verify_simple.py",
    r"docs\knowledge_engine\verify_unified_integration.py",
]


def test_import(filepath):
    """Test importing a single Python file."""
    full_path = project_root / filepath
    
    if not full_path.exists():
        return False, f"File not found: {filepath}"
    
    try:
        # Convert file path to module name
        module_path = str(filepath).replace("\\", ".").replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        
        # Handle __init__.py - import the package instead
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]
        
        # Try to import using importlib with suppressed output
        with suppress_output():
            spec = importlib.util.spec_from_file_location(module_path, full_path)
            if spec is None or spec.loader is None:
                return False, "Could not create module spec"
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
        
        return True, None
    except ImportError as e:
        return False, f"ImportError: {str(e)}"
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def main():
    results = {
        "total_files": 0,
        "successful_imports": 0,
        "failed_imports": 0,
        "success_rate": "0%",
        "successful": [],
        "failed": []
    }
    
    print(f"Testing imports for {len(files_to_test)} Python files...")
    print("=" * 60)
    
    for filepath in files_to_test:
        results["total_files"] += 1
        success, error = test_import(filepath)
        
        if success:
            results["successful_imports"] += 1
            results["successful"].append(filepath)
            print(f"[OK] {filepath}")
        else:
            results["failed_imports"] += 1
            results["failed"].append({"file": filepath, "error": error})
            print(f"[FAIL] {filepath}")
            print(f"  Error: {error}")
        
        # Save intermediate results every 10 files
        if results["total_files"] % 10 == 0:
            with open(project_root / "import_test_batch8.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    
    # Calculate success rate
    if results["total_files"] > 0:
        rate = (results["successful_imports"] / results["total_files"]) * 100
        results["success_rate"] = f"{rate:.1f}%"
    
    print("=" * 60)
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful_imports']}")
    print(f"Failed: {results['failed_imports']}")
    print(f"Success rate: {results['success_rate']}")
    
    # Write final JSON report
    output_path = project_root / "import_test_batch8.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
