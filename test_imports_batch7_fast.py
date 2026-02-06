#!/usr/bin/env python3
"""Fast import test for batch 7 files - shorter timeouts"""

import json
import sys
import os
import importlib.util
import subprocess
from pathlib import Path

# Change to the project directory
os.chdir(r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend')

# All files to test
files_to_test = [
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
    "validate_adversarial_maker_integration.py", "validate_all_fixes.py",
    "validate_end_to_end_invention.py", "validate_enhanced_adversarial.py",
    "validate_evolution_maker_integration.py", "validate_generic_maker_integration.py",
    "validate_hybrid_maker_integration.py", "validate_imports.py",
    "validate_integration.py", "validate_leanaide_tests.py", "validate_maker_integration.py",
    "validate_performance.py", "validate_phase1_complete.py", "validate_production_ready.py",
    "validate_ragbits_integration.py", "validate_sop_components.py",
    "validate_sop_generator.py", "validate_sop_integrated.py", "validate_task_15.py",
    "validate_task_16.py",
    "verify_additional_math_bubbles.py", "verify_all_lean_wiring.py",
    "verify_bubblelabs_integration.py", "verify_causal_learn_final.py",
    "verify_complete_integration.py", "verify_dts_imports.py", "verify_dts_integration.py",
    "verify_gauntlet_wiring.py", "verify_global_dspy_integration.py",
    "verify_integration.py", "verify_integrations.py", "verify_knowledge_extraction.py",
    "verify_lean_wiring.py", "verify_leanaide_integration.py", "verify_leanaide_true_100.py",
    "verify_math_bubbles.py", "verify_mcp.py", "verify_rese_health_apis.py",
    "verify_roma_fix.py", "verify_true_100_integration.py",
    "verify_true_100_knowledge_extraction.py",
    "run_agents_config_tests.py", "run_all_ace_tests.py", "run_all_batch2_tests.py",
    "run_all_gauntlet_tests.py", "run_all_tests.py", "run_evolution_mdap_tests.py",
    "run_evolutionary_tests.py", "run_full_rese_e2e_pipeline.py", "run_gauntlet_tests.py",
    "run_import_test_batch3.py", "run_import_test_batch4.py", "run_integration_tests.py",
    "run_leanaide_tests.py", "run_mcts_mdap_tests.py", "run_mcts_tests.py",
    "run_mdap_tests.py", "run_real_security_tests.py", "run_rese_tests.py",
    "run_security_tests.py", "run_security_true_100_tests.py", "run_tests.py",
    "analyze_ke_imports.py", "analyze_openevolve_integration.py", "analyze_problem_analyzer.py",
    "apply_ace_phase4_fixes.py", "apply_ace_security_fixes.py",
    "apply_api_consistency_fixes.py", "apply_code_quality_fixes.py",
    "apply_component_alerting.py", "apply_phase4_validation.py",
    "fix_critical_imports.py", "fix_demo.py", "fix_demo_mcts.py", "fix_demo_mcts_final.py",
    "fix_high_severity.py", "fix_leanaide.py", "fix_logger_calls.py",
    "fix_manual_security_issues.py", "fix_mcts.py", "fix_medium_severity.py",
    "fix_non_security_issues.py", "fix_subprocess_shell.py", "fix_syntax_errors.py",
    "fix_unicode_characters.py",
]


def test_import_subprocess(filename, timeout=8):
    """Test import using a subprocess."""
    filepath = Path(r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend') / filename
    
    if not filepath.exists():
        return 'failed', "File not found"
    
    test_script = f'''
import sys
sys.path.insert(0, r'c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("test_module", r"{filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("IMPORT_SUCCESS")
except ImportError as e:
    print(f"IMPORT_ERROR: {{e}}")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{e}}")
except Exception as e:
    print(f"OTHER_ERROR: {{type(e).__name__}}: {{e}}")
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout.strip() + result.stderr.strip()
        
        if "IMPORT_SUCCESS" in output:
            return 'success', None
        elif "IMPORT_ERROR:" in output:
            error = output.split("IMPORT_ERROR:", 1)[-1].strip()
            return 'failed', f"ImportError: {error[:150]}"
        elif "SYNTAX_ERROR:" in output:
            error = output.split("SYNTAX_ERROR:", 1)[-1].strip()
            return 'failed', f"SyntaxError: {error[:150]}"
        elif "OTHER_ERROR:" in output:
            error = output.split("OTHER_ERROR:", 1)[-1].strip()
            return 'failed', error[:150]
        elif result.returncode != 0:
            return 'failed', f"Exit code {result.returncode}: {output[:150]}"
        else:
            return 'success', None
            
    except subprocess.TimeoutExpired:
        return 'timeout', f"Timeout after {timeout}s"
    except Exception as e:
        return 'failed', f"Subprocess error: {e}"


def main():
    print("=" * 70)
    print("BATCH 7 IMPORT TEST (FAST)")
    print("=" * 70)
    
    results = {
        "total_files": len(files_to_test),
        "successful_imports": 0,
        "failed_imports": 0,
        "skipped": 0,
        "successful": [],
        "failed": [],
        "skipped_files": [],
        "timeout_files": []
    }
    
    for i, filename in enumerate(files_to_test, 1):
        print(f"[{i}/{len(files_to_test)}] {filename} ... ", end="", flush=True)
        
        status, error = test_import_subprocess(filename, timeout=8)
        
        if status == 'success':
            print("OK")
            results["successful_imports"] += 1
            results["successful"].append(filename)
        elif status == 'timeout':
            print("TIMEOUT (skipped)")
            results["skipped"] += 1
            results["skipped_files"].append(filename)
            results["timeout_files"].append({"file": filename, "error": error})
        else:
            error_str = error or "Unknown error"
            print(f"FAILED: {error_str[:60]}")
            results["failed_imports"] += 1
            results["failed"].append({"file": filename, "error": error_str})
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(files_to_test)
    success = results["successful_imports"]
    failed = results["failed_imports"]
    skipped = results["skipped"]
    success_rate = (success / total * 100) if total > 0 else 0
    
    print(f"Total: {total} | Success: {success} | Failed: {failed} | Skipped: {skipped}")
    print(f"Success rate: {success_rate:.1f}%")
    
    report = {
        "total_files": total,
        "successful_imports": success,
        "failed_imports": failed,
        "skipped": skipped,
        "success_rate": f"{success_rate:.1f}%",
        "successful": results["successful"],
        "failed": results["failed"],
        "skipped_files": results["skipped_files"],
        "timeout_files": results["timeout_files"]
    }
    
    report_path = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_test_batch7.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
