#!/usr/bin/env python3
"""Test imports for batch 7 files (demo_*, validate_*, verify_*, run_*, analyze_*, apply_*, fix_*)"""

import json
import sys
import os
import importlib.util
import time
import subprocess
from pathlib import Path

# Change to the project directory
os.chdir(r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend')

# List of all files to test
files_to_test = [
    # demo_*.py (46 files)
    "demo_adversarial_maker.py",
    "demo_app.py",
    "demo_crewai_research_features.py",
    "demo_database_cleanup.py",
    "demo_e2e_invention_enhanced.py",
    "demo_end_to_end_invention.py",
    "demo_enhanced_adversarial.py",
    "demo_enhanced_decomposition_recomposition.py",
    "demo_evolution_maker.py",
    "demo_evolution_mdap.py",
    "demo_evolutionary_tests.py",
    "demo_generic_maker.py",
    "demo_hierarchical_indexing.py",
    "demo_hybrid_maker.py",
    "demo_hybrid_mcts.py",
    "demo_integration.py",
    "demo_knowledge_extraction_ml.py",
    "demo_leanaide_autoformalization_mdap_maker.py",
    "demo_leanaide_client.py",
    "demo_leanaide_config.py",
    "demo_leanaide_redflagging.py",
    "demo_maker_complete.py",
    "demo_matryoshka_auto.py",
    "demo_matryoshka_unified_memory.py",
    "demo_mcts.py",
    "demo_mcts_mdap.py",
    "demo_mdap_maker.py",
    "demo_mdap_maker_matryoshka.py",
    "demo_mdap_maker_mcts_unified.py",
    "demo_openevolve_bubblelabs.py",
    "demo_openevolve_integration.py",
    "demo_openevolve_pes_integration.py",
    "demo_pes_workflow.py",
    "demo_pes_workflow_language_agnostic.py",
    "demo_pes_workflow_universal.py",
    "demo_problem_classifier.py",
    "demo_quality_calculator.py",
    "demo_reliability_system.py",
    "demo_roma_mdap_maker.py",
    "demo_sop_components.py",
    "demo_sop_generator.py",
    "demo_sop_integrated.py",
    "demo_team_assignment.py",
    "demo_ui_integration.py",
    "demo_unified_memory_system.py",
    "demo_z3_leanaide_integration.py",
    # validate_*.py (20 files)
    "validate_adversarial_maker_integration.py",
    "validate_all_fixes.py",
    "validate_end_to_end_invention.py",
    "validate_enhanced_adversarial.py",
    "validate_evolution_maker_integration.py",
    "validate_generic_maker_integration.py",
    "validate_hybrid_maker_integration.py",
    "validate_imports.py",
    "validate_integration.py",
    "validate_leanaide_tests.py",
    "validate_maker_integration.py",
    "validate_performance.py",
    "validate_phase1_complete.py",
    "validate_production_ready.py",
    "validate_ragbits_integration.py",
    "validate_sop_components.py",
    "validate_sop_generator.py",
    "validate_sop_integrated.py",
    "validate_task_15.py",
    "validate_task_16.py",
    # verify_*.py (21 files)
    "verify_additional_math_bubbles.py",
    "verify_all_lean_wiring.py",
    "verify_bubblelabs_integration.py",
    "verify_causal_learn_final.py",
    "verify_complete_integration.py",
    "verify_dts_imports.py",
    "verify_dts_integration.py",
    "verify_gauntlet_wiring.py",
    "verify_global_dspy_integration.py",
    "verify_integration.py",
    "verify_integrations.py",
    "verify_knowledge_extraction.py",
    "verify_lean_wiring.py",
    "verify_leanaide_integration.py",
    "verify_leanaide_true_100.py",
    "verify_math_bubbles.py",
    "verify_mcp.py",
    "verify_rese_health_apis.py",
    "verify_roma_fix.py",
    "verify_true_100_integration.py",
    "verify_true_100_knowledge_extraction.py",
    # run_*.py (21 files)
    "run_agents_config_tests.py",
    "run_all_ace_tests.py",
    "run_all_batch2_tests.py",
    "run_all_gauntlet_tests.py",
    "run_all_tests.py",
    "run_evolution_mdap_tests.py",
    "run_evolutionary_tests.py",
    "run_full_rese_e2e_pipeline.py",
    "run_gauntlet_tests.py",
    "run_import_test_batch3.py",
    "run_import_test_batch4.py",
    "run_integration_tests.py",
    "run_leanaide_tests.py",
    "run_mcts_mdap_tests.py",
    "run_mcts_tests.py",
    "run_mdap_tests.py",
    "run_real_security_tests.py",
    "run_rese_tests.py",
    "run_security_tests.py",
    "run_security_true_100_tests.py",
    "run_tests.py",
    # analyze_*.py (3 files)
    "analyze_ke_imports.py",
    "analyze_openevolve_integration.py",
    "analyze_problem_analyzer.py",
    # apply_*.py (6 files)
    "apply_ace_phase4_fixes.py",
    "apply_ace_security_fixes.py",
    "apply_api_consistency_fixes.py",
    "apply_code_quality_fixes.py",
    "apply_component_alerting.py",
    "apply_phase4_validation.py",
    # fix_*.py (14 files)
    "fix_critical_imports.py",
    "fix_demo.py",
    "fix_demo_mcts.py",
    "fix_demo_mcts_final.py",
    "fix_high_severity.py",
    "fix_leanaide.py",
    "fix_logger_calls.py",
    "fix_manual_security_issues.py",
    "fix_mcts.py",
    "fix_medium_severity.py",
    "fix_non_security_issues.py",
    "fix_subprocess_shell.py",
    "fix_syntax_errors.py",
    "fix_unicode_characters.py",
]


def check_file_content(filepath):
    """Check if file has main guard and dangerous patterns."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return False, [], f"Cannot read: {e}"
    
    has_main_guard = 'if __name__' in content and "'__main__'" in content
    
    dangerous_patterns = [
        'BubbleLab UI run()',
        'app.run(',
        'uvicorn.run(',
        'server.run(',
        'mainloop()',
        'plt.show()',
        'plt.pause(',
        'input(',
        'raw_input(',
        'time.sleep(',
        'asyncio.run(',
    ]
    
    found_dangerous = [p for p in dangerous_patterns if p in content]
    might_execute = not has_main_guard and len(found_dangerous) > 0
    
    return has_main_guard, found_dangerous, might_execute


def test_import_subprocess(filename, timeout=15):
    """Test import using a subprocess to handle timeouts and isolation."""
    filepath = Path(r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend') / filename
    
    if not filepath.exists():
        return 'failed', f"File not found"
    
    # Create a test script
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
            return 'failed', f"ImportError: {error[:200]}"
        elif "SYNTAX_ERROR:" in output:
            error = output.split("SYNTAX_ERROR:", 1)[-1].strip()
            return 'failed', f"SyntaxError: {error[:200]}"
        elif "OTHER_ERROR:" in output:
            error = output.split("OTHER_ERROR:", 1)[-1].strip()
            return 'failed', error[:200]
        elif result.returncode != 0:
            return 'failed', f"Exit code {result.returncode}: {output[:200]}"
        else:
            return 'success', None
            
    except subprocess.TimeoutExpired:
        return 'timeout', f"Import timed out after {timeout}s"
    except Exception as e:
        return 'failed', f"Subprocess error: {e}"


def main():
    print("=" * 80)
    print("BATCH 7 IMPORT TEST")
    print("Testing: demo_*, validate_*, verify_*, run_*, analyze_*, apply_*, fix_*.py")
    print("=" * 80)
    print()
    
    results = {
        "total_files": len(files_to_test),
        "successful_imports": 0,
        "failed_imports": 0,
        "skipped": 0,
        "successful": [],
        "failed": [],
        "skipped_files": [],
        "timeout_files": [],
        "executes_on_import": []
    }
    
    for i, filename in enumerate(files_to_test, 1):
        filepath = Path(r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend') / filename
        
        print(f"[{i}/{len(files_to_test)}] Testing: {filename} ... ", end="", flush=True)
        
        if not filepath.exists():
            print("FAILED: File not found")
            results["failed_imports"] += 1
            results["failed"].append({"file": filename, "error": "File not found"})
            continue
        
        # Check file content
        has_main_guard, dangerous_patterns, might_execute = check_file_content(filepath)
        if might_execute:
            results["executes_on_import"].append(filename)
        
        # Test import
        status, error = test_import_subprocess(filename)
        
        if status == 'success':
            print("OK")
            results["successful_imports"] += 1
            results["successful"].append(filename)
        elif status == 'timeout':
            print(f"TIMEOUT (skipped)")
            results["skipped"] += 1
            results["skipped_files"].append(filename)
            results["timeout_files"].append({"file": filename, "error": error})
        else:
            error_str = error or "Unknown error"
            print(f"FAILED: {error_str[:50]}..." if len(error_str) > 50 else f"FAILED: {error_str}")
            results["failed_imports"] += 1
            results["failed"].append({"file": filename, "error": error_str})
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(files_to_test)
    success = results["successful_imports"]
    failed = results["failed_imports"]
    skipped = results["skipped"]
    
    success_rate = (success / total * 100) if total > 0 else 0
    
    print(f"Total files tested: {total}")
    print(f"Successful imports: {success}")
    print(f"Failed imports: {failed}")
    print(f"Skipped (timeout): {skipped}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if results["executes_on_import"]:
        print(f"\nFiles that may execute code on import: {len(results['executes_on_import'])}")
        for f in results["executes_on_import"][:10]:
            print(f"  - {f}")
        if len(results["executes_on_import"]) > 10:
            print(f"  ... and {len(results['executes_on_import']) - 10} more")
    
    # Prepare final report
    report = {
        "total_files": total,
        "successful_imports": success,
        "failed_imports": failed,
        "skipped": skipped,
        "success_rate": f"{success_rate:.1f}%",
        "successful": results["successful"],
        "failed": results["failed"],
        "skipped_files": results["skipped_files"],
        "timeout_files": results["timeout_files"],
        "executes_on_import": results["executes_on_import"]
    }
    
    # Write report
    report_path = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_test_batch7.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport written to: {report_path}")
    
    return report


if __name__ == '__main__':
    main()
