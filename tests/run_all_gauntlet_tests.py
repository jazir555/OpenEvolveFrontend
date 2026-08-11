"""
Run all gauntlet and monitoring tests and generate a summary
"""
import subprocess
import sys
from pathlib import Path

test_files = [
    "tests/gauntlets/test_edge_cases_adaptive_learner.py",
    "tests/gauntlets/test_edge_cases_ml_optimizer.py",
    "tests/gauntlets/test_edge_cases_predictive_executor.py",
    "tests/gauntlets/test_edge_cases_websocket.py",
    "tests/gauntlets/test_enhanced_gauntlet_system.py",
    "tests/gauntlets/test_loongflow_gauntlet.py",
    "tests/gauntlets/test_ml_optimizer.py",
    "tests/gauntlets/test_multi_round_orchestrator.py",
    "tests/gauntlets/test_predictive_executor.py",
    "tests/gauntlets/test_three_round_orchestrator.py",
    "tests/gauntlets/test_websocket.py",
    "tests/gauntlet_monitoring/test_monitoring.py",
]

project_root = Path(__file__).parent
results = []

for test_file in test_files:
    test_path = project_root / test_file

    if not test_path.exists():
        results.append((test_file, "NOT FOUND", 0, 0))
        continue

    print(f"\n{'='*80}")
    print(f"Testing: {test_file}")
    print(f"{'='*80}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=no"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )

    # Parse output
    output = result.stdout + result.stderr
    if "passed" in output:
        # Extract passed count
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0:
                        passed = int(part)
                        failed = 0
                        if 'failed' in line:
                            j = i + 1
                            while j < len(parts) and not parts[j].isdigit():
                                j += 1
                            if j < len(parts):
                                failed = int(parts[j])
                        results.append((test_file, "PASS", passed, failed))
                        break
                break
    else:
        results.append((test_file, "ERROR", 0, 0))

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

total_passed = 0
total_failed = 0

for test_file, status, passed, failed in results:
    if status == "NOT FOUND":
        print(f"[NOT FOUND] {test_file}")
    elif status == "ERROR":
        print(f"[ERROR] {test_file}")
    else:
        total_passed += passed
        total_failed += failed
        if failed > 0:
            print(f"[FAIL] {test_file}: {passed} passed, {failed} failed")
        else:
            print(f"[PASS] {test_file}: {passed} passed")

print(f"\n{'='*80}")
print(f"TOTAL: {total_passed} passed, {total_failed} failed")
print(f"{'='*80}")
