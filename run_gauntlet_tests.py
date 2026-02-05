"""
Run gauntlet tests with proper imports
"""
import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent

# Run pytest
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

# Run tests one by one
for test_file in test_files:
    test_path = project_root / test_file
    if test_path.exists():
        print(f"\n{'='*80}")
        print(f"Running: {test_file}")
        print(f"{'='*80}\n")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
            cwd=str(project_root),
            capture_output=False
        )

        if result.returncode != 0:
            print(f"\n[FAILED]: {test_file}")
        else:
            print(f"\n[PASSED]: {test_file}")
    else:
        print(f"\n[NOT FOUND]: {test_file}")

print("\n" + "="*80)
print("Test run complete!")
print("="*80)
