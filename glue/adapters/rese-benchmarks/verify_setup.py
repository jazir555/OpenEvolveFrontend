#!/usr/bin/env python3
"""
RESE Benchmark Suite Verification Script

Verifies that all dependencies and imports are correctly configured
for running the RESE benchmark suite.

Usage:
    python verify_setup.py

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def verify_file_exists(filepath: Path, description: str) -> bool:
    """Verify a file exists."""
    if filepath.exists():
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} NOT found: {filepath}")
        return False


def verify_import(
    module_path: List[str],
    import_name: str,
    description: str
) -> bool:
    """Verify an import works."""
    import sys
    from importlib import import_module

    # Add paths
    benchmark_dir = Path(__file__).parent
    sys.path.insert(0, str(benchmark_dir.parent / "rese-phase1" / "src"))
    sys.path.insert(0, str(benchmark_dir.parent / "rese-phase2" / "src"))
    sys.path.insert(0, str(benchmark_dir.parent / "rese-phase3" / "src"))
    sys.path.insert(0, str(benchmark_dir.parent / "rese-phase4" / "src"))

    try:
        module = import_module(".".join(module_path))
        getattr(module, import_name)
        print_success(f"{description}: {import_name}")
        return True
    except ImportError as e:
        print_error(f"{description} failed: {e}")
        return False
    except Exception as e:
        print_warning(f"{description} error: {e}")
        return False


def verify_directory_structure() -> Tuple[bool, bool, bool, bool]:
    """Verify benchmark directory structure."""
    print("\n" + "=" * 70)
    print("Verifying Directory Structure")
    print("=" * 70)

    benchmark_dir = Path(__file__).parent
    all_ok = True

    # Check benchmark scripts
    scripts = [
        ("benchmark_phase1.py", "Phase I benchmark"),
        ("benchmark_phase2.py", "Phase II benchmark"),
        ("benchmark_phase3.py", "Phase III benchmark"),
        ("benchmark_phase4.py", "Phase IV benchmark"),
        ("benchmark_full_pipeline.py", "Full pipeline benchmark"),
        ("run_all_benchmarks.py", "Orchestrator"),
        ("init_baseline.py", "Baseline initializer"),
    ]

    for script, desc in scripts:
        if not verify_file_exists(benchmark_dir / script, desc):
            all_ok = False

    # Check results directory
    results_dir = benchmark_dir / "results"
    if not results_dir.exists():
        print_warning("Results directory not found (will be created)")
        results_dir.mkdir(exist_ok=True)
    else:
        print_success(f"Results directory: {results_dir}")

    return all_ok, True, True, True


def verify_phase_executors() -> Tuple[bool, bool, bool, bool]:
    """Verify phase executor imports."""
    print("\n" + "=" * 70)
    print("Verifying Phase Executors")
    print("=" * 70)

    phase1_ok = verify_import(
        ["phase1_executor"],
        "EpistemicAuditExecutor",
        "Phase I Executor"
    )

    phase2_ok = verify_import(
        ["phase2_executor"],
        "IsomorphicMappingExecutor",
        "Phase II Executor"
    )

    phase3_ok = verify_import(
        ["phase3_executor"],
        "MCTSSearchExecutor",
        "Phase III Executor"
    )

    phase4_ok = verify_import(
        ["phase4_executor"],
        "ArchitectureAssemblyExecutor",
        "Phase IV Executor"
    )

    return phase1_ok, phase2_ok, phase3_ok, phase4_ok


def verify_python_version() -> bool:
    """Verify Python version."""
    print("\n" + "=" * 70)
    print("Verifying Python Version")
    print("=" * 70)

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version too old: {version_str} (need 3.8+)")
        return False


def verify_environment() -> bool:
    """Verify environment configuration."""
    print("\n" + "=" * 70)
    print("Verifying Environment Configuration")
    print("=" * 70)

    # Check for required env vars (can be defaults)
    env_vars = [
        "PHASE1_TIMEOUT_MS",
        "PHASE2_TIMEOUT_MS",
        "PHASE3_ITERATIONS",
        "PHASE4_ASSEMBLY_TIMEOUT_MS",
    ]

    all_set = True
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var}={value}")
        else:
            print_warning(f"{var} not set (will use default)")

    return all_set


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("RESE Benchmark Suite Verification")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")

    # Run verifications
    python_ok = verify_python_version()
    structure_ok, _, _, _ = verify_directory_structure()
    phase1_ok, phase2_ok, phase3_ok, phase4_ok = verify_phase_executors()
    env_ok = verify_environment()

    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    checks = [
        ("Python Version", python_ok),
        ("Directory Structure", structure_ok),
        ("Phase I Executor", phase1_ok),
        ("Phase II Executor", phase2_ok),
        ("Phase III Executor", phase3_ok),
        ("Phase IV Executor", phase4_ok),
        ("Environment", env_ok),
    ]

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "FAILED"
        color = Colors.GREEN if ok else Colors.RED
        print(f"{color}{status}{Colors.RESET} - {name}")
        if not ok:
            all_ok = False

    # Final verdict
    print("\n" + "=" * 70)
    if all_ok:
        print_success("All verifications passed!")
        print("\nYou can now run benchmarks:")
        print("  python run_all_benchmarks.py")
        print("\nOr run individual phases:")
        print("  python benchmark_phase1.py")
        print("  python benchmark_phase2.py")
        print("  python benchmark_phase3.py")
        print("  python benchmark_phase4.py")
        print("  python benchmark_full_pipeline.py")
        return 0
    else:
        print_error("Some verifications failed!")
        print("\nPlease fix the issues above before running benchmarks.")
        print("\nCommon fixes:")
        print("  - Ensure phase executors are in ../rese-phaseX/src/")
        print("  - Install missing dependencies")
        print("  - Set environment variables (see BENCHMARKS_README.md)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
