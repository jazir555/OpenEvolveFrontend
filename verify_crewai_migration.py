"""
CrewAI Migration Verification Script

This script verifies that:
1. All CrewAI files are deleted
2. CrewAI modules can be imported
3. No CrewAI imports remain
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}[PASS] {msg}{RESET}")

def print_error(msg):
    print(f"{RED}[FAIL] {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}[WARN] {msg}{RESET}")

def check_CrewAI_deleted():
    """Check that all CrewAI files are deleted"""
    print("\n=== Phase 1: CrewAI File Cleanup ===")

    frontend_dir = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend")

    # Check main CrewAI directory
    CrewAI_dir = frontend_dir / "CrewAI"
    if CrewAI_dir.exists():
        print_error("CrewAI directory still exists!")
        return False
    else:
        print_success("CrewAI directory deleted")

    # Check for CrewAI Python files
    CrewAI_files = list(frontend_dir.glob("*CrewAI*.py"))
    if CrewAI_files:
        print_error(f"Found {len(CrewAI_files)} CrewAI Python files:")
        for f in CrewAI_files:
            print(f"  - {f.name}")
        return False
    else:
        print_success("No CrewAI Python files in root")

    # Check for CrewAI backup files
    CrewAI_backups = list(frontend_dir.glob("*CrewAI*.backup"))
    if CrewAI_backups:
        print_warning(f"Found {len(CrewAI_backups)} backup files (can be ignored)")
    else:
        print_success("No CrewAI backup files")

    # Check for CrewAI .md files
    CrewAI_docs = list(frontend_dir.glob("*CrewAI*.md"))
    if CrewAI_docs:
        print_warning(f"Found {len(CrewAI_docs)} documentation files")
        for f in CrewAI_docs:
            print(f"  - {f.name}")

    return True

def check_imports():
    """Check that CrewAI modules can be imported"""
    print("\n=== Phase 2: CrewAI Import Tests ===")

    modules_to_test = [
        "crewai_state_management",
        "bubblelabs_crewai_bridge",
        "datapizza_crewai_bridge",
        "claudiomiro_crewai_bridge",
        "decomposition_crewai_bridge",
        "ace_crewai_bridge",
    ]

    results = {}
    for module in modules_to_test:
        try:
            __import__(module)
            print_success(f"{module} imports OK")
            results[module] = True
        except Exception as e:
            print_error(f"{module} import failed: {str(e)[:100]}")
            results[module] = False

    return all(results.values()), results

def check_no_CrewAI_imports():
    """Check that no Python files import from crewai # MIGRATED: was CrewAI"""
    print("\n=== Phase 3: CrewAI Import Check ===")

    frontend_dir = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend")
    python_files = list(frontend_dir.rglob("*.py"))

    CrewAI_imports = []

    for py_file in python_files:
        # Skip __pycache__ and virtual environments
        if any(skip in str(py_file) for skip in ['__pycache__', '.venv', 'node_modules']):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for CrewAI imports (case-insensitive)
                if any(pattern in content.lower() for pattern in [
                    'from crewai # MIGRATED: was CrewAI',
                    'import crewai # MIGRATED: was CrewAI',
                    'CrewAI_bridge',
                    'crewai_integration',
                ]):
                    # Only flag if not in comments
                    for line in content.split('\n'):
                        if 'CrewAI' in line.lower() and not line.strip().startswith('#'):
                            CrewAI_imports.append((py_file.name, line.strip()))
                            break
        except Exception as e:
            print_warning(f"Could not read {py_file.name}: {e}")

    if CrewAI_imports:
        print_error(f"Found {len(CrewAI_imports)} files with CrewAI imports:")
        for filename, line in CrewAI_imports[:10]:  # Show first 10
            # Encode to avoid Unicode errors
            safe_line = line.encode('ascii', 'ignore').decode('ascii')[:80]
            print(f"  - {filename}: {safe_line}")
        return False
    else:
        print_success("No active CrewAI imports found")
        return True

def check_crewai_files():
    """Check that CrewAI files exist"""
    print("\n=== Phase 4: CrewAI File Existence ===")

    frontend_dir = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend")

    required_files = [
        "crewai_state_management.py",
        "crewai_client.py",
        "bubblelabs_crewai_bridge.py",
        "datapizza_crewai_bridge.py",
        "claudiomiro_crewai_bridge.py",
        "decomposition_crewai_bridge.py",
        "ace_crewai_bridge.py",
        "crewai_unified_flow.py",
    ]

    all_exist = True
    for filename in required_files:
        filepath = frontend_dir / filename
        if filepath.exists():
            print_success(f"{filename} exists")
        else:
            print_error(f"{filename} missing")
            all_exist = False

    return all_exist

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("CrewAI Migration Verification")
    print("=" * 60)

    results = {
        "CrewAI Deleted": check_CrewAI_deleted(),
        "CrewAI Imports": None,
        "No CrewAI Imports": check_no_CrewAI_imports(),
        "CrewAI Files": check_crewai_files(),
    }

    # Check imports separately to capture results
    import_success, import_results = check_imports()
    results["CrewAI Imports"] = import_success

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        if passed:
            print_success(f"{check}")
        elif passed is None:
            print_warning(f"{check}: Skipped")
        else:
            print_error(f"{check}")

    all_passed = all(r is True for r in results.values())

    if all_passed:
        print(f"\n{GREEN}All checks passed! Migration complete.{RESET}")
        return 0
    else:
        print(f"\n{RED}Some checks failed. Please review.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
