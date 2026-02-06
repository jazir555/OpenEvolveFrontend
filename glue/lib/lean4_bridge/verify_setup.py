#!/usr/bin/env python3
"""
Lean 4 Bridge Verification Script

Following CLAUDE.md principles:
- Law of Runtime Truth: Verify all components work
- Law of Configuration Explicitness: All config from env vars
- Exit non-zero if verification fails

Usage:
    python verify_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Import real Lean interface for verification
try:
    from lean4_interface import Lean4Interface, Lean4Error
    LEAN4_INTERFACE_AVAILABLE = True
except ImportError:
    LEAN4_INTERFACE_AVAILABLE = False

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}[OK]{NC} {text}")

def print_error(text):
    """Print error message."""
    print(f"{RED}[FAIL]{NC} {text}")

def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}[WARN]{NC} {text}")

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} not found: {filepath}")
        return False

def check_python_import(module_name, description):
    """Check if Python module can be imported."""
    try:
        __import__(module_name)
        print_success(f"{description}: {module_name}")
        return True
    except ImportError as e:
        print_error(f"{description} not available: {module_name}")
        print(f"  Error: {e}")
        return False

def verify_docker_files():
    """Verify Docker environment files."""
    print_header("Docker Environment Verification")

    all_ok = True

    # Check Dockerfile
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/infra/lean4-docker/Dockerfile",
        "Dockerfile"
    )

    # Check docker-compose
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/infra/lean4-docker/docker-compose.lean4.yml",
        "Docker Compose config"
    )

    # Check requirements.txt
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/infra/lean4-docker/requirements.txt",
        "Python requirements"
    )

    return all_ok

def verify_python_bridge():
    """Verify Python bridge code."""
    print_header("Python Bridge Verification")

    all_ok = True

    # Check main interface
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/lean4_interface.py",
        "Lean 4 interface"
    )

    # Check translator
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/src/constraint_translator.py",
        "Constraint translator"
    )

    # Check __init__ files
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/__init__.py",
        "Package __init__"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/src/__init__.py",
        "Source __init__"
    )

    return all_ok

def verify_lean4_library():
    """Verify Lean 4 library files."""
    print_header("Lean 4 Library Verification")

    all_ok = True

    # Check Lean 4 files
    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/lean4/RESE.lean",
        "RESE.lean"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/lean4/Constraints.lean",
        "Constraints.lean"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/lean4/FDG.lean",
        "FDG.lean"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/lakefile.lean",
        "lakefile.lean"
    )

    return all_ok

def verify_documentation():
    """Verify documentation files."""
    print_header("Documentation Verification")

    all_ok = True

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/ARCHITECTURE.md",
        "Architecture documentation"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/README.md",
        "README"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/IMPLEMENTATION_SUMMARY.md",
        "Implementation summary"
    )

    return all_ok

def verify_tests():
    """Verify test files."""
    print_header("Tests Verification")

    all_ok = True

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/tests/test_lean4_interface.py",
        "Unit tests"
    )

    all_ok &= check_file_exists(
        "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/glue/lib/lean4_bridge/probes/check_lean4.sh",
        "Probe script"
    )

    return all_ok

def verify_python_dependencies():
    """Verify Python dependencies."""
    print_header("Python Dependencies Verification")

    all_ok = True

    # Check required packages
    required_packages = [
        ("psutil", "psutil"),
        ("structlog", "structlog"),
        ("pydantic", "pydantic"),
    ]

    for module_name, package_name in required_packages:
        all_ok &= check_python_import(module_name, package_name)

    return all_ok

def verify_lean_installation():
    """Verify REAL Lean 4 installation."""
    print_header("Lean 4 Installation Verification")

    all_ok = True

    # Check if Lean4Interface is available
    if LEAN4_INTERFACE_AVAILABLE:
        print_success("Lean4Interface module available")

        try:
            # Try to initialize Lean4Interface (this verifies lean executable)
            lean = Lean4Interface()
            print_success("Lean4Interface initialized successfully")
            print_success(f"  Lean path: {lean.lean_path}")
            print_success(f"  Lake path: {lean.lake_path}")
            print_success(f"  Workspace: {lean.workspace_dir}")
            print_success(f"  Timeout: {lean.timeout_ms}ms")
        except Exception as e:
            print_error(f"Lean4Interface initialization failed: {e}")
            all_ok = False
    else:
        print_error("Lean4Interface module not available")
        print_warning("  Cannot import lean4_interface - check Python path")
        all_ok = False

    # Check for lean executable in PATH
    try:
        result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success(f"Lean executable found: {result.stdout.strip()}")
        else:
            print_error("Lean executable returned error")
            all_ok = False
    except FileNotFoundError:
        print_error("Lean executable not found in PATH")
        print_warning("  Install Lean 4 or set LEAN4_PATH environment variable")
        all_ok = False
    except subprocess.TimeoutExpired:
        print_error("Lean version check timed out")
        all_ok = False
    except Exception as e:
        print_error(f"Lean version check failed: {e}")
        all_ok = False

    # Check for lake executable
    try:
        result = subprocess.run(
            ["lake", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success(f"Lake executable found: {result.stdout.strip()}")
        else:
            print_warning("Lake executable returned error (may be OK if not using lake)")
    except FileNotFoundError:
        print_warning("Lake executable not found in PATH")
    except Exception as e:
        print_warning(f"Lake version check failed: {e}")

    return all_ok

def verify_docker_integration():
    """Verify Docker integration in main stack."""
    print_header("Docker Integration Verification")

    docker_compose_path = "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/infra/docker-compose.yml"

    if not Path(docker_compose_path).exists():
        print_warning("Main docker-compose.yml not found")
        return True  # Not critical

    # Check if Lean 4 service is included
    with open(docker_compose_path, 'r') as f:
        content = f.read()
        if 'rese-lean4' in content:
            print_success("Lean 4 service integrated in docker-compose.yml")
            return True
        else:
            print_warning("Lean 4 service not found in docker-compose.yml")
            return True  # Not critical for verification

def count_lines_of_code():
    """Count lines of code."""
    print_header("Lines of Code Summary")

    files_to_count = [
        ("Python", "glue/lib/lean4_bridge/lean4_interface.py"),
        ("Python", "glue/lib/lean4_bridge/src/constraint_translator.py"),
        ("Lean 4", "glue/lib/lean4_bridge/lean4/RESE.lean"),
        ("Lean 4", "glue/lib/lean4_bridge/lean4/Constraints.lean"),
        ("Lean 4", "glue/lib/lean4_bridge/lean4/FDG.lean"),
        ("Tests", "glue/lib/lean4_bridge/tests/test_lean4_interface.py"),
        ("Docker", "infra/lean4-docker/Dockerfile"),
        ("Docs", "glue/lib/lean4_bridge/ARCHITECTURE.md"),
        ("Docs", "glue/lib/lean4_bridge/README.md"),
    ]

    base_path = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend")
    total_lines = 0
    category_totals = {}

    for category, file_path in files_to_count:
        full_path = base_path / file_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
                total_lines += lines
                category_totals[category] = category_totals.get(category, 0) + lines
                print(f"  {category}: {file_path} - {lines} lines")

    print(f"\n  Total by category:")
    for category, total in category_totals.items():
        print(f"    {category}: {total} lines")

    print(f"\n  Grand total: {total_lines} lines")

def main():
    """Main verification function."""
    print_header("Lean 4 Bridge Setup Verification")
    print("Verifying all components are in place...")
    print("Working directory: C:/Users/mmeadow/Documents/OpenEvolve/Frontend")

    all_ok = True

    # Run all verifications
    all_ok &= verify_docker_files()
    all_ok &= verify_python_bridge()
    all_ok &= verify_lean4_library()
    all_ok &= verify_documentation()
    all_ok &= verify_tests()
    all_ok &= verify_python_dependencies()
    all_ok &= verify_docker_integration()
    all_ok &= verify_lean_installation()  # NEW: Real Lean installation check

    # Count lines of code
    count_lines_of_code()

    # Final summary
    print_header("Verification Summary")

    if all_ok:
        print_success("All verifications passed!")
        print("\nLean 4 Bridge is ready for use with REAL Lean 4 integration.")
        print("\nNext steps:")
        print("  1. Build Docker image:")
        print("     cd infra/lean4-docker")
        print("     docker build -t rese-lean4:latest .")
        print("\n  2. Start Lean 4 service:")
        print("     docker-compose -f docker-compose.lean4.yml up -d")
        print("\n  3. Run tests:")
        print("     cd glue/lib/lean4_bridge")
        print("     python -m pytest tests/ -v")
        print("\n  4. Run probe:")
        print("     cd probes")
        print("     ./check_lean4.sh")
        print("\n  5. Test Lean4Interface:")
        print("     python -c \"from glue.lib.lean4_bridge import Lean4Interface; li = Lean4Interface(); print(li)\"")
        return 0
    else:
        print_error("Some verifications failed!")
        print("\nPlease check the errors above and fix them.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
