#!/usr/bin/env python3
"""
Z3 API Probe Script for RESE Phase II

Probes Z3 availability for behavioral equivalence verification.

Following CLAUDE.md Law of Runtime Truth:
- Verify Z3 is actually available before using it
- Test both Python bindings and CLI
- Test basic constraint solving

Usage: python check_z3_api.py

Exit codes:
    0 - Z3 fully available (both Python and CLI)
    1 - Z3 Python bindings only
    2 - Z3 CLI only
    3 - Z3 not available
"""

import sys
import subprocess
import tempfile
import os

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50 + "\n")

def check_python_bindings():
    """Check if Z3 Python bindings are available."""
    print("Test 1: Checking Z3 Python bindings...")

    try:
        import z3
        version = z3.get_version()
        print(f"[OK] Z3 Python bindings available")
        print(f"  Version: {version}")
        return True
    except ImportError:
        print("[FAIL] Z3 Python bindings NOT available")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking Z3: {e}")
        return False

def check_z3_cli():
    """Check if Z3 CLI is available."""
    print("\nTest 2: Checking Z3 CLI...")

    try:
        result = subprocess.run(
            ['z3', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("[OK] Z3 CLI available")
            print(f"  {result.stdout.strip()}")
            return True
        else:
            print("[FAIL] Z3 CLI NOT available")
            return False
    except FileNotFoundError:
        print("[FAIL] Z3 CLI NOT found")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking Z3 CLI: {e}")
        return False

def check_constraint_solving():
    """Check if Z3 can solve simple constraints."""
    print("\nTest 3: Testing basic constraint solving...")

    smtlib_content = """
; Simple satisfiability test
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(assert (< x 10))
(check-sat)
(get-model)
"""

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib_content)
            temp_file = f.name

        try:
            # Run Z3
            result = subprocess.run(
                ['z3', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if 'sat' in result.stdout.lower():
                print("[OK] Z3 can solve simple constraints")
                print("  Output:")
                for line in result.stdout.split('\n')[:5]:
                    print(f"    {line}")
                return True
            else:
                print("[FAIL] Z3 constraint solving failed")
                return False
        finally:
            # Clean up
            os.unlink(temp_file)

    except Exception as e:
        print(f"[FAIL] Error during constraint solving: {e}")
        return False

def check_theorem_proving():
    """Check if Z3 can prove theorems."""
    print("\nTest 4: Testing theorem proving...")

    smtlib_content = """
; Simple theorem: x > 0 implies x + 1 > 0
(set-logic LIA)
(declare-const x Int)
(assert (> x 0))
(assert (not (> (+ x 1) 0)))
(check-sat)
"""

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib_content)
            temp_file = f.name

        try:
            # Run Z3
            result = subprocess.run(
                ['z3', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if 'unsat' in result.stdout.lower():
                print("[OK] Z3 can prove theorems")
                print("  Theorem 'x > 0 -> x + 1 > 0': PROVEN")
                return True
            else:
                print("[FAIL] Z3 theorem proving failed")
                return False
        finally:
            # Clean up
            os.unlink(temp_file)

    except Exception as e:
        print(f"[FAIL] Error during theorem proving: {e}")
        return False

def check_bridge():
    """Check if Z3-LeanAide bridge is available."""
    print("\nTest 5: Checking Z3-LeanAide bridge...")

    try:
        from z3_leanaide_bridge import Z3LeanAideBridge
        print("[OK] Z3-LeanAide bridge available")
        return True
    except ImportError:
        print("[FAIL] Z3-LeanAide bridge NOT available (optional)")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking bridge: {e}")
        return False

def main():
    """Run all probes and report results."""
    print_section("Z3 API Probe for RESE Phase II")

    # Run checks
    python_available = check_python_bindings()
    cli_available = check_z3_cli()
    solving_works = check_constraint_solving() if cli_available else False
    proving_works = check_theorem_proving() if cli_available else False
    bridge_available = check_bridge()

    # Summary
    print_section("Probe Summary")

    print(f"Python bindings: {'Available' if python_available else 'Not available'}")
    print(f"CLI:            {'Available' if cli_available else 'Not available'}")
    print(f"Bridge:         {'Available' if bridge_available else 'Not available (optional)'}")

    # Determine result
    print_section("Result")

    if python_available and cli_available:
        print("[OK] Z3 fully available (recommended)")
        print("\nRecommended configuration:")
        print("  export RESE_Z3_PHASE2_ENABLED=true")
        print("  export Z3_TIMEOUT=10000")
        return 0
    elif python_available:
        print("[WARN] Z3 Python bindings only (usable)")
        print("\nConfiguration:")
        print("  export RESE_Z3_PHASE2_ENABLED=true")
        print("  export Z3_TIMEOUT=10000")
        return 1
    elif cli_available:
        print("[WARN] Z3 CLI only (usable)")
        print("\nConfiguration:")
        print("  export RESE_Z3_PHASE2_ENABLED=true")
        print("  export Z3_TIMEOUT=10000")
        return 2
    else:
        print("[FAIL] Z3 not available")
        print("\nTo install Z3:")
        print("  pip install z3-solver")
        print("\nOr download binary from:")
        print("  https://github.com/Z3Prover/z3/releases")
        print("\nFallback configuration:")
        print("  export RESE_Z3_PHASE2_ENABLED=false")
        return 3

if __name__ == "__main__":
    sys.exit(main())
