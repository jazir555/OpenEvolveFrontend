"""
Lean 4 Integration Activation Script

This script:
1. Detects Lean 4 installation
2. Verifies mathlib4 is available
3. Starts the LeanAide server if needed
4. Tests the integration
5. Generates a status report

Usage:
    python activate_lean_integration.py [--start-server] [--test] [--fix-paths]

Author: OpenEvolve
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Detection Functions
# =============================================================================

def detect_lean_executable() -> Optional[str]:
    """Detect Lean 4 executable path."""
    # Check environment variable first
    lean_exe = os.environ.get('LEAN_EXECUTABLE')
    if lean_exe:
        return lean_exe
    
    # Check common locations
    possible_paths = [
        "lean",
        Path.home() / ".elan" / "bin" / "lean",
        Path.home() / ".local" / "bin" / "lean",
        "/usr/local/bin/lean",
        "/usr/bin/lean",
        "C:\\Users\\mmeadow\\.elan\\bin\\lean.exe",
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run(
                [str(path), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Found Lean at: {path}")
                return str(path)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            continue
    
    return None


def detect_lake_executable() -> Optional[str]:
    """Detect lake executable path."""
    lake_exe = os.environ.get('LAKE_EXECUTABLE')
    if lake_exe:
        return lake_exe
    
    possible_paths = [
        "lake",
        Path.home() / ".elan" / "bin" / "lake",
        Path.home() / ".local" / "bin" / "lake",
        "/usr/local/bin/lake",
        "/usr/bin/lake",
        "C:\\Users\\mmeadow\\.elan\\bin\\lake.exe",
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run(
                [str(path), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Found lake at: {path}")
                return str(path)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            continue
    
    return None


def detect_mathlib_project() -> Optional[str]:
    """Detect mathlib4 project path."""
    possible_paths = [
        Path.cwd() / "lean_workspace" / "mathlib_project",
        Path.cwd() / "mathlib_project",
        Path.home() / ".lean" / "mathlib4",
        Path.home() / "Documents" / "OpenEvolve" / "Frontend" / "lean_workspace" / "mathlib_project",
    ]
    
    for path in possible_paths:
        lakefile = path / "lakefile.lean"
        if lakefile.exists():
            logger.info(f"Found mathlib project at: {path}")
            return str(path)
    
    return None


def check_lean_version(lean_exe: str) -> Optional[str]:
    """Check Lean version."""
    try:
        result = subprocess.run(
            [lean_exe, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to check Lean version: {e}")
    
    return None


def verify_mathlib_build(mathlib_path: str, lake_exe: str) -> bool:
    """Verify mathlib project is built."""
    build_path = Path(mathlib_path) / ".lake" / "build"
    if build_path.exists():
        logger.info("mathlib build directory found")
        return True
    
    logger.warning("mathlib build directory not found. Building...")
    try:
        result = subprocess.run(
            [lake_exe, 'build'],
            cwd=mathlib_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        if result.returncode == 0:
            logger.info("mathlib built successfully")
            return True
        else:
            logger.error(f"mathlib build failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("mathlib build timed out")
    except Exception as e:
        logger.error(f"mathlib build error: {e}")
    
    return False


# =============================================================================
# Server Management
# =============================================================================

def is_server_running(host: str = "localhost", port: int = 7654) -> bool:
    """Check if LeanAide server is running."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_leanaide_server(
    mathlib_path: str,
    host: str = "localhost",
    port: int = 7654
) -> Optional[subprocess.Popen]:
    """Start the LeanAide server."""
    logger.info(f"Starting LeanAide server on {host}:{port}")
    
    env = os.environ.copy()
    env['MATHLIB_PATH'] = mathlib_path
    
    try:
        # Try to start the server
        # Note: This assumes there's a server script available
        server_script = Path(__file__).parent / "leanaide_server.py"
        if server_script.exists():
            process = subprocess.Popen(
                [sys.executable, str(server_script), '--host', host, '--port', str(port)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for server to start
            import time
            time.sleep(3)
            
            if is_server_running(host, port):
                logger.info("LeanAide server started successfully")
                return process
            else:
                logger.error("LeanAide server failed to start")
                process.terminate()
                return None
        else:
            logger.warning(f"Server script not found: {server_script}")
            return None
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None


# =============================================================================
# Integration Tests
# =============================================================================

async def test_leanaide_client() -> Dict[str, Any]:
    """Test LeanAide client integration."""
    results = {
        "client_import": False,
        "autoformalize": False,
        "verify": False,
        "errors": []
    }
    
    try:
        from leanaide_client import LeanAideClient, LeanAideConfig
        results["client_import"] = True
        
        config = LeanAideConfig()
        client = LeanAideClient(config)
        
        # Test autoformalize
        test_statement = "The sum of two even numbers is even"
        try:
            formalized = await client.autoformalize(test_statement)
            results["autoformalize"] = True
            results["formalized_output"] = formalized[:200] if formalized else None
        except Exception as e:
            results["errors"].append(f"Autoformalize failed: {e}")
        
        # Test verify
        try:
            test_theorem = "theorem add_even (n m : Nat) : Even n -> Even m -> Even (n + m)"
            verify_result = await client.verify(test_theorem)
            results["verify"] = verify_result.verified
        except Exception as e:
            results["errors"].append(f"Verify failed: {e}")
        
    except ImportError as e:
        results["errors"].append(f"Client import failed: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")
    
    return results


def test_integration_imports() -> Dict[str, Any]:
    """Test that all integration modules can be imported."""
    results = {
        "leanaide_integration": False,
        "leanaide_client": False,
        "lean4_integration": False,
        "config": False,
        "errors": []
    }
    
    modules = [
        ("leanaide_integration", "leanaide_integration"),
        ("leanaide_client", "leanaide_client"),
        ("lean4_integration", "lean4_integration"),
        ("config", "config"),
    ]
    
    for key, module_name in modules:
        try:
            __import__(module_name)
            results[key] = True
        except ImportError as e:
            results["errors"].append(f"{module_name}: {e}")
    
    return results


# =============================================================================
# Path Fixes
# =============================================================================

def fix_python_paths():
    """Add necessary paths to Python path."""
    paths_to_add = [
        str(Path.cwd()),
        str(Path.cwd() / "lean_workspace"),
        str(Path.cwd() / "glue" / "lib"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.info(f"Added to Python path: {path}")


def set_environment_variables(
    lean_exe: Optional[str],
    lake_exe: Optional[str],
    mathlib_path: Optional[str]
):
    """Set environment variables for Lean integration."""
    if lean_exe:
        os.environ['LEAN_EXECUTABLE'] = lean_exe
        logger.info(f"Set LEAN_EXECUTABLE={lean_exe}")
    
    if lake_exe:
        os.environ['LAKE_EXECUTABLE'] = lake_exe
        logger.info(f"Set LAKE_EXECUTABLE={lake_exe}")
    
    if mathlib_path:
        os.environ['MATHLIB_PATH'] = mathlib_path
        logger.info(f"Set MATHLIB_PATH={mathlib_path}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Activate and verify Lean 4 integration"
    )
    parser.add_argument(
        '--start-server',
        action='store_true',
        help='Start the LeanAide server if not running'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run integration tests'
    )
    parser.add_argument(
        '--fix-paths',
        action='store_true',
        help='Fix Python paths and environment variables'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    report = {
        "status": "checking",
        "lean": {},
        "mathlib": {},
        "server": {},
        "tests": {},
        "recommendations": []
    }
    
    # Step 1: Detect Lean
    logger.info("Step 1: Detecting Lean 4...")
    lean_exe = detect_lean_executable()
    if lean_exe:
        version = check_lean_version(lean_exe)
        report["lean"] = {
            "found": True,
            "path": lean_exe,
            "version": version
        }
    else:
        report["lean"] = {"found": False}
        report["recommendations"].append(
            "Install Lean 4 via elan: https://elan.readthedocs.io"
        )
    
    # Step 2: Detect lake
    logger.info("Step 2: Detecting lake...")
    lake_exe = detect_lake_executable()
    report["lean"]["lake_found"] = lake_exe is not None
    report["lean"]["lake_path"] = lake_exe
    
    # Step 3: Detect mathlib
    logger.info("Step 3: Detecting mathlib project...")
    mathlib_path = detect_mathlib_project()
    if mathlib_path:
        report["mathlib"] = {
            "found": True,
            "path": mathlib_path,
            "built": verify_mathlib_build(mathlib_path, lake_exe or "lake")
        }
    else:
        report["mathlib"] = {"found": False}
        report["recommendations"].append(
            "Set up mathlib4 project in lean_workspace/mathlib_project"
        )
    
    # Step 4: Fix paths if requested
    if args.fix_paths:
        logger.info("Step 4: Fixing Python paths...")
        fix_python_paths()
        set_environment_variables(lean_exe, lake_exe, mathlib_path)
        report["paths_fixed"] = True
    
    # Step 5: Check server
    logger.info("Step 5: Checking LeanAide server...")
    server_running = is_server_running()
    report["server"]["running"] = server_running
    
    if args.start_server and not server_running and mathlib_path:
        process = start_leanaide_server(mathlib_path)
        report["server"]["started"] = process is not None
        server_running = process is not None
    
    # Step 6: Run tests
    if args.test:
        logger.info("Step 6: Running integration tests...")
        
        # Import tests
        import_results = test_integration_imports()
        report["tests"]["imports"] = import_results
        
        # Client tests
        if import_results.get("leanaide_client"):
            client_results = await test_leanaide_client()
            report["tests"]["client"] = client_results
    
    # Determine overall status
    if lean_exe and mathlib_path:
        if report["mathlib"].get("built"):
            report["status"] = "ready"
        else:
            report["status"] = "mathlib_not_built"
    else:
        report["status"] = "missing_components"
    
    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n" + "="*60)
        print("LEAN 4 INTEGRATION STATUS REPORT")
        print("="*60)
        print(f"\nOverall Status: {report['status'].upper()}")
        
        print("\n--- Lean 4 ---")
        if report["lean"].get("found"):
            print(f"  [OK] Found: {report['lean']['path']}")
            print(f"  [OK] Version: {report['lean'].get('version', 'unknown')}")
            print(f"  [{'OK' if report['lean'].get('lake_found') else 'FAIL'}] lake: {report['lean'].get('lake_path') or 'not found'}")
        else:
            print("  [FAIL] Lean 4 not found")
        
        print("\n--- Mathlib4 ---")
        if report["mathlib"].get("found"):
            print(f"  [OK] Found: {report['mathlib']['path']}")
            print(f"  [{'OK' if report['mathlib'].get('built') else 'FAIL'}] Built: {report['mathlib'].get('built')}")
        else:
            print("  [FAIL] Mathlib4 not found")
        
        print("\n--- Server ---")
        print(f"  [{'OK' if report['server'].get('running') else 'INFO'}] Running: {report['server'].get('running')}")
        
        if report["recommendations"]:
            print("\n--- Recommendations ---")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        
        print("\n" + "="*60)
    
    # Exit code
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
