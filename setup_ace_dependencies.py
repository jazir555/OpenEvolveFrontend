#!/usr/bin/env python3
"""
Setup ACE Dependencies

This script ensures all required dependencies for ACE (Agentic Context Engine)
are properly installed. Run this before using the ACE CrewAI Bridge.

Usage:
    python setup_ace_dependencies.py
"""

import subprocess
import sys
import os
from typing import List, Tuple

def run_command(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_package(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_spec: str) -> bool:
    """Install a package using pip."""
    print(f"Installing {package_spec}...")
    success, output = run_command([sys.executable, "-m", "pip", "install", package_spec])
    if success:
        print(f"  ✓ Successfully installed {package_spec}")
        return True
    else:
        print(f"  ✗ Failed to install {package_spec}")
        print(f"  Error: {output}")
        return False

def main():
    """Main setup function."""
    print("=" * 70)
    print("ACE (Agentic Context Engine) Dependency Setup")
    print("=" * 70)
    print()

    # Core ACE dependencies from agentic-context-engine/pyproject.toml
    dependencies = [
        ("litellm", "litellm>=1.78.0"),
        ("pydantic", "pydantic>=2.0.0"),
        ("python-dotenv", "python-dotenv>=1.0.0"),
        ("toon", "python-toon>=0.1.0"),
        ("tenacity", "tenacity>=8.0.0"),
        ("instructor", "instructor>=1.0.0"),
    ]

    print("Checking dependencies...")
    print()

    missing_deps = []
    for module_name, package_spec in dependencies:
        if check_package(module_name):
            print(f"  ✓ {module_name} is installed")
        else:
            print(f"  ✗ {module_name} is missing")
            missing_deps.append(package_spec)

    print()

    if not missing_deps:
        print("✓ All ACE dependencies are already installed!")
        print()
        print("You can now use the ACE CrewAI Bridge:")
        print("  from ace_crewai_bridge import ACECrewAIWorkflowBridge")
        print()
        return 0

    print(f"Installing {len(missing_deps)} missing dependencies...")
    print()

    failed_installs = []
    for package_spec in missing_deps:
        if not install_package(package_spec):
            failed_installs.append(package_spec)

    print()

    if failed_installs:
        print(f"✗ Failed to install {len(failed_installs)} packages:")
        for pkg in failed_installs:
            print(f"  - {pkg}")
        print()
        print("Please install them manually:")
        print(f"  pip install {' '.join(failed_installs)}")
        return 1

    print("=" * 70)
    print("✓ ACE dependency setup complete!")
    print("=" * 70)
    print()
    print("You can now use the ACE CrewAI Bridge:")
    print("  from ace_crewai_bridge import ACECrewAIWorkflowBridge")
    print()
    print("Example usage:")
    print("  bridge = ACECrewAIWorkflowBridge(model='gpt-4o-mini')")
    print("  result = bridge.execute_phase_1_setup('Solve this problem...')")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
