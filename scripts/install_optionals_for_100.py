#!/usr/bin/env python3
"""
Install Optional Dependencies for TRUE 100% - License: Apache 2.0

Installs all optional dependencies to achieve TRUE 100% integration.
"""

import subprocess
import sys

def install_package(package, version=None):
    """Install a package."""
    spec = f"{package}>={version}" if version else package
    print(f"Installing {spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        print(f"  [OK] {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Failed to install {package}: {e}")
        return False

def main():
    print("=" * 70)
    print("Installing Optional Dependencies for TRUE 100% Integration")
    print("=" * 70)
    print()
    
    packages = [
        ("mcp", "1.0.0"),              # For Unified MCP Server
        ("strawberry-graphql", "0.215.0"),  # For GraphQL API
        ("valkey-py", "0.1.0"),        # For production Event Bus
        ("opentelemetry-api", "1.21.0"),    # For full Telemetry
        ("opentelemetry-sdk", "1.21.0"),
        ("opentelemetry-instrumentation-fastapi", "0.42b0"),
    ]
    
    installed = 0
    failed = 0
    
    for package, version in packages:
        if install_package(package, version):
            installed += 1
        else:
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Installation Complete: {installed} succeeded, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n[OK] All optional dependencies installed!")
        print("Run TRUE_100_INTEGRATION.py to verify TRUE 100% completion")
        return 0
    else:
        print(f"\n[WARN] {failed} packages failed to install")
        return 1

if __name__ == "__main__":
    sys.exit(main())
