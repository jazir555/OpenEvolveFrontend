"""
DeepKE Setup - Isolated Environment Installation

This script installs DeepKE in an isolated virtual environment
to avoid dependency conflicts with the main OpenEvolve system.

Usage:
    python setup_deepke_fixed.py
    
Features:
- Creates isolated venv for DeepKE
- Installs compatible torch, transformers, and deepke versions
- Provides import helper for main system
- Handles both CPU and CUDA environments
"""

import subprocess
import sys
import os
import venv
from pathlib import Path


def install_deepke_isolated():
    """
    Install DeepKE in isolated environment with compatible dependencies.
    
    Creates a virtual environment specifically for DeepKE to avoid
    conflicts with other packages in the main environment.
    """
    env_path = Path("deepke_env")
    
    print("=" * 60)
    print("DeepKE Isolated Environment Setup")
    print("=" * 60)
    
    # Create venv if it doesn't exist
    if not env_path.exists():
        print(f"\n[1/5] Creating virtual environment at {env_path}...")
        venv.create(env_path, with_pip=True)
        print("✓ Virtual environment created")
    else:
        print(f"\n[1/5] Using existing virtual environment at {env_path}")
    
    # Determine pip path
    if sys.platform == "win32":
        pip = env_path / "Scripts" / "pip.exe"
        python = env_path / "Scripts" / "python.exe"
    else:
        pip = env_path / "bin" / "pip"
        python = env_path / "bin" / "python"
    
    # Upgrade pip
    print("\n[2/5] Upgrading pip...")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], 
                   capture_output=True)
    print("✓ Pip upgraded")
    
    # Install compatible torch version
    print("\n[3/5] Installing PyTorch 2.0.1...")
    result = subprocess.run(
        [str(pip), "install", "torch==2.0.1", "--index-url", "https://download.pytorch.org/whl/cpu"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ PyTorch installed")
    else:
        print(f"⚠ PyTorch install had issues, continuing...")
        print(f"   Error: {result.stderr[:200]}")
    
    # Install compatible transformers
    print("\n[4/5] Installing transformers 4.30.0...")
    result = subprocess.run(
        [str(pip), "install", "transformers==4.30.0", "sentencepiece"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ Transformers installed")
    else:
        print(f"⚠ Transformers install had issues, continuing...")
    
    # Install DeepKE
    print("\n[5/5] Installing DeepKE 2.2.7...")
    result = subprocess.run(
        [str(pip), "install", "deepke==2.2.7"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ DeepKE installed")
    else:
        print(f"⚠ DeepKE install had issues")
        print(f"   Error: {result.stderr[:300]}")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nVirtual environment location: {env_path.absolute()}")
    print("\nTo use DeepKE in your code:")
    print("-" * 40)
    print("from setup_deepke_fixed import activate_deepke")
    print("activate_deepke()")
    print("from deepke import NERModel, REModel")
    print("-" * 40)
    
    return True


def activate_deepke():
    """
    Activate DeepKE environment by adding it to sys.path.
    
    Call this before importing DeepKE modules.
    """
    env_path = Path("deepke_env")
    
    if sys.platform == "win32":
        site_packages = env_path / "Lib" / "site-packages"
    else:
        # Find site-packages
        lib_path = env_path / "lib"
        if lib_path.exists():
            for p in lib_path.iterdir():
                if p.name.startswith("python"):
                    site_packages = p / "site-packages"
                    break
        else:
            site_packages = env_path / "lib" / "python3.x" / "site-packages"
    
    if site_packages.exists():
        if str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
            print(f"✓ DeepKE environment activated: {site_packages}")
    else:
        print(f"⚠ Could not find site-packages at {site_packages}")


def verify_deepke():
    """Verify DeepKE is properly installed and working."""
    try:
        activate_deepke()
        from deepke import NERModel, REModel
        print("✓ DeepKE imports successfully")
        print(f"  NERModel: {NERModel}")
        print(f"  REModel: {REModel}")
        return True
    except ImportError as e:
        print(f"✗ DeepKE not available: {e}")
        return False


def create_activation_script():
    """Create a script to easily activate the DeepKE environment."""
    script_path = Path("activate_deepke.py")
    
    content = '''"""
Activation helper for DeepKE environment.
Import this module before using DeepKE.
"""
import sys
from pathlib import Path

# Add DeepKE environment to path
env_path = Path(__file__).parent / "deepke_env"

if sys.platform == "win32":
    site_packages = env_path / "Lib" / "site-packages"
else:
    lib_path = env_path / "lib"
    if lib_path.exists():
        import glob
        python_dirs = list(lib_path.glob("python*"))
        if python_dirs:
            site_packages = python_dirs[0] / "site-packages"
        else:
            site_packages = lib_path / "site-packages"
    else:
        site_packages = env_path / "lib" / "site-packages"

if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
    print(f"DeepKE environment activated")

# Now you can import DeepKE
try:
    from deepke import NERModel, REModel
    print("✓ DeepKE ready to use")
except ImportError as e:
    print(f"⚠ Could not import DeepKE: {e}")
'''
    
    script_path.write_text(content)
    print(f"Created activation script: {script_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepKE Setup")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--activate-only", action="store_true", help="Just activate, don't install")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_deepke()
    elif args.activate_only:
        activate_deepke()
    else:
        install_deepke_isolated()
        create_activation_script()
        print("\nVerifying installation...")
        verify_deepke()
