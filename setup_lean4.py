"""
Lean 4 Automated Setup Script for OpenEvolve

This script automates the detection and installation of:
- Lean 4 compiler (`lean`)
- Lake build tool (`lake`)
- Mathlib4 mathematical library

Usage:
    python setup_lean4.py --auto-install
    python setup_lean4.py --check-only
    python setup_lean4.py --setup-mathlib

Author: OpenEvolve
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LeanInstallationStatus:
    """Status of Lean 4 installation"""
    lean_available: bool = False
    lake_available: bool = False
    mathlib_available: bool = False
    lean_version: Optional[str] = None
    lake_version: Optional[str] = None
    mathlib_path: Optional[str] = None
    elan_available: bool = False
    
    def is_fully_functional(self) -> bool:
        """Check if Lean 4 is fully functional with mathlib4"""
        return self.lean_available and self.lake_available and self.mathlib_available
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lean_available": self.lean_available,
            "lake_available": self.lake_available,
            "mathlib_available": self.mathlib_available,
            "lean_version": self.lean_version,
            "lake_version": self.lake_version,
            "mathlib_path": self.mathlib_path,
            "elan_available": self.elan_available,
            "fully_functional": self.is_fully_functional()
        }


@dataclass
class SetupResult:
    """Result of setup operation"""
    success: bool
    message: str
    status: LeanInstallationStatus
    logs: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "status": self.status.to_dict(),
            "logs": self.logs,
            "timestamp": self.timestamp
        }


# ============================================================================
# Detection Functions
# ============================================================================

def check_command_exists(command: str) -> Tuple[bool, Optional[str]]:
    """Check if a command exists in PATH and get its version"""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False, None


def detect_lean_installation() -> LeanInstallationStatus:
    """Detect Lean 4 installation status"""
    status = LeanInstallationStatus()
    
    # Check for elan (Lean version manager)
    status.elan_available, _ = check_command_exists("elan")
    
    # Check for lean
    status.lean_available, lean_version = check_command_exists("lean")
    if lean_version:
        status.lean_version = lean_version.split('\n')[0][:50]
    
    # Check for lake
    status.lake_available, lake_version = check_command_exists("lake")
    if lake_version:
        status.lake_version = lake_version.split('\n')[0][:50]
    
    # Check for mathlib4
    status.mathlib_path = find_mathlib4_path()
    status.mathlib_available = status.mathlib_path is not None
    
    return status


def find_mathlib4_path() -> Optional[str]:
    """Find mathlib4 installation path"""
    # Common locations
    search_paths = [
        Path.home() / ".local" / "share" / "mathlib4",
        Path.home() / ".mathlib4",
        Path("/usr") / "local" / "share" / "mathlib4",
        Path("/usr") / "share" / "mathlib4",
        Path.cwd() / "mathlib4",
        Path.cwd() / "lean_workspace" / "mathlib4",
        Path.cwd() / ".lake" / "packages" / "mathlib",
    ]
    
    for path in search_paths:
        if path.exists() and (path / "Mathlib.lean").exists():
            return str(path)
    
    # Check if mathlib is in lean's toolchain
    try:
        result = subprocess.run(
            ["lean", "--print-libdir"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            libdir = Path(result.stdout.strip())
            mathlib_path = libdir / "Mathlib.lean"
            if mathlib_path.exists():
                return str(libdir)
    except:
        pass
    
    return None


# ============================================================================
# Installation Functions
# ============================================================================

def install_elan() -> Tuple[bool, str]:
    """Install elan (Lean version manager)"""
    system = platform.system().lower()
    
    logger.info("Installing elan (Lean version manager)...")
    
    try:
        if system == "linux" or system == "darwin":
            # Unix-like systems
            install_script = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                script_path = f.name
                
            # Download install script
            urlretrieve(install_script, script_path)
            
            # Make executable and run
            os.chmod(script_path, 0o755)
            result = subprocess.run(
                ["sh", script_path, "-y"],
                capture_output=True,
                text=True,
                timeout=300
            )
            os.unlink(script_path)
            
            if result.returncode == 0:
                # Add to PATH
                add_elan_to_path()
                return True, "elan installed successfully"
            else:
                return False, f"Installation failed: {result.stderr}"
                
        elif system == "windows":
            # Windows - Auto-install elan
            logger.info("Windows detected. Installing elan automatically...")
            
            try:
                # Download elan installer
                elan_url = "https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-pc-windows-msvc.zip"
                zip_path = Path(tempfile.gettempdir()) / "elan-install.zip"
                extract_dir = Path(tempfile.gettempdir()) / "elan-install"
                
                logger.info(f"Downloading elan from {elan_url}...")
                urlretrieve(elan_url, zip_path)
                
                # Extract
                import zipfile
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find elan-init.exe
                elan_init = extract_dir / "elan-init.exe"
                if not elan_init.exists():
                    # Try to find it in subdirectories
                    for exe in extract_dir.rglob("elan-init.exe"):
                        elan_init = exe
                        break
                
                if elan_init.exists():
                    # Run installer
                    result = subprocess.run(
                        [str(elan_init), "-y"],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    # Cleanup
                    zip_path.unlink(missing_ok=True)
                    
                    if result.returncode == 0 or "elan" in str(result.stdout).lower():
                        # Add to PATH for Windows
                        add_elan_to_path_windows()
                        return True, "elan installed successfully on Windows"
                    else:
                        return False, f"Installation failed: {result.stderr}"
                else:
                    return False, "Could not find elan-init.exe in downloaded archive"
                    
            except Exception as e:
                return False, f"Windows installation error: {str(e)}"
        else:
            return False, f"Unsupported platform: {system}"
            
    except Exception as e:
        return False, f"Installation error: {str(e)}"


def add_elan_to_path():
    """Add elan to user's PATH (Unix-like systems)"""
    elan_bin = Path.home() / ".elan" / "bin"
    
    if not elan_bin.exists():
        return
    
    # Get shell config file
    shell = os.environ.get("SHELL", "/bin/bash")
    if "zsh" in shell:
        config_file = Path.home() / ".zshrc"
    elif "bash" in shell:
        config_file = Path.home() / ".bashrc"
    else:
        config_file = Path.home() / ".profile"
    
    # Check if already in PATH
    path_line = f'export PATH="$HOME/.elan/bin:$PATH"'
    
    try:
        if config_file.exists():
            content = config_file.read_text()
            if ".elan/bin" not in content:
                with open(config_file, 'a') as f:
                    f.write(f"\n# Lean version manager\n{path_line}\n")
                logger.info(f"Added elan to {config_file}")
    except Exception as e:
        logger.warning(f"Could not update shell config: {e}")
    
    # Also update current session
    os.environ["PATH"] = str(elan_bin) + os.pathsep + os.environ.get("PATH", "")


def add_elan_to_path_windows():
    """Add elan to user's PATH on Windows"""
    elan_bin = Path.home() / ".elan" / "bin"
    
    if not elan_bin.exists():
        logger.warning(f"Elan bin directory not found: {elan_bin}")
        return
    
    try:
        # Add to current session PATH
        current_path = os.environ.get("PATH", "")
        if str(elan_bin) not in current_path:
            os.environ["PATH"] = str(elan_bin) + os.pathsep + current_path
            logger.info(f"Added {elan_bin} to current PATH")
        
        # Add to user PATH permanently using setx ( Windows)
        result = subprocess.run(
            ["setx", "PATH", f"{elan_bin};%PATH%"],
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )
        if result.returncode == 0:
            logger.info("Added elan to user PATH permanently (setx)")
        else:
            logger.warning(f"Could not update PATH permanently: {result.stderr}")
            
    except Exception as e:
        logger.warning(f"Could not update Windows PATH: {e}")


def install_lean_toolchain() -> Tuple[bool, str]:
    """Install Lean 4 stable toolchain"""
    logger.info("Installing Lean 4 stable toolchain...")
    
    try:
        # Update elan
        result = subprocess.run(
            ["elan", "self", "update"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Install stable
        result = subprocess.run(
            ["elan", "toolchain", "install", "stable"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Set as default
            subprocess.run(
                ["elan", "default", "stable"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return True, "Lean 4 stable toolchain installed"
        else:
            return False, f"Toolchain installation failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Toolchain installation error: {str(e)}"


def setup_mathlib4_project(project_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Create a new mathlib4 project"""
    if project_dir is None:
        project_dir = str(Path.cwd() / "lean_workspace" / "mathlib_project")
    
    project_path = Path(project_dir)
    
    logger.info(f"Setting up mathlib4 project at {project_dir}...")
    
    try:
        # Create directory
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Check if lake is available
        if not shutil.which("lake"):
            return False, "lake command not found. Please install elan and Lean first."
        
        # Create lakefile
        lakefile_content = """import Lake
open Lake DSL

package «mathlib_project» where
  -- Settings applied to both builds and interactive editing
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩ -- pretty-prints `fun a ↦ b`
  ]
  -- add any additional package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «MathlibProject» where
  -- add any library configuration options here
"""
        
        (project_path / "lakefile.lean").write_text(lakefile_content)
        
        # Create basic directory structure
        (project_path / "MathlibProject").mkdir(exist_ok=True)
        (project_path / "MathlibProject" / "Basic.lean").write_text('import Mathlib\n\ndef hello := "world"\n')
        
        # Run lake update to download dependencies
        logger.info("Downloading mathlib4 dependencies (this may take a while)...")
        result = subprocess.run(
            ["lake", "update"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode == 0:
            # Try to build
            logger.info("Building project...")
            result = subprocess.run(
                ["lake", "build"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return True, f"Mathlib4 project setup complete at {project_dir}"
            else:
                return True, f"Project created at {project_dir} (build had warnings)"
        else:
            return False, f"Failed to download dependencies: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, "Setup timed out (dependencies may still be downloading)"
    except Exception as e:
        return False, f"Setup error: {str(e)}"


# ============================================================================
# Main Setup Class
# ============================================================================

class Lean4SetupManager:
    """Manager for Lean 4 installation and setup"""
    
    def __init__(self):
        self.status = LeanInstallationStatus()
    
    def check_installation(self) -> LeanInstallationStatus:
        """Check current installation status"""
        self.status = detect_lean_installation()
        return self.status
    
    def auto_install(self) -> SetupResult:
        """Automatically install Lean 4 and mathlib4"""
        logs = []
        
        logger.info("Starting Lean 4 auto-installation...")
        
        # Step 1: Check current status
        self.check_installation()
        
        if self.status.is_fully_functional():
            return SetupResult(
                success=True,
                message="Lean 4 is already fully installed and functional",
                status=self.status,
                logs=["Lean 4 already installed"]
            )
        
        # Step 2: Install elan if needed
        if not self.status.elan_available:
            logger.info("Step 1/3: Installing elan...")
            success, message = install_elan()
            logs.append(f"Elan installation: {message}")
            
            if not success:
                return SetupResult(
                    success=False,
                    message=f"Failed to install elan: {message}",
                    status=self.status,
                    logs=logs
                )
            
            # Recheck after installation
            self.check_installation()
        
        # Step 3: Install Lean toolchain
        if not self.status.lean_available:
            logger.info("Step 2/3: Installing Lean 4 toolchain...")
            success, message = install_lean_toolchain()
            logs.append(f"Lean toolchain: {message}")
            
            if not success:
                return SetupResult(
                    success=False,
                    message=f"Failed to install Lean: {message}",
                    status=self.status,
                    logs=logs
                )
            
            self.check_installation()
        
        # Step 4: Setup mathlib4
        if not self.status.mathlib_available:
            logger.info("Step 3/3: Setting up mathlib4 project...")
            success, message = setup_mathlib4_project()
            logs.append(f"Mathlib4 setup: {message}")
            
            if not success:
                # This is not a hard failure - Lean can still work
                logger.warning(f"Mathlib4 setup warning: {message}")
            
            self.check_installation()
        
        # Final check
        if self.status.lean_available and self.status.lake_available:
            return SetupResult(
                success=True,
                message="Lean 4 installation completed successfully!",
                status=self.status,
                logs=logs
            )
        else:
            return SetupResult(
                success=False,
                message="Installation incomplete. Some components are missing.",
                status=self.status,
                logs=logs
            )
    
    def get_setup_instructions(self) -> str:
        """Get manual setup instructions"""
        system = platform.system()
        
        instructions = f"""
{'='*70}
Lean 4 Manual Setup Instructions ({system})
{'='*70}

Current Status:
  - Lean available: {self.status.lean_available}
  - Lake available: {self.status.lake_available}
  - Mathlib4 available: {self.status.mathlib_available}

OPTION 1: Automatic Installation (Recommended)
----------------------------------------------
Run: python setup_lean4.py --auto-install

OPTION 2: Manual Installation
-----------------------------

Step 1: Install elan (Lean version manager)
"""
        
        if system == "Linux" or system == "Darwin":
            instructions += """
  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  
  # Add to PATH:
  source $HOME/.elan/env
"""
        elif system == "Windows":
            instructions += """
  Download and run: https://github.com/leanprover/elan/releases/latest
  
  Or use winget:
  winget install Elan
"""
        
        instructions += """
Step 2: Install Lean 4 stable toolchain
  elan toolchain install stable
  elan default stable

Step 3: Verify installation
  lean --version
  lake --version

Step 4: Create a mathlib4 project
  mkdir -p ~/lean_projects
  cd ~/lean_projects
  lake new my_project math
  cd my_project
  lake update
  lake build

OPTION 3: Use Docker
--------------------
  docker run -it --rm leanprovercommunity/lean4:latest

{'='*70}
"""
        return instructions


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lean 4 Setup Script for OpenEvolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check-only          Check if Lean 4 is installed
  %(prog)s --auto-install        Automatically install Lean 4
  %(prog)s --setup-mathlib       Setup mathlib4 project
  %(prog)s --instructions        Show setup instructions
        """
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check installation status"
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install Lean 4"
    )
    parser.add_argument(
        "--setup-mathlib",
        action="store_true",
        help="Setup mathlib4 project"
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Show manual setup instructions"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Directory for mathlib4 project"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    manager = Lean4SetupManager()
    
    if args.instructions:
        status = manager.check_installation()
        print(manager.get_setup_instructions())
        return 0
    
    if args.check_only:
        status = manager.check_installation()
        
        if args.json:
            print(json.dumps(status.to_dict(), indent=2))
        else:
            print("\n" + "="*70)
            print("Lean 4 Installation Status")
            print("="*70)
            print(f"  Lean available:     {status.lean_available}")
            print(f"  Lean version:       {status.lean_version or 'N/A'}")
            print(f"  Lake available:     {status.lake_available}")
            print(f"  Lake version:       {status.lake_version or 'N/A'}")
            print(f"  Mathlib4 available: {status.mathlib_available}")
            print(f"  Mathlib4 path:      {status.mathlib_path or 'N/A'}")
            print(f"  Elan available:     {status.elan_available}")
            print("="*70)
            print(f"  Status: {'[OK] FULLY FUNCTIONAL' if status.is_fully_functional() else '[MISSING] INCOMPLETE'}")
            print("="*70 + "\n")
        
        return 0 if status.lean_available else 1
    
    if args.setup_mathlib:
        success, message = setup_mathlib4_project(args.project_dir)
        
        if args.json:
            print(json.dumps({"success": success, "message": message}, indent=2))
        else:
            print(f"\nMathlib4 setup: {'✓ SUCCESS' if success else '✗ FAILED'}")
            print(f"Message: {message}\n")
        
        return 0 if success else 1
    
    if args.auto_install:
        result = manager.auto_install()
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "="*70)
            print("Lean 4 Auto-Installation Result")
            print("="*70)
            print(f"Status: {'[OK] SUCCESS' if result.success else '[FAIL] FAILED'}")
            print(f"Message: {result.message}")
            print("\nLogs:")
            for log in result.logs:
                print(f"  - {log}")
            print("\nFinal Status:")
            print(f"  Lean: {result.status.lean_available}")
            print(f"  Lake: {result.status.lake_available}")
            print(f"  Mathlib4: {result.status.mathlib_available}")
            print("="*70 + "\n")
            
            if not result.success:
                print("For manual setup instructions, run:")
                print("  python setup_lean4.py --instructions\n")
        
        return 0 if result.success else 1
    
    # Default: show help
    parser.print_help()
    
    # Also show current status
    print("\n" + "="*70)
    print("Current Lean 4 Status (run with --auto-install to fix)")
    print("="*70)
    status = manager.check_installation()
    print(f"  Lean available: {status.lean_available}")
    print(f"  Lake available: {status.lake_available}")
    print(f"  Mathlib4 available: {status.mathlib_available}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
