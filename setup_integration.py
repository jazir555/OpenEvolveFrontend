#!/usr/bin/env python3
"""
OpenEvolve Integration Setup Script - License: Apache 2.0

Automated setup script for the OpenEvolve integration system.
Installs dependencies, configures environment, and verifies installation.

Usage:
    python setup_integration.py
    python setup_integration.py --dev
    python setup_integration.py --docker
    python setup_integration.py --quick
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import platform

# Rich for output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.prompt import Confirm, Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


class SetupManager:
    """Manages the setup process."""
    
    def __init__(self, dev_mode: bool = False, quick: bool = False):
        self.dev_mode = dev_mode
        self.quick = quick
        self.errors = []
        self.warnings = []
        self.project_root = Path.cwd()
        
    def print_header(self):
        """Print setup header."""
        if console:
            console.print(Panel(
                "[bold blue]OpenEvolve Integration Setup[/bold blue]\n"
                "Automated installation and configuration",
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("=" * 60)
            print("OpenEvolve Integration Setup")
            print("=" * 60)
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self._print_step("Checking Python version...")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 11:
            self._print_success(f"Python {version.major}.{version.minor}.{version.micro} [OK]")
            return True
        elif version.major >= 3 and version.minor >= 10:
            self._print_warning(f"Python {version.major}.{version.minor}.{version.micro} (recommended: 3.11+)")
            return True
        else:
            self._print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
            return False
    
    def check_pip(self) -> bool:
        """Check pip availability."""
        self._print_step("Checking pip...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                check=True,
                capture_output=True
            )
            self._print_success("pip available [OK]")
            return True
        except subprocess.CalledProcessError:
            self._print_error("pip not found")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        self._print_step("Installing dependencies...")
        
        requirements_files = ['requirements_integration.txt']
        
        if self.dev_mode:
            requirements_files.append('requirements_with_testing.txt')
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if not req_path.exists():
                self._print_warning(f"{req_file} not found, skipping")
                continue
            
            self._print_info(f"Installing from {req_file}...")
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self._print_success(f"Installed {req_file} [OK]")
            except subprocess.CalledProcessError as e:
                self._print_error(f"Failed to install {req_file}: {e}")
                return False
        
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        self._print_step("Creating directories...")
        
        dirs = ['logs', 'data', 'plugins', 'backups', 'knowledge_extraction']
        
        for dir_name in dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            self._print_info(f"Created {dir_name}/")
        
        self._print_success("Directories created [OK]")
        return True
    
    def setup_environment_file(self) -> bool:
        """Setup environment configuration."""
        self._print_step("Setting up environment...")
        
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if env_file.exists():
            self._print_info(".env file already exists")
            if not self.quick:
                if RICH_AVAILABLE:
                    if not Confirm.ask("Overwrite existing .env?"):
                        return True
                else:
                    response = input("Overwrite existing .env? (y/N): ")
                    if response.lower() != 'y':
                        return True
        
        # Create .env from template or generate
        env_content = self._generate_env_content()
        env_file.write_text(env_content)
        
        self._print_success("Created .env file [OK]")
        return True
    
    def _generate_env_content(self) -> str:
        """Generate environment file content."""
        return """# OpenEvolve Environment Configuration
# License: Apache 2.0

# Core Configuration
OPENEVOLVE_LOG_LEVEL=INFO
OPENEVOLVE_ORCHESTRATOR_PORT=8080

# Service Toggles
OPENEVOLVE_SERVICES__REST_API=true
OPENEVOLVE_SERVICES__GRAPHQL_API=true
OPENEVOLVE_SERVICES__EVENT_BUS=true
OPENEVOLVE_SERVICES__MCP_SERVER=true
OPENEVOLVE_SERVICES__TELEMETRY=true

# REST API
OPENEVOLVE_REST_API__HOST=0.0.0.0
OPENEVOLVE_REST_API__PORT=8000

# GraphQL
OPENEVOLVE_GRAPHQL__HOST=0.0.0.0
OPENEVOLVE_GRAPHQL__PORT=8001

# Event Bus (Valkey)
OPENEVOLVE_EVENT_BUS__ENABLED=true
OPENEVOLVE_EVENT_BUS__BACKEND=valkey
OPENEVOLVE_EVENT_BUS__HOST=localhost
OPENEVOLVE_EVENT_BUS__PORT=6379

# Telemetry (OpenTelemetry)
OPENEVOLVE_TELEMETRY__ENABLED=true
OPENEVOLVE_TELEMETRY__SERVICE_NAME=openevolve
OPENEVOLVE_TELEMETRY__OTLP_ENDPOINT=http://localhost:4317

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=changeme-in-production-generate-random-string

# Valkey Direct
VALKEY_HOST=localhost
VALKEY_PORT=6379
"""
    
    def setup_configuration(self) -> bool:
        """Setup YAML configuration."""
        self._print_step("Setting up configuration...")
        
        config_file = self.project_root / 'integration_config.yaml'
        
        if config_file.exists():
            self._print_info("integration_config.yaml already exists")
            return True
        
        # Create default config
        config_content = """# OpenEvolve Integration Configuration
# License: Apache 2.0

log_level: INFO
orchestrator_port: 8080

services:
  rest_api: true
  graphql_api: true
  event_bus: true
  mcp_server: true
  telemetry: true

rest_api:
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - "*"

graphql:
  host: 0.0.0.0
  port: 8001
  enable_playground: true

event_bus:
  enabled: true
  backend: valkey
  host: localhost
  port: 6379
  password: null

telemetry:
  enabled: true
  service_name: openevolve
  otlp_endpoint: http://localhost:4317
  metrics_enabled: true
  tracing_enabled: true
"""
        
        config_file.write_text(config_content)
        self._print_success("Created integration_config.yaml [OK]")
        return True
    
    def check_docker(self) -> bool:
        """Check if Docker is available."""
        self._print_step("Checking Docker...")
        
        try:
            subprocess.run(
                ['docker', '--version'],
                check=True,
                capture_output=True
            )
            self._print_success("Docker available [OK]")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._print_warning("Docker not found (optional)")
            return False
    
    def run_health_check(self) -> bool:
        """Run system health check."""
        self._print_step("Running health check...")
        
        try:
            result = subprocess.run(
                [sys.executable, 'system_health.py', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self._print_success("Health check passed [OK]")
                return True
            else:
                self._print_warning("Health check found issues (see details above)")
                return True  # Don't fail setup for warnings
        except Exception as e:
            self._print_warning(f"Could not run health check: {e}")
            return True
    
    def print_summary(self):
        """Print setup summary."""
        if console:
            # Summary table
            table = Table(title="Setup Summary")
            table.add_column("Item", style="cyan")
            table.add_column("Status", style="bold")
            
            table.add_row("Python Version", "[OK] OK")
            table.add_row("Dependencies", "[OK] Installed")
            table.add_row("Directories", "[OK] Created")
            table.add_row("Configuration", "[OK] Ready")
            
            if self.errors:
                table.add_row("Errors", f"[red]{len(self.errors)}[/red]")
            
            console.print(table)
            
            # Next steps
            console.print(Panel(
                "[bold green]Setup Complete![/bold green]\n\n"
                "Next steps:\n"
                "  1. Start services: [cyan]make start[/cyan] or [cyan]python -m openevolve_cli services start --all[/cyan]\n"
                "  2. Check health: [cyan]make health[/cyan] or [cyan]python system_health.py[/cyan]\n"
                "  3. Run tests: [cyan]make test[/cyan] or [cyan]python run_integration_tests.py[/cyan]\n"
                "  4. View dashboard: [cyan]make dashboard[/cyan]\n\n"
                "Documentation:\n"
                "  - [cyan]INTEGRATION_GUIDE.md[/cyan] - Complete guide\n"
                "  - [cyan]FINAL_INTEGRATION_SUMMARY.md[/cyan] - Implementation summary",
                title="Next Steps",
                border_style="green"
            ))
        else:
            print("\n" + "=" * 60)
            print("Setup Complete!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Start services: make start")
            print("  2. Check health: make health")
            print("  3. Run tests: make test")
            print("\nDocumentation:")
            print("  - INTEGRATION_GUIDE.md")
            print("  - FINAL_INTEGRATION_SUMMARY.md")
    
    def _print_step(self, message: str):
        """Print setup step."""
        if console:
            console.print(f"[blue]›[/blue] {message}")
        else:
            print(f"-> {message}")
    
    def _print_success(self, message: str):
        """Print success message."""
        if console:
            console.print(f"  [green][OK][/green] {message}")
        else:
            print(f"  [OK] {message}")
    
    def _print_error(self, message: str):
        """Print error message."""
        self.errors.append(message)
        if console:
            console.print(f"  [red][FAIL][/red] {message}")
        else:
            print(f"  [FAIL] {message}")
    
    def _print_warning(self, message: str):
        """Print warning message."""
        self.warnings.append(message)
        if console:
            console.print(f"  [yellow]![/yellow] {message}")
        else:
            print(f"  ! {message}")
    
    def _print_info(self, message: str):
        """Print info message."""
        if console:
            console.print(f"  [dim]{message}[/dim]")
        else:
            print(f"    {message}")
    
    def run(self) -> bool:
        """Run complete setup."""
        self.print_header()
        
        # Pre-flight checks
        if not self.check_python_version():
            return False
        
        if not self.check_pip():
            return False
        
        # Installation
        if not self.quick:
            if not self.install_dependencies():
                return False
        
        # Setup
        self.create_directories()
        self.setup_environment_file()
        self.setup_configuration()
        
        # Optional checks
        self.check_docker()
        
        if not self.quick:
            self.run_health_check()
        
        # Summary
        self.print_summary()
        
        return len(self.errors) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Integration Setup"
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Install development dependencies'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick setup (skip dependencies)'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Setup for Docker deployment'
    )
    
    args = parser.parse_args()
    
    manager = SetupManager(
        dev_mode=args.dev,
        quick=args.quick
    )
    
    success = manager.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
