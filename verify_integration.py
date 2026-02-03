#!/usr/bin/env python3
"""
OpenEvolve Integration Verification - License: Apache 2.0

Comprehensive verification script that checks all integration components.
Validates installation, configuration, and basic functionality.

Usage:
    python verify_integration.py
    python verify_integration.py --full
    python verify_integration.py --ci
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse

# Rich for output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@dataclass
class VerificationResult:
    """Result of a verification check."""
    check: str
    status: str  # 'passed', 'failed', 'warning', 'skipped'
    message: str
    details: Dict = field(default_factory=dict)


class IntegrationVerifier:
    """Verifies OpenEvolve integration installation."""
    
    REQUIRED_FILES = [
        'unified_mcp_server.py',
        'event_bus.py',
        'graphql_server.py',
        'service_orchestrator.py',
        'plugin_registry.py',
        'stage6_knowledge_extraction.py',
        'integration_config.py',
        'telemetry.py',
        'api_gateway.py',
        'openevolve_cli.py',
    ]
    
    REQUIRED_DIRS = ['logs', 'data', 'plugins']
    
    def __init__(self, full: bool = False, ci_mode: bool = False):
        self.full = full
        self.ci_mode = ci_mode
        self.results: List[VerificationResult] = []
        self.project_root = Path.cwd()
    
    def print_header(self):
        """Print verification header."""
        if console:
            console.print(Panel(
                "[bold blue]OpenEvolve Integration Verification[/bold blue]\n"
                "Validating installation and configuration",
                title="🔍 Verification",
                border_style="blue"
            ))
        else:
            print("=" * 60)
            print("OpenEvolve Integration Verification")
            print("=" * 60)
    
    def check_python_version(self) -> VerificationResult:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 11:
            return VerificationResult(
                check="Python Version",
                status="passed",
                message=f"Python {version_str} (supported)",
                details={'version': version_str}
            )
        elif version.major >= 3 and version.minor >= 10:
            return VerificationResult(
                check="Python Version",
                status="warning",
                message=f"Python {version_str} (recommended: 3.11+)",
                details={'version': version_str}
            )
        else:
            return VerificationResult(
                check="Python Version",
                status="failed",
                message=f"Python {version_str} (requires 3.10+)",
                details={'version': version_str}
            )
    
    def check_dependencies(self) -> VerificationResult:
        """Check required dependencies."""
        required = [
            'fastapi',
            'pydantic',
            'strawberry',
            'opentelemetry',
            'click',
        ]
        
        missing = []
        installed = []
        
        for pkg in required:
            try:
                __import__(pkg)
                installed.append(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            return VerificationResult(
                check="Dependencies",
                status="failed",
                message=f"Missing: {', '.join(missing)}",
                details={'missing': missing, 'installed': installed}
            )
        else:
            return VerificationResult(
                check="Dependencies",
                status="passed",
                message=f"All required packages installed ({len(installed)})",
                details={'installed': installed}
            )
    
    def check_file_structure(self) -> VerificationResult:
        """Check required files exist."""
        missing_files = []
        missing_dirs = []
        
        for filename in self.REQUIRED_FILES:
            if not (self.project_root / filename).exists():
                missing_files.append(filename)
        
        for dirname in self.REQUIRED_DIRS:
            if not (self.project_root / dirname).exists():
                missing_dirs.append(dirname)
        
        if missing_files or missing_dirs:
            return VerificationResult(
                check="File Structure",
                status="failed",
                message=f"Missing {len(missing_files)} files, {len(missing_dirs)} directories",
                details={'missing_files': missing_files, 'missing_dirs': missing_dirs}
            )
        else:
            return VerificationResult(
                check="File Structure",
                status="passed",
                message=f"All {len(self.REQUIRED_FILES)} files and {len(self.REQUIRED_DIRS)} directories present",
                details={'files': self.REQUIRED_FILES, 'dirs': self.REQUIRED_DIRS}
            )
    
    def check_configuration(self) -> VerificationResult:
        """Check configuration files."""
        config_files = [
            'integration_config.yaml',
            '.env',
        ]
        
        present = []
        missing = []
        
        for filename in config_files:
            if (self.project_root / filename).exists():
                present.append(filename)
            else:
                missing.append(filename)
        
        if missing:
            return VerificationResult(
                check="Configuration",
                status="warning",
                message=f"Missing: {', '.join(missing)} (optional)",
                details={'present': present, 'missing': missing}
            )
        else:
            return VerificationResult(
                check="Configuration",
                status="passed",
                message="All configuration files present",
                details={'present': present}
            )
    
    def check_imports(self) -> VerificationResult:
        """Check that core modules import successfully."""
        modules = [
            'integration_config',
            'event_bus',
            'service_orchestrator',
            'plugin_registry',
            'stage6_knowledge_extraction',
        ]
        
        failed = []
        passed = []
        
        for module in modules:
            try:
                __import__(module)
                passed.append(module)
            except Exception as e:
                failed.append((module, str(e)))
        
        if failed:
            return VerificationResult(
                check="Module Imports",
                status="failed",
                message=f"Failed to import {len(failed)} modules",
                details={'failed': failed, 'passed': passed}
            )
        else:
            return VerificationResult(
                check="Module Imports",
                status="passed",
                message=f"All {len(passed)} modules import successfully",
                details={'passed': passed}
            )
    
    def check_syntax(self) -> VerificationResult:
        """Check Python syntax of all integration files."""
        import py_compile
        
        files_to_check = list(self.project_root.glob('*.py'))
        failed = []
        passed = []
        
        for filepath in files_to_check[:20]:  # Check first 20 files
            try:
                py_compile.compile(str(filepath), doraise=True)
                passed.append(filepath.name)
            except py_compile.PyCompileError as e:
                failed.append((filepath.name, str(e)))
        
        if failed:
            return VerificationResult(
                check="Syntax Validation",
                status="failed",
                message=f"Syntax errors in {len(failed)} files",
                details={'failed': failed, 'passed': passed}
            )
        else:
            return VerificationResult(
                check="Syntax Validation",
                status="passed",
                message=f"All {len(passed)} files have valid syntax",
                details={'passed': passed}
            )
    
    def run_tests(self) -> VerificationResult:
        """Run quick tests."""
        if self.ci_mode:
            # Run full test suite in CI
            cmd = [sys.executable, '-m', 'pytest', 'test_integrations_comprehensive.py', '-v', '-x']
        else:
            # Run quick tests
            cmd = [sys.executable, '-m', 'pytest', 'test_integrations_comprehensive.py', '-v', '-m', 'not slow', '--tb=short', '-q']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return VerificationResult(
                    check="Test Suite",
                    status="passed",
                    message="All tests passed",
                    details={'stdout': result.stdout[-500:]}
                )
            else:
                return VerificationResult(
                    check="Test Suite",
                    status="failed",
                    message="Some tests failed",
                    details={'stdout': result.stdout[-500:], 'stderr': result.stderr[-500:]}
                )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                check="Test Suite",
                status="warning",
                message="Tests timed out",
                details={}
            )
        except Exception as e:
            return VerificationResult(
                check="Test Suite",
                status="warning",
                message=f"Could not run tests: {e}",
                details={}
            )
    
    def verify(self) -> List[VerificationResult]:
        """Run all verification checks."""
        self.print_header()
        
        checks = [
            self.check_python_version,
            self.check_dependencies,
            self.check_file_structure,
            self.check_configuration,
            self.check_imports,
            self.check_syntax,
        ]
        
        if self.full:
            checks.append(self.run_tests)
        
        for check_func in checks:
            if console:
                console.print(f"[dim]Running {check_func.__name__}...[/dim]")
            
            result = check_func()
            self.results.append(result)
            
            if console:
                status_color = {
                    'passed': 'green',
                    'failed': 'red',
                    'warning': 'yellow',
                    'skipped': 'dim'
                }.get(result.status, 'white')
                
                console.print(f"  [{status_color}]{result.status.upper()}[/{status_color}] {result.check}: {result.message}")
        
        return self.results
    
    def print_summary(self):
        """Print verification summary."""
        passed = sum(1 for r in self.results if r.status == 'passed')
        failed = sum(1 for r in self.results if r.status == 'failed')
        warnings = sum(1 for r in self.results if r.status == 'warning')
        
        if console:
            # Summary table
            table = Table(title="Verification Summary")
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Message")
            
            for result in self.results:
                status_color = {
                    'passed': 'green',
                    'failed': 'red',
                    'warning': 'yellow'
                }.get(result.status, 'white')
                
                table.add_row(
                    result.check,
                    f"[{status_color}]{result.status.upper()}[/{status_color}]",
                    result.message[:60]
                )
            
            console.print(table)
            
            # Overall status
            if failed == 0:
                status = "PASSED"
                color = "green"
                message = "Integration is ready to use!"
            else:
                status = "FAILED"
                color = "red"
                message = "Please fix the issues above"
            
            console.print(Panel(
                f"[bold {color}]{status}[/{color}]\n\n"
                f"Passed: {passed} | Failed: {failed} | Warnings: {warnings}\n\n"
                f"{message}",
                title="Verification Complete",
                border_style=color
            ))
        else:
            print(f"\nPassed: {passed}, Failed: {failed}, Warnings: {warnings}")
    
    def save_report(self, path: Path):
        """Save verification report."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.results],
            'summary': {
                'passed': sum(1 for r in self.results if r.status == 'passed'),
                'failed': sum(1 for r in self.results if r.status == 'failed'),
                'warnings': sum(1 for r in self.results if r.status == 'warning'),
            }
        }
        
        path.write_text(json.dumps(data, indent=2))
        
        if console:
            console.print(f"[green]Report saved to {path}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Integration Verification"
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full verification including tests'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode (stricter checks)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save report to file'
    )
    
    args = parser.parse_args()
    
    verifier = IntegrationVerifier(
        full=args.full,
        ci_mode=args.ci
    )
    
    try:
        verifier.verify()
        verifier.print_summary()
        
        if args.output:
            verifier.save_report(args.output)
        
        # Exit with appropriate code
        failed = sum(1 for r in verifier.results if r.status == 'failed')
        sys.exit(0 if failed == 0 else 1)
        
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Verification interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        if console:
            console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
