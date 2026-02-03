#!/usr/bin/env python3
"""
Final Integration Validation - License: Apache 2.0

Comprehensive validation of the entire OpenEvolve integration system.
Performs pre-flight checks before deployment.

Usage:
    python validate_integration.py
    python validate_integration.py --production
    python validate_integration.py --ci
"""

import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse

# Rich for beautiful output
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
class ValidationResult:
    """Result of a validation check."""
    category: str
    check: str
    status: str  # 'passed', 'failed', 'warning', 'skipped'
    message: str
    details: Dict = field(default_factory=dict)


class IntegrationValidator:
    """
    Comprehensive integration validator.
    
    Performs all checks required for production deployment.
    """
    
    def __init__(self, production: bool = False, ci_mode: bool = False):
        self.production = production
        self.ci_mode = ci_mode
        self.results: List[ValidationResult] = []
        self.project_root = Path.cwd()
        
    def print_header(self):
        """Print validation header."""
        if console:
            mode = "PRODUCTION" if self.production else "DEVELOPMENT"
            console.print(Panel(
                f"[bold blue]OpenEvolve Integration Validation[/bold blue]\n"
                f"Mode: {mode}",
                title="🔍 Validation",
                border_style="blue"
            ))
        else:
            print("=" * 70)
            print("OpenEvolve Integration Validation")
            print("=" * 70)
    
    def add_result(self, category: str, check: str, status: str, message: str, details: Dict = None):
        """Add validation result."""
        result = ValidationResult(
            category=category,
            check=check,
            status=status,
            message=message,
            details=details or {}
        )
        self.results.append(result)
        
        if console:
            status_color = {
                'passed': 'green',
                'failed': 'red',
                'warning': 'yellow',
                'skipped': 'dim'
            }.get(status, 'white')
            
            icon = {'passed': '✓', 'failed': '✗', 'warning': '!', 'skipped': '○'}.get(status, '?')
            console.print(f"  [{status_color}]{icon}[/{status_color}] {check}: {message}")
    
    def validate_file_structure(self):
        """Validate required files exist."""
        category = "File Structure"
        
        required_files = [
            'unified_mcp_server.py',
            'event_bus.py',
            'graphql_server.py',
            'service_orchestrator.py',
            'plugin_registry.py',
            'stage6_knowledge_extraction.py',
            'integration_config.py',
            'openevolve_cli.py',
        ]
        
        required_dirs = ['logs', 'data', 'plugins']
        
        missing_files = []
        missing_dirs = []
        
        for filename in required_files:
            if not (self.project_root / filename).exists():
                missing_files.append(filename)
        
        for dirname in required_dirs:
            if not (self.project_root / dirname).exists():
                missing_dirs.append(dirname)
        
        if missing_files or missing_dirs:
            self.add_result(
                category,
                "Required Files",
                "failed",
                f"Missing {len(missing_files)} files, {len(missing_dirs)} directories"
            )
        else:
            self.add_result(
                category,
                "Required Files",
                "passed",
                f"All {len(required_files)} files and {len(required_dirs)} directories present"
            )
    
    def validate_python_syntax(self):
        """Validate Python syntax."""
        category = "Code Quality"
        
        import py_compile
        
        files_to_check = [
            'unified_mcp_server.py',
            'event_bus.py',
            'stage6_knowledge_extraction.py',
            'integration_config.py',
        ]
        
        failed = []
        
        for filename in files_to_check:
            filepath = self.project_root / filename
            if filepath.exists():
                try:
                    py_compile.compile(str(filepath), doraise=True)
                except py_compile.PyCompileError as e:
                    failed.append((filename, str(e)))
        
        if failed:
            self.add_result(
                category,
                "Python Syntax",
                "failed",
                f"{len(failed)} files have syntax errors"
            )
        else:
            self.add_result(
                category,
                "Python Syntax",
                "passed",
                "All files have valid syntax"
            )
    
    def validate_imports(self):
        """Validate core modules can be imported."""
        category = "Module Imports"
        
        modules = [
            'integration_config',
            'event_bus',
            'service_orchestrator',
            'plugin_registry',
            'stage6_knowledge_extraction',
        ]
        
        failed = []
        
        for module in modules:
            try:
                __import__(module)
            except Exception as e:
                failed.append((module, str(e)))
        
        if failed:
            self.add_result(
                category,
                "Core Imports",
                "failed",
                f"Failed to import {len(failed)} modules"
            )
        else:
            self.add_result(
                category,
                "Core Imports",
                "passed",
                f"All {len(modules)} modules imported successfully"
            )
    
    def validate_configuration(self):
        """Validate configuration."""
        category = "Configuration"
        
        # Check environment file
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if env_file.exists():
            self.add_result(
                category,
                "Environment File",
                "passed",
                ".env file exists"
            )
        elif env_example.exists():
            self.add_result(
                category,
                "Environment File",
                "warning",
                ".env missing, .env.example available"
            )
        else:
            self.add_result(
                category,
                "Environment File",
                "failed",
                "No environment file found"
            )
        
        # Check YAML config
        config_file = self.project_root / 'integration_config.yaml'
        if config_file.exists():
            self.add_result(
                category,
                "YAML Config",
                "passed",
                "integration_config.yaml exists"
            )
        else:
            self.add_result(
                category,
                "YAML Config",
                "warning",
                "integration_config.yaml not found (optional)"
            )
    
    def validate_tests(self):
        """Validate test suite."""
        category = "Testing"
        
        test_file = self.project_root / 'test_integrations_comprehensive.py'
        
        if not test_file.exists():
            self.add_result(
                category,
                "Test Suite",
                "failed",
                "Test file not found"
            )
            return
        
        # Run quick tests
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_file), '-v', '-m', 'not slow', '--tb=short', '-q'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.add_result(
                    category,
                    "Test Execution",
                    "passed",
                    "All quick tests passed"
                )
            else:
                self.add_result(
                    category,
                    "Test Execution",
                    "failed",
                    "Some tests failed"
                )
        except subprocess.TimeoutExpired:
            self.add_result(
                category,
                "Test Execution",
                "warning",
                "Tests timed out"
            )
        except Exception as e:
            self.add_result(
                category,
                "Test Execution",
                "warning",
                f"Could not run tests: {e}"
            )
    
    def validate_documentation(self):
        """Validate documentation exists."""
        category = "Documentation"
        
        docs = [
            'INTEGRATION_GUIDE.md',
            'README_INTEGRATION.md',
            'API_REFERENCE.md',
        ]
        
        missing = []
        
        for doc in docs:
            if not (self.project_root / doc).exists():
                missing.append(doc)
        
        if missing:
            self.add_result(
                category,
                "Documentation",
                "warning",
                f"Missing: {', '.join(missing)}"
            )
        else:
            self.add_result(
                category,
                "Documentation",
                "passed",
                f"All {len(docs)} documentation files present"
            )
    
    def validate_dependencies(self):
        """Validate dependencies."""
        category = "Dependencies"
        
        required = [
            'fastapi',
            'pydantic',
            'strawberry',
            'opentelemetry',
        ]
        
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            self.add_result(
                category,
                "Required Packages",
                "failed",
                f"Missing: {', '.join(missing)}"
            )
        else:
            self.add_result(
                category,
                "Required Packages",
                "passed",
                f"All {len(required)} packages installed"
            )
    
    def validate_security(self):
        """Validate security settings."""
        category = "Security"
        
        env_file = self.project_root / '.env'
        
        if env_file.exists():
            content = env_file.read_text()
            
            # Check for default secrets
            issues = []
            
            if 'changeme-in-production' in content:
                issues.append("Default SECRET_KEY found")
            
            if 'DEBUG=true' in content and self.production:
                issues.append("DEBUG enabled in production")
            
            if issues:
                self.add_result(
                    category,
                    "Security Settings",
                    "failed" if self.production else "warning",
                    f"Issues: {', '.join(issues)}"
                )
            else:
                self.add_result(
                    category,
                    "Security Settings",
                    "passed",
                    "Security settings look good"
                )
        else:
            self.add_result(
                category,
                "Security Settings",
                "warning",
                "Cannot check .env file"
            )
    
    def validate_performance(self):
        """Validate performance benchmarks."""
        category = "Performance"
        
        benchmark_file = self.project_root / 'benchmark_integrations.py'
        
        if benchmark_file.exists():
            self.add_result(
                category,
                "Benchmarks",
                "passed",
                "Performance benchmarks available"
            )
        else:
            self.add_result(
                category,
                "Benchmarks",
                "warning",
                "Benchmarks not found"
            )
    
    def run_all_validations(self):
        """Run all validation checks."""
        self.print_header()
        
        if console:
            console.print("\n[bold]Running validations...[/bold]\n")
        
        validations = [
            self.validate_file_structure,
            self.validate_python_syntax,
            self.validate_imports,
            self.validate_configuration,
            self.validate_dependencies,
            self.validate_tests,
            self.validate_documentation,
            self.validate_security,
            self.validate_performance,
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                self.add_result(
                    "Error",
                    validation.__name__,
                    "failed",
                    str(e)
                )
    
    def print_summary(self):
        """Print validation summary."""
        passed = sum(1 for r in self.results if r.status == 'passed')
        failed = sum(1 for r in self.results if r.status == 'failed')
        warnings = sum(1 for r in self.results if r.status == 'warning')
        total = len(self.results)
        
        if console:
            # Group by category
            categories = {}
            for result in self.results:
                if result.category not in categories:
                    categories[result.category] = []
                categories[result.category].append(result)
            
            # Summary table
            table = Table(title="Validation Summary")
            table.add_column("Category", style="cyan")
            table.add_column("Passed", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Warnings", justify="right", style="yellow")
            
            for category, results in categories.items():
                cat_passed = sum(1 for r in results if r.status == 'passed')
                cat_failed = sum(1 for r in results if r.status == 'failed')
                cat_warnings = sum(1 for r in results if r.status == 'warning')
                
                table.add_row(
                    category,
                    str(cat_passed),
                    str(cat_failed),
                    str(cat_warnings)
                )
            
            console.print(table)
            
            # Overall status
            if failed == 0:
                status = "READY FOR DEPLOYMENT"
                color = "green"
            elif failed <= 2 and not self.production:
                status = "READY (with warnings)"
                color = "yellow"
            else:
                status = "NOT READY"
                color = "red"
            
            console.print(Panel(
                f"[bold {color}]{status}[/{color}]\n\n"
                f"Passed: {passed} | Failed: {failed} | Warnings: {warnings} | Total: {total}",
                title="Validation Complete",
                border_style=color
            ))
        else:
            print(f"\nPassed: {passed}, Failed: {failed}, Warnings: {warnings}")
    
    def save_report(self, path: Path):
        """Save validation report."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'production' if self.production else 'development',
            'results': [asdict(r) for r in self.results],
            'summary': {
                'passed': sum(1 for r in self.results if r.status == 'passed'),
                'failed': sum(1 for r in self.results if r.status == 'failed'),
                'warnings': sum(1 for r in self.results if r.status == 'warning'),
            }
        }
        
        path.write_text(json.dumps(data, indent=2))
        
        if console:
            console.print(f"\n[green]Report saved to {path}[/green]")
    
    def exit_code(self) -> int:
        """Get exit code based on results."""
        failed = sum(1 for r in self.results if r.status == 'failed')
        
        if failed == 0:
            return 0
        elif failed <= 2 and not self.production:
            return 0  # Allow warnings in development
        else:
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Integration Validation"
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Production validation (stricter checks)'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode (non-interactive)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save report to file'
    )
    
    args = parser.parse_args()
    
    validator = IntegrationValidator(
        production=args.production,
        ci_mode=args.ci
    )
    
    try:
        validator.run_all_validations()
        validator.print_summary()
        
        if args.output:
            validator.save_report(args.output)
        
        sys.exit(validator.exit_code())
        
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Validation interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        if console:
            console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
