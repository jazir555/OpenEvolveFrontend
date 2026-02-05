"""
Integration Test Runner - License: Apache 2.0

Comprehensive test runner for OpenEvolve integration components.
Runs all tests and generates detailed reports.

Usage:
    python run_integration_tests.py
    python run_integration_tests.py --quick
    python run_integration_tests.py --category stage6
    python run_integration_tests.py --html-report
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

# Rich for output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.tree import Tree
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration_ms: float
    message: str = ""
    traceback: str = ""
    category: str = ""


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    suite_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    tests: List[TestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'tests': [asdict(t) for t in self.tests]
        }


# =============================================================================
# TEST RUNNER
# =============================================================================

class IntegrationTestRunner:
    """Runs integration tests and collects results."""
    
    TEST_FILES = {
        'core': ['test_integrations_comprehensive.py'],
        'stage6': ['test_integrations_comprehensive.py::TestStage6KnowledgeExtraction'],
        'event_bus': ['test_integrations_comprehensive.py::TestEventBus'],
        'api': ['test_integrations_comprehensive.py::TestAPIGateway'],
        'plugins': ['test_integrations_comprehensive.py::TestPluginRegistry'],
        'orchestrator': ['test_integrations_comprehensive.py::TestServiceOrchestrator'],
        'performance': ['test_integrations_comprehensive.py::TestPerformance'],
    }
    
    def __init__(self):
        self.results: List[TestSuiteResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def _run_pytest(
        self,
        test_paths: List[str],
        markers: Optional[List[str]] = None,
        verbose: bool = True
    ) -> subprocess.CompletedProcess:
        """Run pytest with specified parameters."""
        cmd = [
            sys.executable, '-m', 'pytest',
            *test_paths,
            '-v' if verbose else '-q',
            '--tb=short',
            '--json-report',
            '--json-report-file=test_report.json'
        ]
        
        if markers:
            cmd.extend(['-m', ' or '.join(markers)])
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
    
    def _parse_test_report(self, report_path: Path) -> TestSuiteResult:
        """Parse pytest JSON report."""
        if not report_path.exists():
            return TestSuiteResult(suite_name="unknown")
        
        with open(report_path) as f:
            data = json.load(f)
        
        suite = TestSuiteResult(
            suite_name=data.get('environment', {}).get('Python', 'unknown'),
            total=data.get('summary', {}).get('total', 0),
            passed=data.get('summary', {}).get('passed', 0),
            failed=data.get('summary', {}).get('failed', 0),
            skipped=data.get('summary', {}).get('skipped', 0),
            errors=data.get('summary', {}).get('error', 0),
            duration_seconds=data.get('duration', 0)
        )
        
        for test in data.get('tests', []):
            result = TestResult(
                name=test.get('nodeid', 'unknown'),
                status=test.get('outcome', 'unknown'),
                duration_ms=test.get('duration', 0) * 1000,
                message=test.get('setup', {}).get('longrepr', ''),
                traceback=test.get('call', {}).get('longrepr', ''),
                category=self._categorize_test(test.get('nodeid', ''))
            )
            suite.tests.append(result)
        
        return suite
    
    def _categorize_test(self, test_name: str) -> str:
        """Categorize a test by name."""
        categories = {
            'stage6': ['stage6', 'knowledge'],
            'event_bus': ['event', 'bus'],
            'api': ['api', 'gateway'],
            'plugins': ['plugin', 'registry'],
            'orchestrator': ['orchestrator', 'service'],
            'config': ['config'],
            'telemetry': ['telemetry'],
        }
        
        test_lower = test_name.lower()
        for category, keywords in categories.items():
            if any(kw in test_lower for kw in keywords):
                return category
        
        return 'other'
    
    async def run_all_tests(self, quick: bool = False) -> List[TestSuiteResult]:
        """Run all integration tests."""
        self.start_time = datetime.now()
        
        if console:
            console.print(Panel(
                "[bold blue]OpenEvolve Integration Test Suite[/bold blue]\n"
                f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                title="Test Runner"
            ))
        
        # Determine which tests to run
        if quick:
            categories = ['core']
            markers = ['not slow', 'not integration']
        else:
            categories = list(self.TEST_FILES.keys())
            markers = None
        
        # Run tests for each category
        for category in categories:
            if console:
                console.print(f"\n[cyan]Running {category} tests...[/cyan]")
            
            test_paths = self.TEST_FILES.get(category, [])
            if not test_paths:
                continue
            
            result = self._run_pytest(test_paths, markers)
            
            # Parse results
            suite = self._parse_test_report(Path('test_report.json'))
            suite.suite_name = category
            self.results.append(suite)
            
            # Print summary
            if console:
                status_color = 'green' if suite.failed == 0 else 'red'
                console.print(
                    f"  [{status_color}]{suite.passed} passed, "
                    f"{suite.failed} failed, "
                    f"{suite.skipped} skipped[/{status_color}]"
                )
        
        self.end_time = datetime.now()
        return self.results
    
    async def run_category(self, category: str) -> TestSuiteResult:
        """Run tests for a specific category."""
        test_paths = self.TEST_FILES.get(category, [])
        if not test_paths:
            raise ValueError(f"Unknown category: {category}")
        
        if console:
            console.print(f"[cyan]Running {category} tests...[/cyan]")
        
        result = self._run_pytest(test_paths)
        suite = self._parse_test_report(Path('test_report.json'))
        suite.suite_name = category
        
        return suite
    
    def print_summary(self):
        """Print test summary."""
        if not self.results:
            print("No test results available")
            return
        
        if not RICH_AVAILABLE:
            self._print_summary_plain()
            return
        
        # Overall statistics
        total = sum(r.total for r in self.results)
        passed = sum(r.passed for r in self.results)
        failed = sum(r.failed for r in self.results)
        skipped = sum(r.skipped for r in self.results)
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        # Overall status
        status = "PASSED" if failed == 0 else "FAILED"
        status_color = "green" if failed == 0 else "red"
        
        console.print(Panel(
            f"[bold {status_color}]{status}[/{status_color}]\n\n"
            f"Total: {total} | "
            f"[green]Passed: {passed}[/green] | "
            f"[red]Failed: {failed}[/red] | "
            f"[yellow]Skipped: {skipped}[/yellow]\n"
            f"Duration: {duration:.2f}s",
            title="Test Summary"
        ))
        
        # Detailed results table
        table = Table(title="Test Results by Category")
        table.add_column("Category", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Skipped", justify="right", style="yellow")
        table.add_column("Duration", justify="right")
        table.add_column("Status")
        
        for suite in self.results:
            status = "[OK]" if suite.failed == 0 else "[FAIL]"
            status_style = "green" if suite.failed == 0 else "red"
            
            table.add_row(
                suite.suite_name,
                str(suite.total),
                str(suite.passed),
                str(suite.failed),
                str(suite.skipped),
                f"{suite.duration_seconds:.2f}s",
                f"[{status_style}]{status}[/{status_style}]"
            )
        
        console.print(table)
        
        # Failed tests
        failed_tests = [
            t for suite in self.results
            for t in suite.tests
            if t.status == 'failed'
        ]
        
        if failed_tests:
            console.print("\n[red bold]Failed Tests:[/red bold]")
            for test in failed_tests[:10]:  # Show first 10
                console.print(f"  [red][FAIL][/red] {test.name}")
                if test.message:
                    console.print(f"    {test.message[:100]}")
    
    def _print_summary_plain(self):
        """Print summary without Rich."""
        total = sum(r.total for r in self.results)
        passed = sum(r.passed for r in self.results)
        failed = sum(r.failed for r in self.results)
        
        print(f"\nTest Summary:")
        print(f"  Total: {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Status: {'PASSED' if failed == 0 else 'FAILED'}")
    
    def save_report(self, path: Path):
        """Save test report to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'suites': [s.to_dict() for s in self.results]
        }
        
        path.write_text(json.dumps(data, indent=2))
        
        if console:
            console.print(f"[green]Report saved to {path}[/green]")
    
    def generate_html_report(self, path: Path):
        """Generate HTML test report."""
        total = sum(r.total for r in self.results)
        passed = sum(r.passed for r in self.results)
        failed = sum(r.failed for r in self.results)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEvolve Integration Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat {{ text-align: center; padding: 20px; border-radius: 8px; }}
        .stat.total {{ background: #e3f2fd; }}
        .stat.passed {{ background: #e8f5e9; }}
        .stat.failed {{ background: #ffebee; }}
        .stat.skipped {{ background: #fff8e1; }}
        .stat-value {{ font-size: 36px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ color: #666; font-size: 14px; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f5f5f5; padding: 12px; text-align: left; font-weight: 600; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        .status-pass {{ color: #4caf50; }}
        .status-fail {{ color: #f44336; }}
        .timestamp {{ color: #999; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OpenEvolve Integration Test Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="stat total">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat passed">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat failed">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat skipped">
                <div class="stat-value">{sum(r.skipped for r in self.results)}</div>
                <div class="stat-label">Skipped</div>
            </div>
        </div>
        
        <h2>Results by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Total</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Duration</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for suite in self.results:
            status_class = 'status-pass' if suite.failed == 0 else 'status-fail'
            status_text = '[OK] Pass' if suite.failed == 0 else '[FAIL] Fail'
            
            html += f"""
                <tr>
                    <td>{suite.suite_name}</td>
                    <td>{suite.total}</td>
                    <td>{suite.passed}</td>
                    <td>{suite.failed}</td>
                    <td>{suite.skipped}</td>
                    <td>{suite.duration_seconds:.2f}s</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        path.write_text(html)
        
        if console:
            console.print(f"[green]HTML report saved to {path}[/green]")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Integration Test Runner"
    )
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--category', help='Run specific category (stage6, api, event_bus, etc.)')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', type=Path, help='Save JSON report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner()
    
    try:
        if args.category:
            # Run specific category
            result = await runner.run_category(args.category)
            runner.results.append(result)
        else:
            # Run all tests
            await runner.run_all_tests(quick=args.quick)
        
        # Print summary
        runner.print_summary()
        
        # Save reports
        if args.output:
            runner.save_report(args.output)
        
        if args.html_report:
            runner.generate_html_report(Path('test_report.html'))
        
        # Exit with appropriate code
        total_failed = sum(r.failed for r in runner.results)
        sys.exit(0 if total_failed == 0 else 1)
        
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Test run interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        if console:
            console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
