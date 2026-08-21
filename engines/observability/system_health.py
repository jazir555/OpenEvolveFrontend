"""
System Health Checker - License: Apache 2.0

Comprehensive health diagnostics for OpenEvolve integration system.
Checks services, dependencies, configuration, and performance.

Run: python system_health.py
"""
from __future__ import annotations


import asyncio
import json
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import subprocess
import importlib

# Rich for output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

# =============================================================================
# HEALTH CHECK DATA MODELS
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'unknown'
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: datetime
    overall_status: str
    checks: List[HealthCheckResult]
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status,
            'checks': [c.to_dict() for c in self.checks],
            'summary': self.summary
        }


# =============================================================================
# HEALTH CHECKERS
# =============================================================================

class HealthChecker:
    """Base health checker."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        # Default implementation returns unknown status
        return HealthCheckResult(
            component=self.name,
            status="unknown",
            message="Health check not implemented",
            details={}
        )


class PythonVersionChecker(HealthChecker):
    """Check Python version."""
    
    def __init__(self):
        super().__init__("Python Version")
    
    async def check(self) -> HealthCheckResult:
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 11:
            return HealthCheckResult(
                component="Python Version",
                status="healthy",
                message=f"Python {version_str} (supported)",
                details={'version': version_str}
            )
        elif version.major >= 3 and version.minor >= 10:
            return HealthCheckResult(
                component="Python Version",
                status="degraded",
                message=f"Python {version_str} (minimum recommended is 3.11)",
                details={'version': version_str}
            )
        else:
            return HealthCheckResult(
                component="Python Version",
                status="unhealthy",
                message=f"Python {version_str} (requires 3.10+)",
                details={'version': version_str}
            )


class DependenciesChecker(HealthChecker):
    """Check required dependencies."""
    
    REQUIRED_PACKAGES = [
        'fastapi',
        'pydantic',
        'uvicorn',
        'strawberry',
        'opentelemetry',
        'valkey',
        'mcp',
        'click',
        'rich',
    ]
    
    OPTIONAL_PACKAGES = [
        'sklearn',
        'networkx',
        'plotly',
        'pandas',
        'httpx',
    ]
    
    def __init__(self):
        super().__init__("Dependencies")
    
    async def check(self) -> HealthCheckResult:
        missing_required = []
        missing_optional = []
        installed = []
        
        for pkg in self.REQUIRED_PACKAGES:
            try:
                importlib.import_module(pkg)
                installed.append(pkg)
            except ImportError:
                missing_required.append(pkg)
        
        for pkg in self.OPTIONAL_PACKAGES:
            try:
                importlib.import_module(pkg)
                installed.append(pkg)
            except ImportError:
                missing_optional.append(pkg)
        
        if missing_required:
            return HealthCheckResult(
                component="Dependencies",
                status="unhealthy",
                message=f"Missing required: {', '.join(missing_required)}",
                details={
                    'missing_required': missing_required,
                    'missing_optional': missing_optional,
                    'installed': installed
                }
            )
        elif missing_optional:
            return HealthCheckResult(
                component="Dependencies",
                status="healthy",
                message=f"All required present, missing optional: {', '.join(missing_optional)}",
                details={
                    'missing_optional': missing_optional,
                    'installed': installed
                }
            )
        else:
            return HealthCheckResult(
                component="Dependencies",
                status="healthy",
                message=f"All {len(installed)} packages installed",
                details={'installed': installed}
            )


class FileStructureChecker(HealthChecker):
    """Check required files exist."""
    
    REQUIRED_FILES = [
        'unified_mcp_server.py',
        'event_bus.py',
        'graphql_server.py',
        'api_server.py',
        'service_orchestrator.py',
        'plugin_registry.py',
        'stage6_knowledge_extraction.py',
        'openevolve_cli.py',
        'api_gateway.py',
        'integration_config.py',
        'telemetry.py',
    ]
    
    def __init__(self, project_root: Path = None):
        super().__init__("File Structure")
        self.project_root = project_root or Path.cwd()
    
    async def check(self) -> HealthCheckResult:
        missing = []
        present = []
        
        for filename in self.REQUIRED_FILES:
            filepath = self.project_root / filename
            if filepath.exists():
                present.append(filename)
            else:
                missing.append(filename)
        
        if missing:
            return HealthCheckResult(
                component="File Structure",
                status="unhealthy",
                message=f"Missing {len(missing)} required files",
                details={'missing': missing, 'present': present}
            )
        else:
            return HealthCheckResult(
                component="File Structure",
                status="healthy",
                message=f"All {len(present)} required files present",
                details={'present': present}
            )


class ConfigurationChecker(HealthChecker):
    """Check configuration validity."""
    
    def __init__(self):
        super().__init__("Configuration")
    
    async def check(self) -> HealthCheckResult:
        try:
            from integration_config import get_config
            config = get_config()
            
            return HealthCheckResult(
                component="Configuration",
                status="healthy",
                message="Configuration loaded successfully",
                details={
                    'log_level': config.log_level,
                    'orchestrator_port': config.orchestrator_port,
                    'services': list(config.services.keys())
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="Configuration",
                status="unhealthy",
                message=f"Configuration error: {str(e)}",
                details={'error': str(e)}
            )


class ServiceHealthChecker(HealthChecker):
    """Check service health via HTTP endpoints."""
    
    SERVICES = {
        'REST API': {'url': 'http://localhost:8000/health', 'method': 'GET'},
        'GraphQL': {'url': 'http://localhost:8001/graphql', 'method': 'POST', 'data': '{"query": "{ __typename }"}'},
        'Orchestrator': {'url': 'http://localhost:8080/health', 'method': 'GET'},
    }
    
    def __init__(self):
        super().__init__("Services")
    
    async def check(self) -> HealthCheckResult:
        import httpx
        
        results = {}
        any_healthy = False
        any_unhealthy = False
        
        async with httpx.AsyncClient() as client:
            for service_name, config in self.SERVICES.items():
                try:
                    if config['method'] == 'POST':
                        response = await client.post(
                            config['url'],
                            data=config.get('data'),
                            timeout=5.0
                        )
                    else:
                        response = await client.get(config['url'], timeout=5.0)
                    
                    if response.status_code == 200:
                        results[service_name] = 'healthy'
                        any_healthy = True
                    else:
                        results[service_name] = f'unhealthy ({response.status_code})'
                        any_unhealthy = True
                        
                except Exception as e:
                    results[service_name] = f'unreachable: {str(e)[:50]}'
                    any_unhealthy = True
        
        if any_unhealthy and not any_healthy:
            status = "unhealthy"
            message = "All services unreachable"
        elif any_unhealthy:
            status = "degraded"
            message = "Some services unhealthy"
        else:
            status = "healthy"
            message = "All services healthy"
        
        return HealthCheckResult(
            component="Services",
            status=status,
            message=message,
            details=results
        )


class EventBusChecker(HealthChecker):
    """Check event bus connectivity."""
    
    def __init__(self):
        super().__init__("Event Bus")
    
    async def check(self) -> HealthCheckResult:
        try:
            from event_bus import InMemoryEventBus
            
            bus = InMemoryEventBus()
            await bus.connect()
            
            # Try publish/subscribe
            received = []
            async def handler(event):
                received.append(event)
            
            await bus.subscribe("health_check", handler)
            
            from event_bus import WorkflowEvent, EventType
            test_event = WorkflowEvent(
                id="health_test",
                type=EventType.WORKFLOW_STARTED,
                payload={"test": True},
                timestamp=datetime.now(),
                priority=1
            )
            
            await bus.publish("health_check", test_event)
            await asyncio.sleep(0.1)
            
            await bus.disconnect()
            
            if received:
                return HealthCheckResult(
                    component="Event Bus",
                    status="healthy",
                    message="Event bus functional",
                    details={'events_received': len(received)}
                )
            else:
                return HealthCheckResult(
                    component="Event Bus",
                    status="degraded",
                    message="Event bus connected but events not received",
                    details={}
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="Event Bus",
                status="unhealthy",
                message=f"Event bus error: {str(e)}",
                details={'error': str(e)}
            )


class Stage6Checker(HealthChecker):
    """Check Stage 6 Knowledge Extraction."""
    
    def __init__(self):
        super().__init__("Stage 6 Knowledge")
    
    async def check(self) -> HealthCheckResult:
        try:
            from stage6_knowledge_extraction import (
                Stage6KnowledgeExtraction, ExecutionTrace
            )
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmp:
                engine = Stage6KnowledgeExtraction(storage_path=Path(tmp))
                
                # Create test trace
                trace = ExecutionTrace(
                    trace_id="health_check",
                    workflow_id="wf_health",
                    problem_description="Health check test",
                    stages=[{"stage_name": "test", "parameters": {}}],
                    final_result={"status": "ok"},
                    execution_time_ms=100.0,
                    timestamp=datetime.now()
                )
                
                result = await engine.process_trace(trace)
                
                return HealthCheckResult(
                    component="Stage 6 Knowledge",
                    status="healthy",
                    message="Knowledge extraction functional",
                    details={'traces_processed': result.get('traces_processed', 0)}
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="Stage 6 Knowledge",
                status="unhealthy",
                message=f"Stage 6 error: {str(e)}",
                details={'error': str(e)}
            )


# =============================================================================
# HEALTH CHECK RUNNER
# =============================================================================

class SystemHealthRunner:
    """Runs all health checks."""
    
    def __init__(self):
        self.checkers: List[HealthChecker] = [
            PythonVersionChecker(),
            DependenciesChecker(),
            FileStructureChecker(),
            ConfigurationChecker(),
            ServiceHealthChecker(),
            EventBusChecker(),
            Stage6Checker(),
        ]
    
    async def run_all(self) -> SystemHealthReport:
        """Run all health checks."""
        if console:
            console.print("[bold]Running System Health Checks...[/bold]\n")
        
        checks = []
        
        for checker in self.checkers:
            if console:
                console.print(f"  Checking {checker.name}...", end=" ")
            
            try:
                result = await checker.check()
                checks.append(result)
                
                if console:
                    status_emoji = {
                        'healthy': '[OK]',
                        'degraded': '[WARN]',
                        'unhealthy': '[FAIL]',
                        'unknown': '❓'
                    }.get(result.status, '❓')
                    console.print(f"{status_emoji} {result.status}")
                    
            except Exception as e:
                error_result = HealthCheckResult(
                    component=checker.name,
                    status="unknown",
                    message=f"Check failed: {str(e)}"
                )
                checks.append(error_result)
                
                if console:
                    console.print(f"[FAIL] error")
        
        # Determine overall status
        statuses = [c.status for c in checks]
        if 'unhealthy' in statuses:
            overall = 'unhealthy'
        elif 'degraded' in statuses:
            overall = 'degraded'
        else:
            overall = 'healthy'
        
        # Summary
        summary = {
            'total_checks': len(checks),
            'healthy': sum(1 for c in checks if c.status == 'healthy'),
            'degraded': sum(1 for c in checks if c.status == 'degraded'),
            'unhealthy': sum(1 for c in checks if c.status == 'unhealthy'),
        }
        
        return SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=overall,
            checks=checks,
            summary=summary
        )
    
    def print_report(self, report: SystemHealthReport):
        """Print formatted report."""
        if not RICH_AVAILABLE:
            print(json.dumps(report.to_dict(), indent=2))
            return
        
        # Overall status
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red'
        }.get(report.overall_status, 'white')
        
        console.print(Panel(
            f"[bold]Overall Status: [{status_color}]{report.overall_status.upper()}[/{status_color}][/bold]\n"
            f"Healthy: {report.summary['healthy']} | "
            f"Degraded: {report.summary['degraded']} | "
            f"Unhealthy: {report.summary['unhealthy']}",
            title="System Health Report",
            subtitle=report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        # Detailed table
        table = Table()
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Message")
        
        for check in report.checks:
            status_style = {
                'healthy': 'green',
                'degraded': 'yellow',
                'unhealthy': 'red',
                'unknown': 'dim'
            }.get(check.status, 'white')
            
            table.add_row(
                check.component,
                f"[{status_style}]{check.status}[/{status_style}]",
                check.message[:60]
            )
        
        console.print(table)
    
    def save_report(self, report: SystemHealthReport, path: Path):
        """Save report to file."""
        path.write_text(json.dumps(report.to_dict(), indent=2))
        if console:
            console.print(f"\n[green]Report saved to {path}[/green]")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenEvolve System Health Check")
    parser.add_argument("--output", "-o", type=Path, help="Save report to file")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    runner = SystemHealthRunner()
    report = await runner.run_all()
    
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        runner.print_report(report)
    
    if args.output:
        runner.save_report(report, args.output)
    
    # Exit with appropriate code
    if report.overall_status == 'unhealthy':
        sys.exit(1)
    elif report.overall_status == 'degraded':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
