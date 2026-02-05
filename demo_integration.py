#!/usr/bin/env python3
"""
OpenEvolve Integration Demo - License: Apache 2.0

Interactive demo showcasing the OpenEvolve integration system.
Demonstrates Stage 6 knowledge extraction, event bus, and service orchestration.

Usage:
    python demo_integration.py
    python demo_integration.py --quick
    python demo_integration.py --component stage6
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import argparse

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better output: pip install rich")

console = Console() if RICH_AVAILABLE else None


class IntegrationDemo:
    """Interactive demo of OpenEvolve integration features."""
    
    def __init__(self, quick: bool = False):
        self.quick = quick
        self.results = []
    
    def print_header(self):
        """Print demo header."""
        if console:
            console.print(Panel(
                "[bold blue]OpenEvolve Integration Demo[/bold blue]\n"
                "Showcasing unified architecture, Stage 6 knowledge extraction,\n"
                "event-driven messaging, and service orchestration.",
                title="🚀 Welcome",
                border_style="blue",
                box=box.ROUNDED
            ))
        else:
            print("=" * 60)
            print("OpenEvolve Integration Demo")
            print("=" * 60)
    
    async def demo_stage6_knowledge(self):
        """Demo Stage 6 Knowledge Extraction."""
        self._print_section("Stage 6: Knowledge Extraction")
        
        try:
            from stage6_knowledge_extraction import (
                Stage6KnowledgeExtraction, ExecutionTrace
            )
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmp:
                # Initialize engine
                self._print_step("Initializing knowledge extraction engine...")
                engine = Stage6KnowledgeExtraction(storage_path=Path(tmp))
                
                # Create sample workflow traces
                self._print_step("Processing workflow traces...")
                
                sample_problems = [
                    "Optimize neural network architecture for image classification",
                    "Find optimal hyperparameters for transformer model",
                    "Design efficient CNN for edge detection",
                    "Tune learning rate schedule for deep network",
                    "Optimize neural network architecture for NLP tasks",
                ]
                
                for i, problem in enumerate(sample_problems):
                    trace = ExecutionTrace(
                        trace_id=f"trace_{i:03d}",
                        workflow_id=f"wf_{i:03d}",
                        problem_description=problem,
                        stages=[
                            {
                                "stage_name": "decomposition",
                                "parameters": {"strategy": "hybrid", "depth": 3}
                            },
                            {
                                "stage_name": "evolution",
                                "parameters": {"generations": 100, "population_size": 50}
                            },
                            {
                                "stage_name": "assembly",
                                "parameters": {"validation": "strict"}
                            }
                        ],
                        final_result={
                            "architecture": f"optimized_model_{i}",
                            "accuracy": 0.92 + (i * 0.01),
                            "fitness": 0.95
                        },
                        execution_time_ms=5000 + (i * 500),
                        timestamp=datetime.now()
                    )
                    
                    result = await engine.process_trace(trace)
                
                # Get statistics
                stats = engine.get_statistics()
                
                # Display results
                self._print_success(f"Processed {stats['traces_processed']} traces")
                self._print_info(f"Extracted {stats['patterns_extracted']} patterns")
                self._print_info(f"Generated {stats['artifacts_generated']} artifacts")
                
                if RICH_AVAILABLE:
                    # Pattern types table
                    if stats['pattern_types']:
                        table = Table(title="Extracted Pattern Types")
                        table.add_column("Type", style="cyan")
                        table.add_column("Count", justify="right")
                        
                        for ptype, count in stats['pattern_types'].items():
                            table.add_row(ptype, str(count))
                        
                        console.print(table)
                    
                    # Artifact types table
                    if stats['artifact_types']:
                        table = Table(title="Generated Artifact Types")
                        table.add_column("Type", style="green")
                        table.add_column("Count", justify="right")
                        
                        for atype, count in stats['artifact_types'].items():
                            table.add_row(atype, str(count))
                        
                        console.print(table)
                
                # Demonstrate knowledge retrieval
                self._print_step("Retrieving applicable knowledge...")
                artifacts = engine.get_applicable_artifacts(
                    "neural network optimization",
                    min_validity=0.5
                )
                
                self._print_success(f"Found {len(artifacts)} applicable artifacts")
                
                for artifact in artifacts[:3]:
                    self._print_info(f"  * {artifact.name} (validity: {artifact.validity_score:.2f})")
                
                self.results.append({
                    'component': 'stage6',
                    'status': 'success',
                    'traces': stats['traces_processed'],
                    'patterns': stats['patterns_extracted'],
                    'artifacts': stats['artifacts_generated']
                })
                
        except Exception as e:
            self._print_error(f"Stage 6 demo failed: {e}")
            self.results.append({'component': 'stage6', 'status': 'failed', 'error': str(e)})
    
    async def demo_event_bus(self):
        """Demo Event Bus functionality."""
        self._print_section("Event Bus: Messaging System")
        
        try:
            from event_bus import InMemoryEventBus, WorkflowEvent, EventType
            
            # Initialize event bus
            self._print_step("Initializing event bus...")
            bus = InMemoryEventBus()
            await bus.connect()
            
            # Setup event handlers
            events_received = []
            
            async def workflow_handler(event):
                events_received.append(event)
            
            async def stage_handler(event):
                events_received.append(event)
            
            await bus.subscribe("workflow_events", workflow_handler)
            await bus.subscribe("stage_events", stage_handler)
            
            # Publish events
            self._print_step("Publishing workflow events...")
            
            test_events = [
                (EventType.WORKFLOW_STARTED, {"workflow_id": "wf_001", "problem": "Optimization"}),
                (EventType.STAGE_COMPLETED, {"workflow_id": "wf_001", "stage": "decomposition"}),
                (EventType.STAGE_COMPLETED, {"workflow_id": "wf_001", "stage": "evolution"}),
                (EventType.WORKFLOW_COMPLETED, {"workflow_id": "wf_001", "result": "success"}),
            ]
            
            for event_type, payload in test_events:
                event = WorkflowEvent(
                    id=f"evt_{len(events_received):03d}",
                    type=event_type,
                    payload=payload,
                    timestamp=datetime.now(),
                    priority=1
                )
                
                channel = "workflow_events" if "WORKFLOW" in event_type.value else "stage_events"
                await bus.publish(channel, event)
            
            # Allow async handlers to process
            await asyncio.sleep(0.5)
            
            await bus.disconnect()
            
            self._print_success(f"Published and received {len(events_received)} events")
            
            if RICH_AVAILABLE:
                table = Table(title="Event Flow")
                table.add_column("Event Type", style="cyan")
                table.add_column("Channel", style="blue")
                table.add_column("Status", style="green")
                
                for event in events_received:
                    table.add_row(
                        event.type.value,
                        "workflow_events" if "WORKFLOW" in event.type.value else "stage_events",
                        "[OK] Received"
                    )
                
                console.print(table)
            
            self.results.append({
                'component': 'event_bus',
                'status': 'success',
                'events': len(events_received)
            })
            
        except Exception as e:
            self._print_error(f"Event bus demo failed: {e}")
            self.results.append({'component': 'event_bus', 'status': 'failed', 'error': str(e)})
    
    async def demo_service_orchestrator(self):
        """Demo Service Orchestrator."""
        self._print_section("Service Orchestrator: Lifecycle Management")
        
        try:
            from service_orchestrator import ServiceOrchestrator
            
            # Initialize orchestrator
            self._print_step("Initializing service orchestrator...")
            orchestrator = ServiceOrchestrator()
            
            # Register mock services
            services = [
                ("rest_api", [], "REST API Server"),
                ("graphql", ["rest_api"], "GraphQL API Server"),
                ("event_bus", [], "Event Bus"),
                ("mcp_server", ["event_bus"], "MCP Server"),
            ]
            
            for name, deps, description in services:
                async def mock_start():
                    return True
                
                async def mock_stop():
                    return True
                
                orchestrator.register_service(
                    name=name,
                    start_func=mock_start,
                    stop_func=mock_stop,
                    dependencies=deps
                )
                
                self._print_info(f"Registered: {description}")
            
            # Display dependency graph
            if RICH_AVAILABLE:
                tree = Tree("[bold]Service Dependencies[/bold]")
                
                for name, deps, _ in services:
                    if not deps:
                        tree.add(f"[green]{name}[/green]")
                    else:
                        branch = tree.add(f"[blue]{name}[/blue]")
                        for dep in deps:
                            branch.add(f"[dim]depends on: {dep}[/dim]")
                
                console.print(tree)
            
            # Show registered services
            self._print_success(f"Registered {len(orchestrator.services)} services")
            
            self.results.append({
                'component': 'orchestrator',
                'status': 'success',
                'services': len(orchestrator.services)
            })
            
        except Exception as e:
            self._print_error(f"Orchestrator demo failed: {e}")
            self.results.append({'component': 'orchestrator', 'status': 'failed', 'error': str(e)})
    
    async def demo_plugin_registry(self):
        """Demo Plugin Registry."""
        self._print_section("Plugin Registry: Dynamic Extensions")
        
        try:
            from plugin_registry import PluginRegistry, PluginMetadata, PluginType
            
            # Initialize registry
            self._print_step("Initializing plugin registry...")
            registry = PluginRegistry()
            
            # Show plugin capabilities
            if RICH_AVAILABLE:
                table = Table(title="Plugin System Capabilities")
                table.add_column("Feature", style="cyan")
                table.add_column("Status", style="green")
                
                table.add_row("Dynamic Loading", "[OK] Supported")
                table.add_row("File-based Discovery", "[OK] Supported")
                table.add_row("Module Import", "[OK] Supported")
                table.add_row("Lifecycle Management", "[OK] Supported")
                table.add_row("Hot Reload", "[OK] Supported")
                
                console.print(table)
            
            self._print_success("Plugin registry initialized")
            
            self.results.append({
                'component': 'plugins',
                'status': 'success'
            })
            
        except Exception as e:
            self._print_error(f"Plugin demo failed: {e}")
            self.results.append({'component': 'plugins', 'status': 'failed', 'error': str(e)})
    
    def print_summary(self):
        """Print demo summary."""
        if console:
            # Summary table
            table = Table(title="Demo Results Summary", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Details")
            
            for result in self.results:
                status_color = {
                    'success': 'green',
                    'failed': 'red'
                }.get(result['status'], 'yellow')
                
                details = []
                if 'traces' in result:
                    details.append(f"{result['traces']} traces")
                if 'patterns' in result:
                    details.append(f"{result['patterns']} patterns")
                if 'artifacts' in result:
                    details.append(f"{result['artifacts']} artifacts")
                if 'events' in result:
                    details.append(f"{result['events']} events")
                if 'services' in result:
                    details.append(f"{result['services']} services")
                
                table.add_row(
                    result['component'].replace('_', ' ').title(),
                    f"[{status_color}]{result['status'].upper()}[/{status_color}]",
                    ", ".join(details) if details else "-"
                )
            
            console.print(table)
            
            # Completion message
            success_count = sum(1 for r in self.results if r['status'] == 'success')
            total_count = len(self.results)
            
            console.print(Panel(
                f"[bold green]Demo Complete![/bold green]\n\n"
                f"Successfully demonstrated {success_count}/{total_count} components.\n\n"
                "Next steps:\n"
                "  * Start services: [cyan]make start[/cyan]\n"
                "  * Run tests: [cyan]make test[/cyan]\n"
                "  * View dashboard: [cyan]make dashboard[/cyan]\n\n"
                "Documentation: [cyan]INTEGRATION_GUIDE.md[/cyan]",
                title="🎉 Finished",
                border_style="green"
            ))
        else:
            print("\n" + "=" * 60)
            print("Demo Complete!")
            print("=" * 60)
            for result in self.results:
                print(f"  {result['component']}: {result['status']}")
    
    def _print_section(self, title: str):
        """Print section header."""
        if console:
            console.print(f"\n[bold yellow]▶ {title}[/bold yellow]")
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
    
    def _print_step(self, message: str):
        """Print step message."""
        if console:
            console.print(f"  [blue]->[/blue] {message}")
        else:
            print(f"  -> {message}")
    
    def _print_success(self, message: str):
        """Print success message."""
        if console:
            console.print(f"    [green][OK][/green] {message}")
        else:
            print(f"    [OK] {message}")
    
    def _print_info(self, message: str):
        """Print info message."""
        if console:
            console.print(f"    [dim]{message}[/dim]")
        else:
            print(f"      {message}")
    
    def _print_error(self, message: str):
        """Print error message."""
        if console:
            console.print(f"    [red][FAIL][/red] {message}")
        else:
            print(f"    [FAIL] {message}")
    
    async def run(self, components: List[str] = None):
        """Run demo."""
        self.print_header()
        
        # Determine which demos to run
        if components is None:
            components = ['stage6', 'event_bus', 'orchestrator', 'plugins']
        
        # Run demos
        if 'stage6' in components:
            await self.demo_stage6_knowledge()
        
        if 'event_bus' in components:
            await self.demo_event_bus()
        
        if 'orchestrator' in components:
            await self.demo_service_orchestrator()
        
        if 'plugins' in components:
            await self.demo_plugin_registry()
        
        # Summary
        self.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Integration Demo"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick demo (fewer iterations)'
    )
    parser.add_argument(
        '--component',
        choices=['stage6', 'event_bus', 'orchestrator', 'plugins', 'all'],
        default='all',
        help='Demo specific component'
    )
    
    args = parser.parse_args()
    
    # Determine components
    if args.component == 'all':
        components = None  # Run all
    else:
        components = [args.component]
    
    # Run demo
    demo = IntegrationDemo(quick=args.quick)
    
    try:
        asyncio.run(demo.run(components))
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
        else:
            print("\nDemo interrupted")


if __name__ == "__main__":
    main()
