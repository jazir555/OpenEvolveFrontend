"""
Example: Using OpenEvolve-Hephaestus Delegation

This script demonstrates how to use the OpenEvolve-Hephaestus delegation
to solve complex problems with multi-agent orchestration.

Prerequisites:
1. Qdrant running: docker run -p 6333:6333 qdrant/qdrant
2. ANTHROPIC_API_KEY or OPENAI_API_KEY set
3. Git repository initialized in working directory
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openevolve_hephaestus_delegation import (
    create_openevolve_delegator,
    OpenEvolveHephaestusDelegator,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
import time

console = Console()


def print_header(title: str):
    """Print a formatted header"""
    console.print(Panel(title, style="bold blue", padding=(0, 2)))


def print_workflow_summary(execution):
    """Print workflow execution summary"""
    table = Table(title="Workflow Summary", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Workflow ID", execution.id)
    table.add_row("Description", execution.description[:50] + "..." if len(execution.description) > 50 else execution.description)
    table.add_row("Status", f"[bold]{execution.status}[/bold]")
    table.add_row("Active Tasks", str(execution.active_tasks))
    table.add_row("Total Tasks", str(execution.total_tasks))
    table.add_row("Done Tasks", str(execution.done_tasks))
    table.add_row("Failed Tasks", str(execution.failed_tasks))
    table.add_row("Active Agents", str(execution.active_agents))

    console.print(table)


async def example_simple_workflow():
    """Example 1: Simple decomposition workflow"""
    print_header("Example 1: Simple Decomposition Workflow")

    delegator = create_openevolve_delegator(
        working_directory=str(Path(__file__).parent / "workspace" / "simple"),
        auto_start=True,
    )

    try:
        # Start workflow
        console.print("\n[yellow]Starting workflow...[/yellow]")
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement="Implement a binary search tree in Python with insert, delete, and search operations",
            problem_domain="Software Development",
            complexity_level="Medium (4-7)",
            max_sub_problems=5,
        )

        console.print(f"[green]✓[/green] Workflow started: {workflow_id}")

        # Monitor with progress display
        console.print("\n[yellow]Monitoring workflow...[/yellow]\n")

        with Live(console, refresh_per_second=4) as live:
            last_status = None
            while True:
                execution = await delegator.get_workflow_status(workflow_id)
                metrics = delegator.get_metrics(workflow_id)

                # Build status display
                status_text = f"""
[bold cyan]Workflow:[/bold cyan] {execution.id}
[bold cyan]Status:[/bold cyan] {execution.status}
[bold cyan]Progress:[/bold cyan] {execution.done_tasks}/{execution.total_tasks} tasks ({metrics.completion_percentage:.1f}%)
[bold cyan]Active Agents:[/bold cyan] {execution.active_agents}
[bold cyan]Duration:[/bold cyan] {metrics.duration_seconds:.1f}s
                """.strip()

                if execution.status != last_status:
                    live.update(Panel(status_text, title=f"[bold blue]Workflow Status[/bold blue]"))
                    last_status = execution.status

                # Check if complete
                if execution.status in ["completed", "failed"]:
                    live.stop()
                    break

                await asyncio.sleep(2)

        # Print final summary
        console.print("\n")
        print_workflow_summary(execution)

        if execution.status == "completed":
            console.print("\n[green]✓ Workflow completed successfully![/green]")
        else:
            console.print("\n[red]✗ Workflow failed[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        delegator.shutdown()
        console.print("[green]✓ Shutdown complete[/green]")


async def example_custom_configuration():
    """Example 2: Custom configuration"""
    print_header("Example 2: Custom Configuration")

    from openevolve_hephaestus_delegation import OpenEvolveHephaestusDelegator
    from src.sdk.config import HephaestusConfig

    # Custom configuration
    config = HephaestusConfig(
        database_path="./custom_hephaestus.db",
        qdrant_url="http://localhost:6333",
        mcp_port=8000,
        llm_provider="anthropic",
        # Uses ANTHROPIC_API_KEY from environment
        working_directory=str(Path(__file__).parent / "workspace" / "custom"),
        main_repo_path=str(Path(__file__).parent / "workspace" / "custom"),
        project_root=str(Path(__file__).parent / "workspace" / "custom"),
        monitoring_interval=30,
        log_level="DEBUG",
    )

    delegator = OpenEvolveHephaestusDelegator(
        hephaestus_config=config,
        working_directory=str(Path(__file__).parent / "workspace" / "custom"),
        auto_start=True,
    )

    try:
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement="Design a RESTful API for a task management system",
            problem_domain="Software Development",
            complexity_level="High (8-10)",
            max_sub_problems=8,
        )

        console.print(f"[green]✓[/green] Workflow started: {workflow_id}")

        # Monitor with callback
        async def status_callback(execution):
            console.print(
                f"  Status: {execution.status} | "
                f"Tasks: {execution.done_tasks}/{execution.total_tasks} | "
                f"Agents: {execution.active_agents}"
            )

        execution = await delegator.monitor_workflow(
            workflow_id,
            callback=status_callback,
            poll_interval=5,
        )

        print_workflow_summary(execution)

    finally:
        delegator.shutdown()


async def example_list_workflows():
    """Example 3: List and inspect workflows"""
    print_header("Example 3: List and Inspect Workflows")

    delegator = create_openevolve_delegator(auto_start=True)

    try:
        # List all workflows
        console.print("\n[yellow]Listing all workflows...[/yellow]\n")

        workflows = await delegator.list_workflows(status="all")

        if not workflows:
            console.print("[yellow]No workflows found[/yellow]")
            return

        table = Table(title="All Workflows")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Tasks", style="magenta")
        table.add_column("Agents", style="blue")

        for wf in workflows:
            table.add_row(
                wf.id[:20] + "...",
                (wf.description[:30] + "...") if len(wf.description) > 30 else wf.description,
                wf.status,
                f"{wf.done_tasks}/{wf.total_tasks}",
                str(wf.active_agents),
            )

        console.print(table)

        # Get detailed status of first workflow
        if workflows:
            first_wf = workflows[0]
            console.print(f"\n[yellow]Details for workflow: {first_wf.id}[/yellow]\n")
            execution = await delegator.get_workflow_status(first_wf.id)
            print_workflow_summary(execution)

            # Get metrics
            metrics = delegator.get_metrics(first_wf.id)
            console.print(f"\n[bold]Metrics:[/bold]")
            console.print(f"  Duration: {metrics.duration_seconds:.1f}s")
            console.print(f"  Completion: {metrics.completion_percentage:.1f}%")
            console.print(f"  Tasks: {metrics.completed_tasks}/{metrics.total_tasks}")
            console.print(f"  Failed: {metrics.failed_tasks}")

    finally:
        delegator.shutdown()


async def example_context_manager():
    """Example 4: Using context manager for automatic cleanup"""
    print_header("Example 4: Context Manager Usage")

    async with create_openevolve_delegator(auto_start=True) as delegator:
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement="Solve the N-Queens problem using backtracking",
            problem_domain="Mathematics",
            complexity_level="Medium (4-7)",
        )

        console.print(f"[green]✓[/green] Workflow started: {workflow_id}")

        execution = await delegator.monitor_workflow(
            workflow_id,
            poll_interval=3,
        )

        console.print(f"\n[green]✓[/green] Workflow {execution.status}")

    # Automatic shutdown happens here
    console.print("[green]✓ Automatically shut down[/green]")


async def example_health_check():
    """Example 5: Health check and diagnostics"""
    print_header("Example 5: Health Check and Diagnostics")

    delegator = create_openevolve_delegator(auto_start=True)

    try:
        # Check health
        console.print("\n[yellow]Checking system health...[/yellow]\n")

        health = delegator.is_healthy()

        table = Table(title="System Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for component, status in health.items():
            status_text = "[green]✓ Healthy[/green]" if status else "[red]✗ Unhealthy[/red]"
            table.add_row(component, status_text)

        console.print(table)

        if health["overall"]:
            console.print("\n[green]✓ All systems healthy[/green]")
        else:
            console.print("\n[red]✗ Some systems unhealthy[/red]")

        # List workflow definitions
        console.print("\n[yellow]Registered workflow definitions:[/yellow]\n")
        definitions = delegator.sdk.list_workflow_definitions()

        for defn in definitions:
            console.print(f"  • [cyan]{defn.name}[/cyan] ({defn.id})")
            console.print(f"    {defn.description}")
            console.print(f"    Phases: {len(defn.phases)}")
            console.print("")

    finally:
        delegator.shutdown()


async def main():
    """Run all examples"""
    console.print("\n[bold cyan]OpenEvolve-Hephaestus Delegation Examples[/bold cyan]\n")

    examples = [
        ("Simple Workflow", example_simple_workflow),
        ("Custom Configuration", example_custom_configuration),
        ("List Workflows", example_list_workflows),
        ("Context Manager", example_context_manager),
        ("Health Check", example_health_check),
    ]

    # Create menu
    console.print("[bold]Available Examples:[/bold]\n")
    for i, (name, _) in enumerate(examples, 1):
        console.print(f"  {i}. {name}")
    console.print(f"  0. Run all examples")
    console.print("")

    # Get user choice
    try:
        choice = console.input("[bold cyan]Select example (0-5): [/bold cyan]")
        choice = int(choice)
    except (ValueError, KeyboardInterrupt):
        console.print("\n[yellow]Exiting...[/yellow]")
        return

    console.print("")

    # Run selected example(s
    if choice == 0:
        for name, example_func in examples:
            try:
                await example_func()
                console.print("\n" + "="*60 + "\n")
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Example failed: {e}[/red]\n")
    elif 1 <= choice <= len(examples):
        name, example_func = examples[choice - 1]
        try:
            await example_func()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Example failed: {e}[/red]")
    else:
        console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]")
