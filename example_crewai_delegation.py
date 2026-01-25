"""
Example: Using OpenEvolve-CrewAI Delegation

This script demonstrates how to use the OpenEvolve-CrewAI delegation
to solve complex problems with multi-agent orchestration.

MIGRATED FROM: example_hephaestus_delegation.py
MIGRATION DATE: 2026-01-21
LICENSE CHANGE: Hephaestus (AGPL) → CrewAI (MIT)

NEW PATTERNS SHOWCASED:
- CrewAI Flows for multi-agent orchestration
- Event-driven workflow design (@start, @listen, @router)
- CrewAI-native state management (Pydantic models)
- MDAP/MAKER integration within CrewAI

Prerequisites:
1. ANTHROPIC_API_KEY or OPENAI_API_KEY set
2. Git repository initialized in working directory
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CREWAI IMPORTS
# =============================================================================

from crewai_unified_bridge import CrewAIUnifiedBridge
from crewai_state_management import WorkflowState, DecompositionPlan
from crewai_client import CrewAIClient

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


def print_workflow_summary(state: WorkflowState):
    """Print workflow state summary"""
    table = Table(title="Workflow Summary", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Workflow ID", state.workflow_id)
    table.add_row("Description", state.problem_statement[:50] + "..." if len(state.problem_statement) > 50 else state.problem_statement)
    table.add_row("Status", f"[bold]{state.status}[/bold]")
    table.add_row("Current Phase", state.current_phase)
    table.add_row("Sub-Problems", str(len(state.sub_problems)))
    table.add_row("Completed", str(state.completed))

    console.print(table)


async def example_simple_workflow():
    """Example 1: Simple CrewAI workflow"""
    print_header("Example 1: Simple CrewAI Workflow")

    bridge = CrewAIUnifiedBridge(
        working_directory=str(Path(__file__).parent / "workspace" / "simple"),
        auto_start=True,
    )

    try:
        # Start workflow
        console.print("\n[yellow]Starting CrewAI workflow...[/yellow]")

        state = await bridge.execute_workflow(
            method="auto",
            problem_statement="Implement a binary search tree in Python with insert, delete, and search operations",
            problem_domain="Software Development",
            complexity_level="Medium (4-7)",
            max_sub_problems=5,
        )

        console.print(f"[green]✓[/green] Workflow started: {state.workflow_id}")

        # Monitor with progress display
        console.print("\n[yellow]Monitoring workflow...[/yellow]\n")

        with Live(console, refresh_per_second=4) as live:
            last_status = None
            while True:
                state = await bridge.get_workflow_state(state.workflow_id)

                # Build status display
                status_text = f"""
[bold cyan]Workflow:[/bold cyan] {state.workflow_id}
[bold cyan]Status:[/bold cyan] {state.status}
[bold cyan]Phase:[/bold cyan] {state.current_phase}
[bold cyan]Progress:[/bold cyan] {state.completed}/{state.total_tasks} tasks
[bold cyan]Duration:[/bold cyan] {state.duration_seconds:.1f}s
                """.strip()

                if state.status != last_status:
                    live.update(Panel(status_text, title=f"[bold blue]Workflow Status[/bold blue]"))
                    last_status = state.status

                # Check if complete
                if state.status in ["completed", "failed"]:
                    live.stop()
                    break

                await asyncio.sleep(2)

        # Print final summary
        console.print("\n")
        print_workflow_summary(state)

        if state.status == "completed":
            console.print("\n[green]✓ Workflow completed successfully![/green]")
        else:
            console.print("\n[red]✗ Workflow failed[/red]")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        await bridge.shutdown()
        console.print("[green]✓ Shutdown complete[/green]")


async def example_custom_configuration():
    """Example 2: Custom configuration with ROMA/MDAP/MAKER"""
    print_header("Example 2: Custom Configuration (ROMA + MDAP + MAKER)")

    # Custom configuration with ROMA decomposition, MDAP debate, MAKER voting
    config = {
        "execution_method": "roma_mdap_maker",
        "llm_provider": "anthropic",
        "working_directory": str(Path(__file__).parent / "workspace" / "custom"),
        "roma_config": {
            "max_recursion_depth": 5,
            "use_associative_recomposition": True,
        },
        "mdap_config": {
            "num_agents": 5,
            "debate_rounds": 3,
        },
        "maker_config": {
            "voting_method": "first_to_ahead_by_k",
            "k_value": 2,
        },
    }

    bridge = CrewAIUnifiedBridge(
        working_directory=str(Path(__file__).parent / "workspace" / "custom"),
        auto_start=True,
        **config
    )

    try:
        state = await bridge.execute_workflow(
            method="roma_mdap_maker",
            problem_statement="Design a RESTful API for a task management system",
            problem_domain="Software Development",
            complexity_level="High (8-10)",
            max_sub_problems=8,
        )

        console.print(f"[green]✓[/green] Workflow started: {state.workflow_id}")

        # Monitor with callback
        async def status_callback(state: WorkflowState):
            console.print(
                f"  Status: {state.status} | "
                f"Phase: {state.current_phase} | "
                f"Tasks: {state.completed}/{state.total_tasks}"
            )

        final_state = await bridge.monitor_workflow(
            state.workflow_id,
            callback=status_callback,
            poll_interval=5,
        )

        print_workflow_summary(final_state)

    finally:
        await bridge.shutdown()


async def example_list_workflows():
    """Example 3: List and inspect workflows"""
    print_header("Example 3: List and Inspect Workflows")

    bridge = CrewAIUnifiedBridge(auto_start=True)

    try:
        # List all workflows
        console.print("\n[yellow]Listing all workflows...[/yellow]\n")

        workflows = await bridge.list_workflows(status="all")

        if not workflows:
            console.print("[yellow]No workflows found[/yellow]")
            return

        table = Table(title="All Workflows")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Tasks", style="magenta")
        table.add_column("Phase", style="blue")

        for wf in workflows:
            table.add_row(
                wf.workflow_id[:20] + "...",
                (wf.problem_statement[:30] + "...") if len(wf.problem_statement) > 30 else wf.problem_statement,
                wf.status,
                f"{wf.completed}/{wf.total_tasks}",
                wf.current_phase,
            )

        console.print(table)

        # Get detailed state of first workflow
        if workflows:
            first_wf = workflows[0]
            console.print(f"\n[yellow]Details for workflow: {first_wf.workflow_id}[/yellow]\n")
            state = await bridge.get_workflow_state(first_wf.workflow_id)
            print_workflow_summary(state)

    finally:
        await bridge.shutdown()


async def example_context_manager():
    """Example 4: Using context manager for automatic cleanup"""
    print_header("Example 4: Context Manager Usage")

    async with CrewAIUnifiedBridge(auto_start=True) as bridge:
        state = await bridge.execute_workflow(
            method="traditional",
            problem_statement="Solve the N-Queens problem using backtracking",
            problem_domain="Mathematics",
            complexity_level="Medium (4-7)",
        )

        console.print(f"[green]✓[/green] Workflow started: {state.workflow_id}")

        final_state = await bridge.monitor_workflow(
            state.workflow_id,
            poll_interval=3,
        )

        console.print(f"\n[green]✓[/green] Workflow {final_state.status}")

    # Automatic shutdown happens here
    console.print("[green]✓ Automatically shut down[/green]")


async def example_health_check():
    """Example 5: Health check and diagnostics"""
    print_header("Example 5: Health Check and Diagnostics")

    bridge = CrewAIUnifiedBridge(auto_start=True)

    try:
        # Check health
        console.print("\n[yellow]Checking system health...[/yellow]\n")

        health = await bridge.is_healthy()

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

        # List available execution methods
        console.print("\n[yellow]Available execution methods:[/yellow]\n")
        methods = bridge.list_execution_methods()

        for method in methods:
            console.print(f"  • [cyan]{method}[/cyan]")

    finally:
        await bridge.shutdown()


async def main():
    """Run all examples"""
    console.print("\n[bold cyan]OpenEvolve-CrewAI Delegation Examples[/bold cyan]\n")

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

    # Run selected example(s)
    if choice == 0:
        for name, example_func in examples:
            try:
                await example_func()
                console.print("\n" + "="*60 + "\n")
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                console.print(f"\n[red]Example failed: {e}[/red]")
                import traceback
                traceback.print_exc()
    elif 1 <= choice <= len(examples):
        name, example_func = examples[choice - 1]
        try:
            await example_func()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            console.print(f"\n[red]Example failed: {e}[/red]")
            import traceback
            traceback.print_exc()
    else:
        console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]")
