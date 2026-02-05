"""
OpenEvolve CLI - License: Apache 2.0

Command-line interface for managing OpenEvolve services,
plugins, configuration, and deployments.

Dependencies (all permissive licenses):
- click: MIT License
- rich: MIT License
- pydantic: MIT License

Author: OpenEvolve
Date: 2026-02-02
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Openevolve Cli
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import asyncio
import json
import sys
from typing import Optional, List
from pathlib import Path

# Click - MIT License
import click

# Rich - MIT License
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

console = Console()

# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """
    OpenEvolve CLI - Manage services, plugins, and configuration.
    
    Examples:
        openevolve services start
        openevolve plugins list
        openevolve config show
        openevolve status
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


# =============================================================================
# SERVICES COMMANDS
# =============================================================================

@cli.group()
def services():
    """Manage OpenEvolve services."""
    pass


@services.command()
@click.option('--all', 'start_all', is_flag=True, help='Start all services')
@click.option('--rest', is_flag=True, help='Start REST API')
@click.option('--graphql', is_flag=True, help='Start GraphQL API')
@click.option('--mcp', is_flag=True, help='Start MCP Server')
@click.option('--event-bus', is_flag=True, help='Start Event Bus')
@click.pass_context
def start(ctx, start_all, rest, graphql, mcp, event_bus):
    """Start OpenEvolve services."""
    from service_orchestrator import get_orchestrator, RESTAPIService, GraphQLService, MCPService, EventBusService
    from integration_config import get_config
    
    config = get_config(ctx.obj.get('config_path'))
    orchestrator = get_orchestrator()
    
    # Determine which services to start
    if start_all or not any([rest, graphql, mcp, event_bus]):
        rest = graphql = mcp = event_bus = True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Starting services...", total=None)
        
        # Register requested services
        if rest and config.services.get('rest_api', True):
            progress.update(task, description="Registering REST API...")
            orchestrator.register_service(RESTAPIService(port=config.rest_api.port))
        
        if graphql and config.services.get('graphql_api', True):
            progress.update(task, description="Registering GraphQL API...")
            orchestrator.register_service(GraphQLService(port=config.graphql.port))
        
        if event_bus and config.services.get('event_bus', True):
            progress.update(task, description="Registering Event Bus...")
            orchestrator.register_service(EventBusService())
        
        if mcp and config.services.get('mcp_server', True):
            progress.update(task, description="Registering MCP Server...")
            orchestrator.register_service(MCPService())
        
        # Start all registered services
        progress.update(task, description="Starting services...")
        
        async def do_start():
            return await orchestrator.start_all()
        
        results = asyncio.run(do_start())
    
    # Display results
    table = Table(title="Service Start Results")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    for name, success in results.items():
        status = "[OK] Started" if success else "[FAIL] Failed"
        style = "green" if success else "red"
        
        service = orchestrator.get_service(name)
        details = f"Port: {service.info.port}" if service and service.info.port else ""
        
        table.add_row(name, f"[{style}]{status}[/{style}]", details)
    
    console.print(table)
    
    if all(results.values()):
        console.print(Panel(
            f"[green]All services started successfully![/green]\n\n"
            f"REST API: http://localhost:{config.rest_api.port}\n"
            f"GraphQL:  http://localhost:{config.graphql.port}/graphql\n"
            f"MCP:      stdio/sse for Claude/Cursor",
            title="Access URLs"
        ))


@services.command()
@click.pass_context
def stop(ctx):
    """Stop all OpenEvolve services."""
    from service_orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    with console.status("[bold red]Stopping services..."):
        async def do_stop():
            return await orchestrator.stop_all()
        
        results = asyncio.run(do_stop())
    
    table = Table(title="Service Stop Results")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    
    for name, success in results.items():
        status = "[OK] Stopped" if success else "[FAIL] Error"
        style = "green" if success else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]")
    
    console.print(table)


@services.command()
@click.pass_context
def status(ctx):
    """Check service status."""
    from service_orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    table = Table(title="OpenEvolve Services")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Port", style="dim")
    table.add_column("Uptime", style="dim")
    
    for name, service in orchestrator.services.items():
        status_color = {
            "running": "green",
            "stopped": "dim",
            "error": "red",
            "starting": "yellow"
        }.get(service.info.status.value, "white")
        
        uptime = ""
        if service.info.uptime_seconds:
            mins = int(service.info.uptime_seconds // 60)
            secs = int(service.info.uptime_seconds % 60)
            uptime = f"{mins}m {secs}s"
        
        table.add_row(
            name,
            f"[{status_color}]{service.info.status.value}[/{status_color}]",
            str(service.info.port) if service.info.port else "-",
            uptime
        )
    
    console.print(table)


@services.command()
@click.argument('service_name')
@click.pass_context
def restart(ctx, service_name):
    """Restart a specific service."""
    from service_orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    service = orchestrator.get_service(service_name)
    
    if not service:
        console.print(f"[red]Service not found: {service_name}[/red]")
        sys.exit(1)
    
    with console.status(f"[yellow]Restarting {service_name}..."):
        asyncio.run(service.stop())
        success = asyncio.run(service.start())
    
    if success:
        console.print(f"[green][OK] {service_name} restarted successfully[/green]")
    else:
        console.print(f"[red][FAIL] Failed to restart {service_name}[/red]")


@services.command()
@click.pass_context
def health(ctx):
    """Check health of all services."""
    from service_orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    table = Table(title="Service Health")
    table.add_column("Service", style="cyan")
    table.add_column("Health", style="bold")
    table.add_column("Details", style="dim")
    
    async def check_health():
        for name, service in orchestrator.services.items():
            try:
                health = await service.health_check()
                status = health.get("status", "unknown")
                status_color = "green" if status == "healthy" else "yellow" if status == "degraded" else "red"
                
                details = ", ".join([f"{k}: {v}" for k, v in health.items() if k != "status"])
                table.add_row(name, f"[{status_color}]{status}[/{status_color}]", details[:50])
            except Exception as e:
                table.add_row(name, "[red]error[/red]", str(e)[:50])
    
    asyncio.run(check_health())
    console.print(table)


# =============================================================================
# PLUGINS COMMANDS
# =============================================================================

@cli.group()
def plugins():
    """Manage OpenEvolve plugins."""
    pass


@plugins.command()
def list():
    """List all registered plugins."""
    from plugin_registry import get_plugin_registry
    
    registry = get_plugin_registry()
    plugins_list = registry.list_plugins()
    
    if not plugins_list:
        console.print("[dim]No plugins registered[/dim]")
        return
    
    table = Table(title="Registered Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="dim")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="bold")
    table.add_column("Capabilities", style="dim")
    
    for info in plugins_list:
        status_color = {
            "registered": "dim",
            "loaded": "blue",
            "initialized": "green",
            "error": "red"
        }.get(info.status.value, "white")
        
        caps = ", ".join([c.value for c in info.metadata.capabilities[:3]])
        
        table.add_row(
            info.metadata.name,
            info.metadata.version,
            info.metadata.plugin_type.value,
            f"[{status_color}]{info.status.value}[/{status_color}]",
            caps
        )
    
    console.print(table)


@plugins.command()
@click.argument('path')
def load(path):
    """Load a plugin from file or module."""
    from plugin_registry import get_plugin_registry
    
    registry = get_plugin_registry()
    
    with console.status(f"[yellow]Loading plugin from {path}..."):
        if Path(path).exists():
            # Load from file
            success = asyncio.run(registry.load_from_file(path))
        else:
            # Load from module
            success = asyncio.run(registry.load_from_module(path))
    
    if success:
        console.print(f"[green][OK] Plugin loaded successfully[/green]")
    else:
        console.print(f"[red][FAIL] Failed to load plugin[/red]")


@plugins.command()
@click.argument('directory')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories')
def load_dir(directory, recursive):
    """Load all plugins from directory."""
    from plugin_registry import get_plugin_registry
    
    registry = get_plugin_registry()
    
    with console.status(f"[yellow]Loading plugins from {directory}..."):
        loaded = asyncio.run(registry.load_from_directory(directory, recursive))
    
    if loaded:
        console.print(f"[green][OK] Loaded {len(loaded)} plugins:[/green]")
        for name in loaded:
            console.print(f"  * {name}")
    else:
        console.print("[yellow]No plugins found[/yellow]")


@plugins.command()
@click.argument('name')
def init(name):
    """Initialize a registered plugin."""
    from plugin_registry import get_plugin_registry
    
    registry = get_plugin_registry()
    
    with console.status(f"[yellow]Initializing plugin {name}..."):
        success = asyncio.run(registry.initialize_plugin(name))
    
    if success:
        console.print(f"[green][OK] Plugin {name} initialized[/green]")
    else:
        console.print(f"[red][FAIL] Failed to initialize plugin {name}[/red]")


@plugins.command()
@click.argument('name')
def unload(name):
    """Unload a plugin."""
    from plugin_registry import get_plugin_registry
    
    registry = get_plugin_registry()
    
    with console.status(f"[yellow]Unloading plugin {name}..."):
        success = asyncio.run(registry.unload_plugin(name))
    
    if success:
        console.print(f"[green][OK] Plugin {name} unloaded[/green]")
    else:
        console.print(f"[red][FAIL] Failed to unload plugin {name}[/red]")


@plugins.command()
def create_example():
    """Create an example plugin file."""
    example_code = '''"""
Example OpenEvolve Plugin
"""

from plugin_registry import MCPToolPlugin, PluginMetadata, PluginType, PluginCapability

class MyPlugin(MCPToolPlugin):
    """Example plugin demonstrating MCP tool registration."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_example_plugin",
            version="1.0.0",
            description="An example plugin",
            author="Your Name",
            license="Apache-2.0",
            plugin_type=PluginType.MCP_TOOL,
            capabilities=[PluginCapability.WORKFLOW]
        )
    
    def __init__(self):
        super().__init__()
        
        # Register a custom tool
        self.register_tool(
            "my_custom_tool",
            self.handle_my_tool,
            {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input data"}
                },
                "required": ["input"],
                "description": "My custom tool description"
            }
        )
    
    async def handle_my_tool(self, args: dict) -> str:
        """Handle the custom tool."""
        user_input = args.get("input", "")
        return f"Processed: {user_input}"
    
    async def initialize(self, config: dict) -> bool:
        """Initialize the plugin."""
        console.print(f"Initializing {self.metadata.name}")
        return await super().initialize(config)
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        return True
'''
    
    output_path = Path("example_plugin.py")
    output_path.write_text(example_code)
    console.print(f"[green][OK] Example plugin created: {output_path}[/green]")


# =============================================================================
# CONFIG COMMANDS
# =============================================================================

@cli.group()
def config():
    """Manage OpenEvolve configuration."""
    pass


@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    from integration_config import get_config
    
    config = get_config(ctx.obj.get('config_path'))
    
    console.print(Panel(
        f"[bold cyan]OpenEvolve Configuration[/bold cyan]\n\n"
        f"Log Level: {config.log_level}\n"
        f"Orchestrator Port: {config.orchestrator_port}\n\n"
        f"[bold]Services:[/bold]\n"
        + "\n".join([f"  {'[OK]' if enabled else '[FAIL]'} {name}" 
                    for name, enabled in config.services.items()]),
        title="Configuration"
    ))
    
    # Service-specific config
    console.print(f"\n[bold]REST API:[/bold] http://{config.rest_api.host}:{config.rest_api.port}")
    console.print(f"[bold]GraphQL:[/bold] http://{config.graphql.host}:{config.graphql.port}/graphql")
    console.print(f"[bold]Telemetry:[/bold] {'enabled' if config.telemetry.enabled else 'disabled'}")
    console.print(f"[bold]Event Bus:[/bold] {'enabled' if config.event_bus.enabled else 'disabled'}")


@config.command()
@click.argument('output', type=click.Path())
@click.option('--format', 'fmt', type=click.Choice(['yaml', 'json']), default='yaml')
@click.pass_context
def generate(ctx, output, fmt):
    """Generate default configuration file."""
    from integration_config import default_config_yaml, IntegrationConfig
    
    output_path = Path(output)
    
    if fmt == 'yaml':
        output_path.write_text(default_config_yaml)
    else:
        config = IntegrationConfig()
        output_path.write_text(json.dumps(config.dict(), indent=2))
    
    console.print(f"[green][OK] Configuration written to {output}[/green]")


@config.command()
@click.pass_context
def validate(ctx):
    """Validate configuration file."""
    from integration_config import get_config
    
    try:
        config = get_config(ctx.obj.get('config_path'))
        console.print("[green][OK] Configuration is valid[/green]")
        
        # Show validation details
        tree = Tree("[bold]Configuration Structure[/bold]")
        
        services = tree.add("Services")
        for name, enabled in config.services.items():
            services.add(f"{'[OK]' if enabled else '[FAIL]'} {name}")
        
        tree.add(f"REST API: port {config.rest_api.port}")
        tree.add(f"GraphQL: port {config.graphql.port}")
        tree.add(f"Telemetry: {config.telemetry.service_name}")
        
        console.print(tree)
        
    except Exception as e:
        console.print(f"[red][FAIL] Configuration error: {e}[/red]")
        sys.exit(1)


# =============================================================================
# STATUS COMMAND
# =============================================================================

@cli.command()
@click.pass_context
def status(ctx):
    """Show complete system status."""
    from service_orchestrator import get_orchestrator
    from plugin_registry import get_plugin_registry
    from integration_config import get_config
    
    config = get_config(ctx.obj.get('config_path'))
    orchestrator = get_orchestrator()
    registry = get_plugin_registry()
    
    # Main status panel
    services_running = sum(
        1 for s in orchestrator.services.values()
        if s.info.status.value == 'running'
    )
    plugins_loaded = len(registry.list_plugins())
    
    console.print(Panel(
        f"[bold green]OpenEvolve System Status[/bold green]\n\n"
        f"Services Running: {services_running}/{len(orchestrator.services)}\n"
        f"Plugins Loaded: {plugins_loaded}\n"
        f"Config File: {ctx.obj.get('config_path', 'default')}\n",
        title="Status"
    ))
    
    # Services table
    if orchestrator.services:
        console.print("\n[bold]Services:[/bold]")
        services()
    
    # Plugins summary
    if plugins_loaded > 0:
        console.print(f"\n[bold]Plugins:[/bold] {plugins_loaded} loaded")


# =============================================================================
# DOCKER COMMAND
# =============================================================================

@cli.group()
def docker():
    """Docker deployment commands."""
    pass


@docker.command()
def generate():
    """Generate Docker Compose configuration."""
    docker_compose = '''version: '3.8'

services:
  # Valkey (Redis alternative - Apache 2.0)
  valkey:
    image: valkey/valkey:latest
    ports:
      - "6379:6379"
    volumes:
      - valkey_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # OpenEvolve Services
  openevolve:
    build: .
    ports:
      - "8000:8000"   # REST API
      - "8001:8001"   # GraphQL
      - "8080:8080"   # Orchestrator
    environment:
      - VALKEY_HOST=valkey
      - VALKEY_PORT=6379
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
      - LOG_LEVEL=INFO
    volumes:
      - ./openevolve.yaml:/app/openevolve.yaml:ro
      - ./plugins:/app/plugins:ro
    depends_on:
      valkey:
        condition: service_healthy
    restart: unless-stopped
    command: ["python", "-m", "openevolve_cli", "services", "start", "--all"]

  # Jaeger (OpenTelemetry collector - Apache 2.0)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped

volumes:
  valkey_data:
'''
    
    dockerfile = '''FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY *.py .
COPY valkey/ valkey/  # If building Valkey from source

# Create directories
RUN mkdir -p plugins configs

# Expose ports
EXPOSE 8000 8001 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Default command
CMD ["python", "-m", "openevolve_cli", "services", "start", "--all"]
'''
    
    Path('docker-compose.yml').write_text(docker_compose)
    Path('Dockerfile').write_text(dockerfile)
    
    console.print("[green][OK] Docker files generated:[/green]")
    console.print("  * docker-compose.yml")
    console.print("  * Dockerfile")
    console.print("\n[yellow]To start:[/yellow]")
    console.print("  docker-compose up -d")


# =============================================================================
# ADAPTIVE MDAP COMMANDS
# =============================================================================

@cli.group()
def adaptive():
    """Adaptive MDAP resource allocation commands."""
    pass


@adaptive.command()
@click.option('--description', '-d', required=True, help='Sub-problem description')
@click.option('--domain', '-D', default='general', help='Problem domain (e.g., security, math)')
@click.option('--depth', type=int, default=1, help='Decomposition depth')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def classify(description, domain, depth, json_output):
    """Classify sub-problem complexity."""
    try:
        from adaptive_mdap import TaskComplexityClassifier
        from adaptive_mdap.core.types import SubProblem
        
        # Create sub-problem
        sp = SubProblem(
            id="cli-classify",
            description=description,
            domain=domain,
            depth=depth,
            dependencies=[],
            metadata={}
        )
        
        # Classify
        classifier = TaskComplexityClassifier()
        score = classifier.compute_complexity(sp)
        
        if json_output:
            click.echo(json.dumps({
                "overall_score": score.overall_score,
                "text_length_score": score.text_length_score,
                "domain_rarity_score": score.domain_rarity_score,
                "depth_score": score.depth_score,
                "historical_error_score": score.historical_error_score,
                "dependency_score": score.dependency_score,
                "keyword_complexity_score": score.keyword_complexity_score,
                "constraint_density_score": score.constraint_density_score,
            }, indent=2))
        else:
            console.print(Panel(
                f"[bold]Complexity Score:[/bold] {score.overall_score:.3f}\n\n"
                f"Text Length: {score.text_length_score:.3f}\n"
                f"Domain Rarity: {score.domain_rarity_score:.3f}\n"
                f"Depth: {score.depth_score:.3f}\n"
                f"Historical Error: {score.historical_error_score:.3f}\n"
                f"Dependency: {score.dependency_score:.3f}\n"
                f"Keyword Complexity: {score.keyword_complexity_score:.3f}\n"
                f"Constraint Density: {score.constraint_density_score:.3f}",
                title="Complexity Classification"
            ))
            
    except ImportError:
        console.print("[red]Adaptive MDAP not available[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@adaptive.command()
@click.argument('complexity', type=float)
@click.option('--profile', '-p', type=click.Choice(['conservative', 'balanced', 'aggressive']), 
              default='balanced', help='Allocation profile')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def allocate(complexity, profile, json_output):
    """Allocate resources for a complexity score."""
    try:
        from adaptive_mdap import AdaptiveMDAPAllocator
        from adaptive_mdap.config.profiles import load_profile
        
        # Create allocator with profile
        profile_config = load_profile(profile)
        allocator = AdaptiveMDAPAllocator(profile=profile_config)
        
        # Allocate
        config = allocator.allocate_resources(complexity)
        
        if json_output:
            click.echo(json.dumps({
                "strategy": config.strategy.value,
                "n_agents": config.n_agents,
                "k_ahead": config.k_ahead,
                "timeout_ms": config.timeout_ms,
                "profile": profile
            }, indent=2))
        else:
            console.print(Panel(
                f"[bold]Allocation for complexity {complexity:.3f}[/bold]\n\n"
                f"Strategy: {config.strategy.value}\n"
                f"Agents: {config.n_agents}\n"
                f"K-Ahead: {config.k_ahead}\n"
                f"Timeout: {config.timeout_ms}ms\n"
                f"Profile: {profile}",
                title="Resource Allocation"
            ))
            
    except ImportError:
        console.print("[red]Adaptive MDAP not available[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@adaptive.command()
def status():
    """Show Adaptive MDAP status."""
    try:
        from adaptive_mdap import check_health
        from adaptive_mdap.utils.cache import get_cache_stats
        
        health = check_health()
        cache_stats = get_cache_stats()
        
        # Status table
        table = Table(title="Adaptive MDAP Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        for component, info in health.items():
            status_icon = "[OK]" if info.get('healthy', False) else "[FAIL]"
            status_color = "green" if info.get('healthy', False) else "red"
            table.add_row(
                component,
                f"[{status_color}]{status_icon} {info.get('status', 'unknown')}[/{status_color}]",
                info.get('message', '')
            )
        
        console.print(table)
        
        # Cache stats
        console.print(f"\n[bold]Cache Statistics:[/bold]")
        console.print(f"  Embedding cache: {cache_stats.get('embedding', {}).get('size', 0)} entries")
        console.print(f"  Feature cache: {cache_stats.get('feature', {}).get('size', 0)} entries")
        
    except ImportError:
        console.print("[red]Adaptive MDAP not available[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@adaptive.command()
def profiles():
    """List available allocation profiles."""
    try:
        from adaptive_mdap.config.profiles import get_profile_config
        
        profiles = ['conservative', 'balanced', 'aggressive']
        
        table = Table(title="Allocation Profiles")
        table.add_column("Profile", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Cost Focus", style="dim")
        
        table.add_row("conservative", "Maximum cost savings", "High")
        table.add_row("balanced", "Optimal cost-quality tradeoff", "Medium")
        table.add_row("aggressive", "Maximum quality", "Low")
        
        console.print(table)
        
        console.print("\n[dim]Use with: openevolve adaptive allocate --profile <name>[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
