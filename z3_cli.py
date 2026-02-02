"""
Z3 Integration Command Line Interface

Unified CLI for the Z3-LeanAIDE-OpenEvolve-BubbleLabs integration.

Commands:
- solve: Solve constraint problems
- optimize: Run optimization
- prove: Prove theorems
- translate: Translate between formats
- server: Run API server
- monitor: Show performance metrics
- config: Manage configuration
- knowledge: Query knowledge base

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# CLI framework
try:
    import click
    from click import echo, style
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    # Create dummy click module
    class click:
        @staticmethod
        def command(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def option(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def argument(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def group(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def pass_context(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def echo(message, *args, **kwargs):
            print(message)
    echo = print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI Group
# =============================================================================

if CLICK_AVAILABLE:
    @click.group()
    @click.version_option(version="2.0.0")
    @click.option('--config', '-c', help='Configuration file path')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.pass_context
    def cli(ctx, config, verbose):
        """Z3-LeanAIDE-OpenEvolve Integration CLI"""
        ctx.ensure_object(dict)
        ctx.obj['config_path'] = config
        ctx.obj['verbose'] = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)


    # =============================================================================
    # Solve Command
    # =============================================================================

    @cli.command()
    @click.argument('problem', type=str)
    @click.option('--variables', '-v', help='Variables JSON')
    @click.option('--constraints', '-c', help='Constraints JSON')
    @click.option('--timeout', '-t', default=60.0, help='Timeout in seconds')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def solve(problem, variables, constraints, timeout, output, output_format):
        """Solve a constraint satisfaction problem."""
        try:
            from z3prover_integration import get_z3_solver_engine, Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style("Solving constraint problem...", fg='blue'))
            echo(f"Problem: {problem[:100]}...")
            
            solver = get_z3_solver_engine()
            
            # Parse inputs
            vars_list = json.loads(variables) if variables else []
            constraints_list = json.loads(constraints) if constraints else []
            
            z3_vars = [
                Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()])
                for v in vars_list
            ]
            
            z3_constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in constraints_list
            ]
            
            # If problem is provided and not empty, add it as a constraint 
            # or treat as SMT-LIB
            is_smtlib = any(kw in problem for kw in ['(assert', '(declare-fun', '(check-sat)'])
            
            # Solve
            import time
            start = time.time()
            
            if is_smtlib:
                result = solver.solve_smtlib(problem)
            else:
                if problem and problem.strip():
                    z3_constraints.append(Z3Constraint(problem, Z3ConstraintType.INTEGER))
                result = solver.solve_constraints(z3_vars, z3_constraints)
            
            elapsed = (time.time() - start) * 1000
            
            # Format output
            output_data = {
                "success": True,
                "status": result.status.value,
                "satisfiable": result.is_sat(),
                "model": result.model.assignments if result.model else None,
                "execution_time_ms": elapsed
            }
            
            _output_result(output_data, output, output_format)
            
            if result.is_sat():
                echo(style(f"✓ SATISFIABLE", fg='green'))
                if result.model:
                    echo("Solution:")
                    for var, val in result.model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style(f"✗ {result.status.value.upper()}", fg='red'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Optimize Command
    # =============================================================================

    @cli.command()
    @click.argument('objective', type=str)
    @click.option('--variables', '-v', required=True, help='Variables JSON')
    @click.option('--constraints', '-c', required=True, help='Constraints JSON')
    @click.option('--direction', '-d', default='minimize', type=click.Choice(['minimize', 'maximize']))
    @click.option('--output', '-o', help='Output file')
    def optimize(objective, variables, constraints, direction, output):
        """Run optimization."""
        try:
            from z3prover_advanced import get_z3_advanced_solver, OptimizationObjective
            from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style("Running optimization...", fg='blue'))
            
            solver = get_z3_advanced_solver()
            
            vars_list = json.loads(variables)
            constraints_list = json.loads(constraints)
            
            z3_vars = [Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()]) for v in vars_list]
            z3_constraints = [Z3Constraint(c, Z3ConstraintType.INTEGER) for c in constraints_list]
            
            obj_type = OptimizationObjective.MINIMIZE if direction == 'minimize' else OptimizationObjective.MAXIMIZE
            
            result = solver.optimize(z3_vars, z3_constraints, [(objective, obj_type)])
            
            if result.success:
                echo(style(f"✓ Optimal value: {result.optimal_value}", fg='green'))
                if result.optimal_model:
                    echo("Optimal solution:")
                    for var, val in result.optimal_model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style("✗ Optimization failed", fg='red'))
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Prove Command
    # =============================================================================

    @cli.command()
    @click.argument('theorem_file', type=click.Path(exists=True))
    @click.option('--extract-proof', is_flag=True, help='Extract detailed proof')
    @click.option('--timeout', '-t', default=300.0, help='Timeout in seconds')
    def prove(theorem_file, extract_proof, timeout):
        """Prove a theorem from file."""
        try:
            from z3prover_integration import get_z3_theorem_prover
            
            echo(style(f"Proving theorem from {theorem_file}...", fg='blue'))
            
            theorem = Path(theorem_file).read_text()
            
            prover = get_z3_theorem_prover()
            result = prover.prove_theorem(theorem)
            
            if result.proven:
                echo(style("✓ Theorem PROVEN", fg='green'))
                echo(f"Tactic used: {result.tactic_used}")
                if result.proof and extract_proof:
                    echo("\nProof:")
                    echo(result.proof[:500] + "..." if len(result.proof) > 500 else result.proof)
            else:
                echo(style("✗ Could not prove theorem", fg='red'))
                if result.counterexample:
                    echo("Counterexample found:")
                    echo(json.dumps(result.counterexample, indent=2))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Server Command
    # =============================================================================

    @cli.command()
    @click.option('--host', default='0.0.0.0', help='Host to bind to')
    @click.option('--port', '-p', default=8765, help='Port to bind to')
    @click.option('--reload', is_flag=True, help='Enable auto-reload')
    def server(host, port, reload):
        """Run the API server."""
        try:
            import uvicorn
            
            echo(style(f"Starting API server on {host}:{port}...", fg='blue'))
            echo(f"Documentation: http://{host}:{port}/docs")
            
            uvicorn.run(
                "z3_api_server:app",
                host=host,
                port=port,
                reload=reload
            )
        
        except ImportError:
            echo(style("Error: uvicorn not installed. Run: pip install uvicorn", fg='red'), err=True)
            sys.exit(1)
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Monitor Command
    # =============================================================================

    @cli.command()
    @click.option('--watch', '-w', is_flag=True, help='Continuous monitoring')
    @click.option('--interval', '-i', default=5.0, help='Update interval')
    def monitor(watch, interval):
        """Show performance metrics."""
        try:
            from z3_performance_monitor import get_z3_performance_monitor
            
            monitor = get_z3_performance_monitor()
            
            if watch:
                echo(style("Monitoring (press Ctrl+C to stop)...", fg='blue'))
                try:
                    while True:
                        _print_metrics(monitor)
                        echo("\n" + "-" * 50)
                        import time
                        time.sleep(interval)
                except KeyboardInterrupt:
                    echo(style("\nMonitoring stopped.", fg='yellow'))
            else:
                _print_metrics(monitor)
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    def _print_metrics(monitor):
        """Print current metrics."""
        dashboard = monitor.get_dashboard_data()
        
        echo(style("\n=== Performance Metrics ===", fg='blue', bold=True))
        
        summary = dashboard.get('summary', {})
        echo(f"Total Operations: {summary.get('total_operations', 0)}")
        echo(f"Total Calls: {summary.get('total_calls', 0)}")
        echo(f"Success Rate: {summary.get('overall_success_rate', 'N/A')}")
        echo(f"Active Alerts: {summary.get('active_alerts', 0)}")
        
        bottlenecks = dashboard.get('top_bottlenecks', [])
        if bottlenecks:
            echo(style("\nTop Bottlenecks:", fg='yellow'))
            for b in bottlenecks[:5]:
                echo(f"  {b['operation']}: {b['avg_time_s']:.3f}s")


    # =============================================================================
    # Config Command
    # =============================================================================

    @cli.group()
    def config():
        """Manage configuration."""
        pass

    @config.command('show')
    def config_show():
        """Show current configuration."""
        try:
            from z3_config_manager import get_config_manager
            
            cfg = get_config_manager()
            echo(json.dumps(cfg.to_dict(), indent=2))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    @config.command('validate')
    def config_validate():
        """Validate configuration."""
        try:
            from z3_config_manager import get_config_manager
            
            cfg = get_config_manager()
            errors = cfg.validate()
            
            if errors:
                echo(style("Validation errors:", fg='red'))
                for error in errors:
                    echo(f"  - {error}")
                sys.exit(1)
            else:
                echo(style("✓ Configuration is valid", fg='green'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)


    # =============================================================================
    # Knowledge Command
    # =============================================================================

    @cli.group()
    def knowledge():
        """Query knowledge base."""
        pass

    @knowledge.command('patterns')
    @click.option('--domain', '-d', help='Filter by domain')
    @click.option('--limit', '-l', default=10, help='Number of results')
    def knowledge_patterns(domain, limit):
        """Show learned proof patterns."""
        try:
            from z3_knowledge_extraction import get_z3_knowledge_extractor
            
            extractor = get_z3_knowledge_extractor()
            summary = extractor.get_knowledge_summary()
            
            echo(style(f"\n=== Proof Patterns ({summary['proof_patterns']['count']} total) ===", fg='blue'))
            
            patterns = summary['proof_patterns'].get('top_patterns', [])
            for p in patterns[:limit]:
                echo(f"\n  {p['name']}")
                echo(f"    Success rate: {p['success_rate']}")
                echo(f"    Usage count: {p['usage_count']}")
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    @knowledge.command('insights')
    @click.option('--category', '-c', help='Filter by category')
    def knowledge_insights(category):
        """Show mathematical insights."""
        try:
            from z3_knowledge_extraction import get_z3_knowledge_extractor
            
            extractor = get_z3_knowledge_extractor()
            
            insights = extractor.find_related_insights(category=category)
            
            echo(style(f"\n=== Mathematical Insights ({len(insights)} found) ===", fg='blue'))
            
            for i in insights[:10]:
                echo(f"\n  [{i.category}] {i.statement[:80]}...")
                echo(f"    Confidence: {i.confidence:.1%}")
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)


    # =============================================================================
    # Utility Functions
    # =============================================================================

    def _output_result(data: dict, output_file: Optional[str], output_format: str):
        """Output result to file or stdout."""
        if output_format == 'json':
            content = json.dumps(data, indent=2)
        elif output_format == 'yaml':
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False)
            except ImportError:
                content = json.dumps(data, indent=2)
        else:  # text
            content = str(data)
        
        if output_file:
            Path(output_file).write_text(content)
            echo(style(f"Output written to {output_file}", fg='green'))
        else:
            echo(content)


    # =============================================================================
    # Main Entry Point
    # =============================================================================

    def main():
        """Run the CLI."""
        cli()

else:
    # Fallback if click not available
    def main():
        print("Click is required for CLI. Install with: pip install click")
        print("\nAvailable commands would be:")
        print("  z3-cli solve <problem>")
        print("  z3-cli optimize <objective>")
        print("  z3-cli prove <theorem-file>")
        print("  z3-cli server")
        print("  z3-cli monitor")
        print("  z3-cli config show")
        print("  z3-cli knowledge patterns")


if __name__ == "__main__":
    main()
