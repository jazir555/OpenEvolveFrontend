"""
Mathematical Knowledge CLI

Command-line interface for:
- Solving problems with Z3/LeanAIDE
- Searching knowledge base
- Managing configurations
- Running benchmarks
- Exporting/Importing data
- CAV-NLP enhanced operations

Usage:
    python math_knowledge_cli.py solve --problem "x + y = 10"
    python math_knowledge_cli.py search --query "linear system"
    python math_knowledge_cli.py config --show
    python math_knowledge_cli.py benchmark --iterations 10
    python math_knowledge_cli.py formalize --text "x is greater than zero"

Author: OpenEvolve
Created: 2026-01-31
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Optional, List
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None


class MathKnowledgeCLI:
    """Command-line interface for mathematical knowledge system."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="math-knowledge",
            description="Mathematical Knowledge Integration CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s solve --problem "x + y = 10, x - y = 2" --solver z3
  %(prog)s prove --theorem "forall n, n + 0 = n"
  %(prog)s search --query "linear system" --top-k 5
  %(prog)s config --set z3.timeout_ms=60000
  %(prog)s benchmark --suite basic --iterations 10
  %(prog)s server --start --port 8765
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Solve command
        solve_parser = subparsers.add_parser("solve", help="Solve a problem")
        solve_parser.add_argument("--problem", "-p", required=True, help="Problem statement")
        solve_parser.add_argument("--solver", "-s", choices=["z3", "lean", "hybrid", "auto"],
                                  default="auto", help="Solver to use")
        solve_parser.add_argument("--timeout", "-t", type=int, default=30,
                                  help="Timeout in seconds")
        solve_parser.add_argument("--format", "-f", choices=["smtlib", "lean", "natural"],
                                  default="natural", help="Input format")
        solve_parser.add_argument("--output", "-o", help="Output file")
        
        # Prove command
        prove_parser = subparsers.add_parser("prove", help="Prove a theorem")
        prove_parser.add_argument("--theorem", "-t", required=True, help="Theorem statement")
        prove_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
        prove_parser.add_argument("--output", "-o", help="Output file")
        
        # Search command
        search_parser = subparsers.add_parser("search", help="Search knowledge base")
        search_parser.add_argument("--query", "-q", required=True, help="Search query")
        search_parser.add_argument("--top-k", "-k", type=int, default=5,
                                   help="Number of results")
        search_parser.add_argument("--type", choices=["all", "proof", "strategy", "tactic"],
                                   default="all", help="Result type")
        
        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_parser.add_argument("--show", action="store_true", help="Show current config")
        config_parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                                   help="Set configuration value")
        config_parser.add_argument("--file", "-f", help="Config file path")
        config_parser.add_argument("--save", help="Save config to file")
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
        benchmark_parser.add_argument("--suite", choices=["basic", "comprehensive", "stress"],
                                      default="basic", help="Benchmark suite")
        benchmark_parser.add_argument("--iterations", "-n", type=int, default=10,
                                      help="Number of iterations")
        benchmark_parser.add_argument("--output", "-o", help="Output file for results")
        benchmark_parser.add_argument("--compare", help="Compare with previous results")
        
        # Server command
        server_parser = subparsers.add_parser("server", help="Manage API server")
        server_parser.add_argument("--start", action="store_true", help="Start server")
        server_parser.add_argument("--stop", action="store_true", help="Stop server")
        server_parser.add_argument("--status", action="store_true", help="Check status")
        server_parser.add_argument("--port", "-p", type=int, default=8765, help="Port")
        server_parser.add_argument("--host", default="0.0.0.0", help="Host")
        
        # Knowledge command
        knowledge_parser = subparsers.add_parser("knowledge", help="Knowledge base operations")
        knowledge_parser.add_argument("--stats", action="store_true", help="Show statistics")
        knowledge_parser.add_argument("--export", help="Export to file")
        knowledge_parser.add_argument("--import-file", dest="import_file", help="Import from file")
        knowledge_parser.add_argument("--clear", action="store_true", help="Clear knowledge base")
        
        # Health command
        subparsers.add_parser("health", help="Check system health")
        
        # Version command
        subparsers.add_parser("version", help="Show version")
        
        return parser
    
    async def run(self, args: Optional[List[str]] = None):
        """Run CLI with arguments."""
        parsed = self.parser.parse_args(args)
        
        if not parsed.command:
            self.parser.print_help()
            return 0
        
        command_map = {
            "solve": self._cmd_solve,
            "prove": self._cmd_prove,
            "search": self._cmd_search,
            "config": self._cmd_config,
            "benchmark": self._cmd_benchmark,
            "server": self._cmd_server,
            "knowledge": self._cmd_knowledge,
            "health": self._cmd_health,
            "version": self._cmd_version,
        }
        
        handler = command_map.get(parsed.command)
        if handler:
            try:
                return await handler(parsed)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        else:
            print(f"Unknown command: {parsed.command}", file=sys.stderr)
            return 1
    
    async def _cmd_solve(self, args) -> int:
        """Handle solve command."""
        print(f"Solving with {args.solver}...")
        print(f"Problem: {args.problem}")
        
        from unified_math_bridge_complete import get_unified_bridge_complete, SolverSystem
        
        bridge = await get_unified_bridge_complete()
        
        solver_map = {
            "z3": SolverSystem.Z3,
            "lean": SolverSystem.LEANAIDE,
            "hybrid": SolverSystem.HYBRID,
            "auto": SolverSystem.AUTO
        }
        
        result = await bridge.solve(
            problem=args.problem,
            preferred_solver=solver_map[args.solver],
            timeout=args.timeout
        )
        
        print(f"\nResult:")
        print(f"  Status: {result.get('result_status')}")
        print(f"  Solver: {result.get('primary_solver')}")
        print(f"  Verified: {result.get('verified', False)}")
        
        if result.get('result'):
            print(f"  Solution: {result['result']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to {args.output}")
        
        return 0
    
    async def _cmd_prove(self, args) -> int:
        """Handle prove command."""
        print(f"Proving theorem...")
        print(f"Theorem: {args.theorem}")
        
        from leanaide_real_connector import get_leanaide_connector
        
        connector = await get_leanaide_connector()
        
        result = await connector.prove_theorem(
            args.theorem,
            timeout=args.timeout
        )
        
        print(f"\nResult:")
        print(f"  Success: {result.get('success')}")
        
        if result.get('success'):
            print(f"  Proof: {result.get('proof', 'N/A')[:200]}...")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to {args.output}")
        
        return 0
    
    async def _cmd_search(self, args) -> int:
        """Handle search command."""
        print(f"Searching for: {args.query}")
        
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        manager = await get_z3_knowledge_manager()
        
        results = await manager.find_similar_solutions(
            problem_statement=args.query,
            constraints=[],
            top_k=args.top_k
        )
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] Similarity: {result.get('similarity', 0):.2f}")
            print(f"      Problem: {result.get('problem', 'N/A')[:60]}...")
            print(f"      Strategy: {result.get('metadata', {}).get('strategy', 'unknown')}")
        
        return 0
    
    async def _cmd_config(self, args) -> int:
        """Handle config command."""
        from math_knowledge_config import MathKnowledgeConfig, load_config
        
        if args.show:
            config = load_config(args.file) if args.file else MathKnowledgeConfig()
            print("Current configuration:")
            print(config.to_yaml())
        
        if args.set:
            config = MathKnowledgeConfig()
            for setting in args.set:
                key, value = setting.split('=', 1)
                # Parse nested keys like z3.timeout_ms
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                # Try to convert to int/float/bool
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                setattr(obj, parts[-1], value)
            
            if args.save:
                config.save(args.save)
                print(f"Configuration saved to {args.save}")
            else:
                print("Updated configuration:")
                print(config.to_yaml())
        
        return 0
    
    async def _cmd_benchmark(self, args) -> int:
        """Handle benchmark command."""
        print(f"Running {args.suite} benchmark ({args.iterations} iterations)...")
        
        from z3_solver_connector import get_z3_connector, Z3SolverConfig
        import time
        
        z3 = get_z3_connector()
        
        # Test problems
        problems = {
            "basic": [
                "(declare-fun x () Int) (assert (> x 0)) (check-sat)",
                "(declare-fun y () Int) (assert (< y 10)) (check-sat)",
            ],
            "comprehensive": [
                "(declare-fun x () Int) (declare-fun y () Int) (assert (= (+ x y) 10)) (check-sat)",
                "(declare-fun a () Int) (assert (and (> a 0) (< a 100))) (check-sat)",
            ],
            "stress": [
                "(declare-fun x () Int) (assert (> (* x x) 1000)) (check-sat)",
            ]
        }
        
        results = []
        for problem in problems.get(args.suite, problems["basic"]):
            times = []
            for _ in range(args.iterations):
                start = time.time()
                await z3.solve_smtlib(problem, Z3SolverConfig())
                times.append(time.time() - start)
            
            results.append({
                "problem": problem[:50],
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            })
        
        print(f"\nResults:")
        for r in results:
            print(f"  {r['problem']}...")
            print(f"    Avg: {r['avg_time']*1000:.1f}ms, Min: {r['min_time']*1000:.1f}ms, Max: {r['max_time']*1000:.1f}ms")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {args.output}")
        
        return 0
    
    async def _cmd_server(self, args) -> int:
        """Handle server command."""
        if args.start:
            print(f"Starting server on {args.host}:{args.port}...")
            
            import uvicorn
            from z3_api import app
            
            if app:
                uvicorn.run(app, host=args.host, port=args.port)
            else:
                print("Error: FastAPI app not available")
                return 1
        
        elif args.stop:
            print("Stopping server...")
            # Would need process management
            print("Not implemented - use Ctrl+C or kill command")
        
        elif args.status:
            import urllib.request
            try:
                with urllib.request.urlopen(f"http://{args.host}:{args.port}/health") as resp:
                    print(f"Server status: {resp.status}")
                    print(resp.read().decode())
            except Exception as e:
                print(f"Server not responding: {e}")
                return 1
        
        return 0
    
    async def _cmd_knowledge(self, args) -> int:
        """Handle knowledge command."""
        from z3_knowledge_complete import get_z3_knowledge_manager
        
        manager = await get_z3_knowledge_manager()
        
        if args.stats:
            stats = manager.get_statistics()
            print("Knowledge Base Statistics:")
            print(json.dumps(stats, indent=2))
        
        if args.export:
            print(f"Exporting knowledge base to {args.export}...")
            # Implementation would export all records
            print("Not yet implemented")
        
        if args.import_file:
            print(f"Importing from {args.import_file}...")
            print("Not yet implemented")
        
        if args.clear:
            confirm = input("Are you sure you want to clear the knowledge base? (yes/no): ")
            if confirm.lower() == "yes":
                print("Clearing knowledge base...")
                print("Not yet implemented")
            else:
                print("Cancelled")
        
        return 0
    
    async def _cmd_health(self, args) -> int:
        """Handle health command."""
        print("Checking system health...")
        
        from math_mcp_tools import get_math_mcp_tools
        
        tools = await get_math_mcp_tools()
        result = await tools.execute_tool("math_health_check", {})
        
        print(f"\nHealth Status:")
        print(f"  Z3 available: {result.get('z3_available', False)}")
        print(f"  LeanAIDE available: {result.get('leanaide_available', False)}")
        
        if result.get('z3_stats'):
            print(f"\n  Z3 Statistics:")
            print(f"    {json.dumps(result['z3_stats'], indent=4)}")
        
        return 0
    
    async def _cmd_version(self, args) -> int:
        """Handle version command."""
        print("Mathematical Knowledge Integration")
        print("Version: 1.0.0")
        print("Author: OpenEvolve")
        print("License: MIT")
        
        # Check component versions
        try:
            import z3
            print(f"\nZ3 version: {z3.get_version_string()}")
        except:
            print("\nZ3: not available")
        
        try:
            import sqlalchemy
            print(f"SQLAlchemy version: {sqlalchemy.__version__}")
        except:
            print("SQLAlchemy: not available")
        
        return 0


def main():
    """Main entry point."""
    cli = MathKnowledgeCLI()
    return asyncio.run(cli.run())


if __name__ == "__main__":
    sys.exit(main())
