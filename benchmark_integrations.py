"""
Integration Benchmarking Suite - License: Apache 2.0

Performance benchmarking tools for OpenEvolve integration components.
Measures throughput, latency, and resource utilization.

Run: python benchmark_integrations.py --all
"""

import asyncio
import time
import statistics
from typing import List, Dict, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import argparse

# Rich for output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TaskID
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# =============================================================================
# BENCHMARK DATA MODELS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_rps: float
    success_rate: float
    memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        return {
            'suite_name': self.name,
            'total_benchmarks': len(self.results),
            'total_time_seconds': (self.end_time - self.start_time).total_seconds(),
            'avg_throughput': statistics.mean([r.throughput_rps for r in self.results]) if self.results else 0,
            'results': [r.to_dict() for r in self.results]
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs benchmarks and collects results."""
    
    def __init__(self, warmup_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []
    
    async def run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        concurrent: int = 1,
        setup_func: Callable = None,
        teardown_func: Callable = None,
        metadata: Dict = None
    ) -> BenchmarkResult:
        """Run a benchmark."""
        if console:
            console.print(f"[cyan]Running benchmark: {name}[/cyan]")
        
        # Warmup
        if setup_func:
            await setup_func()
        
        for _ in range(self.warmup_iterations):
            await func()
        
        # Benchmark
        times = []
        success_count = 0
        
        start_mem = self._get_memory_usage()
        
        if concurrent == 1:
            # Sequential
            for i in range(iterations):
                start = time.perf_counter()
                try:
                    await func()
                    success_count += 1
                except Exception as e:
                    if console:
                        console.print(f"[red]Error in iteration {i}: {e}[/red]")
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        else:
            # Concurrent
            semaphore = asyncio.Semaphore(concurrent)
            
            async def run_with_limit():
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        await func()
                        nonlocal success_count
                        success_count += 1
                    except Exception:
                        pass
                    return (time.perf_counter() - start) * 1000
            
            tasks = [run_with_limit() for _ in range(iterations)]
            times = await asyncio.gather(*tasks)
        
        end_mem = self._get_memory_usage()
        
        if teardown_func:
            await teardown_func()
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = iterations / (total_time / 1000) if total_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_rps=throughput,
            success_rate=success_count / iterations,
            memory_mb=end_mem - start_mem,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def print_results(self):
        """Print results in table format."""
        if not RICH_AVAILABLE:
            print(json.dumps([r.to_dict() for r in self.results], indent=2))
            return
        
        table = Table(title="Benchmark Results")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Iterations", justify="right")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")
        table.add_column("StdDev", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Success", justify="right")
        
        for result in self.results:
            success_color = "green" if result.success_rate >= 0.99 else "yellow" if result.success_rate >= 0.9 else "red"
            
            table.add_row(
                result.name,
                str(result.iterations),
                f"{result.avg_time_ms:.2f}",
                f"{result.min_time_ms:.2f}",
                f"{result.max_time_ms:.2f}",
                f"{result.std_dev_ms:.2f}",
                f"{result.throughput_rps:.1f} req/s",
                f"[{success_color}]{result.success_rate*100:.1f}%[/{success_color}]"
            )
        
        console.print(table)
    
    def save_results(self, path: Path):
        """Save results to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': [r.to_dict() for r in self.results]
        }
        path.write_text(json.dumps(data, indent=2))
        if console:
            console.print(f"[green]Results saved to {path}[/green]")


# =============================================================================
# SPECIFIC BENCHMARKS
# =============================================================================

async def benchmark_stage6_knowledge_extraction(runner: BenchmarkRunner):
    """Benchmark Stage 6 Knowledge Extraction."""
    from stage6_knowledge_extraction import (
        Stage6KnowledgeExtraction, ExecutionTrace, PatternExtractor
    )
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        engine = Stage6KnowledgeExtraction(storage_path=Path(tmp))
        
        # Create test traces
        traces = [
            ExecutionTrace(
                trace_id=f"t{i}",
                workflow_id=f"w{i}",
                problem_description=f"Optimization problem {i % 5}",
                stages=[
                    {"stage_name": "decompose", "parameters": {"strategy": "hybrid"}},
                    {"stage_name": "evolve", "parameters": {"generations": 50 + i}}
                ],
                final_result={"fitness": 0.9 + i * 0.01},
                execution_time_ms=1000.0,
                timestamp=datetime.now()
            )
            for i in range(100)
        ]
        
        # Benchmark single trace processing
        trace_iter = iter(traces)
        await runner.run_benchmark(
            name="Stage6: Process Single Trace",
            func=lambda: engine.process_trace(next(trace_iter)),
            iterations=50,
            metadata={'component': 'stage6'}
        )
        
        # Benchmark pattern extraction
        extractor = PatternExtractor()
        await runner.run_benchmark(
            name="Stage6: Extract Sequence Patterns",
            func=lambda: extractor.extract_sequence_patterns(traces[:50]),
            iterations=20,
            metadata={'component': 'stage6', 'pattern_type': 'sequence'}
        )


async def benchmark_event_bus(runner: BenchmarkRunner):
    """Benchmark Event Bus."""
    from event_bus import InMemoryEventBus, WorkflowEvent, EventType
    
    bus = InMemoryEventBus()
    await bus.connect()
    
    # Setup subscriber
    received_count = 0
    async def handler(event):
        nonlocal received_count
        received_count += 1
    
    await bus.subscribe("benchmark_channel", handler)
    
    event_counter = 0
    def create_event():
        nonlocal event_counter
        event_counter += 1
        return WorkflowEvent(
            id=f"evt_{event_counter}",
            type=EventType.WORKFLOW_STARTED,
            payload={"test": "data"},
            timestamp=datetime.now(),
            priority=1
        )
    
    # Benchmark publish
    await runner.run_benchmark(
        name="EventBus: Publish Event",
        func=lambda: bus.publish("benchmark_channel", create_event()),
        iterations=1000,
        metadata={'component': 'event_bus', 'operation': 'publish'}
    )
    
    await bus.disconnect()


async def benchmark_api_gateway(runner: BenchmarkRunner):
    """Benchmark API Gateway."""
    try:
        from fastapi.testclient import TestClient
        from api_gateway import APIGateway
        
        gateway = APIGateway()
        client = TestClient(gateway.app)
        
        # Benchmark root endpoint
        await runner.run_benchmark(
            name="API Gateway: Root Endpoint",
            func=lambda: client.get("/"),
            iterations=1000,
            metadata={'component': 'api_gateway', 'endpoint': 'root'}
        )
        
        # Benchmark health endpoint
        await runner.run_benchmark(
            name="API Gateway: Health Endpoint",
            func=lambda: client.get("/health"),
            iterations=500,
            metadata={'component': 'api_gateway', 'endpoint': 'health'}
        )
        
    except ImportError:
        if console:
            console.print("[yellow]FastAPI not available, skipping API Gateway benchmark[/yellow]")


async def benchmark_plugin_registry(runner: BenchmarkRunner):
    """Benchmark Plugin Registry."""
    from plugin_registry import PluginRegistry, PluginMetadata, PluginType
    from unittest.mock import Mock
    
    registry = PluginRegistry()
    
    # Create mock plugins
    plugins = []
    for i in range(100):
        mock_plugin = Mock()
        mock_plugin.metadata = PluginMetadata(
            name=f"plugin_{i}",
            version="1.0.0",
            description="Test plugin",
            author="Test",
            license="Apache-2.0",
            plugin_type=PluginType.MCP_TOOL,
            capabilities=[]
        )
        registry._plugins[f"plugin_{i}"] = mock_plugin
        plugins.append(mock_plugin)
    
    # Benchmark plugin listing
    await runner.run_benchmark(
        name="Plugin Registry: List Plugins",
        func=registry.list_plugins,
        iterations=1000,
        metadata={'component': 'plugin_registry', 'operation': 'list'}
    )
    
    # Benchmark plugin lookup
    counter = 0
    def get_next_plugin():
        nonlocal counter
        counter = (counter + 1) % 100
        return registry._plugins.get(f"plugin_{counter}")
    
    await runner.run_benchmark(
        name="Plugin Registry: Lookup Plugin",
        func=get_next_plugin,
        iterations=10000,
        metadata={'component': 'plugin_registry', 'operation': 'lookup'}
    )


# =============================================================================
# MAIN
# =============================================================================

async def run_all_benchmarks(output_dir: Path = None):
    """Run all benchmarks."""
    runner = BenchmarkRunner(warmup_iterations=5)
    
    if console:
        console.print("[bold green]OpenEvolve Integration Benchmark Suite[/bold green]")
        console.print(f"Started at: {datetime.now().isoformat()}\n")
    
    benchmarks = [
        ("Stage 6 Knowledge Extraction", benchmark_stage6_knowledge_extraction),
        ("Event Bus", benchmark_event_bus),
        ("API Gateway", benchmark_api_gateway),
        ("Plugin Registry", benchmark_plugin_registry),
    ]
    
    for name, benchmark_func in benchmarks:
        if console:
            console.print(f"\n[bold]{name}[/bold]")
        
        try:
            await benchmark_func(runner)
        except Exception as e:
            if console:
                console.print(f"[red]Error in {name}: {e}[/red]")
    
    # Print results
    runner.print_results()
    
    # Save results
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runner.save_results(output_dir / f"benchmark_{timestamp}.json")
    
    return runner.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenEvolve Integration Benchmarks")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for results")
    parser.add_argument("--component", "-c", help="Run specific component benchmark")
    parser.add_argument("--iterations", "-i", type=int, default=100, help="Iterations per benchmark")
    
    args = parser.parse_args()
    
    asyncio.run(run_all_benchmarks(args.output))


if __name__ == "__main__":
    main()
