"""
Performance Profiler for OpenEvolve Decomposition Engine

Provides comprehensive profiling capabilities including:
- Line-by-line function profiling
- Memory usage tracking
- Function call tracing
- Bottleneck detection
- Performance regression detection
- Statistical analysis of execution patterns
"""
from __future__ import annotations


import time
import functools
import inspect
import threading
import sys
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FunctionProfile:
    """Profile data for a single function"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_time: float = 0.0
    memory_usage: int = 0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    source_file: str = ""
    line_number: int = 0

    def update(self, execution_time: float, memory_delta: int = 0, error: bool = False):
        """Update profile with new execution data"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.last_time = execution_time
        self.memory_usage += memory_delta
        if error:
            self.error_count += 1
        self.timestamp = datetime.now()


@dataclass
class CallTrace:
    """Trace of a single function call"""
    function_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    memory_delta: int
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    exception: Optional[Exception] = None
    call_stack: List[str] = field(default_factory=list)
    thread_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "function_name": self.function_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "memory_delta": self.memory_delta,
            "args": str(self.args)[:200],  # Truncate long args
            "kwargs": str(self.kwargs)[:200],
            "exception": str(self.exception) if self.exception else None,
            "call_stack": self.call_stack,
            "thread_id": self.thread_id,
        }


class PerformanceProfiler:
    """
    Advanced performance profiler for decomposition engine.

    Features:
    - Function-level profiling with timing statistics
    - Memory usage tracking
    - Call stack tracing
    - Bottleneck detection
    - Performance regression detection
    - Statistical analysis
    """

    def __init__(self, enable_memory_profiling: bool = True,
                 max_traces: int = 10000,
                 auto_detect_bottlenecks: bool = True,
                 output_dir: str = "./profiling_output"):
        """
        Initialize the performance profiler.

        Args:
            enable_memory_profiling: Whether to track memory usage
            max_traces: Maximum number of call traces to store
            auto_detect_bottlenecks: Whether to automatically detect bottlenecks
            output_dir: Directory to save profiling reports
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.max_traces = max_traces
        self.auto_detect_bottlenecks = auto_detect_bottlenecks
        self.output_dir = output_dir

        # Profile storage
        self.profiles: Dict[str, FunctionProfile] = {}
        self.traces: List[CallTrace] = []
        self.bottlenecks: List[Dict[str, Any]] = []

        # Thread-local storage for concurrent profiling
        self._local = threading.local()

        # Lock for thread-safe operations
        self._lock = threading.RLock()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Performance baselines for regression detection
        self.baselines: Dict[str, Dict[str, float]] = {}

        logger.info("Performance profiler initialized")

    def profile(self, func: Optional[Callable] = None,
                name: Optional[str] = None,
                track_memory: bool = True,
                trace_calls: bool = False) -> Callable:
        """
        Decorator to profile a function.

        Args:
            func: Function to profile
            name: Custom name for the profile (defaults to function name)
            track_memory: Whether to track memory for this function
            trace_calls: Whether to trace individual calls

        Returns:
            Decorated function
        """
        def decorator(f: Callable) -> Callable:
            profile_name = name or f"{f.__module__}.{f.__name__}"

            # Get source information
            try:
                source_file = inspect.getsourcefile(f) or ""
                line_number = inspect.getsourcelines(f)[1]
            except (OSError, IOError, TypeError):
                source_file = ""
                line_number = 0

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return self._profile_execution(
                    f, args, kwargs, profile_name, source_file, line_number,
                    track_memory, trace_calls
                )

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    def _profile_execution(self, func: Callable, args: Tuple, kwargs: Dict,
                          profile_name: str, source_file: str, line_number: int,
                          track_memory: bool, trace_calls: bool) -> Any:
        """Execute function with profiling"""
        # Initialize profile if needed
        with self._lock:
            if profile_name not in self.profiles:
                self.profiles[profile_name] = FunctionProfile(
                    name=profile_name,
                    source_file=source_file,
                    line_number=line_number
                )

        # Get memory before (if enabled)
        memory_before = 0
        if self.enable_memory_profiling and track_memory:
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception

        # Record call stack
        call_stack = []
        if trace_calls:
            call_stack = [frame.function for frame in inspect.stack()[1:]]

        # Execute and time
        start_time = time.perf_counter()
        exception = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            exception = e
            raise
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Get memory after
            memory_after = 0
            if self.enable_memory_profiling and track_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_after = process.memory_info().rss
                except ImportError:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {__name__}", exc_info=True)
                    raise  # Re-raise the exception

            memory_delta = memory_after - memory_before

            # Update profile
            with self._lock:
                self.profiles[profile_name].update(
                    execution_time, memory_delta, exception is not None
                )

            # Store trace if requested
            if trace_calls and len(self.traces) < self.max_traces:
                trace = CallTrace(
                    function_name=profile_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=execution_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_delta=memory_delta,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    exception=exception,
                    call_stack=call_stack,
                    thread_id=threading.get_ident(),
                )
                self.traces.append(trace)

            # Auto-detect bottlenecks if enabled
            if self.auto_detect_bottlenecks:
                self._check_bottleneck(profile_name, execution_time)

    def _check_bottleneck(self, func_name: str, execution_time: float):
        """Check if function execution indicates a bottleneck"""
        with self._lock:
            if func_name not in self.profiles:
                return

            profile = self.profiles[func_name]

            # Consider it a bottleneck if:
            # 1. Average execution time > 1 second
            # 2. Function called frequently (> 100 times)
            # 3. High variance (max_time > 10 * avg_time)
            is_bottleneck = (
                profile.avg_time > 1.0 and profile.call_count > 100
            ) or (
                profile.max_time > 10 * profile.avg_time and profile.call_count > 50
            )

            if is_bottleneck:
                bottleneck_info = {
                    "function": func_name,
                    "avg_time": profile.avg_time,
                    "max_time": profile.max_time,
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "source": f"{profile.source_file}:{profile.line_number}",
                    "detected_at": datetime.now().isoformat(),
                }

                # Avoid duplicates
                if not any(b["function"] == func_name for b in self.bottlenecks):
                    self.bottlenecks.append(bottleneck_info)
                    logger.warning(f"Bottleneck detected: {func_name} (avg={profile.avg_time:.3f}s)")

    def get_profile(self, func_name: str) -> Optional[FunctionProfile]:
        """Get profile for a specific function"""
        return self.profiles.get(func_name)

    def get_all_profiles(self) -> Dict[str, FunctionProfile]:
        """Get all function profiles"""
        with self._lock:
            return dict(self.profiles)

    def get_bottlenecks(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N bottlenecks sorted by total time"""
        with self._lock:
            return sorted(
                self.bottlenecks,
                key=lambda b: b["total_time"],
                reverse=True
            )[:top_n]

    def get_hot_paths(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze traces to find hot execution paths"""
        with self._lock:
            path_counts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "count": 0,
                "total_time": 0.0,
                "functions": set(),
            })

            for trace in self.traces:
                # Create path signature from call stack
                path = " -> ".join(trace.call_stack[-5:])  # Last 5 frames
                path_counts[path]["count"] += 1
                path_counts[path]["total_time"] += trace.duration
                path_counts[path]["functions"].add(trace.function_name)

            # Convert to list and sort
            hot_paths = []
            for path, data in path_counts.items():
                hot_paths.append({
                    "path": path,
                    "count": data["count"],
                    "total_time": data["total_time"],
                    "avg_time": data["total_time"] / data["count"],
                    "functions": list(data["functions"]),
                })

            return sorted(hot_paths, key=lambda p: p["total_time"], reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall profiling statistics"""
        with self._lock:
            if not self.profiles:
                return {"message": "No profiling data available"}

            total_calls = sum(p.call_count for p in self.profiles.values())
            total_time = sum(p.total_time for p in self.profiles.values())
            total_memory = sum(p.memory_usage for p in self.profiles.values())

            # Find slowest functions
            sorted_by_avg = sorted(
                self.profiles.items(),
                key=lambda x: x[1].avg_time,
                reverse=True
            )

            return {
                "total_functions_profiled": len(self.profiles),
                "total_calls": total_calls,
                "total_execution_time": total_time,
                "total_memory_used": total_memory,
                "avg_execution_time": total_time / total_calls if total_calls > 0 else 0,
                "slowest_functions": [
                    {"name": name, "avg_time": p.avg_time, "call_count": p.call_count}
                    for name, p in sorted_by_avg[:10]
                ],
                "bottlenecks_detected": len(self.bottlenecks),
                "traces_collected": len(self.traces),
            }

    def set_baseline(self, func_name: str, metrics: Dict[str, float]):
        """Set performance baseline for a function"""
        self.baselines[func_name] = metrics
        logger.info(f"Set baseline for {func_name}: {metrics}")

    def check_regression(self, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Check for performance regressions compared to baselines.

        Args:
            threshold: Regression threshold (e.g., 0.2 = 20% slower)

        Returns:
            List of detected regressions
        """
        regressions = []

        with self._lock:
            for func_name, profile in self.profiles.items():
                if func_name not in self.baselines:
                    continue

                baseline = self.baselines[func_name]

                # Check average time regression
                if "avg_time" in baseline:
                    slowdown = (profile.avg_time - baseline["avg_time"]) / baseline["avg_time"]
                    if slowdown > threshold:
                        regressions.append({
                            "function": func_name,
                            "type": "avg_time",
                            "baseline": baseline["avg_time"],
                            "current": profile.avg_time,
                            "slowdown": slowdown,
                        })

                # Check memory regression
                if "memory_usage" in baseline and profile.call_count > 0:
                    avg_memory = profile.memory_usage / profile.call_count
                    baseline_memory = baseline.get("memory_usage", 0)
                    if baseline_memory > 0:
                        memory_increase = (avg_memory - baseline_memory) / baseline_memory
                        if memory_increase > threshold:
                            regressions.append({
                                "function": func_name,
                                "type": "memory",
                                "baseline": baseline_memory,
                                "current": avg_memory,
                                "increase": memory_increase,
                            })

        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")

        return regressions

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive profiling report.

        Args:
            output_file: Optional file path to save report

        Returns:
            Report as JSON string
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "profiles": {
                name: {
                    "call_count": p.call_count,
                    "total_time": p.total_time,
                    "avg_time": p.avg_time,
                    "min_time": p.min_time,
                    "max_time": p.max_time,
                    "memory_usage": p.memory_usage,
                    "error_count": p.error_count,
                    "source": f"{p.source_file}:{p.line_number}",
                }
                for name, p in self.profiles.items()
            },
            "bottlenecks": self.get_bottlenecks(),
            "hot_paths": self.get_hot_paths(),
            "regressions": self.check_regression(),
        }

        report_json = json.dumps(report, indent=2)

        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w') as f:
                f.write(report_json)
            logger.info(f"Profiling report saved to {output_path}")

        return report_json

    def reset(self):
        """Reset all profiling data"""
        with self._lock:
            self.profiles.clear()
            self.traces.clear()
            self.bottlenecks.clear()
        logger.info("Profiling data reset")

    def export_traces(self, output_file: str):
        """Export call traces to file"""
        with self._lock:
            traces_data = [trace.to_dict() for trace in self.traces]

        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(traces_data, f, indent=2)

        logger.info(f"Exported {len(traces_data)} traces to {output_path}")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_function(func: Optional[Callable] = None, **kwargs) -> Callable:
    """Convenient decorator to profile functions using global profiler"""
    profiler = get_profiler()
    return profiler.profile(func, **kwargs)


# Context manager for temporary profiling
class ProfileContext:
    """Context manager for profiling a block of code"""

    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        self.name = name
        self.profiler = profiler or get_profiler()
        self.start_time = None
        self.memory_before = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.profiler.enable_memory_profiling:
            try:
                import psutil
                process = psutil.Process()
                self.memory_before = process.memory_info().rss
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time

        memory_after = None
        if self.memory_before is not None:
            try:
                import psutil
                process = psutil.Process()
                memory_after = process.memory_info().rss
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception

        # Record as synthetic profile
        with self.profiler._lock:
            if self.name not in self.profiler.profiles:
                self.profiler.profiles[self.name] = FunctionProfile(name=self.name)

            memory_delta = (memory_after - self.memory_before) if (
                memory_after and self.memory_before
            ) else 0

            self.profiler.profiles[self.name].update(duration, memory_delta, exc_type is not None)

        return False


def profile_block(name: str):
    """Decorator/context manager for profiling code blocks"""
    return ProfileContext(name)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create profiler
    profiler = PerformanceProfiler()

    # Example 1: Profile a function
    @profiler.profile(name="example.fibonacci")
    def fibonacci(n: int) -> int:
        """Calculate fibonacci number (inefficiently for testing)"""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    # Example 2: Profile with decorator
    @profile_function
    def calculate_sum(n: int) -> int:
        """Calculate sum of numbers"""
        total = 0
        for i in range(n):
            total += i
            time.sleep(0.001)  # Simulate work
        return total

    # Example 3: Profile a code block
    with ProfileContext("example.block"):
        for i in range(5):
            fibonacci(10)
            time.sleep(0.01)

    # Run some work
    logger.info("Running profiled functions...")
    for i in range(3):
        fibonacci(15)
        calculate_sum(100)

    # Generate report
    report = profiler.generate_report("profiling_report.json")
    logger.info(f"Profiling report generated")

    # Get statistics
    stats = profiler.get_statistics()
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

    # Check bottlenecks
    bottlenecks = profiler.get_bottlenecks()
    logger.info(f"Bottlenecks: {json.dumps(bottlenecks, indent=2)}")
