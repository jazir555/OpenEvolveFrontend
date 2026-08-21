"""
Offline unit tests for SecureCodeExecutor.

All tests use small limits and run in well under a second. No network access.
"""

import sys
import time

import pytest

from openevolve.secure_executor import (
    SecureCodeExecutor,
    SecureExecutorConfig,
    SecurityError,
    safe_eval_expression,
)


def make_executor(timeout=1.0, memory_mb=128.0, cpu=0.5, validate_static=False):
    return SecureCodeExecutor(
        SecureExecutorConfig(
            timeout=timeout,
            memory_limit_mb=memory_mb,
            cpu_time_limit=cpu,
            validate_static=validate_static,
        )
    )


def test_safe_program_succeeds():
    res = make_executor().execute_code_blocking("print('hello'); print(' world')")
    assert res.success
    assert res.returncode == 0
    assert "hello" in res.stdout and "world" in res.stdout
    assert not res.timed_out
    assert not res.oom


def test_sleep_exceeds_timeout_is_killed():
    # Sleep far longer than the timeout -> wall-clock kill.
    start = time.perf_counter()
    res = make_executor(timeout=0.4).execute_code_blocking("import time; time.sleep(5)")
    elapsed = time.perf_counter() - start
    assert res.timed_out
    assert res.returncode is not None and res.returncode != 0
    # The wall-clock kill should fire well before the 5s sleep completes.
    assert elapsed < 3.0


def test_infinite_loop_is_handled():
    # Tight infinite loop should be killed by either the wall-clock timeout
    # or the CPU-time limit (POSIX RLIMIT_CPU / SIGXCPU).
    res = make_executor(timeout=0.6, cpu=0.3).execute_code_blocking("while True:\n    pass")
    assert not res.success
    assert res.timed_out or res.cpu_exceeded or res.signal is not None
    assert res.returncode != 0


def test_oom_is_handled():
    # Try to allocate a giant list; with a small address-space limit this
    # should be caught as an OOM (heuristic on POSIX, timeout fallback on Win).
    code = "a=[]\nwhile True:\n    a.append('x'*10_000_000)"
    res = make_executor(timeout=2.0, memory_mb=64.0).execute_code_blocking(code)
    assert not res.success
    # On POSIX we expect a clean OOM/MemoryError; on Windows the timeout
    # is the safety net, so accept either.
    assert res.oom or res.timed_out or res.signal is not None


def test_forbidden_builtin_blocked_by_validation():
    exec_ = make_executor(validate_static=True)
    with pytest.raises(SecurityError):
        exec_.execute_code_blocking("exec('print(1)')")


def test_forbidden_import_blocked_by_validation():
    exec_ = make_executor(validate_static=True)
    with pytest.raises(SecurityError):
        exec_.execute_code_blocking("import os")


def test_async_execute_code_works():
    import asyncio

    res = asyncio.run(make_executor().execute_code("print('async-ok')"))
    assert res.success
    assert "async-ok" in res.stdout


def test_safe_eval_expression_basic():
    assert safe_eval_expression("2 + 3 * 4") == 14
    assert safe_eval_expression("abs(-7)") == 7
    assert abs(safe_eval_expression("math.sqrt(16)", namespace={"math": __import__("math")}) - 4.0) < 1e-9


def test_safe_eval_expression_rejects_imports():
    with pytest.raises(ValueError):
        safe_eval_expression("__import__('os')")


def test_safe_eval_expression_timeout():
    # A runaway expression inside safe_eval_expression must time out.
    with pytest.raises(RuntimeError):
        safe_eval_expression("sum(i for i in range(10**9))", timeout=0.4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
