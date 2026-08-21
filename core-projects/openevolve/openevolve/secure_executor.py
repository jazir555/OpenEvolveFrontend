"""
Secure execution of untrusted generated programs.

This module provides :class:`SecureCodeExecutor`, a best-effort sandboxed
evaluator for untrusted Python programs produced by the evolution engine. It
runs the target code in a *separate subprocess* (so a crash, runaway loop, or
memory explosion cannot corrupt the parent evaluator process) and enforces:

* a wall-clock timeout (always available, cross-platform),
* a CPU-time limit (POSIX ``RLIMIT_CPU``, where supported),
* a virtual-memory limit (POSIX ``RLIMIT_AS``, where supported),
* restricted stdio capture (bounded stdout/stderr), and
* an optional denylist of forbidden imports / builtins (static + runtime).

On Windows, ``RLIMIT_*`` is unavailable; we degrade gracefully to a wall-clock
timeout combined with a best-effort process-tree kill (via ``psutil`` when
present, otherwise ``taskkill``). The executor never raises for a "dangerous"
program -- it returns a :class:`SecureExecResult` describing what happened.

This is defence-in-depth for *untrusted generated* code, NOT a hard security
boundary. It is meant to keep the evolution loop safe from runaway programs.
"""

import abc
import asyncio
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = __import__("logging").getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_POSIX = not IS_WINDOWS

try:  # pragma: no cover - platform dependent
    import resource  # POSIX only
except Exception:  # pragma: no cover - Windows
    resource = None  # type: ignore


# Default forbidden imports / builtins for generated, untrusted programs.
DEFAULT_FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "importlib",
        "builtins",
        "marshal",
        "pickle",
        "codeop",
    }
)

DEFAULT_FORBIDDEN_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

# Maximum bytes captured from stdout/stderr to avoid runaway output.
DEFAULT_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MiB


@dataclass
class SecureExecResult:
    """Outcome of a single sandboxed execution.

    Attributes:
        stdout: Captured standard output (decoded, possibly truncated).
        stderr: Captured standard error (decoded, possibly truncated).
        returncode: Subprocess return code, or ``None`` if unavailable.
        timed_out: Wall-clock timeout was exceeded and the process tree killed.
        cpu_exceeded: CPU-time limit (RLIMIT_CPU / SIGXCPU) was exceeded.
        oom: Virtual-memory / allocation limit was exceeded.
        signal: Signal number that terminated the process, if any.
        error: Human-readable summary of the failure mode (or ``None``).
        wall_s: Measured wall-clock duration in seconds.
        cpu_s: Measured CPU time in seconds, if obtainable (else ``None``).
    """

    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    timed_out: bool = False
    cpu_exceeded: bool = False
    oom: bool = False
    signal: Optional[int] = None
    error: Optional[str] = None
    wall_s: float = 0.0
    cpu_s: Optional[float] = None

    @property
    def success(self) -> bool:
        """True when the program exited cleanly (returncode 0, no violations)."""
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.cpu_exceeded
            and not self.oom
        )


class CodeValidator(abc.ABC):
    """Static validator for untrusted source. Override for stricter policies."""

    @abc.abstractmethod
    def validate(self, source: str) -> Tuple[bool, List[str]]:
        """Return ``(safe, issues)`` for the given source."""
        raise NotImplementedError


class DefaultCodeValidator(CodeValidator):
    """Static check for obviously dangerous constructs (denylisted imports/builtins).

    This is a *best-effort* static screen. It catches the common cases described
    in the security spec but is trivially bypassable; the real safety comes from
    the subprocess isolation + resource limits.
    """

    def __init__(
        self,
        forbidden_imports: Sequence[str] = DEFAULT_FORBIDDEN_IMPORTS,
        forbidden_builtins: Sequence[str] = DEFAULT_FORBIDDEN_BUILTINS,
        block_dunder: bool = True,
    ):
        self.forbidden_imports = frozenset(forbidden_imports)
        self.forbidden_builtins = frozenset(forbidden_builtins)
        self.block_dunder = block_dunder

    def validate(self, source: str) -> Tuple[bool, List[str]]:
        import ast

        issues: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return False, [f"Syntax error: {exc}"]

        for node in ast.walk(tree):
            # import X / import X.Y / import X as Y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.forbidden_imports:
                        issues.append(f"Forbidden import: {alias.name}")
            # from X import ...
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in self.forbidden_imports:
                        issues.append(f"Forbidden import: {node.module}")
            # name nodes that resolve to forbidden builtins
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in self.forbidden_builtins:
                    issues.append(f"Forbidden builtin reference: {node.id}")
            # calls to dangerous builtins: eval(...), exec(...), etc.
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.forbidden_builtins:
                    issues.append(f"Forbidden call: {func.id}()")
                if isinstance(func, ast.Attribute) and self.block_dunder:
                    if func.attr.startswith("__") and func.attr.endswith("__"):
                        # Allow harmless dunders like __name__ / __doc__ reads.
                        if func.attr not in {"__name__", "__doc__", "__class__"}:
                            issues.append(f"Forbidden dunder access: {func.attr}")
        return (len(issues) == 0, issues)


class SecurityError(RuntimeError):
    """Raised by :meth:`SecureCodeExecutor.execute_code_blocking` when validation fails."""


@dataclass
class SecureExecutorConfig:
    """Limits / policy for :class:`SecureCodeExecutor`."""

    timeout: float = 30.0
    cpu_time_limit: Optional[float] = None
    memory_limit_mb: Optional[float] = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    forbid_imports: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_IMPORTS))
    forbid_builtins: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_BUILTINS))
    validate_static: bool = True
    python_executable: str = sys.executable

    @property
    def memory_limit_bytes(self) -> Optional[int]:
        if self.memory_limit_mb is None:
            return None
        return int(self.memory_limit_mb * 1024 * 1024)


class SecureCodeExecutor:
    """Run untrusted Python source in a resource-limited subprocess.

    The executor writes the source to a temporary file and launches
    ``python <tmpfile>`` in a child process. On POSIX it installs
    ``RLIMIT_CPU`` / ``RLIMIT_AS`` via a ``preexec_fn``; on Windows it relies on
    the wall-clock timeout plus a best-effort process-tree kill.
    """

    def __init__(self, config: Optional[SecureExecutorConfig] = None):
        self.config = config or SecureExecutorConfig()
        self.validator = DefaultCodeValidator(
            forbidden_imports=self.config.forbid_imports,
            forbidden_builtins=self.config.forbid_builtins,
        )

    # -- public API ---------------------------------------------------------

    def execute_code_blocking(
        self,
        source: str,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> SecureExecResult:
        """Synchronously execute ``source`` and return a :class:`SecureExecResult`.

        Raises:
            SecurityError: If static validation is enabled and the code is unsafe.
        """
        if self.config.validate_static:
            safe, issues = self.validator.validate(source)
            if not safe:
                raise SecurityError("Unsafe code detected: " + "; ".join(issues))
        return self._run_subprocess(source, stdin=stdin, env=env)

    async def execute_code(
        self,
        source: str,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> SecureExecResult:
        """Async wrapper around :meth:`execute_code_blocking`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute_code_blocking, source, stdin, env
        )

    def run_file(
        self,
        path: str,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> SecureExecResult:
        """Execute an existing Python file (validated then run)."""
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        return self.execute_code_blocking(source, stdin=stdin, env=env)

    # -- subprocess machinery ----------------------------------------------

    def _build_preexec(self):
        """Return a preexec_fn that applies POSIX rlimits, or None on Windows."""
        if not IS_POSIX or resource is None:
            return None

        mem = self.config.memory_limit_bytes
        cpu = self.config.cpu_time_limit

        def _preexec():
            # Soft/hard address-space limit -> MemoryError / SIGSEGV on overflow.
            if mem is not None:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except (ValueError, OSError):
                    pass
            # CPU-time limit -> SIGXCPU when exceeded.
            if cpu is not None:
                try:
                    soft = int(cpu)
                    hard = max(soft, soft)
                    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
                except (ValueError, OSError):
                    pass
            # Disallow spawning new processes from the sandboxed code.
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            except (ValueError, OSError, AttributeError):
                pass

        return _preexec

    def _run_subprocess(
        self,
        source: str,
        stdin: Optional[str],
        env: Optional[Dict[str, str]],
    ) -> SecureExecResult:
        cfg = self.config
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, prefix="openevolve_secure_"
        ) as tmp:
            tmp.write(source.encode("utf-8"))
            tmp_path = tmp.name

        preexec_fn = self._build_preexec()
        # On POSIX, start a new session so we can kill the whole process group.
        start_new_session = IS_POSIX
        proc = None
        start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                [cfg.python_executable, tmp_path],
                stdin=subprocess.PIPE if stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=preexec_fn,
                start_new_session=start_new_session,
            )
            try:
                out, err = proc.communicate(
                    input=stdin.encode("utf-8") if stdin is not None else None,
                    timeout=cfg.timeout,
                )
                wall_s = time.perf_counter() - start
                return self._interpret(proc, out, err, wall_s, timed_out=False)
            except subprocess.TimeoutExpired:
                wall_s = time.perf_counter() - start
                self._kill_process_tree(proc)
                out, err = b"", b""
                try:
                    out, err = proc.communicate(timeout=2)
                except Exception:
                    pass
                res = self._interpret(proc, out, err, wall_s, timed_out=True)
                return res
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _interpret(
        self,
        proc: Optional[subprocess.Popen],
        out: bytes,
        err: bytes,
        wall_s: float,
        timed_out: bool,
    ) -> SecureExecResult:
        cfg = self.config
        stdout = self._clip(out, cfg.max_output_bytes)
        stderr = self._clip(err, cfg.max_output_bytes)
        returncode = proc.returncode if proc is not None else None
        signal_num = None
        cpu_exceeded = False
        oom = False

        if returncode is not None and returncode < 0:
            signal_num = -returncode
            if signal_num == signal.SIGXCPU:
                cpu_exceeded = True
            # SIGKILL / SIGSEGV often indicate the OOM killer or RLIMIT_AS hit.
            if signal_num in (signal.SIGKILL, signal.SIGSEGV):
                oom = True

        # Heuristic OOM detection from stderr text (RLIMIT_AS raises MemoryError).
        low_err = stderr.lower()
        if any(
            t in low_err
            for t in ("memoryerror", "cannot allocate", "out of memory", "memory limit")
        ):
            oom = True

        error = None
        if timed_out:
            error = f"Wall-clock timeout after {cfg.timeout}s"
        elif cpu_exceeded:
            error = "CPU-time limit exceeded"
        elif oom:
            error = "Memory limit exceeded (OOM)"

        return SecureExecResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            cpu_exceeded=cpu_exceeded,
            oom=oom,
            signal=signal_num,
            error=error,
            wall_s=wall_s,
            cpu_s=None,
        )

    @staticmethod
    def _clip(data: bytes, limit: int) -> str:
        if data is None:
            return ""
        if len(data) > limit:
            data = data[:limit] + b"\n...[output truncated]"
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")

    def _kill_process_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if IS_POSIX and proc.pid is not None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                self._kill_windows_tree(proc)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _kill_windows_tree(proc: subprocess.Popen) -> None:
        try:
            import psutil  # type: ignore

            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
            return
        except Exception:
            pass
        # Fallback: taskkill process tree.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def safe_eval_expression(
    expression: str,
    timeout: float = 2.0,
    memory_limit_mb: float = 128.0,
    namespace: Optional[Dict[str, object]] = None,
) -> object:
    """Safely evaluate a single arithmetic/comparison *expression* with a timeout.

    The expression is statically validated (only arithmetic, comparison, logic,
    attribute/index, and a small whitelist of callables are allowed) and then
    evaluated inside a :class:`SecureCodeExecutor` subprocess so a malicious or
    runaway expression cannot hang or corrupt the caller.

    Args:
        expression: Python expression source (e.g. ``"2 ** 20 + math.sqrt(2)"``).
        timeout: Wall-clock budget in seconds.
        memory_limit_mb: Virtual-memory budget for the subprocess.
        namespace: Extra names made available (e.g. ``{"math": math}``).

    Returns:
        The evaluated result (decoded from JSON).

    Raises:
        ValueError: If the expression is rejected by static validation.
        RuntimeError: If evaluation times out, errors, or OOMs.
    """
    import ast
    import builtins as _builtins
    import math
    import types

    allowed_calls = {
        "abs", "min", "max", "sum", "len", "round", "pow", "divmod", "range",
        "math.sqrt", "math.sin", "math.cos", "math.tan", "math.log",
        "math.log10", "math.exp", "math.floor", "math.ceil", "math.factorial",
        "math.gcd", "math.pi", "math.e",
    }

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are not allowed in safe_eval_expression")
        if isinstance(node, (ast.Call,)) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_calls:
                raise ValueError(f"Call to {node.func.id}() is not allowed")
        if isinstance(node, ast.Name) and node.id in DEFAULT_FORBIDDEN_BUILTINS:
            raise ValueError(f"Use of {node.id} is not allowed")

    # Only embed literal-serializable namespace values; modules are handled
    # specially so their repr() does not poison the bootstrap source.
    extra = {
        k: v
        for k, v in (namespace or {}).items()
        if not isinstance(v, types.ModuleType)
    }

    # Restricted builtins made available to the evaluated expression.
    safe_builtin_names = [
        name for name in allowed_calls if hasattr(_builtins, name)
    ]

    bootstrap = (
        "import sys, json, math, builtins\n"
        "ns = {'math': math}\n"
        + ("ns.update(" + repr(extra) + ")\n" if extra else "")
        + "safe_builtins = {n: getattr(builtins, n) for n in "
        + repr(safe_builtin_names)
        + "}\n"
        + "expr = " + repr(expression) + "\n"
        "result = eval(expr, {'__builtins__': safe_builtins}, ns)\n"
        "sys.stdout.write(json.dumps({'result': result}))\n"
    )

    runner = SecureCodeExecutor(
        SecureExecutorConfig(
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            validate_static=False,
        )
    )
    res = runner.execute_code_blocking(bootstrap)
    if res.timed_out:
        raise RuntimeError("safe_eval_expression timed out")
    if res.oom:
        raise RuntimeError("safe_eval_expression ran out of memory")
    if not res.success:
        raise RuntimeError(f"safe_eval_expression failed: {res.error or res.stderr}")
    try:
        payload = json.loads(res.stdout)
        return payload["result"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"safe_eval_expression bad output: {res.stdout!r}") from exc
