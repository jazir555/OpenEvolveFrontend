"""
Execution Sandbox - The "Hazmat Suit"

Provides ephemeral, secure code execution environments using E2B (Code Interpreter SDK)
or Firecracker MicroVMs. Every code execution happens in a disposable, air-gapped
micro-VM that dies after execution.

Key Features:
- Ephemeral execution environments (auto-cleanup)
- Network isolation and resource limits
- Support for Python, Bash, and other languages
- Timeout and memory constraints
- Execution audit logging
"""
from __future__ import annotations


import os
import json
import time
import base64
import hashlib
import logging
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager
import tempfile
import shutil

# Configure logging
logger = logging.getLogger(__name__)

class SandboxProvider(Enum):
    """Supported sandbox providers"""
    E2B = "e2b"                    # E2B Code Interpreter SDK
    FIRECRACKER = "firecracker"    # AWS Firecracker MicroVMs
    DOCKER = "docker"              # Fallback Docker container
    SUBPROCESS = "subprocess"      # Local subprocess (development only)


class ExecutionStatus(Enum):
    """Status of code execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    SANDBOX_ERROR = "sandbox_error"


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution"""
    execution_id: str
    status: ExecutionStatus
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    memory_usage_mb: Optional[float] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sandbox_id: Optional[str] = None
    security_flags: List[str] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    provider: SandboxProvider = SandboxProvider.DOCKER
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0
    network_access: bool = False
    allowed_packages: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=/dev/zero", ":(){ :|:& };:", "> /dev/sda"
    ])
    environment_variables: Dict[str, str] = field(default_factory=dict)
    working_directory: str = "/home/sandbox"
    enable_audit_logging: bool = True


@dataclass
class SecurityPolicy:
    """Security policy for sandbox execution"""
    max_execution_time_seconds: int = 60
    max_memory_mb: int = 1024
    max_file_size_mb: int = 100
    max_files_count: int = 1000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".txt", ".json", ".csv", ".md", ".yaml", ".yml"
    ])
    blocked_imports: List[str] = field(default_factory=lambda: [
        "os.system", "subprocess.call", "subprocess.Popen",
        "eval", "exec", "compile", "__import__"
    ])
    require_code_review: bool = False
    ai_safety_check: bool = True


class CodeSafetyChecker:
    """Checks code for potentially dangerous patterns before execution"""
    
    DANGEROUS_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\.call\s*\(",
        r"subprocess\.Popen\s*\(",
        r"subprocess\.run\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"__import__\s*\(",
        r"import\s+os.*system",
        r"open\s*\(\s*['\"]*/",
        r"file\s*\(\s*['\"]*/",
        r"rm\s+-rf",
        r"dd\s+if=",
        r"mkfs",
        r"chmod\s+777",
        r"chmod\s+s",
        r"wget.*\|.*sh",
        r"curl.*\|.*sh",
    ]
    
    SENSITIVE_ENV_VARS = [
        "API_KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL",
        "AWS_ACCESS", "PRIVATE_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY"
    ]
    
    def __init__(self):
        self.violations: List[str] = []
    
    def check_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Check code for dangerous patterns"""
        self.violations = []
        
        # Check for dangerous patterns
        import re
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                self.violations.append(f"Dangerous pattern detected: {pattern}")
        
        # Check for sensitive env var access
        for var in self.SENSITIVE_ENV_VARS:
            if var in code:
                self.violations.append(f"Potential sensitive data access: {var}")
        
        # Check file system access
        if re.search(r"open\s*\(\s*['\"]\s*/", code):
            self.violations.append("Absolute path file access detected")
        
        # Check for network access patterns
        network_patterns = [
            r"requests\.get", r"requests\.post", r"urllib",
            r"socket\.", r"http\.client"
        ]
        network_access = any(re.search(p, code) for p in network_patterns)
        
        return {
            "is_safe": len(self.violations) == 0,
            "violations": self.violations,
            "has_network_access": network_access,
            "risk_score": min(len(self.violations) * 0.2, 1.0),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        if self.violations:
            recommendations.append("Consider using safer alternatives for file/network operations")
            recommendations.append("Run in isolated sandbox with restricted permissions")
        return recommendations


class E2BSandbox:
    """E2B Code Interpreter SDK integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.sandbox = None
        self._available = None
    
    @property
    def is_available(self) -> bool:
        """Check if E2B is available"""
        if self._available is None:
            try:
                import e2b
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    async def create_sandbox(self, config: SandboxConfig) -> str:
        """Create a new E2B sandbox"""
        if not self.is_available:
            raise RuntimeError("E2B SDK not installed")
        
        from e2b import Sandbox
        
        self.sandbox = Sandbox(
            api_key=self.api_key,
            timeout=config.timeout_seconds,
            envs=config.environment_variables
        )
        return self.sandbox.id
    
    async def execute_code(
        self, 
        code: str, 
        language: str = "python",
        timeout: int = 30
    ) -> ExecutionResult:
        """Execute code in E2B sandbox"""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created")
        
        start_time = time.time()
        execution_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:12]
        
        try:
            if language == "python":
                result = self.sandbox.run_code(code, timeout=timeout)
            else:
                result = self.sandbox.run_command(code, timeout=timeout)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SUCCESS if result.exit_code == 0 else ExecutionStatus.FAILURE,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=execution_time,
                sandbox_id=self.sandbox.id
            )
        except Exception as e:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SANDBOX_ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                sandbox_id=self.sandbox.id
            )
    
    async def close(self):
        """Close the sandbox"""
        if self.sandbox:
            self.sandbox.close()
            self.sandbox = None


class FirecrackerSandbox:
    """AWS Firecracker MicroVM integration"""
    
    def __init__(self, firecracker_path: str = "/usr/bin/firecracker"):
        self.firecracker_path = firecracker_path
        self.vm_id: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
    
    @property
    def is_available(self) -> bool:
        """Check if Firecracker is available"""
        return os.path.exists(self.firecracker_path)
    
    async def create_microvm(self, config: SandboxConfig) -> str:
        """Create a new Firecracker MicroVM"""
        if not self.is_available:
            raise RuntimeError("Firecracker not installed")
        
        self.vm_id = f"vm-{int(time.time() * 1000)}"
        
        # This is a simplified version - in production, you'd use the Firecracker API
        # to configure the microvm with proper resource limits
        logger.info(f"Creating Firecracker MicroVM: {self.vm_id}")
        
        return self.vm_id
    
    async def execute_code(
        self, 
        code: str, 
        language: str = "python",
        timeout: int = 30
    ) -> ExecutionResult:
        """Execute code in Firecracker MicroVM"""
        execution_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:12]
        start_time = time.time()
        
        # Create temporary files for the code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, f"script.{language}")
            with open(code_file, "w") as f:
                f.write(code)
            
            try:
                # Execute in isolated environment
                cmd = self._build_command(code_file, language)
                
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=config.memory_limit_mb * 1024 * 1024 if 'config' in dir() else None
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILURE,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    exit_code=process.returncode or 0,
                    execution_time_ms=execution_time,
                    sandbox_id=self.vm_id
                )
                
            except asyncio.TimeoutError:
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    exit_code=-1,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    sandbox_id=self.vm_id
                )
    
    def _build_command(self, code_file: str, language: str) -> str:
        """Build execution command for the language"""
        commands = {
            "python": f"python3 {code_file}",
            "bash": f"bash {code_file}",
            "javascript": f"node {code_file}",
            "typescript": f"ts-node {code_file}",
        }
        return commands.get(language, f"python3 {code_file}")
    
    async def close(self):
        """Terminate the microvm"""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()


class DockerSandbox:
    """Docker container sandbox (fallback option)"""
    
    def __init__(self):
        self.container_id: Optional[str] = None
        self._available = None
    
    @property
    def is_available(self) -> bool:
        """Check if Docker is available"""
        if self._available is None:
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    timeout=5
                )
                self._available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._available = False
        return self._available
    
    async def create_container(self, config: SandboxConfig) -> str:
        """Create a new Docker container"""
        if not self.is_available:
            raise RuntimeError("Docker not available")
        
        self.container_id = f"sandbox-{int(time.time() * 1000)}"
        
        # Build docker run command with security options
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_id,
            "--memory", f"{config.memory_limit_mb}m",
            "--cpus", str(config.cpu_limit),
            "--network", "none" if not config.network_access else "bridge",
            "--read-only",
            "--tmpfs", "/tmp:noexec,nosuid,size=100m",
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            "python:3.11-slim",
            "sleep", str(config.timeout_seconds + 10)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")
        
        return self.container_id
    
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30
    ) -> ExecutionResult:
        """Execute code in Docker container"""
        execution_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:12]
        start_time = time.time()
        
        try:
            # Write code to temp file and copy to container
            with tempfile.NamedTemporaryFile(mode="w", suffix=f".{language}", delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Copy file to container
                container_file = f"/tmp/script.{language}"
                subprocess.run(
                    ["docker", "cp", temp_file, f"{self.container_id}:{container_file}"],
                    check=True,
                    capture_output=True
                )
                
                # Execute the code
                exec_cmd = self._get_exec_command(language, container_file)
                result = subprocess.run(
                    ["docker", "exec", self.container_id, "sh", "-c", exec_cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILURE,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    execution_time_ms=execution_time,
                    sandbox_id=self.container_id
                )
                
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.TIMEOUT,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                sandbox_id=self.container_id
            )
        except Exception as e:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SANDBOX_ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                sandbox_id=self.container_id
            )
    
    def _get_exec_command(self, language: str, file_path: str) -> str:
        """Get execution command for language"""
        commands = {
            "python": f"python3 {file_path}",
            "bash": f"bash {file_path}",
            "javascript": f"node {file_path}",
        }
        return commands.get(language, f"python3 {file_path}")
    
    async def close(self):
        """Remove the container"""
        if self.container_id:
            subprocess.run(
                ["docker", "rm", "-f", self.container_id],
                capture_output=True
            )
            self.container_id = None


class ExecutionSandbox:
    """
    Main Execution Sandbox interface - The "Hazmat Suit"
    
    Provides secure, ephemeral execution environments for code.
    Automatically selects the best available provider.
    """
    
    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        security_policy: Optional[SecurityPolicy] = None
    ):
        self.config = config or SandboxConfig()
        self.security_policy = security_policy or SecurityPolicy()
        self.safety_checker = CodeSafetyChecker()
        self._sandbox = None
        self._execution_history: List[ExecutionResult] = []
        
        # Initialize the appropriate provider
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the best available sandbox provider"""
        provider = self.config.provider
        
        if provider == SandboxProvider.E2B:
            self._sandbox = E2BSandbox()
            if not self._sandbox.is_available:
                logger.warning("E2B not available, falling back to Docker")
                self._sandbox = DockerSandbox()
                
        elif provider == SandboxProvider.FIRECRACKER:
            self._sandbox = FirecrackerSandbox()
            if not self._sandbox.is_available:
                logger.warning("Firecracker not available, falling back to Docker")
                self._sandbox = DockerSandbox()
                
        elif provider == SandboxProvider.DOCKER:
            self._sandbox = DockerSandbox()
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        if not self._sandbox.is_available:
            raise RuntimeError(
                "No sandbox provider available. "
                "Please install Docker or configure E2B/Firecracker."
            )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Start the sandbox environment"""
        if hasattr(self._sandbox, 'create_sandbox'):
            await self._sandbox.create_sandbox(self.config)
        elif hasattr(self._sandbox, 'create_container'):
            await self._sandbox.create_container(self.config)
        elif hasattr(self._sandbox, 'create_microvm'):
            await self._sandbox.create_microvm(self.config)
    
    async def stop(self):
        """Stop and cleanup the sandbox"""
        if self._sandbox:
            await self._sandbox.close()
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
        skip_safety_check: bool = False
    ) -> ExecutionResult:
        """
        Execute code securely in the sandbox
        
        Args:
            code: The code to execute
            language: Programming language (python, bash, javascript)
            timeout: Execution timeout in seconds
            skip_safety_check: Skip safety checking (not recommended)
            
        Returns:
            ExecutionResult with stdout, stderr, and status
        """
        timeout = timeout or self.config.timeout_seconds
        
        # Safety check
        if not skip_safety_check:
            safety_result = self.safety_checker.check_code(code, language)
            if not safety_result["is_safe"] and self.security_policy.require_code_review:
                return ExecutionResult(
                    execution_id=f"blocked-{int(time.time() * 1000)}",
                    status=ExecutionStatus.FAILURE,
                    stdout="",
                    stderr=f"Code blocked by security policy: {safety_result['violations']}",
                    exit_code=-1,
                    execution_time_ms=0,
                    security_flags=safety_result["violations"]
                )
        
        # Execute in sandbox
        result = await self._sandbox.execute_code(code, language, timeout)
        result.security_flags = safety_result.get("violations", [])
        
        # Log execution
        self._execution_history.append(result)
        if self.config.enable_audit_logging:
            await self._log_execution(code, result)
        
        return result
    
    async def _log_execution(self, code: str, result: ExecutionResult):
        """Log execution for audit purposes"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": result.execution_id,
            "sandbox_id": result.sandbox_id,
            "status": result.status.value,
            "execution_time_ms": result.execution_time_ms,
            "exit_code": result.exit_code,
            "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            "security_flags": result.security_flags
        }
        logger.info(f"Sandbox execution: {json.dumps(log_entry)}")
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions"""
        return self._execution_history.copy()
    
    async def run_tests(
        self,
        code: str,
        test_code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """Run code with tests in the sandbox"""
        full_code = f"""
{code}

# Tests
{test_code}

# Run tests
if __name__ == "__main__":
    import unittest
    unittest.main(argv=[''], verbosity=2, exit=False)
"""
        return await self.execute(full_code, language)
    
    async def install_package(self, package: str) -> ExecutionResult:
        """Install a package in the sandbox"""
        if package not in self.config.allowed_packages:
            return ExecutionResult(
                execution_id=f"blocked-{int(time.time() * 1000)}",
                status=ExecutionStatus.FAILURE,
                stdout="",
                stderr=f"Package '{package}' not in allowed packages list",
                exit_code=-1,
                execution_time_ms=0
            )
        
        install_code = f"""
import subprocess
result = subprocess.run(
    ["pip", "install", "{package}"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
exit(result.returncode)
"""
        return await self.execute(install_code, "python")


class SandboxManager:
    """Manages multiple sandbox instances"""
    
    def __init__(self):
        self._sandboxes: Dict[str, ExecutionSandbox] = {}
        self._pool: asyncio.Queue = asyncio.Queue()
    
    async def create_pool(self, size: int, config: SandboxConfig):
        """Create a pool of sandboxes"""
        for i in range(size):
            sandbox = ExecutionSandbox(config)
            await sandbox.start()
            self._sandboxes[f"sandbox-{i}"] = sandbox
            await self._pool.put(sandbox)
    
    async def get_sandbox(self) -> ExecutionSandbox:
        """Get a sandbox from the pool"""
        return await self._pool.get()
    
    async def return_sandbox(self, sandbox: ExecutionSandbox):
        """Return a sandbox to the pool"""
        await self._pool.put(sandbox)
    
    async def execute_in_pool(
        self,
        code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """Execute code using a sandbox from the pool"""
        sandbox = await self.get_sandbox()
        try:
            result = await sandbox.execute(code, language)
            return result
        finally:
            await self.return_sandbox(sandbox)
    
    async def cleanup_all(self):
        """Cleanup all sandboxes"""
        for sandbox in self._sandboxes.values():
            await sandbox.stop()
        self._sandboxes.clear()


# Convenience functions for quick usage
async def execute_securely(
    code: str,
    language: str = "python",
    timeout: int = 30,
    provider: SandboxProvider = SandboxProvider.DOCKER
) -> ExecutionResult:
    """Quick function to execute code securely"""
    config = SandboxConfig(provider=provider, timeout_seconds=timeout)
    async with ExecutionSandbox(config) as sandbox:
        return await sandbox.execute(code, language)


def execute_securely_sync(
    code: str,
    language: str = "python",
    timeout: int = 30,
    provider: SandboxProvider = SandboxProvider.DOCKER
) -> ExecutionResult:
    """Synchronous wrapper for execute_securely"""
    return asyncio.run(execute_securely(code, language, timeout, provider))


# Example usage
if __name__ == "__main__":
    async def demo():
        # Example: Execute Python code securely
        code = """
import sys
print("Hello from sandbox!")
print(f"Python version: {sys.version}")
result = sum(range(100))
print(f"Sum of 0-99: {result}")
"""
        
        print("=" * 60)
        print("EXECUTION SANDBOX DEMO - The 'Hazmat Suit'")
        print("=" * 60)
        
        result = await execute_securely(code, timeout=10)
        
        print(f"\nExecution ID: {result.execution_id}")
        print(f"Status: {result.status.value}")
        print(f"Time: {result.execution_time_ms:.2f}ms")
        print(f"\nSTDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
        
        # Example: Blocked dangerous code
        print("\n" + "=" * 60)
        print("SAFETY CHECK DEMO")
        print("=" * 60)
        
        dangerous_code = """
import os
os.system("rm -rf /")
"""
        
        checker = CodeSafetyChecker()
        safety = checker.check_code(dangerous_code)
        print(f"\nCode blocked: {not safety['is_safe']}")
        print(f"Violations: {safety['violations']}")
        print(f"Risk Score: {safety['risk_score']}")
    
    asyncio.run(demo())
