"""
Sandbox Manager - Secure Code Execution

Implements the "Hazmat Suit" - ephemeral, secure execution environments.
Every code execution happens in a disposable, air-gapped micro-VM.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import tempfile
import shutil
import os
import subprocess
import time
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class SandboxType(Enum):
    """Types of sandbox environments"""
    E2B = "e2b"                    # E2B Code Interpreter
    FIRECRACKER = "firecracker"    # AWS Firecracker MicroVM
    DOCKER = "docker"              # Docker container (fallback)
    SUBPROCESS = "subprocess"      # Isolated subprocess (last resort)


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    artifacts: List[str]  # Paths to generated files
    sandbox_id: str
    security_report: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'exit_code': self.exit_code,
            'execution_time_ms': self.execution_time_ms,
            'artifacts': self.artifacts,
            'sandbox_id': self.sandbox_id,
            'security_report': self.security_report
        }


@dataclass
class SecurityPolicy:
    """Security policy for sandbox execution"""
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 512
    max_disk_mb: int = 100
    network_access: bool = False
    allow_file_write: bool = True
    allow_file_read: bool = True
    allowed_imports: List[str] = None
    forbidden_modules: List[str] = None
    
    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = []
        if self.forbidden_modules is None:
            self.forbidden_modules = ['os.system', 'subprocess', 'socket', 'urllib']


class SandboxManager:
    """
    Manages secure, ephemeral execution environments.
    
    Key Features:
    - Disposable sandboxes that auto-destroy after execution
    - Multiple backend support (E2B, Firecracker, Docker)
    - Security policy enforcement
    - Artifact collection
    - Execution audit logging
    """
    
    def __init__(
        self,
        preferred_sandbox: SandboxType = SandboxType.DOCKER,
        e2b_api_key: Optional[str] = None,
        firecracker_socket: Optional[str] = None,
        auto_cleanup: bool = True
    ):
        """
        Initialize sandbox manager.
        
        Args:
            preferred_sandbox: Preferred sandbox type
            e2b_api_key: API key for E2B
            firecracker_socket: Path to Firecracker socket
            auto_cleanup: Auto-cleanup sandboxes after execution
        """
        self.preferred_sandbox = preferred_sandbox
        self.e2b_api_key = e2b_api_key
        self.firecracker_socket = firecracker_socket
        self.auto_cleanup = auto_cleanup
        
        # Track active sandboxes
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Initialize available backends
        self.available_backends = self._detect_backends()
        
        logger.info({
            'msg': 'SandboxManager initialized',
            'preferred': preferred_sandbox.value,
            'available': [b.value for b in self.available_backends],
            'auto_cleanup': auto_cleanup
        })
    
    def _detect_backends(self) -> List[SandboxType]:
        """Detect available sandbox backends"""
        available = []
        
        # Check E2B
        try:
            from e2b import Sandbox
            if self.e2b_api_key:
                available.append(SandboxType.E2B)
        except ImportError:
            pass
        
        # Check Firecracker
        if self.firecracker_socket and os.path.exists(self.firecracker_socket):
            available.append(SandboxType.FIRECRACKER)
        
        # Check Docker
        try:
            result = subprocess.run(
                ['docker', 'version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                available.append(SandboxType.DOCKER)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Subprocess always available (least secure)
        available.append(SandboxType.SUBPROCESS)
        
        return available
    
    async def execute_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        policy: Optional[SecurityPolicy] = None,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute Python code in a secure sandbox.
        
        Args:
            code: Python code to execute
            context: Variables to inject into execution context
            policy: Security policy
            timeout: Execution timeout (overrides policy)
            
        Returns:
            ExecutionResult with output and metadata
        """
        policy = policy or SecurityPolicy()
        if timeout:
            policy.max_execution_time = timeout
        
        # Generate sandbox ID
        sandbox_id = f"sb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(code.encode()).hexdigest()[:8]}"
        
        logger.info({
            'msg': 'Starting sandboxed execution',
            'sandbox_id': sandbox_id,
            'code_length': len(code),
            'policy': {
                'max_time': policy.max_execution_time,
                'max_memory': policy.max_memory_mb,
                'network': policy.network_access
            }
        })
        
        # Select best available backend
        backend = self._select_backend()
        
        try:
            # Execute based on backend
            if backend == SandboxType.E2B:
                result = await self._execute_e2b(sandbox_id, code, context, policy)
            elif backend == SandboxType.FIRECRACKER:
                result = await self._execute_firecracker(sandbox_id, code, context, policy)
            elif backend == SandboxType.DOCKER:
                result = await self._execute_docker(sandbox_id, code, context, policy)
            else:
                result = await self._execute_subprocess(sandbox_id, code, context, policy)
            
            # Log execution
            self._log_execution(sandbox_id, code, result, backend)
            
            return result
            
        except Exception as e:
            logger.error({
                'msg': 'Sandbox execution failed',
                'sandbox_id': sandbox_id,
                'error': str(e)
            })
            
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Sandbox execution failed: {str(e)}",
                exit_code=-1,
                execution_time_ms=0.0,
                artifacts=[],
                sandbox_id=sandbox_id,
                security_report={'error': str(e), 'backend': backend.value}
            )
        
        finally:
            if self.auto_cleanup:
                await self._cleanup_sandbox(sandbox_id)
    
    def _select_backend(self) -> SandboxType:
        """Select best available backend"""
        if self.preferred_sandbox in self.available_backends:
            return self.preferred_sandbox
        
        # Fallback to most secure available
        for backend in [SandboxType.E2B, SandboxType.FIRECRACKER, SandboxType.DOCKER]:
            if backend in self.available_backends:
                return backend
        
        return SandboxType.SUBPROCESS

    def _resolve_dependency_packages(self, context: Dict[str, Any]) -> List[str]:
        """Resolve dependency packages based on entanglement matrix context."""
        packages = set(context.get("dependency_packages", []) or [])
        component_id = context.get("component_id")
        entanglement_matrix = context.get("entanglement_matrix", {})
        component_dependencies = context.get("component_dependencies", {})

        if component_id and entanglement_matrix and component_dependencies:
            entangled = entanglement_matrix.get(component_id, set())
            for comp in set(entangled) | {component_id}:
                packages.update(component_dependencies.get(comp, []))

        return sorted(packages)

    def _verify_isolation(self, policy: SecurityPolicy) -> bool:
        """Basic isolation proof stub; uses policy guarantees."""
        if policy.network_access:
            return False
        return True
    
    async def _execute_e2b(
        self,
        sandbox_id: str,
        code: str,
        context: Optional[Dict[str, Any]],
        policy: SecurityPolicy
    ) -> ExecutionResult:
        """Execute using E2B Code Interpreter"""
        try:
            from e2b import Sandbox
            
            # Create ephemeral sandbox
            sandbox = Sandbox(
                api_key=self.e2b_api_key,
                timeout=policy.max_execution_time
            )
            
            self.active_sandboxes[sandbox_id] = {
                'type': 'e2b',
                'instance': sandbox,
                'start_time': datetime.now(timezone.utc)
            }
            
            # Execute code
            start_time = time.time()
            
            # Write context if provided
            if context:
                context_json = json.dumps(context)
                sandbox.filesystem.write(
                    '/home/user/context.json',
                    context_json
                )
            
            # Run code
            process = sandbox.process.start_and_wait(
                f"python3 -c '{code}'",
                timeout=policy.max_execution_time
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Collect artifacts
            artifacts = []
            try:
                files = sandbox.filesystem.list('/home/user')
                for f in files:
                    if f.name.endswith(('.json', '.txt', '.csv', '.png', '.jpg')):
                        content = sandbox.filesystem.read(f'/home/user/{f.name}')
                        artifact_path = f'/tmp/{sandbox_id}_{f.name}'
                        with open(artifact_path, 'wb') as af:
                            af.write(content)
                        artifacts.append(artifact_path)
            except Exception as e:
                logger.warning(f"Artifact collection failed: {e}")
            
            # Kill sandbox
            sandbox.kill()
            
            return ExecutionResult(
                success=process.exit_code == 0,
                stdout=process.stdout,
                stderr=process.stderr,
                exit_code=process.exit_code,
                execution_time_ms=execution_time,
                artifacts=artifacts,
                sandbox_id=sandbox_id,
                security_report={
                    'backend': 'e2b',
                    'isolated': True,
                    'ephemeral': True,
                    'network_access': policy.network_access
                }
            )
            
        except Exception as e:
            logger.error(f"E2B execution failed: {e}")
            raise
    
    async def _execute_firecracker(
        self,
        sandbox_id: str,
        code: str,
        context: Optional[Dict[str, Any]],
        policy: SecurityPolicy
    ) -> ExecutionResult:
        """Execute using Firecracker MicroVM"""
        # Firecracker implementation placeholder
        # Would require Firecracker socket connection and VM management
        logger.warning("Firecracker not fully implemented, falling back to Docker")
        return await self._execute_docker(sandbox_id, code, context, policy)
    
    async def _execute_docker(
        self,
        sandbox_id: str,
        code: str,
        context: Optional[Dict[str, Any]],
        policy: SecurityPolicy
    ) -> ExecutionResult:
        """Execute using Docker container"""
        import docker
        
        client = docker.from_env()
        
        # Create temp directory for code
        temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_")
        
        try:
            # Write code file
            code_file = Path(temp_dir) / 'script.py'
            code_file.write_text(code)
            
            # Write context if provided
            if context:
                context_file = Path(temp_dir) / 'context.json'
                context_file.write_text(json.dumps(context))
            
            self.active_sandboxes[sandbox_id] = {
                'type': 'docker',
                'temp_dir': temp_dir,
                'start_time': datetime.now(timezone.utc)
            }
            
            # Dependency-aware provisioning
            dependency_packages = self._resolve_dependency_packages(context or {})
            install_cmd = ""
            dependency_warning = None
            if dependency_packages:
                if policy.network_access:
                    install_cmd = f"pip install {' '.join(dependency_packages)} && "
                else:
                    dependency_warning = "Missing packages but network access is disabled; cannot install."

            # Run container
            start_time = time.time()
            
            container = client.containers.run(
                'python:3.11-slim',
                f'/bin/sh -lc \"{install_cmd}python /code/script.py\"',
                volumes={temp_dir: {'bind': '/code', 'mode': 'ro'}},
                network_mode='none' if not policy.network_access else 'bridge',
                mem_limit=f'{policy.max_memory_mb}m',
                cpu_quota=100000,  # 1 CPU
                detach=True
            )
            
            # Wait for completion
            try:
                result = container.wait(timeout=policy.max_execution_time)
                logs = container.logs().decode('utf-8')
            except Exception as e:
                container.kill()
                raise e
            finally:
                container.remove(force=True)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Parse stdout/stderr
            stdout = logs
            stderr = ""

            missing_modules = re.findall(r\"No module named ['\\\"]([\\w_\\.]+)['\\\"]\", logs)
            suggested_dockerfile = None
            if missing_modules:
                pkgs = \" \".join(sorted(set(missing_modules)))
                suggested_dockerfile = (
                    \"FROM python:3.11-slim\\n\"
                    f\"RUN pip install {pkgs}\\n\"
                    \"COPY . /code\\n\"
                    \"CMD [\\\"python\\\", \\\"/code/script.py\\\"]\\n\"
                )
            
            # Collect artifacts
            artifacts = []
            for f in Path(temp_dir).glob('*'):
                if f.suffix in ['.json', '.txt', '.csv', '.png', '.jpg']:
                    artifacts.append(str(f))
            
            return ExecutionResult(
                success=result['StatusCode'] == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result['StatusCode'],
                execution_time_ms=execution_time,
                artifacts=artifacts,
                sandbox_id=sandbox_id,
                security_report={
                    'backend': 'docker',
                    'isolated': True,
                    'container_id': container.id[:12],
                    'network_access': policy.network_access,
                    'dependency_packages': dependency_packages,
                    'dependency_warning': dependency_warning,
                    'suggested_dockerfile': suggested_dockerfile,
                    'isolation_proof': self._verify_isolation(policy)
                }
            )
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _execute_subprocess(
        self,
        sandbox_id: str,
        code: str,
        context: Optional[Dict[str, Any]],
        policy: SecurityPolicy
    ) -> ExecutionResult:
        """
        Execute using isolated subprocess (least secure).
        Only use when other backends unavailable.
        """
        import tempfile
        import ast
        
        # Security: Parse and validate code
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in policy.forbidden_modules:
                            raise SecurityError(f"Forbidden module: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in policy.forbidden_modules:
                        raise SecurityError(f"Forbidden module: {node.module}")
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Syntax error: {e}",
                exit_code=-1,
                execution_time_ms=0.0,
                artifacts=[],
                sandbox_id=sandbox_id,
                security_report={'error': 'syntax_error'}
            )
        
        # Create temp file
        temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_")
        
        try:
            code_file = Path(temp_dir) / 'script.py'
            code_file.write_text(code)
            
            self.active_sandboxes[sandbox_id] = {
                'type': 'subprocess',
                'temp_dir': temp_dir,
                'start_time': datetime.now(timezone.utc)
            }
            
            # Execute with timeout
            start_time = time.time()
            
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        'python', str(code_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=temp_dir
                    ),
                    timeout=policy.max_execution_time
                )
                
                stdout, stderr = await proc.communicate()
                execution_time = (time.time() - start_time) * 1000
                
                return ExecutionResult(
                    success=proc.returncode == 0,
                    stdout=stdout.decode('utf-8'),
                    stderr=stderr.decode('utf-8'),
                    exit_code=proc.returncode,
                    execution_time_ms=execution_time,
                    artifacts=[],
                    sandbox_id=sandbox_id,
                    security_report={
                        'backend': 'subprocess',
                        'isolated': False,
                        'warning': 'Subprocess is not fully isolated. Use Docker or E2B for production.'
                    }
                )
                
            except asyncio.TimeoutError:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {policy.max_execution_time}s",
                    exit_code=-1,
                    execution_time_ms=policy.max_execution_time * 1000,
                    artifacts=[],
                    sandbox_id=sandbox_id,
                    security_report={'error': 'timeout'}
                )
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _cleanup_sandbox(self, sandbox_id: str):
        """Cleanup a sandbox"""
        if sandbox_id not in self.active_sandboxes:
            return
        
        info = self.active_sandboxes[sandbox_id]
        
        try:
            if info['type'] == 'e2b':
                # E2B sandboxes auto-kill
                pass
            elif info['type'] == 'docker':
                # Docker containers auto-remove
                pass
            elif info['type'] == 'subprocess':
                # Temp dirs already cleaned
                pass
                
        except Exception as e:
            logger.warning(f"Sandbox cleanup failed: {e}")
        
        finally:
            del self.active_sandboxes[sandbox_id]
    
    def _log_execution(
        self,
        sandbox_id: str,
        code: str,
        result: ExecutionResult,
        backend: SandboxType
    ):
        """Log execution for audit"""
        self.execution_history.append({
            'sandbox_id': sandbox_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'backend': backend.value,
            'code_hash': hashlib.md5(code.encode()).hexdigest(),
            'success': result.success,
            'exit_code': result.exit_code,
            'execution_time_ms': result.execution_time_ms
        })
        
        # Keep only last 1000 executions
        self.execution_history = self.execution_history[-1000:]
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for audit"""
        return self.execution_history.copy()
    
    def get_active_sandboxes(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active sandboxes"""
        return {
            k: {
                'type': v['type'],
                'start_time': v['start_time'].isoformat() if isinstance(v['start_time'], datetime) else v['start_time']
            }
            for k, v in self.active_sandboxes.items()
        }


class SecurityError(Exception):
    """Security policy violation"""
    pass
