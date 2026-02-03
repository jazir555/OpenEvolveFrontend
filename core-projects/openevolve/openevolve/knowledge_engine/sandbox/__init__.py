"""
Execution Sandbox Module for OpenEvolve Knowledge Engine

Provides secure, ephemeral execution environments for code.
Integrates E2B (Code Interpreter SDK) and Firecracker MicroVMs.

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

from .sandbox_manager import SandboxManager, SandboxType, ExecutionResult, SecurityPolicy

__all__ = [
    'SandboxManager',
    'SandboxType',
    'ExecutionResult',
    'SecurityPolicy'
]
