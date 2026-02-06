"""
Lean 4 Enhanced Integration with LLM Support for OpenEvolve

Complete REST API integration for Lean 4 compiler with:
- Automatic Lean installation detection and setup
- Real LLM integration for autoformalization (OpenAI/Anthropic)
- Proof checking service with mathlib4 support
- Batch processing capability
- Error recovery and logging

Author: OpenEvolve
Version: 2.0.0 - Enhanced with LLM Integration
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import hashlib
import shutil

# Configure logging
logger = logging.getLogger(__name__)

# Try to import LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")

# Import setup module
try:
    from setup_lean4 import Lean4SetupManager, detect_lean_installation, LeanInstallationStatus
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False
    logger.warning("setup_lean4 not available")

try:
    from leanaide_integration import create_integration as _create_root_leanaide_integration
    ROOT_LEANAIDE_STATUS_AVAILABLE = True
except ImportError:
    _create_root_leanaide_integration = None
    ROOT_LEANAIDE_STATUS_AVAILABLE = False
    logger.warning("Root LeanAide status provider not available")


def _default_web3_formal_status() -> Dict[str, Any]:
    """Return a stable Web3 formal-status payload."""
    formal_capabilities = {
        "solidity_invariant_translation": False,
        "invariant_translation_verification": False,
        "symbolic_exploit_witness": False,
        "composite_exploit_verification": False,
    }
    return {
        "web3_formal_available": False,
        "web3_formal_verification_available": False,
        "web3_formal_tools": [],
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": False,
    }


def _collect_web3_formal_status() -> Dict[str, Any]:
    """Collect Web3 formal status from root LeanAide integration when available."""
    if not ROOT_LEANAIDE_STATUS_AVAILABLE or _create_root_leanaide_integration is None:
        return _default_web3_formal_status()

    try:
        status = _create_root_leanaide_integration().get_web3_formal_status()
        if not isinstance(status, dict):
            return _default_web3_formal_status()
        default_status = _default_web3_formal_status()
        formal_capabilities = status.get("formal_capabilities")
        if not isinstance(formal_capabilities, dict):
            formal_capabilities = default_status["formal_capabilities"]
        web3_formal_tools = status.get("web3_formal_tools")
        if not isinstance(web3_formal_tools, list):
            web3_formal_tools = []
        web3_formal_tools = sorted(set(str(tool) for tool in web3_formal_tools if tool))
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(value) for value in formal_capabilities.values()
        )
        return {
            "web3_formal_available": bool(
                status.get("web3_formal_available", inferred_formal_available)
            ),
            "web3_formal_verification_available": bool(
                status.get("web3_formal_verification_available", inferred_formal_available)
            ),
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "audit_exploit_verification_available": bool(
                status.get("audit_exploit_verification_available")
            ),
        }
    except Exception:
        logger.debug("Failed to collect root LeanAide Web3 status", exc_info=True)
        return _default_web3_formal_status()


# ============================================================================
# Enums and Data Structures
# ============================================================================

class Lean4TaskType(Enum):
    """Types of Lean 4 tasks"""
    CHECK_PROOF = "check_proof"
    BUILD_PROJECT = "build_project"
    AUTOFORMALIZE = "autoformalize"
    COMPLETE_PROOF = "complete_proof"
    SUGGEST_TACTICS = "suggest_tactics"
    PARSE_EXPRESSION = "parse_expression"
    TYPE_CHECK = "type_check"


class VerificationStatus(Enum):
    """Verification status"""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    PROOF_ERROR = "proof_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    PENDING = "pending"
    LEAN_NOT_INSTALLED = "lean_not_installed"


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


@dataclass
class Lean4ServerConfig:
    """Configuration for Lean 4 server"""
    lean_executable: str = "lean"
    lake_executable: str = "lake"
    mathlib_path: Optional[str] = None
    working_dir: str = "./lean_workspace/mathlib_project"
    timeout_seconds: float = 60.0
    max_memory_mb: int = 4096
    enable_caching: bool = True
    cache_dir: str = ".lean_cache"
    parallel_jobs: int = 4
    server_host: str = "localhost"
    server_port: int = 7654
    
    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.NONE
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-opus-20240229"
    autoformalization_temperature: float = 0.2
    max_llm_retries: int = 3


@dataclass
class VerificationResult:
    """Result of Lean 4 verification"""
    status: VerificationStatus
    success: bool
    code: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output: str = ""
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lean_available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "output": self.output,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "lean_available": self.lean_available
        }


@dataclass
class AutoformalizationResult:
    """Result of autoformalization"""
    success: bool
    natural_language: str
    lean_code: str
    domain: str
    confidence: float = 0.0
    iterations: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    llm_provider: str = "none"
    verification_result: Optional[VerificationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "natural_language": self.natural_language,
            "lean_code": self.lean_code,
            "domain": self.domain,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "errors_encountered": self.errors_encountered,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "llm_provider": self.llm_provider,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None
        }


@dataclass
class ProofSuggestion:
    """Suggested proof tactics"""
    tactic: str
    confidence: float
    explanation: str
    expected_outcome: str


@dataclass
class ProofCompletionResult:
    """Result of proof completion"""
    success: bool
    original_code: str
    completed_code: str
    tactics_used: List[str]
    proof_length: int
    confidence: float
    execution_time: float


# ============================================================================
# Lean 4 Installation Manager
# ============================================================================

class Lean4InstallationManager:
    """Manages Lean 4 installation detection and auto-setup"""
    
    def __init__(self):
        self.status: LeanInstallationStatus = None
        self.setup_manager: Optional[Any] = None
        if SETUP_AVAILABLE:
            self.setup_manager = Lean4SetupManager()
        self._checked = False
    
    def check_installation(self, force: bool = False) -> LeanInstallationStatus:
        """Check if Lean 4 is installed"""
        if self._checked and not force and self.status:
            return self.status
        
        if SETUP_AVAILABLE:
            self.status = detect_lean_installation()
        else:
            # Basic check without setup module
            self.status = self._basic_detection()
        
        self._checked = True
        return self.status
    
    def _basic_detection(self) -> 'LeanInstallationStatus':
        """Basic detection without setup module"""
        @dataclass
        class BasicStatus:
            lean_available: bool = False
            lake_available: bool = False
            mathlib_available: bool = False
            lean_version: Optional[str] = None
            lake_version: Optional[str] = None
            mathlib_path: Optional[str] = None
            elan_available: bool = False
            
            def is_fully_functional(self) -> bool:
                return self.lean_available and self.lake_available
        
        status = BasicStatus()
        
        # Check lean
        try:
            result = subprocess.run(["lean", "--version"], capture_output=True, text=True, timeout=5)
            status.lean_available = result.returncode == 0
            if status.lean_available:
                status.lean_version = result.stdout.strip()[:50]
        except:
            pass
        
        # Check lake
        try:
            result = subprocess.run(["lake", "--version"], capture_output=True, text=True, timeout=5)
            status.lake_available = result.returncode == 0
            if status.lake_available:
                status.lake_version = result.stdout.strip()[:50]
        except:
            pass
        
        return status
    
    def ensure_installed(self, auto_setup: bool = True) -> bool:
        """Ensure Lean 4 is installed, optionally auto-setup"""
        status = self.check_installation()
        
        if status.lean_available and status.lake_available:
            return True
        
        if auto_setup and self.setup_manager:
            logger.info("Lean 4 not found. Attempting auto-setup...")
            result = self.setup_manager.auto_install()
            
            if result.success:
                self.check_installation(force=True)
                return True
            else:
                logger.error(f"Auto-setup failed: {result.message}")
                return False
        
        return False
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions"""
        if self.setup_manager:
            return self.setup_manager.get_setup_instructions()
        
        return """
Lean 4 is not installed. Please install it manually:

1. Install elan (Lean version manager):
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

2. Install Lean 4:
   elan toolchain install stable
   elan default stable

3. Verify:
   lean --version
   lake --version
"""


# ============================================================================
# LLM Client Integration
# ============================================================================

class LLMClient:
    """Client for LLM API calls (OpenAI/Anthropic)"""
    
    def __init__(self, config: Lean4ServerConfig):
        self.config = config
        self.openai_client: Optional[Any] = None
        self.anthropic_client: Optional[Any] = None
        
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and config.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Anthropic if available
        if ANTHROPIC_AVAILABLE and config.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")
    
    def is_available(self) -> bool:
        """Check if any LLM is available"""
        return self.openai_client is not None or self.anthropic_client is not None
    
    def get_provider(self) -> LLMProvider:
        """Get the active LLM provider"""
        if self.config.llm_provider == LLMProvider.OPENAI and self.openai_client:
            return LLMProvider.OPENAI
        elif self.config.llm_provider == LLMProvider.ANTHROPIC and self.anthropic_client:
            return LLMProvider.ANTHROPIC
        elif self.openai_client:
            return LLMProvider.OPENAI
        elif self.anthropic_client:
            return LLMProvider.ANTHROPIC
        return LLMProvider.NONE
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Generate text using LLM"""
        temp = temperature if temperature is not None else self.config.autoformalization_temperature
        
        provider = self.get_provider()
        
        if provider == LLMProvider.OPENAI and self.openai_client:
            return await self._generate_openai(prompt, system_message, temp)
        elif provider == LLMProvider.ANTHROPIC and self.anthropic_client:
            return await self._generate_anthropic(prompt, system_message, temp)
        
        return False, "No LLM provider available"
    
    async def _generate_openai(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float
    ) -> Tuple[bool, str]:
        """Generate using OpenAI"""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000
                )
            )
            
            return True, response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return False, str(e)
    
    async def _generate_anthropic(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float
    ) -> Tuple[bool, str]:
        """Generate using Anthropic"""
        try:
            system = system_message or ""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=2000,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            return True, response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return False, str(e)


# ============================================================================
# Lean 4 Verification Engine
# ============================================================================

class Lean4VerificationEngine:
    """
    Complete verification engine for Lean 4 code.
    
    Supports:
    - Automatic Lean installation detection
    - Syntax checking
    - Type checking
    - Proof verification
    - Mathlib4 integration
    - Batch processing
    """
    
    def __init__(self, config: Optional[Lean4ServerConfig] = None):
        """Initialize the verification engine"""
        self.config = config or Lean4ServerConfig()
        self.cache: Dict[str, VerificationResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_jobs)
        self.installation_manager = Lean4InstallationManager()
        
        # Check Lean installation
        self.lean_available = self.installation_manager.check_installation().lean_available
        
        if not self.lean_available:
            logger.warning("Lean 4 is not installed. Run: python setup_lean4.py --auto-install")
        
        # Ensure working directory exists
        os.makedirs(self.config.working_dir, exist_ok=True)
        if self.config.enable_caching:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"Lean4VerificationEngine initialized (Lean available: {self.lean_available})")
    
    def _get_cache_key(self, code: str) -> str:
        """Generate cache key for code"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    async def verify(self, code: str, use_cache: bool = True) -> VerificationResult:
        """
        Verify Lean 4 code.
        
        Args:
            code: Lean 4 code to verify
            use_cache: Whether to use caching
            
        Returns:
            VerificationResult with status and errors
        """
        start_time = time.time()
        
        # Check if Lean is available
        if not self.lean_available:
            return VerificationResult(
                status=VerificationStatus.LEAN_NOT_INSTALLED,
                success=False,
                code=code,
                errors=["Lean 4 is not installed. Run: python setup_lean4.py --auto-install"],
                execution_time=time.time() - start_time,
                lean_available=False
            )
        
        # Check cache
        if use_cache and self.config.enable_caching:
            cache_key = self._get_cache_key(code)
            if cache_key in self.cache:
                logger.info("Cache hit for verification")
                return self.cache[cache_key]
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.lean', delete=False, dir=self.config.working_dir
            ) as f:
                # Add imports if not present
                if not code.strip().startswith('import'):
                    f.write("import Mathlib\n\n")
                f.write(code)
                temp_file = f.name
            
            # Run lean compiler
            result = await self._run_lean_compiler(temp_file)
            
            # Cleanup
            os.unlink(temp_file)
            
            # Update cache
            if use_cache and self.config.enable_caching:
                cache_key = self._get_cache_key(code)
                self.cache[cache_key] = result
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                status=VerificationStatus.SERVER_ERROR,
                success=False,
                code=code,
                errors=[str(e)],
                execution_time=time.time() - start_time,
                lean_available=self.lean_available
            )
    
    async def _run_lean_compiler(self, file_path: str) -> VerificationResult:
        """Run Lean 4 compiler on file"""
        try:
            # Use lake env to ensure mathlib and other dependencies are available
            cmd = [
                self.config.lake_executable, "env",
                self.config.lean_executable,
                file_path,
                "-M", str(self.config.max_memory_mb),
                "-T", str(int(self.config.timeout_seconds * 1000))
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.working_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                return VerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    success=False,
                    code="",
                    errors=[f"Timeout after {self.config.timeout_seconds}s"],
                    lean_available=True
                )
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            # Parse errors
            errors = []
            warnings = []
            
            # Parse Lean 4 error format
            error_pattern = r'(\S+\.lean):(\d+):(\d+):\s*(error|warning):\s*(.+)'
            for match in re.finditer(error_pattern, stderr_str):
                file, line, col, level, msg = match.groups()
                if level == 'error':
                    errors.append(f"Line {line}:{col}: {msg}")
                else:
                    warnings.append(f"Line {line}:{col}: {msg}")
            
            success = proc.returncode == 0 and not errors
            
            return VerificationResult(
                status=VerificationStatus.SUCCESS if success else VerificationStatus.PROOF_ERROR,
                success=success,
                code="",
                errors=errors,
                warnings=warnings,
                output=stdout_str,
                lean_available=True
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.SERVER_ERROR,
                success=False,
                code="",
                errors=[str(e)],
                lean_available=True
            )
    
    async def verify_batch(
        self,
        codes: List[str],
        use_cache: bool = True
    ) -> List[VerificationResult]:
        """Verify multiple Lean 4 code snippets in parallel"""
        tasks = [self.verify(code, use_cache) for code in codes]
        return await asyncio.gather(*tasks)
    
    def get_setup_instructions(self) -> str:
        """Get Lean setup instructions"""
        return self.installation_manager.get_setup_instructions()


# ============================================================================
# Enhanced Autoformalization Engine with LLM
# ============================================================================

class Lean4AutoformalizationEngine:
    """
    Enhanced autoformalization engine with real LLM integration.
    
    Supports:
    - Natural language -> Lean 4 via LLM
    - LaTeX formula -> Lean 4
    - Python/numpy -> Lean 4
    - Proof sketch -> formal proof
    - Auto-correction with verification
    """
    
    def __init__(
        self,
        config: Optional[Lean4ServerConfig] = None,
        verification_engine: Optional[Lean4VerificationEngine] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize autoformalization engine"""
        self.config = config or Lean4ServerConfig()
        self.verification = verification_engine or Lean4VerificationEngine(self.config)
        self.llm = llm_client or LLMClient(self.config)
        self.max_iterations = 3
        
        # Domain mappings
        self.domain_mappings = self._initialize_domain_mappings()
        
        logger.info(f"Lean4AutoformalizationEngine initialized (LLM available: {self.llm.is_available()})")
    
    def _initialize_domain_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize mappings for different mathematical domains"""
        return {
            "real_analysis": {
                "limit": "Filter.Tendsto",
                "continuous": "Continuous",
                "differentiable": "Differentiable",
                "derivative": "deriv",
                "integral": "integral",
                "open_set": "IsOpen",
                "closed_set": "IsClosed"
            },
            "complex_analysis": {
                "holomorphic": "DifferentiableOn ℂ",
                "analytic": "AnalyticOnNhd",
                "meromorphic": "MeromorphicOn",
                "residue": "residue"
            },
            "topology": {
                "neighborhood": "nhds",
                "compact": "CompactSpace",
                "connected": "ConnectedSpace",
                "hausdorff": "T2Space"
            },
            "measure_theory": {
                "measurable": "Measurable",
                "integrable": "Integrable",
                "almost_everywhere": "∀ᵐ",
                "sigma_algebra": "MeasurableSpace"
            },
            "algebra": {
                "group": "Group",
                "ring": "Ring",
                "field": "Field",
                "homomorphism": "MonoidHom",
                "isomorphism": "RingEquiv"
            }
        }
    
    async def autoformalize(
        self,
        natural_language: str,
        domain: str = "general",
        statement_type: str = "theorem",
        context: Optional[Dict[str, Any]] = None
    ) -> AutoformalizationResult:
        """
        Convert natural language to Lean 4 code using LLM.
        
        Args:
            natural_language: Natural language description
            domain: Mathematical domain hint
            statement_type: theorem, definition, or lemma
            context: Additional context
            
        Returns:
            AutoformalizationResult with Lean 4 code
        """
        start_time = time.time()
        context = context or {}
        
        # Check if LLM is available
        if not self.llm.is_available():
            logger.warning("No LLM available. Using template-based formalization.")
            return await self._template_autoformalize(
                natural_language, domain, statement_type, context
            )
        
        try:
            # Step 1: Generate initial formalization using LLM
            lean_code = await self._generate_with_llm(
                natural_language, domain, statement_type, context
            )
            
            # Step 2: Verify and iterate
            best_result = lean_code
            best_confidence = 0.0
            errors_encountered = []
            final_verification = None
            
            for iteration in range(self.max_iterations):
                final_verification = await self.verification.verify(lean_code)
                
                if final_verification.success:
                    confidence = self._calculate_confidence(lean_code, natural_language)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = lean_code
                    break
                else:
                    errors_encountered.extend(final_verification.errors)
                    # Attempt correction with LLM
                    if iteration < self.max_iterations - 1:
                        lean_code = await self._correct_with_llm(
                            lean_code, final_verification.errors, natural_language
                        )
            
            return AutoformalizationResult(
                success=final_verification.success if final_verification else False,
                natural_language=natural_language,
                lean_code=best_result,
                domain=domain,
                confidence=best_confidence,
                iterations=iteration + 1,
                errors_encountered=errors_encountered,
                metadata={
                    "statement_type": statement_type,
                    "execution_time": time.time() - start_time,
                    "verification_success": final_verification.success if final_verification else False
                },
                llm_provider=self.llm.get_provider().value,
                verification_result=final_verification
            )
            
        except Exception as e:
            logger.error(f"Autoformalization failed: {e}")
            return AutoformalizationResult(
                success=False,
                natural_language=natural_language,
                lean_code="",
                domain=domain,
                errors_encountered=[str(e)],
                llm_provider=self.llm.get_provider().value
            )
    
    async def _generate_with_llm(
        self,
        nl: str,
        domain: str,
        statement_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate Lean code using LLM"""
        
        system_message = """You are an expert in Lean 4 theorem proving and formal mathematics.
Convert the given natural language mathematical statement into valid Lean 4 code.

Rules:
1. Always include `import Mathlib` at the beginning
2. Use proper Lean 4 syntax
3. For incomplete proofs, use `sorry` as a placeholder
4. Include comments explaining the mathematical meaning
5. Use appropriate domain-specific definitions from Mathlib

Output ONLY the Lean 4 code, no additional explanation."""

        prompt = f"""Convert the following {statement_type} statement to Lean 4 code:

Domain: {domain}
Statement: {nl}

Generate valid Lean 4 code:"""

        success, result = await self.llm.generate(prompt, system_message)
        
        if success:
            # Clean up the response
            code = self._extract_lean_code(result)
            return code
        else:
            # Fall back to template
            logger.warning(f"LLM generation failed: {result}. Using template fallback.")
            return self._generate_template_code(nl, domain, statement_type)
    
    async def _correct_with_llm(
        self,
        code: str,
        errors: List[str],
        original_nl: str
    ) -> str:
        """Correct Lean code using LLM"""
        
        system_message = """You are an expert in Lean 4 theorem proving.
Fix the errors in the provided Lean 4 code."""

        prompt = f"""The following Lean 4 code has errors:

```lean
{code}
```

Errors:
{chr(10).join(errors)}

Original mathematical statement:
{original_nl}

Please provide corrected Lean 4 code with the errors fixed.
Output ONLY the corrected code:"""

        success, result = await self.llm.generate(prompt, system_message)
        
        if success:
            return self._extract_lean_code(result)
        return code
    
    def _extract_lean_code(self, text: str) -> str:
        """Extract Lean code from LLM response"""
        # Look for code blocks
        code_block_pattern = r'```lean\s*(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for any code blocks
        code_block_pattern = r'```\s*(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Return as-is, but clean up
        lines = text.split('\n')
        # Remove explanation lines
        code_lines = []
        for line in lines:
            if not line.strip().startswith(('Here', 'This', 'The', 'Note', 'Please')):
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    def _generate_template_code(self, nl: str, domain: str, statement_type: str) -> str:
        """Generate template code when LLM fails"""
        theorem_name = self._generate_theorem_name(nl)
        
        return f"""import Mathlib

-- {nl}
theorem {theorem_name} :
  True := by
  sorry
"""
    
    async def _template_autoformalize(
        self,
        natural_language: str,
        domain: str,
        statement_type: str,
        context: Dict[str, Any]
    ) -> AutoformalizationResult:
        """Fallback template-based autoformalization"""
        lean_code = self._generate_template_code(natural_language, domain, statement_type)
        
        return AutoformalizationResult(
            success=False,
            natural_language=natural_language,
            lean_code=lean_code,
            domain=domain,
            confidence=0.3,
            errors_encountered=["LLM not available - template used"],
            llm_provider="none"
        )
    
    def _generate_theorem_name(self, nl: str) -> str:
        """Generate a descriptive theorem name"""
        words = nl.split()[:5]
        name = "_".join(w.lower() for w in words if w.isalnum())
        hash_suffix = hashlib.sha256(nl.encode()).hexdigest()[:6]
        return f"{name}_{hash_suffix}"
    
    def _calculate_confidence(self, code: str, nl: str) -> float:
        """Calculate confidence score for formalization"""
        confidence = 0.5
        
        if "import Mathlib" in code:
            confidence += 0.1
        
        if "theorem" in code or "def" in code or "lemma" in code:
            confidence += 0.1
        
        if "sorry" not in code and "by" in code:
            confidence += 0.2
        
        if "trivial" in code or "rfl" in code or "simp" in code:
            confidence += 0.1
        
        return min(confidence, 1.0)


# ============================================================================
# Main Enhanced LeanAide Service
# ============================================================================

class LeanAideServiceEnhanced:
    """
    Enhanced LeanAide service with LLM integration and auto-setup.
    
    Provides unified interface for:
    - Verification with auto-setup
    - LLM-powered autoformalization
    - Proof completion
    """
    
    def __init__(
        self,
        config: Optional[Lean4ServerConfig] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None
    ):
        """Initialize the enhanced LeanAide service"""
        self.config = config or Lean4ServerConfig()
        
        # Set API keys if provided
        if openai_api_key:
            self.config.openai_api_key = openai_api_key
            self.config.llm_provider = LLMProvider.OPENAI
        if anthropic_api_key:
            self.config.anthropic_api_key = anthropic_api_key
            if not openai_api_key:
                self.config.llm_provider = LLMProvider.ANTHROPIC
        
        self.verification = Lean4VerificationEngine(self.config)
        self.llm = LLMClient(self.config)
        self.autoformalization = Lean4AutoformalizationEngine(
            self.config, self.verification, self.llm
        )
        
        logger.info(f"LeanAideServiceEnhanced initialized")
    
    async def verify(self, code: str) -> VerificationResult:
        """Verify Lean 4 code"""
        return await self.verification.verify(code)
    
    async def autoformalize(
        self,
        natural_language: str,
        domain: str = "general",
        statement_type: str = "theorem"
    ) -> AutoformalizationResult:
        """Autoformalize using LLM"""
        return await self.autoformalization.autoformalize(
            natural_language, domain, statement_type
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        status = self.verification.installation_manager.check_installation()
        web3_status = _collect_web3_formal_status()
        return {
            "lean_available": status.lean_available if status else False,
            "lake_available": status.lake_available if status else False,
            "mathlib_available": status.mathlib_available if status else False,
            "llm_available": self.llm.is_available(),
            "llm_provider": self.llm.get_provider().value,
            "web3_formal_available": web3_status["web3_formal_available"],
            "web3_formal_verification_available": web3_status[
                "web3_formal_verification_available"
            ],
            "web3_formal_tools": web3_status["web3_formal_tools"],
            "formal_capabilities": web3_status["formal_capabilities"],
            "audit_exploit_verification_available": web3_status[
                "audit_exploit_verification_available"
            ],
        }
    
    async def setup_lean(self, auto_install: bool = True) -> bool:
        """Setup Lean 4 if not installed"""
        return self.verification.installation_manager.ensure_installed(auto_install)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_lean4_service(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> LeanAideServiceEnhanced:
    """Create a LeanAideServiceEnhanced instance"""
    return LeanAideServiceEnhanced(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key
    )


async def main():
    """Example usage of enhanced Lean 4 integration"""
    
    print("=" * 70)
    print("Lean 4 Enhanced Integration - LLM-Powered Autoformalization")
    print("=" * 70)
    
    # Create service (optionally with API keys)
    service = create_lean4_service()
    
    # Check status
    status = service.get_status()
    print("\n📊 Service Status:")
    print(f"  Lean available: {status['lean_available']}")
    print(f"  Lake available: {status['lake_available']}")
    print(f"  LLM available:  {status['llm_available']} ({status['llm_provider']})")
    
    # Check Lean installation
    if not status['lean_available']:
        print("\nWARNING: Lean 4 is not installed.")
        print("Run: python setup_lean4.py --auto-install")
        return
    
    # Example 1: Verify simple proof
    print("\n1. VERIFY LEAN 4 CODE")
    print("-" * 40)
    code = """
theorem test_theorem : 1 + 1 = 2 := by
  rfl
"""
    result = await service.verify(code)
    print(f"   Status: {result.status.value}")
    print(f"   Success: {result.success}")
    print(f"   Errors: {result.errors if result.errors else 'None'}")
    
    # Example 2: LLM Autoformalization (if LLM available)
    if status['llm_available']:
        print("\n2. LLM AUTOFORMALIZATION")
        print("-" * 40)
        nl_statement = "The limit as x approaches 0 of sin(x)/x equals 1"
        print(f"   Input: {nl_statement}")
        result = await service.autoformalize(nl_statement, domain="real_analysis")
        print(f"   Success: {result.success}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   LLM Provider: {result.llm_provider}")
        print(f"   Generated Code:\n{result.lean_code[:300]}...")
    else:
        print("\n2. LLM AUTOFORMALIZATION - SKIPPED (No LLM API key)")
        print("   To enable: export OPENAI_API_KEY=your_key")


if __name__ == "__main__":
    asyncio.run(main())
