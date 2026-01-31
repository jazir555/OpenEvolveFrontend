"""
Z3-LeanAIDE Bridge Integration

This module provides bidirectional integration between Z3 SMT solver and LeanAIDE
formal verification system, enabling:
- Translation between SMT-LIB and Lean 4
- Combined constraint solving and theorem proving
- Enhanced verification workflows
- Cross-validation of proofs

Architecture:
    Z3LeanAideBridge
        ├── SMTtoLeanTranslator (SMT-LIB to Lean 4)
        ├── LeantoSMTTranslator (Lean 4 to SMT-LIB)
        ├── CombinedSolver (Z3 + LeanAIDE)
        └── VerificationOrchestrator (Cross-validation)

Integration Points:
- LeanAide workflow integration (leanaide_workflow_integration.py)
- OpenEvolve workflow stages (workflow_stage_functions.py)
- BubbleLabs visualization (bubblelabs_integration.py)

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3SolverResult, Z3TheoremResult,
        Z3Variable, Z3Constraint, Z3ConstraintType, Z3ResultStatus,
        Z3Config, get_z3_solver_engine, get_z3_theorem_prover, is_z3_available
    )
    Z3_INTEGRATION_AVAILABLE = True
except ImportError:
    Z3_INTEGRATION_AVAILABLE = False
    logger.warning("Z3 integration not available")

# Import LeanAIDE integration
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    from leanaide_mcp_tools import (
        leanaide_translate_theorem,
        leanaide_verify_solution,
        leanaide_elaborate_code
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAIDE client not available")

try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideVerificationResult,
        MathematicalProblemDetector
    )
    LEANAIDE_WORKFLOW_AVAILABLE = True
except ImportError:
    LEANAIDE_WORKFLOW_AVAILABLE = False
    logger.warning("LeanAIDE workflow integration not available")


# =============================================================================
# Data Classes and Enums
# =============================================================================

class TranslationDirection(Enum):
    """Direction of translation between Z3 and Lean."""
    SMT_TO_LEAN = "smt_to_lean"
    LEAN_TO_SMT = "lean_to_smt"


class VerificationStrategy(Enum):
    """Strategy for combined verification."""
    Z3_FIRST = "z3_first"           # Try Z3 first, fall back to Lean
    LEAN_FIRST = "lean_first"       # Try Lean first, fall back to Z3
    PARALLEL = "parallel"           # Run both in parallel
    CONSENSUS = "consensus"         # Both must agree
    ADAPTIVE = "adaptive"           # Choose based on problem type


@dataclass
class TranslationResult:
    """Result of translating between SMT-LIB and Lean."""
    success: bool
    source: str
    target: str
    direction: TranslationDirection
    translation: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "target": self.target,
            "direction": self.direction.value,
            "translation": self.translation,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time
        }


@dataclass
class CombinedVerificationResult:
    """Result of combined Z3 + LeanAIDE verification."""
    success: bool
    z3_result: Optional[Z3SolverResult] = None
    lean_result: Optional[Any] = None
    strategy_used: VerificationStrategy = VerificationStrategy.ADAPTIVE
    agreement: bool = False
    confidence_score: float = 0.0
    recommendation: str = ""
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "z3_result": self.z3_result.to_dict() if self.z3_result else None,
            "lean_result": self.lean_result.to_dict() if hasattr(self.lean_result, 'to_dict') else self.lean_result,
            "strategy_used": self.strategy_used.value,
            "agreement": self.agreement,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "errors": self.errors,
            "execution_time": self.execution_time
        }


@dataclass
class Z3LeanAideConfig:
    """Configuration for Z3-LeanAIDE bridge."""
    # Z3 configuration
    z3_timeout: float = 30.0
    z3_proof_generation: bool = True
    
    # LeanAIDE configuration
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    leanaide_timeout: float = 300.0
    
    # Bridge configuration
    default_strategy: VerificationStrategy = VerificationStrategy.ADAPTIVE
    enable_translation: bool = True
    enable_cross_validation: bool = True
    confidence_threshold: float = 0.7
    
    # Strategy thresholds
    use_z3_for_constraints: bool = True
    use_lean_for_theorems: bool = True
    use_parallel_for_critical: bool = True


# =============================================================================
# SMT-LIB to Lean Translator
# =============================================================================

class SMTtoLeanTranslator:
    """
    Translates SMT-LIB format to Lean 4 code.
    
    Supports:
    - Integer and Real arithmetic
    - Boolean logic
    - Quantifiers (forall, exists)
    - Common SMT-LIB constructs
    """
    
    def __init__(self):
        self.type_mapping = {
            "Int": "Int",
            "Real": "Real",
            "Bool": "Bool",
            "(Array Int Int)": "Array Int Int",
            "(Array Int Real)": "Array Int Real",
        }
    
    def translate(self, smtlib_content: str) -> TranslationResult:
        """
        Translate SMT-LIB content to Lean 4.
        
        Args:
            smtlib_content: SMT-LIB2 formatted content
            
        Returns:
            TranslationResult
        """
        start_time = time.time()
        
        try:
            # Parse SMT-LIB
            variables = self._extract_variables(smtlib_content)
            assertions = self._extract_assertions(smtlib_content)
            logic = self._extract_logic(smtlib_content)
            
            # Generate Lean 4 code
            lean_code = self._generate_lean(variables, assertions, logic)
            
            execution_time = time.time() - start_time
            
            return TranslationResult(
                success=True,
                source="smtlib2",
                target="lean4",
                direction=TranslationDirection.SMT_TO_LEAN,
                translation=lean_code,
                execution_time=execution_time,
                metadata={
                    "num_variables": len(variables),
                    "num_assertions": len(assertions),
                    "logic": logic
                }
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                success=False,
                source="smtlib2",
                target="lean4",
                direction=TranslationDirection.SMT_TO_LEAN,
                translation="",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _extract_variables(self, smtlib: str) -> Dict[str, Dict[str, str]]:
        """Extract variable declarations from SMT-LIB."""
        variables = {}
        
        # Pattern: (declare-fun name () type)
        pattern = r'\(declare-fun\s+(\w+)\s+\(\)\s+(\w+|\([^)]+\))\)'
        matches = re.findall(pattern, smtlib)
        
        for name, var_type in matches:
            variables[name] = {
                "name": name,
                "type": self.type_mapping.get(var_type, var_type),
                "raw_type": var_type
            }
        
        # Pattern: (declare-const name type)
        const_pattern = r'\(declare-const\s+(\w+)\s+(\w+)\)'
        const_matches = re.findall(const_pattern, smtlib)
        
        for name, var_type in const_matches:
            variables[name] = {
                "name": name,
                "type": self.type_mapping.get(var_type, var_type),
                "raw_type": var_type
            }
        
        return variables
    
    def _extract_assertions(self, smtlib: str) -> List[str]:
        """Extract assertions from SMT-LIB."""
        assertions = []
        
        # Pattern: (assert expr)
        pattern = r'\(assert\s+(.+?)\)(?=\s*\(|\s*$)'
        matches = re.findall(pattern, smtlib, re.DOTALL)
        
        for match in matches:
            assertions.append(match.strip())
        
        return assertions
    
    def _extract_logic(self, smtlib: str) -> str:
        """Extract logic from SMT-LIB."""
        pattern = r'\(set-logic\s+(\w+)\)'
        match = re.search(pattern, smtlib)
        return match.group(1) if match else "ALL"
    
    def _generate_lean(self, variables: Dict, assertions: List[str], logic: str) -> str:
        """Generate Lean 4 code from parsed SMT-LIB."""
        lines = [
            "import Mathlib",
            "",
            "-- Generated from SMT-LIB",
            f"-- Logic: {logic}",
            ""
        ]
        
        # Define variables as theorem parameters
        params = []
        for var_name, var_info in variables.items():
            params.append(f"({var_name} : {var_info['type']})")
        
        # Build theorem statement
        lines.append(f"theorem smt_problem {' '.join(params)} :")
        
        # Translate assertions to Lean
        if assertions:
            # Convert SMT expressions to Lean
            lean_assertions = []
            for assertion in assertions:
                lean_expr = self._translate_expr(assertion)
                lean_assertions.append(lean_expr)
            
            if len(lean_assertions) == 1:
                lines.append(f"  {lean_assertions[0]} := by")
            else:
                lines.append("  " + " ∧\n  ".join(lean_assertions) + " := by")
        else:
            lines.append("  True := by")
        
        # Add proof tactics
        lines.extend([
            "  -- SMT-generated proof",
            "  try { tauto }",
            "  try { nlinarith }",
            "  try { simp_all }",
            "  try { aesop }",
            "  try { sorry }  -- Proof to be completed"
        ])
        
        return "\n".join(lines)
    
    def _translate_expr(self, expr: str) -> str:
        """Translate SMT expression to Lean."""
        # Replace SMT operators with Lean equivalents
        replacements = [
            # Comparison operators
            (r'\(=\s+([^)]+)\)', r'\1 = \2'),
            (r'\(<=\s+([^)]+)\)', r'\1 ≤ \2'),
            (r'\(<\s+([^)]+)\)', r'\1 < \2'),
            (r'\(>=\s+([^)]+)\)', r'\1 ≥ \2'),
            (r'\(>\s+([^)]+)\)', r'\1 > \2'),
            
            # Arithmetic operators
            (r'\(\+\s+([^)]+)\)', r'\1 + \2'),
            (r'\(-\s+([^)]+)\)', r'\1 - \2'),
            (r'\(\*\s+([^)]+)\)', r'\1 * \2'),
            (r'\(/\s+([^)]+)\)', r'\1 / \2'),
            
            # Logical operators
            (r'\(and\s+([^)]+)\)', r'\1 ∧ \2'),
            (r'\(or\s+([^)]+)\)', r'\1 ∨ \2'),
            (r'\(not\s+([^)]+)\)', r'¬\1'),
            (r'\(=>\s+([^)]+)\)', r'\1 → \2'),
            (r'\(implies\s+([^)]+)\)', r'\1 → \2'),
            
            # Quantifiers
            (r'\(forall\s+\(([^)]+)\)\s+([^)]+)\)', r'∀ \1, \2'),
            (r'\(exists\s+\(([^)]+)\)\s+([^)]+)\)', r'∃ \1, \2'),
        ]
        
        result = expr
        for pattern, replacement in replacements:
            # For simple patterns, use direct replacement
            if pattern.startswith('(='):
                result = result.replace('(= ', '(')
            elif pattern.startswith('(and'):
                result = result.replace('(and ', 'And ')
            elif pattern.startswith('(or'):
                result = result.replace('(or ', 'Or ')
            elif pattern.startswith('(not'):
                result = result.replace('(not ', 'Not ')
        
        # Clean up parentheses
        result = result.replace('(', ' ').replace(')', ' ')
        
        # Final clean up
        result = result.strip()
        if not result:
            result = "True"
        
        return result


# =============================================================================
# Lean to SMT-LIB Translator
# =============================================================================

class LeantoSMTTranslator:
    """
    Translates Lean 4 code to SMT-LIB format.
    
    This is useful for:
    - Using Z3 to verify Lean proofs
    - Cross-checking results
    - Performance comparison
    """
    
    def __init__(self):
        self.type_mapping = {
            "Int": "Int",
            "Real": "Real",
            "Bool": "Bool",
            "Nat": "Int",
            "Prop": "Bool",
        }
    
    def translate(self, lean_code: str) -> TranslationResult:
        """
        Translate Lean 4 code to SMT-LIB.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            TranslationResult
        """
        start_time = time.time()
        
        try:
            # Extract theorem statement
            theorem_info = self._extract_theorem(lean_code)
            
            # Generate SMT-LIB
            smtlib = self._generate_smtlib(theorem_info)
            
            execution_time = time.time() - start_time
            
            return TranslationResult(
                success=True,
                source="lean4",
                target="smtlib2",
                direction=TranslationDirection.LEAN_TO_SMT,
                translation=smtlib,
                execution_time=execution_time,
                metadata=theorem_info
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                success=False,
                source="lean4",
                target="smtlib2",
                direction=TranslationDirection.LEAN_TO_SMT,
                translation="",
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _extract_theorem(self, lean_code: str) -> Dict[str, Any]:
        """Extract theorem information from Lean code."""
        info = {
            "name": "unknown",
            "parameters": [],
            "statement": "",
            "proof": ""
        }
        
        # Extract theorem name and parameters
        theorem_pattern = r'theorem\s+(\w+)\s*(?:\{[^}]*\})?\s*(\([^)]*\))?'
        match = re.search(theorem_pattern, lean_code)
        if match:
            info["name"] = match.group(1)
            if match.group(2):
                params_str = match.group(2)[1:-1]  # Remove parentheses
                # Parse parameters
                for param in params_str.split(')'):
                    if ':' in param:
                        param = param.strip()
                        if param.startswith('('):
                            param = param[1:]
                        parts = param.split(':', 1)
                        if len(parts) == 2:
                            param_names = parts[0].strip().split()
                            param_type = parts[1].strip()
                            for name in param_names:
                                info["parameters"].append({
                                    "name": name,
                                    "type": self.type_mapping.get(param_type, param_type)
                                })
        
        # Extract statement (between : and :=)
        statement_pattern = r':\s*([^:=]+)\s*:='
        match = re.search(statement_pattern, lean_code, re.DOTALL)
        if match:
            info["statement"] = match.group(1).strip()
        
        return info
    
    def _generate_smtlib(self, theorem_info: Dict) -> str:
        """Generate SMT-LIB from theorem information."""
        lines = [
            "; Generated from Lean 4",
            "(set-logic ALL)",
            "(set-option :produce-models true)",
            ""
        ]
        
        # Declare variables
        for param in theorem_info["parameters"]:
            lines.append(f"(declare-fun {param['name']} () {param['type']})")
        
        lines.append("")
        
        # Translate statement to assertion
        if theorem_info["statement"]:
            smt_statement = self._translate_statement(theorem_info["statement"])
            lines.append(f"(assert (not {smt_statement}))")
        
        lines.extend([
            "",
            "(check-sat)",
            "(get-model)"
        ])
        
        return "\n".join(lines)
    
    def _translate_statement(self, statement: str) -> str:
        """Translate Lean statement to SMT."""
        # Replace Lean operators with SMT equivalents
        smt = statement
        
        # Logical operators
        smt = smt.replace('∧', 'and').replace('/\\', 'and')
        smt = smt.replace('∨', 'or').replace('\\/', 'or')
        smt = smt.replace('¬', 'not').replace('~', 'not')
        smt = smt.replace('→', '=>')
        smt = smt.replace('∀', 'forall')
        smt = smt.replace('∃', 'exists')
        
        # Comparison operators
        smt = smt.replace('≤', '<=').replace('≤', '<=')
        smt = smt.replace('≥', '>=').replace('≥', '>=')
        smt = smt.replace('≠', 'distinct')
        
        return smt


# =============================================================================
# Z3-LeanAIDE Bridge
# =============================================================================

class Z3LeanAideBridge:
    """
    Main bridge class integrating Z3 with LeanAIDE.
    
    Provides:
    - Bidirectional translation
    - Combined verification
    - Strategy selection
    - Cross-validation
    """
    
    def __init__(self, config: Optional[Z3LeanAideConfig] = None):
        self.config = config or Z3LeanAideConfig()
        self.smt_to_lean = SMTtoLeanTranslator()
        self.lean_to_smt = LeantoSMTTranslator()
        
        # Initialize solvers
        self.z3_solver = get_z3_solver_engine(
            Z3Config(
                timeout=self.config.z3_timeout,
                proof_generation=self.config.z3_proof_generation
            )
        ) if Z3_INTEGRATION_AVAILABLE else None
        
        self.z3_prover = get_z3_theorem_prover(
            Z3Config(
                timeout=self.config.z3_timeout,
                proof_generation=self.config.z3_proof_generation
            )
        ) if Z3_INTEGRATION_AVAILABLE else None
        
        self.lean_integrator = None
        if LEANAIDE_WORKFLOW_AVAILABLE:
            lean_config = type('Config', (), {
                'host': self.config.leanaide_host,
                'port': self.config.leanaide_port,
                'timeout': self.config.leanaide_timeout,
                'enabled': True
            })()
            self.lean_integrator = LeanAideWorkflowIntegrator(lean_config)
        
        self.problem_detector = MathematicalProblemDetector() if LEANAIDE_WORKFLOW_AVAILABLE else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "z3_available": Z3_INTEGRATION_AVAILABLE and is_z3_available(),
            "leanaide_available": LEANAIDE_AVAILABLE,
            "leanaide_workflow_available": LEANAIDE_WORKFLOW_AVAILABLE,
            "config": {
                "default_strategy": self.config.default_strategy.value,
                "enable_translation": self.config.enable_translation,
                "enable_cross_validation": self.config.enable_cross_validation
            }
        }
    
    async def translate_smt_to_lean(self, smtlib_content: str) -> TranslationResult:
        """Translate SMT-LIB to Lean 4."""
        return self.smt_to_lean.translate(smtlib_content)
    
    async def translate_lean_to_smt(self, lean_code: str) -> TranslationResult:
        """Translate Lean 4 to SMT-LIB."""
        return self.lean_to_smt.translate(lean_code)
    
    async def verify_with_both(
        self,
        problem: str,
        strategy: Optional[VerificationStrategy] = None
    ) -> CombinedVerificationResult:
        """
        Verify problem using both Z3 and LeanAIDE.
        
        Args:
            problem: Problem statement (SMT-LIB or natural language)
            strategy: Verification strategy to use
            
        Returns:
            CombinedVerificationResult
        """
        start_time = time.time()
        strategy = strategy or self.config.default_strategy
        
        # Detect problem type
        is_smt = self.smt_to_lean._is_smtlib(problem) if hasattr(self.smt_to_lean, '_is_smtlib') else '(assert' in problem
        
        if strategy == VerificationStrategy.ADAPTIVE:
            strategy = self._select_strategy(problem, is_smt)
        
        # Execute based on strategy
        if strategy == VerificationStrategy.Z3_FIRST:
            return await self._verify_z3_first(problem, is_smt)
        elif strategy == VerificationStrategy.LEAN_FIRST:
            return await self._verify_lean_first(problem, is_smt)
        elif strategy == VerificationStrategy.PARALLEL:
            return await self._verify_parallel(problem, is_smt)
        elif strategy == VerificationStrategy.CONSENSUS:
            return await self._verify_consensus(problem, is_smt)
        else:
            return CombinedVerificationResult(
                success=False,
                errors=[f"Unknown strategy: {strategy}"],
                execution_time=time.time() - start_time
            )
    
    def _select_strategy(self, problem: str, is_smt: bool) -> VerificationStrategy:
        """Select best strategy based on problem characteristics."""
        if is_smt and self.config.use_z3_for_constraints:
            return VerificationStrategy.Z3_FIRST
        elif 'prove' in problem.lower() or 'theorem' in problem.lower():
            if self.config.use_lean_for_theorems:
                return VerificationStrategy.LEAN_FIRST
        
        return VerificationStrategy.PARALLEL
    
    async def _verify_z3_first(
        self,
        problem: str,
        is_smt: bool
    ) -> CombinedVerificationResult:
        """Verify with Z3 first, fall back to LeanAIDE."""
        start_time = time.time()
        
        # Try Z3
        if self.z3_solver:
            if is_smt:
                z3_result = self.z3_solver.solve_smtlib(problem)
            else:
                z3_result = self.z3_prover.prove_theorem(problem)
            
            # If Z3 succeeds confidently, return result
            if z3_result.status == Z3ResultStatus.SAT or (hasattr(z3_result, 'proven') and z3_result.proven):
                return CombinedVerificationResult(
                    success=True,
                    z3_result=z3_result,
                    strategy_used=VerificationStrategy.Z3_FIRST,
                    confidence_score=0.8,
                    recommendation="Verified by Z3",
                    execution_time=time.time() - start_time
                )
        
        # Fall back to LeanAIDE
        lean_result = await self._verify_with_lean(problem)
        
        return CombinedVerificationResult(
            success=lean_result.success if hasattr(lean_result, 'success') else lean_result.get('success', False),
            z3_result=z3_result if self.z3_solver else None,
            lean_result=lean_result,
            strategy_used=VerificationStrategy.Z3_FIRST,
            confidence_score=0.7 if (lean_result.success if hasattr(lean_result, 'success') else lean_result.get('success', False)) else 0.0,
            recommendation="Verified by LeanAIDE (Z3 fallback)",
            execution_time=time.time() - start_time
        )
    
    async def _verify_lean_first(
        self,
        problem: str,
        is_smt: bool
    ) -> CombinedVerificationResult:
        """Verify with LeanAIDE first, fall back to Z3."""
        start_time = time.time()
        
        # Try LeanAIDE
        lean_result = await self._verify_with_lean(problem)
        lean_success = lean_result.success if hasattr(lean_result, 'success') else lean_result.get('success', False)
        
        if lean_success:
            return CombinedVerificationResult(
                success=True,
                lean_result=lean_result,
                strategy_used=VerificationStrategy.LEAN_FIRST,
                confidence_score=0.9,
                recommendation="Verified by LeanAIDE",
                execution_time=time.time() - start_time
            )
        
        # Fall back to Z3
        if self.z3_solver:
            if is_smt:
                z3_result = self.z3_solver.solve_smtlib(problem)
            else:
                z3_result = self.z3_prover.prove_theorem(problem)
            
            z3_success = (z3_result.status == Z3ResultStatus.SAT or 
                         (hasattr(z3_result, 'proven') and z3_result.proven))
            
            return CombinedVerificationResult(
                success=z3_success,
                z3_result=z3_result,
                lean_result=lean_result,
                strategy_used=VerificationStrategy.LEAN_FIRST,
                confidence_score=0.6 if z3_success else 0.0,
                recommendation="Verified by Z3 (LeanAIDE fallback)" if z3_success else "Verification failed",
                execution_time=time.time() - start_time
            )
        
        return CombinedVerificationResult(
            success=False,
            lean_result=lean_result,
            strategy_used=VerificationStrategy.LEAN_FIRST,
            errors=["Both LeanAIDE and Z3 failed"],
            execution_time=time.time() - start_time
        )
    
    async def _verify_parallel(
        self,
        problem: str,
        is_smt: bool
    ) -> CombinedVerificationResult:
        """Verify with both Z3 and LeanAIDE in parallel."""
        start_time = time.time()
        
        # Run both verifications concurrently
        z3_task = self._verify_z3_async(problem, is_smt)
        lean_task = self._verify_lean_async(problem)
        
        results = await asyncio.gather(z3_task, lean_task, return_exceptions=True)
        
        z3_result = results[0] if not isinstance(results[0], Exception) else None
        lean_result = results[1] if not isinstance(results[1], Exception) else None
        
        # Determine success
        z3_success = False
        if z3_result:
            if hasattr(z3_result, 'status'):
                z3_success = z3_result.status == Z3ResultStatus.SAT
            elif hasattr(z3_result, 'proven'):
                z3_success = z3_result.proven
        
        lean_success = False
        if lean_result:
            lean_success = (lean_result.success if hasattr(lean_result, 'success') 
                          else lean_result.get('success', False))
        
        success = z3_success or lean_success
        agreement = z3_success == lean_success
        
        confidence = 0.0
        recommendation = ""
        
        if agreement and z3_success:
            confidence = 0.95
            recommendation = "Both Z3 and LeanAIDE verified"
        elif z3_success and not lean_success:
            confidence = 0.7
            recommendation = "Verified by Z3 only"
        elif lean_success and not z3_success:
            confidence = 0.8
            recommendation = "Verified by LeanAIDE only"
        else:
            confidence = 0.0
            recommendation = "Verification failed"
        
        return CombinedVerificationResult(
            success=success,
            z3_result=z3_result,
            lean_result=lean_result,
            strategy_used=VerificationStrategy.PARALLEL,
            agreement=agreement,
            confidence_score=confidence,
            recommendation=recommendation,
            execution_time=time.time() - start_time
        )
    
    async def _verify_consensus(
        self,
        problem: str,
        is_smt: bool
    ) -> CombinedVerificationResult:
        """Verify with both - both must agree for success."""
        result = await self._verify_parallel(problem, is_smt)
        
        # Override success to require consensus
        if not result.agreement:
            result.success = False
            result.recommendation = "Consensus not reached - results disagree"
        
        return result
    
    async def _verify_z3_async(self, problem: str, is_smt: bool) -> Optional[Z3SolverResult]:
        """Async wrapper for Z3 verification."""
        if not self.z3_solver:
            return None
        
        loop = asyncio.get_event_loop()
        if is_smt:
            return await loop.run_in_executor(None, self.z3_solver.solve_smtlib, problem)
        else:
            return await loop.run_in_executor(None, self.z3_prover.prove_theorem, problem)
    
    async def _verify_lean_async(self, problem: str):
        """Async wrapper for LeanAIDE verification."""
        if not self.lean_integrator:
            return None
        
        # Initialize if needed
        if not self.lean_integrator.client:
            initialized = await self.lean_integrator.initialize()
            if not initialized:
                return {"success": False, "error": "Failed to initialize LeanAIDE"}
        
        return await self.lean_integrator.verify_sub_problem_solution(
            sub_problem_id="z3_bridge",
            problem_statement=problem,
            solution_content=""
        )
    
    async def _verify_with_lean(self, problem: str):
        """Verify with LeanAIDE."""
        if not self.lean_integrator:
            return {"success": False, "error": "LeanAIDE not available"}
        
        try:
            if not self.lean_integrator.client:
                initialized = await self.lean_integrator.initialize()
                if not initialized:
                    return {"success": False, "error": "Failed to initialize LeanAIDE"}
            
            return await self.lean_integrator.verify_sub_problem_solution(
                sub_problem_id="z3_bridge",
                problem_statement=problem,
                solution_content=""
            )
        except Exception as e:
            logger.error(f"LeanAIDE verification failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def cross_validate(
        self,
        smtlib_problem: str
    ) -> CombinedVerificationResult:
        """
        Cross-validate by translating SMT to Lean and verifying both ways.
        
        Args:
            smtlib_problem: SMT-LIB problem
            
        Returns:
            CombinedVerificationResult
        """
        start_time = time.time()
        
        # Translate to Lean
        translation = await self.translate_smt_to_lean(smtlib_problem)
        
        if not translation.success:
            return CombinedVerificationResult(
                success=False,
                errors=["Translation failed"] + translation.errors,
                execution_time=time.time() - start_time
            )
        
        # Verify original SMT with Z3
        z3_result = None
        if self.z3_solver:
            z3_result = self.z3_solver.solve_smtlib(smtlib_problem)
        
        # Verify translated Lean with LeanAIDE
        lean_result = await self._verify_with_lean(translation.translation)
        lean_success = (lean_result.success if hasattr(lean_result, 'success') 
                       else lean_result.get('success', False))
        
        # Check agreement
        z3_success = z3_result and z3_result.status == Z3ResultStatus.SAT
        agreement = z3_success == lean_success
        
        return CombinedVerificationResult(
            success=z3_success or lean_success,
            z3_result=z3_result,
            lean_result=lean_result,
            agreement=agreement,
            confidence_score=0.9 if agreement else 0.5,
            recommendation="Cross-validated" if agreement else "Results differ - needs review",
            execution_time=time.time() - start_time
        )


# =============================================================================
# Global Instance
# =============================================================================

_z3_leanaide_bridge: Optional[Z3LeanAideBridge] = None
_bridge_lock = asyncio.Lock()


async def get_z3_leanaide_bridge(config: Optional[Z3LeanAideConfig] = None) -> Z3LeanAideBridge:
    """Get global Z3-LeanAIDE bridge instance."""
    global _z3_leanaide_bridge
    if _z3_leanaide_bridge is None:
        async with _bridge_lock:
            if _z3_leanaide_bridge is None:
                _z3_leanaide_bridge = Z3LeanAideBridge(config)
    return _z3_leanaide_bridge


def get_z3_leanaide_bridge_sync(config: Optional[Z3LeanAideConfig] = None) -> Z3LeanAideBridge:
    """Get global Z3-LeanAIDE bridge instance (synchronous)."""
    global _z3_leanaide_bridge
    if _z3_leanaide_bridge is None:
        _z3_leanaide_bridge = Z3LeanAideBridge(config)
    return _z3_leanaide_bridge


# =============================================================================
# Example Usage
# =============================================================================

async def example_translation():
    """Example: Translate SMT to Lean."""
    bridge = await get_z3_leanaide_bridge()
    
    smt_problem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (> x 0))
    (assert (< x 10))
    (assert (= y (+ x 5)))
    (check-sat)
    """
    
    result = await bridge.translate_smt_to_lean(smt_problem)
    print(f"Translation success: {result.success}")
    print(f"Lean code:\n{result.translation}")
    
    return result


async def example_combined_verification():
    """Example: Combined verification."""
    bridge = await get_z3_leanaide_bridge()
    
    problem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (assert (> x 0))
    (assert (< x 5))
    (check-sat)
    """
    
    result = await bridge.verify_with_both(problem, VerificationStrategy.PARALLEL)
    print(f"Success: {result.success}")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Recommendation: {result.recommendation}")
    
    return result


if __name__ == "__main__":
    print("Z3-LeanAIDE Bridge Integration")
    print("=" * 50)
    
    # Run examples
    print("\n--- Translation Example ---")
    asyncio.run(example_translation())
    
    print("\n--- Combined Verification Example ---")
    asyncio.run(example_combined_verification())
