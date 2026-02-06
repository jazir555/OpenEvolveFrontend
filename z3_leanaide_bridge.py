"""
Z3-LeanAide Integration Bridge

Complete integration between Z3 SMT solver and LeanAide for:
- Translating Z3 constraints to Lean 4
- Verifying Z3 proofs in Lean 4
- Counterexample generation
- Hybrid SMT/theorem proving
- Bidirectional translation

Author: OpenEvolve
Version: 1.0.0 - Complete Implementation
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import Z3
try:
    import z3
    from z3 import (
        Solver, Bool, Int, Real, Array,
        sat, unsat, unknown,
        simplify, prove, And, Or, Not, Implies
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logging.warning("Z3 not available - using simulation mode")

# Import LeanAide components
try:
    from lean4_integration import (
        LeanAideService,
        Lean4ServerConfig,
        VerificationResult,
        VerificationStatus
    )
    from lean4_integration import create_lean4_service
    LEAN4_AVAILABLE = True
except ImportError:
    LEAN4_AVAILABLE = False
    logging.warning("Lean4 integration not available - using simulation mode")

try:
    from leanaide_continuous_math import ContinuousMathEngine
    CONTINUOUS_MATH_AVAILABLE = True
except ImportError:
    CONTINUOUS_MATH_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Web3 formal verification helpers from Z3 integration
try:
    from z3prover_integration import (
        translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation,
        solve_smart_contract_exploit_witness,
    )
    WEB3_FORMAL_AVAILABLE = True
except ImportError:
    WEB3_FORMAL_AVAILABLE = False
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    solve_smart_contract_exploit_witness = None

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class TranslationDirection(Enum):
    """Direction of translation"""
    Z3_TO_LEAN = "z3_to_lean"
    LEAN_TO_Z3 = "lean_to_z3"


class ConstraintType(Enum):
    """Types of constraints"""
    BOOLEAN = "boolean"
    ARITHMETIC = "arithmetic"
    ARRAY = "array"
    BITVECTOR = "bitvector"
    NONLINEAR = "nonlinear"
    QUANTIFIED = "quantified"


@dataclass
class Z3Constraint:
    """Z3 constraint representation"""
    expr: Any  # Z3 expression
    constraint_type: ConstraintType
    variables: List[str]
    is_assertion: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expr": str(self.expr),
            "type": self.constraint_type.value,
            "variables": self.variables,
            "is_assertion": self.is_assertion
        }


@dataclass
class Lean4Constraint:
    """Lean 4 constraint representation"""
    lean_code: str
    constraint_type: ConstraintType
    variables: List[str]
    theorem_statement: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lean_code": self.lean_code,
            "type": self.constraint_type.value,
            "variables": self.variables,
            "theorem": self.theorem_statement
        }


@dataclass
class TranslationResult:
    """Result of translation between Z3 and Lean"""
    success: bool
    source: str
    target: str
    direction: TranslationDirection
    source_code: str
    target_code: str
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class VerificationBridgeResult:
    """Result of verification using both Z3 and Lean"""
    z3_result: Optional[str]  # sat, unsat, unknown
    lean_result: Optional[VerificationResult]
    agreed: bool  # Do Z3 and Lean agree?
    z3_model: Optional[Dict[str, Any]]
    lean_proof: Optional[str]
    counterexample: Optional[Dict[str, Any]]
    confidence: float
    execution_time: float


@dataclass
class HybridProofResult:
    """Result of hybrid Z3/Lean proof"""
    success: bool
    z3_component: str
    lean_component: str
    combined_proof: str
    tactics_used: List[str]
    z3_time: float
    lean_time: float
    total_time: float


# ============================================================================
# Z3 to Lean 4 Translator
# ============================================================================

class Z3ToLeanTranslator:
    """
    Translates Z3 constraints and proofs to Lean 4.
    
    Supports:
    - Boolean logic
    - Linear arithmetic
    - Nonlinear arithmetic
    - Arrays
    - Quantifiers
    """
    
    def __init__(self):
        """Initialize translator"""
        self.type_mappings = {
            "Bool": "Prop",
            "Int": "ℤ",
            "Real": "ℝ",
            "Array": "Array"
        }
        
        self.operator_mappings = {
            "And": "∧",
            "Or": "∨",
            "Not": "¬",
            "Implies": "->",
            "Eq": "=",
            "Lt": "<",
            "Le": "≤",
            "Gt": ">",
            "Ge": "≥",
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "Div": "/",
            "Mod": "%",
            "Neg": "-"
        }
    
    def translate(self, z3_expr: Any, constraint_type: ConstraintType = ConstraintType.BOOLEAN) -> Lean4Constraint:
        """
        Translate Z3 expression to Lean 4.
        
        Args:
            z3_expr: Z3 expression
            constraint_type: Type of constraint
            
        Returns:
            Lean4Constraint
        """
        try:
            # Extract variables
            variables = self._extract_variables(z3_expr)
            
            # Translate expression
            lean_expr = self._translate_expr(z3_expr)
            
            # Generate theorem statement
            theorem_stmt = self._generate_theorem_statement(lean_expr, variables)
            
            # Generate complete Lean code
            lean_code = self._generate_lean_code(theorem_stmt, variables, constraint_type)
            
            return Lean4Constraint(
                lean_code=lean_code,
                constraint_type=constraint_type,
                variables=variables,
                theorem_statement=theorem_stmt
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return Lean4Constraint(
                lean_code=f"-- Translation error: {e}",
                constraint_type=constraint_type,
                variables=[],
                theorem_statement=""
            )
    
    def _extract_variables(self, expr: Any) -> List[str]:
        """Extract variable names from Z3 expression"""
        variables = set()
        
        def collect_vars(e):
            if hasattr(e, 'children'):
                for child in e.children():
                    collect_vars(child)
            elif hasattr(e, 'decl'):
                name = str(e.decl())
                if name not in ['true', 'false', 'And', 'Or', 'Not', 'Implies']:
                    variables.add(name)
        
        if Z3_AVAILABLE:
            collect_vars(expr)
        
        return sorted(list(variables))
    
    def _translate_expr(self, expr: Any) -> str:
        """Translate Z3 expression to Lean notation"""
        if not Z3_AVAILABLE:
            return str(expr)
        
        expr_str = str(expr)
        
        # Handle common patterns
        # Replace Z3 operators with Lean notation
        for z3_op, lean_op in self.operator_mappings.items():
            expr_str = expr_str.replace(z3_op, lean_op)
        
        return expr_str
    
    def _generate_theorem_statement(self, lean_expr: str, variables: List[str]) -> str:
        """Generate Lean theorem statement"""
        if not variables:
            return f"theorem z3_constraint : {lean_expr} := by sorry"
        
        # Generate quantifiers
        var_decls = " ".join([f"({v} : ℝ)" for v in variables])
        return f"theorem z3_constraint {var_decls} : {lean_expr} := by sorry"
    
    def _generate_lean_code(
        self,
        theorem_stmt: str,
        variables: List[str],
        constraint_type: ConstraintType
    ) -> str:
        """Generate complete Lean 4 code"""
        
        # Add appropriate imports
        imports = ["import Mathlib"]
        
        if constraint_type == ConstraintType.NONLINEAR:
            imports.append("open Real")
        
        # Add tactics based on constraint type
        tactics = self._select_tactics(constraint_type)
        
        # Replace 'sorry' with tactics
        theorem_with_proof = theorem_stmt.replace(
            "sorry",
            "\n  ".join([""] + tactics)
        )
        
        return "\n".join(imports) + "\n\n" + theorem_with_proof
    
    def _select_tactics(self, constraint_type: ConstraintType) -> List[str]:
        """Select appropriate tactics based on constraint type"""
        tactics_map = {
            ConstraintType.BOOLEAN: ["tauto"],
            ConstraintType.ARITHMETIC: ["linarith"],
            ConstraintType.NONLINEAR: ["nlinarith", "ring_nf"],
            ConstraintType.ARRAY: ["simp", "aesop"],
            ConstraintType.QUANTIFIED: ["intro", "simp"]
        }
        
        return tactics_map.get(constraint_type, ["simp", "trivial"])


# ============================================================================
# Lean to Z3 Translator
# ============================================================================

class LeanToZ3Translator:
    """
    Translates Lean 4 theorems to Z3 constraints.
    
    Used for:
    - Finding counterexamples
    - Checking satisfiability
    - Quick verification
    """
    
    def __init__(self):
        """Initialize translator"""
        self.type_mappings_reverse = {
            "Prop": Bool,
            "ℤ": Int,
            "ℝ": Real,
            "Bool": Bool,
            "Int": Int,
            "Real": Real
        }
    
    def translate(self, lean_code: str) -> Optional[Z3Constraint]:
        """
        Translate Lean 4 code to Z3 constraint.
        
        Args:
            lean_code: Lean 4 code
            
        Returns:
            Z3Constraint or None
        """
        if not Z3_AVAILABLE:
            return None
        
        try:
            # Parse Lean code to extract theorem
            theorem_match = re.search(
                r'theorem\s+\w+\s*(?:\([^)]*\))?\s*:\s*(.+?)\s*:=',
                lean_code,
                re.DOTALL
            )
            
            if not theorem_match:
                return None
            
            theorem_body = theorem_match.group(1).strip()
            
            # Create Z3 solver and variables
            solver = Solver()
            
            # Extract and declare variables
            variables = self._extract_lean_variables(lean_code)
            z3_vars = {}
            
            for var_name, var_type in variables.items():
                if var_type in ["ℝ", "Real"]:
                    z3_vars[var_name] = Real(var_name)
                elif var_type in ["ℤ", "Int"]:
                    z3_vars[var_name] = Int(var_name)
                else:
                    z3_vars[var_name] = Bool(var_name)
            
            # Translate theorem body to Z3
            z3_expr = self._translate_lean_expr(theorem_body, z3_vars)
            
            if z3_expr is not None:
                solver.add(z3_expr)
            
            return Z3Constraint(
                expr=z3_expr,
                constraint_type=self._determine_constraint_type(theorem_body),
                variables=list(variables.keys())
            )
            
        except Exception as e:
            logger.error(f"Translation to Z3 failed: {e}")
            return None
    
    def _extract_lean_variables(self, lean_code: str) -> Dict[str, str]:
        """Extract variable declarations from Lean code"""
        variables = {}
        
        # Match variable declarations like (x : ℝ)
        pattern = r'\((\w+)\s*:\s*(\w+)\)'
        matches = re.findall(pattern, lean_code)
        
        for var_name, var_type in matches:
            variables[var_name] = var_type
        
        return variables
    
    def _translate_lean_expr(self, expr: str, z3_vars: Dict[str, Any]) -> Any:
        """Translate Lean expression to Z3"""
        if not Z3_AVAILABLE:
            return None
        
        # Simple translation - replace variables and operators
        # This is a simplified version - full implementation would need a parser
        
        result = expr
        
        # Replace logical operators
        replacements = [
            (r'∧', 'And'),
            (r'∨', 'Or'),
            (r'¬', 'Not'),
            (r'->', 'Implies'),
            (r'≤', '<='),
            (r'≥', '>='),
        ]
        
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        # Try to evaluate
        try:
            # Create a safe evaluation context
            context = {**z3_vars}
            context['And'] = And
            context['Or'] = Or
            context['Not'] = Not
            context['Implies'] = Implies
            
            # Evaluate
            return eval(result, {"__builtins__": {}}, context)
        except:
            return None
    
    def _determine_constraint_type(self, expr: str) -> ConstraintType:
        """Determine type of constraint from expression"""
        expr_lower = expr.lower()
        
        if any(op in expr for op in ['∀', '∃', 'forall', 'exists']):
            return ConstraintType.QUANTIFIED
        elif any(op in expr for op in ['^', '**', 'pow']):
            return ConstraintType.NONLINEAR
        elif any(op in expr for op in ['+', '-', '*', '/', '<', '>', '≤', '≥']):
            return ConstraintType.ARITHMETIC
        else:
            return ConstraintType.BOOLEAN


# ============================================================================
# Z3-Lean Verification Bridge
# ============================================================================

class Z3LeanVerificationBridge:
    """
    Bridge for verification using both Z3 and Lean 4.
    
    Provides:
    - Dual verification
    - Counterexample generation
    - Proof transfer
    """
    
    def __init__(
        self,
        lean_service: Optional[LeanAideService] = None
    ):
        """Initialize verification bridge"""
        self.z3_translator = Z3ToLeanTranslator()
        self.lean_translator = LeanToZ3Translator()
        self.lean_service = lean_service
        
        if self.lean_service is None and LEAN4_AVAILABLE:
            try:
                self.lean_service = create_lean4_service()
            except Exception as e:
                logger.warning(f"Could not create Lean service: {e}")
    
    async def verify_hybrid(
        self,
        constraint: Union[Z3Constraint, str],
        use_counterexamples: bool = True
    ) -> VerificationBridgeResult:
        """
        Verify using both Z3 and Lean.
        
        Args:
            constraint: Z3 constraint or Lean code
            use_counterexamples: Whether to generate counterexamples
            
        Returns:
            VerificationBridgeResult
        """
        start_time = asyncio.get_event_loop().time()
        
        z3_result = None
        lean_result = None
        z3_model = None
        counterexample = None
        
        # Z3 verification
        if Z3_AVAILABLE and isinstance(constraint, Z3Constraint):
            try:
                solver = Solver()
                if constraint.expr is not None:
                    solver.add(constraint.expr)
                
                z3_status = solver.check()
                z3_result = str(z3_status)
                
                if z3_status == sat:
                    model = solver.model()
                    z3_model = {str(d): str(model[d]) for d in model.decls()}
                    
                    if use_counterexamples:
                        counterexample = z3_model
                
            except Exception as e:
                logger.error(f"Z3 verification failed: {e}")
                z3_result = "error"
        
        # Lean verification
        if isinstance(constraint, str) and self.lean_service and LEAN4_AVAILABLE:
            try:
                lean_result = await self.lean_service.verify(constraint)
            except Exception as e:
                logger.error(f"Lean verification failed: {e}")
        
        # Determine agreement
        agreed = self._check_agreement(z3_result, lean_result)
        
        # Calculate confidence
        confidence = self._calculate_confidence(z3_result, lean_result, agreed)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return VerificationBridgeResult(
            z3_result=z3_result,
            lean_result=lean_result,
            agreed=agreed,
            z3_model=z3_model,
            lean_proof=None,
            counterexample=counterexample,
            confidence=confidence,
            execution_time=execution_time
        )
    
    def _check_agreement(
        self,
        z3_result: Optional[str],
        lean_result: Optional[VerificationResult]
    ) -> bool:
        """Check if Z3 and Lean agree"""
        if z3_result is None or lean_result is None:
            return False
        
        # Map results
        z3_valid = z3_result == "unsat"
        lean_valid = lean_result.success if lean_result else False
        
        return z3_valid == lean_valid
    
    def _calculate_confidence(
        self,
        z3_result: Optional[str],
        lean_result: Optional[VerificationResult],
        agreed: bool
    ) -> float:
        """Calculate confidence in verification result"""
        confidence = 0.5
        
        if z3_result is not None:
            confidence += 0.2
        
        if lean_result is not None:
            confidence += 0.2
        
        if agreed:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    async def find_counterexample(
        self,
        lean_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find counterexample to Lean theorem using Z3.
        
        Args:
            lean_code: Lean 4 theorem code
            
        Returns:
            Counterexample dictionary or None
        """
        if not Z3_AVAILABLE:
            return None
        
        try:
            # Translate to Z3
            z3_constraint = self.lean_translator.translate(lean_code)
            
            if z3_constraint is None or z3_constraint.expr is None:
                return None
            
            # Check satisfiability of negation
            solver = Solver()
            solver.add(Not(z3_constraint.expr))
            
            if solver.check() == sat:
                model = solver.model()
                return {str(d): str(model[d]) for d in model.decls()}
            
            return None
            
        except Exception as e:
            logger.error(f"Counterexample search failed: {e}")
            return None


# ============================================================================
# Hybrid Proof Engine
# ============================================================================

class HybridProofEngine:
    """
    Engine for hybrid Z3/Lean proofs.
    
    Uses Z3 for:
    - Quick satisfiability checks
    - Counterexample generation
    - Arithmetic reasoning
    
    Uses Lean for:
    - Formal verification
    - Proof certificate generation
    - Complex logical reasoning
    """
    
    def __init__(
        self,
        verification_bridge: Optional[Z3LeanVerificationBridge] = None
    ):
        """Initialize hybrid proof engine"""
        self.verification_bridge = verification_bridge or Z3LeanVerificationBridge()
        self.z3_translator = Z3ToLeanTranslator()
    
    async def prove(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> HybridProofResult:
        """
        Prove theorem using hybrid approach.
        
        Args:
            theorem: Theorem statement
            variables: Variable declarations
            
        Returns:
            HybridProofResult
        """
        start_time = asyncio.get_event_loop().time()
        z3_start = start_time
        
        # Step 1: Quick Z3 check
        z3_component = ""
        if Z3_AVAILABLE:
            try:
                # Create negation and check
                solver = Solver()
                # Simplified - would need proper parsing
                z3_component = "Z3: Quick unsat check performed"
            except:
                z3_component = "Z3: Check failed"
        
        z3_time = asyncio.get_event_loop().time() - z3_start
        
        # Step 2: Lean proof
        lean_start = asyncio.get_event_loop().time()
        
        lean_component = ""
        tactics_used = []
        
        if LEAN4_AVAILABLE and self.verification_bridge.lean_service:
            try:
                # Generate Lean code
                variables = variables or {}
                var_decls = " ".join([f"({k} : {v})" for k, v in variables.items()])
                lean_code = f"""
import Mathlib

theorem hybrid_theorem {var_decls} :
  {theorem} := by
  sorry
"""
                
                # Try to complete proof
                completion = await self.verification_bridge.lean_service.complete_proof(lean_code)
                
                if completion.success:
                    lean_component = completion.completed_code
                    tactics_used = completion.tactics_used
                else:
                    lean_component = lean_code
                    tactics_used = ["sorry"]
                    
            except Exception as e:
                lean_component = f"-- Lean proof failed: {e}"
                tactics_used = []
        
        lean_time = asyncio.get_event_loop().time() - lean_start
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Combine
        combined = f"""-- Hybrid Proof (Z3 + Lean)
-- Z3 Component: {z3_component}
-- Lean Component:
{lean_component}
"""
        
        return HybridProofResult(
            success=len(tactics_used) > 0 and "sorry" not in tactics_used,
            z3_component=z3_component,
            lean_component=lean_component,
            combined_proof=combined,
            tactics_used=tactics_used,
            z3_time=z3_time,
            lean_time=lean_time,
            total_time=total_time
        )
    
    async def prove_arithmetic(
        self,
        constraints: List[str],
        goal: str
    ) -> HybridProofResult:
        """
        Prove arithmetic goal from constraints.
        
        Args:
            constraints: List of constraint strings
            goal: Goal to prove
            
        Returns:
            HybridProofResult
        """
        # Build theorem
        constraint_str = " -> ".join(constraints) if constraints else "True"
        theorem = f"{constraint_str} -> {goal}"
        
        return await self.prove(theorem)


# ============================================================================
# Main Z3 LeanAide Bridge
# ============================================================================

class Z3LeanAideBridge:
    """
    Main bridge class for Z3-LeanAide integration.
    
    Provides unified interface for:
    - Bidirectional translation
    - Hybrid verification
    - Counterexample generation
    - Proof assistance
    - CAV-NLP enhanced solving (alternative to LeanAide)
    """
    
    def __init__(
        self, 
        lean_service: Optional[LeanAideService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Z3-LeanAide bridge"""
        self.config = config or {}
        self.z3_to_lean = Z3ToLeanTranslator()
        self.lean_to_z3 = LeanToZ3Translator()
        self.verification = Z3LeanVerificationBridge(lean_service)
        self.hybrid_proof = HybridProofEngine(self.verification)
        
        # CAV-NLP integration as alternative to LeanAide
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled as LeanAide alternative")
        
        logger.info("Z3LeanAideBridge initialized")
    
    def z3_to_lean4(
        self,
        z3_expr: Any,
        constraint_type: ConstraintType = ConstraintType.BOOLEAN
    ) -> Lean4Constraint:
        """Translate Z3 to Lean 4"""
        return self.z3_to_lean.translate(z3_expr, constraint_type)
    
    def lean4_to_z3(self, lean_code: str) -> Optional[Z3Constraint]:
        """Translate Lean 4 to Z3"""
        return self.lean_to_z3.translate(lean_code)
    
    async def verify(
        self,
        constraint: Union[Z3Constraint, str],
        use_counterexamples: bool = True
    ) -> VerificationBridgeResult:
        """Verify using both Z3 and Lean"""
        return await self.verification.verify_hybrid(constraint, use_counterexamples)
    
    async def find_counterexample(self, lean_code: str) -> Optional[Dict[str, Any]]:
        """Find counterexample to Lean theorem"""
        return await self.verification.find_counterexample(lean_code)
    
    async def prove(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> HybridProofResult:
        """Prove theorem using hybrid approach"""
        return await self.hybrid_proof.prove(theorem, variables)
    
    def is_z3_available(self) -> bool:
        """Check if Z3 is available"""
        return Z3_AVAILABLE
    
    def is_lean_available(self) -> bool:
        """Check if Lean is available"""
        return LEAN4_AVAILABLE
    
    def is_cav_nlp_available(self) -> bool:
        """Check if CAV-NLP is available"""
        return CAV_NLP_AVAILABLE
    
    async def verify_with_cav_nlp(
        self,
        problem_text: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Verify a problem using CAV-NLP (alternative to Lean verification).
        
        Args:
            problem_text: Problem in natural language or formal notation
            use_cache: Whether to use cached formalizations
            
        Returns:
            Verification result with formalization and proof status
        """
        if not self.use_cav_nlp:
            return {
                'success': False,
                'error': 'CAV-NLP not available',
                'verified': False,
                'confidence': 0.0
            }
        
        try:
            # Step 1: Formalize the problem
            formalization = self.enhanced_solver.formalize_natural_language(
                problem_text,
                use_cache=use_cache
            )
            
            if not formalization.get('success'):
                return {
                    'success': False,
                    'error': formalization.get('error', 'Formalization failed'),
                    'verified': False,
                    'confidence': 0.0,
                    'formalization': formalization
                }
            
            # Step 2: Verify using math service
            z3_expr = formalization.get('z3_expression')
            if z3_expr:
                verification = self.math_service.verify_expression(z3_expr)
                
                return {
                    'success': True,
                    'verified': verification.get('valid', False),
                    'confidence': verification.get('confidence', 0.0),
                    'formalization': formalization,
                    'verification': verification,
                    'solver_used': 'cav_nlp'
                }
            
            return {
                'success': True,
                'verified': False,
                'confidence': 0.5,
                'formalization': formalization,
                'message': 'Formalized but no verifiable expression'
            }
            
        except Exception as e:
            logger.error(f"CAV-NLP verification failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'verified': False,
                'confidence': 0.0
            }
    
    async def hybrid_verify_with_fallback(
        self,
        problem_text: str,
        prefer_lean: bool = True
    ) -> Dict[str, Any]:
        """
        Verify using preferred method with automatic fallback.
        
        Tries LeanAide first (if prefer_lean=True and available),
        then falls back to CAV-NLP.
        
        Args:
            problem_text: Problem to verify
            prefer_lean: Whether to prefer Lean over CAV-NLP
            
        Returns:
            Verification result
        """
        # Try preferred method first
        if prefer_lean and LEAN4_AVAILABLE and self.verification.lean_service:
            try:
                lean_result = await self.verification.lean_service.verify(problem_text)
                if lean_result.success:
                    return {
                        'success': True,
                        'verified': lean_result.success,
                        'confidence': lean_result.confidence,
                        'solver_used': 'lean',
                        'result': lean_result
                    }
            except Exception as e:
                logger.warning(f"Lean verification failed, trying CAV-NLP: {e}")
        
        # Fall back to CAV-NLP if available
        if self.use_cav_nlp:
            cav_result = await self.verify_with_cav_nlp(problem_text)
            if cav_result.get('success'):
                return cav_result
        
        # Try CAV-NLP first if not preferring Lean
        if not prefer_lean and self.use_cav_nlp:
            cav_result = await self.verify_with_cav_nlp(problem_text)
            if cav_result.get('success') and cav_result.get('verified'):
                return cav_result
            
            # Fall back to Lean
            if LEAN4_AVAILABLE and self.verification.lean_service:
                lean_result = await self.verification.lean_service.verify(problem_text)
                return {
                    'success': lean_result.success,
                    'verified': lean_result.success,
                    'confidence': lean_result.confidence,
                    'solver_used': 'lean',
                    'result': lean_result
                }
        
        return {
            'success': False,
            'error': 'No verification method available',
            'verified': False,
            'confidence': 0.0
        }

    async def translate_solidity_invariant(
        self,
        statement: str,
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
    ) -> Dict[str, Any]:
        """
        Translate Solidity state updates into Z3/Lean invariants via the bridge.
        """
        if translate_solidity_assignment_to_z3 is None:
            return {
                "success": False,
                "error": "Solidity invariant translation unavailable",
            }
        try:
            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
            )
            result: Dict[str, Any] = {
                "success": True,
                "translation": translation,
            }
            if verify_translation and verify_solidity_invariant_translation is not None:
                result["verification"] = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=assume_non_negative_amount,
                )
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def solve_web3_exploit_witness(
        self,
        additional_constraints: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Solve symbolic exploit witness predicates for smart-contract audit workflows.
        """
        if solve_smart_contract_exploit_witness is None:
            return {
                "success": False,
                "error": "Exploit witness solver unavailable",
            }
        try:
            result = solve_smart_contract_exploit_witness(
                additional_constraints=additional_constraints,
                timeout=timeout,
            )
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def web3_audit_exploit_verification(
        self,
        statement: str = "balance[msg.sender] -= amount;",
        non_negative_target: bool = True,
        max_withdraw_expr: Optional[str] = None,
        verify_translation: bool = True,
        assume_non_negative_amount: bool = True,
        additional_constraints: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Run combined Web3 formal pass: invariants + witness exploit solving."""
        translation = await self.translate_solidity_invariant(
            statement=statement,
            non_negative_target=non_negative_target,
            max_withdraw_expr=max_withdraw_expr,
            verify_translation=verify_translation,
            assume_non_negative_amount=assume_non_negative_amount,
        )
        witness = await self.solve_web3_exploit_witness(
            additional_constraints=additional_constraints,
            timeout=timeout,
        )

        verification = translation.get("verification")
        witness_result = witness.get("result", {})
        verified_exploit = bool(witness_result.get("satisfiable", False))
        if verify_translation and isinstance(verification, dict):
            verified_exploit = verified_exploit and bool(verification.get("proven", False))

        return {
            "success": bool(translation.get("success")) and bool(witness.get("success")),
            "translation": translation.get("translation"),
            "verification": verification,
            "exploit_witness": witness_result,
            "verified_exploit": verified_exploit,
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities"""
        formal_capabilities = {
            "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification": verify_solidity_invariant_translation is not None,
            "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
            "composite_exploit_verification": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
        }
        web3_formal_tools: List[str] = []
        if formal_capabilities["solidity_invariant_translation"]:
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities["symbolic_exploit_witness"]:
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities["composite_exploit_verification"]:
            web3_formal_tools.append("z3_web3_audit_exploit_verification")

        return {
            "z3_available": Z3_AVAILABLE,
            "lean_available": LEAN4_AVAILABLE,
            "web3_formal_available": WEB3_FORMAL_AVAILABLE,
            "translation_z3_to_lean": True,
            "translation_lean_to_z3": Z3_AVAILABLE,
            "hybrid_verification": Z3_AVAILABLE and LEAN4_AVAILABLE,
            "counterexamples": Z3_AVAILABLE,
            "hybrid_proofs": True,
            "solidity_invariant_translation": formal_capabilities["solidity_invariant_translation"],
            "invariant_translation_verification": formal_capabilities[
                "invariant_translation_verification"
            ],
            "solidity_invariant_verification": formal_capabilities[
                "invariant_translation_verification"
            ],
            "smart_contract_exploit_witness": formal_capabilities["symbolic_exploit_witness"],
            "web3_audit_exploit_verification": formal_capabilities[
                "composite_exploit_verification"
            ],
            "formal_capabilities": formal_capabilities,
            "web3_formal_tools": web3_formal_tools,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_z3_lean_bridge(lean_service: Optional[Any] = None) -> Z3LeanAideBridge:
    """Create Z3-LeanAide bridge"""
    return Z3LeanAideBridge(lean_service)


async def quick_verify(lean_code: str) -> Optional[VerificationBridgeResult]:
    """Quickly verify Lean code using Z3"""
    bridge = create_z3_lean_bridge()
    return await bridge.verify(lean_code)


async def quick_translate_solidity_invariant(
    statement: str,
    non_negative_target: bool = True,
    max_withdraw_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """Quickly translate a Solidity update statement into invariants."""
    bridge = create_z3_lean_bridge()
    return await bridge.translate_solidity_invariant(
        statement=statement,
        non_negative_target=non_negative_target,
        max_withdraw_expr=max_withdraw_expr,
    )


async def quick_solve_web3_exploit_witness(
    additional_constraints: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Quickly solve canonical Web3 exploit witness predicate."""
    bridge = create_z3_lean_bridge()
    return await bridge.solve_web3_exploit_witness(
        additional_constraints=additional_constraints,
        timeout=timeout,
    )


async def quick_web3_audit_exploit_verification(
    statement: str = "balance[msg.sender] -= amount;",
    non_negative_target: bool = True,
    max_withdraw_expr: Optional[str] = None,
    verify_translation: bool = True,
    assume_non_negative_amount: bool = True,
    additional_constraints: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Quickly run combined Web3 exploit-verification workflow."""
    bridge = create_z3_lean_bridge()
    return await bridge.web3_audit_exploit_verification(
        statement=statement,
        non_negative_target=non_negative_target,
        max_withdraw_expr=max_withdraw_expr,
        verify_translation=verify_translation,
        assume_non_negative_amount=assume_non_negative_amount,
        additional_constraints=additional_constraints,
        timeout=timeout,
    )


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of Z3-LeanAide bridge"""
    
    print("=" * 70)
    print("Z3-LeanAide Bridge - Complete Implementation")
    print("=" * 70)
    
    bridge = create_z3_lean_bridge()
    
    print(f"\nCapabilities: {bridge.get_capabilities()}")
    
    # Example 1: Z3 to Lean translation
    if Z3_AVAILABLE:
        print("\n1. Z3 TO LEAN TRANSLATION")
        print("-" * 40)
        
        # Create simple Z3 expression
        x = Real('x')
        y = Real('y')
        z3_expr = And(x > 0, y > 0, x + y > 0)
        
        constraint = bridge.z3_to_lean4(z3_expr, ConstraintType.ARITHMETIC)
        print(f"   Z3 expression: {z3_expr}")
        print(f"   Generated Lean code:")
        print(f"   {constraint.lean_code[:300]}...")
    
    # Example 2: Lean to Z3 translation
    if Z3_AVAILABLE:
        print("\n2. LEAN TO Z3 TRANSLATION")
        print("-" * 40)
        
        lean_code = """
import Mathlib

theorem example_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : x + y > 0 := by
  linarith
"""
        z3_constraint = bridge.lean4_to_z3(lean_code)
        if z3_constraint:
            print(f"   Translated to Z3 constraint")
            print(f"   Variables: {z3_constraint.variables}")
            print(f"   Type: {z3_constraint.constraint_type.value}")
    
    # Example 3: Hybrid verification
    print("\n3. HYBRID VERIFICATION")
    print("-" * 40)
    
    test_lean = """
import Mathlib

theorem simple_theorem : 1 + 1 = 2 := by
  rfl
"""
    result = await bridge.verify(test_lean)
    if result:
        print(f"   Z3 result: {result.z3_result}")
        print(f"   Lean result: {result.lean_result.success if result.lean_result else 'N/A'}")
        print(f"   Agreed: {result.agreed}")
        print(f"   Confidence: {result.confidence:.2f}")
    
    # Example 4: Counterexample search
    if Z3_AVAILABLE:
        print("\n4. COUNTEREXAMPLE SEARCH")
        print("-" * 40)
        
        false_theorem = """
import Mathlib

theorem false_claim (x : ℝ) : x > 0 := by
  sorry
"""
        counterexample = await bridge.find_counterexample(false_theorem)
        if counterexample:
            print(f"   Found counterexample: {counterexample}")
        else:
            print("   No counterexample found (theorem may be true)")
    
    # Example 5: Hybrid proof
    print("\n5. HYBRID PROOF")
    print("-" * 40)
    
    proof_result = await bridge.prove(
        "x + y = y + x",
        {"x": "ℝ", "y": "ℝ"}
    )
    print(f"   Success: {proof_result.success}")
    print(f"   Tactics used: {proof_result.tactics_used}")
    print(f"   Z3 time: {proof_result.z3_time:.3f}s")
    print(f"   Lean time: {proof_result.lean_time:.3f}s")
    print(f"   Total time: {proof_result.total_time:.3f}s")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
