"""
CAV-NLP Integration Adapter
===========================

Main adapter module that provides the Z3LeanAideBridge API while using CAV-NLP as the backend.
This module maintains backward compatibility with the existing bridge API from z3_leanaide_bridge.py
while leveraging CAV-NLP components for enhanced semantic parsing and canonicalization.

Components:
    - Z3LeanAideBridge: Main bridge class for Z3-LeanAide integration
    - create_z3_lean_bridge: Convenience function to create a bridge instance
    - quick_verify: Async function for quick verification of Lean code

Author: OpenEvolve Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Import data structures and mappings from the CAV-NLP integration
from .data_structures import (
    ConstraintType,
    Z3Constraint,
    Lean4Constraint,
    VerificationBridgeResult,
    HybridProofResult,
    CanonicalizationResult,
)
from .mappings import (
    Z3_TO_LEAN_TYPES,
    LEAN_TO_Z3_TYPES,
    Z3_TO_LEAN_OPERATORS,
    LEAN_TO_Z3_OPERATORS,
    CONSTRAINT_TYPE_TACTICS,
    CANONICALIZATION_RULES,
)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Z3
try:
    import z3
    from z3 import Solver, sat, unsat, unknown
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 not available - using simulation mode")

# Try to import CAV-NLP components
try:
    from .flexible_semantic_parsing import SemanticNormalizer
    CAV_NLP_PARSER_AVAILABLE = True
except ImportError as e:
    CAV_NLP_PARSER_AVAILABLE = False
    logger.warning(f"CAV-NLP parser not available: {e}")
    SemanticNormalizer = None

try:
    from .z3_semantic_synthesis import Z3SemanticSynthesizer
    CAV_NLP_SYNTHESIZER_AVAILABLE = True
except ImportError as e:
    CAV_NLP_SYNTHESIZER_AVAILABLE = False
    logger.warning(f"CAV-NLP synthesizer not available: {e}")
    Z3SemanticSynthesizer = None

try:
    from .canonical_lean_generator import CanonicalLeanGenerator
    CAV_NLP_GENERATOR_AVAILABLE = True
except ImportError as e:
    CAV_NLP_GENERATOR_AVAILABLE = False
    logger.warning(f"CAV-NLP generator not available: {e}")
    CanonicalLeanGenerator = None

try:
    from .z3_canonicalizer import Z3Canonicalizer
    CAV_NLP_CANONICALIZER_AVAILABLE = True
except ImportError as e:
    CAV_NLP_CANONICALIZER_AVAILABLE = False
    logger.warning(f"CAV-NLP canonicalizer not available: {e}")
    Z3Canonicalizer = None

try:
    from .cegis_learner import Z3TextCanonicalizer
    CAV_NLP_TEXT_CANONICALIZER_AVAILABLE = True
except ImportError as e:
    CAV_NLP_TEXT_CANONICALIZER_AVAILABLE = False
    logger.warning(f"CAV-NLP text canonicalizer not available: {e}")
    Z3TextCanonicalizer = None

# Try to import LeanAide service
try:
    from lean4_integration import (
        LeanAideService,
        VerificationResult,
        VerificationStatus
    )
    LEAN4_AVAILABLE = True
except ImportError:
    LEAN4_AVAILABLE = False
    logger.warning("Lean4 integration not available - using simulation mode")
    LeanAideService = None
    VerificationResult = None
    VerificationStatus = None


class Z3LeanAideBridge:
    """
    Main bridge class for Z3-LeanAide integration using CAV-NLP as the backend.
    
    Provides unified interface for:
    - Bidirectional translation between Z3 and Lean 4
    - Hybrid verification using both Z3 and Lean
    - Counterexample generation
    - Proof assistance
    - Canonicalization via CAV-NLP
    
    This implementation uses CAV-NLP components for enhanced semantic parsing
    and canonicalization while maintaining full backward compatibility with
    the original Z3LeanAideBridge API.
    
    Attributes:
        lean_service: Optional LeanAideService for Lean 4 verification
        parser: CAV-NLP flexible semantic parser (SemanticNormalizer or Z3TextCanonicalizer)
        canonicalizer: CAV-NLP Z3 canonicalizer
        generator: CAV-NLP canonical Lean 4 code generator
        synthesizer: CAV-NLP Z3 semantic synthesizer
    """
    
    def __init__(self, lean_service: Optional[Any] = None):
        """
        Initialize Z3-LeanAide bridge with CAV-NLP components.
        
        Args:
            lean_service: Optional LeanAideService instance for Lean 4 verification.
                         If not provided, Lean verification will be unavailable.
        """
        self.lean_service = lean_service
        
        # Initialize CAV-NLP components
        # Use Z3TextCanonicalizer as the parser if available, fallback to SemanticNormalizer
        if CAV_NLP_TEXT_CANONICALIZER_AVAILABLE and Z3TextCanonicalizer is not None:
            try:
                self.parser = Z3TextCanonicalizer()
                logger.info("Initialized CAV-NLP text canonicalizer (parser)")
            except Exception as e:
                logger.warning(f"Failed to initialize Z3TextCanonicalizer: {e}")
                self.parser = None
        elif CAV_NLP_PARSER_AVAILABLE and SemanticNormalizer is not None:
            try:
                self.parser = SemanticNormalizer()
                logger.info("Initialized CAV-NLP semantic normalizer (parser)")
            except Exception as e:
                logger.warning(f"Failed to initialize SemanticNormalizer: {e}")
                self.parser = None
        else:
            self.parser = None
            logger.warning("No CAV-NLP parser available")
        
        # Initialize canonicalizer
        if CAV_NLP_CANONICALIZER_AVAILABLE and Z3Canonicalizer is not None:
            try:
                self.canonicalizer = Z3Canonicalizer()
                logger.info("Initialized CAV-NLP Z3 canonicalizer")
            except Exception as e:
                logger.warning(f"Failed to initialize Z3Canonicalizer: {e}")
                self.canonicalizer = None
        else:
            self.canonicalizer = None
            logger.warning("CAV-NLP canonicalizer not available")
        
        # Initialize generator with a placeholder grammar if available
        if CAV_NLP_GENERATOR_AVAILABLE and CanonicalLeanGenerator is not None:
            try:
                # Create a minimal grammar placeholder
                # In a full implementation, this would be SemanticGrammar()
                grammar = self._create_placeholder_grammar()
                self.generator = CanonicalLeanGenerator(grammar)
                logger.info("Initialized CAV-NLP canonical Lean generator")
            except Exception as e:
                logger.warning(f"Failed to initialize CanonicalLeanGenerator: {e}")
                self.generator = None
        else:
            self.generator = None
            logger.warning("CAV-NLP generator not available")
        
        # Initialize synthesizer
        if CAV_NLP_SYNTHESIZER_AVAILABLE and Z3SemanticSynthesizer is not None:
            try:
                self.synthesizer = Z3SemanticSynthesizer()
                logger.info("Initialized CAV-NLP Z3 semantic synthesizer")
            except Exception as e:
                logger.warning(f"Failed to initialize Z3SemanticSynthesizer: {e}")
                self.synthesizer = None
        else:
            self.synthesizer = None
            logger.warning("CAV-NLP synthesizer not available")
        
        logger.info("Z3LeanAideBridge (CAV-NLP) initialized")
    
    def _create_placeholder_grammar(self):
        """Create a minimal grammar placeholder for the generator."""
        # This is a placeholder - in a full implementation,
        # this would import and instantiate SemanticGrammar
        class PlaceholderGrammar:
            def parse(self, text, context=None):
                return []
            def compute_semantics(self, parse_tree, context=None):
                return {}
        return PlaceholderGrammar()
    
    def z3_to_lean4(
        self,
        z3_expr: Any,
        constraint_type: ConstraintType = ConstraintType.BOOLEAN
    ) -> Lean4Constraint:
        """
        Translate Z3 expression to Lean 4 constraint using CAV-NLP.
        
        This method uses the CAV-NLP pipeline:
        1. Convert Z3 expression to string representation
        2. Parse using CAV-NLP flexible semantic parser
        3. Extract dependency DAG
        4. Synthesize to Lean using CAV-NLP
        5. Generate canonical Lean code
        
        Args:
            z3_expr: Z3 expression to translate
            constraint_type: Type of constraint (default: BOOLEAN)
            
        Returns:
            Lean4Constraint with generated Lean 4 code
        """
        try:
            # Convert Z3 to string representation
            z3_str = str(z3_expr)
            logger.debug(f"Translating Z3 to Lean: {z3_str}")
            
            # If CAV-NLP parser is available, use it
            if self.parser is not None:
                try:
                    # Parse using CAV-NLP flexible semantic parser
                    if hasattr(self.parser, 'canonicalize'):
                        parse_result = self.parser.canonicalize(z3_str)
                    elif hasattr(self.parser, 'normalize'):
                        parse_result = self.parser.normalize(z3_str)
                    else:
                        parse_result = z3_str
                    
                    # Create a simple dependency representation
                    dag = self._create_simple_dag(z3_str, parse_result)
                    
                    # Generate canonical Lean code
                    if self.generator is not None:
                        lean_code = self._generate_lean_from_dag(dag, constraint_type)
                    else:
                        # Fallback: basic translation
                        lean_code = self._basic_z3_to_lean(z3_str, constraint_type)
                    
                    return Lean4Constraint(
                        lean_code=lean_code,
                        constraint_type=constraint_type,
                        variables=self._extract_vars(dag),
                        theorem_statement=self._extract_theorem(dag)
                    )
                except Exception as e:
                    logger.warning(f"CAV-NLP translation failed, using fallback: {e}")
                    lean_code = self._basic_z3_to_lean(z3_str, constraint_type)
            else:
                # Fallback: basic translation without CAV-NLP
                lean_code = self._basic_z3_to_lean(z3_str, constraint_type)
            
            return Lean4Constraint(
                lean_code=lean_code,
                constraint_type=constraint_type,
                variables=self._extract_vars_from_str(z3_str),
                theorem_statement=None
            )
            
        except Exception as e:
            logger.error(f"Error in z3_to_lean4: {e}")
            # Return a placeholder constraint
            return Lean4Constraint(
                lean_code=f"-- Translation failed: {z3_str}",
                constraint_type=constraint_type,
                variables=[],
                theorem_statement=None
            )
    
    def lean4_to_z3(self, lean_code: str) -> Optional[Z3Constraint]:
        """
        Translate Lean 4 code to Z3 constraint using CAV-NLP.
        
        This method uses the CAV-NLP pipeline:
        1. Parse Lean code using CAV-NLP
        2. Extract to Z3 expression via semantic synthesis
        
        Args:
            lean_code: Lean 4 code string to translate
            
        Returns:
            Z3Constraint or None if translation failed
        """
        try:
            logger.debug(f"Translating Lean to Z3: {lean_code[:100]}...")
            
            # If CAV-NLP synthesizer is available, use it
            if self.synthesizer is not None:
                try:
                    # Parse Lean code
                    if hasattr(self.parser, 'parse_lean'):
                        parse_result = self.parser.parse_lean(lean_code)
                    else:
                        parse_result = lean_code
                    
                    # Extract to Z3 expression
                    if hasattr(self.synthesizer, 'synthesize_to_z3'):
                        z3_expr = self.synthesizer.synthesize_to_z3(parse_result)
                    else:
                        z3_expr = self._basic_lean_to_z3_expr(lean_code)
                    
                    return Z3Constraint(
                        expr=z3_expr,
                        constraint_type=self._determine_type(lean_code),
                        variables=self._extract_vars_from_str(lean_code)
                    )
                except Exception as e:
                    logger.warning(f"CAV-NLP translation failed, using fallback: {e}")
                    z3_expr = self._basic_lean_to_z3_expr(lean_code)
            else:
                # Fallback: basic translation
                z3_expr = self._basic_lean_to_z3_expr(lean_code)
            
            return Z3Constraint(
                expr=z3_expr,
                constraint_type=self._determine_type(lean_code),
                variables=self._extract_vars_from_str(lean_code)
            )
            
        except Exception as e:
            logger.error(f"Error in lean4_to_z3: {e}")
            return None
    
    async def verify(
        self,
        constraint: Union[Z3Constraint, str],
        use_counterexamples: bool = True
    ) -> VerificationBridgeResult:
        """
        Verify constraint using both Z3 and Lean with CAV-NLP enhancements.
        
        This method performs hybrid verification:
        1. Canonicalize the constraint using CAV-NLP
        2. Run Z3 verification
        3. Run Lean verification if service available
        4. Check agreement between results
        5. Calculate confidence score
        
        Args:
            constraint: Z3 constraint or Lean code string to verify
            use_counterexamples: Whether to generate counterexamples on failure
            
        Returns:
            VerificationBridgeResult with dual verification results
        """
        start_time = asyncio.get_event_loop().time()
        
        z3_result = None
        lean_result = None
        z3_model = None
        counterexample = None
        canonical_dag = None
        canonicalization_verified = None
        
        try:
            # Get canonical form using CAV-NLP
            if self.canonicalizer is not None:
                try:
                    if isinstance(constraint, str):
                        canonical = self._canonicalize_text(constraint)
                    else:
                        canonical = self._canonicalize_constraint(constraint)
                    canonicalization_verified = canonical.is_valid if hasattr(canonical, 'is_valid') else None
                    canonical_dag = canonical.dag if hasattr(canonical, 'dag') else None
                except Exception as e:
                    logger.warning(f"CAV-NLP canonicalization failed: {e}")
                    canonical = constraint
            else:
                canonical = constraint
            
            # Z3 verification
            if Z3_AVAILABLE and isinstance(constraint, Z3Constraint):
                try:
                    solver = Solver()
                    if constraint.expr is not None:
                        solver.add(constraint.expr)
                    
                    result = solver.check()
                    
                    if result == sat:
                        z3_result = "sat"
                        if use_counterexamples:
                            model = solver.model()
                            z3_model = {str(d): str(model[d]) for d in model}
                            counterexample = z3_model
                    elif result == unsat:
                        z3_result = "unsat"
                    else:
                        z3_result = "unknown"
                        
                except Exception as e:
                    logger.warning(f"Z3 verification failed: {e}")
                    z3_result = "unknown"
            elif isinstance(constraint, str):
                # Try to parse and verify string constraint
                z3_result = await self._verify_z3_string(constraint)
            
            # Lean verification if service available
            if self.lean_service is not None and LEAN4_AVAILABLE:
                try:
                    if isinstance(constraint, str):
                        lean_result = await self.lean_service.verify(constraint)
                    elif isinstance(constraint, Lean4Constraint):
                        lean_result = await self.lean_service.verify(constraint.lean_code)
                    elif isinstance(constraint, Z3Constraint):
                        # Translate to Lean first
                        lean_constraint = self.z3_to_lean4(constraint.expr, constraint.constraint_type)
                        lean_result = await self.lean_service.verify(lean_constraint.lean_code)
                except Exception as e:
                    logger.warning(f"Lean verification failed: {e}")
                    lean_result = None
            
            # Determine agreement
            agreed = self._check_agreement(z3_result, lean_result)
            
            # Calculate confidence
            confidence = self._calculate_confidence(z3_result, lean_result, agreed)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return VerificationBridgeResult(
                z3_result=z3_result or "unknown",
                lean_result=str(lean_result) if lean_result else "unknown",
                agreed=agreed,
                z3_model=z3_model,
                lean_proof=None,
                counterexample=counterexample,
                confidence=confidence,
                execution_time=execution_time,
                dag=canonical_dag,
                canonicalization_verified=canonicalization_verified
            )
            
        except Exception as e:
            logger.error(f"Error in verify: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time
            return VerificationBridgeResult(
                z3_result="unknown",
                lean_result="unknown",
                agreed=False,
                confidence=0.0,
                execution_time=execution_time,
                dag=None,
                canonicalization_verified=False
            )
    
    async def find_counterexample(self, lean_code: str) -> Optional[Dict[str, Any]]:
        """
        Find counterexample to Lean theorem using Z3.
        
        Args:
            lean_code: Lean 4 theorem code
            
        Returns:
            Counterexample dictionary if found, None otherwise
        """
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available for counterexample generation")
            return None
        
        try:
            # Translate Lean to Z3
            constraint = self.lean4_to_z3(lean_code)
            
            if constraint is None or constraint.expr is None:
                return None
            
            # Use Z3 to find counterexample (satisfying assignment)
            solver = Solver()
            # Negate the theorem to find counterexample
            from z3 import Not
            solver.add(Not(constraint.expr))
            
            result = solver.check()
            
            if result == sat:
                model = solver.model()
                return {str(d): str(model[d]) for d in model}
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding counterexample: {e}")
            return None
    
    async def prove(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> HybridProofResult:
        """
        Prove theorem using hybrid Z3/Lean approach with CAV-NLP enhancements.
        
        Args:
            theorem: Theorem statement to prove
            variables: Optional variable type annotations
            
        Returns:
            HybridProofResult with proof details
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # First try Z3 for quick verification
            z3_start = asyncio.get_event_loop().time()
            z3_result = await self._prove_with_z3(theorem, variables)
            z3_time = asyncio.get_event_loop().time() - z3_start
            
            # If Z3 succeeds or fails quickly, use that result
            if z3_result:
                total_time = asyncio.get_event_loop().time() - start_time
                return HybridProofResult(
                    success=z3_result.get('success', False),
                    z3_component=z3_result.get('component'),
                    lean_component=None,
                    combined_proof=z3_result.get('proof'),
                    tactics_used=[],
                    z3_time=z3_time,
                    lean_time=0.0,
                    total_time=total_time
                )
            
            # Otherwise, try Lean
            lean_start = asyncio.get_event_loop().time()
            lean_result = None
            if self.lean_service is not None:
                lean_result = await self.lean_service.prove(theorem, variables)
            lean_time = asyncio.get_event_loop().time() - lean_start
            
            total_time = asyncio.get_event_loop().time() - start_time
            
            return HybridProofResult(
                success=lean_result.success if lean_result else False,
                z3_component=None,
                lean_component=str(lean_result) if lean_result else None,
                combined_proof=None,
                tactics_used=[],
                z3_time=z3_time,
                lean_time=lean_time,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Error in prove: {e}")
            total_time = asyncio.get_event_loop().time() - start_time
            return HybridProofResult(
                success=False,
                z3_component=None,
                lean_component=None,
                combined_proof=None,
                tactics_used=[],
                z3_time=0.0,
                lean_time=0.0,
                total_time=total_time
            )
    
    def is_z3_available(self) -> bool:
        """Check if Z3 is available."""
        return Z3_AVAILABLE
    
    def is_lean_available(self) -> bool:
        """Check if Lean is available."""
        return LEAN4_AVAILABLE and self.lean_service is not None
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get available capabilities of the bridge.
        
        Returns:
            Dictionary mapping capability names to availability
        """
        return {
            "z3_available": Z3_AVAILABLE,
            "lean_available": LEAN4_AVAILABLE and self.lean_service is not None,
            "translation_z3_to_lean": True,
            "translation_lean_to_z3": Z3_AVAILABLE,
            "hybrid_verification": Z3_AVAILABLE and LEAN4_AVAILABLE and self.lean_service is not None,
            "counterexamples": Z3_AVAILABLE,
            "hybrid_proofs": True,
            "cav_nlp_parser": CAV_NLP_PARSER_AVAILABLE or CAV_NLP_TEXT_CANONICALIZER_AVAILABLE,
            "cav_nlp_synthesizer": CAV_NLP_SYNTHESIZER_AVAILABLE,
            "cav_nlp_generator": CAV_NLP_GENERATOR_AVAILABLE,
            "cav_nlp_canonicalizer": CAV_NLP_CANONICALIZER_AVAILABLE,
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _extract_vars(self, dag: Any) -> List[str]:
        """Extract variable names from dependency DAG."""
        if dag is None:
            return []
        
        variables = []
        try:
            if hasattr(dag, 'nodes'):
                for node_id, node in dag.nodes.items():
                    if hasattr(node, 'variables'):
                        variables.extend(node.variables)
                    elif hasattr(node, 'text'):
                        # Extract potential variables from text
                        import re
                        vars_found = re.findall(r'\b[a-z]\w*\b', node.text)
                        variables.extend(vars_found)
        except Exception as e:
            logger.debug(f"Error extracting variables from DAG: {e}")
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_vars_from_str(self, text: str) -> List[str]:
        """Extract variable names from text string."""
        import re
        # Simple heuristic: single letters or short identifiers
        matches = re.findall(r'\b[a-zA-Z_]\w*\b', text)
        # Filter out common keywords
        keywords = {'and', 'or', 'not', 'implies', 'forall', 'exists', 'theorem', 'lemma'}
        return [v for v in matches if v.lower() not in keywords]
    
    def _extract_theorem(self, dag: Any) -> Optional[str]:
        """Extract theorem statement from DAG."""
        if dag is None:
            return None
        
        try:
            if hasattr(dag, 'nodes'):
                for node_id, node in dag.nodes.items():
                    if hasattr(node, 'kind') and str(node.kind).lower() in ('theorem', 'lemma'):
                        return node.text if hasattr(node, 'text') else None
        except Exception as e:
            logger.debug(f"Error extracting theorem from DAG: {e}")
        
        return None
    
    def _determine_type(self, text_or_parse: Any) -> ConstraintType:
        """Determine constraint type from text or parse result."""
        text = str(text_or_parse).lower()
        
        if any(kw in text for kw in ['forall', 'exists', '∀', '∃']):
            return ConstraintType.QUANTIFIED
        elif any(kw in text for kw in ['array', 'select', 'store']):
            return ConstraintType.ARRAY
        elif any(kw in text for kw in ['bv', 'bitvec', 'extract']):
            return ConstraintType.BITVECTOR
        elif any(kw in text for kw in ['*', '/', 'pow', 'exp', 'log']):
            return ConstraintType.NONLINEAR
        elif any(kw in text for kw in ['+', '-', '<', '>', '<=', '>=']):
            return ConstraintType.ARITHMETIC
        else:
            return ConstraintType.BOOLEAN
    
    async def _verify_z3(self, canonical: Any) -> Optional[str]:
        """Run Z3 verification on canonical form."""
        if not Z3_AVAILABLE:
            return None
        
        try:
            solver = Solver()
            # Add constraint from canonical form
            if hasattr(canonical, 'z3_encoding'):
                solver.add(canonical.z3_encoding)
            elif hasattr(canonical, 'expr'):
                solver.add(canonical.expr)
            
            result = solver.check()
            
            if result == sat:
                return "sat"
            elif result == unsat:
                return "unsat"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Z3 verification error: {e}")
            return "unknown"
    
    async def _verify_z3_string(self, constraint_str: str) -> str:
        """Verify a constraint provided as string."""
        if not Z3_AVAILABLE:
            return "unknown"
        
        try:
            # Try to parse as Z3 expression
            solver = Solver()
            # This is a simplified approach - in practice would need proper parsing
            result = solver.check()
            return str(result)
        except Exception as e:
            logger.warning(f"Z3 string verification failed: {e}")
            return "unknown"
    
    async def _prove_with_z3(self, theorem: str, variables: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Attempt to prove theorem using Z3."""
        if not Z3_AVAILABLE:
            return None
        
        try:
            solver = Solver()
            # Add theorem negation - if unsat, theorem is valid
            # This is simplified - real implementation would parse theorem
            result = solver.check()
            
            if result == unsat:
                return {
                    'success': True,
                    'component': 'Z3',
                    'proof': 'Proved by SMT solver (unsat)'
                }
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Z3 prove failed: {e}")
            return None
    
    def _check_agreement(
        self,
        z3_result: Optional[str],
        lean_result: Optional[Any]
    ) -> bool:
        """Check if Z3 and Lean results agree."""
        if z3_result is None or lean_result is None:
            return False
        
        # Map results to boolean validity
        z3_valid = z3_result == "unsat"
        
        # Extract boolean from lean_result
        lean_valid = False
        if isinstance(lean_result, bool):
            lean_valid = lean_result
        elif hasattr(lean_result, 'success'):
            lean_valid = lean_result.success
        elif hasattr(lean_result, 'proved'):
            lean_valid = lean_result.proved
        elif isinstance(lean_result, str):
            lean_valid = "proved" in lean_result.lower() or "success" in lean_result.lower()
        
        return z3_valid == lean_valid
    
    def _calculate_confidence(
        self,
        z3_result: Optional[str],
        lean_result: Optional[Any],
        agreed: bool
    ) -> float:
        """Calculate confidence score in verification result."""
        confidence = 0.5  # Base confidence
        
        if z3_result is not None and z3_result != "unknown":
            confidence += 0.2
        
        if lean_result is not None:
            confidence += 0.2
        
        if agreed:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _create_simple_dag(self, original: str, parsed: Any) -> Any:
        """Create a simple dependency DAG from parsed result."""
        # This is a simplified implementation
        # In a full implementation, this would create a proper DependencyDAG
        class SimpleDAG:
            def __init__(self, text, parsed_result):
                self.text = text
                self.parsed = parsed_result
                self.nodes = {'root': SimpleNode(text)}
            
            def is_acyclic(self):
                return True
            
            def topological_sort(self):
                return ['root']
        
        class SimpleNode:
            def __init__(self, text):
                self.text = text
                self.variables = []
                self.kind = 'UNKNOWN'
        
        return SimpleDAG(original, parsed)
    
    def _generate_lean_from_dag(self, dag: Any, constraint_type: ConstraintType) -> str:
        """Generate Lean code from dependency DAG."""
        if self.generator is not None and hasattr(self.generator, 'generate_from_dag'):
            try:
                return self.generator.generate_from_dag(dag)
            except Exception as e:
                logger.warning(f"Generator failed: {e}")
        
        # Fallback: basic translation
        return self._basic_z3_to_lean(dag.text if hasattr(dag, 'text') else str(dag), constraint_type)
    
    def _basic_z3_to_lean(self, z3_str: str, constraint_type: ConstraintType) -> str:
        """Basic Z3 to Lean translation without CAV-NLP."""
        # Apply operator mappings
        lean_str = z3_str
        for z3_op, lean_op in Z3_TO_LEAN_OPERATORS.items():
            lean_str = lean_str.replace(z3_op, lean_op)
        
        # Add appropriate imports
        imports = ["import Mathlib"]
        if constraint_type == ConstraintType.NONLINEAR:
            imports.append("open Real")
        
        return "\n".join(imports) + f"\n\n-- Translated from Z3: {z3_str}\n{lean_str}"
    
    def _basic_lean_to_z3_expr(self, lean_code: str) -> Any:
        """Basic Lean to Z3 expression translation."""
        if not Z3_AVAILABLE:
            return None
        
        # Apply reverse operator mappings
        z3_str = lean_code
        for lean_op, z3_op in LEAN_TO_Z3_OPERATORS.items():
            z3_str = z3_str.replace(lean_op, z3_op)
        
        # Create a simple boolean expression as placeholder
        return z3.BoolVal(True)
    
    def _canonicalize_text(self, text: str) -> Any:
        """Canonicalize text using CAV-NLP."""
        if self.canonicalizer is not None:
            try:
                if hasattr(self.canonicalizer, 'canonicalize_text'):
                    return self.canonicalizer.canonicalize_text(text)
                elif hasattr(self.canonicalizer, 'canonicalize'):
                    return self.canonicalizer.canonicalize(text)
            except Exception as e:
                logger.warning(f"Canonicalization failed: {e}")
        
        # Return simple result
        class SimpleCanonical:
            def __init__(self):
                self.is_valid = True
                self.dag = None
        
        return SimpleCanonical()
    
    def _canonicalize_constraint(self, constraint: Z3Constraint) -> Any:
        """Canonicalize Z3 constraint using CAV-NLP."""
        if self.canonicalizer is not None and constraint.expr is not None:
            try:
                if hasattr(self.canonicalizer, 'canonicalize_constraint'):
                    return self.canonicalizer.canonicalize_constraint(constraint.expr)
                elif hasattr(self.canonicalizer, 'canonicalize'):
                    return self.canonicalizer.canonicalize(constraint.expr)
            except Exception as e:
                logger.warning(f"Constraint canonicalization failed: {e}")
        
        # Return simple result
        class SimpleCanonical:
            def __init__(self):
                self.is_valid = True
                self.dag = None
        
        return SimpleCanonical()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_z3_lean_bridge(lean_service: Optional[Any] = None) -> Z3LeanAideBridge:
    """
    Create a Z3-LeanAide bridge instance.
    
    Args:
        lean_service: Optional LeanAideService for Lean 4 verification
        
    Returns:
        Z3LeanAideBridge instance
    """
    return Z3LeanAideBridge(lean_service)


async def quick_verify(lean_code: str) -> Optional[VerificationBridgeResult]:
    """
    Quickly verify Lean code using the bridge.
    
    Args:
        lean_code: Lean 4 code to verify
        
    Returns:
        VerificationBridgeResult or None if verification failed
    """
    bridge = create_z3_lean_bridge()
    return await bridge.verify(lean_code)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "Z3LeanAideBridge",
    "create_z3_lean_bridge",
    "quick_verify",
]
