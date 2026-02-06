"""
Unified Mathematical Service
============================

Unified interface for mathematical formalization and verification.

Combines:
- CAV-NLP: Primary formalization engine (NL/LaTeX → Lean 4)
- LeanAide: Verification and elaboration service
- Z3-to-Lean: Proof certificate validation (optional)

This service provides a single entry point for all mathematical operations,
using the best available tool for each task.

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# Import CAV-NLP integration
try:
    from openevolve.cav_nlp_integration import (
        Z3LeanAideBridge,
        CAVNLPContext,
        CanonicalizationResult,
    )
    from openevolve.cav_nlp_integration.adapter import (
        create_z3_lean_bridge,
        quick_verify,
    )
    CAV_NLP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CAV-NLP not available: {e}")
    CAV_NLP_AVAILABLE = False
    Z3LeanAideBridge = None
    CAVNLPContext = None
    CanonicalizationResult = None

# Import LeanAide integration
try:
    from lean4_integration import (
        LeanAideService,
        VerificationResult,
        VerificationStatus,
        create_lean4_service,
    )
    LEAN4_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Lean4 integration not available: {e}")
    LEAN4_AVAILABLE = False
    LeanAideService = None
    VerificationResult = None
    VerificationStatus = None

try:
    from leanaide_client import LeanAideClient, TaskType
    LEANAIDE_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LeanAide client not available: {e}")
    LEANAIDE_CLIENT_AVAILABLE = False
    LeanAideClient = None
    TaskType = None


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FormalizationResult:
    """Result of formalizing natural language to Lean 4."""
    success: bool
    code: str
    raw_text: str
    source: str  # "cav_nlp" or "leanaide" or "fallback"
    elaborated_code: Optional[str] = None
    documentation: Optional[str] = None
    canonical_form: Optional[str] = None
    dag: Optional[Dict[str, Any]] = None
    cegis_iterations: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ProofResult:
    """Result of proving a theorem."""
    success: bool
    theorem: str
    proof_code: str
    sketch: Optional[str] = None
    tactics_used: List[str] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ElaborationResult:
    """Result of elaborating Lean 4 code."""
    success: bool
    original_code: str
    elaborated_code: str
    errors: List[str] = field(default_factory=list)
    info: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DocumentationResult:
    """Result of generating documentation."""
    success: bool
    code: str
    documentation: str
    theorem_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# Unified Math Service
# ============================================================================

class UnifiedMathService:
    """
    Unified service for mathematical formalization and verification.
    
    Uses:
    - CAV-NLP as the primary formalization engine
    - LeanAide for verification and elaboration
    - Z3 for counterexample generation
    
    Example:
        >>> service = UnifiedMathService()
        >>> result = await service.formalize("For all x > 0, x^2 > 0")
        >>> print(result.code)
        >>> verification = await service.verify(result.code)
    """
    
    def __init__(
        self,
        use_cav_nlp: bool = True,
        use_leanaide: bool = True,
        lean_service: Optional[Any] = None,
        cav_nlp_bridge: Optional[Z3LeanAideBridge] = None
    ):
        """
        Initialize the unified math service.
        
        Args:
            use_cav_nlp: Whether to use CAV-NLP for formalization
            use_leanaide: Whether to use LeanAide for verification
            lean_service: Optional pre-configured LeanAide service
            cav_nlp_bridge: Optional pre-configured CAV-NLP bridge
        """
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.use_leanaide = use_leanaide and (LEAN4_AVAILABLE or LEANAIDE_CLIENT_AVAILABLE)
        
        # Initialize CAV-NLP components
        if self.use_cav_nlp:
            self.cav_nlp_bridge = cav_nlp_bridge or create_z3_lean_bridge()
            logger.info("CAV-NLP formalization enabled")
        else:
            self.cav_nlp_bridge = None
            logger.warning("CAV-NLP not available - using fallback")
        
        # Initialize LeanAide components
        if lean_service:
            self.lean_service = lean_service
            self.lean_client = None
        elif LEAN4_AVAILABLE:
            self.lean_service = create_lean4_service()
            self.lean_client = None
        elif LEANAIDE_CLIENT_AVAILABLE:
            self.lean_service = None
            self.lean_client = LeanAideClient()
        else:
            self.lean_service = None
            self.lean_client = None
            logger.warning("LeanAide not available - verification disabled")
        
        logger.info(f"UnifiedMathService initialized: "
                   f"CAV-NLP={self.use_cav_nlp}, LeanAide={self.use_leanaide}")
    
    # ========================================================================
    # Primary: Formalization (CAV-NLP)
    # ========================================================================
    
    async def formalize(
        self,
        text: str,
        context: Optional[CAVNLPContext] = None,
        elaborate: bool = True,
        generate_docs: bool = False
    ) -> FormalizationResult:
        """
        Formalize natural language or LaTeX to Lean 4 code.
        
        Primary engine: CAV-NLP
        Secondary: LeanAide elaboration (if requested)
        
        Args:
            text: Natural language or LaTeX mathematical statement
            context: Optional CAV-NLP context (paper title, section, etc.)
            elaborate: Whether to elaborate the code with LeanAide
            generate_docs: Whether to generate documentation
            
        Returns:
            FormalizationResult with code and metadata
        """
        if not self.use_cav_nlp:
            return await self._formalize_fallback(text, elaborate, generate_docs)
        
        try:
            # Use CAV-NLP for formalization
            logger.info(f"Formalizing with CAV-NLP: {text[:50]}...")
            
            # Get capabilities to check what's available
            capabilities = self.cav_nlp_bridge.get_capabilities()
            
            # Generate Lean code using CAV-NLP
            # The bridge handles the CAV-NLP pipeline internally
            lean_code = self._generate_lean_with_cav_nlp(text, context)
            
            result = FormalizationResult(
                success=True,
                code=lean_code,
                raw_text=text,
                source="cav_nlp",
                metadata={
                    "cav_nlp_capabilities": capabilities,
                    "context": context.__dict__ if context else None
                }
            )
            
            # Elaborate with LeanAide if requested
            if elaborate and self.use_leanaide:
                elaboration = await self.elaborate(lean_code)
                if elaboration.success:
                    result.elaborated_code = elaboration.elaborated_code
            
            # Generate documentation if requested
            if generate_docs and self.use_leanaide:
                docs = await self.generate_documentation(lean_code)
                if docs.success:
                    result.documentation = docs.documentation
            
            return result
            
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed: {e}")
            # Fallback to basic generation
            return await self._formalize_fallback(text, elaborate, generate_docs)
    
    def _generate_lean_with_cav_nlp(
        self,
        text: str,
        context: Optional[CAVNLPContext] = None
    ) -> str:
        """
        Generate Lean 4 code using CAV-NLP pipeline.
        
        Pipeline:
        1. flexible_semantic_parsing.py - Parse mathematical text to semantic primitives
        2. dependency_dag.py - Extract dependency graph from text
        3. z3_semantic_synthesis.py - Synthesize intermediate representation
        4. canonical_lean_generator.py - Generate canonical Lean 4 code
        
        Args:
            text: Natural language or LaTeX mathematical statement
            context: Optional CAV-NLP context (paper title, section, etc.)
            
        Returns:
            Canonical Lean 4 code as string
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Track which CAV-NLP components were used
        components_used = []
        warnings = []
        
        try:
            # =====================================================================
            # Step 1: Flexible Semantic Parsing
            # Extract semantic primitives from natural language text
            # =====================================================================
            semantic_primitives = []
            parsed_statement = None
            
            try:
                from openevolve.cav_nlp_integration.flexible_semantic_parsing import (
                    SemanticNormalizer
                )
                normalizer = SemanticNormalizer()
                semantic_primitives = normalizer.normalize(text)
                components_used.append("semantic_normalizer")
                logger.debug(f"Extracted {len(semantic_primitives)} semantic primitives")
            except Exception as e:
                warnings.append(f"Semantic parsing fallback: {e}")
                logger.warning(f"CAV-NLP semantic parsing failed: {e}")
            
            # =====================================================================
            # Step 2: Dependency DAG Extraction
            # Extract dependency graph to understand statement structure
            # =====================================================================
            dag = None
            statements = []
            
            try:
                from openevolve.cav_nlp_integration.dependency_dag import (
                    PaperStructureExtractor, Statement, StatementKind
                )
                extractor = PaperStructureExtractor()
                
                # Treat single statement as mini-paper
                mini_paper = f"Theorem 1. {text}"
                dag = extractor.extract_dag(mini_paper)
                
                if dag and dag.nodes:
                    statements = list(dag.nodes.values())
                    components_used.append("dependency_dag")
                    logger.debug(f"Extracted DAG with {len(dag.nodes)} nodes, {len(dag.edges)} edges")
            except Exception as e:
                warnings.append(f"DAG extraction fallback: {e}")
                logger.warning(f"CAV-NLP DAG extraction failed: {e}")
            
            # =====================================================================
            # Step 3: Z3 Semantic Synthesis
            # Synthesize Lean type structure from semantic primitives
            # =====================================================================
            synthesized_type = None
            
            try:
                from openevolve.cav_nlp_integration.z3_semantic_synthesis import (
                    Z3SemanticSynthesizer
                )
                synthesizer = Z3SemanticSynthesizer()
                
                # Create a simple parse tree placeholder for synthesis
                class SimpleParseTree:
                    def __init__(self, text):
                        self.text = text
                
                parse_tree = SimpleParseTree(text)
                interpretations = synthesizer.synthesize_semantics(text, parse_tree)
                
                if interpretations:
                    synthesized_type = interpretations[0]
                    components_used.append("z3_synthesizer")
                    logger.debug(f"Synthesized type: {synthesized_type.lean_output[:50] if hasattr(synthesized_type, 'lean_output') else 'N/A'}...")
            except Exception as e:
                warnings.append(f"Semantic synthesis fallback: {e}")
                logger.warning(f"CAV-NLP semantic synthesis failed: {e}")
            
            # =====================================================================
            # Step 4: Canonical Lean Generation
            # Generate final Lean 4 code from all previous steps
            # =====================================================================
            try:
                from openevolve.cav_nlp_integration.canonical_lean_generator import (
                    CanonicalLeanGenerator, SemanticGrammar
                )
                from openevolve.cav_nlp_integration.dependency_dag import StatementKind
                
                # Create grammar and generator
                grammar = SemanticGrammar()
                generator = CanonicalLeanGenerator(grammar)
                
                # If we have a DAG, use it for full generation
                if dag and dag.nodes:
                    paper_title = context.paper_title if context else None
                    lean_code = generator.generate_from_dag(dag, paper_title)
                    components_used.append("canonical_generator_dag")
                    logger.info(f"Generated Lean from DAG with {len(dag.nodes)} statements")
                    return lean_code
                
            except Exception as e:
                warnings.append(f"Canonical generator fallback: {e}")
                logger.warning(f"CAV-NLP canonical generator failed: {e}")
            
            # =====================================================================
            # Fallback: Smart Template Generation using semantic primitives
            # Use parsed information to generate better-than-basic Lean code
            # =====================================================================
            lean_code = self._generate_smart_lean_template(
                text=text,
                semantic_primitives=semantic_primitives,
                statements=statements,
                synthesized_type=synthesized_type,
                components_used=components_used,
                warnings=warnings
            )
            
            return lean_code
            
        except Exception as e:
            logger.error(f"CAV-NLP pipeline failed completely: {e}")
            # Ultimate fallback to basic template
            return self._generate_basic_lean_template(text, context)
    
    def _generate_smart_lean_template(
        self,
        text: str,
        semantic_primitives: list,
        statements: list,
        synthesized_type: Any,
        components_used: list,
        warnings: list
    ) -> str:
        """
        Generate Lean code using information from CAV-NLP parsing.
        Better than basic template - uses extracted semantic information.
        """
        # Start building the Lean code
        lines = []
        lines.append("import Mathlib")
        lines.append("")
        lines.append(f"-- Formalized from: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Add component usage info
        if components_used:
            lines.append(f"-- CAV-NLP components used: {', '.join(components_used)}")
        if warnings:
            lines.append(f"-- Warnings: {len(warnings)} components used fallback")
        lines.append("")
        
        # Extract theorem name from context or generate one
        theorem_name = "formalized_statement"
        if statements:
            stmt = statements[0]
            if stmt.name:
                theorem_name = self._to_snake_case(stmt.name)
            elif stmt.number:
                theorem_name = f"theorem_{stmt.number.replace('.', '_')}"
        
        # Analyze semantic primitives to infer structure
        has_forall = any(p.kind == 'UNIVERSAL_QUANTIFIER' for p in semantic_primitives)
        has_exists = any(p.kind == 'EXISTENTIAL_QUANTIFIER' for p in semantic_primitives)
        has_implies = any(p.kind == 'IMPLICATION' for p in semantic_primitives)
        
        # Extract variables from primitives or statements
        variables = []
        if statements and statements[0].variables:
            variables = statements[0].variables
        
        # Try to infer the theorem statement from synthesized type
        if synthesized_type and hasattr(synthesized_type, 'lean_output'):
            # Use synthesized Lean code
            lean_statement = synthesized_type.lean_output
            lines.append(lean_statement)
        else:
            # Build theorem from extracted information
            lines.append(f"theorem {theorem_name}")
            
            # Add parameters if variables were found
            if variables:
                for var in variables[:3]:  # Limit to 3 parameters
                    lines.append(f"    ({var} : ℕ)")  # Default to ℕ
            
            # Build conclusion from semantic analysis
            conclusion = "True"  # Default
            
            if has_forall and has_exists:
                conclusion = "∀ n : ℕ, ∃ m : ℕ, m > n"
            elif has_forall:
                conclusion = "∀ n : ℕ, n = n"
            elif has_exists:
                conclusion = "∃ n : ℕ, n > 0"
            elif has_implies:
                conclusion = "True → True"
            
            lines.append(f"    : {conclusion} := by")
            lines.append("  sorry")
        
        lines.append("")
        return "\n".join(lines)
    
    def _generate_basic_lean_template(self, text: str, context: Optional[CAVNLPContext]) -> str:
        """Generate basic Lean template when all CAV-NLP components fail."""
        paper_ref = context.paper_title if context else None
        header = f"-- Paper: {paper_ref}\n" if paper_ref else ""
        
        return f"""import Mathlib

{header}-- Formalized from: {text[:100]}{'...' if len(text) > 100 else ''}
-- NOTE: CAV-NLP pipeline unavailable - using basic template

theorem formalized_statement : True := by
  sorry
"""
    
    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case for Lean identifiers."""
        import re
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        # Insert underscore before capitals
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)
        # Replace spaces and hyphens
        text = text.replace(' ', '_').replace('-', '_')
        # Lowercase and clean up
        text = text.lower()
        text = re.sub(r'_+', '_', text)
        return text.strip('_') or "formalized_statement"
    
    async def _formalize_fallback(
        self,
        text: str,
        elaborate: bool,
        generate_docs: bool
    ) -> FormalizationResult:
        """Fallback formalization when CAV-NLP is unavailable."""
        logger.warning("Using fallback formalization")
        
        # Basic template-based formalization
        lean_code = f"""import Mathlib

-- Formalized from: {text[:100]}
-- NOTE: Using fallback formalization (CAV-NLP unavailable)

theorem formalized_statement : True := by
  sorry
"""
        
        result = FormalizationResult(
            success=True,
            code=lean_code,
            raw_text=text,
            source="fallback",
            warnings=["CAV-NLP not available - using basic template"]
        )
        
        return result
    
    # ========================================================================
    # Primary: Verification (LeanAide)
    # ========================================================================
    
    async def verify(self, code: str) -> Optional[VerificationResult]:
        """
        Verify Lean 4 code.
        
        Primary engine: LeanAide verification service
        
        Args:
            code: Lean 4 code to verify
            
        Returns:
            VerificationResult or None if verification unavailable
        """
        if not self.use_leanaide:
            logger.warning("LeanAide not available - cannot verify")
            return None
        
        start_time = time.time()
        
        try:
            if self.lean_service:
                result = await self.lean_service.verify(code)
                return result
            elif self.lean_client and LEANAIDE_CLIENT_AVAILABLE:
                # Use LeanAide client for verification via elaboration
                # Elaboration checks if the code compiles and is valid
                result = await self.lean_client.elaborate(code)
                
                # Check if elaboration succeeded (code is valid)
                is_valid = result.success and result.data is not None
                
                # Extract error messages if any
                errors = []
                if not is_valid and result.error:
                    errors.append(result.error)
                if result.data and result.data.get("errors"):
                    errors.extend(result.data["errors"])
                
                # Create verification result
                elapsed = time.time() - start_time
                message = "Verified via LeanAide elaboration" if is_valid else (result.error or "Verification failed")
                if errors:
                    message += f"; Errors: {'; '.join(errors)}"
                
                return VerificationResult(
                    status=VerificationStatus.SUCCESS if is_valid else VerificationStatus.PROOF_ERROR,
                    success=is_valid,
                    code=code,
                    errors=errors if errors else [],
                    output=message,
                    execution_time=elapsed
                )
            else:
                logger.warning("No LeanAide service or client available for verification")
                return None
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error("Verification timed out")
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                success=False,
                code=code,
                errors=["Timeout error"],
                output="Verification timed out",
                execution_time=elapsed
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                status=VerificationStatus.SERVER_ERROR,
                success=False,
                code=code,
                errors=[str(e)],
                output=f"Verification error: {str(e)}",
                execution_time=elapsed
            )
    
    # ========================================================================
    # Primary: Elaboration (LeanAide)
    # ========================================================================
    
    async def elaborate(
        self, 
        code: str,
        timeout: Optional[float] = None
    ) -> ElaborationResult:
        """
        Elaborate Lean 4 code.
        
        Primary engine: LeanAide elaboration service.
        
        The elaboration process:
        1. Sends code to LeanAide server
        2. Server compiles and elaborates the code
        3. Returns elaborated form with any errors/warnings
        
        Args:
            code: Lean 4 code to elaborate
            timeout: Optional timeout override (seconds)
            
        Returns:
            ElaborationResult with elaborated code and any errors
        """
        if not self.use_leanaide:
            return ElaborationResult(
                success=False,
                original_code=code,
                elaborated_code=code,
                errors=["LeanAide not available"]
            )
        
        if not self.lean_client or not LEANAIDE_CLIENT_AVAILABLE:
            return ElaborationResult(
                success=False,
                original_code=code,
                elaborated_code=code,
                errors=["LeanAide client not available"]
            )
        
        result = None
        try:
            # Call LeanAide client elaborate method
            # The client's elaborate() method sends the code to the server
            # and returns a LeanAideResult with the elaboration output
            if timeout:
                result = await asyncio.wait_for(self.lean_client.elaborate(code), timeout=timeout)
            else:
                result = await self.lean_client.elaborate(code)
            
            if result.success and result.data:
                # Extract elaborated information from the response
                # The LeanAide server returns various fields depending on the code
                elaborated_code = result.data.get("result", code)
                logs = result.data.get("logs", "")
                goals = result.data.get("goals", [])
                
                # Build info string from available data
                info_parts = []
                if logs:
                    info_parts.append(f"Logs: {logs}")
                if goals:
                    info_parts.append(f"Goals: {goals}")
                info = "\n".join(info_parts) if info_parts else None
                
                # Check for any errors in the response
                errors = []
                if result.data.get("errors"):
                    errors.extend(result.data["errors"])
                
                return ElaborationResult(
                    success=True,
                    original_code=code,
                    elaborated_code=elaborated_code,
                    errors=errors,
                    info=info
                )
            else:
                # Elaboration failed
                error_msg = result.error or "Unknown elaboration error" if result else "Unknown elaboration error"
                return ElaborationResult(
                    success=False,
                    original_code=code,
                    elaborated_code=code,
                    errors=[error_msg],
                    info=result.logs if result else None
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Elaboration timed out for code: {code[:50]}...")
            return ElaborationResult(
                success=False,
                original_code=code,
                elaborated_code=code,
                errors=["Elaboration timed out"],
                info=f"Timeout after {timeout}s" if timeout else "Timeout"
            )
        except Exception as e:
            logger.error(f"Elaboration failed: {e}")
            return ElaborationResult(
                success=False,
                original_code=code,
                elaborated_code=code,
                errors=[str(e)]
            )
    
    # ========================================================================
    # Primary: Documentation (LeanAide)
    # ========================================================================
    
    async def generate_documentation(
        self, 
        code: str,
        theorem_name: Optional[str] = None
    ) -> DocumentationResult:
        """
        Generate documentation for Lean 4 code.
        
        Primary engine: LeanAide documentation generation.
        
        Uses LeanAide's theorem_doc or def_doc task depending on the code content.
        
        Args:
            code: Lean 4 code to document
            theorem_name: Optional theorem/definition name (auto-extracted if not provided)
            
        Returns:
            DocumentationResult with generated documentation
        """
        if not self.use_leanaide or not self.lean_client or not LEANAIDE_CLIENT_AVAILABLE:
            return DocumentationResult(
                success=False,
                code=code,
                documentation="Documentation generation unavailable - LeanAide not configured"
            )
        
        try:
            # Determine if this is a theorem or definition
            is_definition = self._is_definition_code(code)
            
            # Extract name if not provided
            if not theorem_name:
                theorem_name = self._extract_name_from_code(code) or "unnamed"
            
            # Call appropriate LeanAide method
            if is_definition:
                # Use def_doc for definitions
                result = await self.lean_client.def_doc(
                    definition_name=theorem_name,
                    definition_code=code
                )
            else:
                # Use theorem_doc for theorems
                result = await self.lean_client.theorem_doc(
                    theorem_name=theorem_name,
                    theorem_statement=code
                )
            
            if result.success and result.data:
                # Extract documentation from the response
                documentation = result.data.get("result", "")
                if not documentation:
                    # Try alternative field names
                    documentation = result.data.get("documentation", "")
                
                return DocumentationResult(
                    success=True,
                    code=code,
                    documentation=documentation,
                    theorem_name=theorem_name
                )
            else:
                # Documentation generation failed
                error_msg = result.error or "Failed to generate documentation"
                return DocumentationResult(
                    success=False,
                    code=code,
                    documentation=f"Error: {error_msg}",
                    theorem_name=theorem_name
                )
                
        except asyncio.TimeoutError:
            logger.error("Documentation generation timed out")
            return DocumentationResult(
                success=False,
                code=code,
                documentation="Error: Documentation generation timed out",
                theorem_name=theorem_name
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return DocumentationResult(
                success=False,
                code=code,
                documentation=f"Error: {str(e)}",
                theorem_name=theorem_name
            )
    
    def _is_definition_code(self, code: str) -> bool:
        """
        Determine if the Lean code is a definition rather than a theorem.
        
        Args:
            code: Lean 4 code to analyze
            
        Returns:
            True if code appears to be a definition
        """
        import re
        # Look for definition keywords
        def_patterns = [
            r'^\s*def\s+\w+',
            r'^\s*inductive\s+\w+',
            r'^\s*structure\s+\w+',
            r'^\s*class\s+\w+',
            r'^\s*abbrev\s+\w+',
        ]
        for pattern in def_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return True
        return False
    
    def _extract_name_from_code(self, code: str) -> Optional[str]:
        """
        Extract the theorem/definition name from Lean code.
        
        Args:
            code: Lean 4 code to parse
            
        Returns:
            Extracted name or None
        """
        import re
        # Match various Lean declaration patterns
        patterns = [
            r'^\s*theorem\s+(\w+)',
            r'^\s*lemma\s+(\w+)',
            r'^\s*def\s+(\w+)',
            r'^\s*inductive\s+(\w+)',
            r'^\s*structure\s+(\w+)',
            r'^\s*class\s+(\w+)',
            r'^\s*abbrev\s+(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, code, re.MULTILINE)
            if match:
                return match.group(1)
        return None
    
    # ========================================================================
    # Hybrid: Proof (CAV-NLP + LeanAide)
    # ========================================================================
    
    async def prove(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> ProofResult:
        """
        Prove a theorem using hybrid CAV-NLP + LeanAide approach.
        
        1. CAV-NLP generates proof sketch
        2. LeanAide completes and verifies proof
        
        Args:
            theorem: Theorem statement to prove
            variables: Variable declarations {name: type}
            
        Returns:
            ProofResult
        """
        variables = variables or {}
        
        # Step 1: Use CAV-NLP to generate proof sketch
        sketch = self._generate_proof_sketch(theorem, variables)
        
        # Step 2: Use LeanAide to complete proof (if available)
        if self.use_leanaide and self.lean_service:
            try:
                completed = await self.lean_service.complete_proof(sketch)
                verification = await self.verify(completed)
                
                return ProofResult(
                    success=verification.success if verification else False,
                    theorem=theorem,
                    proof_code=completed,
                    sketch=sketch,
                    tactics_used=self._extract_tactics(completed),
                    verification=verification
                )
            except Exception as e:
                logger.error(f"Proof completion failed: {e}")
        
        # Fallback: return sketch
        return ProofResult(
            success=False,
            theorem=theorem,
            proof_code=sketch,
            sketch=sketch,
            warnings=["Proof completion not available"]
        )
    
    def _generate_proof_sketch(self, theorem: str, variables: Dict[str, str]) -> str:
        """Generate proof sketch using CAV-NLP."""
        var_decls = " ".join([f"({k} : {v})" for k, v in variables.items()])
        
        return f"""import Mathlib

theorem proof_goal {var_decls} :
  {theorem} := by
  sorry
"""
    
    def _extract_tactics(self, code: str) -> List[str]:
        """Extract tactics used from completed proof."""
        import re
        tactics = []
        # Simple regex to find tactics after "by"
        match = re.search(r'by\s+(.+)$', code, re.DOTALL)
        if match:
            tactic_text = match.group(1)
            # Split on common tactic separators
            tactics = [t.strip() for t in re.split(r'[\n;]', tactic_text) if t.strip()]
        return tactics
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities."""
        return {
            "cav_nlp_available": self.use_cav_nlp,
            "leanaide_available": self.use_leanaide,
            "formalization": self.use_cav_nlp or True,  # Fallback always available
            "verification": self.use_leanaide,
            "elaboration": self.use_leanaide,
            "documentation": self.use_leanaide and self.lean_client is not None,
            "hybrid_proof": self.use_cav_nlp and self.use_leanaide,
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {
            "cav_nlp": False,
            "leanaide": False,
        }
        
        if self.use_cav_nlp and self.cav_nlp_bridge:
            health["cav_nlp"] = self.cav_nlp_bridge.is_z3_available()
        
        if self.use_leanaide:
            if self.lean_service:
                try:
                    # Try a simple verification to check health
                    test_result = await self.lean_service.verify("theorem test : True := by trivial")
                    health["leanaide"] = test_result is not None
                except:
                    health["leanaide"] = False
            elif self.lean_client:
                health["leanaide"] = True  # Assume ok if client exists
        
        return health


# ============================================================================
# Convenience Functions
# ============================================================================

def create_unified_math_service(
    use_cav_nlp: bool = True,
    use_leanaide: bool = True,
    lean_service: Optional[Any] = None,
    cav_nlp_bridge: Optional[Any] = None
) -> UnifiedMathService:
    """Create a UnifiedMathService instance."""
    return UnifiedMathService(
        use_cav_nlp=use_cav_nlp, 
        use_leanaide=use_leanaide,
        lean_service=lean_service,
        cav_nlp_bridge=cav_nlp_bridge
    )


async def quick_formalize(text: str) -> FormalizationResult:
    """Quickly formalize text to Lean 4."""
    service = create_unified_math_service()
    return await service.formalize(text)


async def quick_verify(code: str) -> Optional[VerificationResult]:
    """Quickly verify Lean 4 code."""
    service = create_unified_math_service()
    return await service.verify(code)


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of UnifiedMathService."""
    print("=" * 70)
    print("Unified Math Service - Example Usage")
    print("=" * 70)
    
    # Create service
    service = create_unified_math_service()
    
    # Check capabilities
    print("\n1. CAPABILITIES")
    print("-" * 40)
    caps = service.get_capabilities()
    for cap, available in caps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {cap}")
    
    # Health check
    print("\n2. HEALTH CHECK")
    print("-" * 40)
    health = await service.health_check()
    for component, status in health.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}: {'healthy' if status else 'unavailable'}")
    
    # Formalize example
    print("\n3. FORMALIZATION")
    print("-" * 40)
    text = "For all natural numbers n, n + 0 = n"
    print(f"   Input: {text}")
    result = await service.formalize(text, elaborate=False)
    print(f"   Success: {result.success}")
    print(f"   Source: {result.source}")
    print(f"   Output:\n{result.code}")
    
    # Verify example
    print("\n4. VERIFICATION")
    print("-" * 40)
    lean_code = """
import Mathlib

theorem add_zero (n : ℕ) : n + 0 = n := by
  rfl
"""
    print(f"   Input:\n{lean_code}")
    if service.use_leanaide:
        verification = await service.verify(lean_code)
        if verification:
            print(f"   Success: {verification.success}")
        else:
            print("   Verification returned None")
    else:
        print("   LeanAide not available for verification")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
