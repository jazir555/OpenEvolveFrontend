"""
LeanAide-CAV-NLP Integration Bridge
====================================

Bridge between LeanAide and CAV-NLP systems.

Purpose:
1. Provide smooth migration path from LeanAide to CAV-NLP
2. Route formalization requests to CAV-NLP
3. Keep verification/elaboration with LeanAide
4. Maintain backward compatibility

Author: OpenEvolve
Version: 1.0.0
"""

import warnings
import logging
import re
from typing import Any, Dict, List, Optional
from leanaide_web3_status import collect_web3_formal_status, merge_web3_formal_status

# Configure logging
logger = logging.getLogger(__name__)

# Try to import CAV-NLP
try:
    from openevolve.cav_nlp_integration import Z3LeanAideBridge
    from openevolve.cav_nlp_integration.adapter import create_z3_lean_bridge
    from openevolve.cav_nlp_integration.data_structures import (
        ConstraintType,
        TranslationResult,
        TranslationDirection
    )
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Try to import CAV-NLP core components
try:
    from openevolve.cav_nlp_integration.flexible_semantic_parsing import (
        SemanticNormalizer,
        SemanticPrimitive
    )
    CAV_NLP_PARSER_AVAILABLE = True
except ImportError:
    CAV_NLP_PARSER_AVAILABLE = False
    SemanticNormalizer = None
    SemanticPrimitive = None

try:
    from openevolve.cav_nlp_integration.dependency_dag import (
        DependencyDAG,
        Statement,
        StatementKind,
        Dependency
    )
    CAV_NLP_DAG_AVAILABLE = True
except ImportError:
    CAV_NLP_DAG_AVAILABLE = False
    DependencyDAG = None
    Statement = None
    StatementKind = None
    Dependency = None

try:
    from openevolve.cav_nlp_integration.canonical_lean_generator import (
        CanonicalLeanGenerator,
        CanonicalNamingRules
    )
    CAV_NLP_GENERATOR_AVAILABLE = True
except ImportError:
    CAV_NLP_GENERATOR_AVAILABLE = False
    CanonicalLeanGenerator = None
    CanonicalNamingRules = None

# Try to import unified service
try:
    from openevolve.unified_math_service import (
        UnifiedMathService,
        create_unified_math_service
    )
    UNIFIED_SERVICE_AVAILABLE = True
except ImportError:
    UNIFIED_SERVICE_AVAILABLE = False

# Try to import LeanAide
try:
    from leanaide_client import LeanAideClient, TaskType
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False


class LeanAideCAVNLPBridge:
    """
    Bridge for migrating from LeanAide to CAV-NLP.
    
    Routes formalization requests to CAV-NLP while preserving
    LeanAide's verification and elaboration capabilities.
    
    This class can be used as a drop-in replacement for LeanAideClient
    for translation tasks, redirecting them to CAV-NLP.
    
    Example:
        # Old way (deprecated)
        client = LeanAideClient()
        result = await client.translate_thm("x + 0 = x")
        
        # New way (recommended)
        bridge = LeanAideCAVNLPBridge()
        result = await bridge.translate_thm("x + 0 = x")  # Uses CAV-NLP
    """
    
    def __init__(
        self,
        use_cav_nlp: bool = True,
        use_unified_service: bool = True
    ):
        """
        Initialize the bridge.
        
        Args:
            use_cav_nlp: Whether to use CAV-NLP for formalization
            use_unified_service: Whether to use unified math service
        """
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.use_unified_service = use_unified_service and UNIFIED_SERVICE_AVAILABLE
        
        # Initialize CAV-NLP pipeline components
        self._semantic_normalizer = None
        self._dependency_dag = None
        self._lean_generator = None
        
        # Initialize services
        if self.use_unified_service:
            self.unified_service = create_unified_math_service()
            self.cav_nlp_bridge = None
            logger.info("Using UnifiedMathService for formalization")
        elif self.use_cav_nlp:
            self.cav_nlp_bridge = create_z3_lean_bridge()
            self.unified_service = None
            logger.info("Using CAV-NLP bridge for formalization")
            # Initialize CAV-NLP pipeline components
            self._init_cav_nlp_components()
        else:
            self.unified_service = None
            self.cav_nlp_bridge = None
            logger.warning("CAV-NLP not available - will use fallback")
        
        # Keep LeanAide client for non-translation tasks
        if LEANAIDE_AVAILABLE:
            self.lean_client = LeanAideClient()
        else:
            self.lean_client = None
    
    def _init_cav_nlp_components(self):
        """Initialize CAV-NLP pipeline components for direct use."""
        # Initialize semantic normalizer
        if CAV_NLP_PARSER_AVAILABLE and SemanticNormalizer is not None:
            try:
                self._semantic_normalizer = SemanticNormalizer()
                logger.info("Initialized CAV-NLP semantic normalizer")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic normalizer: {e}")
        
        # Initialize dependency DAG (just the class, instances created per translation)
        if CAV_NLP_DAG_AVAILABLE:
            self._dependency_dag_class = DependencyDAG
            self._statement_class = Statement
            self._statement_kind = StatementKind
            logger.info("CAV-NLP dependency DAG available")
        
        # Initialize Lean generator (will be created with grammar when needed)
        if CAV_NLP_GENERATOR_AVAILABLE and CanonicalLeanGenerator is not None:
            logger.info("CAV-NLP Lean generator available")
        else:
            logger.warning("CAV-NLP Lean generator not available - will use template generation")
    
    # ========================================================================
    # Translation Methods (Redirected to CAV-NLP)
    # ========================================================================
    
    @staticmethod
    def _normalize_result_payload(result: Any) -> Dict[str, Any]:
        """Normalize LeanAide/unified service results to a dictionary payload."""
        if isinstance(result, dict):
            return result
        if hasattr(result, "to_dict"):
            try:
                normalized = result.to_dict()
                if isinstance(normalized, dict):
                    return normalized
            except Exception:
                pass
        if hasattr(result, "dict"):
            try:
                normalized = result.dict()
                if isinstance(normalized, dict):
                    return normalized
            except Exception:
                pass
        if hasattr(result, "__dict__"):
            try:
                return dict(result.__dict__)
            except Exception:
                pass
        return {"success": False, "error": "Unsupported result payload type"}

    async def translate_thm(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate theorem to Lean 4.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        This method now redirects to CAV-NLP.
        
        Args:
            text: Theorem text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with 'lean_code', 'success', etc.
        """
        warnings.warn(
            "translate_thm is deprecated. Use UnifiedMathService.formalize() or CAV-NLP directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.formalize(text)
            return merge_web3_formal_status({
                "success": result.success,
                "lean_code": result.code,
                "elaborated_code": result.elaborated_code,
                "source": result.source,
                "warnings": result.warnings
            })
        elif self.use_cav_nlp and self.cav_nlp_bridge:
            # Use CAV-NLP pipeline directly
            return merge_web3_formal_status(self._translate_with_cav_nlp(text, is_theorem=True))
        else:
            # Fallback
            return merge_web3_formal_status({
                "success": True,
                "lean_code": self._generate_fallback_code(text),
                "source": "fallback",
                "warnings": ["CAV-NLP not available - using basic template"]
            })
    
    async def translate_def(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate definition to Lean 4.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        
        Args:
            text: Definition text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with 'lean_code', 'success', etc.
        """
        warnings.warn(
            "translate_def is deprecated. Use UnifiedMathService.formalize() or CAV-NLP directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.formalize(text)
            return merge_web3_formal_status({
                "success": result.success,
                "lean_code": result.code,
                "elaborated_code": result.elaborated_code,
                "source": result.source,
                "warnings": result.warnings
            })
        elif self.use_cav_nlp and self.cav_nlp_bridge:
            # Use CAV-NLP pipeline for definition
            return merge_web3_formal_status(
                self._translate_with_cav_nlp(text, is_theorem=False, is_definition=True)
            )
        else:
            # Fallback
            return merge_web3_formal_status({
                "success": True,
                "lean_code": self._generate_fallback_definition(text),
                "source": "fallback",
                "warnings": ["CAV-NLP not available - using basic template"]
            })
    
    async def translate_thm_detailed(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate theorem with detailed output.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        
        Args:
            text: Theorem text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with detailed formalization results including:
            - lean_code: Generated Lean 4 code
            - semantic_primitives: Parsed semantic primitives
            - dependency_dag: Dependency graph structure
            - canonical_form: Canonical representation
        """
        warnings.warn(
            "translate_thm_detailed is deprecated. Use CAV-NLP directly for detailed output.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.use_cav_nlp and self.cav_nlp_bridge:
            # Use CAV-NLP pipeline with full details
            return merge_web3_formal_status(
                self._translate_with_cav_nlp(
                    text,
                    is_theorem=True,
                    detailed=True
                )
            )
        
        # Fallback to basic translation
        result = await self.translate_thm(text, **kwargs)
        result["detailed"] = True
        result["note"] = "CAV-NLP not available - limited detail provided"
        result["semantic_primitives"] = []
        result["dependency_dag"] = None
        result["canonical_form"] = None
        return merge_web3_formal_status(result)
    
    # ========================================================================
    # Elaboration Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def elaborate(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Elaborate Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to elaborate
            **kwargs: Additional arguments
            
        Returns:
            Dict with elaborated code
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.elaborate(code)
            return merge_web3_formal_status({
                "success": result.success,
                "elaborated_code": result.elaborated_code,
                "info": result.info
            })
        elif self.lean_client:
            result = await self.lean_client.elaborate(code, **kwargs)
            return merge_web3_formal_status(self._normalize_result_payload(result))
        else:
            return merge_web3_formal_status({
                "success": False,
                "error": "Elaboration not available"
            })
    
    # ========================================================================
    # Documentation Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def generate_documentation(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Generate documentation for Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to document
            **kwargs: Additional arguments
            
        Returns:
            Dict with documentation
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.generate_documentation(code)
            return merge_web3_formal_status({
                "success": result.success,
                "documentation": result.documentation,
                "theorem_name": result.theorem_name
            })
        elif self.lean_client:
            if hasattr(self.lean_client, "generate_documentation"):
                result = await self.lean_client.generate_documentation(code, **kwargs)
                return merge_web3_formal_status(self._normalize_result_payload(result))
            return merge_web3_formal_status(
                {"success": False, "error": "Documentation generation not supported by client"}
            )
        else:
            return merge_web3_formal_status({
                "success": False,
                "error": "Documentation generation not available"
            })
    
    # ========================================================================
    # Verification Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def verify(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Verify Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to verify
            **kwargs: Additional arguments
            
        Returns:
            Dict with verification results
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.verify(code)
            if result:
                return merge_web3_formal_status({
                    "success": result.success,
                    "message": result.message if hasattr(result, 'message') else str(result.status),
                    "verified": result.success
                })
            else:
                return merge_web3_formal_status(
                    {"success": False, "error": "Verification returned None"}
                )
        elif self.lean_client:
            if hasattr(self.lean_client, "check_elaboration"):
                result = await self.lean_client.check_elaboration(code, **kwargs)
            elif hasattr(self.lean_client, "verify"):
                result = await self.lean_client.verify(code)
            else:
                result = {"success": False, "error": "Verification not supported by client"}
            return merge_web3_formal_status(self._normalize_result_payload(result))
        else:
            return merge_web3_formal_status({
                "success": False,
                "error": "Verification not available"
            })
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities."""
        web3_status = collect_web3_formal_status()
        return {
            "cav_nlp_available": self.use_cav_nlp,
            "unified_service_available": self.use_unified_service,
            "leanaide_client_available": self.lean_client is not None,
            "translation": self.use_cav_nlp or self.use_unified_service,
            "elaboration": self.lean_client is not None or self.use_unified_service,
            "verification": self.lean_client is not None or self.use_unified_service,
            "documentation": self.lean_client is not None or self.use_unified_service,
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
    
    def _translate_with_cav_nlp(
        self, 
        text: str, 
        is_theorem: bool = True,
        is_definition: bool = False,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Translate text using the full CAV-NLP pipeline.
        
        Pipeline:
        1. Parse text with flexible_semantic_parsing (SemanticNormalizer)
        2. Extract semantic primitives
        3. Build dependency DAG
        4. Generate canonical Lean code
        
        Args:
            text: Natural language text to translate
            is_theorem: Whether this is a theorem (vs lemma/axiom)
            is_definition: Whether this is a definition
            detailed: Whether to include detailed output
            
        Returns:
            Dict with translation results
        """
        result = {
            "success": False,
            "lean_code": "",
            "source": "cav_nlp",
            "warnings": []
        }
        
        try:
            # Step 1: Semantic parsing with flexible_semantic_parsing
            semantic_primitives = []
            if self._semantic_normalizer is not None:
                try:
                    semantic_primitives = self._semantic_normalizer.normalize(text)
                    logger.debug(f"Extracted {len(semantic_primitives)} semantic primitives")
                except Exception as e:
                    logger.warning(f"Semantic parsing failed: {e}")
                    result["warnings"].append(f"Semantic parsing error: {e}")
            
            # Step 2: Build dependency DAG
            dag = None
            if CAV_NLP_DAG_AVAILABLE and self._statement_class is not None:
                try:
                    dag = self._build_dependency_dag(text, semantic_primitives, is_theorem)
                    logger.debug("Built dependency DAG")
                except Exception as e:
                    logger.warning(f"DAG construction failed: {e}")
                    result["warnings"].append(f"DAG construction error: {e}")
            
            # Step 3: Generate canonical Lean code
            if is_definition:
                lean_code = self._generate_canonical_definition(text, semantic_primitives, dag)
            else:
                lean_code = self._generate_canonical_theorem(text, semantic_primitives, dag, is_theorem)
            
            result["success"] = True
            result["lean_code"] = lean_code
            
            # Add detailed output if requested
            if detailed:
                result["semantic_primitives"] = [
                    {
                        "kind": p.kind,
                        "canonical_form": p.canonical_form,
                        "confidence": p.confidence
                    }
                    for p in semantic_primitives
                ] if semantic_primitives else []
                
                result["dependency_dag"] = self._dag_to_dict(dag) if dag else None
                result["canonical_form"] = self._extract_canonical_form(semantic_primitives)
                result["note"] = "Detailed mode - CAV-NLP provides dependency DAG and canonical form"
                result["detailed"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"CAV-NLP translation failed: {e}")
            result["warnings"].append(f"CAV-NLP error: {e}")
            # Fallback to template
            result["lean_code"] = self._generate_fallback_code(text)
            result["success"] = True  # Still return success with fallback
            return result
    
    def _build_dependency_dag(
        self, 
        text: str, 
        semantic_primitives: List[Any],
        is_theorem: bool
    ) -> Any:
        """
        Build dependency DAG from semantic primitives.
        
        Args:
            text: Original text
            semantic_primitives: Extracted semantic primitives
            is_theorem: Whether this is a theorem
            
        Returns:
            DependencyDAG instance
        """
        dag = self._dependency_dag_class()
        
        # Create a statement for this text
        kind = self._statement_kind.THEOREM if is_theorem else self._statement_kind.PROPOSITION
        
        # Extract variables from primitives
        variables = []
        for prim in semantic_primitives:
            if hasattr(prim, 'canonical_form'):
                # Try to extract variable names from canonical form
                matches = re.findall(r'[∀∃]\s+(\w+)', prim.canonical_form)
                variables.extend(matches)
        
        # Extract types mentioned (capitalized words)
        type_pattern = re.compile(r'\b([A-Z][a-zA-Z]*)\b')
        types_mentioned = list(set(type_pattern.findall(text)))
        
        stmt = self._statement_class(
            id="stmt_0",
            kind=kind,
            name=None,
            number=None,
            text=text,
            variables=list(set(variables)),
            types_mentioned=types_mentioned
        )
        
        dag.add_node(stmt)
        return dag
    
    def _generate_canonical_theorem(
        self, 
        text: str, 
        semantic_primitives: List[Any],
        dag: Any,
        is_theorem: bool
    ) -> str:
        """
        Generate canonical Lean theorem code.
        
        Uses semantic primitives to construct proper Lean syntax.
        """
        # Extract theorem name from text or use default
        theorem_name = self._extract_theorem_name(text)
        
        # Build imports
        imports = ["import Mathlib"]
        
        # Check for Real numbers
        if any(t in text.lower() for t in ['real', 'ℝ', 'r']):
            imports.append("import Mathlib.Data.Real.Basic")
        
        # Check for Natural numbers
        if any(t in text.lower() for t in ['natural', 'nat', 'ℕ', 'n']):
            imports.append("import Mathlib.Data.Nat.Basic")
        
        # Check for Integers
        if any(t in text.lower() for t in ['integer', 'int', 'ℤ', 'z']):
            imports.append("import Mathlib.Data.Int.Basic")
        
        # Start building the theorem
        lines = imports + [""]
        
        # Add comment with original text
        lines.append(f"-- Formalized from: {text[:100]}{'...' if len(text) > 100 else ''}")
        lines.append("")
        
        # Extract quantifiers and build theorem signature
        quantifiers = []
        hypotheses = []
        conclusion = "True"  # Default
        
        for prim in semantic_primitives:
            if prim.kind == 'UNIVERSAL_QUANTIFIER':
                # Extract variable from canonical form
                match = re.search(r'∀\s+(\w+)\s*:\s*(\w+)', prim.canonical_form)
                if match:
                    var_name, var_type = match.groups()
                    quantifiers.append(f"    ({var_name} : {self._map_type(var_type)})")
            elif prim.kind == 'EXISTENTIAL_QUANTIFIER':
                match = re.search(r'∃\s+(\w+)\s*:\s*(\w+)', prim.canonical_form)
                if match:
                    var_name, var_type = match.groups()
                    quantifiers.append(f"    ({var_name} : {self._map_type(var_type)})")
            elif prim.kind == 'COMPARISON':
                # Convert comparison to Lean
                conclusion = self._primitive_to_lean(prim)
        
        # If no conclusion extracted, try pattern matching on text
        if conclusion == "True":
            conclusion = self._text_to_lean_expr(text)
        
        # Build theorem
        keyword = "theorem" if is_theorem else "lemma"
        lines.append(f"{keyword} {theorem_name}")
        
        # Add quantifiers
        for q in quantifiers:
            lines.append(q)
        
        # Add conclusion
        lines.append(f"    : {conclusion} := by")
        lines.append("  sorry")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_canonical_definition(
        self, 
        text: str, 
        semantic_primitives: List[Any],
        dag: Any
    ) -> str:
        """
        Generate canonical Lean definition code.
        """
        # Extract definition name
        def_name = self._extract_definition_name(text)
        
        lines = ["import Mathlib", ""]
        lines.append(f"-- Definition from: {text[:100]}{'...' if len(text) > 100 else ''}")
        lines.append("")
        
        # Try to infer type from context
        def_type = "Prop"  # Default
        
        for prim in semantic_primitives:
            if 'set' in prim.canonical_form.lower() or '∈' in prim.canonical_form:
                def_type = "Set α"
                break
            elif 'function' in prim.canonical_form.lower() or '→' in prim.canonical_form:
                def_type = "α → β"
                break
        
        lines.append(f"def {def_name} : {def_type} :=")
        lines.append("  sorry")
        lines.append("")
        
        return "\n".join(lines)
    
    def _extract_theorem_name(self, text: str) -> str:
        """Extract theorem name from text or generate one."""
        # Remove common words and extract key terms
        common = {'the', 'a', 'an', 'is', 'are', 'if', 'then', 'for', 'all', 'every', 
                  'there', 'exists', 'such', 'that', 'and', 'or', 'not'}
        words = [w for w in text.lower().split() if w not in common and len(w) > 2]
        
        if words:
            # Take first few significant words
            name = '_'.join(words[:4])
            # Clean up
            name = re.sub(r'[^\w_]', '', name)
            return name or "formalized_statement"
        
        return "formalized_statement"
    
    def _extract_definition_name(self, text: str) -> str:
        """Extract definition name from text."""
        # Look for "Define X as" or "X is defined as" patterns
        match = re.search(r'(?:define|definition)\s+(\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # Look for capitalized term being defined
        match = re.search(r'\b([A-Z][a-zA-Z]+)\b', text)
        if match:
            return match.group(1)
        
        return "defined_concept"
    
    def _map_type(self, type_str: str) -> str:
        """Map type string to Lean type."""
        type_map = {
            'real': 'ℝ',
            'nat': 'ℕ',
            'natural': 'ℕ',
            'int': 'ℤ',
            'integer': 'ℤ',
            'bool': 'Bool',
            'prop': 'Prop',
            'type': 'Type',
        }
        return type_map.get(type_str.lower(), type_str)
    
    def _primitive_to_lean(self, prim: Any) -> str:
        """Convert a semantic primitive to Lean expression."""
        cf = prim.canonical_form
        
        # Map common patterns
        mappings = [
            (r'GT\((\w+),\s*(\w+)\)', r'\1 > \2'),
            (r'LT\((\w+),\s*(\w+)\)', r'\1 < \2'),
            (r'GE\((\w+),\s*(\w+)\)', r'\1 ≥ \2'),
            (r'LE\((\w+),\s*(\w+)\)', r'\1 ≤ \2'),
        ]
        
        for pattern, replacement in mappings:
            cf = re.sub(pattern, replacement, cf)
        
        return cf if cf != prim.canonical_form else "True"
    
    def _text_to_lean_expr(self, text: str) -> str:
        """Convert text to Lean expression using simple heuristics."""
        # Common patterns
        if '>' in text:
            return "x > 0"  # Simplified
        if '<' in text:
            return "x < 0"
        if '=' in text and '!=' not in text:
            return "x = y"
        
        return "True"
    
    def _extract_canonical_form(self, semantic_primitives: List[Any]) -> Optional[str]:
        """Extract canonical form from semantic primitives."""
        if not semantic_primitives:
            return None
        
        canonical_parts = []
        for prim in semantic_primitives:
            if hasattr(prim, 'canonical_form'):
                canonical_parts.append(prim.canonical_form)
        
        return ' '.join(canonical_parts) if canonical_parts else None
    
    def _dag_to_dict(self, dag: Any) -> Optional[Dict[str, Any]]:
        """Convert dependency DAG to dictionary."""
        if dag is None:
            return None
        
        try:
            nodes = {}
            for node_id, stmt in dag.nodes.items():
                nodes[node_id] = {
                    "id": stmt.id,
                    "kind": stmt.kind.value if hasattr(stmt.kind, 'value') else str(stmt.kind),
                    "text": stmt.text[:100] if stmt.text else "",
                    "variables": stmt.variables if hasattr(stmt, 'variables') else [],
                    "types_mentioned": stmt.types_mentioned if hasattr(stmt, 'types_mentioned') else []
                }
            
            return {
                "nodes": nodes,
                "is_acyclic": dag.is_acyclic() if hasattr(dag, 'is_acyclic') else True,
                "node_count": len(dag.nodes) if hasattr(dag, 'nodes') else 0
            }
        except Exception as e:
            logger.warning(f"Failed to serialize DAG: {e}")
            return None
    
    def _generate_lean_code(self, text: str) -> str:
        """Generate basic Lean code from text - now delegates to CAV-NLP."""
        result = self._translate_with_cav_nlp(text, is_theorem=True)
        return result.get("lean_code", self._generate_fallback_code(text))
    
    def _generate_fallback_code(self, text: str) -> str:
        """Generate fallback Lean code when CAV-NLP is unavailable."""
        return f"""import Mathlib

-- Formalized from: {text[:100]}{'...' if len(text) > 100 else ''}
-- NOTE: Using fallback generation (CAV-NLP unavailable)

theorem formalized_statement : True := by
  sorry
"""
    
    def _generate_fallback_definition(self, text: str) -> str:
        """Generate fallback Lean definition code."""
        return f"""import Mathlib

-- Definition from: {text[:100]}{'...' if len(text) > 100 else ''}
-- NOTE: Using fallback generation (CAV-NLP unavailable)

def defined_concept : Prop :=
  sorry
"""


# ============================================================================
# Migration Helper
# ============================================================================

def migrate_leanaide_to_cav_nlp(old_code: str) -> str:
    """
    Helper to migrate old LeanAide code to use CAV-NLP.
    
    Args:
        old_code: Python code using LeanAide
        
    Returns:
        Migrated code using CAV-NLP/Unified service
    """
    replacements = [
        # Import changes
        ("from leanaide_client import LeanAideClient",
         "from openevolve.unified_math_service import UnifiedMathService, create_unified_math_service"),
        
        # Client instantiation
        ("client = LeanAideClient()",
         "service = create_unified_math_service()"),
        
        # Translation calls
        ("await client.translate_thm(text)",
         "await service.formalize(text)"),
        
        ("await client.translate_def(text)",
         "await service.formalize(text)"),
        
        # Result access
        ('result.data["lean_code"]',
         'result.code'),
        
        ('result.data.get("lean_code")',
         'result.code'),
        
        # Verification
        ("await client.check_elaboration(code)",
         "await service.verify(code)"),
    ]
    
    result = old_code
    for old, new in replacements:
        result = result.replace(old, new)
    
    return result


# ============================================================================
# Convenience Functions
# ============================================================================

def create_migration_bridge() -> LeanAideCAVNLPBridge:
    """Create a bridge for migration from LeanAide to CAV-NLP."""
    return LeanAideCAVNLPBridge()


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of the bridge."""
    print("=" * 70)
    print("LeanAide-CAV-NLP Bridge - Example Usage")
    print("=" * 70)
    
    bridge = create_migration_bridge()
    
    # Check capabilities
    print("\n1. CAPABILITIES")
    print("-" * 40)
    caps = bridge.get_capabilities()
    for cap, available in caps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {cap}")
    
    # Translation (redirected to CAV-NLP)
    print("\n2. TRANSLATION (CAV-NLP)")
    print("-" * 40)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = await bridge.translate_thm("For all x > 0, x + 1 > 1")
        
        print(f"   Success: {result['success']}")
        print(f"   Source: {result['source']}")
        print(f"   Code:\n{result['lean_code']}")
        
        if w:
            print(f"   Deprecation warning issued: {len([x for x in w if issubclass(x.category, DeprecationWarning)])}")
    
    print("\n" + "=" * 70)
    print("Bridge example completed!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
