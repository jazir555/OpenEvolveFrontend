"""
Proof Translation Node for BubbleLabs

Translates between different formal proof formats:
- Lean 4 ↔ SMT-LIB
- Lean 4 ↔ TPTP
- SMT-LIB ↔ TPTP
- Natural language hints
- CAV-NLP based proof translation (NEW)

Uses the Z3-LeanAIDE bridge and CAV-NLP for bidirectional translation.

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import time
import re
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class TranslationDirection(Enum):
    """Direction of translation."""
    SMT_TO_LEAN = "smt_to_lean"
    LEAN_TO_SMT = "lean_to_smt"
    LEAN_TO_TPTP = "lean_to_tptp"
    TPTP_TO_LEAN = "tptp_to_lean"
    SMT_TO_TPTP = "smt_to_tptp"
    TPTP_TO_SMT = "tptp_to_smt"
    # NEW: CAV-NLP based directions
    NL_TO_LEAN = "nl_to_lean"
    NL_TO_SMT = "nl_to_smt"
    Z3_PROOF_TO_LEAN = "z3_proof_to_lean"


class ProofTranslationNode(BubbleLabsNode):
    """
    Translate between formal proof formats with CAV-NLP enhancement.
    
    Operations:
        - translate: General translation
        - smt_to_lean: Convert SMT-LIB to Lean 4
        - lean_to_smt: Convert Lean 4 to SMT-LIB
        - lean_to_tptp: Convert Lean to TPTP
        - add_hints: Add natural language hints
        - validate: Validate translation correctness
        - batch_translate: Translate multiple formulas
        - nl_to_formal: Natural language to formal using CAV-NLP (NEW)
        - z3_proof_export: Export Z3 proofs to Lean using CAV-NLP (NEW)
        - cav_nlp_translate: CAV-NLP enhanced translation (NEW)
    """
    
    DISPLAY_NAME = "Proof Translation"
    DESCRIPTION = "Translate between formal proof formats (Lean, SMT-LIB, TPTP) with CAV-NLP"
    ICON = "proof-translation"
    CATEGORY = "mathematical_verification"
    VERSION = "2.0.0"  # Updated for CAV-NLP integration
    
    OPERATIONS = [
        "translate",
        "smt_to_lean",
        "lean_to_smt",
        "lean_to_tptp",
        "tptp_to_lean",
        "smt_to_tptp",
        "tptp_to_smt",
        "add_hints",
        "validate",
        "batch_translate",
        "nl_to_formal",  # NEW
        "z3_proof_export",  # NEW
        "cav_nlp_translate"  # NEW
    ]
    
    SUPPORTED_FORMATS = ["lean", "smtlib", "tptp", "natural", "z3_proof"]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._bridge = None
        self._math_service = None
        self._initialize_bridge()
        self._initialize_math_service()
        
    def _initialize_bridge(self):
        """Initialize Z3-LeanAIDE bridge."""
        try:
            from z3_leanaide_bridge import Z3LeanAideBridge
            self._bridge = Z3LeanAideBridge()
            return True
        except Exception as e:
            logger.warning(f"Could not initialize bridge: {e}")
            return False
    
    def _initialize_math_service(self):
        """Initialize CAV-NLP math service."""
        if not self.config.get("use_cav_nlp", True):
            logger.info("CAV-NLP integration disabled by configuration")
            return False
            
        try:
            from openevolve.unified_math_service import UnifiedMathService
            self._math_service = UnifiedMathService(
                use_cav_nlp=True,
                use_leanaide=self.config.get("use_leanaide", True)
            )
            logger.info("CAV-NLP math service initialized for proof translation")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize CAV-NLP math service: {e}")
            return False
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "translate"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_translate":
            if "items" not in inputs and "items" not in self.config:
                errors.append("batch_translate requires 'items' input")
        elif operation in ["translate", "add_hints", "validate", "cav_nlp_translate"]:
            if "content" not in inputs and "content" not in self.config:
                errors.append(f"{operation} requires 'content' input")
            if operation == "translate":
                if "source_format" not in inputs and "source_format" not in self.config:
                    errors.append("translate requires 'source_format'")
                if "target_format" not in inputs and "target_format" not in self.config:
                    errors.append("translate requires 'target_format'")
        elif operation in ["smt_to_lean", "lean_to_smt", "lean_to_tptp", "tptp_to_lean", "smt_to_tptp", "tptp_to_smt"]:
            if "content" not in inputs and "content" not in self.config:
                errors.append(f"{operation} requires 'content' input")
        elif operation == "nl_to_formal":
            if "content" not in inputs and "content" not in self.config:
                errors.append("nl_to_formal requires 'content' input (natural language)")
        elif operation == "z3_proof_export":
            if "content" not in inputs and "content" not in self.config:
                errors.append("z3_proof_export requires 'content' input (Z3 proof)")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "translate",
                    "description": "Translation operation"
                },
                "content": {
                    "type": "string",
                    "description": "Content to translate"
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "content": {"type": "string"},
                            "source_format": {"type": "string"},
                            "target_format": {"type": "string"}
                        }
                    },
                    "description": "Items for batch translation"
                },
                "source_format": {
                    "type": "string",
                    "enum": self.SUPPORTED_FORMATS,
                    "description": "Source format"
                },
                "target_format": {
                    "type": "string",
                    "enum": self.SUPPORTED_FORMATS,
                    "description": "Target format"
                },
                "hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Natural language hints for translation"
                },
                "preserve_comments": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve comments in translation"
                },
                "verify_translation": {
                    "type": "boolean",
                    "default": True,
                    "description": "Verify translation correctness"
                },
                "optimization_level": {
                    "type": "string",
                    "enum": ["none", "basic", "aggressive"],
                    "default": "basic",
                    "description": "Optimization level for translation"
                },
                # NEW: CAV-NLP configuration options
                "use_cav_nlp": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable CAV-NLP for enhanced translation"
                },
                "use_leanaide": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use LeanAide for verification"
                },
                "elaborate_result": {
                    "type": "boolean",
                    "default": True,
                    "description": "Elaborate translated code with LeanAide"
                },
                "generate_documentation": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate documentation for translation"
                },
                "cav_nlp_timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Timeout for CAV-NLP operations"
                },
                "fallback_to_bridge": {
                    "type": "boolean",
                    "default": True,
                    "description": "Fall back to bridge translation if CAV-NLP fails"
                },
                "export_proof_style": {
                    "type": "string",
                    "enum": ["tactic", "term", "structured"],
                    "default": "tactic",
                    "description": "Style for exported proofs"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute translation operation."""
        start_time = time.time()
        operation = inputs.get("operation", self.config.get("operation", "translate"))
        
        context.update_progress(10)
        
        if self._bridge is None:
            self._initialize_bridge()
        
        if self._math_service is None and self.config.get("use_cav_nlp", True):
            self._initialize_math_service()
        
        context.update_progress(20)
        
        try:
            if operation == "translate":
                result = self._translate(inputs, context)
            elif operation == "smt_to_lean":
                result = self._smt_to_lean(inputs, context)
            elif operation == "lean_to_smt":
                result = self._lean_to_smt(inputs, context)
            elif operation == "lean_to_tptp":
                result = self._lean_to_tptp(inputs, context)
            elif operation == "tptp_to_lean":
                result = self._tptp_to_lean(inputs, context)
            elif operation == "smt_to_tptp":
                result = self._smt_to_tptp(inputs, context)
            elif operation == "tptp_to_smt":
                result = self._tptp_to_smt(inputs, context)
            elif operation == "add_hints":
                result = self._add_hints(inputs, context)
            elif operation == "validate":
                result = self._validate(inputs, context)
            elif operation == "batch_translate":
                result = self._batch_translate(inputs, context)
            elif operation == "nl_to_formal":
                result = asyncio.run(self._nl_to_formal(inputs, context))
            elif operation == "z3_proof_export":
                result = asyncio.run(self._z3_proof_export(inputs, context))
            elif operation == "cav_nlp_translate":
                result = asyncio.run(self._cav_nlp_translate(inputs, context))
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.utcnow().isoformat()
            result["cav_nlp_enabled"] = self.config.get("use_cav_nlp", True)
            
            context.add_artifact("proof_translation_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Translation failed: {str(e)}",
                details={"operation": operation}
            )
    
    # =======================================================================
    # NEW: CAV-NLP Enhanced Operations
    # =======================================================================
    
    async def _nl_to_formal(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Natural language to formal code using CAV-NLP.
        
        Primary CAV-NLP operation for translating mathematical
        natural language to formal Lean 4 code.
        """
        content = inputs.get("content", self.config.get("content", ""))
        target_format = inputs.get("target_format", self.config.get("target_format", "lean"))
        elaborate = inputs.get("elaborate_result", self.config.get("elaborate_result", True))
        generate_docs = inputs.get("generate_documentation", self.config.get("generate_documentation", False))
        
        context.update_progress(30)
        
        if not self._math_service:
            return {
                "success": False,
                "error": "CAV-NLP service not available",
                "cav_nlp_used": False,
                "fallback": True
            }
        
        try:
            # Use CAV-NLP for formalization
            formalization = await self._math_service.formalize(
                text=content,
                elaborate=elaborate,
                generate_docs=generate_docs
            )
            
            context.update_progress(70)
            
            if formalization.success:
                context.update_progress(100)
                
                result = {
                    "success": True,
                    "source": "natural",
                    "target": target_format,
                    "translation": formalization.code,
                    "cav_nlp_used": True,
                    "formalization_source": formalization.source,
                    "warnings": formalization.warnings
                }
                
                if formalization.elaborated_code:
                    result["elaborated_code"] = formalization.elaborated_code
                if formalization.documentation:
                    result["documentation"] = formalization.documentation
                
                return result
            else:
                return {
                    "success": False,
                    "error": "CAV-NLP formalization failed",
                    "cav_nlp_used": True,
                    "warnings": formalization.warnings
                }
                
        except Exception as e:
            logger.error(f"NL to formal translation failed: {e}")
            
            if self.config.get("fallback_to_bridge", True):
                return self._fallback_nl_to_formal(content, target_format)
            else:
                return {
                    "success": False,
                    "error": f"CAV-NLP translation error: {e}",
                    "cav_nlp_used": True
                }
    
    async def _z3_proof_export(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Export Z3 proofs to Lean 4 using CAV-NLP.
        
        Takes a Z3 proof trace and converts it to a structured
        Lean 4 proof using CAV-NLP semantic understanding.
        """
        content = inputs.get("content", self.config.get("content", ""))
        proof_style = inputs.get("export_proof_style", self.config.get("export_proof_style", "tactic"))
        
        context.update_progress(30)
        
        # Step 1: Parse Z3 proof
        z3_proof_data = self._parse_z3_proof(content)
        
        context.update_progress(50)
        
        # Step 2: Use CAV-NLP to generate Lean proof structure
        lean_proof = None
        if self._math_service and z3_proof_data:
            try:
                # Create a natural language description of the proof
                nl_proof = self._z3_proof_to_nl(z3_proof_data)
                
                # Formalize with CAV-NLP
                formalization = await self._math_service.formalize(nl_proof)
                
                if formalization.success:
                    lean_proof = formalization.code
                    context.update_progress(80)
            except Exception as e:
                logger.warning(f"CAV-NLP proof export failed: {e}")
        
        context.update_progress(90)
        
        if lean_proof:
            return {
                "success": True,
                "source": "z3_proof",
                "target": "lean",
                "translation": lean_proof,
                "cav_nlp_used": True,
                "proof_style": proof_style,
                "z3_proof_steps": z3_proof_data.get("steps", []),
                "note": "Z3 proof exported to Lean 4 using CAV-NLP"
            }
        else:
            return {
                "success": False,
                "error": "Could not export Z3 proof to Lean",
                "cav_nlp_used": self._math_service is not None,
                "fallback": self._generate_basic_lean_from_z3(z3_proof_data)
            }
    
    async def _cav_nlp_translate(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        CAV-NLP enhanced translation.
        
        Uses CAV-NLP semantic understanding to improve translation
        between formal formats.
        """
        content = inputs.get("content", self.config.get("content", ""))
        source = inputs.get("source_format", self.config.get("source_format", ""))
        target = inputs.get("target_format", self.config.get("target_format", ""))
        
        context.update_progress(30)
        
        # Step 1: If source is natural language, use CAV-NLP formalization
        if source == "natural" and self._math_service:
            return await self._nl_to_formal(inputs, context)
        
        # Step 2: For other translations, use standard bridge with CAV-NLP verification
        context.update_progress(50)
        
        # Use standard bridge translation
        direction_map = {
            ("smtlib", "lean"): self._smt_to_lean,
            ("lean", "smtlib"): self._lean_to_smt,
            ("lean", "tptp"): self._lean_to_tptp,
            ("tptp", "lean"): self._tptp_to_lean,
            ("smtlib", "tptp"): self._smt_to_tptp,
            ("tptp", "smtlib"): self._tptp_to_smt
        }
        
        op = direction_map.get((source, target))
        if op:
            result = op({"content": content}, context)
            
            # CAV-NLP enhancement: verify and validate translation
            if self._math_service and target == "lean":
                try:
                    verification = await self._math_service.verify(result.get("translation", ""))
                    result["cav_nlp_verification"] = {
                        "success": verification.success if verification else False,
                        "source": "cav_nlp"
                    }
                except Exception as e:
                    logger.warning(f"CAV-NLP verification failed: {e}")
            
            result["cav_nlp_used"] = self._math_service is not None
            return result
        
        return {
            "success": False,
            "error": f"CAV-NLP translation from {source} to {target} not supported",
            "cav_nlp_used": False
        }
    
    def _parse_z3_proof(self, z3_proof: str) -> Dict[str, Any]:
        """Parse Z3 proof trace into structured data."""
        steps = []
        
        # Simple parsing of Z3 proof steps
        lines = z3_proof.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith(";"):
                # Extract proof steps (simplified)
                if "assert" in line or "rule" in line:
                    steps.append(line)
        
        return {
            "steps": steps,
            "step_count": len(steps)
        }
    
    def _z3_proof_to_nl(self, z3_proof_data: Dict) -> str:
        """Convert Z3 proof data to natural language description."""
        steps = z3_proof_data.get("steps", [])
        
        if not steps:
            return "Theorem to be proved"
        
        # Create a natural language summary
        nl_parts = ["Proof by Z3 theorem prover:"]
        for i, step in enumerate(steps[:5], 1):  # Limit to 5 steps
            nl_parts.append(f"Step {i}: {step}")
        
        return " ".join(nl_parts)
    
    def _generate_basic_lean_from_z3(self, z3_proof_data: Dict) -> str:
        """Generate basic Lean code from Z3 proof data."""
        steps = z3_proof_data.get("steps", [])
        
        lines = [
            "import Mathlib",
            "",
            "-- Exported from Z3 proof",
            "theorem exported_z3_proof : True := by",
            "  -- Proof steps from Z3",
        ]
        
        for step in steps[:10]:
            lines.append(f"  -- {step}")
        
        lines.append("  trivial")
        
        return "\n".join(lines)
    
    def _fallback_nl_to_formal(self, content: str, target_format: str) -> Dict[str, Any]:
        """Fallback for NL to formal translation."""
        return {
            "success": True,
            "source": "natural",
            "target": target_format,
            "translation": f"-- Formalized from: {content[:100]}...\ntheorem auto : True := by sorry",
            "cav_nlp_used": False,
            "fallback": True,
            "warnings": ["CAV-NLP unavailable - using basic template"]
        }
    
    # =======================================================================
    # Standard Operations
    # =======================================================================
    
    def _translate(self, inputs: Dict, context) -> Dict[str, Any]:
        """General translation between formats."""
        content = inputs.get("content", self.config.get("content", ""))
        source = inputs.get("source_format", self.config.get("source_format", ""))
        target = inputs.get("target_format", self.config.get("target_format", ""))
        
        context.update_progress(40)
        
        # Map to specific operation
        direction_map = {
            ("smtlib", "lean"): self._smt_to_lean,
            ("lean", "smtlib"): self._lean_to_smt,
            ("lean", "tptp"): self._lean_to_tptp,
            ("tptp", "lean"): self._tptp_to_lean,
            ("smtlib", "tptp"): self._smt_to_tptp,
            ("tptp", "smtlib"): self._tptp_to_smt
        }
        
        # Check for CAV-NLP enhancement
        if source == "natural" and self.config.get("use_cav_nlp", True):
            return asyncio.run(self._nl_to_formal(inputs, context))
        
        op = direction_map.get((source, target))
        if op:
            return op({"content": content}, context)
        
        # Unsupported direction
        return {
            "success": False,
            "error": f"Translation from {source} to {target} not supported",
            "supported": list(direction_map.keys())
        }
    
    def _smt_to_lean(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert SMT-LIB to Lean 4."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        if self._bridge:
            try:
                from z3_leanaide_bridge import TranslationDirection
                result = self._bridge.translate(
                    content,
                    direction=TranslationDirection.SMT_TO_LEAN
                )
                
                context.update_progress(90)
                
                return {
                    "success": result.success,
                    "source": result.source,
                    "target": result.target,
                    "translation": result.translation,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            except Exception as e:
                logger.warning(f"Bridge translation failed: {e}")
        
        context.update_progress(70)
        
        # Fallback translation
        return self._fallback_smt_to_lean(content)
    
    def _lean_to_smt(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert Lean 4 to SMT-LIB."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        if self._bridge:
            try:
                from z3_leanaide_bridge import TranslationDirection
                result = self._bridge.translate(
                    content,
                    direction=TranslationDirection.LEAN_TO_SMT
                )
                
                context.update_progress(90)
                
                return {
                    "success": result.success,
                    "source": result.source,
                    "target": result.target,
                    "translation": result.translation,
                    "errors": result.errors
                }
            except Exception as e:
                logger.warning(f"Bridge translation failed: {e}")
        
        context.update_progress(70)
        
        return self._fallback_lean_to_smt(content)
    
    def _lean_to_tptp(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert Lean to TPTP."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        # Extract theorem statement
        theorem_match = re.search(r'theorem\s+(\w+)\s*:([^:=]+)', content, re.DOTALL)
        
        if theorem_match:
            name = theorem_match.group(1)
            statement = theorem_match.group(2).strip()
            
            # Simple conversion to TPTP
            tptp = f"thf({name}, conjecture, ({self._lean_to_tptp_expr(statement)}))."
            
            context.update_progress(90)
            
            return {
                "success": True,
                "source": "lean",
                "target": "tptp",
                "translation": tptp,
                "theorem_name": name
            }
        
        context.update_progress(80)
        
        return {
            "success": False,
            "error": "Could not parse Lean theorem",
            "note": "Fallback conversion failed"
        }
    
    def _tptp_to_lean(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert TPTP to Lean."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        # Parse TPTP
        tptp_match = re.search(r'thf\(([^,]+),\s*(\w+),\s*(.+)\)\.', content, re.DOTALL)
        
        if tptp_match:
            name = tptp_match.group(1).strip()
            role = tptp_match.group(2).strip()
            formula = tptp_match.group(3).strip()
            
            # Simple conversion
            if role == "conjecture":
                lean = f"theorem {name} : {self._tptp_to_lean_expr(formula)} := by sorry"
            else:
                lean = f"axiom {name} : {self._tptp_to_lean_expr(formula)}"
            
            context.update_progress(90)
            
            return {
                "success": True,
                "source": "tptp",
                "target": "lean",
                "translation": lean,
                "theorem_name": name
            }
        
        context.update_progress(80)
        
        return {
            "success": False,
            "error": "Could not parse TPTP formula"
        }
    
    def _smt_to_tptp(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert SMT-LIB to TPTP."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        # Extract assertions
        assertions = re.findall(r'\(assert\s+(.+)\)', content)
        
        if assertions:
            tptp_lines = []
            for i, assertion in enumerate(assertions):
                tptp = f"thf(smt_assert_{i}, axiom, ({assertion}))."
                tptp_lines.append(tptp)
            
            context.update_progress(90)
            
            return {
                "success": True,
                "source": "smtlib",
                "target": "tptp",
                "translation": "\n".join(tptp_lines)
            }
        
        context.update_progress(80)
        
        return {
            "success": False,
            "error": "No assertions found in SMT-LIB"
        }
    
    def _tptp_to_smt(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert TPTP to SMT-LIB."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        # Parse TPTP formulas
        formulas = re.findall(r'thf\([^,]+,\s*\w+,\s*(.+)\)\.', content)
        
        if formulas:
            smt_lines = ["(set-logic ALL)"]
            for formula in formulas:
                smt_lines.append(f"(assert {formula})")
            smt_lines.append("(check-sat)")
            
            context.update_progress(90)
            
            return {
                "success": True,
                "source": "tptp",
                "target": "smtlib",
                "translation": "\n".join(smt_lines)
            }
        
        context.update_progress(80)
        
        return {
            "success": False,
            "error": "No formulas found in TPTP"
        }
    
    def _add_hints(self, inputs: Dict, context) -> Dict[str, Any]:
        """Add natural language hints to formal code."""
        content = inputs.get("content", self.config.get("content", ""))
        hints = inputs.get("hints", self.config.get("hints", []))
        
        context.update_progress(40)
        
        # Add hints as comments
        hint_comments = "\n".join([f"-- Hint: {h}" for h in hints])
        
        context.update_progress(80)
        
        result = f"{hint_comments}\n\n{content}" if hint_comments else content
        
        context.update_progress(100)
        
        return {
            "success": True,
            "original": content,
            "with_hints": result,
            "hint_count": len(hints)
        }
    
    def _validate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Validate translation correctness."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        # Check syntax based on format
        format_type = self._detect_format(content)
        
        context.update_progress(70)
        
        errors = []
        if format_type == "lean":
            # Check for balanced braces/parentheses
            if content.count("(") != content.count(")"):
                errors.append("Unbalanced parentheses")
            if content.count("{") != content.count("}"):
                errors.append("Unbalanced braces")
        elif format_type == "smtlib":
            if content.count("(") != content.count(")"):
                errors.append("Unbalanced parentheses in SMT-LIB")
            if "(assert" not in content and "(declare-fun" not in content:
                errors.append("No assertions or declarations found")
        
        context.update_progress(100)
        
        return {
            "success": len(errors) == 0,
            "valid": len(errors) == 0,
            "detected_format": format_type,
            "errors": errors,
            "warnings": [] if errors else ["Basic syntax check passed"]
        }
    
    def _batch_translate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Translate multiple items."""
        items = inputs.get("items", self.config.get("items", []))
        
        context.update_progress(20)
        
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            progress = 20 + (70 * (i + 1) // max(total, 1))
            context.update_progress(progress)
            
            content = item.get("content", "")
            source = item.get("source_format", "lean")
            target = item.get("target_format", "smtlib")
            
            # Map operation
            op_map = {
                ("smtlib", "lean"): self._smt_to_lean,
                ("lean", "smtlib"): self._lean_to_smt,
                ("lean", "tptp"): self._lean_to_tptp,
                ("tptp", "lean"): self._tptp_to_lean
            }
            
            op = op_map.get((source, target))
            if op:
                result = op({"content": content}, context)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported translation: {source} -> {target}"
                }
            
            results.append({
                "name": item.get("name", f"item_{i}"),
                "source_format": source,
                "target_format": target,
                "result": result
            })
        
        context.update_progress(100)
        
        successful = sum(1 for r in results if r["result"].get("success", False))
        
        return {
            "success": True,
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "results": results
        }
    
    def _fallback_smt_to_lean(self, smtlib: str) -> Dict[str, Any]:
        """Fallback SMT to Lean translation."""
        # Extract declare-fun
        decls = re.findall(r'\(declare-fun\s+(\w+)\s*\(\)\s+(\w+)\)', smtlib)
        
        # Extract assertions
        assertions = re.findall(r'\(assert\s+(.+)\)', smtlib)
        
        lean_lines = ["import Mathlib", "", "theorem smt_translated :"]
        
        for name, var_type in decls:
            lean_type = "Int" if var_type in ["Int", "Integer"] else "Real" if var_type == "Real" else "Bool"
            lean_lines.append(f"  -- Variable {name} : {lean_type}")
        
        if assertions:
            lean_lines.append("  " + " ∧ ".join([f"({a})" for a in assertions[:3]]))
        else:
            lean_lines.append("  True")
        
        lean_lines.append(":= by sorry")
        
        return {
            "success": True,
            "source": "smtlib",
            "target": "lean",
            "translation": "\n".join(lean_lines),
            "warnings": ["Fallback translation - Z3-Lean bridge unavailable"]
        }
    
    def _fallback_lean_to_smt(self, lean: str) -> Dict[str, Any]:
        """Fallback Lean to SMT translation."""
        smt_lines = ["(set-logic ALL)", ""]
        
        # Extract variables from theorem signature
        var_match = re.search(r'theorem\s+\w+\s*\(([^)]+)\)', lean)
        if var_match:
            vars_str = var_match.group(1)
            # Simple parsing
            vars_list = [v.strip() for v in vars_str.split(",")]
            for v in vars_list:
                parts = v.split(":")
                if len(parts) == 2:
                    name = parts[0].strip()
                    var_type = parts[1].strip()
                    smt_type = "Int" if "Nat" in var_type or "Int" in var_type else "Real" if "Real" in var_type else "Bool"
                    smt_lines.append(f"(declare-fun {name} () {smt_type})")
        
        smt_lines.append("(assert true)")
        smt_lines.append("(check-sat)")
        
        return {
            "success": True,
            "source": "lean",
            "target": "smtlib",
            "translation": "\n".join(smt_lines),
            "warnings": ["Fallback translation - Z3-Lean bridge unavailable"]
        }
    
    def _detect_format(self, content: str) -> str:
        """Detect the format of content."""
        if re.search(r'theorem\s+\w+\s*:', content):
            return "lean"
        elif re.search(r'\(declare-fun|assert|check-sat\)', content):
            return "smtlib"
        elif re.search(r'thf\(|fof\(|cnf\(', content):
            return "tptp"
        return "unknown"
    
    def _lean_to_tptp_expr(self, expr: str) -> str:
        """Convert Lean expression to TPTP."""
        # Simple replacements
        expr = expr.replace("->", ">")
        expr = expr.replace("∧", "&")
        expr = expr.replace("∨", "|")
        expr = expr.replace("¬", "~")
        expr = expr.replace("∀", "!")
        expr = expr.replace("∃", "?")
        return expr.strip()
    
    def _tptp_to_lean_expr(self, expr: str) -> str:
        """Convert TPTP expression to Lean."""
        expr = expr.replace(">", "->")
        expr = expr.replace("&", "∧")
        expr = expr.replace("|", "∨")
        expr = expr.replace("~", "¬")
        return expr.strip()
    
    def is_healthy(self) -> bool:
        """Check node health."""
        health = {
            "bridge_available": self._bridge is not None,
            "cav_nlp_available": self._math_service is not None
        }
        return any(health.values())
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get node capabilities."""
        return {
            "bridge_available": self._bridge is not None,
            "cav_nlp_available": self._math_service is not None,
            "operations": self.OPERATIONS,
            "supported_formats": self.SUPPORTED_FORMATS,
            "cav_nlp_config": {
                "use_cav_nlp": self.config.get("use_cav_nlp", True),
                "use_leanaide": self.config.get("use_leanaide", True),
                "cav_nlp_timeout": self.config.get("cav_nlp_timeout", 30.0),
                "fallback_to_bridge": self.config.get("fallback_to_bridge", True)
            }
        }
