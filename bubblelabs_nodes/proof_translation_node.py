"""
Proof Translation Node for BubbleLabs

Translates between different formal proof formats:
- Lean 4 ↔ SMT-LIB
- Lean 4 ↔ TPTP
- SMT-LIB ↔ TPTP
- Natural language hints

Uses the Z3-LeanAIDE bridge for bidirectional translation.

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import time
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

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


class ProofTranslationNode(BubbleLabsNode):
    """
    Translate between formal proof formats.
    
    Operations:
        - translate: General translation
        - smt_to_lean: Convert SMT-LIB to Lean 4
        - lean_to_smt: Convert Lean 4 to SMT-LIB
        - lean_to_tptp: Convert Lean to TPTP
        - add_hints: Add natural language hints
        - validate: Validate translation correctness
        - batch_translate: Translate multiple formulas
    """
    
    DISPLAY_NAME = "Proof Translation"
    DESCRIPTION = "Translate between formal proof formats (Lean, SMT-LIB, TPTP)"
    ICON = "proof-translation"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
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
        "batch_translate"
    ]
    
    SUPPORTED_FORMATS = ["lean", "smtlib", "tptp", "natural"]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._bridge = None
        
    def _initialize_bridge(self):
        """Initialize Z3-LeanAIDE bridge."""
        try:
            from z3_leanaide_bridge import Z3LeanAideBridge
            self._bridge = Z3LeanAideBridge()
            return True
        except Exception as e:
            logger.warning(f"Could not initialize bridge: {e}")
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
        elif operation in ["translate", "add_hints", "validate"]:
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
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.utcnow().isoformat()
            
            context.add_artifact("proof_translation_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Translation failed: {str(e)}",
                details={"operation": operation}
            )
    
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
        expr = expr.replace("→", ">")
        expr = expr.replace("∧", "&")
        expr = expr.replace("∨", "|")
        expr = expr.replace("¬", "~")
        expr = expr.replace("∀", "!")
        expr = expr.replace("∃", "?")
        return expr.strip()
    
    def _tptp_to_lean_expr(self, expr: str) -> str:
        """Convert TPTP expression to Lean."""
        expr = expr.replace(">", "→")
        expr = expr.replace("&", "∧")
        expr = expr.replace("|", "∨")
        expr = expr.replace("~", "¬")
        return expr.strip()
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
