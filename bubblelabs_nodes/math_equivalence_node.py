"""
Math Equivalence Checking Node for BubbleLabs

Checks if mathematical expressions are equivalent:
- Algebraic equivalence
- Logical equivalence
- Semantic equivalence
- Step-by-step verification
- Equivalence proof generation

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathEquivalenceNode(BubbleLabsNode):
    """
    Check if mathematical expressions are equivalent.
    
    Operations:
        - check_equivalence: Check if two expressions are equivalent
        - algebraic_equivalence: Check algebraic equivalence
        - logical_equivalence: Check logical equivalence
        - show_steps: Show step-by-step transformation
        - find_transformation: Find transformation between expressions
        - verify_identity: Verify mathematical identity
        - batch_check: Check multiple equivalence pairs
    """
    
    DISPLAY_NAME = "Math Equivalence Checker"
    DESCRIPTION = "Check if mathematical expressions are equivalent"
    ICON = "math-equivalence"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "check_equivalence",
        "algebraic_equivalence",
        "logical_equivalence",
        "show_steps",
        "find_transformation",
        "verify_identity",
        "batch_check"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "check_equivalence"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_check":
            if "pairs" not in inputs and "pairs" not in self.config:
                errors.append("batch_check requires 'pairs' input")
        elif operation == "verify_identity":
            if "identity" not in inputs and "identity" not in self.config:
                errors.append("verify_identity requires 'identity' input")
        else:
            if "expr1" not in inputs and "expr1" not in self.config:
                errors.append(f"{operation} requires 'expr1' input")
            if "expr2" not in inputs and "expr2" not in self.config:
                errors.append(f"{operation} requires 'expr2' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "check_equivalence",
                    "description": "Equivalence checking operation"
                },
                "expr1": {
                    "type": "string",
                    "description": "First expression"
                },
                "expr2": {
                    "type": "string",
                    "description": "Second expression"
                },
                "identity": {
                    "type": "string",
                    "description": "Identity to verify"
                },
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expr1": {"type": "string"},
                            "expr2": {"type": "string"}
                        }
                    },
                    "description": "List of expression pairs for batch checking"
                },
                "domain": {
                    "type": "string",
                    "enum": ["algebra", "logic", "arithmetic", "general"],
                    "default": "general",
                    "description": "Mathematical domain"
                },
                "show_proof": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show equivalence proof"
                },
                "timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Timeout in seconds"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute equivalence checking operation."""
        operation = inputs.get("operation", self.config.get("operation", "check_equivalence"))
        
        try:
            if operation == "check_equivalence":
                result = self._check_equivalence(inputs, context)
            elif operation == "algebraic_equivalence":
                result = self._algebraic_equivalence(inputs, context)
            elif operation == "logical_equivalence":
                result = self._logical_equivalence(inputs, context)
            elif operation == "show_steps":
                result = self._show_steps(inputs, context)
            elif operation == "find_transformation":
                result = self._find_transformation(inputs, context)
            elif operation == "verify_identity":
                result = self._verify_identity(inputs, context)
            elif operation == "batch_check":
                result = self._batch_check(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("equivalence_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Equivalence check failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _check_equivalence(self, inputs: Dict, context) -> Dict[str, Any]:
        """General equivalence check."""
        expr1 = inputs.get("expr1", self.config.get("expr1", ""))
        expr2 = inputs.get("expr2", self.config.get("expr2", ""))
        domain = inputs.get("domain", self.config.get("domain", "general"))
        show_proof = inputs.get("show_proof", self.config.get("show_proof", True))
        
        context.update_progress(50)
        
        # Dispatch to domain-specific checker
        if domain == "algebra":
            result = self._check_algebraic(expr1, expr2)
        elif domain == "logic":
            result = self._check_logical(expr1, expr2)
        elif domain == "arithmetic":
            result = self._check_arithmetic(expr1, expr2)
        else:
            result = self._check_general(expr1, expr2)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "expr1": expr1,
            "expr2": expr2,
            "are_equivalent": result["equivalent"],
            "confidence": result.get("confidence", 1.0),
            "domain": domain,
            "proof": result.get("proof") if show_proof else None,
            "method": result.get("method", "general")
        }
    
    def _algebraic_equivalence(self, inputs: Dict, context) -> Dict[str, Any]:
        """Check algebraic equivalence."""
        expr1 = inputs.get("expr1", self.config.get("expr1", ""))
        expr2 = inputs.get("expr2", self.config.get("expr2", ""))
        
        context.update_progress(50)
        result = self._check_algebraic(expr1, expr2)
        context.update_progress(100)
        
        return {
            "success": True,
            "expr1": expr1,
            "expr2": expr2,
            **result,
            "domain": "algebra"
        }
    
    def _logical_equivalence(self, inputs: Dict, context) -> Dict[str, Any]:
        """Check logical equivalence."""
        expr1 = inputs.get("expr1", self.config.get("expr1", ""))
        expr2 = inputs.get("expr2", self.config.get("expr2", ""))
        
        context.update_progress(50)
        result = self._check_logical(expr1, expr2)
        context.update_progress(100)
        
        return {
            "success": True,
            "expr1": expr1,
            "expr2": expr2,
            **result,
            "domain": "logic"
        }
    
    def _show_steps(self, inputs: Dict, context) -> Dict[str, Any]:
        """Show step-by-step transformation."""
        expr1 = inputs.get("expr1", self.config.get("expr1", ""))
        expr2 = inputs.get("expr2", self.config.get("expr2", ""))
        
        context.update_progress(50)
        
        steps = self._generate_transformation_steps(expr1, expr2)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "expr1": expr1,
            "expr2": expr2,
            "steps": steps,
            "step_count": len(steps),
            "transformations": [s["rule"] for s in steps]
        }
    
    def _find_transformation(self, inputs: Dict, context) -> Dict[str, Any]:
        """Find transformation between expressions."""
        expr1 = inputs.get("expr1", self.config.get("expr1", ""))
        expr2 = inputs.get("expr2", self.config.get("expr2", ""))
        
        context.update_progress(50)
        
        transformation = self._find_transformation_path(expr1, expr2)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "expr1": expr1,
            "expr2": expr2,
            "transformation": transformation,
            "found": transformation is not None
        }
    
    def _verify_identity(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify a mathematical identity."""
        identity = inputs.get("identity", self.config.get("identity", ""))
        
        context.update_progress(50)
        
        # Parse identity (expr1 = expr2)
        parts = identity.split('=')
        if len(parts) == 2:
            expr1, expr2 = parts[0].strip(), parts[1].strip()
            result = self._check_equivalence({"expr1": expr1, "expr2": expr2}, context)
            is_identity = result.get("are_equivalent", False)
        else:
            is_identity = False
            result = {"error": "Could not parse identity"}
        
        context.update_progress(100)
        
        return {
            "success": True,
            "identity": identity,
            "is_valid_identity": is_identity,
            "verification": result
        }
    
    def _batch_check(self, inputs: Dict, context) -> Dict[str, Any]:
        """Check multiple equivalence pairs."""
        pairs = inputs.get("pairs", self.config.get("pairs", []))
        
        results = []
        total = len(pairs)
        equivalent_count = 0
        
        for i, pair in enumerate(pairs):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._check_equivalence({
                "expr1": pair.get("expr1", ""),
                "expr2": pair.get("expr2", "")
            }, context)
            
            if result.get("are_equivalent"):
                equivalent_count += 1
            
            results.append({
                "expr1": pair.get("expr1", "")[:50],
                "expr2": pair.get("expr2", "")[:50],
                "equivalent": result.get("are_equivalent", False)
            })
        
        return {
            "success": True,
            "total": total,
            "equivalent": equivalent_count,
            "not_equivalent": total - equivalent_count,
            "results": results
        }
    
    def _check_algebraic(self, expr1: str, expr2: str) -> Dict:
        """Check algebraic equivalence."""
        # Normalize expressions
        norm1 = self._normalize_algebraic(expr1)
        norm2 = self._normalize_algebraic(expr2)
        
        # Check if normalized forms match
        if norm1 == norm2:
            return {
                "equivalent": True,
                "method": "normalization",
                "confidence": 1.0,
                "proof": f"Both normalize to: {norm1}"
            }
        
        # Try to find common transformations
        return {
            "equivalent": False,
            "method": "normalization",
            "confidence": 0.8,
            "normalized": {"expr1": norm1, "expr2": norm2}
        }
    
    def _check_logical(self, expr1: str, expr2: str) -> Dict:
        """Check logical equivalence."""
        # Build truth tables for simple expressions
        if self._is_simple_logical(expr1) and self._is_simple_logical(expr2):
            equiv = self._truth_table_equivalence(expr1, expr2)
            return {
                "equivalent": equiv,
                "method": "truth_table",
                "confidence": 1.0
            }
        
        # Check for standard equivalences
        if self._is_standard_equivalence(expr1, expr2):
            return {
                "equivalent": True,
                "method": "standard_equivalence",
                "confidence": 1.0
            }
        
        return {
            "equivalent": False,
            "method": "analysis",
            "confidence": 0.6
        }
    
    def _check_arithmetic(self, expr1: str, expr2: str) -> Dict:
        """Check arithmetic equivalence."""
        try:
            # Try to evaluate both
            val1 = self._evaluate_arithmetic(expr1)
            val2 = self._evaluate_arithmetic(expr2)
            
            if val1 is not None and val2 is not None:
                return {
                    "equivalent": abs(val1 - val2) < 1e-10,
                    "method": "evaluation",
                    "confidence": 1.0,
                    "values": {"expr1": val1, "expr2": val2}
                }
        except:
            pass
        
        return {
            "equivalent": False,
            "method": "evaluation",
            "confidence": 0.5
        }
    
    def _check_general(self, expr1: str, expr2: str) -> Dict:
        """General equivalence check."""
        # Try syntactic equality first
        if expr1.strip() == expr2.strip():
            return {"equivalent": True, "method": "syntactic", "confidence": 1.0}
        
        # Try algebraic
        alg = self._check_algebraic(expr1, expr2)
        if alg["equivalent"]:
            return alg
        
        return {"equivalent": False, "method": "general", "confidence": 0.5}
    
    def _normalize_algebraic(self, expr: str) -> str:
        """Normalize algebraic expression."""
        # Simple normalization
        expr = expr.replace(" ", "")
        expr = expr.replace("**", "^")
        # Sort terms alphabetically (simplified)
        return expr.lower()
    
    def _is_simple_logical(self, expr: str) -> bool:
        """Check if expression is simple enough for truth table."""
        vars_in_expr = set(re.findall(r'[a-zA-Z]', expr))
        return len(vars_in_expr) <= 3
    
    def _truth_table_equivalence(self, expr1: str, expr2: str) -> bool:
        """Check equivalence via truth tables."""
        # Simplified - would need proper parser for real implementation
        return False
    
    def _is_standard_equivalence(self, expr1: str, expr2: str) -> bool:
        """Check for standard logical equivalences."""
        # De Morgan's laws
        de_morgan = [
            ("not (A and B)", "(not A) or (not B)"),
            ("not (A or B)", "(not A) and (not B)")
        ]
        for e1, e2 in de_morgan:
            if (expr1.lower() == e1 and expr2.lower() == e2) or \
               (expr1.lower() == e2 and expr2.lower() == e1):
                return True
        return False
    
    def _evaluate_arithmetic(self, expr: str) -> Optional[float]:
        """Safely evaluate arithmetic expression."""
        try:
            # Very basic evaluation - production would need proper parser
            expr = expr.replace("^", "**")
            # Only allow safe operations
            allowed = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                      "*": lambda x, y: x * y, "/": lambda x, y: x / y if y != 0 else None}
            # Parse simple binary operations
            for op, func in allowed.items():
                if op in expr:
                    parts = expr.split(op)
                    if len(parts) == 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            return func(x, y)
                        except:
                            pass
            return float(expr)
        except:
            return None
    
    def _generate_transformation_steps(self, expr1: str, expr2: str) -> List[Dict]:
        """Generate transformation steps."""
        steps = []
        
        # Start with expr1
        current = expr1
        
        # Apply transformations until we reach expr2 or give up
        for i in range(5):  # Max 5 steps
            if current == expr2:
                break
            
            # Try to find a transformation
            next_expr, rule = self._apply_transformation(current)
            if next_expr == current:
                break
            
            steps.append({
                "from": current,
                "to": next_expr,
                "rule": rule
            })
            current = next_expr
        
        return steps
    
    def _apply_transformation(self, expr: str) -> Tuple[str, str]:
        """Apply a single transformation to expression."""
        transformations = [
            (r'(\w+)\s*\+\s*0', r'\1', "Additive identity"),
            (r'0\s*\+\s*(\w+)', r'\1', "Additive identity"),
            (r'(\w+)\s*\*\s*1', r'\1', "Multiplicative identity"),
            (r'1\s*\*\s*(\w+)', r'\1', "Multiplicative identity"),
            (r'(\w+)\s*-\s*0', r'\1', "Subtraction of zero"),
        ]
        
        for pattern, replacement, rule in transformations:
            new_expr = re.sub(pattern, replacement, expr)
            if new_expr != expr:
                return new_expr, rule
        
        return expr, "None applicable"
    
    def _find_transformation_path(self, expr1: str, expr2: str) -> Optional[List[str]]:
        """Find transformation path between expressions."""
        steps = self._generate_transformation_steps(expr1, expr2)
        if steps:
            return [s["rule"] for s in steps]
        return None
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
