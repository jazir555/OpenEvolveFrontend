"""
Math Induction Helper Node for BubbleLabs

Helps with mathematical induction proofs:
- Identify base case
- Formulate inductive hypothesis
- Guide inductive step
- Verify induction structure
- Suggest induction variants (strong, structural, transfinite)

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathInductionHelperNode(BubbleLabsNode):
    """
    Help construct and verify mathematical induction proofs.
    
    Operations:
        - setup_induction: Set up induction proof structure
        - identify_base_case: Identify the base case
        - formulate_hypothesis: Formulate inductive hypothesis
        - guide_inductive_step: Guide the inductive step
        - verify_structure: Verify induction proof structure
        - suggest_variant: Suggest appropriate induction variant
        - analyze_pattern: Analyze pattern for induction
        - complete_induction: Generate complete induction outline
    """
    
    DISPLAY_NAME = "Math Induction Helper"
    DESCRIPTION = "Help construct and verify mathematical induction proofs"
    ICON = "math-induction"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "setup_induction",
        "identify_base_case",
        "formulate_hypothesis",
        "guide_inductive_step",
        "verify_structure",
        "suggest_variant",
        "analyze_pattern",
        "complete_induction"
    ]
    
    INDUCTION_VARIANTS = {
        "simple": {
            "description": "Standard mathematical induction",
            "use_when": "Proving P(n) for all natural numbers",
            "structure": ["base_case", "inductive_hypothesis", "inductive_step"]
        },
        "strong": {
            "description": "Strong (complete) induction",
            "use_when": "Inductive step depends on multiple previous values",
            "structure": ["base_case", "strong_hypothesis", "inductive_step"]
        },
        "structural": {
            "description": "Structural induction",
            "use_when": "Proving properties of recursive data types",
            "structure": ["base_cases", "inductive_cases"]
        },
        "course_of_values": {
            "description": "Course-of-values induction",
            "use_when": "Need all previous cases, not just n",
            "structure": ["base_case", "course_hypothesis", "inductive_step"]
        },
        "transfinite": {
            "description": "Transfinite induction",
            "use_when": "Well-ordered sets beyond natural numbers",
            "structure": ["base_case", "successor_case", "limit_case"]
        },
        "double": {
            "description": "Double induction",
            "use_when": "Two variables that need simultaneous induction",
            "structure": ["double_base", "double_hypothesis", "double_step"]
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "setup_induction"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation in ["setup_induction", "identify_base_case", "formulate_hypothesis", 
                         "guide_inductive_step", "complete_induction"]:
            if "statement" not in inputs and "statement" not in self.config:
                errors.append(f"{operation} requires 'statement' input")
        
        if operation == "verify_structure":
            if "proof" not in inputs and "proof" not in self.config:
                errors.append("verify_structure requires 'proof' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "setup_induction",
                    "description": "Induction helper operation"
                },
                "statement": {
                    "type": "string",
                    "description": "Theorem statement to prove by induction"
                },
                "proof": {
                    "type": "string",
                    "description": "Proof to verify"
                },
                "induction_variable": {
                    "type": "string",
                    "default": "n",
                    "description": "Variable to induct on"
                },
                "variant": {
                    "type": "string",
                    "enum": list(self.INDUCTION_VARIANTS.keys()),
                    "default": "simple",
                    "description": "Induction variant to use"
                },
                "base_value": {
                    "type": "integer",
                    "default": 0,
                    "description": "Base case value (usually 0 or 1)"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute induction helper operation."""
        operation = inputs.get("operation", self.config.get("operation", "setup_induction"))
        
        try:
            if operation == "setup_induction":
                result = self._setup_induction(inputs, context)
            elif operation == "identify_base_case":
                result = self._identify_base_case(inputs, context)
            elif operation == "formulate_hypothesis":
                result = self._formulate_hypothesis(inputs, context)
            elif operation == "guide_inductive_step":
                result = self._guide_inductive_step(inputs, context)
            elif operation == "verify_structure":
                result = self._verify_structure(inputs, context)
            elif operation == "suggest_variant":
                result = self._suggest_variant(inputs, context)
            elif operation == "analyze_pattern":
                result = self._analyze_pattern(inputs, context)
            elif operation == "complete_induction":
                result = self._complete_induction(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("induction_helper_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Induction helper failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _setup_induction(self, inputs: Dict, context) -> Dict[str, Any]:
        """Set up induction proof structure."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        var = inputs.get("induction_variable", self.config.get("induction_variable", "n"))
        variant = inputs.get("variant", self.config.get("variant", "simple"))
        
        context.update_progress(50)
        
        variant_info = self.INDUCTION_VARIANTS.get(variant, self.INDUCTION_VARIANTS["simple"])
        
        structure = {
            "variant": variant,
            "description": variant_info["description"],
            "template": self._generate_template(statement, var, variant),
            "steps": variant_info["structure"]
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "statement": statement[:100] + "..." if len(statement) > 100 else statement,
            "induction_variable": var,
            "structure": structure
        }
    
    def _identify_base_case(self, inputs: Dict, context) -> Dict[str, Any]:
        """Identify the base case for induction."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        base_val = inputs.get("base_value", self.config.get("base_value", 0))
        var = inputs.get("induction_variable", self.config.get("induction_variable", "n"))
        
        context.update_progress(50)
        
        # Substitute base value into statement
        base_statement = self._substitute_variable(statement, var, str(base_val))
        
        context.update_progress(100)
        
        return {
            "success": True,
            "base_value": base_val,
            "base_case_statement": base_statement,
            "suggested_proof_approach": self._suggest_base_proof(base_statement),
            "verification": f"Verify that P({base_val}) holds"
        }
    
    def _formulate_hypothesis(self, inputs: Dict, context) -> Dict[str, Any]:
        """Formulate the inductive hypothesis."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        var = inputs.get("induction_variable", self.config.get("induction_variable", "n"))
        variant = inputs.get("variant", self.config.get("variant", "simple"))
        
        context.update_progress(50)
        
        if variant == "simple":
            hypothesis = f"Assume P({var}) holds: {self._substitute_variable(statement, var, var)}"
        elif variant == "strong":
            hypothesis = f"Assume P(k) holds for all k ≤ {var}"
        elif variant == "course_of_values":
            hypothesis = f"Assume P(k) holds for all k < {var}"
        else:
            hypothesis = f"Assume P({var}) holds"
        
        context.update_progress(100)
        
        return {
            "success": True,
            "inductive_hypothesis": hypothesis,
            "variant": variant,
            "assumption": f"P({var})"
        }
    
    def _guide_inductive_step(self, inputs: Dict, context) -> Dict[str, Any]:
        """Guide the inductive step."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        var = inputs.get("induction_variable", self.config.get("induction_variable", "n"))
        
        context.update_progress(50)
        
        # Goal for inductive step
        next_var = f"{var} + 1"
        goal = self._substitute_variable(statement, var, next_var)
        
        # Suggest approach
        suggestions = self._suggest_inductive_approach(statement, var)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "goal": f"Prove P({next_var})",
            "goal_statement": goal,
            "starting_point": f"Using P({var})...",
            "suggested_approaches": suggestions,
            "common_techniques": ["Algebraic manipulation", "Use definitions", "Apply lemmas"]
        }
    
    def _verify_structure(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify induction proof structure."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        checks = {
            "has_base_case": "base" in proof.lower() or "case" in proof.lower(),
            "has_inductive_hypothesis": "hypothesis" in proof.lower() or "assume" in proof.lower(),
            "has_inductive_step": "step" in proof.lower() or "induction" in proof.lower(),
            "uses_induction_tactic": "induction" in proof.lower() or "induct" in proof.lower(),
            "properly_terminated": "qed" in proof.lower() or "done" in proof.lower() or "sorry" not in proof.lower()
        }
        
        score = sum(checks.values()) / len(checks)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "structure_score": round(score, 2),
            "checks": checks,
            "is_valid": score >= 0.8,
            "suggestions": self._structure_suggestions(checks)
        }
    
    def _suggest_variant(self, inputs: Dict, context) -> Dict[str, Any]:
        """Suggest appropriate induction variant."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(50)
        
        # Analyze statement to suggest variant
        suggested = "simple"
        reason = "Standard pattern"
        
        if "recursive" in statement.lower() or "tree" in statement.lower():
            suggested = "structural"
            reason = "Statement involves recursive structure"
        elif "fibonacci" in statement.lower() or "two previous" in statement.lower():
            suggested = "strong"
            reason = "May depend on multiple previous values"
        elif "well-order" in statement.lower() or "ordinal" in statement.lower():
            suggested = "transfinite"
            reason = "Involves well-ordered sets"
        elif "pair" in statement.lower() or "double" in statement.lower():
            suggested = "double"
            reason = "Two interdependent variables"
        
        context.update_progress(100)
        
        variant_info = self.INDUCTION_VARIANTS[suggested]
        
        return {
            "success": True,
            "suggested_variant": suggested,
            "reason": reason,
            "description": variant_info["description"],
            "alternative_variants": [v for v in self.INDUCTION_VARIANTS.keys() if v != suggested][:2]
        }
    
    def _analyze_pattern(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze pattern for induction."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(50)
        
        # Extract pattern from statement
        pattern_analysis = {
            "recurrence_relation": self._detect_recurrence(statement),
            "closed_form_candidate": self._suggest_closed_form(statement),
            "pattern_type": self._classify_pattern(statement)
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "pattern_analysis": pattern_analysis,
            "suggests_induction": pattern_analysis["pattern_type"] in ["recursive", "iterative"]
        }
    
    def _complete_induction(self, inputs: Dict, context) -> Dict[str, Any]:
        """Generate complete induction outline."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        var = inputs.get("induction_variable", self.config.get("induction_variable", "n"))
        variant = inputs.get("variant", self.config.get("variant", "simple"))
        
        context.update_progress(30)
        
        # Generate complete outline
        base = self._identify_base_case(inputs, context)
        context.update_progress(50)
        
        hypothesis = self._formulate_hypothesis(inputs, context)
        context.update_progress(70)
        
        step = self._guide_inductive_step(inputs, context)
        context.update_progress(100)
        
        outline = f"""-- Proof by {variant} induction on {var}

-- Base case: {base['base_value']}
-- {base['base_case_statement']}
-- Proof: [to be filled]

-- Inductive hypothesis:
-- {hypothesis['inductive_hypothesis']}

-- Inductive step:
-- Goal: {step['goal']}
-- {step['goal_statement']}
-- Proof: [to be filled]

-- Therefore, by induction, {statement}
"""
        
        return {
            "success": True,
            "complete_outline": outline,
            "components": {
                "base_case": base,
                "hypothesis": hypothesis,
                "inductive_step": step
            },
            "variant_used": variant
        }
    
    def _generate_template(self, statement: str, var: str, variant: str) -> str:
        """Generate proof template."""
        templates = {
            "simple": f"""theorem by_induction : ∀ {var}, P({var}) := by
  intro {var}
  induction {var} with
  | zero =>
    -- Base case
    sorry
  | succ {var} ih =>
    -- Inductive step using ih
    sorry""",
            "strong": f"""theorem by_strong_induction : ∀ {var}, P({var}) := by
  intro {var}
  induction {var} using Nat.strongRecOn with
  | ind {var} ih =>
    -- Use ih for all k ≤ {var}
    sorry"""
        }
        return templates.get(variant, templates["simple"])
    
    def _substitute_variable(self, statement: str, var: str, value: str) -> str:
        """Substitute variable with value in statement."""
        # Simple substitution
        return statement.replace(var, value)
    
    def _suggest_base_proof(self, base_statement: str) -> str:
        """Suggest approach for proving base case."""
        if "0" in base_statement or "1" in base_statement:
            return "Direct computation or definition"
        elif "=" in base_statement:
            return "Reflexivity or normalization"
        else:
            return "Simplify and check"
    
    def _suggest_inductive_approach(self, statement: str, var: str) -> List[str]:
        """Suggest approaches for inductive step."""
        suggestions = []
        
        if "+" in statement:
            suggestions.append("Use associativity/commutativity of addition")
        if "*" in statement:
            suggestions.append("Distribute multiplication over addition")
        if "sum" in statement.lower():
            suggestions.append("Split sum: sum to n+1 = sum to n + (n+1)th term")
        if "divisible" in statement.lower() or "|" in statement:
            suggestions.append("Use properties of divisibility")
        
        if not suggestions:
            suggestions.append("Algebraic manipulation")
            suggestions.append("Apply definitions directly")
        
        return suggestions
    
    def _structure_suggestions(self, checks: Dict[str, bool]) -> List[str]:
        """Generate suggestions based on structure checks."""
        suggestions = []
        if not checks["has_base_case"]:
            suggestions.append("Add explicit base case")
        if not checks["has_inductive_hypothesis"]:
            suggestions.append("State inductive hypothesis clearly")
        if not checks["has_inductive_step"]:
            suggestions.append("Add inductive step section")
        if not checks["uses_induction_tactic"]:
            suggestions.append("Consider using induction tactic")
        return suggestions
    
    def _detect_recurrence(self, statement: str) -> Optional[str]:
        """Detect if statement involves recurrence."""
        if any(word in statement.lower() for word in ["recursive", "recurrence", "fibonacci"]):
            return "Likely recurrence relation"
        return None
    
    def _suggest_closed_form(self, statement: str) -> Optional[str]:
        """Suggest closed form candidate."""
        if "sum" in statement.lower():
            return "Look for arithmetic/geometric series formula"
        return None
    
    def _classify_pattern(self, statement: str) -> str:
        """Classify the pattern type."""
        if "recursive" in statement.lower():
            return "recursive"
        elif "iterative" in statement.lower() or "step" in statement.lower():
            return "iterative"
        elif "closed" in statement.lower():
            return "closed_form"
        return "general"
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
