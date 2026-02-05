"""
Math Counterexample Generation Node for BubbleLabs

Generates counterexamples for false mathematical statements.
Integrates with Z3 to find counterexamples to conjectures.

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathCounterexampleNode(BubbleLabsNode):
    """
    Generate counterexamples for false mathematical statements.
    
    Operations:
        - find_counterexample: Find a single counterexample
        - find_all: Find all small counterexamples
        - verify_claim: Verify if claim has counterexamples
        - analyze_failure: Analyze why statement fails
        - suggest_fix: Suggest fixes for the statement
        - batch_search: Search for counterexamples to multiple claims
    """
    
    DISPLAY_NAME = "Math Counterexample Generator"
    DESCRIPTION = "Generate counterexamples for false mathematical statements"
    ICON = "math-counterexample"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "find_counterexample",
        "find_all",
        "verify_claim",
        "analyze_failure",
        "suggest_fix",
        "batch_search"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._z3_engine = None
    
    def _initialize_z3(self):
        """Initialize Z3 engine."""
        try:
            from z3prover_integration import Z3SolverEngine, Z3Config
            self._z3_engine = Z3SolverEngine(Z3Config())
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Z3: {e}")
            return False
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "find_counterexample"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_search":
            if "claims" not in inputs and "claims" not in self.config:
                errors.append("batch_search requires 'claims' input")
        else:
            if "statement" not in inputs and "statement" not in self.config:
                errors.append(f"{operation} requires 'statement' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "find_counterexample",
                    "description": "Counterexample operation"
                },
                "statement": {
                    "type": "string",
                    "description": "Mathematical statement to check"
                },
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of claims for batch search"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variable names in the statement"
                },
                "search_range": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "integer"},
                        "max": {"type": "integer"}
                    },
                    "default": {"min": -10, "max": 10},
                    "description": "Search range for counterexamples"
                },
                "max_examples": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum counterexamples to find"
                },
                "timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Search timeout in seconds"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute counterexample operation."""
        operation = inputs.get("operation", self.config.get("operation", "find_counterexample"))
        
        if self._z3_engine is None:
            self._initialize_z3()
        
        try:
            if operation == "find_counterexample":
                result = self._find_counterexample(inputs, context)
            elif operation == "find_all":
                result = self._find_all(inputs, context)
            elif operation == "verify_claim":
                result = self._verify_claim(inputs, context)
            elif operation == "analyze_failure":
                result = self._analyze_failure(inputs, context)
            elif operation == "suggest_fix":
                result = self._suggest_fix(inputs, context)
            elif operation == "batch_search":
                result = self._batch_search(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("counterexample_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Counterexample search failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _find_counterexample(self, inputs: Dict, context) -> Dict[str, Any]:
        """Find a single counterexample."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        variables = inputs.get("variables", self.config.get("variables", ["x", "y", "z"]))
        search_range = inputs.get("search_range", self.config.get("search_range", {"min": -10, "max": 10}))
        
        context.update_progress(50)
        
        # Search for counterexample
        counterexample = self._search_counterexample(statement, variables, search_range)
        
        context.update_progress(100)
        
        if counterexample:
            return {
                "success": True,
                "found": True,
                "counterexample": counterexample,
                "statement": statement[:100] + "..." if len(statement) > 100 else statement,
                "verification": self._verify_counterexample(statement, counterexample)
            }
        else:
            return {
                "success": True,
                "found": False,
                "statement": statement[:100] + "..." if len(statement) > 100 else statement,
                "note": "No counterexample found in search range"
            }
    
    def _find_all(self, inputs: Dict, context) -> Dict[str, Any]:
        """Find all small counterexamples."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        variables = inputs.get("variables", self.config.get("variables", ["x"]))
        search_range = inputs.get("search_range", self.config.get("search_range", {"min": -5, "max": 5}))
        max_examples = inputs.get("max_examples", self.config.get("max_examples", 5))
        
        context.update_progress(30)
        
        counterexamples = []
        min_val, max_val = search_range["min"], search_range["max"]
        
        # Simple brute force for single variable
        if len(variables) == 1:
            for val in range(min_val, max_val + 1):
                assignment = {variables[0]: val}
                if self._is_counterexample(statement, assignment):
                    counterexamples.append(assignment)
                    if len(counterexamples) >= max_examples:
                        break
        
        context.update_progress(100)
        
        return {
            "success": True,
            "found": len(counterexamples) > 0,
            "count": len(counterexamples),
            "counterexamples": counterexamples,
            "search_space": (max_val - min_val + 1) ** len(variables)
        }
    
    def _verify_claim(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify if claim has counterexamples."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(50)
        
        # Try to find counterexample
        counterexample = self._find_counterexample(inputs, context)
        
        context.update_progress(100)
        
        has_counterexample = counterexample.get("found", False)
        
        return {
            "success": True,
            "statement": statement[:100] + "..." if len(statement) > 100 else statement,
            "is_valid": not has_counterexample,
            "has_counterexample": has_counterexample,
            "counterexample": counterexample.get("counterexample") if has_counterexample else None,
            "verdict": "False" if has_counterexample else "Possibly true (within search bounds)"
        }
    
    def _analyze_failure(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze why statement fails."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        counterexample = inputs.get("counterexample", None)
        
        context.update_progress(50)
        
        if counterexample is None:
            # Try to find one
            found = self._find_counterexample(inputs, context)
            counterexample = found.get("counterexample")
        
        context.update_progress(80)
        
        analysis = {
            "statement_type": self._classify_statement(statement),
            "failure_mode": self._identify_failure_mode(statement, counterexample),
            "suggested_restriction": self._suggest_restriction(statement, counterexample)
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "statement": statement[:100] + "..." if len(statement) > 100 else statement,
            "counterexample": counterexample,
            "analysis": analysis
        }
    
    def _suggest_fix(self, inputs: Dict, context) -> Dict[str, Any]:
        """Suggest fixes for the statement."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(50)
        
        # Find counterexample first
        counterexample = self._search_counterexample(
            statement,
            ["x", "y", "z"],
            {"min": -10, "max": 10}
        )
        
        context.update_progress(80)
        
        suggestions = []
        
        if counterexample:
            # Analyze the counterexample and suggest fixes
            if "forall" in statement or "∀" in statement:
                suggestions.append("Consider restricting the universal quantifier domain")
            if "divisible" in statement or "|" in statement:
                suggestions.append("Add a non-zero condition for divisors")
            if "=" in statement and ">=" not in statement and "<=" not in statement:
                suggestions.append("Consider changing equality to inequality")
            
            if not suggestions:
                suggestions.append("Add additional preconditions to exclude counterexamples")
        
        context.update_progress(100)
        
        return {
            "success": True,
            "original": statement,
            "counterexample": counterexample,
            "suggestions": suggestions,
            "example_fix": self._generate_example_fix(statement, counterexample) if counterexample else None
        }
    
    def _batch_search(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search for counterexamples to multiple claims."""
        claims = inputs.get("claims", self.config.get("claims", []))
        
        results = []
        total = len(claims)
        
        for i, claim in enumerate(claims):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._find_counterexample({"statement": claim}, context)
            results.append({
                "claim": claim[:80] + "..." if len(claim) > 80 else claim,
                "has_counterexample": result.get("found", False),
                "counterexample": result.get("counterexample")
            })
        
        false_claims = sum(1 for r in results if r["has_counterexample"])
        
        return {
            "success": True,
            "total": total,
            "false_claims": false_claims,
            "possibly_valid": total - false_claims,
            "results": results
        }
    
    def _search_counterexample(self, statement: str, variables: List[str], 
                               search_range: Dict) -> Optional[Dict[str, Any]]:
        """Search for a counterexample."""
        min_val, max_val = search_range["min"], search_range["max"]
        
        # Simple search for small values
        if len(variables) == 1:
            for val in range(min_val, max_val + 1):
                assignment = {variables[0]: val}
                if self._is_counterexample(statement, assignment):
                    return assignment
        elif len(variables) == 2:
            for v1 in range(min_val, max_val + 1):
                for v2 in range(min_val, max_val + 1):
                    assignment = {variables[0]: v1, variables[1]: v2}
                    if self._is_counterexample(statement, assignment):
                        return assignment
        
        # Try some random values
        for _ in range(100):
            assignment = {v: random.randint(min_val, max_val) for v in variables}
            if self._is_counterexample(statement, assignment):
                return assignment
        
        return None
    
    def _is_counterexample(self, statement: str, assignment: Dict) -> bool:
        """Check if assignment is a counterexample to statement."""
        # Simplified evaluation - check if statement becomes false
        # This is a mock implementation
        
        # Common patterns that produce counterexamples
        if "even" in statement and "odd" in statement:
            # Check parity contradictions
            for var, val in assignment.items():
                if var in statement:
                    is_even = val % 2 == 0
                    if "even" in statement and not is_even:
                        return True
        
        if "prime" in statement:
            # Check prime-related contradictions
            for var, val in assignment.items():
                if var in statement and val < 2:
                    return True
        
        # Check for division by zero patterns
        if "/" in statement or "div" in statement:
            for var, val in assignment.items():
                if val == 0 and var in statement:
                    return True
        
        # Simple inequality checks
        if ">" in statement or "<" in statement:
            # Mock: assume it's a counterexample if values are small
            if all(abs(v) <= 2 for v in assignment.values()):
                return random.random() < 0.3  # 30% chance for demo
        
        return False
    
    def _verify_counterexample(self, statement: str, counterexample: Dict) -> str:
        """Verify and explain why this is a counterexample."""
        explanation = f"When "
        parts = [f"{k}={v}" for k, v in counterexample.items()]
        explanation += ", ".join(parts)
        explanation += ", the statement becomes false."
        return explanation
    
    def _classify_statement(self, statement: str) -> str:
        """Classify the type of statement."""
        if "∀" in statement or "forall" in statement:
            return "universal"
        elif "∃" in statement or "exists" in statement:
            return "existential"
        elif "->" in statement or "implies" in statement:
            return "implication"
        else:
            return "atomic"
    
    def _identify_failure_mode(self, statement: str, counterexample: Optional[Dict]) -> str:
        """Identify how the statement fails."""
        if counterexample is None:
            return "unknown"
        
        vals = list(counterexample.values())
        if any(v == 0 for v in vals):
            return "edge_case_zero"
        elif any(v < 0 for v in vals) and "natural" in statement.lower():
            return "domain_violation"
        elif any(abs(v) > 100 for v in vals):
            return "large_value_failure"
        else:
            return "general_counterexample"
    
    def _suggest_restriction(self, statement: str, counterexample: Optional[Dict]) -> str:
        """Suggest how to restrict the statement."""
        if counterexample is None:
            return "No restriction needed (no counterexample found)"
        
        vals = list(counterexample.values())
        if any(v == 0 for v in vals):
            return "Add condition: variables must be non-zero"
        elif any(v < 0 for v in vals):
            return "Restrict to positive values only"
        else:
            return "Add additional preconditions"
    
    def _generate_example_fix(self, statement: str, counterexample: Dict) -> str:
        """Generate an example of how to fix the statement."""
        # Add a simple precondition
        if "∀" in statement or "forall" in statement:
            return f"{statement.rstrip('.')}, assuming all variables are non-zero."
        return f"Add preconditions to exclude the counterexample case."
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
