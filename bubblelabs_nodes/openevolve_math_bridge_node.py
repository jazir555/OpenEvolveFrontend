"""
OpenEvolve-Math Bridge Node for BubbleLabs

Bridges OpenEvolve's problem-solving workflow with Mathematical Verification:
- Convert OpenEvolve problems to math verification tasks
- Route problems to appropriate math verification bubbles
- Integrate verification results back into OpenEvolve workflow
- Enable formal verification of decomposed subproblems

Part of the OpenEvolve-Math Integration Suite.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class OpenEvolveMathBridgeNode(BubbleLabsNode):
    """
    Bridge between OpenEvolve and Mathematical Verification bubbles.
    
    Operations:
        - route_to_verification: Route problem to math verification
        - convert_problem: Convert OpenEvolve problem to math format
        - integrate_result: Integrate verification result back
        - classify_and_route: Classify then route to appropriate verifier
        - verify_subproblem: Verify a decomposed subproblem
        - batch_verify: Verify multiple subproblems
        - formalize_solution: Formalize OpenEvolve solution
    """
    
    DISPLAY_NAME = "OpenEvolve-Math Bridge"
    DESCRIPTION = "Bridge OpenEvolve workflows with Mathematical Verification"
    ICON = "openevolve-math-bridge"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "route_to_verification",
        "convert_problem",
        "integrate_result",
        "classify_and_route",
        "verify_subproblem",
        "batch_verify",
        "formalize_solution"
    ]
    
    # Problem type to verification strategy mapping
    VERIFICATION_STRATEGIES = {
        "theorem": {
            "primary": "lean",
            "secondary": "z3",
            "pipeline": ["classify", "autoformalize", "lean_verify", "cross_check"]
        },
        "constraint": {
            "primary": "z3",
            "secondary": None,
            "pipeline": ["classify", "z3_solve"]
        },
        "proof": {
            "primary": "lean",
            "secondary": None,
            "pipeline": ["classify", "lean_check"]
        },
        "equation": {
            "primary": "z3",
            "secondary": "lean",
            "pipeline": ["classify", "z3_solve", "equivalence_check"]
        },
        "optimization": {
            "primary": "z3",
            "secondary": None,
            "pipeline": ["classify", "z3_optimize"]
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "route_to_verification"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation in ["route_to_verification", "convert_problem", "classify_and_route", "formalize_solution"]:
            if "problem" not in inputs and "problem" not in self.config:
                errors.append(f"{operation} requires 'problem' input")
        
        if operation == "verify_subproblem":
            if "subproblem" not in inputs and "subproblem" not in self.config:
                errors.append("verify_subproblem requires 'subproblem' input")
        
        if operation == "batch_verify":
            if "subproblems" not in inputs and "subproblems" not in self.config:
                errors.append("batch_verify requires 'subproblems' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "route_to_verification",
                    "description": "Bridge operation"
                },
                "problem": {
                    "type": "object",
                    "description": "OpenEvolve problem structure"
                },
                "subproblem": {
                    "type": "object",
                    "description": "Subproblem to verify"
                },
                "subproblems": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of subproblems"
                },
                "verification_result": {
                    "type": "object",
                    "description": "Math verification result to integrate"
                },
                "preferred_verifier": {
                    "type": "string",
                    "enum": ["lean", "z3", "auto"],
                    "default": "auto",
                    "description": "Preferred verification system"
                },
                "cross_verify": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use cross-verification when possible"
                },
                "autoformalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Autoformalize natural language"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute bridge operation."""
        operation = inputs.get("operation", self.config.get("operation", "route_to_verification"))
        
        try:
            if operation == "route_to_verification":
                result = self._route_to_verification(inputs, context)
            elif operation == "convert_problem":
                result = self._convert_problem(inputs, context)
            elif operation == "integrate_result":
                result = self._integrate_result(inputs, context)
            elif operation == "classify_and_route":
                result = self._classify_and_route(inputs, context)
            elif operation == "verify_subproblem":
                result = self._verify_subproblem(inputs, context)
            elif operation == "batch_verify":
                result = self._batch_verify(inputs, context)
            elif operation == "formalize_solution":
                result = self._formalize_solution(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            result["bridge_version"] = self.VERSION
            context.add_artifact("openevolve_math_bridge_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Bridge operation failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _route_to_verification(self, inputs: Dict, context) -> Dict[str, Any]:
        """Route OpenEvolve problem to math verification."""
        problem = inputs.get("problem", self.config.get("problem", {}))
        preferred = inputs.get("preferred_verifier", self.config.get("preferred_verifier", "auto"))
        cross_verify = inputs.get("cross_verify", self.config.get("cross_verify", True))
        
        context.update_progress(30)
        
        # Extract problem text
        problem_text = problem.get("description", problem.get("text", str(problem)))
        problem_type = problem.get("type", "general")
        
        context.update_progress(60)
        
        # Determine verification strategy
        strategy = self._determine_strategy(problem_type, problem_text, preferred)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "problem_id": problem.get("id", "unknown"),
            "problem_text": problem_text[:200] + "..." if len(problem_text) > 200 else problem_text,
            "routing_decision": {
                "primary_verifier": strategy["primary"],
                "secondary_verifier": strategy["secondary"] if cross_verify else None,
                "pipeline": strategy["pipeline"],
                "reasoning": f"Problem type '{problem_type}' routed to {strategy['primary']}"
            },
            "next_steps": self._generate_next_steps(strategy)
        }
    
    def _convert_problem(self, inputs: Dict, context) -> Dict[str, Any]:
        """Convert OpenEvolve problem to math verification format."""
        problem = inputs.get("problem", self.config.get("problem", {}))
        autoformalize = inputs.get("autoformalize", self.config.get("autoformalize", True))
        
        context.update_progress(50)
        
        # Extract and convert
        converted = {
            "original_id": problem.get("id"),
            "title": problem.get("title", "Untitled"),
            "description": problem.get("description", ""),
            "constraints": problem.get("constraints", []),
            "requirements": problem.get("requirements", []),
        }
        
        # Generate math-specific fields
        converted["math_statement"] = self._to_math_statement(problem)
        converted["formalization_ready"] = autoformalize
        converted["verification_targets"] = self._extract_verification_targets(problem)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "original": problem,
            "converted": converted,
            "format": "math_verification",
            "ready_for": "lean_autoformalization" if autoformalize else "direct_verification"
        }
    
    def _integrate_result(self, inputs: Dict, context) -> Dict[str, Any]:
        """Integrate verification result back into OpenEvolve format."""
        verification_result = inputs.get("verification_result", self.config.get("verification_result", {}))
        problem_id = inputs.get("problem_id", self.config.get("problem_id", "unknown"))
        
        context.update_progress(50)
        
        # Convert verification result to OpenEvolve format
        integrated = {
            "problem_id": problem_id,
            "verification_status": self._map_status(verification_result.get("status", "unknown")),
            "confidence": verification_result.get("confidence", 0.0),
            "formal_proof": verification_result.get("proof", verification_result.get("lean_code", "")),
            "verification_system": verification_result.get("system", "unknown"),
            "verified_at": datetime.utcnow().isoformat()
        }
        
        # Add quality metrics
        integrated["quality_score"] = self._compute_quality_score(verification_result)
        integrated["reliability"] = "high" if integrated["confidence"] > 0.9 else "medium" if integrated["confidence"] > 0.7 else "low"
        
        context.update_progress(100)
        
        return {
            "success": True,
            "verification_result": verification_result,
            "openevolve_format": integrated,
            "integration_complete": True
        }
    
    def _classify_and_route(self, inputs: Dict, context) -> Dict[str, Any]:
        """Classify problem and route to appropriate verifier."""
        problem = inputs.get("problem", self.config.get("problem", {}))
        
        context.update_progress(30)
        
        # Step 1: Extract problem text
        problem_text = problem.get("description", str(problem))
        
        context.update_progress(50)
        
        # Step 2: Classify (would call MathProblemClassificationNode)
        classification = self._mock_classify(problem_text)
        
        context.update_progress(70)
        
        # Step 3: Route based on classification
        routing = self._route_based_on_classification(classification)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "classification": classification,
            "routing": routing,
            "recommended_pipeline": self._build_pipeline(classification, routing)
        }
    
    def _verify_subproblem(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify a decomposed subproblem."""
        subproblem = inputs.get("subproblem", self.config.get("subproblem", {}))
        
        context.update_progress(50)
        
        # Convert subproblem to verification format
        subproblem_text = subproblem.get("description", str(subproblem))
        
        # Determine if formal verification is appropriate
        needs_formal = self._needs_formal_verification(subproblem)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "subproblem_id": subproblem.get("id", "unknown"),
            "needs_formal_verification": needs_formal,
            "verification_type": "lean" if needs_formal else "z3_quick",
            "statement": subproblem_text[:200] + "..." if len(subproblem_text) > 200 else subproblem_text,
            "suggested_verifier": "lean_proof_checking" if needs_formal else "z3_constraint_solving"
        }
    
    def _batch_verify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify multiple subproblems."""
        subproblems = inputs.get("subproblems", self.config.get("subproblems", []))
        
        results = []
        total = len(subproblems)
        
        for i, sub in enumerate(subproblems):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._verify_subproblem({"subproblem": sub}, context)
            results.append({
                "subproblem_id": sub.get("id", f"sp_{i}"),
                "verification_type": result.get("verification_type"),
                "needs_formal": result.get("needs_formal_verification")
            })
        
        formal_count = sum(1 for r in results if r["needs_formal"])
        
        return {
            "success": True,
            "total": total,
            "formal_verification_needed": formal_count,
            "quick_check_sufficient": total - formal_count,
            "results": results,
            "batch_recommendation": f"Use pipeline: {formal_count} formal, {total - formal_count} quick"
        }
    
    def _formalize_solution(self, inputs: Dict, context) -> Dict[str, Any]:
        """Formalize OpenEvolve solution."""
        solution = inputs.get("solution", self.config.get("solution", {}))
        
        context.update_progress(50)
        
        # Extract solution components
        solution_text = solution.get("text", solution.get("description", str(solution)))
        
        # Determine formalization approach
        formalization = {
            "original_solution": solution_text[:300] + "..." if len(solution_text) > 300 else solution_text,
            "can_formalize": len(solution_text) > 10,
            "suggested_approach": "lean_autoformalization",
            "expected_output": "lean_code",
            "verification_steps": ["autoformalize", "type_check", "proof_check"]
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "solution_id": solution.get("id", "unknown"),
            "formalization": formalization,
            "ready_for_verification": formalization["can_formalize"]
        }
    
    def _determine_strategy(self, problem_type: str, problem_text: str, preferred: str) -> Dict:
        """Determine verification strategy."""
        # Check for explicit type match
        if problem_type in self.VERIFICATION_STRATEGIES:
            strategy = self.VERIFICATION_STRATEGIES[problem_type].copy()
        else:
            # Infer from text
            strategy = self._infer_strategy(problem_text)
        
        # Override with preference if specified
        if preferred != "auto":
            strategy["primary"] = preferred
        
        return strategy
    
    def _infer_strategy(self, text: str) -> Dict:
        """Infer verification strategy from problem text."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["theorem", "prove", "lemma", "corollary"]):
            return self.VERIFICATION_STRATEGIES["theorem"]
        elif any(w in text_lower for w in ["constraint", "satisfy", "solve for"]):
            return self.VERIFICATION_STRATEGIES["constraint"]
        elif any(w in text_lower for w in ["proof", "show that"]):
            return self.VERIFICATION_STRATEGIES["proof"]
        elif any(w in text_lower for w in ["equation", "solve"]):
            return self.VERIFICATION_STRATEGIES["equation"]
        elif any(w in text_lower for w in ["optimize", "minimize", "maximize"]):
            return self.VERIFICATION_STRATEGIES["optimization"]
        else:
            return self.VERIFICATION_STRATEGIES["theorem"]  # Default
    
    def _generate_next_steps(self, strategy: Dict) -> List[str]:
        """Generate next steps based on strategy."""
        steps = []
        
        for step in strategy["pipeline"]:
            if step == "classify":
                steps.append("MathProblemClassificationNode: Classify problem")
            elif step == "autoformalize":
                steps.append("LeanAutoformalizationNode: Convert to formal language")
            elif step == "lean_verify":
                steps.append("LeanProofCheckingNode: Verify with Lean")
            elif step == "z3_solve":
                steps.append("Z3ConstraintSolvingNode: Solve constraints")
            elif step == "cross_check":
                steps.append("MathVerificationPipelineNode: Cross-verify")
        
        return steps
    
    def _to_math_statement(self, problem: Dict) -> str:
        """Convert problem to math statement."""
        parts = []
        
        if "title" in problem:
            parts.append(f"** {problem['title']} **")
        if "description" in problem:
            parts.append(problem["description"])
        if "constraints" in problem:
            parts.append("Constraints: " + "; ".join(problem["constraints"]))
        
        return "\n".join(parts)
    
    def _extract_verification_targets(self, problem: Dict) -> List[str]:
        """Extract targets for verification."""
        targets = []
        
        if "requirements" in problem:
            targets.extend(problem["requirements"])
        if "constraints" in problem:
            targets.append("All constraints satisfied")
        if "expected_output" in problem:
            targets.append("Output correctness")
        
        return targets
    
    def _map_status(self, status: str) -> str:
        """Map verification status to OpenEvolve format."""
        mapping = {
            "verified": "verified",
            "proven": "verified",
            "sat": "satisfied",
            "unsat": "failed",
            "unknown": "pending",
            "error": "error"
        }
        return mapping.get(status.lower(), "unknown")
    
    def _compute_quality_score(self, result: Dict) -> float:
        """Compute quality score from verification result."""
        score = 0.0
        
        if result.get("verified") or result.get("proven"):
            score += 0.5
        if result.get("confidence"):
            score += result["confidence"] * 0.5
        
        return round(min(score, 1.0), 3)
    
    def _mock_classify(self, text: str) -> Dict:
        """Mock classification (would call actual classifier)."""
        return {
            "domain": "algebra",
            "type": "theorem",
            "difficulty": "intermediate",
            "confidence": 0.8
        }
    
    def _route_based_on_classification(self, classification: Dict) -> Dict:
        """Route based on classification."""
        domain = classification.get("domain", "general")
        ptype = classification.get("type", "theorem")
        
        if ptype == "theorem":
            return {"primary": "lean", "reason": "Theorem requires formal proof"}
        elif domain in ["algebra", "arithmetic"]:
            return {"primary": "z3", "reason": "Algebraic - good for SMT"}
        else:
            return {"primary": "hybrid", "reason": "Complex - use both systems"}
    
    def _build_pipeline(self, classification: Dict, routing: Dict) -> List[str]:
        """Build verification pipeline."""
        pipeline = ["MathProblemClassificationNode"]
        
        if routing["primary"] == "lean":
            pipeline.extend([
                "LeanAutoformalizationNode",
                "LeanProofCheckingNode"
            ])
        elif routing["primary"] == "z3":
            pipeline.extend([
                "Z3ConstraintSolvingNode",
                "Z3TheoremProvingNode"
            ])
        else:
            pipeline.extend([
                "MathVerificationPipelineNode"
            ])
        
        pipeline.append("MathVerificationDashboardNode")
        
        return pipeline
    
    def _needs_formal_verification(self, subproblem: Dict) -> bool:
        """Determine if subproblem needs formal verification."""
        text = subproblem.get("description", "").lower()
        
        # Indicators that formal proof is needed
        formal_indicators = [
            "prove", "theorem", "lemma", "correctness", "verify"
        ]
        
        return any(ind in text for ind in formal_indicators)
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
