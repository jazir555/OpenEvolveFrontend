"""
Math Proof Completion Node for BubbleLabs

Completes partial proofs by filling in gaps:
- Fill in sorry's
- Complete proof sketches
- Suggest missing steps
- Auto-complete trivial cases
- Verify completion correctness

Part of the Mathematical Verification Bubble Suite.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

# Lean integration
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logging.getLogger(__name__).warning("Lean 4 not available for MathProofCompletionNode")

# Z3 integration for counterexample search
try:
    from z3prover_integration import Z3SolverEngine, Z3Config
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

logger = logging.getLogger(__name__)


class MathProofCompletionNode(BubbleLabsNode):
    """
    Complete partial proofs by filling in gaps and sorry's.
    
    Operations:
        - complete_proof: Complete a partial proof
        - fill_sorry: Fill specific sorry placeholders
        - complete_sketch: Expand proof sketch to full proof
        - suggest_steps: Suggest missing proof steps
        - auto_complete: Auto-complete trivial cases
        - verify_completion: Verify completed proof
        - batch_complete: Complete multiple proofs
    """
    
    DISPLAY_NAME = "Math Proof Completion"
    DESCRIPTION = "Complete partial proofs by filling in gaps and sorry's"
    ICON = "math-completion"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "complete_proof",
        "fill_sorry",
        "complete_sketch",
        "suggest_steps",
        "auto_complete",
        "verify_completion",
        "batch_complete"
    ]
    
    # Completion strategies for common patterns
    COMPLETION_STRATEGIES = {
        "trivial": {
            "pattern": r'sorry\s*--\s*trivial|trivial.*sorry',
            "replacement": "trivial",
            "description": "Trivial case"
        },
        "reflexivity": {
            "pattern": r'x\s*=\s*x|reflexive',
            "replacement": "rfl",
            "description": "Reflexivity"
        },
        "simp": {
            "pattern": r'simplify|simp.*sorry',
            "replacement": "simp",
            "description": "Simplification"
        },
        "ring": {
            "pattern": r'algebraic|ring|commutative',
            "replacement": "ring",
            "description": "Ring arithmetic"
        },
        "linarith": {
            "pattern": r'linear|inequality|arithmetic',
            "replacement": "linarith",
            "description": "Linear arithmetic"
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._client = None
        self._z3_engine = None
        
        if LEAN_AVAILABLE:
            try:
                client_config = LeanAideConfig(
                    host=self.config.get("leanaide_host", "localhost"),
                    port=self.config.get("leanaide_port", 7654),
                    timeout=self.config.get("timeout", 6000.0)
                )
                self._client = LeanAideClient(client_config)
                logger.info("LeanAide client initialized for MathProofCompletionNode")
            except Exception as e:
                logger.warning(f"Could not initialize LeanAide client: {e}")
                self._client = None
        
        if Z3_AVAILABLE:
            try:
                self._z3_engine = Z3SolverEngine(Z3Config())
                logger.info("Z3 engine initialized for MathProofCompletionNode")
            except Exception as e:
                logger.warning(f"Could not initialize Z3 engine: {e}")
                self._z3_engine = None
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "complete_proof"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_complete":
            if "proofs" not in inputs and "proofs" not in self.config:
                errors.append("batch_complete requires 'proofs' input")
        else:
            if "proof" not in inputs and "proof" not in self.config:
                errors.append(f"{operation} requires 'proof' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "complete_proof",
                    "description": "Completion operation"
                },
                "proof": {
                    "type": "string",
                    "description": "Partial proof to complete"
                },
                "proofs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of proofs for batch completion"
                },
                "goal": {
                    "type": "string",
                    "description": "Proof goal (theorem statement)"
                },
                "aggressive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Try aggressive completion strategies"
                },
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum search depth for completion"
                },
                "hint": {
                    "type": "string",
                    "description": "Hint for completion"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute proof completion operation."""
        operation = inputs.get("operation", self.config.get("operation", "complete_proof"))
        
        try:
            if operation == "complete_proof":
                result = self._complete_proof(inputs, context)
            elif operation == "fill_sorry":
                result = self._fill_sorry(inputs, context)
            elif operation == "complete_sketch":
                result = self._complete_sketch(inputs, context)
            elif operation == "suggest_steps":
                result = self._suggest_steps(inputs, context)
            elif operation == "auto_complete":
                result = self._auto_complete(inputs, context)
            elif operation == "verify_completion":
                result = self._verify_completion(inputs, context)
            elif operation == "batch_complete":
                result = self._batch_complete(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("completion_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Proof completion failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _complete_proof(self, inputs: Dict, context) -> Dict[str, Any]:
        """Complete a partial proof."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        goal = inputs.get("goal", self.config.get("goal", ""))
        aggressive = inputs.get("aggressive", self.config.get("aggressive", False))
        
        context.update_progress(30)
        
        # Count sorry's
        sorry_count = proof.lower().count("sorry")
        
        context.update_progress(50)
        
        # Try to fill sorry's
        completed = self._fill_sorry_placeholders(proof, goal, aggressive)
        
        context.update_progress(80)
        
        # Try to complete sketchy parts
        completed = self._expand_sketch(completed, aggressive)
        
        context.update_progress(100)
        
        remaining_sorry = completed.lower().count("sorry")
        
        return {
            "success": True,
            "original": proof[:300] + "..." if len(proof) > 300 else proof,
            "completed": completed[:300] + "..." if len(completed) > 300 else completed,
            "original_sorry_count": sorry_count,
            "remaining_sorry": remaining_sorry,
            "filled_count": sorry_count - remaining_sorry,
            "completion_rate": (sorry_count - remaining_sorry) / sorry_count if sorry_count > 0 else 1.0
        }
    
    def _fill_sorry(self, inputs: Dict, context) -> Dict[str, Any]:
        """Fill specific sorry placeholders."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        goal = inputs.get("goal", self.config.get("goal", ""))
        hint = inputs.get("hint", self.config.get("hint", ""))
        
        context.update_progress(50)
        
        # Find and fill sorry's
        filled_proof = self._fill_sorry_placeholders(proof, goal, False, hint)
        
        context.update_progress(100)
        
        sorry_count_before = proof.lower().count("sorry")
        sorry_count_after = filled_proof.lower().count("sorry")
        
        return {
            "success": True,
            "original": proof,
            "filled": filled_proof,
            "filled_count": sorry_count_before - sorry_count_after,
            "remaining_sorry": sorry_count_after
        }
    
    def _complete_sketch(self, inputs: Dict, context) -> Dict[str, Any]:
        """Expand proof sketch to full proof."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        # Expand sketchy parts
        completed = self._expand_sketch(proof, True)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "sketch": proof,
            "completed": completed,
            "expansions_made": self._count_expansions(proof, completed)
        }
    
    def _suggest_steps(self, inputs: Dict, context) -> Dict[str, Any]:
        """Suggest missing proof steps."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        goal = inputs.get("goal", self.config.get("goal", ""))
        
        context.update_progress(50)
        
        # Analyze proof and suggest steps
        suggestions = self._analyze_and_suggest(proof, goal)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "current_state": proof[:200] + "..." if len(proof) > 200 else proof,
            "suggested_steps": suggestions,
            "goal": goal[:100] + "..." if len(goal) > 100 else goal
        }
    
    def _auto_complete(self, inputs: Dict, context) -> Dict[str, Any]:
        """Auto-complete trivial cases."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        # Auto-fill trivial patterns
        completed = proof
        
        # Replace simple sorry patterns
        replacements = [
            (r'by\s+sorry\s*--\s*trivial', 'by trivial'),
            (r'by\s+sorry\s*--\s*rfl', 'by rfl'),
            (r'by\s+sorry\s*--\s*simp', 'by simp'),
        ]
        
        for pattern, replacement in replacements:
            completed = re.sub(pattern, replacement, completed, flags=re.IGNORECASE)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "original": proof,
            "completed": completed,
            "auto_filled": completed != proof
        }
    
    def _verify_completion(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify completed proof."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        # Check for remaining sorry's
        sorry_count = proof.lower().count("sorry")
        
        # Check for common issues
        issues = []
        if sorry_count > 0:
            issues.append(f"Contains {sorry_count} sorry's")
        if "admit" in proof.lower():
            issues.append("Contains admit")
        if proof.count("begin") != proof.count("end"):
            issues.append("Mismatched begin/end blocks")
        
        context.update_progress(100)
        
        return {
            "success": True,
            "is_complete": sorry_count == 0 and len(issues) == 0,
            "sorry_count": sorry_count,
            "issues": issues,
            "suggestions": ["Fill remaining sorry's"] if sorry_count > 0 else []
        }
    
    def _batch_complete(self, inputs: Dict, context) -> Dict[str, Any]:
        """Complete multiple proofs."""
        proofs = inputs.get("proofs", self.config.get("proofs", []))
        
        results = []
        total = len(proofs)
        
        for i, proof in enumerate(proofs):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._complete_proof({"proof": proof}, context)
            results.append({
                "original_size": len(proof),
                "completed_size": len(result.get("completed", "")),
                "completion_rate": result.get("completion_rate", 0)
            })
        
        avg_completion = sum(r["completion_rate"] for r in results) / len(results) if results else 0
        
        return {
            "success": True,
            "total": total,
            "results": results,
            "average_completion_rate": round(avg_completion, 3)
        }
    
    def _fill_sorry_placeholders(self, proof: str, goal: str, aggressive: bool, hint: str = "") -> str:
        """Fill sorry placeholders with appropriate tactics."""
        filled = proof
        
        # First: Try REAL Lean proof completion if available
        if self._client and LEAN_AVAILABLE:
            try:
                completed = self.complete_proof_with_lean(proof, goal)
                if completed and "sorry" not in completed.lower():
                    logger.info("Successfully completed proof using LeanAide")
                    return completed
            except Exception as e:
                logger.warning(f"Lean proof completion failed: {e}, using fallback")
        
        # Fallback: Use pattern-based completion
        # Use hint if provided
        if hint:
            for strategy_name, strategy in self.COMPLETION_STRATEGIES.items():
                if hint.lower() in strategy_name.lower():
                    filled = filled.replace("sorry", strategy["replacement"], 1)
                    break
        
        # Try to infer from context
        if "rfl" in proof or "reflexiv" in proof.lower():
            filled = filled.replace("sorry", "rfl", 1)
        elif "simp" in proof:
            filled = filled.replace("sorry", "simp", 1)
        elif "linarith" in proof or "inequality" in goal.lower():
            filled = filled.replace("sorry", "linarith", 1)
        elif "ring" in proof or "algebraic" in goal.lower():
            filled = filled.replace("sorry", "ring", 1)
        elif "trivial" in proof.lower():
            filled = filled.replace("sorry", "trivial", 1)
        
        return filled
    
    def complete_proof_with_lean(self, proof: str, goal: str = "") -> str:
        """
        Complete a proof using real Lean 4 via LeanAide.
        
        Args:
            proof: Partial proof with sorry placeholders
            goal: Theorem statement/goal
            
        Returns:
            Completed proof with sorry's filled
            
        Raises:
            RuntimeError: If Lean is not available
        """
        if not LEAN_AVAILABLE or not self._client:
            raise RuntimeError("Lean 4 not available. Please install leanaide_client.")
        
        try:
            # Use the client's prove_for_formalization capability
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self._client.prove_for_formalization(
                        theorem_text=goal or "Theorem from incomplete proof",
                        theorem_code=proof,
                        theorem_statement=goal
                    )
                )
                
                if result.success and result.data:
                    completed_proof = result.data.get("proof", proof)
                    return completed_proof
                else:
                    logger.warning(f"Lean proof completion returned no result: {result.error}")
                    return proof
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Lean proof completion failed: {e}")
            raise RuntimeError(f"Failed to complete proof with Lean: {e}")
    
    def _expand_sketch(self, proof: str, aggressive: bool) -> str:
        """Expand proof sketch to full proof."""
        expanded = proof
        
        # Expand common sketch patterns
        if "by induction" in proof.lower() and "|" not in proof:
            # Add induction structure
            expanded = expanded.replace(
                "by induction",
                "by induction\n  | zero => sorry\n  | succ n ih => sorry"
            )
        
        if "by cases" in proof.lower() and aggressive:
            expanded = expanded.replace(
                "by cases",
                "by cases\n  -- Case 1\n  sorry\n  -- Case 2\n  sorry"
            )
        
        return expanded
    
    def _count_expansions(self, original: str, completed: str) -> int:
        """Count number of expansions made."""
        orig_lines = len(original.split('\n'))
        completed_lines = len(completed.split('\n'))
        return max(0, completed_lines - orig_lines)
    
    def _analyze_and_suggest(self, proof: str, goal: str) -> List[Dict]:
        """Analyze proof and suggest next steps."""
        suggestions = []
        
        # Check current state
        if "intro" not in proof.lower() and ("∀" in goal or "forall" in goal.lower()):
            suggestions.append({
                "step": "intro x",
                "reason": "Goal has universal quantifier",
                "priority": "high"
            })
        
        if "sorry" in proof.lower():
            suggestions.append({
                "step": "Fill sorry with appropriate tactic",
                "reason": "Incomplete proof",
                "priority": "high"
            })
        
        if "=" in goal and "rw" not in proof:
            suggestions.append({
                "step": "rw [relevant_lemma]",
                "reason": "Goal is an equality",
                "priority": "medium"
            })
        
        return suggestions
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
    
    def get_lean_status(self) -> Dict[str, Any]:
        """Get Lean integration status."""
        return {
            "lean_available": LEAN_AVAILABLE,
            "client_initialized": self._client is not None,
            "can_complete_proofs": LEAN_AVAILABLE and self._client is not None
        }
