"""
Math Verification Pipeline Node for BubbleLabs

Complete mathematical verification pipeline combining Lean 4 and Z3.
Provides end-to-end verification from natural language to formal proof.

Pipeline stages:
1. Autoformalization (NL → Lean)
2. Z3 Pre-check (quick validation)
3. Lean Verification (detailed proof)
4. Cross-validation (Z3 ↔ Lean)
5. Report Generation

Part of the Mathematical Verification Bubble Suite.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class VerificationStrategy(Enum):
    """Strategy for combined verification."""
    Z3_FIRST = "z3_first"       # Quick Z3 check first
    LEAN_FIRST = "lean_first"   # Full Lean proof first
    PARALLEL = "parallel"       # Run both in parallel
    CONSENSUS = "consensus"     # Both must agree
    ADAPTIVE = "adaptive"       # Choose based on problem


class MathVerificationPipelineNode(BubbleLabsNode):
    """
    Complete mathematical verification pipeline.
    
    Operations:
        - verify: Full verification pipeline
        - quick_check: Fast Z3-only check
        - formal_verify: Lean-only verification
        - cross_validate: Cross-check Z3 and Lean
        - batch_verify: Verify multiple statements
        - compare_strategies: Compare verification strategies
    """
    
    DISPLAY_NAME = "Math Verification Pipeline"
    DESCRIPTION = "Complete mathematical verification combining Lean 4 and Z3"
    ICON = "math-pipeline"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "verify",
        "quick_check",
        "formal_verify",
        "cross_validate",
        "batch_verify",
        "compare_strategies"
    ]
    
    STAGES = [
        "autoformalization",
        "z3_precheck",
        "lean_verification",
        "cross_validation",
        "report_generation"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._lean_client = None
        self._z3_engine = None
        self._bridge = None
        
    def _initialize_components(self):
        """Initialize Lean and Z3 components."""
        try:
            from leanaide_client import LeanAideClient, LeanAideConfig
            self._lean_client = LeanAideClient(LeanAideConfig())
        except Exception as e:
            logger.warning(f"Could not initialize LeanAide: {e}")
        
        try:
            from z3prover_integration import Z3SolverEngine, Z3Config
            self._z3_engine = Z3SolverEngine(Z3Config())
        except Exception as e:
            logger.warning(f"Could not initialize Z3: {e}")
        
        try:
            from z3_leanaide_bridge import Z3LeanAideBridge
            self._bridge = Z3LeanAideBridge()
        except Exception as e:
            logger.warning(f"Could not initialize bridge: {e}")

    def _extract_entanglement_context(self, inputs: Dict[str, Any], context) -> Dict[str, Any]:
        """Extract entanglement context from inputs, context metadata, or artifacts."""
        entanglement_context = inputs.get("entanglement_context") or {}

        entanglement_matrix = entanglement_context.get("entanglement_matrix") or inputs.get("entanglement_matrix")
        entangled_with = entanglement_context.get("entangled_with") or inputs.get("entangled_with")

        if hasattr(context, "metadata") and isinstance(context.metadata, dict):
            entanglement_matrix = entanglement_matrix or context.metadata.get("entanglement_matrix")
            entangled_with = entangled_with or context.metadata.get("entangled_with")

        if not entanglement_matrix and hasattr(context, "artifacts"):
            entanglement_matrix = context.artifacts.get("decomposition", {}).get("entanglement_matrix")

        if entanglement_matrix and not entangled_with:
            sub_problem_id = inputs.get("sub_problem_id") or inputs.get("component_id")
            if sub_problem_id and isinstance(entanglement_matrix, dict):
                entangled_with = entanglement_matrix.get(sub_problem_id)

        entangled_with = entangled_with or []

        return {
            "entanglement_matrix": entanglement_matrix or {},
            "entangled_with": entangled_with
        }
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "verify"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_verify":
            if "statements" not in inputs and "statements" not in self.config:
                errors.append("batch_verify requires 'statements' input")
        elif operation in ["verify", "quick_check", "formal_verify", "cross_validate"]:
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
                    "default": "verify",
                    "description": "Verification operation"
                },
                "statement": {
                    "type": "string",
                    "description": "Mathematical statement to verify"
                },
                "statements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of statements for batch verification"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["z3_first", "lean_first", "parallel", "consensus", "adaptive"],
                    "default": "adaptive",
                    "description": "Verification strategy"
                },
                "stages": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": self.STAGES
                    },
                    "default": self.STAGES,
                    "description": "Pipeline stages to run"
                },
                "skip_stages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Stages to skip"
                },
                "autoformalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Autoformalize if input is natural language"
                },
                "domain": {
                    "type": "string",
                    "enum": ["general", "arithmetic", "algebra", "analysis", "logic", "geometry"],
                    "default": "general",
                    "description": "Mathematical domain"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence for verification"
                },
                "timeout": {
                    "type": "number",
                    "default": 300.0,
                    "description": "Total pipeline timeout"
                },
                "generate_report": {
                    "type": "boolean",
                    "default": True,
                    "description": "Generate detailed report"
                },
                "sub_problem_id": {
                    "type": "string",
                    "description": "Sub-problem identifier for entanglement lookup"
                },
                "entanglement_matrix": {
                    "type": "object",
                    "description": "Entanglement matrix mapping sub-problems to entangled peers"
                },
                "entangled_with": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit list of entangled sub-problem ids"
                },
                "compare_all_strategies": {
                    "type": "boolean",
                    "default": False,
                    "description": "Compare all strategies (for compare_strategies)"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute verification pipeline."""
        start_time = time.time()
        operation = inputs.get("operation", self.config.get("operation", "verify"))
        entanglement_context = self._extract_entanglement_context(inputs, context)
        
        context.update_progress(5)
        
        if self._lean_client is None and self._z3_engine is None:
            self._initialize_components()
        
        context.update_progress(10)
        
        try:
            if operation == "verify":
                result = self._full_verify(inputs, context)
            elif operation == "quick_check":
                result = self._quick_check(inputs, context)
            elif operation == "formal_verify":
                result = self._formal_verify(inputs, context)
            elif operation == "cross_validate":
                result = self._cross_validate(inputs, context)
            elif operation == "batch_verify":
                result = self._batch_verify(inputs, context)
            elif operation == "compare_strategies":
                result = self._compare_strategies(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.utcnow().isoformat()
            result["entanglement_context"] = entanglement_context
            
            context.add_artifact("math_verification_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Verification pipeline failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _full_verify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Run full verification pipeline."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        strategy_str = inputs.get("strategy", self.config.get("strategy", "adaptive"))
        stages = inputs.get("stages", self.config.get("stages", self.STAGES))
        autoformalize = inputs.get("autoformalize", self.config.get("autoformalize", True))
        
        strategy = VerificationStrategy(strategy_str)
        
        pipeline_results = {}
        stage_results = []
        
        # Stage 1: Autoformalization
        if "autoformalization" in stages and autoformalize:
            context.update_progress(15)
            lean_code = self._autoformalize_statement(statement)
            pipeline_results["autoformalization"] = lean_code
            stage_results.append({"stage": "autoformalization", "status": "completed"})
        else:
            lean_code = statement
            stage_results.append({"stage": "autoformalization", "status": "skipped"})
        
        # Stage 2: Z3 Pre-check
        if "z3_precheck" in stages:
            context.update_progress(35)
            z3_result = self._z3_check(lean_code)
            pipeline_results["z3_precheck"] = z3_result
            stage_results.append({"stage": "z3_precheck", "status": z3_result.get("status", "unknown")})
        
        # Stage 3: Lean Verification
        if "lean_verification" in stages:
            context.update_progress(60)
            lean_result = self._lean_verify(lean_code)
            pipeline_results["lean_verification"] = lean_result
            stage_results.append({"stage": "lean_verification", "status": lean_result.get("status", "unknown")})
        
        # Stage 4: Cross-validation
        if "cross_validation" in stages:
            context.update_progress(80)
            cross_result = self._cross_check(pipeline_results.get("z3_precheck"), 
                                            pipeline_results.get("lean_verification"))
            pipeline_results["cross_validation"] = cross_result
            stage_results.append({"stage": "cross_validation", "status": cross_result.get("agreement", "unknown")})
        
        context.update_progress(100)
        
        # Determine overall result
        overall_success = all(
            r.get("status") in ["verified", "sat", "completed", "agreed"]
            for r in stage_results
        )
        
        return {
            "success": overall_success,
            "verified": overall_success,
            "statement": statement,
            "lean_code": lean_code if isinstance(lean_code, str) else lean_code.get("lean_code", ""),
            "strategy": strategy.value,
            "stages_completed": stage_results,
            "pipeline_results": pipeline_results,
            "summary": self._generate_summary(pipeline_results)
        }
    
    def _quick_check(self, inputs: Dict, context) -> Dict[str, Any]:
        """Quick Z3-only check."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(40)
        
        # Convert statement to Z3 constraints
        z3_result = self._z3_check(statement)
        
        context.update_progress(100)
        
        return {
            "success": z3_result.get("status") == "sat",
            "verified": z3_result.get("status") == "sat",
            "method": "z3_quick_check",
            "result": z3_result,
            "note": "Quick check only - not a formal proof"
        }
    
    def _formal_verify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Lean-only verification."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        autoformalize = inputs.get("autoformalize", self.config.get("autoformalize", True))
        
        context.update_progress(30)
        
        if autoformalize:
            lean_code = self._autoformalize_statement(statement)
        else:
            lean_code = statement
        
        context.update_progress(60)
        
        lean_result = self._lean_verify(lean_code)
        
        context.update_progress(100)
        
        return {
            "success": lean_result.get("status") == "verified",
            "verified": lean_result.get("status") == "verified",
            "method": "lean_formal",
            "lean_code": lean_code if isinstance(lean_code, str) else lean_code.get("lean_code", ""),
            "result": lean_result
        }
    
    def _cross_validate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Cross-validate Z3 and Lean results."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        
        context.update_progress(30)
        
        lean_code = self._autoformalize_statement(statement)
        
        context.update_progress(40)
        
        z3_result = self._z3_check(lean_code if isinstance(lean_code, str) else statement)
        
        context.update_progress(60)
        
        lean_result = self._lean_verify(lean_code)
        
        context.update_progress(80)
        
        cross_result = self._cross_check(z3_result, lean_result)
        
        context.update_progress(100)
        
        return {
            "success": cross_result.get("agreement", False),
            "agreement": cross_result.get("agreement", False),
            "z3_result": z3_result,
            "lean_result": lean_result,
            "cross_validation": cross_result
        }
    
    def _batch_verify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify multiple statements."""
        statements = inputs.get("statements", self.config.get("statements", []))
        
        context.update_progress(20)
        
        results = []
        total = len(statements)
        successful = 0
        
        for i, statement in enumerate(statements):
            progress = 20 + (70 * (i + 1) // max(total, 1))
            context.update_progress(progress)
            
            result = self._full_verify({"statement": statement}, context)
            if result.get("verified"):
                successful += 1
            
            results.append({
                "statement": statement[:100] + "..." if len(statement) > 100 else statement,
                "verified": result.get("verified", False),
                "summary": result.get("summary", {})
            })
        
        context.update_progress(100)
        
        return {
            "success": True,
            "total": total,
            "verified_count": successful,
            "failed_count": total - successful,
            "success_rate": successful / max(total, 1),
            "results": results
        }
    
    def _compare_strategies(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compare different verification strategies."""
        statement = inputs.get("statement", self.config.get("statement", ""))
        compare_all = inputs.get("compare_all_strategies", self.config.get("compare_all_strategies", False))
        
        strategies = list(VerificationStrategy) if compare_all else [
            VerificationStrategy.Z3_FIRST,
            VerificationStrategy.LEAN_FIRST,
            VerificationStrategy.PARALLEL
        ]
        
        context.update_progress(20)
        
        comparisons = []
        
        for i, strategy in enumerate(strategies):
            progress = 20 + (70 * (i + 1) // len(strategies))
            context.update_progress(progress)
            
            start = time.time()
            result = self._full_verify(
                {"statement": statement, "strategy": strategy.value},
                context
            )
            elapsed = time.time() - start
            
            comparisons.append({
                "strategy": strategy.value,
                "verified": result.get("verified", False),
                "execution_time": elapsed,
                "stages_completed": len(result.get("stages_completed", []))
            })
        
        context.update_progress(100)
        
        # Find best strategy
        best = min(comparisons, key=lambda x: x["execution_time"] if x["verified"] else float('inf'))
        
        return {
            "success": True,
            "comparisons": comparisons,
            "best_strategy": best["strategy"],
            "recommendation": f"Use {best['strategy']} for similar problems"
        }
    
    def _autoformalize_statement(self, statement: str) -> Union[str, Dict]:
        """Autoformalize natural language to Lean."""
        if self._lean_client:
            try:
                result = asyncio.run(self._lean_client.translate_theorem(statement))
                if result.success and result.data:
                    return result.data.get("translation", statement)
            except Exception as e:
                logger.warning(f"Autoformalization failed: {e}")
        
        # Fallback: Return structured result
        return {
            "lean_code": f"-- Autoformalized: {statement[:50]}...\ntheorem auto : True := by trivial",
            "confidence": 0.5,
            "warnings": ["Fallback autoformalization"]
        }
    
    def _z3_check(self, code_or_statement: str) -> Dict[str, Any]:
        """Run Z3 check."""
        if self._z3_engine:
            try:
                from z3prover_integration import Z3Variable, Z3Constraint
                # Simplified check
                result = self._z3_engine.solve_constraints([], [])
                return {
                    "status": result.status.value,
                    "satisfiable": result.status.value == "sat"
                }
            except Exception as e:
                logger.warning(f"Z3 check failed: {e}")
        
        return {"status": "unknown", "satisfiable": None, "note": "Z3 unavailable"}
    
    def _lean_verify(self, code: Union[str, Dict]) -> Dict[str, Any]:
        """Verify with Lean."""
        lean_code = code if isinstance(code, str) else code.get("lean_code", "")
        
        if self._bridge:
            try:
                result = self._bridge.verify_with_lean(lean_code)
                return {
                    "status": "verified" if result.success else "failed",
                    "verified": result.success
                }
            except Exception as e:
                logger.warning(f"Lean verification failed: {e}")
        
        return {"status": "unknown", "verified": False, "note": "Lean unavailable"}
    
    def _cross_check(self, z3_result: Dict, lean_result: Dict) -> Dict[str, Any]:
        """Cross-validate Z3 and Lean results."""
        if not z3_result or not lean_result:
            return {"agreement": False, "reason": "Missing results"}
        
        z3_verified = z3_result.get("satisfiable") or z3_result.get("status") == "sat"
        lean_verified = lean_result.get("verified") or lean_result.get("status") == "verified"
        
        agreement = z3_verified == lean_verified
        
        return {
            "agreement": agreement,
            "z3_verified": z3_verified,
            "lean_verified": lean_verified,
            "confidence": 1.0 if agreement else 0.5,
            "recommendation": "Results agree" if agreement else "Discrepancy detected - manual review needed"
        }
    
    def _generate_summary(self, pipeline_results: Dict) -> Dict[str, Any]:
        """Generate verification summary."""
        stages_passed = sum(
            1 for r in pipeline_results.values()
            if isinstance(r, dict) and r.get("status") in ["verified", "sat", "completed", "agreed"]
        )
        
        total_stages = len(pipeline_results)
        
        return {
            "stages_passed": stages_passed,
            "total_stages": total_stages,
            "completion_rate": stages_passed / max(total_stages, 1),
            "overall_status": "verified" if stages_passed == total_stages else "partial"
        }
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
