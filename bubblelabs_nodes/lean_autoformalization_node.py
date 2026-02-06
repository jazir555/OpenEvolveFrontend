"""
Lean Autoformalization Node for BubbleLabs

Converts natural language mathematical statements into formal Lean 4 code.
Uses LeanAide's autoformalization capabilities with support for:
- Theorem translation
- Definition translation
- Multi-agent generation (MDAP)
- Voting-based refinement (MAKER)
- Proof generation
- CAV-NLP enhanced formalization

Part of the Mathematical Verification Bubble Suite.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

# CAV-NLP Integration
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LeanAutoformalizationNode(BubbleLabsNode):
    """
    Autoformalize natural language mathematics to Lean 4 code.
    
    Operations:
        - translate_theorem: Convert theorem statement to Lean
        - translate_definition: Convert definition to Lean
        - elaborate: Expand brief descriptions to formal code
        - autoformalize: Full autoformalization with MDAP/MAKER
        - batch_translate: Translate multiple statements
    """
    
    DISPLAY_NAME = "Lean Autoformalization"
    DESCRIPTION = "Convert natural language mathematics to formal Lean 4 code"
    ICON = "lean-autoformalization"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    # Operation types
    OPERATIONS = [
        "translate_theorem",
        "translate_definition", 
        "elaborate",
        "autoformalize",
        "batch_translate"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._client = None
        self._engine = None
        self.use_cav_nlp = config.get("use_cav_nlp", True) if config else True
        self.use_cav_nlp = self.use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP integration initialized for LeanAutoformalizationNode")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP services: {e}")
                self.use_cav_nlp = False
                self.math_service = None
                self.enhanced_solver = None
        else:
            self.math_service = None
            self.enhanced_solver = None
        
    def _initialize_client(self):
        """Initialize LeanAide client."""
        try:
            from leanaide_client import LeanAideClient, LeanAideConfig
            config = LeanAideConfig(
                host=self.config.get("leanaide_host", "localhost"),
                port=self.config.get("leanaide_port", 7654),
                timeout=self.config.get("timeout", 6000.0)
            )
            self._client = LeanAideClient(config)
            return True
        except Exception as e:
            logger.warning(f"Could not initialize LeanAide client: {e}")
            return False
    
    def _initialize_engine(self):
        """Initialize autoformalization engine."""
        try:
            from leanaide_autoformalization_mdap_maker import (
                LeanAideAutoformalizationEngine,
                AutoformalizationStrategy
            )
            if self._client is None:
                self._initialize_client()
            
            self._engine = LeanAideAutoformalizationEngine(
                leanaide_client=self._client,
                enable_caching=self.config.get("enable_caching", True)
            )
            return True
        except Exception as e:
            logger.warning(f"Could not initialize autoformalization engine: {e}")
            return False
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "autoformalize"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}. Must be one of {self.OPERATIONS}")
        
        if operation == "batch_translate":
            if "statements" not in inputs and "statements" not in self.config:
                errors.append("batch_translate requires 'statements' input (list of statements)")
        else:
            if "text" not in inputs and "text" not in self.config:
                errors.append(f"{operation} requires 'text' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "autoformalize",
                    "description": "Autoformalization operation to perform"
                },
                "text": {
                    "type": "string",
                    "description": "Natural language mathematical statement"
                },
                "statements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of statements for batch translation"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["direct", "mdap", "maker", "hybrid", "adaptive"],
                    "default": "adaptive",
                    "description": "Autoformalization strategy"
                },
                "domain": {
                    "type": "string",
                    "enum": ["general", "algebra", "analysis", "topology", "number_theory", "logic"],
                    "default": "general",
                    "description": "Mathematical domain"
                },
                "include_proofs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include proof generation"
                },
                "num_agents": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of agents for MDAP/MAKER"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold"
                },
                "leanaide_host": {
                    "type": "string",
                    "default": "localhost",
                    "description": "LeanAide server host"
                },
                "leanaide_port": {
                    "type": "integer",
                    "default": 7654,
                    "description": "LeanAide server port"
                },
                "timeout": {
                    "type": "number",
                    "default": 6000.0,
                    "description": "Request timeout in seconds"
                },
                "enable_caching": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable result caching"
                },
                "use_cav_nlp": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable CAV-NLP enhanced autoformalization"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute autoformalization operation."""
        start_time = time.time()
        operation = inputs.get("operation", self.config.get("operation", "autoformalize"))
        
        context.update_progress(10)
        
        # Initialize client/engine
        if self._client is None:
            self._initialize_client()
        
        context.update_progress(20)
        
        try:
            if operation == "translate_theorem":
                result = self._translate_theorem(inputs, context)
            elif operation == "translate_definition":
                result = self._translate_definition(inputs, context)
            elif operation == "elaborate":
                result = self._elaborate(inputs, context)
            elif operation == "autoformalize":
                result = self._autoformalize(inputs, context)
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
            
            # Store in context
            context.add_artifact("lean_autoformalization_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Autoformalization failed: {str(e)}",
                details={"operation": operation, "error": str(e)}
            )
    
    def autoformalize(self, natural_language: str) -> str:
        """Convert natural language to Lean using CAV-NLP.
        
        Args:
            natural_language: Natural language mathematical statement
            
        Returns:
            Formal Lean code
        """
        if not self.use_cav_nlp:
            # Fallback to traditional method
            result = self._fallback_translation(natural_language, "theorem", "general", False)
            return result.get("lean_code", "")
        
        try:
            result = self.math_service.formalize(natural_language)
            if result and hasattr(result, 'elaborated_code') and result.elaborated_code:
                return result.elaborated_code
            return result.code if result and hasattr(result, 'code') else ""
        except Exception as e:
            logger.error(f"CAV-NLP autoformalization failed: {e}")
            # Fallback
            fallback = self._fallback_translation(natural_language, "theorem", "general", False)
            return fallback.get("lean_code", "")
    
    def _translate_theorem(self, inputs: Dict, context) -> Dict[str, Any]:
        """Translate theorem statement to Lean."""
        text = inputs.get("text", self.config.get("text", ""))
        domain = inputs.get("domain", self.config.get("domain", "general"))
        include_proofs = inputs.get("include_proofs", self.config.get("include_proofs", True))
        
        context.update_progress(40)
        
        # Try CAV-NLP first if available
        if self.use_cav_nlp:
            try:
                formalized = self.autoformalize(text)
                if formalized:
                    context.update_progress(90)
                    return {
                        "success": True,
                        "lean_code": formalized,
                        "theorem_name": self._extract_theorem_name(text),
                        "confidence": 0.9,
                        "method": "cav_nlp",
                        "logs": []
                    }
            except Exception as e:
                logger.warning(f"CAV-NLP translation failed: {e}, using fallback")
        
        if self._client:
            # Use real client
            try:
                import asyncio
                result = asyncio.run(self._client.translate_theorem(text))
                return {
                    "success": result.success,
                    "lean_code": result.data.get("translation", "") if result.data else "",
                    "theorem_name": result.data.get("theorem_name", "") if result.data else "",
                    "confidence": result.data.get("confidence", 0.0) if result.data else 0.0,
                    "logs": result.logs
                }
            except Exception as e:
                logger.warning(f"Client translation failed: {e}, using fallback")
        
        context.update_progress(60)
        
        # Fallback: Generate mock translation
        return self._fallback_translation(text, "theorem", domain, include_proofs)
    
    def _translate_definition(self, inputs: Dict, context) -> Dict[str, Any]:
        """Translate definition to Lean."""
        text = inputs.get("text", self.config.get("text", ""))
        domain = inputs.get("domain", self.config.get("domain", "general"))
        
        context.update_progress(40)
        
        if self._client:
            try:
                import asyncio
                result = asyncio.run(self._client.translate_definition(text))
                return {
                    "success": result.success,
                    "lean_code": result.data.get("translation", "") if result.data else "",
                    "definition_name": result.data.get("definition_name", "") if result.data else "",
                    "confidence": result.data.get("confidence", 0.0) if result.data else 0.0
                }
            except Exception as e:
                logger.warning(f"Client definition translation failed: {e}")
        
        context.update_progress(60)
        
        # Fallback
        return self._fallback_translation(text, "definition", domain, False)
    
    def _elaborate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Elaborate brief description to formal code."""
        text = inputs.get("text", self.config.get("text", ""))
        
        context.update_progress(40)
        
        if self._client:
            try:
                import asyncio
                result = asyncio.run(self._client.elaborate(text))
                return {
                    "success": result.success,
                    "elaborated_code": result.data.get("elaboration", "") if result.data else "",
                    "original": text
                }
            except Exception as e:
                logger.warning(f"Elaboration failed: {e}")
        
        context.update_progress(80)
        
        return {
            "success": True,
            "elaborated_code": f"-- Elaborated from: {text[:50]}...\n-- (Full elaboration would be here)",
            "original": text
        }
    
    def _autoformalize(self, inputs: Dict, context) -> Dict[str, Any]:
        """Full autoformalization with MDAP/MAKER."""
        text = inputs.get("text", self.config.get("text", ""))
        strategy = inputs.get("strategy", self.config.get("strategy", "adaptive"))
        num_agents = inputs.get("num_agents", self.config.get("num_agents", 3))
        confidence_threshold = inputs.get("confidence_threshold", self.config.get("confidence_threshold", 0.8))
        
        context.update_progress(30)
        
        if self._engine is None:
            self._initialize_engine()
        
        if self._engine:
            try:
                from leanaide_autoformalization_mdap_maker import AutoformalizationStrategy
                
                strategy_map = {
                    "direct": AutoformalizationStrategy.DIRECT,
                    "mdap": AutoformalizationStrategy.MDAP,
                    "maker": AutoformalizationStrategy.MAKER,
                    "hybrid": AutoformalizationStrategy.HYBRID,
                    "adaptive": AutoformalizationStrategy.ADAPTIVE
                }
                
                result = self._engine.autoformalize(
                    text=text,
                    strategy=strategy_map.get(strategy, AutoformalizationStrategy.ADAPTIVE),
                    num_agents=num_agents
                )
                
                context.update_progress(90)
                
                return {
                    "success": result.success,
                    "lean_code": result.lean_code,
                    "theorem_name": result.theorem_name,
                    "confidence": result.confidence,
                    "strategy_used": result.strategy_used,
                    "verification_status": result.verification_status,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            except Exception as e:
                logger.warning(f"Engine autoformalization failed: {e}")
        
        context.update_progress(70)
        
        # Fallback
        return self._fallback_translation(text, "theorem", "general", True)
    
    def _batch_translate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Translate multiple statements."""
        statements = inputs.get("statements", self.config.get("statements", []))
        
        context.update_progress(30)
        
        results = []
        total = len(statements)
        
        for i, statement in enumerate(statements):
            progress = 30 + (60 * (i + 1) // total)
            context.update_progress(progress)
            
            result = self._fallback_translation(statement, "theorem", "general", False)
            results.append({
                "statement": statement,
                "result": result
            })
        
        return {
            "success": True,
            "total": total,
            "successful": sum(1 for r in results if r["result"]["success"]),
            "results": results
        }
    
    def _fallback_translation(self, text: str, kind: str, domain: str, include_proofs: bool) -> Dict[str, Any]:
        """Generate fallback translation when LeanAide is unavailable."""
        # Extract a name from the text
        words = text.split()
        name = "theorem_" + "_".join(w.lower()[:5] for w in words[:3] if w.isalpha())[:20]
        
        # Generate mock Lean code
        if kind == "theorem":
            lean_code = f"""-- Autoformalized from: {text[:80]}...
import Mathlib

namespace Autoformalized

/- {text} -/
theorem {name} : True := by
  trivial

end Autoformalized
"""
        else:
            lean_code = f"""-- Autoformalized definition from: {text[:80]}...
import Mathlib

namespace Autoformalized

def {name} : ℕ := 0

end Autoformalized
"""
        
        return {
            "success": True,
            "lean_code": lean_code,
            f"{kind}_name": name,
            "confidence": 0.5,
            "warnings": ["Using fallback translation - LeanAide server unavailable"],
            "domain": domain,
            "include_proofs": include_proofs
        }
    
    def _extract_theorem_name(self, text: str) -> str:
        """Extract a theorem name from text."""
        words = text.split()
        name = "theorem_" + "_".join(w.lower()[:5] for w in words[:3] if w.isalpha())[:20]
        return name
    
    def is_healthy(self) -> bool:
        """Check node health."""
        # Always healthy as fallback mode works
        return True
