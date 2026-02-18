"""
Guardrails Adapter - Integration with the Guardrails library.
"""

import logging
from typing import Dict, List, Any, Optional
from .config import ReliabilityConfig

logger = logging.getLogger(__name__)

class GuardrailsAdapter:
    """Bridges OpenEvolve to the Guardrails validation system."""

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()
        self._initialized = False
        self._initialize_guardrails()

    def _initialize_guardrails(self):
        try:
            # Try to import guardrails
            import guardrails as gd
            self._gd = gd
            self._initialized = True
            logger.info("Guardrails initialized successfully")
        except ImportError:
            logger.warning("Guardrails not installed. Using fallback validation.")
            self._initialized = False

    def validate(self, output: str, rail_spec: Optional[str] = None) -> Dict[str, Any]:
        """Validate output against rail specification."""
        if not self._initialized:
            return {"valid": True, "method": "fallback", "issues": []}
            
        try:
            # Create a Guard instance from the rail specification
            if rail_spec:
                guard = self._gd.Guard.from_rail_string(rail_spec)
            else:
                # Use a default rail spec if none provided
                default_rail = """
<rail version="0.1">
<output>
    <string name="validated_output" format="valid-json" on-fail-valid-json="reask" />
</output>
</rail>
"""
                guard = self._gd.Guard.from_rail_string(default_rail)
            
            # Perform validation
            validation_result = guard.parse(output)
            
            return {
                "valid": validation_result.validation_passed,
                "method": "guardrails",
                "validated_output": validation_result.validated_output,
                "issues": validation_result.error if not validation_result.validation_passed else []
            }
        except Exception as e:
            logger.error(f"Guardrails validation failed: {e}")
            return {"valid": False, "error": str(e)}
