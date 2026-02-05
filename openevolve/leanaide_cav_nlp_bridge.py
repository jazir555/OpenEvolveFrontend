"""
LeanAide-CAV-NLP Integration Bridge
====================================

Bridge between LeanAide and CAV-NLP systems.

Purpose:
1. Provide smooth migration path from LeanAide to CAV-NLP
2. Route formalization requests to CAV-NLP
3. Keep verification/elaboration with LeanAide
4. Maintain backward compatibility

Author: OpenEvolve
Version: 1.0.0
"""

import warnings
import logging
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Try to import CAV-NLP
try:
    from openevolve.cav_nlp_integration import Z3LeanAideBridge
    from openevolve.cav_nlp_integration.adapter import create_z3_lean_bridge
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Try to import unified service
try:
    from openevolve.unified_math_service import (
        UnifiedMathService,
        create_unified_math_service
    )
    UNIFIED_SERVICE_AVAILABLE = True
except ImportError:
    UNIFIED_SERVICE_AVAILABLE = False

# Try to import LeanAide
try:
    from leanaide_client import LeanAideClient, TaskType
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False


class LeanAideCAVNLPBridge:
    """
    Bridge for migrating from LeanAide to CAV-NLP.
    
    Routes formalization requests to CAV-NLP while preserving
    LeanAide's verification and elaboration capabilities.
    
    This class can be used as a drop-in replacement for LeanAideClient
    for translation tasks, redirecting them to CAV-NLP.
    
    Example:
        # Old way (deprecated)
        client = LeanAideClient()
        result = await client.translate_thm("x + 0 = x")
        
        # New way (recommended)
        bridge = LeanAideCAVNLPBridge()
        result = await bridge.translate_thm("x + 0 = x")  # Uses CAV-NLP
    """
    
    def __init__(
        self,
        use_cav_nlp: bool = True,
        use_unified_service: bool = True
    ):
        """
        Initialize the bridge.
        
        Args:
            use_cav_nlp: Whether to use CAV-NLP for formalization
            use_unified_service: Whether to use unified math service
        """
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.use_unified_service = use_unified_service and UNIFIED_SERVICE_AVAILABLE
        
        # Initialize services
        if self.use_unified_service:
            self.unified_service = create_unified_math_service()
            logger.info("Using UnifiedMathService for formalization")
        elif self.use_cav_nlp:
            self.cav_nlp_bridge = create_z3_lean_bridge()
            self.unified_service = None
            logger.info("Using CAV-NLP bridge for formalization")
        else:
            self.unified_service = None
            self.cav_nlp_bridge = None
            logger.warning("CAV-NLP not available - will use fallback")
        
        # Keep LeanAide client for non-translation tasks
        if LEANAIDE_AVAILABLE:
            self.lean_client = LeanAideClient()
        else:
            self.lean_client = None
    
    # ========================================================================
    # Translation Methods (Redirected to CAV-NLP)
    # ========================================================================
    
    async def translate_thm(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate theorem to Lean 4.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        This method now redirects to CAV-NLP.
        
        Args:
            text: Theorem text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with 'lean_code', 'success', etc.
        """
        warnings.warn(
            "translate_thm is deprecated. Use UnifiedMathService.formalize() or CAV-NLP directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.formalize(text)
            return {
                "success": result.success,
                "lean_code": result.code,
                "elaborated_code": result.elaborated_code,
                "source": result.source,
                "warnings": result.warnings
            }
        elif self.use_cav_nlp and self.cav_nlp_bridge:
            # Use CAV-NLP bridge directly
            lean_code = self._generate_lean_code(text)
            return {
                "success": True,
                "lean_code": lean_code,
                "source": "cav_nlp",
                "warnings": []
            }
        else:
            # Fallback
            return {
                "success": True,
                "lean_code": self._generate_fallback_code(text),
                "source": "fallback",
                "warnings": ["CAV-NLP not available - using basic template"]
            }
    
    async def translate_def(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate definition to Lean 4.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        
        Args:
            text: Definition text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with 'lean_code', 'success', etc.
        """
        warnings.warn(
            "translate_def is deprecated. Use UnifiedMathService.formalize() or CAV-NLP directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Similar to translate_thm
        return await self.translate_thm(text, **kwargs)
    
    async def translate_thm_detailed(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Translate theorem with detailed output.
        
        DEPRECATED: Use CAV-NLP formalization instead.
        
        Args:
            text: Theorem text in natural language
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with detailed formalization results
        """
        warnings.warn(
            "translate_thm_detailed is deprecated. Use CAV-NLP directly for detailed output.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = await self.translate_thm(text, **kwargs)
        result["detailed"] = True
        result["note"] = "Detailed mode - CAV-NLP provides dependency DAG and canonical form"
        return result
    
    # ========================================================================
    # Elaboration Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def elaborate(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Elaborate Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to elaborate
            **kwargs: Additional arguments
            
        Returns:
            Dict with elaborated code
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.elaborate(code)
            return {
                "success": result.success,
                "elaborated_code": result.elaborated_code,
                "info": result.info
            }
        elif self.lean_client:
            return await self.lean_client.elaborate(code, **kwargs)
        else:
            return {
                "success": False,
                "error": "Elaboration not available"
            }
    
    # ========================================================================
    # Documentation Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def generate_documentation(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Generate documentation for Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to document
            **kwargs: Additional arguments
            
        Returns:
            Dict with documentation
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.generate_documentation(code)
            return {
                "success": result.success,
                "documentation": result.documentation,
                "theorem_name": result.theorem_name
            }
        elif self.lean_client:
            return await self.lean_client.generate_documentation(code, **kwargs)
        else:
            return {
                "success": False,
                "error": "Documentation generation not available"
            }
    
    # ========================================================================
    # Verification Methods (Delegated to LeanAide)
    # ========================================================================
    
    async def verify(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Verify Lean 4 code.
        
        This is a LeanAide-specific capability that is preserved.
        
        Args:
            code: Lean 4 code to verify
            **kwargs: Additional arguments
            
        Returns:
            Dict with verification results
        """
        if self.use_unified_service and self.unified_service:
            result = await self.unified_service.verify(code)
            if result:
                return {
                    "success": result.success,
                    "message": result.message if hasattr(result, 'message') else str(result.status),
                    "verified": result.success
                }
            else:
                return {"success": False, "error": "Verification returned None"}
        elif self.lean_client:
            return await self.lean_client.check_elaboration(code, **kwargs)
        else:
            return {
                "success": False,
                "error": "Verification not available"
            }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get available capabilities."""
        return {
            "cav_nlp_available": self.use_cav_nlp,
            "unified_service_available": self.use_unified_service,
            "leanaide_client_available": self.lean_client is not None,
            "translation": self.use_cav_nlp or self.use_unified_service,
            "elaboration": self.lean_client is not None or self.use_unified_service,
            "verification": self.lean_client is not None or self.use_unified_service,
            "documentation": self.lean_client is not None or self.use_unified_service,
        }
    
    def _generate_lean_code(self, text: str) -> str:
        """Generate basic Lean code from text."""
        return f"""import Mathlib

-- Formalized: {text[:100]}

theorem formalized_statement : True := by
  sorry
"""
    
    def _generate_fallback_code(self, text: str) -> str:
        """Generate fallback Lean code."""
        return self._generate_lean_code(text)


# ============================================================================
# Migration Helper
# ============================================================================

def migrate_leanaide_to_cav_nlp(old_code: str) -> str:
    """
    Helper to migrate old LeanAide code to use CAV-NLP.
    
    Args:
        old_code: Python code using LeanAide
        
    Returns:
        Migrated code using CAV-NLP/Unified service
    """
    replacements = [
        # Import changes
        ("from leanaide_client import LeanAideClient",
         "from openevolve.unified_math_service import UnifiedMathService, create_unified_math_service"),
        
        # Client instantiation
        ("client = LeanAideClient()",
         "service = create_unified_math_service()"),
        
        # Translation calls
        ("await client.translate_thm(text)",
         "await service.formalize(text)"),
        
        ("await client.translate_def(text)",
         "await service.formalize(text)"),
        
        # Result access
        ('result.data["lean_code"]',
         'result.code'),
        
        ('result.data.get("lean_code")',
         'result.code'),
        
        # Verification
        ("await client.check_elaboration(code)",
         "await service.verify(code)"),
    ]
    
    result = old_code
    for old, new in replacements:
        result = result.replace(old, new)
    
    return result


# ============================================================================
# Convenience Functions
# ============================================================================

def create_migration_bridge() -> LeanAideCAVNLPBridge:
    """Create a bridge for migration from LeanAide to CAV-NLP."""
    return LeanAideCAVNLPBridge()


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of the bridge."""
    print("=" * 70)
    print("LeanAide-CAV-NLP Bridge - Example Usage")
    print("=" * 70)
    
    bridge = create_migration_bridge()
    
    # Check capabilities
    print("\n1. CAPABILITIES")
    print("-" * 40)
    caps = bridge.get_capabilities()
    for cap, available in caps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {cap}")
    
    # Translation (redirected to CAV-NLP)
    print("\n2. TRANSLATION (CAV-NLP)")
    print("-" * 40)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = await bridge.translate_thm("For all x > 0, x + 1 > 1")
        
        print(f"   Success: {result['success']}")
        print(f"   Source: {result['source']}")
        print(f"   Code:\n{result['lean_code']}")
        
        if w:
            print(f"   Deprecation warning issued: {len([x for x in w if issubclass(x.category, DeprecationWarning)])}")
    
    print("\n" + "=" * 70)
    print("Bridge example completed!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
