"""
Auto-Extraction Hook for Z3 Solver Operations

Automatically extracts knowledge from Z3 solver results and integrates
with the OpenEvolve knowledge engine.

Usage:
    from knowledge_engine.integrations.z3_auto_extraction import enable_auto_extraction
    enable_auto_extraction()  # Enables for all Z3 operations

Author: OpenEvolve
Created: 2026-01-31
"""

import logging
import asyncio
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime, timezone

# Configure logging
logger = logging.getLogger(__name__)

# Integration imports
try:
    from knowledge_engine.integrations.z3_knowledge_integration import (
        Z3KnowledgeIntegration,
        get_z3_knowledge_integration,
        Z3KnowledgeExtractionHook
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("Z3 knowledge integration not available")

# Z3 imports
try:
    from z3prover_integration import Z3SolverResult, Z3TheoremResult
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None


class Z3AutoExtractionManager:
    """
    Manages automatic knowledge extraction from Z3 operations.
    
    Can be enabled/disabled globally and provides hooks for
    various Z3 solver operations.
    """
    
    def __init__(self):
        self.enabled = False
        self.extraction_hook: Optional[Z3KnowledgeExtractionHook] = None
        self._integration: Optional[Z3KnowledgeIntegration] = None
        self._wrapped_functions: Dict[str, Callable] = {}
        self.stats = {
            "extractions_triggered": 0,
            "extractions_successful": 0,
            "extractions_failed": 0
        }
    
    async def initialize(self):
        """Initialize the auto-extraction system."""
        if not INTEGRATION_AVAILABLE:
            logger.error("Cannot initialize: Z3 knowledge integration not available")
            return False
        
        try:
            self._integration = await get_z3_knowledge_integration()
            self.extraction_hook = Z3KnowledgeExtractionHook(self._integration)
            logger.info("Z3 auto-extraction initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize auto-extraction: {e}")
            return False
    
    def enable(self):
        """Enable auto-extraction."""
        if not self._integration:
            logger.warning("Auto-extraction not initialized, call initialize() first")
            return False
        
        self.enabled = True
        self.extraction_hook.enabled = True
        logger.info("Z3 auto-extraction enabled")
        return True
    
    def disable(self):
        """Disable auto-extraction."""
        self.enabled = False
        if self.extraction_hook:
            self.extraction_hook.enabled = False
        logger.info("Z3 auto-extraction disabled")
    
    async def on_solver_result(
        self,
        result: Any,
        problem: str,
        problem_type: str = "general",
        problem_id: Optional[str] = None
    ):
        """Called when a solver result is available."""
        if not self.enabled or not self.extraction_hook:
            return
        
        self.stats["extractions_triggered"] += 1
        
        try:
            await self.extraction_hook.on_solver_result(
                result, problem, problem_type
            )
            self.stats["extractions_successful"] += 1
            
        except Exception as e:
            self.stats["extractions_failed"] += 1
            logger.error(f"Auto-extraction failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            **self.stats,
            "enabled": self.enabled,
            "initialized": self._integration is not None
        }


# Global manager instance
_auto_extraction_manager: Optional[Z3AutoExtractionManager] = None


async def get_auto_extraction_manager() -> Z3AutoExtractionManager:
    """Get or initialize the auto-extraction manager."""
    global _auto_extraction_manager
    if _auto_extraction_manager is None:
        _auto_extraction_manager = Z3AutoExtractionManager()
        await _auto_extraction_manager.initialize()
    return _auto_extraction_manager


def enable_auto_extraction():
    """
    Enable automatic knowledge extraction for Z3 operations.
    
    This should be called after importing Z3 integration modules.
    """
    async def _enable():
        manager = await get_auto_extraction_manager()
        return manager.enable()
    
    try:
        # Try to get running event loop
        loop = asyncio.get_running_loop()
        # If we're in an async context, return the coroutine
        return _enable()
    except RuntimeError:
        # No event loop running, create one
        return asyncio.run(_enable())


def disable_auto_extraction():
    """Disable automatic knowledge extraction."""
    global _auto_extraction_manager
    if _auto_extraction_manager:
        _auto_extraction_manager.disable()


def auto_extract_knowledge(wrapped_func: Callable = None, problem_type: str = "general"):
    """
    Decorator to automatically extract knowledge from Z3 solver functions.
    
    Usage:
        @auto_extract_knowledge(problem_type="linear")
        async def solve_linear_problem(constraints):
            # ... solving logic ...
            return result
    
    Args:
        wrapped_func: Function to wrap
        problem_type: Type of problem for classification
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Extract knowledge if enabled
            manager = await get_auto_extraction_manager()
            if manager.enabled:
                # Get problem statement from args or kwargs
                problem = kwargs.get('problem', str(args[0]) if args else "")
                
                await manager.on_solver_result(
                    result=result,
                    problem=problem,
                    problem_type=problem_type
                )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            
            # Extract knowledge if enabled (fire and forget)
            async def _extract():
                manager = await get_auto_extraction_manager()
                if manager.enabled:
                    problem = kwargs.get('problem', str(args[0]) if args else "")
                    await manager.on_solver_result(
                        result=result,
                        problem=problem,
                        problem_type=problem_type
                    )
            
            try:
                asyncio.create_task(_extract())
            except RuntimeError:
                # No event loop, skip extraction
                pass
            
            return result
        
        # Return appropriate wrapper based on whether func is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    if wrapped_func is None:
        return decorator
    else:
        return decorator(wrapped_func)


class Z3KnowledgeExtractorMixin:
    """
    Mixin class for Z3 solver classes to enable knowledge extraction.
    
    Usage:
        class MySolver(Z3KnowledgeExtractorMixin):
            async def solve(self, problem):
                result = await self._do_solve(problem)
                await self.extract_knowledge(result, problem)
                return result
    """
    
    def __init__(self):
        self._extraction_enabled = True
        self._auto_manager: Optional[Z3AutoExtractionManager] = None
    
    async def _get_manager(self) -> Optional[Z3AutoExtractionManager]:
        """Get the auto-extraction manager."""
        if self._auto_manager is None:
            self._auto_manager = await get_auto_extraction_manager()
        return self._auto_manager
    
    async def extract_knowledge(
        self,
        result: Any,
        problem: str,
        problem_type: str = "general",
        problem_id: Optional[str] = None
    ):
        """
        Extract knowledge from a solver result.
        
        This method can be called explicitly after solving.
        """
        if not self._extraction_enabled:
            return
        
        manager = await self._get_manager()
        if manager and manager.enabled:
            await manager.on_solver_result(
                result=result,
                problem=problem,
                problem_type=problem_type,
                problem_id=problem_id
            )
    
    def enable_extraction(self):
        """Enable knowledge extraction for this solver."""
        self._extraction_enabled = True
    
    def disable_extraction(self):
        """Disable knowledge extraction for this solver."""
        self._extraction_enabled = False


# =============================================================================
# Integration with z3prover_integration.py
# =============================================================================

def patch_z3_integration():
    """
    Patch Z3 integration module to auto-extract knowledge.
    
    This modifies the Z3 solver functions to automatically extract
    knowledge after solving.
    """
    if not Z3_AVAILABLE:
        logger.warning("Z3 integration not available for patching")
        return False
    
    try:
        import z3prover_integration as z3i
        
        # Store original solve methods
        _original_solve = z3i.Z3Integration.solve
        _original_solve_theorem = z3i.Z3Integration.solve_theorem
        
        async def _patched_solve(self, smt_problem: str, **kwargs):
            """Patched solve method with knowledge extraction."""
            result = await _original_solve(self, smt_problem, **kwargs)
            
            # Extract knowledge
            manager = await get_auto_extraction_manager()
            if manager.enabled:
                await manager.on_solver_result(
                    result=result,
                    problem=smt_problem,
                    problem_type="constraint_solving"
                )
            
            return result
        
        async def _patched_solve_theorem(self, theorem: str, **kwargs):
            """Patched solve_theorem method with knowledge extraction."""
            result = await _original_solve_theorem(self, theorem, **kwargs)
            
            # Extract knowledge
            manager = await get_auto_extraction_manager()
            if manager.enabled:
                await manager.on_solver_result(
                    result=result,
                    problem=theorem,
                    problem_type="theorem_proving"
                )
            
            return result
        
        # Apply patches
        z3i.Z3Integration.solve = _patched_solve
        z3i.Z3Integration.solve_theorem = _patched_solve_theorem
        
        logger.info("Z3 integration patched for auto-extraction")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch Z3 integration: {e}")
        return False


# =============================================================================
# Example Usage
# =============================================================================

async def example_auto_extraction():
    """Example: Using auto-extraction."""
    print("Z3 Auto-Extraction Example")
    print("=" * 50)
    
    # Initialize and enable
    await get_auto_extraction_manager()
    enable_auto_extraction()
    
    # Example with decorator
    @auto_extract_knowledge(problem_type="linear")
    async def solve_example():
        # Mock result
        class MockResult:
            success = True
            model = type('Model', (), {'assignments': {'x': 5}})()
            constraints = ["(> x 0)"]
            solving_time = 0.5
        
        return MockResult()
    
    result = await solve_example()
    print(f"Solved with result: {result.success}")
    
    # Get stats
    manager = await get_auto_extraction_manager()
    stats = manager.get_stats()
    print(f"\nExtraction stats: {stats}")
    
    # Disable
    disable_auto_extraction()


if __name__ == "__main__":
    asyncio.run(example_auto_extraction())
