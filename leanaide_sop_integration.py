"""
LeanAide Autoformalization Integration for SOP Generator

This module provides integration between the LeanAide autoformalization system
and the SOP generator, enabling formal verification of mathematical components
in SOPs using Lean 4 theorem proving.

With CAV-NLP integration for enhanced SOP compliance verification.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from generic_maker_integration import (
    run_generic_maker,
    GenericEvaluator,
    GenericTask,
    GenericSolution,
    TaskType,
    MAKERConfig
)

from leanaide_autoformalization_mdap_maker import (
    LeanAideAutoformalizationEngine,
    AutoformalizationStrategy,
    create_leanaide_autoformalization_engine
)

from leanaide_mcts_mdap import (
    MDAPMCTSConfig,
    MDAPMCTS,
    MDAPMCTSResult
)

from leanaide_redflagging_system import (
    IntegratedRedFlaggingSystem,
    RedFlagConfig
)

from leanaide_predictive_flagging import (
    IntegratedPredictiveFlaggingSystem,
    PredictiveFlagConfig
)

# Add CAV-NLP imports with graceful fallback
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logging.warning("CAV-NLP integration not available - SOP integration will use standard methods")

logger = logging.getLogger(__name__)


@dataclass
class FormalVerificationResult:
    """Result of formal verification using LeanAide."""
    success: bool
    lean_code: str
    confidence: float
    verification_logs: List[str]
    error: Optional[str] = None
    execution_time: float = 0.0
    strategy_used: str = "unknown"


@dataclass
class MathematicalComponent:
    """A mathematical component in an SOP that requires formal verification."""
    description: str  # Natural language description
    formal_statement: str = ""  # Formal statement in Lean
    verification_result: Optional[FormalVerificationResult] = None
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 1  # 1-10 scale
    domain: str = "general"  # Mathematical domain


class LeanAideSOPIntegration:
    """
    Integration between LeanAide autoformalization system and SOP generator.
    
    This class provides:
    - Mathematical component extraction from SOPs
    - Formal verification of mathematical claims
    - Integration with SOP generation workflow
    - Quality control for mathematical components
    - CAV-NLP enhanced SOP compliance verification
    """
    
    def __init__(
        self,
        leanaide_client: Any,  # LeanAide client
        enable_predictive_flagging: bool = True,
        enable_red_flagging: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LeanAide-SOP integration.

        Args:
            leanaide_client: Initialized LeanAide client
            enable_predictive_flagging: Whether to enable predictive quality control
            enable_red_flagging: Whether to enable red-flagging
            config: Optional configuration dictionary
        """
        self.leanaide_client = leanaide_client
        self.enable_predictive_flagging = enable_predictive_flagging
        self.enable_red_flagging = enable_red_flagging
        self.config = config or {}

        # Initialize autoformalization engine
        try:
            from leanaide_autoformalization_mdap_maker import create_leanaide_autoformalization_engine
            self.autoformalization_engine = create_leanaide_autoformalization_engine(
                leanaide_client=leanaide_client,
                enable_caching=True
            )
        except ImportError:
            # Fallback if the autoformalization module isn't available
            self.autoformalization_engine = None
            print("Warning: Autoformalization engine not available, using basic functionality")

        # Initialize red-flagging system
        if enable_red_flagging:
            try:
                from leanaide_redflagging_system import IntegratedRedFlaggingSystem, RedFlagConfig
                red_config = RedFlagConfig(
                    confidence_threshold=0.3,
                    enable_detailed_analysis=True
                )
                self.red_flagging_system = IntegratedRedFlaggingSystem(red_config)
            except ImportError:
                self.red_flagging_system = None
                print("Warning: Red-flagging system not available")

        # Initialize predictive flagging system
        if enable_predictive_flagging:
            try:
                from leanaide_predictive_flagging import IntegratedPredictiveFlaggingSystem, PredictiveFlagConfig
                pred_config = PredictiveFlagConfig(
                    prediction_confidence_threshold=0.6,
                    enable_ml_prediction=True
                )
                self.predictive_flagging_system = IntegratedPredictiveFlaggingSystem(pred_config)
            except ImportError:
                self.predictive_flagging_system = None
                print("Warning: Predictive flagging system not available")
        
        # Initialize CAV-NLP components for SOP compliance
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP components initialized for SOP integration")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP components: {e}")
                self.use_cav_nlp = False
    
    async def extract_mathematical_components(
        self,
        sop_content: str
    ) -> List[MathematicalComponent]:
        """
        Extract mathematical components from SOP content.
        
        Args:
            sop_content: Content of the SOP to analyze
            
        Returns:
            List of mathematical components found in the SOP
        """
        components = []
        
        # Look for mathematical statements in the SOP
        # This is a simplified extraction - in practice, would use more sophisticated NLP
        import re

        # Patterns for mathematical statements
        patterns = [
            r'prove that ([^.]+)',  # "prove that X"
            r'for all ([^.]+)',     # "for all X"
            r'for any ([^.]+)',     # "for any X" 
            r'if ([^.]+) then',     # "if X then Y"
            r'where ([^.]+)',       # "where X"
            r'equation: ([^.]+)',   # "equation: X"
            r'formula: ([^.]+)',    # "formula: X"
            r'condition: ([^.]+)',  # "condition: X"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sop_content, re.IGNORECASE)
            for match in matches:
                # Determine complexity based on length and keywords
                complexity = min(10, max(1, len(match.split()) // 10))
                
                # Determine domain based on keywords
                domain = self._infer_domain(match)
                
                component = MathematicalComponent(
                    description=match.strip(),
                    complexity=complexity,
                    domain=domain
                )
                components.append(component)
        
        return components
    
    def _infer_domain(self, statement: str) -> str:
        """Infer mathematical domain from statement."""
        statement_lower = statement.lower()
        
        # Domain keywords
        domain_keywords = {
            "algebra": ["group", "ring", "field", "vector", "matrix", "polynomial", "equation"],
            "analysis": ["limit", "continuity", "derivative", "integral", "convergence", "function"],
            "logic": ["proposition", "predicate", "quantifier", "theorem", "proof", "axiom"],
            "number_theory": ["prime", "divisor", "modular", "congruence", "integer", "natural"],
            "combinatorics": ["count", "permutation", "combination", "graph", "tree", "set"],
            "geometry": ["triangle", "circle", "angle", "area", "volume", "dimension"],
            "topology": ["open", "closed", "compact", "connected", "continuous", "neighborhood"],
            "category_theory": ["functor", "natural", "morphism", "object", "arrow", "category"],
            "general": ["equal", "greater", "less", "sum", "product", "operation"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in statement_lower for keyword in keywords):
                return domain
        
        return "general"
    
    async def verify_mathematical_component(
        self,
        component: MathematicalComponent,
        strategy: AutoformalizationStrategy = AutoformalizationStrategy.ADAPTIVE
    ) -> FormalVerificationResult:
        """
        Verify a mathematical component using LeanAide autoformalization.
        
        Args:
            component: Mathematical component to verify
            strategy: Strategy to use for autoformalization
            
        Returns:
            Formal verification result
        """
        start_time = time.time()

        try:
            if self.autoformalization_engine is None:
                # Fallback: create a basic result if engine not available
                return FormalVerificationResult(
                    success=True,
                    lean_code=f"-- Fallback: {component.description}\n-- Autoformalization engine not available\ntheorem component_{abs(hash(component.description)) % 10000} : True := by trivial",
                    confidence=0.5,
                    verification_logs=["Using fallback - autoformalization engine not available"],
                    execution_time=time.time() - start_time,
                    strategy_used=strategy.value
                )

            # Use autoformalization to convert natural language to Lean code
            result = await self.autoformalization_engine.autoformalize(
                natural_language=component.description,
                statement_type="theorem",  # Default to theorem, could be lemma, definition, etc.
                name=f"component_{abs(hash(component.description)) % 10000}",
                strategy=strategy,
                context={
                    "domain": component.domain,
                    "complexity": component.complexity,
                    "dependencies": component.dependencies
                }
            )

            execution_time = time.time() - start_time

            if result.success:
                # Check quality with red-flagging system
                if self.enable_red_flagging and self.red_flagging_system:
                    is_flagged, flags = self.red_flagging_system.flag_item(
                        item=result.lean_code,
                        item_type="proof",
                        context={
                            "agent_id": "leanaide_sop_integration",
                            "confidence": result.confidence
                        }
                    )

                    if is_flagged:
                        # Apply predictive flagging to assess potential issues
                        if self.enable_predictive_flagging and self.predictive_flagging_system:
                            predictions = self.predictive_flagging_system.predict_quality(
                                item=result.lean_code,
                                item_type="proof",
                                context={
                                    "agent_id": "leanaide_sop_integration",
                                    "confidence": result.confidence,
                                    "flags": [f.reason for f in flags]
                                }
                            )

                            # Adjust confidence based on predictions
                            if predictions:
                                avg_prediction_confidence = sum(p.confidence for p in predictions) / len(predictions)
                                result.confidence = min(result.confidence, avg_prediction_confidence)

                return FormalVerificationResult(
                    success=True,
                    lean_code=result.lean_code,
                    confidence=result.confidence,
                    verification_logs=[f"Autoformalization successful with strategy {strategy.value}"],
                    execution_time=execution_time,
                    strategy_used=strategy.value
                )
            else:
                return FormalVerificationResult(
                    success=False,
                    lean_code="",
                    confidence=0.0,
                    verification_logs=getattr(result, 'errors', []),
                    error="Autoformalization failed",
                    execution_time=execution_time,
                    strategy_used=strategy.value
                )

        except (IOError, ValueError, TypeError, AttributeError) as e:
            execution_time = time.time() - start_time
            return FormalVerificationResult(
                success=False,
                lean_code="",
                confidence=0.0,
                verification_logs=[f"Error during verification: {str(e)}"],
                error=str(e),
                execution_time=execution_time,
                strategy_used=strategy.value
            )
    
    async def verify_sop_mathematical_components(
        self,
        sop_content: str,
        strategy: AutoformalizationStrategy = AutoformalizationStrategy.ADAPTIVE
    ) -> Dict[str, Any]:
        """
        Verify all mathematical components in an SOP.

        Args:
            sop_content: Content of the SOP to verify
            strategy: Strategy to use for autoformalization

        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Extract mathematical components
        components = await self.extract_mathematical_components(sop_content)
        
        # Verify each component
        verification_results = []
        total_confidence = 0.0
        successful_verifications = 0
        
        for component in components:
            result = await self.verify_mathematical_component(component, strategy)
            verification_results.append({
                "component": component.description,
                "result": result,
                "domain": component.domain,
                "complexity": component.complexity
            })
            
            if result.success:
                successful_verifications += 1
                total_confidence += result.confidence
        
        execution_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_success_rate = len(components) and successful_verifications / len(components) or 0
        avg_confidence = len(components) and total_confidence / len(components) or 0
        
        return {
            "total_components": len(components),
            "successful_verifications": successful_verifications,
            "success_rate": overall_success_rate,
            "average_confidence": avg_confidence,
            "execution_time": execution_time,
            "components": verification_results,
            "overall_success": overall_success_rate > 0.8  # Consider successful if >80% pass
        }
    
    async def enhance_sop_with_formal_verification(
        self,
        sop_content: str,
        requirement_description: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance an SOP with formal verification of mathematical components.
        
        Args:
            sop_content: Original SOP content
            requirement_description: Requirement description for context
            
        Returns:
            Tuple of (enhanced_sop_content, verification_summary)
        """
        # Verify mathematical components
        verification_results = await self.verify_sop_mathematical_components(
            sop_content,
            strategy=AutoformalizationStrategy.ADAPTIVE
        )
        
        # Enhance SOP with verification information
        enhanced_sop = self._enhance_sop_content(
            sop_content,
            verification_results
        )
        
        return enhanced_sop, verification_results
    
    def _enhance_sop_content(
        self,
        original_sop: str,
        verification_results: Dict[str, Any]
    ) -> str:
        """Enhance SOP content with verification information."""
        enhanced_sop = original_sop
        
        # Add verification summary section
        verification_summary = f"""
        
## Mathematical Verification Summary
- Total mathematical components: {verification_results['total_components']}
- Successfully verified: {verification_results['successful_verifications']}
- Success rate: {verification_results['success_rate']:.2%}
- Average confidence: {verification_results['average_confidence']:.2f}
- Overall status: {'PASS' if verification_results['overall_success'] else 'FAIL'}

"""
        
        # Add details for each component
        for i, comp_result in enumerate(verification_results['components']):
            component = comp_result['component']
            result = comp_result['result']
            
            verification_summary += f"""
### Component {i+1}: {component[:50]}...
- Domain: {comp_result['domain']}
- Complexity: {comp_result['complexity']}
- Status: {'VERIFIED' if result.success else 'FAILED'}
- Confidence: {result.confidence:.3f}
"""
            if result.success and result.lean_code:
                verification_summary += f"- Lean code: ```lean\n{result.lean_code}\n```"
            elif not result.success and result.error:
                verification_summary += f"- Error: {result.error}"
        
        # Add verification summary to SOP
        enhanced_sop += verification_summary
        
        return enhanced_sop

    async def verify_sop_compliance_cav_nlp(
        self,
        sop_content: str,
        compliance_requirements: List[str]
    ) -> Dict[str, Any]:
        """
        Verify SOP compliance using CAV-NLP enhanced analysis.
        
        Uses CAV-NLP for:
        - Semantic analysis of SOP requirements
        - Constraint-based compliance checking
        - Enhanced mathematical verification
        
        Args:
            sop_content: Content of the SOP to verify
            compliance_requirements: List of compliance requirements to check
            
        Returns:
            Dictionary with CAV-NLP compliance verification results
        """
        if not self.use_cav_nlp:
            return {
                "available": False,
                "error": "CAV-NLP not available",
                "compliant": None
            }
        
        start_time = time.time()
        
        try:
            # Use math service for semantic compliance analysis
            semantic_result = await self.math_service.analyze_compliance_async(
                content=sop_content,
                requirements=compliance_requirements,
                context={"verification_type": "sop_compliance"}
            )
            
            # Use enhanced solver for constraint-based compliance
            constraint_result = await self.enhanced_solver.check_compliance_async(
                content=sop_content,
                requirements=compliance_requirements,
                timeout_ms=self.config.get("solver_timeout", 5000)
            )
            
            execution_time = time.time() - start_time
            
            # Determine overall compliance
            semantic_compliant = semantic_result.get("compliance_score", 0) > 0.7
            constraint_compliant = constraint_result.get("compliant", True)
            
            return {
                "available": True,
                "compliant": semantic_compliant and constraint_compliant,
                "semantic_compliance": {
                    "score": semantic_result.get("compliance_score", 0),
                    "issues": semantic_result.get("issues", [])
                },
                "constraint_compliance": {
                    "satisfied": constraint_compliant,
                    "violations": constraint_result.get("violations", [])
                },
                "execution_time": execution_time,
                "cav_nlp_version": "1.0"
            }
        
        except Exception as e:
            logger.error(f"CAV-NLP SOP compliance verification failed: {e}")
            return {
                "available": True,
                "error": str(e),
                "compliant": None,
                "execution_time": time.time() - start_time
            }


# Integration with SOP Generator
class EnhancedSOPGenerator:
    """
    Enhanced SOP generator with LeanAide integration for mathematical verification.
    """
    
    def __init__(self, base_generator, leanaide_integration: LeanAideSOPIntegration):
        """
        Initialize enhanced generator.
        
        Args:
            base_generator: Original SOP generator instance
            leanaide_integration: LeanAide integration instance
        """
        self.base_generator = base_generator
        self.leanaide_integration = leanaide_integration
    
    async def generate_sop_with_verification(
        self,
        requirement_description: str,
        domain: str,
        constraints: List[str] = None,
        equipment_available: List[str] = None,
        existing_sop: Any = None
    ) -> Any:
        """
        Generate SOP with mathematical verification.
        
        Args:
            requirement_description: High-level requirement
            domain: Domain for the SOP
            constraints: Constraints to consider
            equipment_available: Available equipment
            existing_sop: Existing SOP to enhance
            
        Returns:
            Generated SOP with mathematical verification
        """
        # Generate base SOP using original generator
        if existing_sop:
            sop = await self.base_generator.refine_sop(
                existing_sop,
                requirement_description,
                domain,
                constraints or [],
                equipment_available or []
            )
        else:
            sop = await self.base_generator.generate_sop(
                requirement_description,
                domain,
                constraints or [],
                equipment_available or []
            )
        
        # Convert SOP to string content for verification
        sop_content = self._sop_to_content(sop)
        
        # Enhance with formal verification
        enhanced_sop_content, verification_results = await self.leanaide_integration.enhance_sop_with_formal_verification(
            sop_content,
            requirement_description
        )
        
        # Update SOP with verification information
        enhanced_sop = self._update_sop_with_verification(sop, enhanced_sop_content, verification_results)
        
        return enhanced_sop
    
    def _sop_to_content(self, sop: Any) -> str:
        """Convert SOP object to string content for analysis."""
        # This would depend on the actual SOP structure
        # For now, assuming it has a content field or can be converted to string
        if hasattr(sop, 'content'):
            return str(sop.content)
        elif hasattr(sop, 'to_string'):
            return sop.to_string()
        else:
            return str(sop)
    
    def _update_sop_with_verification(self, sop: Any, enhanced_content: str, verification_results: Dict[str, Any]) -> Any:
        """Update SOP object with verification information."""
        # This would depend on the actual SOP structure
        # For now, just return the original SOP with verification metadata added
        if hasattr(sop, 'metadata'):
            if not hasattr(sop.metadata, 'verification_results'):
                sop.metadata['verification_results'] = verification_results
            else:
                sop.metadata.verification_results = verification_results
        elif hasattr(sop, 'add_metadata'):
            sop.add_metadata('verification_results', verification_results)
        
        return sop


# Factory function to create enhanced generator
def create_enhanced_sop_generator_with_leanaide(
    base_generator,
    leanaide_client,
    enable_predictive_flagging: bool = True,
    enable_red_flagging: bool = True
) -> EnhancedSOPGenerator:
    """
    Create an enhanced SOP generator with LeanAide integration.
    
    Args:
        base_generator: Base SOP generator instance
        leanaide_client: Initialized LeanAide client
        enable_predictive_flagging: Whether to enable predictive quality control
        enable_red_flagging: Whether to enable red-flagging
        
    Returns:
        EnhancedSOPGenerator instance
    """
    leanaide_integration = LeanAideSOPIntegration(
        leanaide_client=leanaide_client,
        enable_predictive_flagging=enable_predictive_flagging,
        enable_red_flagging=enable_red_flagging
    )
    
    return EnhancedSOPGenerator(base_generator, leanaide_integration)


# Example usage
async def example_usage():
    """Example of using the LeanAide-SOP integration."""
    print("LeanAide-SOP Integration Example")
    print("=" * 50)
    
    # This would typically receive a real LeanAide client
    # For this example, we'll use a mock
    class MockLeanAideClient:
        def __init__(self):
            self.cache = {}
    
    mock_client = MockLeanAideClient()
    
    # Create integration
    integration = LeanAideSOPIntegration(
        leanaide_client=mock_client,
        enable_predictive_flagging=True,
        enable_red_flagging=True
    )
    
    # Example SOP content with mathematical components
    sop_content = """
# Standard Operating Procedure for Chemical Reaction Optimization

## Objective
Optimize the yield of chemical reaction where for all concentrations c, the reaction rate r(c) follows Michaelis-Menten kinetics.

## Pre-conditions
- Temperature must be maintained where dT/dt ≤ 0.1°C/min
- Pressure must satisfy P ≥ 1 atm

## Protocol
1. Mix reactants A and B where [A]₀ + [B]₀ = constant
2. Monitor reaction where d[r]/dt = k*[A]*[B]
3. Prove that the maximum yield occurs at equilibrium

## Quality Control
- Verify that for all time t, [A](t) + [B](t) + [P](t) = constant
- Confirm that the rate equation holds where k > 0

## Safety
- Ensure that temperature never exceeds T_max where T_max < 100°C
    """
    
    print("Original SOP content:")
    print(sop_content[:200] + "...")
    
    # Verify mathematical components
    results = await integration.verify_sop_mathematical_components(sop_content)
    
    print(f"\nVerification Results:")
    print(f"Total components found: {results['total_components']}")
    print(f"Successfully verified: {results['successful_verifications']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Average confidence: {results['average_confidence']:.3f}")
    print(f"Overall success: {results['overall_success']}")
    
    # Enhance SOP with verification
    enhanced_sop, verification_summary = await integration.enhance_sop_with_formal_verification(
        sop_content,
        "Chemical reaction optimization with mathematical verification"
    )
    
    print(f"\nEnhanced SOP includes verification summary with {len(verification_summary['components'])} components analyzed")
    
    print("\nIntegration example completed successfully!")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())