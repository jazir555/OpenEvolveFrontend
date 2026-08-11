"""
Z3 Reliability Checker for ROMA/MDAP Integration

Provides formal verification of reliability constraints in multi-agent workflows:
- Component reliability verification
- Entanglement constraint validation
- Contract satisfaction checking
- Temporal reliability properties
- Failure mode analysis

Integrates with:
- roma_mdap_maker_engine.py
- roma_recomposition_config.py
- sovereign_reliability.py

Author: OpenEvolve
Created: 2026-02-02
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3ResultStatus, Z3SolverResult
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 integration not available")

try:
    from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

# Import solver pool for metrics
try:
    from z3_solver_pool import get_solver_pool, get_active_solvers_count, get_queue_depth_count
    SOLVER_POOL_AVAILABLE = True
except ImportError:
    SOLVER_POOL_AVAILABLE = False

# CAV-NLP integration for enhanced verification
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available")

# Import ROMA components
try:
    from roma_recomposition_config import Component, EntanglementConstraint
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    logger.warning("ROMA components not available")

try:
    from sovereign_reliability import ReliabilityMetrics
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False


# =============================================================================
# Data Classes and Enums
# =============================================================================

class ReliabilityProperty(Enum):
    """Types of reliability properties to verify."""
    AVAILABILITY = "availability"
    MTBF = "mean_time_between_failures"
    MTTR = "mean_time_to_repair"
    FAULT_TOLERANCE = "fault_tolerance"
    REDUNDANCY = "redundancy"
    FAIL_SAFE = "fail_safe"


@dataclass
class ReliabilityConstraint:
    """A reliability constraint for verification."""
    property_type: ReliabilityProperty
    threshold: float
    target_component: Optional[str] = None
    priority: int = 1
    
    def to_smtlib(self) -> str:
        """Convert to SMT-LIB assertion."""
        var_name = f"reliability_{self.property_type.value}"
        if self.target_component:
            var_name = f"{var_name}_{self.target_component}"
        return f"(>= {var_name} {self.threshold})"


@dataclass
class ComponentReliabilityModel:
    """Reliability model for a system component."""
    component_id: str
    availability: float = 0.99
    mtbf_hours: float = 8760.0  # 1 year
    mttr_hours: float = 1.0
    failure_rate: float = 0.0
    redundancy_factor: int = 1
    
    def calculate_availability(self) -> float:
        """Calculate availability from MTBF and MTTR."""
        if self.mtbf_hours + self.mttr_hours == 0:
            return 0.0
        return self.mtbf_hours / (self.mtbf_hours + self.mttr_hours)
    
    def to_z3_variables(self) -> List[Z3Variable]:
        """Convert to Z3 variables."""
        return [
            Z3Variable(f"availability_{self.component_id}", Z3ConstraintType.REAL),
            Z3Variable(f"mtbf_{self.component_id}", Z3ConstraintType.REAL),
            Z3Variable(f"mttr_{self.component_id}", Z3ConstraintType.REAL),
        ]
    
    def to_z3_constraints(self) -> List[Z3Constraint]:
        """Convert to Z3 constraints."""
        constraints = [
            Z3Constraint(f"(>= availability_{self.component_id} 0.0)", Z3ConstraintType.REAL),
            Z3Constraint(f"(<= availability_{self.component_id} 1.0)", Z3ConstraintType.REAL),
            Z3Constraint(f"(>= mtbf_{self.component_id} 0.0)", Z3ConstraintType.REAL),
            Z3Constraint(f"(>= mttr_{self.component_id} 0.0)", Z3ConstraintType.REAL),
        ]
        return constraints


@dataclass
class VerificationResult:
    """Result of reliability verification."""
    success: bool
    verified: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    counterexample: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    proof: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "verified": self.verified,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "counterexample": self.counterexample,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class ContractSpecification:
    """Contract between components."""
    contract_id: str
    provider: str
    consumer: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    reliability_slo: float = 0.99
    
    def to_z3_constraints(self) -> List[Z3Constraint]:
        """Convert contract to Z3 constraints."""
        constraints = []
        
        # Preconditions
        for i, pre in enumerate(self.preconditions):
            constraints.append(Z3Constraint(
                f"contract_{self.contract_id}_pre_{i}",
                Z3ConstraintType.BOOLEAN,
                description=f"Precondition: {pre}"
            ))
        
        # Postconditions
        for i, post in enumerate(self.postconditions):
            constraints.append(Z3Constraint(
                f"contract_{self.contract_id}_post_{i}",
                Z3ConstraintType.BOOLEAN,
                description=f"Postcondition: {post}"
            ))
        
        # Reliability SLO
        constraints.append(Z3Constraint(
            f"(>= reliability_{self.contract_id} {self.reliability_slo})",
            Z3ConstraintType.REAL,
            description=f"Reliability SLO: {self.reliability_slo}"
        ))
        
        return constraints


# =============================================================================
# Z3 Reliability Checker
# =============================================================================

class Z3ReliabilityChecker:
    """
    Formal verification of reliability constraints using Z3.
    
    Capabilities:
    - Verify component reliability meets targets
    - Check entanglement constraints are satisfiable
    - Validate contract satisfaction
    - Analyze failure scenarios
    - Generate counterexamples for violations
    """
    
    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or (Z3Config(timeout=60.0, proof_generation=True) if Z3_AVAILABLE else None)
        self.solver = None
        self.prover = None
        
        if Z3_AVAILABLE and self.config:
            self.solver = Z3SolverEngine(self.config)
            self.prover = Z3TheoremProver(self.config)
        
        # CAV-NLP enhanced verification
        self.use_cav_nlp = True
        if isinstance(config, dict):
            self.use_cav_nlp = config.get("use_cav_nlp", True)
        self.use_cav_nlp = self.use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
        
        # Cache for verification results
        self._verification_cache: Dict[str, VerificationResult] = {}
        
        # Statistics
        self._stats = {
            "total_checks": 0,
            "successful_checks": 0,
            "violations_found": 0,
            "cache_hits": 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get checker status with solver pool metrics and CAV-NLP."""
        status = {
            "z3_available": Z3_AVAILABLE,
            "z3_advanced_available": Z3_ADVANCED_AVAILABLE,
            "cav_nlp_available": CAV_NLP_AVAILABLE,
            "cav_nlp_enabled": self.use_cav_nlp,
            "roma_available": ROMA_AVAILABLE,
            "reliability_available": RELIABILITY_AVAILABLE,
            "statistics": self._stats.copy(),
            "cache_size": len(self._verification_cache)
        }
        
        # Add solver pool metrics if available
        if SOLVER_POOL_AVAILABLE:
            try:
                status["solver_pool"] = {
                    "active_solvers": get_active_solvers_count(),
                    "queue_depth": get_queue_depth_count(),
                    "available": True
                }
            except Exception as e:
                status["solver_pool"] = {
                    "available": False,
                    "error": str(e)
                }
        else:
            status["solver_pool"] = {"available": False}
        
        return status
    
    # =====================================================================
    # Component Reliability Verification
    # =====================================================================
    
    def verify_component_reliability(
        self,
        component: Union[ComponentReliabilityModel, Any],
        requirements: List[ReliabilityConstraint],
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify that a component meets reliability requirements.
        
        Args:
            component: Component model or ROMA Component
            requirements: List of reliability constraints to verify
            context: Additional context for verification
            
        Returns:
            VerificationResult with verification status
        """
        start_time = time.time()
        self._stats["total_checks"] += 1
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": "Z3 not available"}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Convert to reliability model if needed
            if ROMA_AVAILABLE and isinstance(component, Component):
                model = self._component_to_model(component)
            else:
                model = component
            
            # Build verification problem
            variables = model.to_z3_variables()
            constraints = model.to_z3_constraints()
            
            # Add requirement constraints
            for req in requirements:
                req_constraint = self._reliability_constraint_to_z3(req, model.component_id)
                if req_constraint:
                    constraints.append(req_constraint)
            
            # Add context constraints if provided
            if context:
                context_constraints = self._extract_context_constraints(context, model.component_id)
                constraints.extend(context_constraints)
            
            # Solve
            result = self.solver.solve_constraints(variables, constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                self._stats["successful_checks"] += 1
                return VerificationResult(
                    success=True,
                    verified=True,
                    execution_time_ms=execution_time,
                    recommendations=self._generate_recommendations(model, requirements, satisfied=True)
                )
            elif result.is_unsat():
                self._stats["violations_found"] += 1
                violations = self._analyze_violations(model, requirements, result)
                counterexample = self._extract_counterexample(result, variables)
                
                return VerificationResult(
                    success=True,
                    verified=False,
                    violations=violations,
                    counterexample=counterexample,
                    execution_time_ms=execution_time,
                    recommendations=self._generate_recommendations(model, requirements, satisfied=False)
                )
            else:
                return VerificationResult(
                    success=False,
                    verified=False,
                    violations=[{"error": "Verification returned unknown", "reason": result.reason}],
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.error(f"Component reliability verification failed: {e}")
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def verify_hybrid_reliability(
        self,
        component: Union[ComponentReliabilityModel, Any],
        requirements: List[ReliabilityConstraint],
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify component reliability using hybrid Z3 + CAV-NLP approach.
        
        Args:
            component: Component model or ROMA Component
            requirements: List of reliability constraints to verify
            context: Additional context for verification
            
        Returns:
            VerificationResult from hybrid verification
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": "Z3 not available"}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Convert to reliability model if needed
            if ROMA_AVAILABLE and isinstance(component, Component):
                model = self._component_to_model(component)
            else:
                model = component
            
            # Build verification problem
            variables = model.to_z3_variables()
            constraints = model.to_z3_constraints()
            
            # Add requirement constraints
            for req in requirements:
                req_constraint = self._reliability_constraint_to_z3(req, model.component_id)
                if req_constraint:
                    constraints.append(req_constraint)
            
            # Add context constraints if provided
            if context:
                context_constraints = self._extract_context_constraints(context, model.component_id)
                constraints.extend(context_constraints)
            
            # Z3 validation
            z3_result = self.solver.solve_constraints(variables, constraints)
            
            # CAV-NLP verification
            if self.use_cav_nlp and CAV_NLP_AVAILABLE:
                try:
                    # Convert constraints to format for CAV-NLP
                    cav_constraints = [c.expression for c in constraints if hasattr(c, 'expression')]
                    cav_result = await self.math_service.verify(cav_constraints)
                    return self._combine_reliability_results(
                        z3_result, cav_result, model, requirements,
                        execution_time=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    logger.warning(f"CAV-NLP verification failed, using Z3 only: {e}")
            
            # Return Z3-only result
            execution_time = (time.time() - start_time) * 1000
            if z3_result.is_sat():
                return VerificationResult(
                    success=True,
                    verified=True,
                    execution_time_ms=execution_time,
                    recommendations=self._generate_recommendations(model, requirements, satisfied=True)
                )
            elif z3_result.is_unsat():
                violations = self._analyze_violations(model, requirements, z3_result)
                counterexample = self._extract_counterexample(z3_result, variables)
                return VerificationResult(
                    success=True,
                    verified=False,
                    violations=violations,
                    counterexample=counterexample,
                    execution_time_ms=execution_time,
                    recommendations=self._generate_recommendations(model, requirements, satisfied=False)
                )
            else:
                return VerificationResult(
                    success=False,
                    verified=False,
                    violations=[{"error": "Unknown result"}],
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.error(f"Hybrid reliability verification failed: {e}")
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _combine_reliability_results(
        self,
        z3_result,
        cav_result,
        model: ComponentReliabilityModel,
        requirements: List[ReliabilityConstraint],
        execution_time: float
    ) -> VerificationResult:
        """Combine Z3 and CAV-NLP reliability results."""
        z3_sat = hasattr(z3_result, 'is_sat') and z3_result.is_sat()
        cav_verified = isinstance(cav_result, dict) and cav_result.get('verified', False)
        
        if z3_sat and cav_verified:
            # Both agree: constraints satisfiable
            return VerificationResult(
                success=True,
                verified=True,
                execution_time_ms=execution_time,
                recommendations=[
                    "Z3: Reliability constraints satisfied",
                    "CAV-NLP: Mathematically verified"
                ]
            )
        elif not z3_sat:
            # Z3 says unsat - reliability violation
            violations = self._analyze_violations(model, requirements, z3_result)
            counterexample = self._extract_counterexample(z3_result, [])
            return VerificationResult(
                success=True,
                verified=False,
                violations=violations,
                counterexample=counterexample,
                execution_time_ms=execution_time,
                recommendations=self._generate_recommendations(model, requirements, satisfied=False)
            )
        else:
            # Z3 says sat but CAV may have issues
            recommendations = ["Z3: Reliability constraints satisfied"]
            if isinstance(cav_result, dict) and cav_result.get('recommendations'):
                recommendations.extend(cav_result['recommendations'])
            
            return VerificationResult(
                success=True,
                verified=z3_sat,
                execution_time_ms=execution_time,
                recommendations=recommendations
            )
    
    def verify_system_reliability(
        self,
        components: List[ComponentReliabilityModel],
        system_requirements: List[ReliabilityConstraint],
        component_dependencies: Optional[Dict[str, List[str]]] = None
    ) -> VerificationResult:
        """
        Verify reliability of a multi-component system.
        
        Args:
            components: List of component reliability models
            system_requirements: System-level reliability requirements
            component_dependencies: Map of component_id -> list of dependencies
            
        Returns:
            VerificationResult
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            # Collect all variables and constraints
            all_variables = []
            all_constraints = []
            
            for component in components:
                all_variables.extend(component.to_z3_variables())
                all_constraints.extend(component.to_z3_constraints())
            
            # Add dependency constraints
            if component_dependencies:
                dep_constraints = self._build_dependency_constraints(
                    components, component_dependencies
                )
                all_constraints.extend(dep_constraints)
            
            # Add system requirements
            for req in system_requirements:
                req_constraint = self._system_reliability_constraint_to_z3(req, components)
                if req_constraint:
                    all_constraints.append(req_constraint)
            
            # Calculate composite availability for series/parallel systems
            composite_constraints = self._build_composite_constraints(components)
            all_constraints.extend(composite_constraints)
            
            # Solve
            result = self.solver.solve_constraints(all_variables, all_constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                return VerificationResult(
                    success=True,
                    verified=True,
                    execution_time_ms=execution_time
                )
            elif result.is_unsat():
                violations = self._analyze_system_violations(components, system_requirements, result)
                return VerificationResult(
                    success=True,
                    verified=False,
                    violations=violations,
                    execution_time_ms=execution_time,
                    recommendations=self._generate_system_recommendations(components, violations)
                )
            else:
                return VerificationResult(
                    success=False,
                    verified=False,
                    violations=[{"error": "Unknown result"}],
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            logger.error(f"System reliability verification failed: {e}")
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    # =====================================================================
    # Contract Verification
    # =====================================================================
    
    def verify_contract_satisfaction(
        self,
        contract: ContractSpecification,
        provider_model: ComponentReliabilityModel,
        consumer_model: ComponentReliabilityModel
    ) -> VerificationResult:
        """
        Verify that a contract between components is satisfiable.
        
        Args:
            contract: Contract specification
            provider_model: Provider component model
            consumer_model: Consumer component model
            
        Returns:
            VerificationResult
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            # Build variables and constraints
            variables = []
            constraints = []
            
            # Add provider and consumer variables
            variables.extend(provider_model.to_z3_variables())
            variables.extend(consumer_model.to_z3_variables())
            
            # Add component constraints
            constraints.extend(provider_model.to_z3_constraints())
            constraints.extend(consumer_model.to_z3_constraints())
            
            # Add contract constraints
            contract_constraints = contract.to_z3_constraints()
            constraints.extend(contract_constraints)
            
            # Add reliability implication: provider_avail >= contract_slo
            reliability_constraint = Z3Constraint(
                f"(>= availability_{provider_model.component_id} {contract.reliability_slo})",
                Z3ConstraintType.REAL
            )
            constraints.append(reliability_constraint)
            
            # Solve
            result = self.solver.solve_constraints(variables, constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                return VerificationResult(
                    success=True,
                    verified=True,
                    execution_time_ms=execution_time,
                    recommendations=[f"Contract {contract.contract_id} is satisfiable"]
                )
            else:
                return VerificationResult(
                    success=True,
                    verified=False,
                    violations=[{
                        "contract_id": contract.contract_id,
                        "issue": "Contract cannot be satisfied with given component reliabilities"
                    }],
                    execution_time_ms=execution_time,
                    recommendations=[
                        f"Increase {provider_model.component_id} availability to at least {contract.reliability_slo}",
                        "Consider adding redundancy",
                        "Relax contract SLO if possible"
                    ]
                )
                
        except Exception as e:
            logger.error(f"Contract verification failed: {e}")
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    # =====================================================================
    # Entanglement Verification
    # =====================================================================
    
    def verify_entanglement_constraints(
        self,
        entanglements: List[Dict[str, Any]],
        component_models: Dict[str, ComponentReliabilityModel]
    ) -> VerificationResult:
        """
        Verify that entanglement constraints between components are satisfiable.
        
        Args:
            entanglements: List of entanglement specifications
            component_models: Map of component_id to reliability model
            
        Returns:
            VerificationResult
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": "Z3 not available"}]
            )
        
        try:
            variables = []
            constraints = []
            
            # Add all component variables
            for model in component_models.values():
                variables.extend(model.to_z3_variables())
                constraints.extend(model.to_z3_constraints())
            
            # Add entanglement constraints
            for ent in entanglements:
                ent_constraints = self._entanglement_to_z3_constraints(
                    ent, component_models
                )
                constraints.extend(ent_constraints)
            
            # Solve
            result = self.solver.solve_constraints(variables, constraints)
            execution_time = (time.time() - start_time) * 1000
            
            if result.is_sat():
                return VerificationResult(
                    success=True,
                    verified=True,
                    execution_time_ms=execution_time,
                    recommendations=["All entanglement constraints are satisfiable"]
                )
            else:
                violations = []
                for ent in entanglements:
                    violations.append({
                        "entanglement_id": ent.get("id", "unknown"),
                        "issue": "Entanglement constraint is unsatisfiable"
                    })
                
                return VerificationResult(
                    success=True,
                    verified=False,
                    violations=violations,
                    execution_time_ms=execution_time,
                    recommendations=[
                        "Review entanglement constraints for contradictions",
                        "Check component reliability assumptions",
                        "Consider relaxing coupling between components"
                    ]
                )
                
        except Exception as e:
            logger.error(f"Entanglement verification failed: {e}")
            return VerificationResult(
                success=False,
                verified=False,
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    # =====================================================================
    # Failure Mode Analysis
    # =====================================================================
    
    def analyze_failure_modes(
        self,
        components: List[ComponentReliabilityModel],
        failure_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze system behavior under various failure scenarios.
        
        Args:
            components: System components
            failure_scenarios: List of failure scenarios to analyze
            
        Returns:
            Analysis results with impact assessment
        """
        if not Z3_AVAILABLE:
            return {"error": "Z3 not available"}
        
        results = {
            "scenarios_analyzed": 0,
            "system_failures": 0,
            "degraded_operations": 0,
            "resilient_configs": 0,
            "scenario_results": []
        }
        
        for scenario in failure_scenarios:
            scenario_result = self._analyze_single_scenario(components, scenario)
            results["scenario_results"].append(scenario_result)
            results["scenarios_analyzed"] += 1
            
            if scenario_result["system_failure"]:
                results["system_failures"] += 1
            elif scenario_result["degraded"]:
                results["degraded_operations"] += 1
            else:
                results["resilient_configs"] += 1
        
        return results
    
    def _analyze_single_scenario(
        self,
        components: List[ComponentReliabilityModel],
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single failure scenario."""
        failed_components = scenario.get("failed_components", [])
        degraded_components = scenario.get("degraded_components", [])
        
        # Build scenario constraints
        variables = []
        constraints = []
        
        for comp in components:
            variables.extend(comp.to_z3_variables())
            
            # If component fails, availability = 0
            if comp.component_id in failed_components:
                constraints.append(Z3Constraint(
                    f"(= availability_{comp.component_id} 0.0)",
                    Z3ConstraintType.REAL
                ))
            # If degraded, reduce availability
            elif comp.component_id in degraded_components:
                constraints.append(Z3Constraint(
                    f"(<= availability_{comp.component_id} {comp.availability * 0.5})",
                    Z3ConstraintType.REAL
                ))
            else:
                constraints.extend(comp.to_z3_constraints())
        
        # Check if system can still meet requirements
        result = self.solver.solve_constraints(variables, constraints)
        
        return {
            "scenario_id": scenario.get("id", "unknown"),
            "failed_components": failed_components,
            "degraded_components": degraded_components,
            "system_failure": not result.is_sat(),
            "degraded": len(degraded_components) > 0 and result.is_sat(),
            "satisfiable": result.is_sat()
        }
    
    # =====================================================================
    # Helper Methods
    # =====================================================================
    
    def _component_to_model(self, component: Any) -> ComponentReliabilityModel:
        """Convert ROMA Component to reliability model."""
        # Extract reliability attributes from component
        return ComponentReliabilityModel(
            component_id=getattr(component, 'id', str(component)),
            availability=getattr(component, 'availability', 0.99),
            mtbf_hours=getattr(component, 'mtbf_hours', 8760.0),
            mttr_hours=getattr(component, 'mttr_hours', 1.0),
            redundancy_factor=getattr(component, 'redundancy', 1)
        )
    
    def _reliability_constraint_to_z3(
        self,
        constraint: ReliabilityConstraint,
        component_id: str
    ) -> Optional[Z3Constraint]:
        """Convert reliability constraint to Z3 constraint."""
        var_name = f"{constraint.property_type.value}_{component_id}"
        
        if constraint.property_type == ReliabilityProperty.AVAILABILITY:
            return Z3Constraint(
                f"(>= {var_name} {constraint.threshold})",
                Z3ConstraintType.REAL
            )
        elif constraint.property_type == ReliabilityProperty.MTBF:
            return Z3Constraint(
                f"(>= {var_name} {constraint.threshold})",
                Z3ConstraintType.REAL
            )
        elif constraint.property_type == ReliabilityProperty.MTTR:
            return Z3Constraint(
                f"(<= {var_name} {constraint.threshold})",
                Z3ConstraintType.REAL
            )
        return None
    
    def _extract_context_constraints(
        self,
        context: Dict[str, Any],
        component_id: str
    ) -> List[Z3Constraint]:
        """Extract additional constraints from context."""
        constraints = []
        
        # Environmental constraints
        if "max_load" in context:
            constraints.append(Z3Constraint(
                f"(<= load_{component_id} {context['max_load']})",
                Z3ConstraintType.REAL
            ))
        
        return constraints
    
    def _build_dependency_constraints(
        self,
        components: List[ComponentReliabilityModel],
        dependencies: Dict[str, List[str]]
    ) -> List[Z3Constraint]:
        """Build constraints for component dependencies."""
        constraints = []
        
        for comp_id, deps in dependencies.items():
            for dep_id in deps:
                # If dependency fails, component fails
                # availability_comp <= availability_dep
                constraints.append(Z3Constraint(
                    f"(<= availability_{comp_id} availability_{dep_id})",
                    Z3ConstraintType.REAL
                ))
        
        return constraints
    
    def _build_composite_constraints(
        self,
        components: List[ComponentReliabilityModel]
    ) -> List[Z3Constraint]:
        """Build constraints for composite system availability using standard reliability models."""
        constraints = []
        
        if not components:
            return constraints

        # Series system: A_sys = product(A_i)
        # Parallel system: A_sys = 1 - product(1 - A_i)
        
        # SMT-LIB doesn't have a product operator for variables easily,
        # but for small systems we can expand it.
        # Here we add variables for different composition types.
        
        avail_vars = [f"availability_{c.component_id}" for c in components]
        
        # Series Model
        if len(avail_vars) == 1:
            constraints.append(Z3Constraint(
                f"(= system_availability_series {avail_vars[0]})",
                Z3ConstraintType.REAL
            ))
        else:
            series_expr = avail_vars[0]
            for v in avail_vars[1:]:
                series_expr = f"(* {series_expr} {v})"
            constraints.append(Z3Constraint(
                f"(= system_availability_series {series_expr})",
                Z3ConstraintType.REAL
            ))
            
        # Parallel Model (assuming independent failures)
        if len(avail_vars) == 1:
            constraints.append(Z3Constraint(
                f"(= system_availability_parallel {avail_vars[0]})",
                Z3ConstraintType.REAL
            ))
        else:
            # 1 - (1-a1)*(1-a2)*...
            product_unavail = f"(- 1.0 {avail_vars[0]})"
            for v in avail_vars[1:]:
                product_unavail = f"(* {product_unavail} (- 1.0 {v}))"
            
            constraints.append(Z3Constraint(
                f"(= system_availability_parallel (- 1.0 {product_unavail}))",
                Z3ConstraintType.REAL
            ))
            
        # Default to series for 'system_availability' as it's the conservative lower bound
        constraints.append(Z3Constraint(
            "(= system_availability system_availability_series)",
            Z3ConstraintType.REAL
        ))
        
        return constraints
    
    def _system_reliability_constraint_to_z3(
        self,
        constraint: ReliabilityConstraint,
        components: List[ComponentReliabilityModel]
    ) -> Optional[Z3Constraint]:
        """Convert system reliability constraint to Z3."""
        if constraint.target_component:
            return self._reliability_constraint_to_z3(constraint, constraint.target_component)
        
        # System-wide constraint
        return Z3Constraint(
            f"(>= system_{constraint.property_type.value} {constraint.threshold})",
            Z3ConstraintType.REAL
        )
    
    def _entanglement_to_z3_constraints(
        self,
        entanglement: Dict[str, Any],
        component_models: Dict[str, ComponentReliabilityModel]
    ) -> List[Z3Constraint]:
        """Convert entanglement to Z3 constraints."""
        constraints = []
        
        source = entanglement.get("source")
        target = entanglement.get("target")
        coupling_type = entanglement.get("type", "strong")
        
        if source and target:
            if coupling_type == "strong":
                # Strong coupling: availabilities are equal
                constraints.append(Z3Constraint(
                    f"(= availability_{source} availability_{target})",
                    Z3ConstraintType.REAL
                ))
            elif coupling_type == "weak":
                # Weak coupling: if source fails, target affected
                constraints.append(Z3Constraint(
                    f"(<= availability_{target} availability_{source})",
                    Z3ConstraintType.REAL
                ))
        
        return constraints
    
    def _analyze_violations(
        self,
        model: ComponentReliabilityModel,
        requirements: List[ReliabilityConstraint],
        result: Z3SolverResult
    ) -> List[Dict[str, Any]]:
        """Analyze which requirements were violated."""
        violations = []
        
        for req in requirements:
            violations.append({
                "property": req.property_type.value,
                "required": req.threshold,
                "component": model.component_id,
                "issue": f"Component does not meet {req.property_type.value} requirement"
            })
        
        return violations
    
    def _analyze_system_violations(
        self,
        components: List[ComponentReliabilityModel],
        requirements: List[ReliabilityConstraint],
        result: Z3SolverResult
    ) -> List[Dict[str, Any]]:
        """Analyze system-level violations."""
        violations = []
        
        for req in requirements:
            violations.append({
                "property": req.property_type.value,
                "required": req.threshold,
                "scope": "system",
                "issue": f"System does not meet {req.property_type.value} requirement"
            })
        
        return violations
    
    def _extract_counterexample(
        self,
        result: Z3SolverResult,
        variables: List[Z3Variable]
    ) -> Optional[Dict[str, Any]]:
        """Extract counterexample from solver result."""
        if result.model:
            return {
                "variable_assignments": result.model.assignments,
                "explanation": "These values demonstrate the constraint violation"
            }
        return None
    
    def _generate_recommendations(
        self,
        model: ComponentReliabilityModel,
        requirements: List[ReliabilityConstraint],
        satisfied: bool
    ) -> List[str]:
        """Generate recommendations based on verification result."""
        if satisfied:
            return ["Component meets all reliability requirements"]
        
        recommendations = []
        
        for req in requirements:
            if req.property_type == ReliabilityProperty.AVAILABILITY:
                recommendations.append(
                    f"Increase {model.component_id} availability to at least {req.threshold}"
                )
                recommendations.append("Consider adding redundant instances")
            elif req.property_type == ReliabilityProperty.MTBF:
                recommendations.append(
                    f"Improve MTBF to at least {req.threshold} hours"
                )
            elif req.property_type == ReliabilityProperty.MTTR:
                recommendations.append(
                    f"Reduce MTTR to at most {req.threshold} hours"
                )
        
        return recommendations
    
    def _generate_system_recommendations(
        self,
        components: List[ComponentReliabilityModel],
        violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = [
            "System reliability requirements not met",
            "Consider adding redundancy to critical components",
            "Review component dependencies for single points of failure",
            "Implement circuit breakers and graceful degradation"
        ]
        
        return recommendations


# =============================================================================
# Integration Helpers for ROMA
# =============================================================================

class ROMAZ3Integration:
    """Integration layer between ROMA and Z3 reliability checker."""
    
    def __init__(self):
        self.checker = Z3ReliabilityChecker()
    
    def verify_recomposition_plan(
        self,
        components: List[Any],
        entanglements: List[Any],
        reliability_targets: Dict[str, float]
    ) -> VerificationResult:
        """
        Verify a ROMA recomposition plan meets reliability targets.
        
        Args:
            components: ROMA components
            entanglements: Entanglement constraints
            reliability_targets: Map of property -> target value
            
        Returns:
            VerificationResult
        """
        # Convert ROMA components to reliability models
        models = []
        for comp in components:
            model = self.checker._component_to_model(comp)
            models.append(model)
        
        # Build requirements
        requirements = [
            ReliabilityConstraint(
                property_type=ReliabilityProperty(prop),
                threshold=value
            )
            for prop, value in reliability_targets.items()
        ]
        
        # Verify system reliability
        return self.checker.verify_system_reliability(models, requirements)
    
    def validate_component_contracts(
        self,
        contracts: List[ContractSpecification],
        component_models: Dict[str, ComponentReliabilityModel]
    ) -> Dict[str, VerificationResult]:
        """
        Validate all contracts in a composition.
        
        Args:
            contracts: List of contracts
            component_models: Component reliability models
            
        Returns:
            Map of contract_id -> VerificationResult
        """
        results = {}
        
        for contract in contracts:
            provider = component_models.get(contract.provider)
            consumer = component_models.get(contract.consumer)
            
            if provider and consumer:
                result = self.checker.verify_contract_satisfaction(
                    contract, provider, consumer
                )
                results[contract.contract_id] = result
        
        return results


# =============================================================================
# Global Instance
# =============================================================================

_reliability_checker: Optional[Z3ReliabilityChecker] = None
_roma_integration: Optional[ROMAZ3Integration] = None


def get_z3_reliability_checker() -> Z3ReliabilityChecker:
    """Get global Z3 reliability checker."""
    global _reliability_checker
    if _reliability_checker is None:
        _reliability_checker = Z3ReliabilityChecker()
    return _reliability_checker


def get_roma_z3_integration() -> ROMAZ3Integration:
    """Get global ROMA-Z3 integration."""
    global _roma_integration
    if _roma_integration is None:
        _roma_integration = ROMAZ3Integration()
    return _roma_integration


# =============================================================================
# Example Usage
# =============================================================================

def example_component_verification():
    """Example: Verify a single component."""
    checker = get_z3_reliability_checker()
    
    # Create component model
    component = ComponentReliabilityModel(
        component_id="auth_service",
        availability=0.995,
        mtbf_hours=4380,  # 6 months
        mttr_hours=0.5
    )
    
    # Define requirements
    requirements = [
        ReliabilityConstraint(
            property_type=ReliabilityProperty.AVAILABILITY,
            threshold=0.999
        ),
        ReliabilityConstraint(
            property_type=ReliabilityProperty.MTBF,
            threshold=8000
        )
    ]
    
    # Verify
    result = checker.verify_component_reliability(component, requirements)
    
    print("Component Verification Result:")
    print(f"  Verified: {result.verified}")
    print(f"  Violations: {len(result.violations)}")
    for rec in result.recommendations:
        print(f"  Recommendation: {rec}")
    
    return result


def example_system_verification():
    """Example: Verify multi-component system."""
    checker = get_z3_reliability_checker()
    
    # Create system components
    components = [
        ComponentReliabilityModel("api_gateway", availability=0.999, mtbf_hours=8760),
        ComponentReliabilityModel("auth_service", availability=0.995, mtbf_hours=4380),
        ComponentReliabilityModel("database", availability=0.9999, mtbf_hours=20000),
        ComponentReliabilityModel("cache", availability=0.99, mtbf_hours=2000)
    ]
    
    # Dependencies
    dependencies = {
        "api_gateway": ["auth_service"],
        "auth_service": ["database"]
    }
    
    # System requirements
    requirements = [
        ReliabilityConstraint(
            property_type=ReliabilityProperty.AVAILABILITY,
            threshold=0.99
        )
    ]
    
    # Verify
    result = checker.verify_system_reliability(components, requirements, dependencies)
    
    print("\nSystem Verification Result:")
    print(f"  Verified: {result.verified}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    
    return result


def example_contract_verification():
    """Example: Verify inter-component contract."""
    checker = get_z3_reliability_checker()
    
    # Create contract
    contract = ContractSpecification(
        contract_id="auth_contract",
        provider="auth_service",
        consumer="api_gateway",
        preconditions=["valid_credentials"],
        postconditions=["authenticated_session"],
        reliability_slo=0.995
    )
    
    # Component models
    provider = ComponentReliabilityModel("auth_service", availability=0.99)
    consumer = ComponentReliabilityModel("api_gateway", availability=0.999)
    
    # Verify contract
    result = checker.verify_contract_satisfaction(contract, provider, consumer)
    
    print("\nContract Verification Result:")
    print(f"  Verified: {result.verified}")
    print(f"  Recommendations: {result.recommendations}")
    
    return result


if __name__ == "__main__":
    print("Z3 Reliability Checker for ROMA/MDAP")
    print("=" * 60)
    
    example_component_verification()
    example_system_verification()
    example_contract_verification()
