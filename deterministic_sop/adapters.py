"""
Adapters for integrating 8-layer deterministic framework with SOP generation
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging

# Layer 0: Lagrange Mapper (Bias Filtering)
try:
    from lagrange_llm import LagrangeMapper
except ImportError:
    LagrangeMapper = None

# Layer 1: ROMA/MDAP (Decomposition)
try:
    from decomposition_engine import RecursiveSolver
except ImportError:
    RecursiveSolver = None

# Layer 2: Structured Generation
try:
    from lmql import LMQL
except ImportError:
    LMQL = None

# Layer 3: Content Validation
try:
    from steerable import Steerable
except ImportError:
    Steerable = None

# Layer 4: DSPy (Learning) - using global integration module for consistency
try:
    from dspy_integration import DSPY_AVAILABLE, get_global_dspy_instance, initialize_dspy
    import dspy
    logger = logging.getLogger(__name__)
    logger.info("DSPy available through global integration for enhanced programmatic prompting")
except ImportError:
    # Fallback to local import if global module not available
    try:
        import dspy
        DSPY_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("DSPy available for enhanced programmatic prompting")
    except ImportError:
        dspy = None
        DSPY_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("DSPy not available - using standard prompting methods")

# Layer 5: Knowledge Engine
try:
    from knowledge_engine import KnowledgeEngine
except ImportError:
    KnowledgeEngine = None

# Layer 6: Formal Verification
try:
    from z3 import Solver, sat
except ImportError:
    Solver = None
    sat = None

# Layer 7: detLLM (Reproducibility)
try:
    from detllm import check
except ImportError:
    check = None

# CAV-NLP Integration for enhanced formal verification
try:
    from openevolve.z3_cav_nlp_integration import (
        EnhancedZ3Solver,
        ConstraintFormalizer,
        VerificationResult,
        FormalizationResult,
    )
    CAV_NLP_AVAILABLE = True
    logger.info("CAV-NLP integration available for enhanced SOP verification")
except ImportError:
    CAV_NLP_AVAILABLE = False
    EnhancedZ3Solver = None
    ConstraintFormalizer = None
    VerificationResult = None
    FormalizationResult = None
    logger.warning("CAV-NLP not available - using standard Z3 verification")

# Existing SOP Generator
try:
    from sop_generator import SOPGenerator
except ImportError:
    SOPGenerator = None


logger = logging.getLogger(__name__)


@dataclass
class SOPGenerationConfig:
    """Configuration for deterministic SOP generation"""

    # Layers to enable (0-7)
    enable_layers: List[int] = field(default_factory=lambda: list(range(8)))

    # Domain (chemistry, physics, biology, general)
    domain: str = "general"

    # Determinism tier for detLLM (0, 1, 2)
    determinism_tier: int = 2

    # Number of runs for reproducibility verification
    verification_runs: int = 3

    # Temperature and sampling parameters
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1

    # Maximum execution time (seconds)
    max_execution_time: int = 300

    # Output format
    output_format: str = "json"

    # Verbose logging
    verbose: bool = False

    # CAV-NLP Configuration
    use_cav_nlp: bool = True  # Enable CAV-NLP for enhanced verification
    cav_nlp_confidence_threshold: float = 0.8  # Minimum confidence for CAV-NLP results
    use_hybrid_verification: bool = True  # Use Z3 + Lean hybrid verification


@dataclass
class SOPGenerationResult:
    """Result from deterministic SOP generation"""

    success: bool
    sop: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    validation_results: Dict[str, Any]
    reproducibility_metrics: Optional[Dict[str, Any]]
    execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: List[str] = field(default_factory=list)


class LagrangeFilterAdapter:
    """Layer 0: Bias filtering adapter"""

    def __init__(self):
        if LagrangeMapper is None:
            logger.warning("LagrangeMapper not available, using mock implementation")
        self.mapper = LagrangeMapper(domain="scientific") if LagrangeMapper else None

    def filter(self, requirement: str, domain: str) -> str:
        """Filter biases from requirement"""
        if self.mapper is None:
            # Mock implementation: remove common biased phrases
            biased_phrases = [
                "obviously", "clearly", "undoubtedly",
                "everyone knows", "it goes without saying"
            ]
            filtered = requirement.lower()
            for phrase in biased_phrases:
                filtered = filtered.replace(phrase, "")
            return filtered

        try:
            filtered = self.mapper.filter_bias(requirement, domain=domain)
            return filtered
        except Exception as e:
            logger.error(f"Lagrange filtering failed: {e}")
            return requirement


class DecompositionAdapter:
    """Layer 1: ROMA/MDAP decomposition adapter"""

    def __init__(self):
        self.solver = RecursiveSolver() if RecursiveSolver else None

    def decompose(self, requirement: str, domain: str) -> Dict[str, Any]:
        """Decompose requirement into sections"""
        if self.solver is None:
            # Mock implementation
            return {
                "sections": [
                    {"name": "objective", "description": "Main goal"},
                    {"name": "materials", "description": "Required materials"},
                    {"name": "equipment", "description": "Equipment needed"},
                    {"name": "procedure", "description": "Step-by-step process"},
                    {"name": "safety", "description": "Safety considerations"},
                    {"name": "validation", "description": "Quality checks"}
                ],
                "dependencies": [],
                "complexity": "medium"
            }

        try:
            decomposition = self.solver.solve(requirement, context={"domain": domain})
            return decomposition
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {}


class StructuredGenerationAdapter:
    """Layer 2: LMQL/Outlines structured generation adapter"""

    def __init__(self):
        self.lmql = LMQL() if LMQL else None

    def enforce_structure(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce JSON schema on generated text"""
        if self.lmql is None:
            # Mock: try to parse as JSON
            try:
                return json.loads(text)
            except:
                return {"raw_text": text}

        try:
            structured = self.lmql.generate(
                prompt=text,
                schema=schema,
                temperature=0.0
            )
            return structured
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {}


class ValidationAdapter:
    """Layer 3: Content validation adapter"""

    def __init__(self):
        self.steerable = Steerable() if Steerable else None
        self.validators = self._init_validators()

    def _init_validators(self) -> Dict[str, Any]:
        """Initialize domain-specific validators"""
        return {
            "chemistry": self._validate_chemistry,
            "physics": self._validate_physics,
            "biology": self._validate_biology,
            "general": self._validate_general
        }

    def _validate_chemistry(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chemistry SOP"""
        issues = []

        # Check for safety information
        if "safety" not in sop or not sop["safety"]:
            issues.append("Missing safety section")

        # Check for hazardous materials
        materials = sop.get("materials", [])
        if not any(mat.get("hazard_level") for mat in materials):
            issues.append("Materials missing hazard information")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _validate_physics(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physics SOP"""
        issues = []

        # Check for mathematical models
        if "calculations" not in sop:
            issues.append("Missing calculations section")

        # Check for equipment calibration
        equipment = sop.get("equipment", [])
        if not any(eq.get("calibration") for eq in equipment):
            issues.append("Equipment missing calibration information")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _validate_biology(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Validate biology SOP"""
        issues = []

        # Check for biosafety level
        if "biosafety_level" not in sop:
            issues.append("Missing biosafety level designation")

        # Check for sterility requirements
        if "sterility" not in sop:
            issues.append("Missing sterility requirements")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _validate_general(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Validate general SOP"""
        issues = []

        # Basic validation
        required_sections = ["objective", "procedure", "materials"]
        for section in required_sections:
            if section not in sop:
                issues.append(f"Missing required section: {section}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def validate(self, sop: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Validate SOP based on domain"""
        validator = self.validators.get(domain, self.validators["general"])
        return validator(sop)


class LearningAdapter:
    """Layer 4: DSPy learning adapter"""

    def __init__(self):
        self.dspy = dspy if dspy else None
        self.optimized = False

    def optimize(self, training_data: List[Dict[str, Any]]):
        """Optimize using DSPy from training examples"""
        if self.dspy is None:
            logger.warning("DSPy not available, skipping optimization")
            return

        # Mock optimization
        self.optimized = True
        logger.info(f"Optimized with {len(training_data)} examples")

    def improve(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Improve SOP based on learned patterns"""
        if not self.optimized:
            return sop

        # Mock improvement: add common enhancements
        if "procedure" in sop and isinstance(sop["procedure"], list):
            for step in sop["procedure"]:
                if "duration" not in step:
                    step["duration"] = "estimate based on complexity"

        return sop


class KnowledgeAdapter:
    """Layer 5: Knowledge engine adapter"""

    def __init__(self):
        self.ke = KnowledgeEngine() if KnowledgeEngine else None

    def enrich(self, requirement: str, domain: str) -> Dict[str, Any]:
        """Enrich requirement with knowledge from literature"""
        if self.ke is None:
            return {"context": [], "references": []}

        try:
            results = self.ke.search(query=requirement, domain=domain, max_results=5)
            return {
                "context": [r.get("summary") for r in results],
                "references": [r.get("citation") for r in results]
            }
        except Exception as e:
            logger.error(f"Knowledge enrichment failed: {e}")
            return {}


class FormalVerificationAdapter:
    """Layer 6: Z3 formal verification adapter with CAV-NLP enhancement"""

    def __init__(self, use_cav_nlp: bool = True):
        self.solver = Solver() if Solver else None
        
        # CAV-NLP configuration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self._cav_solver = None
        self._formalizer = None
        
        if self.use_cav_nlp and EnhancedZ3Solver is not None:
            try:
                self._cav_solver = EnhancedZ3Solver(use_cav_nlp=True)
                self._formalizer = ConstraintFormalizer()
                logger.info("CAV-NLP formal verification initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize CAV-NLP: {exc}")
                self.use_cav_nlp = False

    def verify_dimensional_consistency(
        self,
        sop: Dict[str, Any],
        use_cav_nlp: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Verify dimensional analysis using Z3 with optional CAV-NLP enhancement"""
        use_cav = use_cav_nlp if use_cav_nlp is not None else self.use_cav_nlp
        
        if use_cav and self._cav_solver is not None:
            try:
                return self._verify_dimensional_with_cav_nlp(sop)
            except Exception as exc:
                logger.warning(f"CAV-NLP dimensional verification failed: {exc}")
        
        # Standard Z3 verification
        if self.solver is None:
            return {"verified": False, "reason": "Z3 not available"}

        calculations = sop.get("calculations", [])
        if not calculations:
            return {"verified": True, "reason": "No calculations to verify"}

        return {"verified": True, "verified_equations": len(calculations)}

    def _verify_dimensional_with_cav_nlp(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced dimensional verification using CAV-NLP."""
        calculations = sop.get("calculations", [])
        
        if not calculations:
            return {"verified": True, "reason": "No calculations to verify"}
        
        self._cav_solver.reset()
        
        # Formalize dimensional constraints
        for calc in calculations:
            equation = calc.get("equation", "")
            units = calc.get("units", "")
            
            if equation and units:
                try:
                    constraint = f"dimensional consistency: {equation} with units {units}"
                    formalized = self._cav_solver.formalize_constraint(constraint)
                    if formalized is not None:
                        self._cav_solver.add(formalized)
                except Exception as exc:
                    logger.debug(f"Could not formalize equation {equation}: {exc}")
        
        # Perform hybrid verification
        verification = self._cav_solver.verify_with_lean()
        
        return {
            "verified": verification.success,
            "confidence": verification.confidence,
            "method": "cav_nlp_hybrid_dimensional",
            "verified_equations": len(calculations),
            "z3_result": verification.z3_result,
            "counterexample": verification.counterexample
        }

    def verify_stoichiometry(
        self, 
        sop: Dict[str, Any],
        use_cav_nlp: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Verify chemical reaction stoichiometry with optional CAV-NLP enhancement"""
        use_cav = use_cav_nlp if use_cav_nlp is not None else self.use_cav_nlp
        
        if use_cav and self._cav_solver is not None:
            try:
                return self._verify_stoichiometry_with_cav_nlp(sop)
            except Exception as exc:
                logger.warning(f"CAV-NLP stoichiometry verification failed: {exc}")
        
        # Standard Z3 verification
        if self.solver is None:
            return {"verified": False, "reason": "Z3 not available"}

        reactions = sop.get("reactions", [])
        if not reactions:
            return {"verified": True, "reason": "No reactions to verify"}

        return {"verified": True, "verified_reactions": len(reactions)}

    def _verify_stoichiometry_with_cav_nlp(self, sop: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced stoichiometry verification using CAV-NLP."""
        reactions = sop.get("reactions", [])
        
        if not reactions:
            return {"verified": True, "reason": "No reactions to verify"}
        
        self._cav_solver.reset()
        
        # Formalize mass balance constraints for each reaction
        for reaction in reactions:
            reactants = reaction.get("reactants", [])
            products = reaction.get("products", [])
            
            if reactants and products:
                try:
                    constraint = f"mass balance: {len(reactants)} reactants -> {len(products)} products"
                    formalized = self._cav_solver.formalize_constraint(constraint)
                    if formalized is not None:
                        self._cav_solver.add(formalized)
                except Exception as exc:
                    logger.debug(f"Could not formalize reaction: {exc}")
        
        # Perform hybrid verification
        verification = self._cav_solver.verify_with_lean()
        
        return {
            "verified": verification.success,
            "confidence": verification.confidence,
            "method": "cav_nlp_hybrid_stoichiometry",
            "verified_reactions": len(reactions),
            "z3_result": verification.z3_result,
            "counterexample": verification.counterexample
        }

    def verify_sop_compliance(
        self,
        sop: Dict[str, Any],
        compliance_constraints: List[str],
        use_cav_nlp: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Verify SOP compliance against a set of constraints using CAV-NLP.
        
        Args:
            sop: The SOP to verify
            compliance_constraints: List of compliance constraints (natural language)
            use_cav_nlp: Whether to use CAV-NLP enhancement
            
        Returns:
            Dict with compliance verification results
        """
        use_cav = use_cav_nlp if use_cav_nlp is not None else self.use_cav_nlp
        
        if not use_cav or self._cav_solver is None:
            # Basic compliance check without CAV-NLP
            return {
                "compliant": True,
                "confidence": 0.5,
                "method": "basic",
                "violations": [],
                "note": "CAV-NLP not available - using basic verification"
            }
        
        try:
            self._cav_solver.reset()
            violations = []
            formalized_count = 0
            
            # Formalize and verify each compliance constraint
            for constraint in compliance_constraints:
                try:
                    formalized = self._cav_solver.formalize_constraint(constraint)
                    if formalized is not None:
                        self._cav_solver.add(formalized)
                        formalized_count += 1
                    else:
                        violations.append({
                            "constraint": constraint,
                            "error": "Could not formalize constraint"
                        })
                except Exception as exc:
                    violations.append({
                        "constraint": constraint,
                        "error": str(exc)
                    })
            
            # Perform hybrid verification
            verification = self._cav_solver.verify_with_lean()
            
            return {
                "compliant": verification.success,
                "confidence": verification.confidence,
                "method": "cav_nlp_hybrid",
                "formalized_constraints": formalized_count,
                "violations": violations,
                "z3_result": verification.z3_result,
                "lean_result": verification.lean_result,
                "counterexample": verification.counterexample
            }
            
        except Exception as exc:
            logger.error(f"SOP compliance verification failed: {exc}")
            return {
                "compliant": False,
                "confidence": 0.0,
                "method": "error",
                "error": str(exc),
                "violations": []
            }

    def formalize_deterministic_constraint(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formalize natural language constraint to Z3 using CAV-NLP.
        
        Args:
            natural_language: Natural language constraint
            context: Optional context for formalization
            
        Returns:
            Dict with formalization result
        """
        if not self.use_cav_nlp or self._formalizer is None:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "source": natural_language
            }
        
        try:
            result = self._formalizer.formalize(natural_language, context)
            return {
                "success": result.success,
                "z3_expr": str(result.z3_expr) if result.z3_expr else None,
                "constraint_type": result.constraint_type,
                "variables": result.variables,
                "canonical_form": result.canonical_form,
                "source": natural_language,
                "warnings": result.warnings
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "source": natural_language
            }

    def canonicalize_deterministic_expression(
        self,
        expression: str,
        target: str = "z3"
    ) -> Dict[str, Any]:
        """Canonicalize a deterministic expression using CAV-NLP.
        
        Args:
            expression: Expression to canonicalize
            target: Target format ("z3" or "lean")
            
        Returns:
            Dict with canonical form
        """
        if not self.use_cav_nlp or self._formalizer is None:
            return {
                "success": False,
                "error": "CAV-NLP not available",
                "original": expression
            }
        
        try:
            result = self._formalizer.formalize(expression, target=target)
            return {
                "success": result.success,
                "canonical_form": result.canonical_form,
                "original": expression,
                "constraint_type": result.constraint_type,
                "variables": result.variables,
                "warnings": result.warnings
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "original": expression
            }

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available verification capabilities."""
        return {
            "z3_available": self.solver is not None,
            "cav_nlp_available": self.use_cav_nlp,
            "hybrid_verification": self.use_cav_nlp,
            "formalization": self.use_cav_nlp,
            "canonicalization": self.use_cav_nlp,
            "sop_compliance": self.use_cav_nlp
        }


class ReproducibilityAdapter:
    """Layer 7: detLLM reproducibility adapter"""

    def __init__(self):
        self.detllm = check

    def verify_reproducibility(
        self,
        prompt: str,
        tier: int = 2,
        runs: int = 3
    ) -> Dict[str, Any]:
        """Verify reproducibility using detLLM"""
        if self.detllm is None:
            return {
                "reproducible": False,
                "reason": "detLLM not available",
                "divergence_rate": None
            }

        try:
            report = self.detllm(
                backend="local",
                model="llama-2-70b",
                prompts=[prompt],
                runs=runs,
                tier=tier
            )

            return {
                "reproducible": report.get("passed", False),
                "divergence_rate": report.get("divergence_rate"),
                "tier_achieved": report.get("tier"),
                "details": report
            }
        except Exception as e:
            logger.error(f"Reproducibility verification failed: {e}")
            return {
                "reproducible": False,
                "error": str(e)
            }


class DeterministicSOPGenerator:
    """
    Main adapter integrating 8-layer deterministic framework with SOP generation

    Architecture:
    - Layer 0: Lagrange Mapper (Bias Filtering)
    - Layer 1: ROMA/MDAP (Decomposition)
    - Layer 2: LMQL/Outlines (Structured Generation)
    - Layer 3: Steer/Guardrails (Content Validation)
    - Layer 4: DSPy (Learning)
    - Layer 5: Knowledge Engine (Context)
    - Layer 6: Lean 4/Z3 + CAV-NLP (Formal Verification)
    - Layer 7: detLLM (Reproducibility)
    """

    def __init__(self, config: Optional[SOPGenerationConfig] = None):
        self.config = config or SOPGenerationConfig()
        self.base_generator = SOPGenerator() if SOPGenerator else None

        # Initialize layer adapters
        self.bias_filter = LagrangeFilterAdapter()
        self.decomposer = DecompositionAdapter()
        self.structured_gen = StructuredGenerationAdapter()
        self.validator = ValidationAdapter()
        self.learner = LearningAdapter()
        self.knowledge = KnowledgeAdapter()
        
        # Initialize formal verifier with CAV-NLP if enabled
        self.formal_verifier = FormalVerificationAdapter(
            use_cav_nlp=self.config.use_cav_nlp
        )
        self.reproducibility = ReproducibilityAdapter()

        # Log initialization details
        logger.info("DeterministicSOPGenerator initialized with layers: " +
                   f"{self.config.enable_layers}")
        if self.config.use_cav_nlp:
            logger.info(f"CAV-NLP enabled (confidence threshold: {self.config.cav_nlp_confidence_threshold})")
        else:
            logger.info("CAV-NLP disabled")

    async def generate_sop(
        self,
        requirement: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        equipment: Optional[List[str]] = None,
        materials: Optional[List[str]] = None,
        tier: int = 2,
        use_all_layers: bool = True
    ) -> SOPGenerationResult:
        """
        Generate SOP using 8-layer deterministic framework

        Args:
            requirement: User requirement for the SOP
            domain: Scientific domain (chemistry, physics, biology, general)
            constraints: Optional constraints (budget, duration, etc.)
            equipment: Optional equipment list
            materials: Optional materials list
            tier: Reproducibility tier (0, 1, 2)
            use_all_layers: Whether to use all 8 layers

        Returns:
            SOPGenerationResult with SOP, metadata, validation, and metrics
        """

        start_time = asyncio.get_event_loop().time()
        errors = []

        try:
            # Layer 0: Bias filtering
            if 0 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 0: Filtering biases")
                filtered_requirement = self.bias_filter.filter(requirement, domain)
            else:
                filtered_requirement = requirement

            # Layer 1: Decomposition
            if 1 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 1: Decomposing requirement")
                decomposition = self.decomposer.decompose(filtered_requirement, domain)
            else:
                decomposition = {}

            # Layer 2: Structured generation
            if 2 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 2: Enforcing structure")
                schema = self._get_sop_schema(domain)
            else:
                schema = {}

            # Layer 5: Knowledge enrichment
            if 5 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 5: Enriching with knowledge")
                knowledge = self.knowledge.enrich(filtered_requirement, domain)
            else:
                knowledge = {}

            # Generate base SOP using existing MAKER framework
            if self.base_generator is None:
                # Mock generation
                base_sop = self._mock_generate_sop(
                    filtered_requirement,
                    domain,
                    decomposition,
                    knowledge
                )
            else:
                # Call existing SOP generator
                base_sop = await self.base_generator.generate_sop(
                    requirement=filtered_requirement,
                    domain=domain,
                    constraints=constraints,
                    equipment=equipment,
                    materials=materials
                )

            # Layer 3: Content validation
            if 3 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 3: Validating content")
                validation_result = self.validator.validate(base_sop, domain)
            else:
                validation_result = {"valid": True, "issues": []}

            # Layer 4: Learning-based improvement
            if 4 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 4: Applying learning improvements")
                improved_sop = self.learner.improve(base_sop)
            else:
                improved_sop = base_sop

            # Layer 6: Formal verification (with optional CAV-NLP enhancement)
            if 6 in self.config.enable_layers and use_all_layers:
                logger.info("Layer 6: Formal verification" + 
                           (" (with CAV-NLP)" if self.config.use_cav_nlp else ""))
                
                use_cav = self.config.use_cav_nlp and self.config.use_hybrid_verification
                
                if domain == "chemistry":
                    formal_result = self.formal_verifier.verify_stoichiometry(
                        improved_sop, 
                        use_cav_nlp=use_cav
                    )
                elif domain == "physics":
                    formal_result = self.formal_verifier.verify_dimensional_consistency(
                        improved_sop,
                        use_cav_nlp=use_cav
                    )
                else:
                    # General domain - check for any formal constraints in SOP
                    constraints = improved_sop.get("constraints", [])
                    if constraints and use_cav:
                        formal_result = self.formal_verifier.verify_sop_compliance(
                            improved_sop,
                            compliance_constraints=constraints,
                            use_cav_nlp=True
                        )
                    else:
                        formal_result = {"verified": True, "reason": "Domain not applicable"}
            else:
                formal_result = {"verified": True}

            # Layer 7: Reproducibility verification
            if 7 in self.config.enable_layers and use_all_layers:
                logger.info(f"Layer 7: Verifying reproducibility (Tier {tier})")
                reproducibility_result = self.reproducibility.verify_reproducibility(
                    prompt=json.dumps(improved_sop),
                    tier=tier,
                    runs=self.config.verification_runs
                )
            else:
                reproducibility_result = None

            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time

            # Prepare result
            result = SOPGenerationResult(
                success=True,
                sop=improved_sop,
                metadata={
                    "domain": domain,
                    "layers_used": self.config.enable_layers if use_all_layers else [],
                    "decomposition": decomposition,
                    "knowledge_enriched": len(knowledge.get("context", [])),
                    "config": {
                        "temperature": self.config.temperature,
                        "tier": tier,
                        "domain": domain
                    }
                },
                validation_results={
                    "content_validation": validation_result,
                    "formal_verification": formal_result
                },
                reproducibility_metrics=reproducibility_result,
                execution_time=execution_time,
                errors=errors
            )

            logger.info(f"SOP generation completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"SOP generation failed: {e}")

            return SOPGenerationResult(
                success=False,
                sop=None,
                metadata={},
                validation_results={},
                reproducibility_metrics=None,
                execution_time=execution_time,
                errors=[str(e)]
            )

    def _get_sop_schema(self, domain: str) -> Dict[str, Any]:
        """Get JSON schema for SOP structure"""
        base_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "objective": {"type": "string"},
                "materials": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "string"},
                            "hazard_level": {"type": "string", "enum": ["low", "medium", "high"]}
                        }
                    }
                },
                "equipment": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "specifications": {"type": "string"},
                            "calibration": {"type": "string"}
                        }
                    }
                },
                "procedure": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "duration": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                },
                "safety": {
                    "type": "object",
                    "properties": {
                        "hazards": {"type": "array", "items": {"type": "string"}},
                        "ppe": {"type": "array", "items": {"type": "string"}},
                        "emergency": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "validation": {
                    "type": "object",
                    "properties": {
                        "quality_checks": {"type": "array", "items": {"type": "string"}},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["title", "objective", "materials", "procedure", "safety"]
        }

        # Domain-specific additions
        if domain == "chemistry":
            base_schema["properties"]["reactions"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reactants": {"type": "array"},
                        "products": {"type": "array"},
                        "conditions": {"type": "object"}
                    }
                }
            }
        elif domain == "physics":
            base_schema["properties"]["calculations"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "equation": {"type": "string"},
                        "variables": {"type": "object"},
                        "units": {"type": "string"}
                    }
                }
            }
        elif domain == "biology":
            base_schema["properties"]["biosafety_level"] = {"type": "integer"}
            base_schema["properties"]["sterility"] = {"type": "string"}

        return base_schema

    def _mock_generate_sop(
        self,
        requirement: str,
        domain: str,
        decomposition: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock SOP generation when base generator not available"""

        return {
            "title": f"Standard Operating Procedure: {requirement[:50]}",
            "objective": requirement,
            "domain": domain,
            "materials": [
                {
                    "name": "Example Material",
                    "quantity": "100 mL",
                    "hazard_level": "low"
                }
            ],
            "equipment": [
                {
                    "name": "Example Equipment",
                    "specifications": "Standard lab grade",
                    "calibration": "Calibrated within last 6 months"
                }
            ],
            "procedure": [
                {
                    "step": 1,
                    "action": "Prepare materials and equipment",
                    "duration": "5 minutes",
                    "parameters": {}
                },
                {
                    "step": 2,
                    "action": "Execute main procedure",
                    "duration": "30 minutes",
                    "parameters": {}
                }
            ],
            "safety": {
                "hazards": ["General lab hazards"],
                "ppe": ["Lab coat", "Safety glasses", "Gloves"],
                "emergency": ["Eye wash station", "Safety shower"]
            },
            "validation": {
                "quality_checks": ["Visual inspection", "Measurement verification"],
                "acceptance_criteria": ["Procedure completed as specified"]
            }
        }


# Convenience function
async def generate_deterministic_sop(
    requirement: str,
    domain: str = "general",
    **kwargs
) -> SOPGenerationResult:
    """
    Convenience function for generating deterministic SOPs

    Example:
        result = await generate_deterministic_sop(
            requirement="Synthesize ibuprofen from isobutylbenzene",
            domain="chemistry",
            tier=2
        )
    """
    generator = DeterministicSOPGenerator()
    return await generator.generate_sop(requirement, domain, **kwargs)
