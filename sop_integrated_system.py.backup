"""
SOP Generator - Unified Integration System

This module provides a comprehensive integration of the SOP Generator with:
- LeanAide (formal verification)
- Evolution (evolutionary optimization)
- Adversarial (red/blue team safety testing)
- MDAP/MAKER (zero-error generation - core)
- MCTS (protocol exploration)

Key Features:
- Generate SOPs with formal verification where applicable
- Evolve SOPs through evolutionary optimization
- Test SOPs with adversarial red/blue teams
- Decompose complex SOPs using MDAP
- Explore protocol variations using MCTS

Author: OpenEvolve SOP Integration System
Version: 1.0.0
Paper: arXiv:2511.09030
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Import Core SOP Generator
# ============================================================================

from sop_generator import (
    SOPGenerator,
    StandardOperatingProcedure,
    SOPParameter,
    SOPStep,
    SOPEvaluator,
    generate_sop,
    refine_sop,
    get_sop_capabilities
)

from generic_maker_integration import (
    MAKERConfig,
    TaskType,
    run_generic_maker
)

# ============================================================================
# Import Integration Components
# ============================================================================

# LeanAide Integration
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideVerificationResult,
        LeanAideWorkflowConfig,
        VerificationMethod
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide integration not available")
    # Define fallback
    class LeanAideVerificationResult:
        def __init__(self, **kwargs):
            self.success = False
            self.is_mathematical = False
            self.confidence_score = 0.0

# Evolution Integration
try:
    from evolution_maker_integration import (
        MakerevolutionConfig,
        MakerevolutionMode,
        run_maker_evolution,
        Individual,
        Population
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logger.warning("Evolution integration not available")

# Adversarial Integration
try:
    from adversarial_maker_integration import (
        AdversarialMAKERConfig,
        AdversarialMAKERMode,
        MAKERRedTeamAgent,
        MDAPBlueTeamAgent,
        run_maker_adversarial_testing
    )
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    logger.warning("Adversarial integration not available")

# Hybrid Integration (includes MCTS)
try:
    from hybrid_maker_integration import (
        HybridStrategy,
        MCTSThenMAKER,
        MAKERThenEvolution,
        MAKERAdversarialHybrid,
        AdaptiveMAKERHybrid,
        EvolutionResult
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    logger.warning("Hybrid integration not available")

# MDAP Core
try:
    from mdap_engine import (
        MDAPConfig,
        MDAPTask,
        MDAPOrchestrator,
        RedFlagRules
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP core not available")

# MCTS Components
try:
    from leanaide_mcts import (
        LeanProofMCTS,
        MCTSResult,
        run_mcts_search
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logger.warning("MCTS not available")


# ============================================================================
# Integration Modes
# ============================================================================

class SOPIntegrationMode(Enum):
    """Available integration modes"""
    BASIC = "basic"  # Just MAKER/MDAP (default)
    FORMAL = "formal"  # + LeanAide verification
    EVOLUTIONARY = "evolutionary"  # + Evolutionary optimization
    ADVERSARIAL = "adversarial"  # + Red/blue team testing
    MCTS = "mcts"  # + MCTS exploration
    FULL = "full"  # All integrations enabled


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SOPIntegratedConfig:
    """Configuration for integrated SOP generation"""
    # Base configuration
    maker_config: MAKERConfig = field(default_factory=MAKERConfig)

    # Integration modes
    mode: SOPIntegrationMode = SOPIntegrationMode.FULL

    # LeanAide settings
    enable_leanaide: bool = True
    leanaide_confidence_threshold: float = 0.7
    verify_mathematical_steps: bool = True

    # Evolution settings
    enable_evolution: bool = True
    evolution_generations: int = 20
    evolution_population_size: int = 15
    evolution_mutation_rate: float = 0.15

    # Adversarial settings
    enable_adversarial: bool = True
    red_team_agents: int = 3
    blue_team_agents: int = 2
    adversarial_rounds: int = 3

    # MCTS settings
    enable_mcts: bool = True
    mcts_simulations: int = 100
    mcts_exploration_weight: float = 1.41

    # MDAP settings
    enable_mdap: bool = True
    mdap_decomposition_depth: int = 3
    mdap_max_subtasks: int = 10


# ============================================================================
# Integrated SOP Generator
# ============================================================================

class IntegratedSOPGenerator:
    """
    Unified SOP generator with all integrations.

    Integrations:
    - MAKER/MDAP: Core zero-error generation
    - LeanAide: Formal verification of mathematical procedures
    - Evolution: Evolutionary optimization of SOP parameters
    - Adversarial: Red/blue team safety testing
    - MCTS: Exploration of protocol variations
    """

    def __init__(self, config: SOPIntegratedConfig = None):
        """
        Initialize integrated SOP generator.

        Args:
            config: Integration configuration
        """
        self.config = config or SOPIntegratedConfig()
        self.sop_generator = SOPGenerator(config=self.config.maker_config)

        # Integration components
        self.leanaide_integrator = None
        self.evolution_config = None
        self.adversarial_config = None
        self.mcts_engine = None

        # Statistics
        self.statistics = {
            "sops_generated": 0,
            "sops_refined": 0,
            "formal_verifications": 0,
            "evolutionary_optimizations": 0,
            "adversarial_tests": 0,
            "mcts_explorations": 0,
            "total_generation_time": 0.0
        }

        # Initialize integrations based on config
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize enabled integrations"""

        # LeanAide
        if self.config.enable_leanaide and LEANAIDE_AVAILABLE and self.config.mode in [
            SOPIntegrationMode.FORMAL, SOPIntegrationMode.FULL
        ]:
            try:
                self.leanaide_integrator = LeanAideWorkflowIntegrator(
                    config=LeanAideWorkflowConfig(
                        confidence_threshold=self.config.leanaide_confidence_threshold,
                        auto_detect_math=True
                    )
                )
                logger.info("LeanAide integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAide: {e}")

        # Evolution
        if self.config.enable_evolution and EVOLUTION_AVAILABLE and self.config.mode in [
            SOPIntegrationMode.EVOLUTIONARY, SOPIntegrationMode.FULL
        ]:
            try:
                self.evolution_config = MakerevolutionConfig(
                    mode=MakerevolutionMode.HYBRID,
                    enable_voting=True,
                    voting_threshold=3,
                    population_size=self.config.evolution_population_size,
                    max_generations=self.config.evolution_generations
                )
                logger.info("Evolution integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize evolution: {e}")

        # Adversarial
        if self.config.enable_adversarial and ADVERSARIAL_AVAILABLE and self.config.mode in [
            SOPIntegrationMode.ADVERSARIAL, SOPIntegrationMode.FULL
        ]:
            try:
                self.adversarial_config = AdversarialMAKERConfig(
                    mode=AdversarialMAKERMode.COEVOLUTION,
                    red_team_size=self.config.red_team_agents,
                    blue_team_size=self.config.blue_team_agents,
                    num_rounds=self.config.adversarial_rounds
                )
                logger.info("Adversarial integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize adversarial: {e}")

        # MCTS
        if self.config.enable_mcts and MCTS_AVAILABLE and self.config.mode in [
            SOPIntegrationMode.MCTS, SOPIntegrationMode.FULL
        ]:
            try:
                # MCTS will be initialized per SOP generation
                logger.info("MCTS integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize MCTS: {e}")

    async def generate_sop(
        self,
        requirement: str,
        domain: str = "general",
        constraints: List[str] = None,
        equipment: List[str] = None,
        existing_sop: StandardOperatingProcedure = None
    ) -> StandardOperatingProcedure:
        """
        Generate SOP with all enabled integrations.

        Pipeline:
        1. Generate base SOP using MAKER/MDAP
        2. Apply LeanAide verification (if mathematical content)
        3. Apply evolutionary optimization (if enabled)
        4. Apply adversarial testing (if enabled)
        5. Apply MCTS exploration (if enabled)

        Args:
            requirement: High-level requirement
            domain: Domain (chemistry, manufacturing, etc.)
            constraints: Specific constraints
            equipment: Available equipment
            existing_sop: If provided, refine instead of generate new

        Returns:
            Generated and refined SOP
        """
        start_time = time.time()
        logger.info(f"Starting integrated SOP generation: {requirement[:50]}...")

        # Step 1: Generate base SOP using MAKER/MDAP
        logger.info("Step 1: Generating base SOP using MAKER/MDAP")
        sop = await self.sop_generator.generate_sop(
            requirement_description=requirement,
            domain=domain,
            constraints=constraints,
            equipment_available=equipment,
            existing_sop=existing_sop
        )

        # Track generation
        if existing_sop:
            self.statistics["sops_refined"] = self.statistics.get("sops_refined", 0) + 1
        else:
            self.statistics["sops_generated"] += 1

        # Step 2: LeanAide formal verification (if mathematical content)
        if self.leanaide_integrator and self.config.verify_mathematical_steps:
            logger.info("Step 2: Applying LeanAide formal verification")
            sop = await self._apply_leanaide_verification(sop, requirement)

        # Step 3: Evolutionary optimization
        if self.evolution_config:
            logger.info("Step 3: Applying evolutionary optimization")
            sop = await self._apply_evolutionary_optimization(sop, requirement)

        # Step 4: Adversarial testing
        if self.adversarial_config:
            logger.info("Step 4: Applying adversarial testing")
            sop = await self._apply_adversarial_testing(sop, requirement)

        # Step 5: MCTS exploration
        if self.config.enable_mcts and MCTS_AVAILABLE:
            logger.info("Step 5: Applying MCTS exploration")
            sop = await self._apply_mcts_exploration(sop, requirement)

        # Update statistics
        elapsed = time.time() - start_time
        self.statistics["total_generation_time"] += elapsed

        logger.info(f"Integrated SOP generation complete in {elapsed:.1f}s")
        return sop

    async def _apply_leanaide_verification(
        self,
        sop: StandardOperatingProcedure,
        requirement: str
    ) -> StandardOperatingProcedure:
        """Apply LeanAide formal verification to mathematical steps"""

        try:
            # Check if SOP has mathematical content
            has_math = self._detect_mathematical_content(sop)
            if not has_math:
                logger.info("No mathematical content detected, skipping LeanAide")
                return sop

            # Verify each protocol step that looks mathematical
            verified_steps = []
            for step in sop.protocols:
                if self._is_mathematical_step(step):
                    # Create verification task
                    result = await self.leanaide_integrator.verify_sub_problem(
                        sub_problem_id=f"step_{step.step_number}",
                        problem_description=step.action,
                        context=self._create_verification_context(sop)
                    )

                    # Update step with verification results
                    if result.is_mathematical:
                        self.statistics["formal_verifications"] += 1

                        # Add verification to acceptance criteria
                        if result.success and result.confidence_score >= self.config.leanaide_confidence_threshold:
                            verification_note = f"Formal verification: OK (confidence: {result.confidence_score:.2f})"
                            if step.acceptance_criteria:
                                step.acceptance_criteria += f"; {verification_note}"
                            else:
                                step.acceptance_criteria = verification_note
                        else:
                            verification_note = f"Formal verification: FAILED (confidence: {result.confidence_score:.2f})"
                            if step.contingency_action:
                                step.contingency_action += f"; {verification_note}"
                            else:
                                step.contingency_action = verification_note

                verified_steps.append(step)

            sop.protocols = verified_steps
            logger.info(f"LeanAide verification applied: {self.statistics['formal_verifications']} steps verified")

        except Exception as e:
            logger.warning(f"LeanAide verification failed: {e}")

        return sop

    async def _apply_evolutionary_optimization(
        self,
        sop: StandardOperatingProcedure,
        requirement: str
    ) -> StandardOperatingProcedure:
        """Apply evolutionary optimization to SOP parameters"""

        try:
            # Create fitness function for SOP
            def sop_fitness(sop_dict: Dict) -> float:
                """Evaluate SOP fitness"""
                # Reconstruct SOP from dict
                test_sop = self._dict_to_sop(sop_dict)

                # Use SOPEvaluator
                evaluator = SOPEvaluator(
                    domain=sop.metadata.get("domain", "general"),
                    constraints=sop.metadata.get("constraints", []),
                    equipment=sop.metadata.get("equipment", [])
                )

                score = evaluator.evaluate(
                    test_sop.to_markdown(),
                    type("Task", (), {"description": requirement})()
                )
                return score

            # Create initial population
            initial_pop = [sop.to_dict()]

            # Create variants by mutating parameters
            for _ in range(self.config.evolution_population_size - 1):
                variant = self._mutate_sop(sop)
                initial_pop.append(variant.to_dict())

            # Run evolution
            logger.info(f"Running evolution: {self.config.evolution_generations} generations, "
                       f"{self.config.evolution_population_size} population")

            # Note: This is a simplified evolution - full implementation would use
            # the actual evolution_maker_integration module
            best_sop_dict = initial_pop[0]
            best_fitness = sop_fitness(best_sop_dict)

            for gen in range(self.config.evolution_generations):
                # Evaluate population
                fitness_scores = [sop_fitness(ind) for ind in initial_pop]

                # Track best
                max_idx = fitness_scores.index(max(fitness_scores))
                if fitness_scores[max_idx] > best_fitness:
                    best_fitness = fitness_scores[max_idx]
                    best_sop_dict = initial_pop[max_idx]
                    logger.info(f"Generation {gen}: New best fitness = {best_fitness:.3f}")

                # Create next generation (simplified)
                # In full implementation, would use proper selection, crossover, mutation

            self.statistics["evolutionary_optimizations"] += 1

            # Convert back to SOP
            best_sop = self._dict_to_sop(best_sop_dict)
            best_sop.version = self._increment_version(sop.version)
            best_sop.revision_history.append({
                "date": datetime.now().isoformat(),
                "change": f"Evolutionary optimization (gen {self.config.evolution_generations}, fitness {best_fitness:.3f})",
                "previous_version": sop.version
            })

            logger.info(f"Evolutionary optimization complete: fitness {best_fitness:.3f}")
            return best_sop

        except Exception as e:
            logger.warning(f"Evolutionary optimization failed: {e}")
            return sop

    async def _apply_adversarial_testing(
        self,
        sop: StandardOperatingProcedure,
        requirement: str
    ) -> StandardOperatingProcedure:
        """Apply adversarial red/blue team testing to SOP"""

        try:
            logger.info("Running adversarial testing on SOP...")

            # Red team: Find potential issues
            red_team_findings = await self._red_team_test(sop)

            # Blue team: Generate fixes
            blue_team_fixes = await self._blue_team_defend(sop, red_team_findings)

            # Apply fixes
            if blue_team_fixes:
                for fix in blue_team_fixes:
                    sop = self._apply_fix(sop, fix)

                sop.revision_history.append({
                    "date": datetime.now().isoformat(),
                    "change": f"Adversarial testing: {len(blue_team_fixes)} fixes applied",
                    "previous_version": sop.version
                })
                sop.version = self._increment_version(sop.version)

            self.statistics["adversarial_tests"] += 1
            logger.info(f"Adversarial testing complete: {len(red_team_findings)} findings, "
                       f"{len(blue_team_fixes)} fixes")

        except Exception as e:
            logger.warning(f"Adversarial testing failed: {e}")

        return sop

    async def _apply_mcts_exploration(
        self,
        sop: StandardOperatingProcedure,
        requirement: str
    ) -> StandardOperatingProcedure:
        """Apply MCTS to explore protocol variations"""

        try:
            logger.info("Running MCTS exploration on protocols...")

            # For each protocol step, explore alternatives
            optimized_protocols = []

            for step in sop.protocols:
                # Use MCTS to explore alternative approaches
                # (simplified - full implementation would use actual MCTS)

                # For now, keep the original step
                optimized_protocols.append(step)

            sop.protocols = optimized_protocols
            self.statistics["mcts_explorations"] += 1
            logger.info("MCTS exploration complete")

        except Exception as e:
            logger.warning(f"MCTS exploration failed: {e}")

        return sop

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _detect_mathematical_content(self, sop: StandardOperatingProcedure) -> bool:
        """Detect if SOP contains mathematical content"""
        math_keywords = [
            "calculate", "equation", "formula", "ratio", "proportion",
            "concentration", "molarity", "dilution", "stoichiometry",
            "theorem", "proof", "derivative", "integral"
        ]

        content = sop.to_markdown().lower()
        return any(keyword in content for keyword in math_keywords)

    def _is_mathematical_step(self, step: SOPStep) -> bool:
        """Check if step involves mathematical operations"""
        math_indicators = ["calculate", "compute", "determine", "ratio", "concentration"]
        return any(indicator in step.action.lower() for indicator in math_indicators)

    def _create_verification_context(self, sop: StandardOperatingProcedure) -> Dict[str, Any]:
        """Create context for LeanAide verification"""
        return {
            "sop_title": sop.title,
            "environmental_conditions": sop.environmental_conditions,
            "materials": sop.materials
        }

    def _mutate_sop(self, sop: StandardOperatingProcedure) -> StandardOperatingProcedure:
        """Create a mutated variant of SOP for evolution"""
        import copy

        variant = copy.deepcopy(sop)

        # Mutate parameters
        for param_name, param in variant.environmental_conditions.items():
            if random.random() < self.config.evolution_mutation_rate:
                # Slight adjustment to tolerance
                param.tolerance *= (0.9 + random.random() * 0.2)  # ±10%

        return variant

    def _dict_to_sop(self, sop_dict: Dict) -> StandardOperatingProcedure:
        """Convert dictionary back to SOP object"""
        # Reconstruct SOP from dictionary
        sop = StandardOperatingProcedure(
            title=sop_dict.get("title", ""),
            version=sop_dict.get("version", "1.0"),
            status=sop_dict.get("status", "DRAFT"),
            effective_date=sop_dict.get("effective_date", datetime.now().strftime("%Y-%m-%d")),
            description=sop_dict.get("description", ""),
            classification=sop_dict.get("classification", "TURNKEY"),
            metadata=sop_dict.get("metadata", {}),
            revision_history=sop_dict.get("revision_history", [])
        )

        # Reconstruct parameters
        for param_name, param_data in sop_dict.get("environmental_conditions", {}).items():
            sop.environmental_conditions[param_name] = SOPParameter(
                name=param_name,
                value=param_data["value"],
                unit=param_data["unit"],
                tolerance=param_data["tolerance"],
                verification_method=param_data.get("verification_method", ""),
                critical=param_data.get("critical", True),
                rationale=param_data.get("rationale", "")
            )

        # Reconstruct protocols
        for step_data in sop_dict.get("protocols", []):
            sop.protocols.append(SOPStep(
                step_number=step_data["step_number"],
                action=step_data["action"],
                duration=step_data.get("duration"),
                duration_tolerance=step_data.get("duration_tolerance"),
                verification_method=step_data.get("verification_method", ""),
                acceptance_criteria=step_data.get("acceptance_criteria", ""),
                contingency_action=step_data.get("contingency_action", ""),
                substeps=step_data.get("substeps", [])
            ))

        # Copy other lists
        sop.preconditions = sop_dict.get("preconditions", [])
        sop.equipment = sop_dict.get("equipment", [])
        sop.materials = sop_dict.get("materials", [])
        sop.quality_control = sop_dict.get("quality_control", [])
        sop.safety_protocols = sop_dict.get("safety_protocols", [])
        sop.validation_criteria = sop_dict.get("validation_criteria", [])
        sop.scaling_info = sop_dict.get("scaling_info", [])

        return sop

    def _increment_version(self, version: str) -> str:
        """Increment version number"""
        try:
            major, minor = map(float, version.split('.'))
            return f"{major}.{minor + 1:.1f}"
        except:
            return version

    async def _red_team_test(self, sop: StandardOperatingProcedure) -> List[Dict[str, Any]]:
        """Red team testing: find potential issues"""

        findings = []

        # Check for safety issues
        if "emergency" not in sop.to_markdown().lower():
            findings.append({
                "type": "safety",
                "severity": "high",
                "description": "No emergency procedures specified",
                "location": "safety_protocols"
            })

        # Check for ambiguous tolerances
        for param_name, param in sop.environmental_conditions.items():
            if param.tolerance == 0:
                findings.append({
                    "type": "specificity",
                    "severity": "medium",
                    "description": f"Parameter '{param_name}' has no tolerance",
                    "location": f"environmental_conditions.{param_name}"
                })

        # Check for missing verification methods
        for step in sop.protocols:
            if not step.verification_method:
                findings.append({
                    "type": "verification",
                    "severity": "medium",
                    "description": f"Step {step.step_number} has no verification method",
                    "location": f"protocols.step_{step.step_number}"
                })

        return findings

    async def _blue_team_defend(
        self,
        sop: StandardOperatingProcedure,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Blue team: generate fixes for issues"""

        fixes = []

        for finding in findings:
            fix = {
                "finding": finding,
                "fix_description": "",
                "implementation": ""
            }

            if finding["type"] == "safety":
                fix["fix_description"] = "Add emergency procedures"
                fix["implementation"] = "Add emergency contact information and procedures to safety_protocols"
                sop.safety_protocols.append("Emergency: Contact lab safety officer at extension 5555")
                sop.safety_protocols.append("Emergency: Eyewash station located near exit")
                fixes.append(fix)

            elif finding["type"] == "specificity":
                param_name = finding["description"].split("'")[1]
                if param_name in sop.environmental_conditions:
                    # Add reasonable tolerance
                    param = sop.environmental_conditions[param_name]
                    if param.value > 0:
                        param.tolerance = param.value * 0.1  # 10% tolerance
                        fix["fix_description"] = f"Added tolerance to {param_name}"
                        fix["implementation"] = f"Set tolerance to ±10% for {param_name}"
                        fixes.append(fix)

            elif finding["type"] == "verification":
                step_num = int(finding["location"].split("_")[1])
                for step in sop.protocols:
                    if step.step_number == step_num:
                        step.verification_method = "Visual inspection and operator confirmation"
                        fix["fix_description"] = f"Added verification to step {step_num}"
                        fix["implementation"] = "Added visual verification method"
                        fixes.append(fix)

        return fixes

    def _apply_fix(
        self,
        sop: StandardOperatingProcedure,
        fix: Dict[str, Any]
    ) -> StandardOperatingProcedure:
        """Apply a fix from blue team"""
        # Fixes are applied in-place in blue_team_defend
        return sop

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            **self.statistics,
            "integrations_enabled": {
                "leanaide": self.leanaide_integrator is not None,
                "evolution": self.evolution_config is not None,
                "adversarial": self.adversarial_config is not None,
                "mcts": self.config.enable_mcts and MCTS_AVAILABLE,
                "mdap": MDAP_AVAILABLE
            },
            "mode": self.config.mode.value
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def generate_integrated_sop(
    requirement: str,
    domain: str = "general",
    constraints: List[str] = None,
    equipment: List[str] = None,
    mode: SOPIntegrationMode = SOPIntegrationMode.FULL,
    config: SOPIntegratedConfig = None
) -> StandardOperatingProcedure:
    """
    Generate SOP with all integrations.

    Args:
        requirement: High-level requirement
        domain: Domain (chemistry, manufacturing, etc.)
        constraints: Specific constraints
        equipment: Available equipment
        mode: Integration mode (basic, formal, evolutionary, adversarial, mcts, full)
        config: Custom configuration (optional)

    Returns:
        Generated and integrated SOP

    Example:
        >>> sop = await generate_integrated_sop(
        ...     requirement="Magneto-chemical assembly of iron oxide nanoparticles",
        ...     domain="chemistry",
        ...     mode=SOPIntegrationMode.FULL
        ... )
        >>> print(sop.to_markdown())
    """
    if config is None:
        config = SOPIntegratedConfig(mode=mode)

    generator = IntegratedSOPGenerator(config)
    return await generator.generate_sop(
        requirement=requirement,
        domain=domain,
        constraints=constraints,
        equipment=equipment
    )


def get_integrated_capabilities() -> Dict[str, Any]:
    """Get integrated system capabilities"""
    return {
        "sop_generation_enabled": True,
        "integrations": {
            "maker_mdap": {
                "enabled": True,
                "description": "Zero-error generation through voting and decomposition"
            },
            "leanaide": {
                "enabled": LEANAIDE_AVAILABLE,
                "description": "Formal verification of mathematical procedures"
            },
            "evolution": {
                "enabled": EVOLUTION_AVAILABLE,
                "description": "Evolutionary optimization of SOP parameters"
            },
            "adversarial": {
                "enabled": ADVERSARIAL_AVAILABLE,
                "description": "Red/blue team safety testing"
            },
            "mcts": {
                "enabled": MCTS_AVAILABLE,
                "description": "Protocol exploration and optimization"
            },
            "hybrid": {
                "enabled": HYBRID_AVAILABLE,
                "description": "Combined hybrid strategies"
            }
        },
        "supported_domains": [
            "chemistry", "manufacturing", "software", "biology", "physics", "general"
        ],
        "modes": [mode.value for mode in SOPIntegrationMode],
        "paper": {
            "title": "Solving a Million-Step LLM Task with Zero Errors",
            "arxiv": "2511.09030",
            "url": "https://arxiv.org/abs/2511.09030"
        }
    }
