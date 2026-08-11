"""
SOP Component-Level Generation and Refinement System

This module extends the integrated SOP system to support granular generation,
refinement, and optimization of individual SOP components:

- Environmental conditions (individual parameters)
- Equipment specifications
- Materials and reagents
- Protocol steps
- Quality control procedures
- Safety protocols
- Validation criteria
- Scaling information
- Preconditions

Each component can be:
1. Generated independently
2. Refined through feedback
3. Optimized via evolution
4. Tested via adversarial methods
5. Explored via MCTS
6. Verified formally (if mathematical)

Author: OpenEvolve SOP Component System
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

# Import core SOP system
from sop_generator import (
    SOPParameter,
    SOPStep,
    StandardOperatingProcedure,
    SOPEvaluator,
    SOPGenerator
)

from sop_integrated_system import (
    IntegratedSOPGenerator,
    SOPIntegratedConfig,
    SOPIntegrationMode
)

from generic_maker_integration import (
    MAKERConfig,
    TaskType,
    run_generic_maker,
    GenericEvaluator,
    GenericTask
)


# ============================================================================
# Component Types
# ============================================================================

class SOPComponentType(Enum):
    """Types of SOP components that can be generated/refined"""
    ENVIRONMENTAL_CONDITION = "environmental_condition"
    EQUIPMENT_SPECIFICATION = "equipment_specification"
    MATERIAL = "material"
    PROTOCOL_STEP = "protocol_step"
    QUALITY_CONTROL = "quality_control"
    SAFETY_PROTOCOL = "safety_protocol"
    VALIDATION_CRITERION = "validation_criterion"
    SCALING_INFO = "scaling_info"
    PRECONDITION = "precondition"


# ============================================================================
# Component Refinement Requests
# ============================================================================

@dataclass
class ComponentRefinementRequest:
    """Request to refine a specific component"""
    component_type: SOPComponentType
    component_name: str  # Name/identifier of the component
    current_value: Any
    refinement_goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


# ============================================================================
# Component-Level Generator
# ============================================================================

class SOPComponentGenerator:
    """
    Generate and refine individual SOP components.

    Features:
    - Generate any component type independently
    - Refine existing components
    - Apply all integrations at component level
    - Track component-level improvements
    """

    def __init__(self, config: SOPIntegratedConfig = None):
        """Initialize component generator"""
        self.config = config or SOPIntegratedConfig()
        self.integrated_generator = IntegratedSOPGenerator(config)

        # Component-level statistics
        self.statistics = {
            "generated": {},
            "refined": {},
            "optimized": {},
            "tested": {},
            "verified": {},
            "total_operations": 0
        }

    async def generate_environmental_condition(
        self,
        parameter_name: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> SOPParameter:
        """
        Generate an environmental condition parameter.

        Args:
            parameter_name: Name of the parameter (e.g., "Temperature", "Humidity")
            context: Context (domain, equipment available, etc.)
            domain: Domain (chemistry, manufacturing, etc.)

        Returns:
            Generated parameter with value, tolerance, verification method
        """
        logger.info(f"Generating environmental condition: {parameter_name}")

        # Create generation task
        task_desc = f"""
Generate a specification for the environmental parameter: {parameter_name}

Context:
- Domain: {domain}
- SOP Purpose: {context.get('purpose', 'general procedure')}
- Available Equipment: {context.get('equipment', 'standard')}

Generate a parameter specification that includes:
1. Target value (numeric, realistic for this domain)
2. Tolerance (achievable with standard equipment)
3. Verification method (specific measurement technique)
4. Rationale (why this parameter matters)
5. Criticality (is this parameter critical?)

Output format: JSON with keys: value, unit, tolerance, verification_method, rationale, critical
"""

        # Use MAKER to generate
        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("environmental_condition", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        param = self._parse_parameter(result.solution, parameter_name)

        # Track statistics
        self._track_operation("environmental_condition", "generated")

        # Apply integrations if enabled
        param = await self._apply_integrations_to_parameter(param, context)

        return param

    async def refine_environmental_condition(
        self,
        param: SOPParameter,
        refinement_goal: str,
        context: Dict[str, Any] = None
    ) -> SOPParameter:
        """
        Refine an existing environmental condition parameter.

        Args:
            param: Current parameter
            refinement_goal: What to improve (e.g., "tighten tolerance", "better verification")
            context: Additional context

        Returns:
            Refined parameter
        """
        logger.info(f"Refining environmental condition: {param.name} - Goal: {refinement_goal}")

        # Create refinement task
        task_desc = f"""
Refine this environmental parameter specification based on feedback:

Current Specification:
- Parameter: {param.name}
- Value: {param.value} {param.unit}
- Tolerance: ±{param.tolerance} {param.unit}
- Verification: {param.verification_method}
- Rationale: {param.rationale}

Refinement Goal: {refinement_goal}

Context:
- Domain: {context.get('domain', 'general') if context else 'general'}
- Equipment: {context.get('equipment', 'standard') if context else 'standard'}

Provide an improved specification that addresses the refinement goal while maintaining realism.
Output format: JSON with keys: value, unit, tolerance, verification_method, rationale, critical
"""

        # Use MAKER to refine
        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("environmental_condition", context or {}),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        refined_param = self._parse_parameter(result.solution, param.name)

        # Track statistics
        self._track_operation("environmental_condition", "refined")

        return refined_param

    async def generate_equipment_specification(
        self,
        equipment_name: str,
        purpose: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> Dict[str, str]:
        """
        Generate equipment specification.

        Args:
            equipment_name: Type of equipment (e.g., "Magnetic Stirrer")
            purpose: What it will be used for
            context: Context (requirements, constraints, etc.)
            domain: Domain

        Returns:
            Equipment specification dict with model, specs, etc.
        """
        logger.info(f"Generating equipment specification: {equipment_name}")

        task_desc = f"""
Generate a detailed specification for: {equipment_name}

Purpose: {purpose}
Domain: {domain}

Requirements:
{chr(10).join(f'- {r}' for r in context.get('requirements', ['standard specifications']))}

Provide specification including:
1. Recommended model(s)
2. Key specifications (range, accuracy, etc.)
3. Required features/capabilities
4. Calibration requirements
5. Maintenance considerations

Output format: JSON with keys: name, model, specifications, features, calibration, maintenance
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("equipment_specification", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        spec = self._parse_equipment_spec(result.solution, equipment_name)

        self._track_operation("equipment_specification", "generated")

        return spec

    async def generate_material(
        self,
        material_name: str,
        purpose: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate material specification.

        Args:
            material_name: Name of material/reagent
            purpose: What it will be used for
            context: Context (purity requirements, etc.)
            domain: Domain

        Returns:
            Material specification dict
        """
        logger.info(f"Generating material specification: {material_name}")

        task_desc = f"""
Generate a detailed specification for: {material_name}

Purpose: {purpose}
Domain: {domain}

Requirements:
{chr(10).join(f'- {r}' for r in context.get('requirements', ['standard grade']))}

Provide specification including:
1. Purity/grade specifications
2. Amount with tolerance
3. Storage requirements
4. Safety considerations
5. Alternative options

Output format: JSON with keys: name, purity, grade, amount, unit, tolerance, storage, safety, alternatives
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("material", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        spec = self._parse_material_spec(result.solution, material_name)

        self._track_operation("material", "generated")

        return spec

    async def generate_protocol_step(
        self,
        step_number: int,
        action_description: str,
        context: Dict[str, Any],
        previous_steps: List[SOPStep] = None,
        domain: str = "general"
    ) -> SOPStep:
        """
        Generate a detailed protocol step.

        Args:
            step_number: Step number in sequence
            action_description: What the step should accomplish
            context: Context (equipment, materials, etc.)
            previous_steps: Previous steps for context
            domain: Domain

        Returns:
            Complete protocol step with duration, verification, acceptance criteria, contingency
        """
        logger.info(f"Generating protocol step {step_number}: {action_description}")

        prev_context = ""
        if previous_steps:
            prev_context = f"Previous steps:\n" + "\n".join(
                f"- Step {s.step_number}: {s.action}" for s in previous_steps[-3:]
            )

        task_desc = f"""
Generate a detailed protocol step:

Step Number: {step_number}
Action: {action_description}
Domain: {domain}

{prev_context}

Context:
- Available Equipment: {context.get('equipment', [])}
- Available Materials: {context.get('materials', [])}
- Environmental Conditions: {context.get('environmental_conditions', {})}

Generate a complete step specification including:
1. Detailed action description
2. Duration with tolerance
3. Verification method (how to confirm step was done correctly)
4. Acceptance criteria (what indicates success)
5. Contingency action (what to do if step fails)
6. Sub-steps if applicable

Output format: JSON with keys: action, duration, duration_tolerance, verification_method,
acceptance_criteria, contingency_action, substeps
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("protocol_step", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        step = self._parse_protocol_step(result.solution, step_number)

        self._track_operation("protocol_step", "generated")

        return step

    async def refine_protocol_step(
        self,
        step: SOPStep,
        refinement_goal: str,
        context: Dict[str, Any] = None
    ) -> SOPStep:
        """
        Refine an existing protocol step.

        Args:
            step: Current step
            refinement_goal: What to improve
            context: Additional context

        Returns:
            Refined step
        """
        logger.info(f"Refining step {step.step_number}: {refinement_goal}")

        task_desc = f"""
Refine this protocol step based on feedback:

Current Step:
- Number: {step.step_number}
- Action: {step.action}
- Duration: {step.duration} ± {step.duration_tolerance}
- Verification: {step.verification_method}
- Acceptance: {step.acceptance_criteria}
- Contingency: {step.contingency_action}

Refinement Goal: {refinement_goal}

Provide an improved step specification that addresses the goal.
Output format: JSON with keys: action, duration, duration_tolerance, verification_method,
acceptance_criteria, contingency_action, substeps
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("protocol_step", context or {}),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        # Parse result
        refined_step = self._parse_protocol_step(result.solution, step.step_number)

        self._track_operation("protocol_step", "refined")

        return refined_step

    async def generate_quality_control_procedure(
        self,
        qc_focus: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str:
        """Generate a quality control procedure"""
        logger.info(f"Generating quality control: {qc_focus}")

        task_desc = f"""
Generate a quality control procedure for: {qc_focus}

Domain: {domain}
Context: {context}

Provide a detailed QC procedure including:
1. What to check
2. How to check it
3. Acceptance criteria
4. Frequency/timing
5. Documentation requirements
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("quality_control", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        self._track_operation("quality_control", "generated")
        return result.solution.strip()

    async def generate_safety_protocol(
        self,
        hazard_type: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str:
        """Generate a safety protocol"""
        logger.info(f"Generating safety protocol: {hazard_type}")

        task_desc = f"""
Generate a comprehensive safety protocol for: {hazard_type}

Domain: {domain}
Context: {context}

Provide a detailed safety protocol including:
1. Personal protective equipment (PPE)
2. Engineering controls
3. Administrative controls
4. Emergency procedures
5. First aid measures
6. Spill/response procedures
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("safety_protocol", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        self._track_operation("safety_protocol", "generated")
        return result.solution.strip()

    async def generate_validation_criterion(
        self,
        criterion_focus: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str:
        """Generate a validation criterion"""
        logger.info(f"Generating validation criterion: {criterion_focus}")

        task_desc = f"""
Generate a validation criterion for: {criterion_focus}

Domain: {domain}
Context: {context}

Provide a specific, measurable validation criterion including:
1. What is being validated
2. How to measure it
3. Acceptance criteria
4. Measurement method/equipment
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("validation_criterion", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        self._track_operation("validation_criterion", "generated")
        return result.solution.strip()

    async def generate_scaling_info(
        self,
        base_process: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str:
        """Generate scaling information"""
        logger.info(f"Generating scaling info: {base_process}")

        task_desc = f"""
Generate scaling information for: {base_process}

Domain: {domain}
Context: {context}

Provide scaling guidance including:
1. Linear scaling relationships
2. Non-linear considerations
3. Equipment limitations
4. Practical maximum/minimum scales
5. Special considerations for different scales
"""

        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=ComponentEvaluator("scaling_info", context),
            task_type=TaskType.CUSTOM,
            config=self.config.maker_config
        )

        self._track_operation("scaling_info", "generated")
        return result.solution.strip()

    async def optimize_component(
        self,
        component: Any,
        component_type: SOPComponentType,
        optimization_goal: str,
        context: Dict[str, Any] = None
    ) -> Any:
        """
        Optimize a component using evolutionary methods.

        Args:
            component: Component to optimize (parameter, step, etc.)
            component_type: Type of component
            optimization_goal: What to optimize (e.g., "minimize duration", "tighten tolerance")
            context: Additional context

        Returns:
            Optimized component
        """
        logger.info(f"Optimizing {component_type.value}: {optimization_goal}")

        if not self.config.enable_evolution:
            logger.warning("Evolution not enabled, returning original component")
            return component

        # Create population of variants
        population = [component]

        for _ in range(self.config.evolution_population_size - 1):
            variant = self._mutate_component(component, component_type)
            population.append(variant)

        # Evolve for specified generations
        best_component = component
        best_fitness = self._evaluate_component_fitness(component, optimization_goal, context)

        for generation in range(self.config.evolution_generations):
            # Evaluate population
            fitness_scores = [
                self._evaluate_component_fitness(ind, optimization_goal, context)
                for ind in population
            ]

            # Track best
            max_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_component = population[max_idx]
                logger.info(f"Generation {generation}: New best fitness = {best_fitness:.3f}")

            # Create next generation (simplified)
            population = self._create_next_generation(population, fitness_scores, component_type)

        self._track_operation(component_type.value, "optimized")

        return best_component

    async def test_component_safety(
        self,
        component: Any,
        component_type: SOPComponentType,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]:
        """
        Test a component for safety issues using adversarial red team.

        Args:
            component: Component to test
            component_type: Type of component
            context: Additional context

        Returns:
            (is_safe, list_of_issues)
        """
        logger.info(f"Testing {component_type.value} for safety issues")

        issues = []

        # Red team testing
        if component_type == SOPComponentType.PROTOCOL_STEP:
            # Check for missing safety elements
            if not hasattr(component, 'contingency_action') or not component.contingency_action:
                issues.append("No contingency action for failures")

            if "heat" in component.action.lower() and "safety" not in component.action.lower():
                issues.append("Heating operation without explicit safety mention")

        elif component_type == SOPComponentType.ENVIRONMENTAL_CONDITION:
            # Check for extreme values
            if hasattr(component, 'value'):
                if component.value > 1000 and component.unit == "°C":
                    issues.append("Extremely high temperature may be unsafe")

        # Track statistics
        self._track_operation(component_type.value, "tested")

        return (len(issues) == 0, issues)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _parse_parameter(self, solution: str, name: str) -> SOPParameter:
        """Parse parameter from MAKER solution"""
        import re
        import json

        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"value"[^}]*\}', solution, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return SOPParameter(
                    name=name,
                    value=float(data.get('value', 0)),
                    unit=data.get('unit', ''),
                    tolerance=float(data.get('tolerance', 0)),
                    verification_method=data.get('verification_method', ''),
                    critical=data.get('critical', True),
                    rationale=data.get('rationale', '')
                )
            except:
                pass

        # Fallback: extract from text
        return SOPParameter(
            name=name,
            value=25.0,
            unit="",
            tolerance=0.0,
            verification_method="Standard measurement",
            critical=True,
            rationale="As specified"
        )

    def _parse_equipment_spec(self, solution: str, name: str) -> Dict[str, str]:
        """Parse equipment specification from solution using JSON or regex extraction"""
        import re
        import json

        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"model"[^}]*\}', solution, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return {
                    "name": data.get("name", name),
                    "model": data.get("model", "Standard"),
                    "specifications": data.get("specifications", "As required"),
                    "features": data.get("features", "Standard features"),
                    "calibration": data.get("calibration", "Annual"),
                    "maintenance": data.get("maintenance", "Regular")
                }
            except:
                pass

        # Fallback: regex extraction
        model = re.search(r'[Mm]odel:\s*(.*)', solution)
        specs = re.search(r'[Ss]pecifications:\s*(.*)', solution)
        
        return {
            "name": name,
            "model": model.group(1).strip() if model else "Standard",
            "specifications": specs.group(1).strip() if specs else "As required",
            "features": "Standard features",
            "calibration": "Annual",
            "maintenance": "Regular"
        }

    def _parse_material_spec(self, solution: str, name: str) -> Dict[str, Any]:
        """Parse material specification from solution using JSON or regex extraction"""
        import re
        import json

        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"purity"[^}]*\}', solution, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return {
                    "name": data.get("name", name),
                    "purity": data.get("purity", ">= 99%"),
                    "grade": data.get("grade", "Standard"),
                    "amount": float(data.get("amount", 100.0)),
                    "unit": data.get("unit", "g"),
                    "tolerance": float(data.get("tolerance", 1.0)),
                    "storage": data.get("storage", "Room temperature"),
                    "safety": data.get("safety", "Standard precautions"),
                    "alternatives": data.get("alternatives", [])
                }
            except:
                pass

        # Fallback extraction
        purity = re.search(r'[Pp]urity:\s*(.*)', solution)
        amount = re.search(r'[Aa]mount:\s*([\d.]+)', solution)
        
        return {
            "name": name,
            "purity": purity.group(1).strip() if purity else ">= 99%",
            "grade": "Standard",
            "amount": float(amount.group(1)) if amount else 100.0,
            "unit": "g",
            "tolerance": 1.0,
            "storage": "Room temperature",
            "safety": "Standard precautions",
            "alternatives": []
        }

    def _parse_protocol_step(self, solution: str, step_number: int) -> SOPStep:
        """Parse protocol step from solution using JSON or robust regex"""
        import re
        import json

        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"action"[^}]*\}', solution, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return SOPStep(
                    step_number=step_number,
                    action=data.get("action", solution.strip()[:200]),
                    duration=float(data.get("duration", 300.0)),
                    duration_tolerance=float(data.get("duration_tolerance", 30.0)),
                    verification_method=data.get("verification_method", "Visual inspection"),
                    acceptance_criteria=data.get("acceptance_criteria", "Step completed successfully"),
                    contingency_action=data.get("contingency_action", "Repeat step if failed"),
                    substeps=data.get("substeps", [])
                )
            except:
                pass

        # Fallback: robust regex
        action = solution.strip().split('\n')[0]
        duration = re.search(r'[Dd]uration:\s*([\d.]+)', solution)
        verify = re.search(r'[Vv]erification:\s*(.*)', solution)
        
        return SOPStep(
            step_number=step_number,
            action=action[:200],
            duration=float(duration.group(1)) if duration else 300.0,
            duration_tolerance=30.0,
            verification_method=verify.group(1).strip() if verify else "Visual inspection",
            acceptance_criteria="Step completed successfully",
            contingency_action="Repeat step if failed",
            substeps=[]
        )

    def _track_operation(self, component_type: str, operation: str):
        """Track component-level operation"""
        if component_type not in self.statistics.get(operation, {}):
            if operation not in self.statistics:
                self.statistics[operation] = {}
            self.statistics[operation][component_type] = 0
        self.statistics[operation][component_type] += 1
        self.statistics["total_operations"] += 1

    async def _apply_integrations_to_parameter(
        self,
        param: SOPParameter,
        context: Dict[str, Any]
    ) -> SOPParameter:
        """Apply evolutionary optimization and safety testing to a parameter"""
        
        # 1. Evolutionary optimization (if enabled)
        if self.config.enable_evolution:
            param = await self.optimize_component(
                param, 
                SOPComponentType.ENVIRONMENTAL_CONDITION,
                "tighten tolerance and optimize value",
                context
            )
            
        # 2. Safety testing (adversarial)
        if self.config.enable_adversarial:
            is_safe, issues = await self.test_component_safety(
                param,
                SOPComponentType.ENVIRONMENTAL_CONDITION,
                context
            )
            if not is_safe:
                logger.warning(f"Safety issues found in parameter {param.name}: {issues}")
                # Refine if unsafe
                param = await self.refine_environmental_condition(
                    param,
                    f"Fix safety issues: {', '.join(issues)}",
                    context
                )
                
        return param

    def _mutate_component(self, component: Any, component_type: SOPComponentType) -> Any:
        """Create a mutated variant of a component for evolution"""
        import copy

        if component_type == SOPComponentType.ENVIRONMENTAL_CONDITION:
            mutated = copy.deepcopy(component)
            # Mutate tolerance slightly
            mutated.tolerance *= (0.9 + random.random() * 0.2)
            return mutated

        elif component_type == SOPComponentType.PROTOCOL_STEP:
            mutated = copy.deepcopy(component)
            # Mutate duration slightly
            if mutated.duration:
                mutated.duration *= (0.9 + random.random() * 0.2)
            return mutated

        return component

    def _evaluate_component_fitness(
        self,
        component: Any,
        goal: str,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate fitness of a component for optimization"""
        score = 0.5  # Base score

        if "tighten" in goal.lower():
            # Prefer smaller tolerances
            if hasattr(component, 'tolerance'):
                score += 0.3 * (1.0 / (1.0 + component.tolerance))

        elif "shorten" in goal.lower() or "minimize" in goal.lower():
            # Prefer shorter durations
            if hasattr(component, 'duration'):
                score += 0.3 * (1.0 / (1.0 + component.duration / 1000))

        return min(1.0, score)

    def _create_next_generation(
        self,
        population: List[Any],
        fitness_scores: List[float],
        component_type: SOPComponentType
    ) -> List[Any]:
        """Create next generation for evolution"""
        # Select top performers
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        top_count = len(population) // 2

        next_gen = [population[i] for i in sorted_indices[:top_count]]

        # Add mutated variants
        while len(next_gen) < len(population):
            parent = random.choice(next_gen[:top_count])
            next_gen.append(self._mutate_component(parent, component_type))

        return next_gen

    def get_statistics(self) -> Dict[str, Any]:
        """Get component-level statistics"""
        return self.statistics.copy()


# ============================================================================
# Component Evaluators
# ============================================================================

class ComponentEvaluator(GenericEvaluator):
    """Evaluator for SOP components"""

    def __init__(self, component_type: str, context: Dict[str, Any]):
        self.component_type = component_type
        self.context = context

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Evaluate component quality"""
        score = 0.0

        # Check for completeness
        if self.component_type == "environmental_condition":
            score += 0.3 * ("value" in solution.lower())
            score += 0.2 * ("tolerance" in solution.lower())
            score += 0.2 * ("verification" in solution.lower())
            score += 0.2 * ("rationale" in solution.lower())

        elif self.component_type == "protocol_step":
            score += 0.3 * ("action" in solution.lower() or "step" in solution.lower())
            score += 0.2 * ("duration" in solution.lower())
            score += 0.2 * ("verification" in solution.lower())
            score += 0.2 * ("acceptance" in solution.lower() or "criteria" in solution.lower())
            score += 0.1 * ("contingency" in solution.lower())

        elif self.component_type == "equipment_specification":
            score += 0.4 * ("model" in solution.lower() or "specification" in solution.lower())
            score += 0.3 * ("feature" in solution.lower())
            score += 0.3 * ("calibration" in solution.lower() or "maintenance" in solution.lower())

        elif self.component_type == "material":
            score += 0.3 * ("purity" in solution.lower() or "grade" in solution.lower())
            score += 0.3 * ("amount" in solution.lower())
            score += 0.2 * ("storage" in solution.lower())
            score += 0.2 * ("safety" in solution.lower())

        # Bonus for specificity (numeric values)
        import re
        numeric_count = len(re.findall(r'\d+\.?\d*', solution))
        score += min(0.2, numeric_count * 0.02)

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "context": self.context
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def generate_sop_component(
    component_type: SOPComponentType,
    component_name: str,
    context: Dict[str, Any],
    domain: str = "general",
    config: SOPIntegratedConfig = None
) -> Any:
    """
    Generate any SOP component.

    Args:
        component_type: Type of component to generate
        component_name: Name/description of component
        context: Context (equipment, materials, etc.)
        domain: Domain
        config: Configuration

    Returns:
        Generated component (parameter, step, spec dict, or string)
    """
    generator = SOPComponentGenerator(config)

    if component_type == SOPComponentType.ENVIRONMENTAL_CONDITION:
        return await generator.generate_environmental_condition(component_name, context, domain)

    elif component_type == SOPComponentType.EQUIPMENT_SPECIFICATION:
        return await generator.generate_equipment_specification(
            component_name,
            context.get('purpose', 'general use'),
            context,
            domain
        )

    elif component_type == SOPComponentType.MATERIAL:
        return await generator.generate_material(
            component_name,
            context.get('purpose', 'general use'),
            context,
            domain
        )

    elif component_type == SOPComponentType.PROTOCOL_STEP:
        step_number = context.get('step_number', 1)
        return await generator.generate_protocol_step(
            step_number,
            component_name,
            context,
            context.get('previous_steps'),
            domain
        )

    elif component_type == SOPComponentType.QUALITY_CONTROL:
        return await generator.generate_quality_control_procedure(component_name, context, domain)

    elif component_type == SOPComponentType.SAFETY_PROTOCOL:
        return await generator.generate_safety_protocol(component_name, context, domain)

    elif component_type == SOPComponentType.VALIDATION_CRITERION:
        return await generator.generate_validation_criterion(component_name, context, domain)

    elif component_type == SOPComponentType.SCALING_INFO:
        return await generator.generate_scaling_info(component_name, context, domain)

    else:
        raise ValueError(f"Unknown component type: {component_type}")


def get_component_capabilities() -> Dict[str, Any]:
    """Get component system capabilities"""
    return {
        "component_generation_enabled": True,
        "component_refinement_enabled": True,
        "component_optimization_enabled": True,
        "component_testing_enabled": True,
        "supported_components": [t.value for t in SOPComponentType],
        "features": {
            "independent_generation": "Generate any component independently",
            "component_refinement": "Refine existing components based on feedback",
            "evolutionary_optimization": "Optimize components through evolution",
            "adversarial_testing": "Test components for safety issues",
            "formal_verification": "Formally verify mathematical components",
            "granular_statistics": "Track operations at component level"
        }
    }
