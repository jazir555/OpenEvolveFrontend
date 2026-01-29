"""
Gauntlet Integration for DecompositionEngine

This module provides integration methods for the formal gauntlet system
with the DecompositionEngine. These methods can be mixed into the
DecompositionEngine class or used as standalone functions.
"""

import logging
from typing import Optional, Dict, Any

from sovereign_data_models import (
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    GauntletAssignment,
    GauntletExecution,
    GauntletDefinition,
    DecompositionStrategy
)
from sovereign_data_models import generate_id

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

logger = logging.getLogger(__name__)


class GauntletDecompositionMixin:
    """
    Mixin class providing gauntlet integration methods for DecompositionEngine.

    This mixin can be used to extend DecompositionEngine with gauntlet functionality
    without modifying the core decomposition logic.
    """

    def _init_roma_engine(self):
        """Initialize ROMA engine for robust decomposition."""
        if not hasattr(self, 'roma_engine'):
            self.roma_engine = None
            if ROMA_MDAP_MAKER_AVAILABLE:
                try:
                    # Use SSOT validation preset for standardized high-reliability config
                    config_roma = get_validation_config()
                    self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                    self.logger.info("ROMAMDAPMakerAssociativeEngine initialized for GauntletDecompositionMixin")
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    self.logger.error(f"Failed to initialize ROMA engine: {e}")

    def decompose_with_gauntlets(
        self,
        problem: ProblemDefinition,
        strategy: Optional[str] = None,
        use_gauntlets: bool = True,
        gauntlet_template: str = "standard",
        **kwargs
    ) -> DecompositionPlan:
        """
        Decompose problem with automatic gauntlet assignment.

        This is an enhanced version of decompose() that automatically assigns
        gauntlets to sub-problems for validation.

        Args:
            problem: The problem to decompose
            strategy: Optional strategy name (auto-selected if not provided)
            use_gauntlets: If True, assign gauntlets to sub-problems
            gauntlet_template: Gauntlet template to use ("standard", "security", "performance", "research")
            **kwargs: Additional arguments passed to decompose()

        Returns:
            DecompositionPlan with gauntlet assignments
        """
        self.logger.info(f"Decomposing problem with gauntlets: {problem.id}, template: {gauntlet_template}")
        self._init_roma_engine()

        # Perform standard decomposition
        plan = self.decompose(problem, strategy=strategy, **kwargs)

        # Assign gauntlets to sub-problems if requested
        if use_gauntlets:
            try:
                from formal_gauntlet_system import GauntletSystem, GauntletTemplates

                gauntlet_system = GauntletSystem(team_manager=getattr(self, 'team_assignment_engine', None))
                template = GauntletTemplates.get_template(gauntlet_template)

                if template:
                    self.logger.info(f"Assigning {gauntlet_template} gauntlet to {len(plan.sub_problems)} sub-problems")

                    for sp in plan.sub_problems:
                        # Assign appropriate gauntlet
                        gauntlet = gauntlet_system.create_gauntlet(
                            gauntlet_id=f"{sp.id}_{gauntlet_template}",
                            name=f"{gauntlet_template.title()} for {sp.title}",
                            rounds=template.rounds,
                            description=template.description,
                            execution_order=template.execution_order,
                            stop_on_first_failure=template.stop_on_first_failure,
                            require_all_rounds=template.require_all_rounds,
                            red_team_required=template.red_team_required,
                            gold_team_required=template.gold_team_required,
                            blue_team_participation=template.blue_team_participation
                        )

                        # Store gauntlet assignment
                        if not sp.ai_suggested_gauntlet_assignment:
                            sp.ai_suggested_gauntlet_assignment = GauntletAssignment()

                        # For compatibility, we store the gauntlet ID as a string
                        sp.ai_suggested_gauntlet_assignment.red_team_gauntlet = gauntlet.gauntlet_id
                        sp.ai_suggested_gauntlet_assignment.gold_team_gauntlet = gauntlet.gauntlet_id

                        # Store the full gauntlet definition in metadata
                        if not sp.metadata:
                            sp.metadata = {}
                        sp.metadata['red_team_gauntlet_definition'] = gauntlet.to_dict()
                        sp.metadata['gold_team_gauntlet_definition'] = gauntlet.to_dict()

                    self.logger.info(f"Gauntlet assignment complete for {len(plan.sub_problems)} sub-problems")

                else:
                    self.logger.warning(f"Gauntlet template '{gauntlet_template}' not found")

            except ImportError as e:
                self.logger.warning(f"Formal gauntlet system not available: {e}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                self.logger.error(f"Failed to assign gauntlets: {e}", exc_info=True)

        return plan

    def execute_solution_gauntlets(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        gauntlet_assignment: GauntletAssignment
    ) -> GauntletExecution:
        """
        Execute gauntlets for a solution.

        This is the validation phase after solution generation.

        Args:
            solution: The solution attempt to validate
            sub_problem: The sub-problem this solution addresses
            gauntlet_assignment: The gauntlet assignment for this sub-problem

        Returns:
            GauntletExecution with results and feedback
        """
        self.logger.info(f"Executing gauntlets for solution {solution.id}")

        try:
            from formal_gauntlet_system import GauntletSystem

            gauntlet_system = GauntletSystem(team_manager=getattr(self, 'team_assignment_engine', None))

            # Get gauntlet definitions from metadata or create from templates
            red_gauntlet_def = None
            gold_gauntlet_def = None

            if sub_problem.metadata:
                red_gauntlet_dict = sub_problem.metadata.get('red_team_gauntlet_definition')
                gold_gauntlet_dict = sub_problem.metadata.get('gold_team_gauntlet_definition')

                if red_gauntlet_dict:
                    red_gauntlet_def = GauntletDefinition.from_dict(red_gauntlet_dict)
                if gold_gauntlet_dict:
                    gold_gauntlet_def = GauntletDefinition.from_dict(gold_gauntlet_dict)

            # If no gauntlets found, use standard template
            if not red_gauntlet_def or not gold_gauntlet_def:
                from formal_gauntlet_system import GauntletTemplates
                template = GauntletTemplates.standard_validation_gauntlet()
                red_gauntlet_def = template
                gold_gauntlet_def = template

            # Execute red team gauntlet
            if red_gauntlet_def:
                self.logger.info(f"Executing red team gauntlet: {red_gauntlet_def.gauntlet_id}")
                red_result = gauntlet_system.execute_gauntlet(
                    red_gauntlet_def,
                    solution,
                    sub_problem
                )
                self.logger.info(f"Red team result: passed={red_result.overall_passed}, score={red_result.final_score:.2f}")
            else:
                red_result = None

            # Execute gold team gauntlet
            if gold_gauntlet_def:
                self.logger.info(f"Executing gold team gauntlet: {gold_gauntlet_def.gauntlet_id}")
                gold_result = gauntlet_system.execute_gauntlet(
                    gold_gauntlet_def,
                    solution,
                    sub_problem
                )
                self.logger.info(f"Gold team result: passed={gold_result.overall_passed}, score={gold_result.final_score:.2f}")
            else:
                gold_result = None

            # Combine results
            # Use the more comprehensive of the two results
            execution = red_result if red_result else gold_result
            if not execution:
                raise ValueError("No gauntlet execution results available")

            # Store both results in metadata
            execution.metadata['red_team_result'] = red_result.to_dict() if red_result else None
            execution.metadata['gold_team_result'] = gold_result.to_dict() if gold_result else None

            # Overall pass requires both to pass if both were executed
            if red_result and gold_result:
                execution.overall_passed = red_result.overall_passed and gold_result.overall_passed
                execution.final_score = (red_result.final_score + gold_result.final_score) / 2.0

            self.logger.info(f"Gauntlet execution complete: passed={execution.overall_passed}, score={execution.final_score:.2f}")
            return execution

        except ImportError as e:
            self.logger.error(f"Formal gauntlet system not available: {e}")
            raise
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            self.logger.error(f"Failed to execute gauntlets: {e}", exc_info=True)
            raise
