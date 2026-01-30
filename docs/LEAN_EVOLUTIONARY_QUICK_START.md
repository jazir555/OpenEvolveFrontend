"""
Quick Start Guide: Integrating LeanAide Evolutionary Workflow

This guide provides step-by-step instructions for integrating the evolutionary
LeanAide workflow into existing workflow_stage_functions.py.
"""

# =============================================================================
# STEP 1: Import the Evolutionary Module
# =============================================================================

# At the top of workflow_stage_functions.py, add:

from leanaide_evolutionary_workflow import (
    LeanEvolutionaryWorkflowStage,
    LeanEvolutionarySubProblemSolver,
    LeanEvolutionaryReassembler,
    EvolutionaryConfig,
    EvolutionStrategy,
    add_evolutionary_config_to_workflow_state,
    extract_evolutionary_config_from_workflow_state,
    is_subproblem_mathematical,
    LEANAIDE_AVAILABLE,
    EVOLUTION_AVAILABLE,
    ADVERSARIAL_AVAILABLE,
    SELFPLAY_AVAILABLE
)


# =============================================================================
# STEP 2: Initialize Evolutionary Stage
# =============================================================================

async def initialize_evolutionary_stage(workflow_state: WorkflowState) -> Optional[LeanEvolutionaryWorkflowStage]:
    """
    Initialize the evolutionary workflow stage based on workflow configuration.

    Call this at the beginning of Stage 3 to set up evolutionary capabilities.
    """

    # Extract configuration from workflow state
    config = extract_evolutionary_config_from_workflow_state(workflow_state)

    # Check if evolution is enabled
    if not config.lean_evolution_enabled:
        logger.info("LeanAide evolution disabled, using standard workflow")
        return None

    # Check component availability
    if not LEANAIDE_AVAILABLE:
        logger.warning("LeanAide not available, evolution disabled")
        return None

    # Create evolutionary stage
    evolutionary_stage = LeanEvolutionaryWorkflowStage(
        config=config,
        workflow_state=workflow_state
    )

    # Initialize LeanAide connection
    if LEANAIDE_AVAILABLE and evolutionary_stage.leanaide_integrator:
        initialized = await evolutionary_stage.leanaide_integrator.initialize()
        if not initialized:
            logger.warning("LeanAide initialization failed")
            return None

    logger.info("Evolutionary workflow stage initialized successfully")
    return evolutionary_stage


# =============================================================================
# STEP 3: Integrate Stage 3A - Solution Generation
# =============================================================================

async def execute_stage3a_solution_loop_evolutionary(
    workflow_state: WorkflowState,
    evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
) -> WorkflowState:
    """
    Execute Stage 3A: Solution Loop with evolutionary integration.

    Enhanced version of existing execute_stage3a_solution_loop() that
    automatically applies evolutionary approach to mathematical sub-problems.
    """

    logger.info("Starting Stage 3A: Solution Generation (with evolutionary support)")

    # Initialize evolutionary stage if not provided
    if evolutionary_stage is None:
        evolutionary_stage = await initialize_evolutionary_stage(workflow_state)

    # Create solver
    if evolutionary_stage:
        sub_problem_solver = LeanEvolutionarySubProblemSolver(evolutionary_stage)
    else:
        # Use standard solver
        sub_problem_solver = StandardSubProblemSolver()  # Existing solver

    # Process each sub-problem
    for sub_problem in workflow_state.decomposition_plan.sub_problems:
        logger.info(f"Solving sub-problem: {sub_problem.id}")

        try:
            # Solve using appropriate approach
            solution = await sub_problem_solver.solve(sub_problem, workflow_state)

            # Store solution
            workflow_state.sub_problem_solutions[sub_problem.id] = solution

            # Track progress if evolutionary
            if evolutionary_stage:
                progress = evolutionary_stage.get_progress(sub_problem.id)
                if progress:
                    logger.info(
                        f"Evolutionary progress: {progress.generation} generations, "
                        f"best fitness: {progress.best_fitness:.2f}"
                    )

        except Exception as e:
            logger.error(f"Failed to solve sub-problem {sub_problem.id}: {e}")
            # Create placeholder solution
            workflow_state.sub_problem_solutions[sub_problem.id] = SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=f"# Error: Failed to solve - {str(e)}",
                generated_by_model="Error",
                timestamp=time.time(),
                status="rejected"
            )

    workflow_state.current_stage = "Stage 3A Complete"
    return workflow_state


# =============================================================================
# STEP 4: Integrate Stage 3B - Adversarial Critique
# =============================================================================

async def execute_stage3b_adversarial_critique_evolutionary(
    workflow_state: WorkflowState,
    evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
) -> WorkflowState:
    """
    Execute Stage 3B: Adversarial Critique with evolutionary enhancement.

    Enhanced version that applies adversarial evolution to mathematical solutions.
    """

    logger.info("Starting Stage 3B: Adversarial Critique (with evolutionary support)")

    # Initialize evolutionary stage if not provided
    if evolutionary_stage is None:
        evolutionary_stage = await initialize_evolutionary_stage(workflow_state)

    # Process each solution
    for sp_id, solution in list(workflow_state.sub_problem_solutions.items()):
        try:
            # Apply adversarial evolution if available
            if evolutionary_stage:
                # Check if mathematical
                sub_problem = get_sub_problem_by_id(
                    workflow_state.decomposition_plan, sp_id
                )

                if sub_problem:
                    is_math, confidence, _ = evolutionary_stage.is_mathematical_subproblem(
                        sub_problem
                    )

                    if is_math and confidence > 0.5:
                        logger.info(f"Applying adversarial evolution to {sp_id}")
                        evolved = await evolutionary_stage.adversarial_evolution_stage3b(
                            solution, workflow_state
                        )
                        workflow_state.sub_problem_solutions[sp_id] = evolved
                    else:
                        # Use standard red team critique
                        critique = await execute_standard_red_team_critique(
                            solution, workflow_state
                        )
                        solution.critique_reports.append(critique)
            else:
                # Use standard red team critique
                critique = await execute_standard_red_team_critique(
                    solution, workflow_state
                )
                solution.critique_reports.append(critique)

        except Exception as e:
            logger.error(f"Failed to critique solution {sp_id}: {e}")

    workflow_state.current_stage = "Stage 3B Complete"
    return workflow_state


# =============================================================================
# STEP 5: Integrate Stage 3C - Gold Team Verification
# =============================================================================

async def execute_stage3c_verification_evolutionary(
    workflow_state: WorkflowState,
    evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
) -> WorkflowState:
    """
    Execute Stage 3C: Gold Team Verification with LeanAide integration.

    Enhanced version that performs formal verification for mathematical proofs.
    """

    logger.info("Starting Stage 3C: Verification (with LeanAide support)")

    # Initialize evolutionary stage if not provided
    if evolutionary_stage is None:
        evolutionary_stage = await initialize_evolutionary_stage(workflow_state)

    # Verify each solution
    for sp_id, solution in list(workflow_state.sub_problem_solutions.items()):
        try:
            if evolutionary_stage:
                # Use LeanAide verification
                verification_report = await evolutionary_stage.verify_evolved_proof_stage3c(
                    solution, workflow_state
                )
                solution.verification_reports.append(verification_report)

                logger.info(
                    f"Verification {sp_id}: approved={verification_report.is_approved}, "
                    f"score={verification_report.average_score:.2f}"
                )
            else:
                # Use standard gold team verification
                verification_report = await execute_standard_gold_team_verification(
                    solution, workflow_state
                )
                solution.verification_reports.append(verification_report)

        except Exception as e:
            logger.error(f"Failed to verify solution {sp_id}: {e}")

    workflow_state.current_stage = "Stage 3C Complete"
    return workflow_state


# =============================================================================
# STEP 6: Integrate Stage 4 - Solution Assembly with Evolutionary Reassembly
# =============================================================================

async def execute_stage4_solution_assembly_evolutionary(
    workflow_state: WorkflowState,
    evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
) -> WorkflowState:
    """
    Execute Stage 4: Solution Assembly with evolutionary reassembly.

    Enhanced version that uses LeanEvolutionaryReassembler for mathematical proofs.
    """

    logger.info("Starting Stage 4: Solution Assembly (with evolutionary reassembly)")

    if evolutionary_stage:
        # Use evolutionary reassembler
        reassembler = LeanEvolutionaryReassembler(evolutionary_stage)
        integrated_solution = await reassembler.reassemble(
            workflow_state.sub_problem_solutions,
            workflow_state
        )
    else:
        # Use standard assembly
        integrated_solution = await standard_solution_assembly(
            workflow_state.sub_problem_solutions,
            workflow_state
        )

    workflow_state.final_solution = integrated_solution
    workflow_state.current_stage = "Stage 4 Complete"
    return workflow_state


# =============================================================================
# STEP 7: Integrate Stage 5 - Final Verification
# =============================================================================

async def execute_stage5_final_verification_evolutionary(
    workflow_state: WorkflowState,
    evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None
) -> WorkflowState:
    """
    Execute Stage 5: Final Verification with LeanAide support.

    Enhanced version that performs comprehensive final verification including
    evolutionary components.
    """

    logger.info("Starting Stage 5: Final Verification (with LeanAide support)")

    if evolutionary_stage and workflow_state.final_solution:
        # Use LeanAide final verification
        final_report = await evolutionary_stage.evolutionary_final_verification_stage5(
            workflow_state.final_solution,
            workflow_state
        )
        workflow_state.all_verification_reports.append(final_report)

        logger.info(
            f"Final verification: approved={final_report.is_approved}, "
            f"score={final_report.average_score:.2f}"
        )
    else:
        # Use standard final verification
        final_report = await execute_standard_final_verification(
            workflow_state.final_solution,
            workflow_state
        )
        workflow_state.all_verification_reports.append(final_report)

    workflow_state.current_stage = "Stage 5 Complete"
    return workflow_state


# =============================================================================
# STEP 8: Updated Main Workflow Execution
# =============================================================================

async def execute_workflow_with_evolutionary_support(
    decomposition_plan: DecompositionPlan,
    evolutionary_config: Optional[EvolutionaryConfig] = None
) -> WorkflowState:
    """
    Execute the complete decomposition workflow with evolutionary support.

    This is the main entry point that replaces or enhances the existing
    execute_workflow() function.
    """

    # Create workflow state
    workflow_state = WorkflowState(
        workflow_id=str(uuid.uuid4()),
        workflow_type=WorkflowType.DECOMPOSITION_WORKFLOW,
        problem_statement=decomposition_plan.problem_statement,
        current_stage="Initializing",
        decomposition_plan=decomposition_plan
    )

    # Add evolutionary configuration if provided
    if evolutionary_config:
        workflow_state = add_evolutionary_config_to_workflow_state(
            workflow_state, evolutionary_config
        )

    try:
        # Execute Stage 1: Problem Analysis (unchanged)
        workflow_state = await execute_stage1_problem_analysis(workflow_state)

        # Execute Stage 2: Decomposition (unchanged)
        workflow_state = await execute_stage2_decomposition(workflow_state)

        # Initialize evolutionary stage
        evolutionary_stage = await initialize_evolutionary_stage(workflow_state)

        # Execute Stage 3A: Solution Generation (with evolutionary support)
        workflow_state = await execute_stage3a_solution_loop_evolutionary(
            workflow_state, evolutionary_stage
        )

        # Execute Stage 3B: Adversarial Critique (with evolutionary support)
        workflow_state = await execute_stage3b_adversarial_critique_evolutionary(
            workflow_state, evolutionary_stage
        )

        # Execute Stage 3C: Verification (with LeanAide support)
        workflow_state = await execute_stage3c_verification_evolutionary(
            workflow_state, evolutionary_stage
        )

        # Execute Stage 4: Solution Assembly (with evolutionary reassembly)
        workflow_state = await execute_stage4_solution_assembly_evolutionary(
            workflow_state, evolutionary_stage
        )

        # Execute Stage 5: Final Verification (with LeanAide support)
        workflow_state = await execute_stage5_final_verification_evolutionary(
            workflow_state, evolutionary_stage
        )

        workflow_state.status = "completed"
        logger.info("Workflow completed successfully")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        workflow_state.status = "failed"
        raise

    return workflow_state


# =============================================================================
# STEP 9: Configuration Helpers
# =============================================================================

def create_evolutionary_config_from_parameters(
    enable_evolution: bool = True,
    strategy: str = "hybrid",
    generations: int = 50,
    population_size: int = 20,
    adversarial_rounds: int = 10
) -> EvolutionaryConfig:
    """
    Create evolutionary configuration from simplified parameters.

    Helper function for easier configuration from UI or config files.
    """

    strategy_map = {
        "standard": EvolutionStrategy.STANDARD,
        "evolution": EvolutionStrategy.EVOLUTION,
        "adversarial": EvolutionStrategy.ADVERSARIAL,
        "self_play": EvolutionStrategy.SELF_PLAY,
        "hybrid": EvolutionStrategy.HYBRID
    }

    return EvolutionaryConfig(
        lean_evolution_enabled=enable_evolution,
        lean_evolution_strategy=strategy_map.get(strategy, EvolutionStrategy.HYBRID),
        lean_evolution_generations=generations,
        lean_evolution_population_size=population_size,
        lean_adversarial_rounds=adversarial_rounds,
        lean_fallback_to_standard=True,  # Always enable fallback
        lean_auto_detect_mathematical=True,  # Auto-detect math problems
        lean_store_evolved_proofs=True,  # Store in knowledge base
        lean_track_evolution_statistics=True  # Track progress
    )


def get_evolutionary_statistics(
    workflow_state: WorkflowState
) -> Dict[str, Any]:
    """
    Extract evolutionary statistics from workflow state.

    Returns summary of evolutionary runs for UI display.
    """

    if not workflow_state.openevolve_parameters:
        return {
            "evolution_used": False,
            "message": "Evolution not configured"
        }

    # Check if evolution was enabled
    if not workflow_state.openevolve_parameters.get("lean_evolution_enabled", False):
        return {
            "evolution_used": False,
            "message": "Evolution disabled in configuration"
        }

    # Extract statistics from solutions
    stats = {
        "evolution_used": True,
        "strategy": workflow_state.openevolve_parameters.get(
            "lean_evolution_strategy", "unknown"
        ),
        "sub_problems_processed": 0,
        "evolutionary_sub_problems": 0,
        "sub_problem_details": []
    }

    for sp_id, solution in workflow_state.sub_problem_solutions.items():
        stats["sub_problems_processed"] += 1

        if solution.openevolve_metrics:
            stats["evolutionary_sub_problems"] += 1
            stats["sub_problem_details"].append({
                "sub_problem_id": sp_id,
                "approach": solution.solution_approach,
                "metrics": solution.openevolve_metrics
            })

    return stats


# =============================================================================
# STEP 10: Helper Functions
# =============================================================================

def get_sub_problem_by_id(
    decomposition_plan: DecompositionPlan,
    sub_problem_id: str
) -> Optional[SubProblem]:
    """Get sub-problem by ID from decomposition plan."""

    for sp in decomposition_plan.sub_problems:
        if sp.id == sub_problem_id:
            return sp
    return None


async def execute_standard_red_team_critique(
    solution: SolutionAttempt,
    workflow_state: WorkflowState
) -> CritiqueReport:
    """Fallback standard red team critique."""
    # Existing implementation
    pass


async def execute_standard_gold_team_verification(
    solution: SolutionAttempt,
    workflow_state: WorkflowState
) -> VerificationReport:
    """Fallback standard gold team verification."""
    # Existing implementation
    pass


async def standard_solution_assembly(
    sub_problem_solutions: Dict[str, SolutionAttempt],
    workflow_state: WorkflowState
) -> SolutionAttempt:
    """Fallback standard solution assembly."""
    # Existing implementation
    pass


async def execute_standard_final_verification(
    final_solution: SolutionAttempt,
    workflow_state: WorkflowState
) -> VerificationReport:
    """Fallback standard final verification."""
    # Existing implementation
    pass


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_usage_basic():
    """Example: Basic usage with default configuration."""

    # Create decomposition plan
    decomposition_plan = DecompositionPlan(
        problem_statement="Prove mathematical theorems",
        analyzed_context={},
        sub_problems=[
            SubProblem(
                id="sp_001",
                description="Prove that sqrt(2) is irrational",
                dependencies=[]
            )
        ]
    )

    # Execute with evolutionary support (default config)
    workflow_state = await execute_workflow_with_evolutionary_support(
        decomposition_plan
    )

    # Check results
    stats = get_evolutionary_statistics(workflow_state)
    print(f"Evolution used: {stats['evolution_used']}")
    print(f"Sub-problems processed: {stats['sub_problems_processed']}")


async def example_usage_custom_config():
    """Example: Custom evolutionary configuration."""

    # Create custom config
    config = create_evolutionary_config_from_parameters(
        enable_evolution=True,
        strategy="adversarial",
        generations=100,
        population_size=50,
        adversarial_rounds=20
    )

    # Execute with custom config
    workflow_state = await execute_workflow_with_evolutionary_support(
        decomposition_plan,
        evolutionary_config=config
    )

    # Get detailed statistics
    stats = get_evolutionary_statistics(workflow_state)
    for detail in stats["sub_problem_details"]:
        print(f"Sub-problem: {detail['sub_problem_id']}")
        print(f"  Approach: {detail['approach']}")
        print(f"  Metrics: {detail['metrics']}")


# =============================================================================
# MIGRATION CHECKLIST
# =============================================================================

"""
To migrate existing workflow_stage_functions.py to use evolutionary integration:

1. Add imports at top of file
2. Add initialize_evolutionary_stage() function
3. Update execute_stage3a_solution_loop() -> execute_stage3a_solution_loop_evolutionary()
4. Update execute_stage3b_adversarial_critique() -> execute_stage3b_adversarial_critique_evolutionary()
5. Update execute_stage3c_verification() -> execute_stage3c_verification_evolutionary()
6. Update execute_stage4_solution_assembly() -> execute_stage4_solution_assembly_evolutionary()
7. Update execute_stage5_final_verification() -> execute_stage5_final_verification_evolutionary()
8. Update execute_workflow() -> execute_workflow_with_evolutionary_support()
9. Add configuration helpers
10. Add statistics helpers
11. Update tests
12. Update documentation

The integration is designed to be:
- Non-breaking: Works without LeanAide components
- Graceful: Falls back to standard approach on errors
- Optional: Can be enabled/disabled via configuration
- Transparent: Minimal changes to existing code
"""

# Export all functions
__all__ = [
    # Initialization
    "initialize_evolutionary_stage",
    "create_evolutionary_config_from_parameters",

    # Stage execution
    "execute_stage3a_solution_loop_evolutionary",
    "execute_stage3b_adversarial_critique_evolutionary",
    "execute_stage3c_verification_evolutionary",
    "execute_stage4_solution_assembly_evolutionary",
    "execute_stage5_final_verification_evolutionary",
    "execute_workflow_with_evolutionary_support",

    # Statistics
    "get_evolutionary_statistics",

    # Helpers
    "get_sub_problem_by_id",
]
