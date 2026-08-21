import uuid
import logging
import random
import time
from dataclasses import dataclass

from openevolve.database import Program, ProgramDatabase
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """
    Run a single iteration using shared memory database

    This is optimized for use with persistent worker processes.
    """
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database
        parent, inspirations = database.sample(
            num_inspirations=config.prompt.num_top_programs
        )

        # Optional genetic-operator layer (OFF unless config.use_genetic_operators).
        # When enabled the documented core-evolution parameters become live:
        #    - selection_method / selection_pressure: choose the parent
        #    - crossover_rate: blend a second parent into the base code
        #    - mutation_rate: mutate the base code + scale LLM temperature
        #    - elitism / elite_ratio: preserve top-N candidates each generation
        # The flag-off path below is unchanged from the original implementation.
        parent_code = parent.code
        llm_temperature = None
        if getattr(config, "use_genetic_operators", False):
            from openevolve.genetic_operators import (
                crossover,
                elite_programs,
                mutate_code,
                mutation_temperature_scale,
                select_parent,
            )

            go_rng = random.Random(config.random_seed)
            go_pop = list(database.programs.values())
            go_fdims = database.config.feature_dimensions
            go_method = getattr(config, "selection_method", "tournament")

            # 1. Genetic parent selection
            if go_method in ("tournament", "roulette", "rank") and len(go_pop) > 1:
                parent = select_parent(
                    go_pop,
                    method=go_method,
                    selection_pressure=getattr(config, "selection_pressure", 1.0),
                    feature_dimensions=go_fdims,
                    rng=go_rng,
                )

            # 2. Crossover: blend a second parent into the base code
            if getattr(config, "crossover_rate", 0.0) > 0.0 and len(go_pop) > 1:
                others = [p for p in go_pop if p.id != parent.id]
                if others and go_rng.random() < getattr(config, "crossover_rate", 0.0):
                    second = select_parent(
                        others,
                        method=go_method,
                        selection_pressure=getattr(config, "selection_pressure", 1.0),
                        feature_dimensions=go_fdims,
                        rng=go_rng,
                    )
                    parent_code = crossover(parent, second, config.crossover_rate, go_rng)
                else:
                    parent_code = parent.code
            else:
                parent_code = parent.code

            # 3. Code-level mutation of the base code
            parent_code = mutate_code(
                parent_code, getattr(config, "mutation_rate", 0.0), go_rng
            )

            # 4. Elitism: preserve top-N candidates across generations
            if getattr(config, "elitism", False):
                elite_n = int(getattr(config, "elite_ratio", 0.1) * len(go_pop))
                if getattr(config, "elite_ratio", 0.1) > 1:
                    elite_n = int(getattr(config, "elite_ratio", 0.1))
                elite_n = max(0, elite_n)
                for e in elite_programs(go_pop, elite_n, go_fdims):
                    database.add(
                        Program(
                            code=e.code,
                            language=config.language or e.language,
                            parent_id=e.id,
                            generation=e.generation,
                            metrics=dict(e.metrics),
                            iteration_found=iteration,
                            metadata={**e.metadata, "elite": True},
                        )
                    )

            # 5. Scale LLM temperature by mutation rate
            base_temp = getattr(config.llm, "temperature", None)
            llm_temperature = mutation_temperature_scale(
                base_temp if base_temp is not None else 0.7,
                getattr(config, "mutation_rate", 0.0),
            )

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # Get island-specific top programs for prompt context (maintain island isolation)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(
            3, island_idx=parent_island
        )

        # Build prompt
        prompt = prompt_sampler.build_prompt(
            current_program=parent_code,
            parent_program=parent_code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
        )

        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        generate_kwargs = {}
        if llm_temperature is not None:
            generate_kwargs["temperature"] = llm_temperature
        llm_response = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
            **generate_kwargs,
        )

        # Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)

            if not diff_blocks:
                logger.warning(
                    f"Iteration {iteration + 1}: No valid diffs found in response"
                )
                return None

            # Apply the diffs
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(
                    f"Iteration {iteration + 1}: No valid code found in response"
                )
                return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Iteration {iteration + 1}: Generated code exceeds maximum length "
                f"({len(child_code)} > {config.max_code_length})"
            )
            return None

        # Evaluate the child program
        child_id = str(uuid.uuid4())
        result.child_metrics = await evaluator.evaluate_program(child_code, child_id)

        # Handle artifacts if they exist
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Set template_key of Prompts
        template_key = (
            "full_rewrite_user" if not config.diff_based_evolution else "diff_user"
        )

        # Create a child program
        result.child_program = Program(
            id=child_id,
            code=child_code,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
            prompts={
                template_key: {
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "responses": [llm_response] if llm_response is not None else [],
                }
            }
            if database.config.log_prompts
            else None,
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None
