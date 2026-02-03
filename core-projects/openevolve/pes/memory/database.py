#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database adapter for OpenEvolve PES framework.

Adapts OpenEvolve's ProgramDatabase to provide the LoongFlow EvolveDatabase interface.
This allows PES components to work with OpenEvolve's native database system.
"""

from typing import Optional, Dict, Any
import logging
import uuid

# Import OpenEvolve's native database
try:
    from openevolve.database import Program, ProgramDatabase
    OPENEVOLVE_DB_AVAILABLE = True
except ImportError:
    OPENEVOLVE_DB_AVAILABLE = False
    logging.warning("OpenEvolve's ProgramDatabase not available. Using fallback implementation.")

from openevolve.pes.config.config import DatabaseConfig

logger = logging.getLogger(__name__)


class EvolveDatabase:
    """
    Database adapter for PES framework.

    Provides LoongFlow-compatible interface using OpenEvolve's ProgramDatabase.
    Implements MAP-Elites algorithm, Boltzmann sampling, and island-based population.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database with given configuration.

        Args:
            config: DatabaseConfig object containing database parameters.
        """
        self.config = config

        if OPENEVOLVE_DB_AVAILABLE:
            # Initialize OpenEvolve's ProgramDatabase
            # Convert DatabaseConfig to OpenEvolve's expected format
            oe_config = {
                "storage_type": config.storage_type,
                "num_islands": config.num_islands,
                "population_size": config.population_size,
                "elite_archive_size": config.elite_archive_size,
                "use_sampling_weight": config.use_sampling_weight,
                "sampling_weight_power": config.sampling_weight_power,
                "migration_interval": config.migration_interval,
                "migration_rate": config.migration_rate,
                "boltzmann_temperature": config.boltzmann_temperature,
                "feature_bins": config.feature_bins,
                "feature_dimensions": config.feature_dimensions,
                "feature_scaling_method": config.feature_scaling_method,
                "output_path": config.output_path,
            }
            self._db = ProgramDatabase(config=oe_config)
        else:
            # Fallback: create simple in-memory storage
            logger.warning("Using fallback in-memory database")
            self._db = _FallbackDatabase()

    @classmethod
    def create_database(cls, config: DatabaseConfig) -> "EvolveDatabase":
        """
        Create an instance of the EvolveDatabase class using the provided configuration.

        Args:
            config: A DatabaseConfig object containing the configuration options.

        Returns:
            An instance of the EvolveDatabase class configured according to the provided configuration.
        """
        return cls(config)

    def sample_solution(self, island_id: Optional[int] = None) -> dict:
        """
        Sample a solution from the database based on the given message.

        Args:
            island_id: Optional island ID to sample from

        Returns:
            The sampled solution dict.
        """
        exploration_rate = self.config.exploration_rate

        # Check for local optimum by examining recent solutions
        # If stuck, increase exploration rate
        previous_solutions = self._db.get_best_programs(limit=5)
        if len(previous_solutions) >= 2:
            # Calculate the delta of the last 5 iterations
            deltas = [
                abs(previous_solutions[i].metrics.get("fitness", 0) -
                    previous_solutions[i + 1].metrics.get("fitness", 0))
                for i in range(len(previous_solutions) - 1)
            ]
            # Check if all deltas are small (local optimum)
            if deltas and all(delta < 0.01 for delta in deltas):
                exploration_rate = exploration_rate * 2
            elif deltas and all(delta < 0.001 for delta in deltas):
                exploration_rate = exploration_rate * 4

        if exploration_rate >= 1:
            exploration_rate = 0.9

        # Use OpenEvolve's sampling mechanism
        program = self._db.sample_program(
            island_id=island_id,
            exploration_rate=exploration_rate
        )

        # Convert Program to dict format expected by PES
        if program:
            return self._program_to_solution_dict(program)
        return {}

    async def add_solution(self, solution: Dict[str, Any]) -> str:
        """
        Add a new solution to the database.

        Args:
            solution: The solution dictionary with keys: solution, evaluation, score, etc.

        Returns:
            The ID of the added solution.
        """
        # Convert PES solution dict to OpenEvolve Program
        program = self._solution_dict_to_program(solution)

        if OPENEVOLVE_DB_AVAILABLE:
            program_id = self._db.add_program(program)
            return program_id
        else:
            # Fallback
            return self._db.add_solution(solution)

    async def update_solution(self, solution_id: str, **kwargs) -> str:
        """
        Update a solution in the database.

        Args:
            solution_id: The ID of the solution to update.
            **kwargs: Keyword arguments representing the fields to update.

        Returns:
            The ID of the updated solution.

        Raises:
            ValueError: If the solution_id is not specified.
        """
        if solution_id is None:
            raise ValueError("Solution id is required.")

        if OPENEVOLVE_DB_AVAILABLE:
            # Update in OpenEvolve database
            program = self._db.get_program(solution_id)
            if program:
                # Update fields
                for key, value in kwargs.items():
                    if hasattr(program, key):
                        setattr(program, key, value)
                return solution_id
        else:
            # Fallback
            return self._db.update_solution(solution_id, **kwargs)

        return solution_id

    def memory_status(self, island_id: Optional[int] = None) -> dict:
        """
        Get current status of the memory.

        Args:
            island_id: Optional island ID to get status for

        Returns:
            Dictionary containing memory status information
        """
        if OPENEVOLVE_DB_AVAILABLE:
            status = self._db.get_status(island_id=island_id)

            # Convert to PES format
            return {
                "global_status": {
                    "best_score": status.get("best_fitness", 0.0),
                    "best_iteration": status.get("best_iteration", 0),
                    "current_iteration": status.get("current_iteration", 0),
                    "total_solutions": status.get("total_programs", 0),
                },
                "island_status": status.get("island_status", {})
            }
        else:
            # Fallback
            return self._db.memory_status(island_id)

    async def save_checkpoint(self, checkpoint_path: str, tag: str):
        """
        Save the current state of the database to a checkpoint.

        Args:
            checkpoint_path: Base path for saving checkpoints
            tag: Checkpoint tag/identifier
        """
        if OPENEVOLVE_DB_AVAILABLE:
            self._db.save_checkpoint(checkpoint_path, tag)
        else:
            await self._db.save_checkpoint(checkpoint_path, tag)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        if OPENEVOLVE_DB_AVAILABLE:
            self._db.load_checkpoint(checkpoint_path)
        else:
            self._db.load_checkpoint(checkpoint_path)

    def get_parents_by_child_id(self, child_id: str, parent_cnt: int) -> list[dict]:
        """
        Get parents by child id.

        Args:
            child_id: Child solution ID
            parent_cnt: Number of parents to retrieve

        Returns:
            List of parent solution dictionaries
        """
        if OPENEVOLVE_DB_AVAILABLE:
            programs = self._db.get_ancestors(child_id, parent_cnt)
            return [self._program_to_solution_dict(p) for p in programs]
        else:
            # Fallback
            return self._db.get_parents_by_child_id(child_id, parent_cnt)

    def get_childs_by_parent_id(self, parent_id: str, child_cnt: int) -> list[dict]:
        """
        Get children by parent id.

        Args:
            parent_id: Parent solution ID
            child_cnt: Number of children to retrieve

        Returns:
            List of child solution dictionaries
        """
        if OPENEVOLVE_DB_AVAILABLE:
            programs = self._db.get_descendants(parent_id, child_cnt)
            return [self._program_to_solution_dict(p) for p in programs]
        else:
            # Fallback
            return self._db.get_childs_by_parent_id(parent_id, child_cnt)

    def get_solutions(self, solution_ids: list[str]) -> list[dict]:
        """
        Get solutions by ids.

        Args:
            solution_ids: List of solution ids

        Returns:
            List of solution dictionaries
        """
        if OPENEVOLVE_DB_AVAILABLE:
            programs = [self._db.get_program(sid) for sid in solution_ids]
            return [self._program_to_solution_dict(p) for p in programs if p]
        else:
            # Fallback
            return self._db.get_solutions(solution_ids)

    def get_best_solutions(
        self, island_id: Optional[int] = None, top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Get the best solutions.

        Args:
            island_id: Optional island ID to filter by
            top_k: Number of top solutions to retrieve

        Returns:
            List of solution dictionaries
        """
        if OPENEVOLVE_DB_AVAILABLE:
            programs = self._db.get_best_programs(island_id=island_id, limit=top_k)
            return [self._program_to_solution_dict(p) for p in programs]
        else:
            # Fallback
            return self._db.get_best_solutions(island_id, top_k)

    # =========================================================================
    # Helper methods for converting between PES format and OpenEvolve format
    # =========================================================================

    def _program_to_solution_dict(self, program) -> dict:
        """
        Convert OpenEvolve Program to PES solution dictionary format.

        Args:
            program: OpenEvolve Program object

        Returns:
            Dictionary in PES solution format
        """
        if program is None:
            return {}

        return {
            "solution_id": program.id,
            "solution": program.code,
            "evaluation": str(program.metrics),  # Convert metrics to string
            "score": program.metrics.get("fitness", 0.0),
            "island_id": getattr(program, "island_id", 0),
            "generate_plan": program.metadata.get("generate_plan", ""),
            "summary": program.metadata.get("summary", ""),
            "complexity": program.complexity,
            "diversity": program.diversity,
            "parent_id": program.parent_id,
            "generation": program.generation,
        }

    def _solution_dict_to_program(self, solution: Dict[str, Any]) -> 'Program':
        """
        Convert PES solution dictionary to OpenEvolve Program format.

        Args:
            solution: PES solution dictionary

        Returns:
            OpenEvolve Program object
        """
        if not OPENEVOLVE_DB_AVAILABLE:
            return None

        # Extract metrics
        metrics = {"fitness": solution.get("score", 0.0)}

        # Build metadata
        metadata = solution.get("metadata", {})
        metadata["generate_plan"] = solution.get("generate_plan", "")
        metadata["summary"] = solution.get("summary", "")

        return Program(
            id=solution.get("solution_id", str(uuid.uuid4())),
            code=solution.get("solution", ""),
            parent_id=solution.get("parent_id"),
            generation=solution.get("generation", 0),
            metrics=metrics,
            complexity=solution.get("complexity", 0.0),
            diversity=solution.get("diversity", 0.0),
            metadata=metadata,
        )


class _FallbackDatabase:
    """
    Fallback in-memory database when OpenEvolve's ProgramDatabase is not available.
    Provides basic functionality for testing and development.
    """

    def __init__(self):
        self._solutions = {}
        self._islands = {}

    def add_solution(self, solution: dict) -> str:
        solution_id = solution.get("solution_id", str(uuid.uuid4()))
        self._solutions[solution_id] = solution
        return solution_id

    def sample_program(self, island_id=None, exploration_rate=0.2):
        return None

    def get_best_programs(self, limit=None, island_id=None):
        return []

    def get_status(self, island_id=None):
        return {"best_fitness": 0.0, "current_iteration": 0}

    def get_program(self, program_id):
        return None

    def save_checkpoint(self, path, tag):
        pass

    def load_checkpoint(self, path):
        pass

    def memory_status(self, island_id=None):
        return {"global_status": {}}
