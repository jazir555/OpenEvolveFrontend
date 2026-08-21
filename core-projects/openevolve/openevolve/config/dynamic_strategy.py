"""
Dynamic Strategy Switching

This module provides functionality to switch between different evolutionary strategies
mid-run, with state migration and preservation capabilities.
"""

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel

from ..unified.config import (
    UnifiedEvolutionConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig
)
from .config_metrics import compute_adaptive_metrics


logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System mode enumeration"""
    PES = "pes"
    QD = "qd"
    MO = "mo"
    ADVERSARIAL = "adversarial"
    OPENEVOLVE = "openevolve"
    HYBRID = "hybrid"


class StrategySwitchRecord(BaseModel):
    """Record of a strategy switch"""
    timestamp: datetime
    from_strategy: SystemMode
    to_strategy: SystemMode
    state_preserved: bool
    migration_success: bool
    reason: Optional[str] = None


class DynamicStrategySwitcher:
    """
    Switch evolutionary strategies mid-run

    Features:
    - Graceful strategy transitions
    - State migration between strategies
    - Validation of compatibility
    - Rollback capabilities
    - Switch history tracking
    """

    def __init__(self, current_strategy: SystemMode):
        """
        Initialize strategy switcher

        Args:
            current_strategy: Current active strategy
        """
        self.current_strategy = current_strategy
        self.strategy_history: List[StrategySwitchRecord] = []
        self.state_migrator = StateMigrator()

        # Track running state
        self.is_running = False
        self.current_state: Optional[Dict] = None

    async def switch_strategy(
        self,
        new_strategy: SystemMode,
        new_config: UnifiedEvolutionConfig,
        preserve_state: bool = True,
        reason: Optional[str] = None
    ) -> bool:
        """
        Switch to different evolutionary strategy

        Args:
            new_strategy: New strategy (SystemMode enum)
            new_config: Configuration for new strategy
            preserve_state: Whether to migrate learned state
            reason: Optional reason for the switch

        Returns:
            True if switch successful
        """
        if new_strategy == self.current_strategy:
            logger.warning(f"Already using strategy {new_strategy}")
            return True

        logger.info(f"Switching strategy: {self.current_strategy} -> {new_strategy}")
        if reason:
            logger.info(f"Reason: {reason}")

        # Validate switch is possible
        if not self._validate_switch(new_strategy, new_config):
            logger.error(f"Strategy switch validation failed")
            return False

        # Capture current state
        old_state = self.current_state if preserve_state else None

        # Migrate state if requested
        migrated_state = None
        migration_success = True

        if preserve_state and old_state:
            try:
                migrated_state = await self.state_migrator.migrate(
                    from_strategy=self.current_strategy,
                    to_strategy=new_strategy,
                    current_state=old_state
                )
                logger.info("State migration completed")
            except Exception as e:
                logger.error(f"State migration failed: {e}")
                migration_success = False

                # Decide whether to continue without state
                if not self._can_continue_without_state():
                    return False

        # Perform switch
        try:
            # Stop current strategy
            await self._stop_current_strategy()

            # Start new strategy
            await self._start_new_strategy(new_strategy, new_config, migrated_state)

            # Record switch
            self.strategy_history.append(StrategySwitchRecord(
                timestamp=datetime.utcnow(),
                from_strategy=self.current_strategy,
                to_strategy=new_strategy,
                state_preserved=preserve_state,
                migration_success=migration_success,
                reason=reason
            ))

            self.current_strategy = new_strategy
            logger.info(f"Strategy switched successfully to {new_strategy}")
            return True

        except Exception as e:
            logger.error(f"Strategy switch failed: {e}", exc_info=True)

            # Attempt rollback
            if await self._rollback_strategy():
                logger.info("Rolled back to previous strategy")
            else:
                logger.error("Rollback failed")

            return False

    def _validate_switch(
        self,
        new_strategy: SystemMode,
        new_config: UnifiedEvolutionConfig
    ) -> bool:
        """
        Check if switch is valid

        Args:
            new_strategy: Target strategy
            new_config: Configuration for new strategy

        Returns:
            True if switch is valid
        """
        # Check if mode is enabled in config
        if new_config.evolution_mode != new_strategy.value:
            logger.error(f"Config evolution_mode mismatch: {new_config.evolution_mode} != {new_strategy.value}")
            return False

        # Check if strategy-specific config is present
        strategy_config = self._get_strategy_config(new_strategy, new_config)
        if strategy_config is None:
            logger.error(f"Missing config for strategy {new_strategy}")
            return False

        # Check resource compatibility
        if not self._check_resource_compatibility(new_strategy, new_config):
            logger.error(f"Resource compatibility check failed for {new_strategy}")
            return False

        return True

    def _get_strategy_config(
        self,
        strategy: SystemMode,
        config: UnifiedEvolutionConfig
    ) -> Optional[Any]:
        """Get strategy-specific configuration"""
        config_map = {
            SystemMode.PES: config.pes,
            SystemMode.QD: config.qd,
            SystemMode.MO: config.mo,
            SystemMode.ADVERSARIAL: config.adversarial,
            SystemMode.OPENEVOLVE: config.openevolve,
        }
        return config_map.get(strategy)

    def _check_resource_compatibility(
        self,
        new_strategy: SystemMode,
        config: UnifiedEvolutionConfig
    ) -> bool:
        """Check if current resources are sufficient for new strategy"""
        # Basic checks - can be extended
        required_concurrency = {
            SystemMode.PES: 3,
            SystemMode.QD: 5,
            SystemMode.MO: 4,
            SystemMode.ADVERSARIAL: 6,
            SystemMode.OPENEVOLVE: 2,
        }

        required = required_concurrency.get(new_strategy, 2)
        available = config.common.concurrency

        if available < required:
            logger.warning(
                f"Insufficient concurrency for {new_strategy}: "
                f"{available} < {required}"
            )
            return False

        return True

    async def _stop_current_strategy(self) -> None:
        """Gracefully stop current strategy"""
        logger.info(f"Stopping strategy {self.current_strategy}")

        # Save current state
        self.current_state = await self._capture_current_state()

        # Perform cleanup
        # This would integrate with the actual evolution system
        # For now, just mark as not running
        self.is_running = False

    async def _start_new_strategy(
        self,
        new_strategy: SystemMode,
        new_config: UnifiedEvolutionConfig,
        migrated_state: Optional[Dict]
    ) -> None:
        """Start new strategy"""
        logger.info(f"Starting strategy {new_strategy}")

        # Initialize new strategy
        # This would integrate with the actual evolution system
        # For now, just mark as running
        self.is_running = True

        # Load migrated state if available
        if migrated_state:
            await self._load_migrated_state(migrated_state)

    async def _capture_current_state(self) -> Dict:
        """Capture current evolutionary state"""
        state = {
            "strategy": self.current_strategy.value,
            "timestamp": datetime.utcnow().isoformat(),
            "population": [],
            "archive": [],
            "metrics": {},
        }

        # Derive real metrics from any state available in the current run when
        # the actual evolution system has populated it (fitness history etc.).
        current = self.current_state or {}
        fitness_history = current.get("fitness_history")
        if fitness_history:
            metrics = compute_adaptive_metrics(
                fitness_history,
                current.get("population_scores"),
            )
            state["metrics"] = {
                "stagnation_index": metrics.stagnation_index,
                "diversity": metrics.diversity,
                "improvement_rate": metrics.improvement_rate,
                "stagnation_generations": metrics.stagnation_generations,
            }

        return state

    def select_strategy(
        self,
        stagnation_index: float,
        diversity: float = 0.0,
        iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select an evolutionary strategy / parameters for the next phase.

        Thin wrapper around :func:`select_strategy` using the switcher's
        current strategy as the baseline.
        """
        return select_strategy(
            stagnation_index=stagnation_index,
            diversity=diversity,
            iteration=iteration,
            current_strategy=self.current_strategy,
        )

    async def _load_migrated_state(self, state: Dict) -> None:
        """Load migrated state into new strategy"""
        # This would integrate with the actual evolution system
        logger.info(f"Loading migrated state with {len(state.get('population', []))} individuals")

    def _can_continue_without_state(self) -> bool:
        """Check if we can continue without state migration"""
        # Some switches are safe even if state migration fails
        safe_transitions = [
            (SystemMode.OPENEVOLVE, SystemMode.QD),
            (SystemMode.OPENEVOLVE, SystemMode.MO),
            (SystemMode.QD, SystemMode.MO),
        ]

        for from_strat, to_strat in safe_transitions:
            if self.current_strategy == from_strat:
                return True

        return False

    async def _rollback_strategy(self) -> bool:
        """Rollback to previous strategy"""
        if not self.strategy_history:
            logger.error("No previous strategy to rollback to")
            return False

        # Get last successful switch
        last_switch = self.strategy_history[-1]

        try:
            # Switch back
            await self._stop_current_strategy()
            await self._start_new_strategy(
                last_switch.from_strategy,
                None,  # Would need to store old config
                None
            )

            self.current_strategy = last_switch.from_strategy
            logger.info(f"Rolled back to {last_switch.from_strategy}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_switch_history(self) -> List[StrategySwitchRecord]:
        """Get history of strategy switches"""
        return self.strategy_history.copy()

    def get_current_strategy(self) -> SystemMode:
        """Get current active strategy"""
        return self.current_strategy


class StateMigrator:
    """
    Migrate state between different evolutionary strategies

    Handles conversion of population, archive, and learned parameters
    between different algorithmic approaches.
    """

    async def migrate(
        self,
        from_strategy: SystemMode,
        to_strategy: SystemMode,
        current_state: Dict
    ) -> Optional[Dict]:
        """
        Migrate state from one strategy to another

        Args:
            from_strategy: Source strategy
            to_strategy: Target strategy
            current_state: Current state to migrate

        Returns:
            Migrated state or None if migration not possible
        """
        logger.info(f"Migrating state from {from_strategy} to {to_strategy}")

        # Extract compatible state
        compatible_state = self._extract_compatible_state(
            from_strategy,
            to_strategy,
            current_state
        )

        if not compatible_state:
            logger.warning("No compatible state found for migration")
            return None

        # Transform to target format
        migrated_state = self._transform_state(
            compatible_state,
            from_strategy,
            to_strategy
        )

        logger.info(f"State migration completed: {len(migrated_state)} keys")
        return migrated_state

    def _extract_compatible_state(
        self,
        from_strategy: SystemMode,
        to_strategy: SystemMode,
        current_state: Dict
    ) -> Dict:
        """
        Extract state that can be migrated between strategies

        Some components are universal, others are strategy-specific:
        - Universal: best solutions, knowledge artifacts, fitness history
        - Conditional: population archive (QD), Pareto front (MO)
        """
        compatible = {}

        # Always migrate best solutions
        if "best_solutions" in current_state:
            compatible["best_solutions"] = current_state["best_solutions"]

        # Always migrate knowledge artifacts
        if "artifacts" in current_state:
            compatible["artifacts"] = current_state["artifacts"]

        # Always migrate fitness history
        if "fitness_history" in current_state:
            compatible["fitness_history"] = current_state["fitness_history"]

        # Conditionally migrate population
        if self._both_use_population(from_strategy, to_strategy):
            if "population" in current_state:
                compatible["population"] = current_state["population"]

        # Conditionally migrate archive
        if self._both_use_archive(from_strategy, to_strategy):
            if "archive" in current_state:
                compatible["archive"] = current_state["archive"]

        return compatible

    def _both_use_population(
        self,
        from_strategy: SystemMode,
        to_strategy: SystemMode
    ) -> bool:
        """Check if both strategies use population-based evolution"""
        population_based = [
            SystemMode.OPENEVOLVE,
            SystemMode.QD,
            SystemMode.MO,
            SystemMode.ADVERSARIAL
        ]
        return from_strategy in population_based and to_strategy in population_based

    def _both_use_archive(
        self,
        from_strategy: SystemMode,
        to_strategy: SystemMode
    ) -> bool:
        """Check if both strategies use archives"""
        archive_based = [SystemMode.QD, SystemMode.MO]
        return from_strategy in archive_based and to_strategy in archive_based

    def _transform_state(
        self,
        state: Dict,
        from_strategy: SystemMode,
        to_strategy: SystemMode
    ) -> Dict:
        """
        Transform state to target format

        Handles conversion of data structures between different
        algorithmic representations.
        """
        transformed = copy.deepcopy(state)

        # Transform population representation
        if "population" in transformed:
            transformed["population"] = self._transform_population(
                transformed["population"],
                from_strategy,
                to_strategy
            )

        # Transform archive representation
        if "archive" in transformed:
            transformed["archive"] = self._transform_archive(
                transformed["archive"],
                from_strategy,
                to_strategy
            )

        return transformed

    def _transform_population(
        self,
        population: List,
        from_strategy: SystemMode,
        to_strategy: SystemMode
    ) -> List:
        """Transform population to target format"""
        # Most population formats are compatible
        # Main differences are in fitness representation
        return population

    def _transform_archive(
        self,
        archive: Dict,
        from_strategy: SystemMode,
        to_strategy: SystemMode
    ) -> Dict:
        """Transform archive to target format"""
        # QD uses grid-based archives
        # MO uses Pareto front archives
        # Need to restructure if switching between these

        if from_strategy == SystemMode.QD and to_strategy == SystemMode.MO:
            # Convert grid archive to Pareto front
            return self._grid_to_pareto(archive)

        elif from_strategy == SystemMode.MO and to_strategy == SystemMode.QD:
            # Convert Pareto front to grid archive
            return self._pareto_to_grid(archive)

        return archive

    def _grid_to_pareto(self, grid_archive: Dict) -> Dict:
        """Convert QD grid archive to MO Pareto front"""
        # Extract all solutions from grid cells
        pareto_front = {
            "solutions": [],
            "objectives": []
        }

        for cell_key, cell_value in grid_archive.items():
            if "solution" in cell_value:
                pareto_front["solutions"].append(cell_value["solution"])
            if "objectives" in cell_value:
                pareto_front["objectives"].extend(cell_value["objectives"])

        return pareto_front

    def _pareto_to_grid(self, pareto_front: Dict) -> Dict:
        """Convert MO Pareto front to QD grid archive"""
        # Distribute Pareto solutions across grid
        grid_archive = {}

        for i, solution in enumerate(pareto_front.get("solutions", [])):
            grid_key = f"cell_{i % 100}"  # Simple distribution
            grid_archive[grid_key] = {
                "solution": solution,
                "fitness": solution.get("fitness", 0.0)
            }

        return grid_archive


def select_strategy(
    stagnation_index: float,
    diversity: float = 0.0,
    iteration: Optional[int] = None,
    current_strategy: Optional[SystemMode] = None,
) -> Dict[str, Any]:
    """
    Dynamically select an evolutionary strategy and its parameters.

    This is deterministic and dependency-free: given the same inputs it always
    returns the same result, so it can be used safely mid-run.

    Inputs:
        stagnation_index: adaptive metric in [0, 1] (higher = more stagnation).
            Produced by ``config_metrics.compute_adaptive_metric``.
        diversity: coefficient of variation of the current population (0..inf,
            typically 0..1 for normalized scores). Higher = more spread out.
        iteration: current generation index (kept for call-site symmetry).
        current_strategy: the strategy currently in use (defaults to OPENEVOLVE).

    Logic:
        - Exploration pressure scales linearly with ``stagnation_index``:
          more stagnation -> higher mutation rate, lower selection pressure.
        - When stagnated *and* the population has collapsed to low diversity,
          switch to a diversity-seeking mode (QD, then MO) to escape the local
          optimum. Otherwise keep the current strategy and exploit progress.

    Returns:
        A concrete strategy/parameter dict with keys: ``strategy`` (SystemMode),
        ``mutation_rate``, ``crossover_rate``, ``selection_pressure``,
        ``exploration`` and ``reason``.
    """
    stagnation_index = max(0.0, min(1.0, float(stagnation_index)))
    diversity = max(0.0, float(diversity))
    exploration = stagnation_index

    mutation_rate = 0.05 + 0.45 * exploration
    crossover_rate = 0.9 - 0.3 * exploration
    selection_pressure = 0.5 + 0.5 * (1.0 - exploration)

    if current_strategy is None:
        current_strategy = SystemMode.OPENEVOLVE
    chosen = current_strategy

    reasons = []
    if stagnation_index >= 0.5:
        reasons.append("stagnation detected; increasing exploration")
    else:
        reasons.append("progress detected; exploiting")

    if stagnation_index >= 0.6 and diversity < 0.15:
        if current_strategy == SystemMode.OPENEVOLVE:
            chosen = SystemMode.QD
            reasons.append("low diversity under stagnation -> switch to QD")
        elif current_strategy == SystemMode.QD:
            chosen = SystemMode.MO
            reasons.append("low diversity under stagnation -> broaden to MO")

    return {
        "strategy": chosen,
        "mutation_rate": round(mutation_rate, 4),
        "crossover_rate": round(crossover_rate, 4),
        "selection_pressure": round(selection_pressure, 4),
        "exploration": round(exploration, 4),
        "reason": "; ".join(reasons),
    }
