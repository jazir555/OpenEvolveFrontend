# -*- coding: utf-8 -*-
"""
Finalizer component for OpenEvolve PES framework.

Provides finalization logic to generate summary results after evolution completes.
Adapted from LoongFlow.
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Finalizer(ABC):
    """
    Abstract interface for the Finalizer component.

    The Finalizer is responsible for generating the final result of the
    evolution process based on the state stored in the database.
    """

    @abstractmethod
    async def finalize(
        self, database: Any, start_time: int, was_interrupted: bool,
        total_cost: float, total_tokens: float
    ) -> Any:
        """
        Generate the final result of the evolution process.

        Args:
            database: The database containing the full history and state.
            start_time: The timestamp when the evolution process started.
            was_interrupted: A boolean indicating if the process was interrupted.
            total_cost: The total cost of the evolution process.
            total_tokens: The total tokens used during the evolution process.

        Returns:
            The final message or result object summarizing the outcome.
        """
        raise NotImplementedError


class PESFinalizer(Finalizer):
    """
    Default implementation of the Finalizer component for PES framework.

    Queries the database for the best result, constructs a structured result,
    and wraps it in a final message/report.
    """

    async def finalize(
        self,
        database: Any,
        start_time: int,
        was_interrupted: bool,
        total_cost: float,
        total_tokens: float
    ) -> Dict[str, Any]:
        """
        Generate final report from database state.

        Queries the database to find the best solution and its metadata,
        then constructs and returns a structured result dictionary.

        Args:
            database: The evolution database to query for results.
            start_time: The timestamp when the evolution process started.
            was_interrupted: A boolean indicating if the process was interrupted.
            total_cost: The total cost of the evolution process.
            total_tokens: The total tokens used during the evolution process.

        Returns:
            A dictionary containing the final evolution results.
        """
        logger.info("PES Finalizer: Generating final report from database.")

        end_time = int(time.time())
        cost_time = end_time - start_time
        status_prefix = "Process concluded."
        if was_interrupted:
            status_prefix = "Process was interrupted."

        try:
            memory_status = database.memory_status()
            global_status: Dict[str, Any] = memory_status.get("global_status", {})

            best_score = global_status.get("best_score")
            best_iteration = global_status.get("best_iteration")
            total_iterations = global_status.get("current_iteration", 0)

            if best_score is None or best_iteration is None:
                summary = {
                    "status": status_prefix,
                    "message": "No solution was successfully scored.",
                    "was_interrupted": was_interrupted,
                }
                logger.info(summary)
                return summary

            # Get best solutions
            best_solution_list = []
            best_evaluation_list = []
            best_solutions = database.get_best_solutions(top_k=3)
            if best_solutions and len(best_solutions) > 0:
                best_solution_list.append(best_solutions[0].get("solution", ""))
                best_evaluation_list.append(best_solutions[0].get("evaluation", ""))

            start_time_str = datetime.fromtimestamp(start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_time_str = datetime.fromtimestamp(end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            result = {
                "status": status_prefix,
                "best_score": best_score,
                "best_solution": best_solution_list,
                "evaluation": best_evaluation_list,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "cost_time": cost_time,
                "last_iteration": best_iteration,
                "total_iterations": total_iterations,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "was_interrupted": was_interrupted,
            }

            logger.info(
                f"{status_prefix}\n"
                f"Best score achieved: {result['best_score']}\n"
                f"Found in iteration: {result['last_iteration']}\n"
                f"Total iterations: {result['total_iterations']}\n"
                f"Total cost time: {result['cost_time']} seconds\n"
                f"Total tokens: {result['total_tokens']}\n"
                f"Total cost: {result['total_cost']}"
            )

            return result

        except Exception as e:
            error_result = {
                "status": status_prefix,
                "error": str(e),
                "was_interrupted": was_interrupted,
            }
            logger.error(f"Error during finalization: {e}", exc_info=True)
            return error_result
