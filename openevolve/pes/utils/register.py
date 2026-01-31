#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker registry for OpenEvolve PES framework.

Provides registration and retrieval mechanism for Planner, Executor, and Summary workers.
Adapted from LoongFlow.
"""

import abc
import inspect
from typing import Any

PLANNER = "planner"
EXECUTOR = "executor"
SUMMARY = "summary"

# Planner registry to hold implementations
_planner_registry = {}

# Executor registry to hold implementations
_executor_registry = {}

# Summary registry to hold implementations
_summary_registry = {}


class Worker(abc.ABC):
    """
    Abstract base class for PES workers.

    All workers (Planner, Executor, Summary) must inherit from this class
    and implement the run method.
    """

    @abc.abstractmethod
    async def run(self, context: Any, message: Any | None) -> Any:
        """
        Execute the worker's logic.

        Args:
            context: Runtime context containing task information, iteration state, etc.
            message: Input message from previous stage (Planner -> Executor -> Summary)

        Returns:
            Output message to pass to next stage
        """
        pass


def register_worker(name: str, phase: str, worker_class: type):
    """
    Register a worker implementation.

    Args:
        name: The name to identify the worker.
        phase: The phase to identify the worker ("planner", "executor", or "summary").
        worker_class: The class of the worker that extends the Worker interface.

    Raises:
        ValueError: If worker_class is not a subclass of Worker or phase is invalid.
    """
    if not issubclass(worker_class, Worker):
        raise ValueError(f"{worker_class.__name__} must be a subclass of Worker.")

    phase = phase.lower()
    if phase == PLANNER:
        _planner_registry[name] = worker_class
    elif phase == EXECUTOR:
        _executor_registry[name] = worker_class
    elif phase == SUMMARY:
        _summary_registry[name] = worker_class
    else:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of [{PLANNER}, {EXECUTOR}, {SUMMARY}]."
        )


def get_worker(name: str, phase: str, **kwargs) -> Worker:
    """
    Retrieve a registered worker (Planner, Executor, or Summary) by name.

    This function looks up the worker in the corresponding phase registry
    and instantiates it using the provided keyword arguments.

    Args:
        name: The name of the worker to retrieve.
        phase: The phase of the worker. Must be one of:
            - "planner"
            - "executor"
            - "summary"
        **kwargs: Arbitrary keyword arguments that will be passed to the
            worker's constructor. Common examples include:
            - config: configuration object for the worker
            - db: database or storage object (planner/summary)
            - evaluator: evaluator instance (executor)

    Returns:
        The instantiated worker.

    Raises:
        KeyError: If the worker is not registered in the specified phase.
    """
    _worker_class = None
    phase = phase.lower()
    if phase == PLANNER:
        _worker_class = _planner_registry[name]
    elif phase == EXECUTOR:
        _worker_class = _executor_registry[name]
    elif phase == SUMMARY:
        _worker_class = _summary_registry[name]

    if _worker_class is None:
        raise KeyError(f"Worker '{name}' not found in Phase '{phase}'.")

    sig = inspect.signature(_worker_class.__init__)

    class_params = set(sig.parameters.keys())
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in class_params
    }

    return _worker_class(**filtered_kwargs)
