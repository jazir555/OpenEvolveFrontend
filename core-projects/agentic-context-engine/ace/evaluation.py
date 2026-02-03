"""
Parallel evaluation framework for ACE agents.

This module provides tools for evaluating agent performance on datasets
with parallel execution and comprehensive error tracking.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single sample.

    Attributes:
        index: Index of the sample in the dataset
        prediction: The agent's predicted answer
        ground_truth: The correct answer from the dataset
        is_correct: Whether the prediction matches ground truth
        skill_ids_used: List of skill IDs cited in the agent's reasoning
        error: Error message if evaluation failed, None otherwise
    """

    index: int
    prediction: str
    ground_truth: str
    is_correct: bool
    skill_ids_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def __repr__(self) -> str:
        """Concise representation showing correctness status."""
        status = "✓" if self.is_correct else "✗"
        return f"EvaluationResult(index={self.index}, status={status})"


def evaluate_single_sample(
    index: int,
    sample: Dict[str, Any],
    agent: Any,  # Agent type to avoid circular import
    skillbook: Any,  # Skillbook type to avoid circular import
    answer_checker: Callable[[str, str], bool],
    **kwargs: Any,
) -> EvaluationResult:
    """
    Evaluate a single test sample.

    This function generates an answer using the agent and validates it
    against the ground truth using the provided answer checker.

    Args:
        index: Index of the sample in the dataset
        sample: Dictionary containing the sample data with keys:
            - 'question': The question to answer
            - 'context': Optional context for the question
            - 'target' or 'ground_truth': The correct answer
        agent: Agent instance with generate() method
        skillbook: Skillbook instance containing strategies
        answer_checker: Callable that takes (prediction, ground_truth) and
                       returns True if the answer is correct
        **kwargs: Additional arguments passed to agent.generate()

    Returns:
        EvaluationResult with prediction, correctness, and error status

    Example:
        >>> from ace import Agent, Skillbook
        >>> def simple_checker(pred, truth):
        ...     return pred.strip().lower() == truth.strip().lower()
        >>> result = evaluate_single_sample(
        ...     index=0,
        ...     sample={'question': 'What is 2+2?', 'context': None, 'target': '4'},
        ...     agent=agent,
        ...     skillbook=skillbook,
        ...     answer_checker=simple_checker
        ... )
        >>> print(result.is_correct)
        True
    """
    try:
        # Extract sample fields
        question = sample.get("question", "")
        context = sample.get("context")
        ground_truth = sample.get("target") or sample.get("ground_truth", "")

        if not question:
            return EvaluationResult(
                index=index,
                prediction="",
                ground_truth=ground_truth,
                is_correct=False,
                skill_ids_used=[],
                error="Sample missing 'question' field",
            )

        # Generate answer using agent
        output = agent.generate(
            question=question,
            context=context,
            skillbook=skillbook,
            **kwargs,
        )

        prediction = output.final_answer
        skill_ids = output.skill_ids if hasattr(output, "skill_ids") else []

        # Validate against ground truth
        is_correct = answer_checker(prediction, ground_truth)

        return EvaluationResult(
            index=index,
            prediction=prediction,
            ground_truth=ground_truth,
            is_correct=is_correct,
            skill_ids_used=skill_ids,
            error=None,
        )

    except Exception as e:
        # Handle any errors gracefully without failing the entire evaluation
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Error evaluating sample {index}: {error_msg}")

        return EvaluationResult(
            index=index,
            prediction="",
            ground_truth=sample.get("target") or sample.get("ground_truth", ""),
            is_correct=False,
            skill_ids_used=[],
            error=error_msg,
        )


def evaluate_dataset(
    samples: List[Dict[str, Any]],
    agent: Any,  # Agent type to avoid circular import
    skillbook: Any,  # Skillbook type to avoid circular import
    answer_checker: Callable[[str, str], bool],
    max_workers: int = 20,
    show_progress: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Parallel evaluation of a dataset using ThreadPoolExecutor.

    This function evaluates multiple samples in parallel, tracking accuracy
    and aggregating errors for incorrect predictions. Progress is shown
    every 50 samples by default.

    Args:
        samples: List of sample dictionaries, each containing:
            - 'question': The question to answer
            - 'context': Optional context for the question
            - 'target' or 'ground_truth': The correct answer
        agent: Agent instance with generate() method
        skillbook: Skillbook instance containing strategies
        answer_checker: Callable that takes (prediction, ground_truth) and
                       returns True if the answer is correct
        max_workers: Maximum number of parallel threads (default: 20)
        show_progress: Whether to print progress updates (default: True)
        **kwargs: Additional arguments passed to each agent.generate() call

    Returns:
        Dictionary with evaluation results:
            - 'accuracy': Overall accuracy (0.0 to 1.0)
            - 'correct': Number of correct predictions
            - 'total': Total number of samples evaluated
            - 'errors': List of error details for incorrect predictions
            - 'results': List of EvaluationResult objects (optional)

    Example:
        >>> from ace import Agent, Skillbook
        >>> samples = [
        ...     {'question': 'What is 2+2?', 'context': None, 'target': '4'},
        ...     {'question': 'What is 3+3?', 'context': None, 'target': '6'},
        ... ]
        >>> def simple_checker(pred, truth):
        ...     return pred.strip().lower() == truth.strip().lower()
        >>> results = evaluate_dataset(
        ...     samples=samples,
        ...     agent=agent,
        ...     skillbook=skillbook,
        ...     answer_checker=simple_checker,
        ...     max_workers=2
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        Accuracy: 100.00%
    """
    if show_progress:
        print(f"\n{'='*40}")
        print(f"EVALUATING DATASET - {len(samples)} samples, {max_workers} workers")
        print(f"{'='*40}")

    # Tracking variables
    correct = 0
    total = 0
    errors: List[Dict[str, Any]] = []
    all_results: List[EvaluationResult] = []

    # Prepare evaluation arguments for each sample
    eval_args = [
        (i, sample, agent, skillbook, answer_checker, kwargs)
        for i, sample in enumerate(samples)
    ]

    # Execute evaluations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_evaluate_wrapper, args): args[0]
            for args in eval_args
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_index), 1):
            result = future.result()

            if result.error:
                # Sample had an error during evaluation
                total += 1
                errors.append({
                    "index": result.index,
                    "error": result.error,
                    "ground_truth": result.ground_truth,
                })
            else:
                # Sample evaluated successfully
                total += 1
                if result.is_correct:
                    correct += 1
                else:
                    # Track incorrect predictions for analysis
                    errors.append({
                        "index": result.index,
                        "prediction": result.prediction,
                        "ground_truth": result.ground_truth,
                        "skill_ids_used": result.skill_ids_used,
                    })

            all_results.append(result)

            # Show progress every 50 samples
            if show_progress and i % 50 == 0:
                current_acc = correct / total if total > 0 else 0.0
                print(f"Progress: {i}/{len(samples)}, Accuracy: {current_acc:.3f}")

    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0

    if show_progress:
        print(f"\n{'='*40}")
        print(f"Final Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"{'='*40}\n")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "results": all_results,
    }


def _evaluate_wrapper(args: tuple) -> EvaluationResult:
    """
    Wrapper function for ThreadPoolExecutor.

    Unpacks arguments and calls evaluate_single_sample.
    """
    index, sample, agent, skillbook, answer_checker, kwargs = args
    return evaluate_single_sample(
        index=index,
        sample=sample,
        agent=agent,
        skillbook=skillbook,
        answer_checker=answer_checker,
        **kwargs,
    )
