#!/usr/bin/env python3
"""
Unified Training CLI for Core-Project ACE.

This script provides a comprehensive command-line interface for training
ACE (Agentic Context Engine) models across different tasks and benchmarks.

Supports three training modes:
- offline: Traditional training with validation splits
- online: Sequential adaptation on test data
- eval_only: Evaluation without training

Example:
    # Offline training with validation
    python train_ace.py --task finance --mode offline \\
        --data-dir ./data/finance --epochs 3

    # Online adaptation
    python train_ace.py --task finance --mode online \\
        --data-dir ./data/finance --initial-skillbook ./skillbook.json

    # Evaluation only
    python train_ace.py --task finance --mode eval_only \\
        --data-dir ./data/finance --initial-skillbook ./skillbook.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ACE core imports
from ace import (
    Agent,
    Reflector,
    SkillManager,
    OfflineACE,
    OnlineACE,
    Skillbook,
    Sample,
    SimpleEnvironment,
    TaskEnvironment,
    EnvironmentResult,
)

# LLM client
try:
    from ace.llm_providers import LiteLLMClient
except ImportError:
    LiteLLMClient = None
    print("Warning: LiteLLM not available. Install with: pip install ace-framework[all]")

# Benchmark processors
try:
    from benchmarks.processors import get_processor
    from benchmarks.environments import (
        FiNEREnvironment,
        XBRLMathEnvironment,
        AppWorldEnvironment,
    )
except ImportError:
    get_processor = None

# Suppress LiteLLM debug messages
try:
    import litellm

    litellm.suppress_debug_info = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_ace.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for ACE training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., finance, finer_ord, xbrl_math, appworld)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "online", "eval_only"],
        help=(
            "Training mode: "
            "'offline' for training with validation, "
            "'online' for sequential adaptation on test set, "
            "'eval_only' for testing only"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data files (train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--initial-skillbook",
        type=str,
        default=None,
        help="Path to initial skillbook file (JSON format)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for LiteLLM (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for LLM responses (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--max-reflection-rounds",
        type=int,
        default=3,
        help="Maximum reflection rounds for incorrect answers (default: 3)",
    )
    parser.add_argument(
        "--curator-frequency",
        type=int,
        default=10,
        help="Run curator/skill manager every N steps (default: 10)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=50,
        help="Evaluate on validation set every N steps (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N successful samples (default: 100)",
    )

    # System configuration
    parser.add_argument(
        "--skillbook-budget",
        type=int,
        default=80000,
        help="Token budget for skillbook (default: 80000)",
    )
    parser.add_argument(
        "--test-workers",
        type=int,
        default=4,
        help="Number of parallel workers for testing (default: 4)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ace_output",
        help="Directory to save results and skillbooks (default: ./ace_output)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing results (default: timestamped)",
    )

    return parser.parse_args()


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON on line {line_num} of {file_path}: {e}"
                        )
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def load_data(
    data_dir: str, mode: str
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load training, validation, and test data.

    Args:
        data_dir: Directory containing data files
        mode: Training mode ('offline', 'online', or 'eval_only')

    Returns:
        Tuple of (train_samples, val_samples, test_samples)

    Raises:
        FileNotFoundError: If required files are missing
    """
    train_samples = []
    val_samples = []
    test_samples = []

    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        test_file = os.path.join(data_dir, "test.jsonl")
        if not os.path.exists(test_file):
            raise FileNotFoundError(
                f"{mode.upper()} mode requires test.jsonl in {data_dir}"
            )
        test_data = load_jsonl(test_file)
        test_samples = convert_to_samples(test_data)
        logger.info(f"{mode.upper()} mode: Loaded {len(test_samples)} test samples")

    # For offline mode, load train, val, and optionally test
    else:  # mode == 'offline'
        train_file = os.path.join(data_dir, "train.jsonl")
        val_file = os.path.join(data_dir, "val.jsonl")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"OFFLINE mode requires train.jsonl in {data_dir}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"OFFLINE mode requires val.jsonl in {data_dir}")

        train_data = load_jsonl(train_file)
        val_data = load_jsonl(val_file)
        train_samples = convert_to_samples(train_data)
        val_samples = convert_to_samples(val_data)

        # Load test data if available
        test_file = os.path.join(data_dir, "test.jsonl")
        if os.path.exists(test_file):
            test_data = load_jsonl(test_file)
            test_samples = convert_to_samples(test_data)

        logger.info(
            f"OFFLINE mode: Loaded {len(train_samples)} train, "
            f"{len(val_samples)} val, {len(test_samples)} test samples"
        )

    return train_samples, val_samples, test_samples


def convert_to_samples(data: List[Dict[str, Any]]) -> List[Sample]:
    """
    Convert raw data to Sample objects.

    Args:
        data: List of data dictionaries

    Returns:
        List of Sample objects
    """
    samples = []
    for item in data:
        # Handle different data formats
        question = item.get("question", item.get("input", item.get("prompt", "")))
        context = item.get("context", item.get("passage", ""))
        ground_truth = item.get("ground_truth", item.get("answer", item.get("output", "")))
        metadata = item.get("metadata", {})

        sample = Sample(
            question=question,
            context=context,
            ground_truth=ground_truth,
            metadata=metadata,
        )
        samples.append(sample)

    return samples


def load_or_create_skillbook(
    skillbook_path: Optional[str], budget: int
) -> Skillbook:
    """
    Load existing skillbook or create new one.

    Args:
        skillbook_path: Path to existing skillbook (optional)
        budget: Token budget for skillbook

    Returns:
        Skillbook instance
    """
    if skillbook_path and os.path.exists(skillbook_path):
        logger.info(f"Loading skillbook from {skillbook_path}")
        try:
            skillbook = Skillbook.from_file(skillbook_path)
            logger.info(f"Loaded {len(skillbook.skills)} skills")
            return skillbook
        except Exception as e:
            logger.warning(f"Failed to load skillbook: {e}. Creating new one.")

    logger.info("Creating new skillbook")
    return Skillbook(token_budget=budget)


def get_environment(task: str) -> TaskEnvironment:
    """
    Get appropriate environment for task evaluation.

    Args:
        task: Task name

    Returns:
        TaskEnvironment instance
    """
    # Use specialized environments if available
    if task == "finer_ord":
        try:
            return FiNEREnvironment()
        except Exception:
            logger.warning("FiNEREnvironment not available, using SimpleEnvironment")

    elif task == "xbrl_math":
        try:
            return XBRLMathEnvironment()
        except Exception:
            logger.warning("XBRLMathEnvironment not available, using SimpleEnvironment")

    elif task == "appworld":
        try:
            return AppWorldEnvironment()
        except Exception:
            logger.warning("AppWorldEnvironment not available, using SimpleEnvironment")

    # Default to simple environment
    return SimpleEnvironment()


def setup_output_directory(
    output_dir: str, experiment_name: Optional[str]
) -> tuple[str, str]:
    """
    Setup output directory for results.

    Args:
        output_dir: Base output directory
        experiment_name: Optional experiment name

    Returns:
        Tuple of (output_path, results_dir)
    """
    # Create experiment name if not provided
    if not experiment_name:
        experiment_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_path = os.path.join(output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Output directory: {output_path}")
    return output_path, output_path


def save_results(
    results: List[Any],
    skillbook: Skillbook,
    output_dir: str,
    mode: str,
    task: str,
) -> None:
    """
    Save training results and skillbook.

    Args:
        results: Training results
        skillbook: Final skillbook
        output_dir: Output directory path
        mode: Training mode
        task: Task name
    """
    # Save skillbook
    skillbook_path = os.path.join(output_dir, f"{task}_skillbook.json")
    skillbook.save_to_file(skillbook_path)
    logger.info(f"Saved skillbook to {skillbook_path}")

    # Save results summary
    summary_path = os.path.join(output_dir, "results_summary.json")

    # Calculate metrics
    total_samples = len(results)
    if total_samples > 0:
        correct = sum(
            1 for r in results if hasattr(r, "environment_result") and r.environment_result
        )
        accuracy = correct / total_samples if total_samples > 0 else 0.0
    else:
        correct = 0
        accuracy = 0.0

    summary = {
        "task": task,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": total_samples,
        "correct": correct,
        "accuracy": accuracy,
        "skillbook_size": len(skillbook.skills),
        "skillbook_tokens": skillbook.token_count(),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved results summary to {summary_path}")
    logger.info(f"Final accuracy: {accuracy:.2%}")


def run_offline_training(
    adapter: OfflineACE,
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    environment: TaskEnvironment,
    args: argparse.Namespace,
) -> List[Any]:
    """
    Run offline training with validation.

    Args:
        adapter: OfflineACE adapter instance
        train_samples: Training data
        val_samples: Validation data
        test_samples: Test data
        environment: Task environment
        args: Command line arguments

    Returns:
        Training results
    """
    logger.info("Starting offline training...")

    # Train on training set
    results = adapter.run(
        samples=train_samples,
        environment=environment,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.output_dir,
    )

    # Evaluate on validation set
    if val_samples:
        logger.info(f"Evaluating on {len(val_samples)} validation samples...")
        val_results = adapter.run(
            samples=val_samples,
            environment=environment,
            epochs=1,
        )
        logger.info("Validation evaluation complete")

    # Evaluate on test set
    if test_samples:
        logger.info(f"Evaluating on {len(test_samples)} test samples...")
        test_results = adapter.run(
            samples=test_samples,
            environment=environment,
            epochs=1,
        )
        logger.info("Test evaluation complete")

    return results


def run_online_adaptation(
    adapter: OnlineACE,
    test_samples: List[Sample],
    environment: TaskEnvironment,
    args: argparse.Namespace,
) -> List[Any]:
    """
    Run online adaptation on test data.

    Args:
        adapter: OnlineACE adapter instance
        test_samples: Test data
        environment: Task environment
        args: Command line arguments

    Returns:
        Training results
    """
    logger.info("Starting online adaptation...")

    results = adapter.run(
        samples=test_samples,
        environment=environment,
    )

    return results


def run_evaluation_only(
    adapter: OfflineACE,
    test_samples: List[Sample],
    environment: TaskEnvironment,
    args: argparse.Namespace,
) -> List[Any]:
    """
    Run evaluation without training.

    Args:
        adapter: OfflineACE adapter instance
        test_samples: Test data
        environment: Task environment
        args: Command line arguments

    Returns:
        Evaluation results
    """
    logger.info("Starting evaluation-only mode...")

    results = adapter.run(
        samples=test_samples,
        environment=environment,
        epochs=1,
    )

    return results


def main() -> None:
    """
    Main execution function.

    Parses arguments, sets up components, runs training, and saves results.
    """
    # Parse arguments
    args = parse_args()

    logger.info("=" * 80)
    logger.info("ACE UNIFIED TRAINING CLI")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 80)

    # Setup output directory
    output_dir, experiment_dir = setup_output_directory(
        args.output_dir, args.experiment_name
    )

    # Load data
    try:
        train_samples, val_samples, test_samples = load_data(args.data_dir, args.mode)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Create or load skillbook
    skillbook = load_or_create_skillbook(args.initial_skillbook, args.skillbook_budget)

    # Initialize LLM client
    if LiteLLMClient is None:
        logger.error("LiteLLM client not available. Install with: pip install ace-framework[all]")
        sys.exit(1)

    llm_client = LiteLLMClient(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=120,
    )
    logger.info(f"Initialized LLM client: {args.model}")

    # Create ACE components
    agent = Agent(llm=llm_client)
    reflector = Reflector(llm=llm_client)
    skill_manager = SkillManager(llm=llm_client)

    logger.info("Created Agent, Reflector, and SkillManager")

    # Get environment for task
    environment = get_environment(args.task)
    logger.info(f"Using environment: {environment.__class__.__name__}")

    # Run appropriate mode
    try:
        if args.mode == "offline":
            adapter = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            results = run_offline_training(
                adapter,
                train_samples,
                val_samples,
                test_samples,
                environment,
                args,
            )

        elif args.mode == "online":
            adapter = OnlineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            results = run_online_adaptation(
                adapter,
                test_samples,
                environment,
                args,
            )

        elif args.mode == "eval_only":
            adapter = OfflineACE(
                skillbook=skillbook,
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
            )

            results = run_evaluation_only(
                adapter,
                test_samples,
                environment,
                args,
            )

        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    # Save results
    try:
        save_results(results, skillbook, experiment_dir, args.mode, args.task)
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
