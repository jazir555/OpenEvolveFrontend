"""
Evaluator for Algorithm Parameters Example

Evaluates how well the algorithm parameters perform.
"""

import sys
import importlib.util
import time


def evaluate(program_path):
    """
    Evaluate algorithm parameters.

    Simulates training with different parameter settings.
    Better parameters = faster convergence + better final accuracy.
    """
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "Failed to load"}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Load error: {e}"}

    if not hasattr(module, 'algorithm_parameters'):
        return {"combined_score": 0.0, "error": "No algorithm_parameters function"}

    try:
        params = module.algorithm_parameters()

        # Extract parameters
        lr = params.get('learning_rate', 0.01)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 10)

        # Simulate training (this would be real training in practice)
        # Better parameters:
        # - Learning rate: 0.001 - 0.01 is good
        # - Batch size: 32 - 128 is reasonable
        # - Epochs: More is better but diminishing returns

        # Score learning rate (optimal around 0.001-0.01)
        if 0.001 <= lr <= 0.01:
            lr_score = 1.0
        elif lr < 0.001:
            lr_score = lr / 0.001  # Too small
        else:
            lr_score = max(0, 1.0 - (lr - 0.01) * 10)  # Too large

        # Score batch size (32-128 is good)
        if 32 <= batch_size <= 128:
            batch_score = 1.0
        else:
            batch_score = max(0, 1.0 - abs(batch_size - 64) / 100)

        # Score epochs (more is better, but diminishing)
        epoch_score = min(1.0, epochs / 50.0)

        # Combined score
        combined = (lr_score * 0.5) + (batch_score * 0.3) + (epoch_score * 0.2)

        return {
            "combined_score": combined,
            "lr_score": lr_score,
            "batch_score": batch_score,
            "epoch_score": epoch_score,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs
        }

    except Exception as e:
        return {"combined_score": 0.0, "error": f"Eval error: {e}"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python algo_evaluator.py <program_path>")
        sys.exit(1)

    metrics = evaluate(sys.argv[1])

    print("Algorithm Parameters Evaluation:")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")
    print(f"  Learning Rate: {metrics.get('learning_rate', 0):.6f} (score: {metrics.get('lr_score', 0):.2f})")
    print(f"  Batch Size: {metrics.get('batch_size', 0)} (score: {metrics.get('batch_score', 0):.2f})")
    print(f"  Epochs: {metrics.get('epochs', 0)} (score: {metrics.get('epoch_score', 0):.2f})")

    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")
