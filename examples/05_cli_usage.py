"""
CLI Usage Example - Command-Line Interface

This example shows how to use OpenEvolve from the command line.

Problem: Optimize algorithm parameters
"""

# EVOLVE-BLOCK-START
def algorithm_parameters():
    """Return optimized parameters for an algorithm"""
    # Initial parameters
    learning_rate = 0.01
    batch_size = 32
    epochs = 10

    return {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
# EVOLVE-BLOCK-END


"""
COMMAND-LINE USAGE:
------------------

1. Basic usage:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py
   ```

2. Specify number of iterations:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py --iterations 50
   ```

3. Set output directory:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py --output my_results
   ```

4. Use custom config file:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py --config config.yaml
   ```

5. Set target score (early stopping):
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py --target-score 0.95
   ```

6. Adjust logging level:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py --log-level DEBUG
   ```

7. Resume from checkpoint:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py \\
       --checkpoint my_results/checkpoints/checkpoint_25
   ```

8. Override LLM settings:
   ```bash
   openevolve algorithm_parameters.py algo_evaluator.py \\
       --api-base https://api.openai.com/v1 \\
       --primary-model gpt-4
   ```

FULL EXAMPLE:
------------
```bash
# Run evolution with all options
openevolve algorithm_parameters.py algo_evaluator.py \\
    --config my_config.yaml \\
    --iterations 100 \\
    --output results/experiment_1 \\
    --target-score 0.90 \\
    --log-level INFO
```

WHAT HAPPENS:
------------
1. OpenEvolve loads the initial program
2. Evaluates it with algo_evaluator.py
3. Generates variations using LLM
4. Tests each variation
5. Keeps best performers
6. Repeats for specified iterations
7. Saves best program to output directory

OUTPUT STRUCTURE:
----------------
```
results/experiment_1/
├── best/
│   ├── best_program.py          # Best evolved code
│   └── best_program_info.json   # Metrics and metadata
├── checkpoints/
│   ├── checkpoint_10/
│   ├── checkpoint_20/
│   └── ...
└── logs/
    └── openevolve_YYYYMMDD_HHMMSS.log
```

CHECKPOINTING:
-------------
Checkpoints are saved automatically every N iterations (default: 100).

To resume from a checkpoint:
```bash
openevolve program.py evaluator.py \\
    --checkpoint results/checkpoints/checkpoint_50
```

This resumes from iteration 50 and continues to the max iterations.
"""
