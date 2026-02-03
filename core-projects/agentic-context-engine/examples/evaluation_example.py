"""
Example: Parallel Evaluation Framework for ACE Agents

This example demonstrates how to use the evaluation framework to assess
agent performance on datasets with parallel execution.
"""

from typing import Any, Dict

from ace import Agent, Skillbook, LiteLLMClient
from ace.evaluation import evaluate_dataset, evaluate_single_sample, EvaluationResult


def simple_answer_checker(prediction: str, ground_truth: str) -> bool:
    """
    Simple answer checker that compares lowercase strings.

    In production, you might use:
    - Exact match for factual questions
    - Semantic similarity with embeddings
    - Custom logic for different question types
    """
    return prediction.strip().lower() == ground_truth.strip().lower()


def main():
    """Run evaluation examples."""
    print("=" * 60)
    print("ACE Parallel Evaluation Framework Examples")
    print("=" * 60)

    # Initialize agent and skillbook
    # Note: Set OPENAI_API_KEY environment variable before running
    try:
        llm = LiteLLMClient(model="gpt-3.5-turbo")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        return

    agent = Agent(llm)
    skillbook = Skillbook()

    # Example 1: Single sample evaluation
    print("\n--- Example 1: Single Sample Evaluation ---")
    sample = {
        "question": "What is the capital of France?",
        "context": "Geography",
        "target": "Paris",
    }

    result = evaluate_single_sample(
        index=0,
        sample=sample,
        agent=agent,
        skillbook=skillbook,
        answer_checker=simple_answer_checker,
    )

    print(f"Question: {sample['question']}")
    print(f"Prediction: {result.prediction}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Correct: {result.is_correct}")
    print(f"Skill IDs Used: {result.skill_ids_used}")

    # Example 2: Dataset evaluation with parallel execution
    print("\n--- Example 2: Parallel Dataset Evaluation ---")
    samples = [
        {
            "question": "What is 2 + 2?",
            "context": "Math",
            "target": "4",
        },
        {
            "question": "What is the capital of Japan?",
            "context": "Geography",
            "target": "Tokyo",
        },
        {
            "question": "What color is the sky?",
            "context": "General knowledge",
            "target": "Blue",
        },
        {
            "question": "What is 10 * 10?",
            "context": "Math",
            "target": "100",
        },
    ]

    results = evaluate_dataset(
        samples=samples,
        agent=agent,
        skillbook=skillbook,
        answer_checker=simple_answer_checker,
        max_workers=2,  # Adjust based on your API rate limits
        show_progress=True,
    )

    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']}/{results['total']}")

    # Show errors if any
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            if "prediction" in error:
                print(f"  - Sample {error['index']}: "
                      f"Predicted '{error['prediction']}', "
                      f"Expected '{error['ground_truth']}'")
            else:
                print(f"  - Sample {error['index']}: {error.get('error', 'Unknown error')}")

    # Example 3: Custom answer checker
    print("\n--- Example 3: Custom Answer Checker ---")

    def flexible_checker(prediction: str, ground_truth: str) -> bool:
        """More flexible checking that allows numeric answers."""
        pred_clean = prediction.strip().lower()
        truth_clean = ground_truth.strip().lower()

        # Exact match
        if pred_clean == truth_clean:
            return True

        # Handle numeric answers
        try:
            pred_num = float(pred_clean)
            truth_num = float(truth_clean)
            return abs(pred_num - truth_num) < 0.01
        except ValueError:
            pass

        return False

    math_samples = [
        {"question": "What is 5 * 5?", "context": None, "target": "25"},
        {"question": "What is 100 / 4?", "context": None, "target": "25"},
    ]

    results = evaluate_dataset(
        samples=math_samples,
        agent=agent,
        skillbook=skillbook,
        answer_checker=flexible_checker,
        max_workers=2,
        show_progress=True,
    )

    print(f"\nMath Evaluation: {results['accuracy']:.2%} accuracy")


if __name__ == "__main__":
    main()
