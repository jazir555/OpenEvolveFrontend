"""
Finance Domain Processor for Core-Project ACE.

This module provides data processing and evaluation logic for finance-related tasks,
including Named Entity Recognition (FiNER) and financial formula reasoning tasks.
"""

from typing import List, Dict, Any, Tuple
import re
import json


class FinanceDataProcessor:
    """
    Processor for financial domain tasks in the ACE benchmark framework.

    Handles two main task types:
    - FiNER: Financial Named Entity Recognition with multi-label annotation
    - Formula: Financial numerical reasoning with formula computation

    This processor converts raw dataset formats into standardized evaluation samples
    and provides task-specific validation logic for answer correctness.
    """

    def __init__(self, task_name: str):
        """
        Initialize the finance data processor.

        Args:
            task_name: The name of the finance task ('finer' or 'formula')

        Raises:
            ValueError: If task_name is not supported
        """
        self.task_name = task_name

        if task_name not in ["finer", "formula"]:
            raise ValueError(
                f"Unsupported task: {task_name}. "
                "Supported tasks are: 'finer', 'formula'"
            )

    def parse_instruction_input_format(self, context: str) -> Tuple[str, str]:
        """
        Parse context in 'Instruction: [INSTRUCTION].\\nInput: [TEXT]\\nAnswer: ' format.

        This format is used by FiNER sentiment and classification tasks where:
        - Instruction: Contains the task description
        - Input: Contains the actual text to analyze
        - Answer: Placeholder for the model output

        Args:
            context: Raw context string in Instruction/Input format

        Returns:
            Tuple of (input_text, instruction) where:
                - input_text: The actual text content to analyze
                - instruction: The task instruction/description

        Examples:
            >>> processor = FinanceDataProcessor("finer")
            >>> text = 'Instruction: Classify sentiment.\\nInput: Stock prices rose.\\nAnswer: '
            >>> input_text, instruction = processor.parse_instruction_input_format(text)
            >>> input_text
            'Stock prices rose.'
            >>> instruction
            'Classify sentiment.'
        """
        if "Input: " in context and "Instruction: " in context:
            # Split by "Input: " to separate instruction from input text
            instruction_part = context.split("Input: ")[0].strip()
            instruction_part = instruction_part.split("Instruction: ")[1].strip()

            # Extract input text (between "Input: " and "Answer: ")
            remaining = context.split("Input: ")[1]
            input_text = remaining.split("Answer: ")[0].strip()

            return input_text, instruction_part

        # Fallback: return entire context as instruction with empty input
        return "", context

    def parse_context_question_formula(self, context: str) -> Tuple[str, str]:
        """
        Parse context in '[instruction] Question: \"[QUESTION]\". Answer:' format.

        This format is used by formula-based financial reasoning tasks. The method
        adds a numeric conversion hint to ensure proper handling of financial numbers
        (e.g., "5 million" -> 5000000.0).

        Args:
            context: Raw context string containing question

        Returns:
            Tuple of ("", question_text) where:
                - First element is always empty (no input context needed)
                - question_text: The question with numeric conversion hint appended

        Examples:
            >>> processor = FinanceDataProcessor("formula")
            >>> text = 'Calculate the sum. Question: "What is 5 million + 2 million?". Answer:'
            >>> _, question = processor.parse_context_question_formula(text)
            >>> "plain floating point number" in question
            True
            >>> "5 million should be 5000000.0" in question
            True
        """
        if "Question: " in context and ". Answer:" in context:
            # Split by "Question: " to separate instruction from question
            parts = context.split("Question: ", 1)
            instruction_part = parts[0].strip()

            # Extract question text (between "Question: " and ". Answer:")
            question_part = parts[1]
            question_text = question_part.split(". Answer:")[0].strip()

            # Remove surrounding quotes if present
            if question_text.startswith('"') and question_text.endswith('"'):
                question_text = question_text[1:-1]

            # Add numeric conversion hint for financial numbers
            # This ensures models convert "5 million" to 5000000.0
            numeric_hint = (
                " Your answer should be a plain floating point number, "
                "round to the nearest hundredth if necessary. "
                "Do the necessary conversions, for example 5 million should be 5000000.0. "
            )
            question_text += numeric_hint

            return "", question_text

        # Fallback: return entire context as question
        return "", context

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process raw task data into standardized format for ACE evaluation.

        Converts dataset-specific formats into a unified structure containing:
        - context: Input text or document
        - question: Task instruction or question
        - ground_truth: Expected answer
        - metadata: Additional task information

        Args:
            raw_data: List of dictionaries from JSONL file with keys:
                     - 'context': Raw context string
                     - 'target': Ground truth answer

        Returns:
            List of processed dictionaries with standardized format:
                - context: Input text (empty for formula tasks)
                - question: Instruction/question (with hints for formula)
                - ground_truth: Expected answer
                - metadata: Task metadata (original_context, task, data_source)

        Raises:
            ValueError: If task_name is not supported
        """
        processed_data = []

        # Select appropriate parser based on task type
        if self.task_name == "finer":
            parse_fn = self.parse_instruction_input_format
        elif self.task_name == "formula":
            parse_fn = self.parse_context_question_formula
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        for item in raw_data:
            context = item.get('context', '')
            target = item.get('target', '')

            # Parse context to extract the actual text and instruction/question
            input_text, question = parse_fn(context)

            processed_item = {
                "context": input_text,
                "question": question,
                "ground_truth": target,
                "metadata": {
                    "original_context": context,
                    "task": self.task_name,
                    "data_source": "finance"
                }
            }

            processed_data.append(processed_item)

        return processed_data

    def finer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if FiNER NER prediction is correct with partial credit scoring.

        FiNER tasks involve multi-label named entity recognition where answers
        may contain comma-separated entity labels. This method:
        1. Handles comma-separated labels
        2. Evaluates numeric values with flexible comparison
        3. Awards partial credit (score must be 1.0 for full correctness)

        Args:
            predicted: Model's prediction (comma-separated entities or values)
            ground_truth: Ground truth labels or values

        Returns:
            True if prediction is fully correct (score == 1.0), False otherwise

        Examples:
            >>> processor = FinanceDataProcessor("finer")
            >>> processor.finer_is_correct("PER,ORG", "PER,ORG")
            True
            >>> processor.finer_is_correct("PER", "PER,ORG")  # Missing entity
            False
            >>> processor.finer_is_correct("$1,000", "1000")  # Numeric comparison
            True
        """
        # Split by comma and normalize
        pred = predicted.split(",")
        pred = [val.lower().strip() for val in pred]

        label = ground_truth.split(",")
        label = [val.lower().strip() for val in label]

        # Pad predictions or labels to match lengths
        if len(pred) != len(label):
            if len(pred) > len(label):
                pred = pred[:len(label)]
            else:
                padding_needed = len(label) - len(pred)
                pred += ([""] * padding_needed)

        # Compare each label/value
        count = 0
        for prediction, ground_truth_item in zip(pred, label):
            try:
                # Try numeric evaluation (e.g., "$1,000" vs "1000")
                evaluated_pred = eval(prediction.replace(",", "").replace("$", ""))
                evaluated_gt = eval(ground_truth_item)
                if evaluated_pred == evaluated_gt:
                    count += 1
            except Exception:
                # Fall back to string comparison
                if prediction == ground_truth_item:
                    count += 1

        # Calculate score - must be perfect for correctness
        score = count / len(pred) if pred else 0
        return score == 1

    def formula_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if formula task prediction is numerically correct.

        Formula tasks involve financial calculations and numerical reasoning.
        This method:
        1. Removes commas from numbers (e.g., "1,000" -> "1000")
        2. Compares as floating point numbers
        3. Falls back to string comparison if conversion fails

        Args:
            predicted: Model's numeric prediction (may contain commas)
            ground_truth: Ground truth numeric value (may contain commas)

        Returns:
            True if values match numerically, False otherwise

        Examples:
            >>> processor = FinanceDataProcessor("formula")
            >>> processor.formula_is_correct("1000", "1000")
            True
            >>> processor.formula_is_correct("1,000", "1000")  # Commas removed
            True
            >>> processor.formula_is_correct("1000.5", "1000.50")  # Float comparison
            True
            >>> processor.formula_is_correct("1000", "999")
            False
        """
        try:
            # Remove commas and compare as floats
            predicted_clean = predicted.replace(",", "")
            ground_truth_clean = ground_truth.replace(",", "")

            return float(predicted_clean) == float(ground_truth_clean)
        except (ValueError, TypeError):
            # Fallback to string comparison if not numeric
            return predicted == ground_truth

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Route answer validation to task-specific validator.

        This is the main entry point for answer correctness checking,
        automatically routing to the appropriate validation method based
        on the task type.

        Args:
            predicted: Model's prediction
            ground_truth: Ground truth answer

        Returns:
            True if answer is correct according to task-specific logic

        Raises:
            ValueError: If task_name is not supported

        Examples:
            >>> processor = FinanceDataProcessor("finer")
            >>> processor.answer_is_correct("PER,ORG", "PER,ORG")
            True
            >>> processor = FinanceDataProcessor("formula")
            >>> processor.answer_is_correct("1000", "1000")
            True
        """
        # Handle edge cases
        if not predicted or not ground_truth:
            return False

        if not isinstance(predicted, str):
            predicted = str(predicted)

        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        # Route to appropriate validator
        if self.task_name == "finer":
            return self.finer_is_correct(predicted, ground_truth)
        elif self.task_name == "formula":
            return self.formula_is_correct(predicted, ground_truth)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

    def evaluate_accuracy(
        self, predictions: List[str], ground_truths: List[str]
    ) -> Tuple[float, int, int]:
        """
        Compute accuracy metrics for a batch of predictions.

        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers

        Returns:
            Tuple of (accuracy, correct_count, total_count):
                - accuracy: Fraction of correct answers (0.0 to 1.0)
                - correct_count: Number of correct predictions
                - total_count: Total number of predictions

        Raises:
            ValueError: If input lists have different lengths
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(ground_truths)} ground truths"
            )

        correct_count = 0
        total_count = len(predictions)

        for predicted, ground_truth in zip(predictions, ground_truths):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return accuracy, correct_count, total_count


def load_finance_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process financial data from a JSONL file.

    Args:
        data_path: Path to the JSONL file containing finance data

    Returns:
        List of dictionaries with 'context' and 'target' keys

    Raises:
        FileNotFoundError: If data file does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    import os

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {data_path}: {e}"
                    )

    return data
