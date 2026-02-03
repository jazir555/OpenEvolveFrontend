# Parallel Evaluation Framework

The evaluation framework provides tools for assessing ACE agent performance on datasets with parallel execution and comprehensive error tracking.

## Overview

The framework consists of three main components:

1. **EvaluationResult**: Dataclass containing results for a single sample
2. **evaluate_single_sample()**: Evaluate one test sample
3. **evaluate_dataset()**: Parallel evaluation of multiple samples

## Quick Start

```python
from ace import Agent, Skillbook, LiteLLMClient
from ace.evaluation import evaluate_dataset

# Initialize
llm = LiteLLMClient(model="gpt-3.5-turbo")
agent = Agent(llm)
skillbook = Skillbook()

# Define answer checker
def checker(pred, truth):
    return pred.strip().lower() == truth.strip().lower()

# Run evaluation
samples = [
    {"question": "What is 2+2?", "context": None, "target": "4"},
    {"question": "Capital of France?", "context": None, "target": "Paris"},
]

results = evaluate_dataset(
    samples=samples,
    agent=agent,
    skillbook=skillbook,
    answer_checker=checker,
    max_workers=2
)

print(f"Accuracy: {results['accuracy']:.2%}")
```

## Components

### EvaluationResult

Dataclass containing results for a single sample evaluation.

**Fields:**
- `index` (int): Sample index in dataset
- `prediction` (str): Agent's predicted answer
- `ground_truth` (str): Correct answer from dataset
- `is_correct` (bool): Whether prediction matches ground truth
- `skill_ids_used` (List[str]): Skill IDs cited in reasoning
- `error` (Optional[str]): Error message if evaluation failed

### evaluate_single_sample()

Evaluate a single test sample.

**Parameters:**
- `index` (int): Sample index
- `sample` (Dict): Sample dictionary with keys:
  - `question` (str): The question
  - `context` (Optional[str]): Additional context
  - `target` or `ground_truth` (str): Correct answer
- `agent` (Agent): ACE Agent instance
- `skillbook` (Skillbook): Current skillbook
- `answer_checker` (Callable): Function that takes (prediction, ground_truth) -> bool
- `**kwargs`: Additional arguments passed to agent.generate()

**Returns:**
- `EvaluationResult`: Result object with prediction and correctness

### evaluate_dataset()

Parallel evaluation of multiple samples using ThreadPoolExecutor.

**Parameters:**
- `samples` (List[Dict]): List of sample dictionaries
- `agent` (Agent): ACE Agent instance
- `skillbook` (Skillbook): Current skillbook
- `answer_checker` (Callable): Function that takes (prediction, ground_truth) -> bool
- `max_workers` (int): Maximum parallel threads (default: 20)
- `show_progress` (bool): Print progress updates (default: True)
- `**kwargs`: Additional arguments passed to each agent.generate() call

**Returns:**
- `Dict` with keys:
  - `accuracy` (float): Overall accuracy (0.0 to 1.0)
  - `correct` (int): Number of correct predictions
  - `total` (int): Total samples evaluated
  - `errors` (List[Dict]): Error details for incorrect predictions
  - `results` (List[EvaluationResult]): All evaluation results

## Answer Checkers

The answer checker determines if a prediction is correct. Choose based on your task:

### Exact Match (Simple)

```python
def exact_match(pred, truth):
    return pred.strip().lower() == truth.strip().lower()
```

### Numeric Tolerance

```python
def numeric_match(pred, truth, tolerance=0.01):
    try:
        return abs(float(pred) - float(truth)) < tolerance
    except ValueError:
        return False
```

### Semantic Similarity (Advanced)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_match(pred, truth, threshold=0.85):
    embeddings = model.encode([pred, truth])
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity >= threshold
```

### Contains Match

```python
def contains_match(pred, truth):
    return truth.lower() in pred.lower()
```

## Error Handling

The framework handles errors gracefully:

1. **Per-sample isolation**: Errors in one sample don't affect others
2. **Error tracking**: All errors are captured in the results
3. **Detailed logging**: Error type and message preserved

**Example error output:**
```python
{
    "index": 5,
    "error": "RuntimeError: LLM API timeout",
    "ground_truth": "Paris"
}
```

## Performance Tips

1. **Adjust max_workers**: Based on API rate limits
   - OpenAI: 5-10 workers
   - Local models: 20+ workers
   - Rate-limited APIs: 1-3 workers

2. **Batch size**: Progress updates every 50 samples

3. **Memory usage**: Results stored in memory; for huge datasets, consider streaming

4. **Custom kwargs**: Pass temperature, max_tokens, etc. to control generation

## Testing

Run the test suite:

```bash
pytest tests/test_evaluation.py -v
```

**Test coverage:**
- Single sample evaluation (success/failure)
- Parallel dataset evaluation
- Error isolation
- Custom kwargs handling
- Empty datasets

## Example Usage

See `examples/evaluation_example.py` for complete examples including:
- Single sample evaluation
- Parallel dataset evaluation
- Custom answer checkers
- Error handling

## Integration with Training

The evaluation framework integrates seamlessly with ACE training:

```python
from ace import OfflineACE, Agent, Skillbook

# Train agent
adapter = OfflineACE(...)
adapter.run(train_samples, environment, epochs=3)

# Evaluate on test set
test_results = evaluate_dataset(
    samples=test_samples,
    agent=adapter.agent,
    skillbook=adapter.skillbook,
    answer_checker=exact_match,
    max_workers=10
)

print(f"Test Accuracy: {test_results['accuracy']:.2%}")
```
