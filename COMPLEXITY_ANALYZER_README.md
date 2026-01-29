# Complexity Analyzer - Production-Ready Implementation

## Overview

The `complexity_analyzer.py` module provides a comprehensive, production-ready implementation for analyzing problem complexity across multiple dimensions. This system integrates seamlessly with the existing `sovereign_data_models.py` and `problem_fractal_pipeline.py`.

## Features

### 1. Multi-Dimensional Complexity Analysis

The analyzer evaluates complexity across **5 key dimensions**:

- **Cognitive Complexity**: Mental effort and cognitive load required
  - Sentence structure and readability
  - Technical terminology density
  - Abstract concept usage
  - Quantifier frequency

- **Computational Complexity**: Algorithmic and resource requirements
  - Time/space complexity indicators
  - Performance constraints
  - Scalability requirements
  - Resource demands

- **Domain Complexity**: Specialized knowledge needed
  - Domain-specific complexity mapping
  - Constraint complexity analysis
  - Expertise level requirements

- **Integration Complexity**: Dependencies and external systems
  - Number of dependencies
  - Integration types (APIs, databases, services)
  - Synchronization requirements

- **Overall Complexity**: Weighted composite score
  - Configurable weights for each dimension
  - Normalized scoring (0.0 to 1.0)
  - Confidence assessment

### 2. Comprehensive Error Handling

- Input validation with descriptive error messages
- Type checking for both dataclass and TypedDict
- Graceful handling of missing attributes
- Edge case coverage (empty inputs, special characters, unicode)

### 3. Production-Ready Features

- Structured logging with context
- Confidence scoring for reliability assessment
- Detailed explanations for each score
- Dimension breakdown with contributing factors
- Extensible configuration system

## Installation

The module is self-contained and requires only Python 3.7+ standard library:

```python
# No additional dependencies required
import complexity_analyzer
```

## Quick Start

### Basic Usage

```python
from complexity_analyzer import quick_complexity_analysis

result = quick_complexity_analysis(
    description="Create a web form for user registration",
    domain="web_development",
    requirements=["Validate input", "Store in database"]
)

print(f"Overall Complexity: {result.overall_score:.2f}")
print(f"Explanation: {result.explanation}")
```

### Advanced Usage

```python
from complexity_analyzer import ComplexityAnalyzer
from sovereign_data_models import ProblemDefinition
from datetime import datetime

# Create analyzer with custom configuration
analyzer = ComplexityAnalyzer(config={
    'cognitive_weight': 0.4,      # Increase cognitive importance
    'computational_weight': 0.3,
    'domain_weight': 0.2,
    'integration_weight': 0.1,
    'normalize_scores': True,
    'min_confidence': 0.6
})

# Analyze a problem
problem = ProblemDefinition(
    problem_id="prob_001",
    title="ML System Design",
    description="Design a deep learning system for real-time predictions",
    domain="machine_learning",
    complexity="high",
    priority="high",
    estimated_effort="weeks",
    requirements=["Real-time processing", "High accuracy"],
    constraints=["Limited resources"],
    created_at=datetime.now()
)

result = analyzer.calculate_complexity(problem)

# Access detailed results
print(f"Overall: {result.overall_score:.2f}")
print(f"Cognitive: {result.cognitive_score:.2f}")
print(f"Computational: {result.computational_score:.2f}")
print(f"Domain: {result.domain_score:.2f}")
print(f"Integration: {result.integration_score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")

# Access dimension breakdown
for dimension, details in result.dimension_breakdown.items():
    print(f"{dimension.capitalize()}:")
    print(f"  Score: {details['score']:.2f}")
    print(f"  Level: {details['level']}")
    print(f"  Factors: {', '.join(details['factors'])}")
```

## API Reference

### ComplexityAnalyzer Class

#### Initialization

```python
analyzer = ComplexityAnalyzer(config={
    'cognitive_weight': 0.3,      # Weight for cognitive complexity (default: 0.3)
    'computational_weight': 0.3,   # Weight for computational complexity (default: 0.3)
    'domain_weight': 0.2,          # Weight for domain complexity (default: 0.2)
    'integration_weight': 0.2,     # Weight for integration complexity (default: 0.2)
    'min_confidence': 0.5,         # Minimum confidence threshold (default: 0.5)
    'normalize_scores': True       # Whether to normalize scores to [0,1] (default: True)
})
```

#### Methods

##### `calculate_complexity(problem, context=None)`

Calculate overall complexity score for a problem.

**Parameters:**
- `problem` (ProblemDefinition): Problem definition
- `context` (Dict, optional): Additional context information

**Returns:**
- `ComplexityScore`: Complexity scores across all dimensions

**Raises:**
- `ValueError`: If problem description is empty
- `TypeError`: If problem is not a valid ProblemDefinition

##### `analyze_cognitive_complexity(description, domain)`

Analyze cognitive complexity based on description.

**Parameters:**
- `description` (str): Problem description
- `domain` (str): Problem domain

**Returns:**
- `float`: Cognitive complexity score (0.0 to 1.0)

##### `analyze_computational_complexity(requirements)`

Analyze computational complexity based on requirements.

**Parameters:**
- `requirements` (List[str]): List of requirement strings

**Returns:**
- `float`: Computational complexity score (0.0 to 1.0)

##### `analyze_domain_complexity(domain, constraints)`

Analyze domain complexity based on domain and constraints.

**Parameters:**
- `domain` (str): Problem domain
- `constraints` (List[str]): List of constraints

**Returns:**
- `float`: Domain complexity score (0.0 to 1.0)

##### `analyze_integration_complexity(dependencies)`

Analyze integration complexity based on dependencies.

**Parameters:**
- `dependencies` (List[str]): List of dependency identifiers

**Returns:**
- `float`: Integration complexity score (0.0 to 1.0)

### ComplexityScore Data Structure

```python
@dataclass
class ComplexityScore:
    overall_score: float          # Overall complexity (0.0 to 1.0)
    cognitive_score: float        # Cognitive complexity (0.0 to 1.0)
    computational_score: float   # Computational complexity (0.0 to 1.0)
    domain_score: float          # Domain complexity (0.0 to 1.0)
    integration_score: float     # Integration complexity (0.0 to 1.0)
    confidence: float            # Confidence in assessment (0.0 to 1.0)
    explanation: str             # Human-readable explanation
    dimension_breakdown: Dict    # Detailed breakdown per dimension
```

### Complexity Levels

Scores are mapped to meaningful complexity levels:

- **0.0 - 0.15**: Trivial
- **0.15 - 0.35**: Simple
- **0.35 - 0.55**: Moderate
- **0.55 - 0.75**: Complex
- **0.75 - 0.90**: Very Complex
- **0.90 - 1.00**: Extreme

## Technical Details

### Scoring Algorithms

#### Cognitive Complexity

1. **Structural Analysis** (30%)
   - Sentence length
   - Sentence count
   - Conditional indicators

2. **Technical Analysis** (40%)
   - Technical term density
   - Domain-specific keywords
   - Concept complexity

3. **Conceptual Analysis** (30%)
   - Abstract concept frequency
   - Quantifier usage
   - Mental effort indicators

#### Computational Complexity

1. **Algorithm Indicators** (60%)
   - Complexity keywords (O(n), O(n²), etc.)
   - Algorithm classes (NP-hard, exponential, etc.)
   - Optimization requirements

2. **Scale Factors** (40%)
   - Number of requirements
   - Performance constraints
   - Resource limitations

#### Domain Complexity

1. **Base Domain Score** (70%)
   - Pre-defined domain complexity mapping
   - 50+ domains categorized
   - Specialized knowledge indicators

2. **Constraint Analysis** (30%)
   - Constraint complexity
   - Number of constraints
   - Interdependencies

#### Integration Complexity

1. **Dependency Count** (60%)
   - Number of dependencies
   - Scaled logarithmically

2. **Type Analysis** (40%)
   - Integration types (APIs, databases, services)
   - Synchronization requirements
   - External vs internal

### Technical Term Detection

The analyzer maintains a comprehensive database of 200+ technical terms across:

- Machine Learning & AI
- Distributed Systems
- Algorithms & Data Structures
- Software Engineering
- Cloud Computing
- DevOps & Infrastructure
- Security & Cryptography
- Database Systems
- Networking

Each term is weighted based on its complexity contribution.

### Domain Complexity Mapping

Pre-configured complexity scores for 50+ domains including:

- `machine_learning`: 0.85
- `deep_learning`: 0.90
- `computer_vision`: 0.80
- `distributed_systems`: 0.85
- `web_development`: 0.50
- `database`: 0.60
- And 40+ more

## Integration with Problem Fractal Pipeline

The complexity analyzer integrates seamlessly with the problem fractal pipeline:

```python
from problem_fractal_pipeline import ProblemFractalPipeline
from complexity_analyzer import ComplexityAnalyzer

# Initialize pipeline
pipeline = ProblemFractalPipeline()

# Analyze complexity before decomposition
analyzer = ComplexityAnalyzer()
complexity = analyzer.calculate_complexity(problem)

# Use complexity to guide decomposition strategy
if complexity.overall_score > 0.7:
    # Use more aggressive decomposition for complex problems
    sub_problems = pipeline.decompose(problem, max_depth=4)
else:
    # Use simpler decomposition for straightforward problems
    sub_problems = pipeline.decompose(problem, max_depth=2)
```

## Testing

Comprehensive test suite with 33+ test cases covering:

- Unit tests for all methods
- Edge case handling
- Integration with sovereign_data_models
- Performance testing
- Error handling

Run tests:

```bash
python test_complexity_analyzer.py
```

Expected output:

```
================================================================================
TEST SUMMARY
================================================================================
Tests run: 33
Successes: 33
Failures: 0
Errors: 0
================================================================================
```

## Examples

### Example 1: Simple Web Form

**Input:**
```python
quick_complexity_analysis(
    description="Create a web form to collect user contact information",
    domain="web_development"
)
```

**Output:**
```
Overall Score: 0.26
Level: simple
Explanation: This problem is simple in complexity.
```

### Example 2: Machine Learning System

**Input:**
```python
quick_complexity_analysis(
    description="Design a deep learning system for real-time object detection",
    domain="computer_vision",
    requirements=["Real-time processing", "Low latency", "High accuracy"]
)
```

**Output:**
```
Overall Score: 0.52
Cognitive: 0.74
Computational: 0.53
Domain: 0.68
Integration: 0.00
Level: moderate
Explanation: This problem is moderate in complexity. It requires
significant mental effort due to complex concepts and terminology.
The computational requirements are moderate. Some domain expertise
will be helpful.
```

### Example 3: Distributed Systems

**Input:**
```python
quick_complexity_analysis(
    description="Design a distributed consensus protocol for a blockchain system",
    domain="distributed_systems",
    requirements=["High throughput", "Byzantine fault tolerance"],
    constraints=["Geographic distribution", "Network partitions"]
)
```

**Output:**
```
Overall Score: 0.46
Cognitive: 0.61
Computational: 0.50
Domain: 0.62
Integration: 0.00
Level: moderate
Explanation: This problem is moderate in complexity. It requires
moderate mental effort to understand the concepts involved.
The computational requirements are moderate. Some domain expertise
will be helpful.
```

## Configuration

### Custom Weights

Adjust the importance of each dimension:

```python
# Emphasize cognitive complexity
analyzer = ComplexityAnalyzer(config={
    'cognitive_weight': 0.5,
    'computational_weight': 0.2,
    'domain_weight': 0.2,
    'integration_weight': 0.1
})
```

### Score Normalization

Control whether scores are normalized to [0, 1]:

```python
analyzer = ComplexityAnalyzer(config={
    'normalize_scores': False  # Allow scores to exceed 1.0
})
```

### Confidence Thresholds

Set minimum confidence for reliable assessments:

```python
analyzer = ComplexityAnalyzer(config={
    'min_confidence': 0.7  # Require higher confidence
})
```

## Performance Considerations

- **Time Complexity**: O(n) where n is the length of the description
- **Space Complexity**: O(1) - constant space usage
- **Typical Runtime**: < 10ms for average problem descriptions
- **Scalability**: Handles descriptions up to 100K+ characters

## Error Handling

The analyzer provides comprehensive error handling:

```python
try:
    result = analyzer.calculate_complexity(problem)
except ValueError as e:
    print(f"Invalid input: {e}")
except TypeError as e:
    print(f"Type error: {e}")
```

Common errors:

- `ValueError: Problem description cannot be empty`
- `TypeError: Expected ProblemDefinition, got <type>`

## Best Practices

1. **Always provide detailed descriptions** for accurate cognitive analysis
2. **Include specific requirements** for better computational assessment
3. **Specify constraints** to improve domain complexity accuracy
4. **List dependencies** for integration complexity evaluation
5. **Use appropriate domain names** from the predefined list
6. **Check confidence scores** before making critical decisions
7. **Review dimension breakdowns** for detailed insights

## Limitations

1. **Language Support**: Optimized for English text
2. **Domain Coverage**: Limited to 50+ predefined domains
3. **Real-time Analysis**: Not suitable for streaming data
4. **Subjectivity**: Cognitive complexity has inherent subjectivity

## Future Enhancements

- [ ] Multi-language support
- [ ] ML-based complexity prediction
- [ ] Historical complexity tracking
- [ ] Team expertise calibration
- [ ] Integration with JIRA/GitHub issues
- [ ] REST API endpoint
- [ ] Real-time complexity monitoring

## Contributing

To extend the analyzer:

1. Add new technical terms to `TECHNICAL_TERMS`
2. Add domain mappings to `DOMAIN_COMPLEXITY_MAP`
3. Add complexity keywords to `COMPLEXITY_KEYWORDS`
4. Implement custom scoring methods
5. Add unit tests for new functionality

## License

This module is part of the OpenEvolve Frontend project.

## Support

For issues, questions, or contributions, please refer to the main project documentation.
