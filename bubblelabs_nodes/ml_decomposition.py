"""
ML-Based Decomposition Prediction for OpenEvolve Gauntlet System

Uses machine learning to predict optimal decomposition depth for problems,
enabling smarter automatic decomposition decisions.

Key Features:
- Feature extraction from problems
- Depth prediction model
- Training data collection
- Model serving and prediction
- Continuous learning and improvement
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DecompositionExample:
    """Training example for decomposition prediction"""
    example_id: str
    problem_features: Dict[str, Any]
    optimal_depth: int
    actual_depth_used: int
    success: bool
    score: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemFeatures:
    """Features extracted from a problem for ML prediction"""
    problem_id: str
    # Text features
    statement_length: int
    word_count: int
    avg_word_length: float
    unique_word_ratio: float

    # Complexity features
    requirements_count: int
    subproblem_count: int
    max_nesting_depth: int
    has_constraints: bool
    has_dependencies: bool

    # Domain features (one-hot encoded)
    domain_web_dev: float = 0.0
    domain_ml: float = 0.0
    domain_data_processing: float = 0.0
    domain_security: float = 0.0
    domain_general: float = 1.0

    # Historical features
    similar_problem_count: int = 0
    avg_success_rate: float = 0.5

    # Resource features
    estimated_effort_hours: float = 0.0
    available_time_hours: float = 0.0
    team_size: int = 3
    team_experience: float = 0.5


class DecompositionDataCollector:
    """
    Collects training data from problem decompositions.
    """

    def __init__(self, storage_path: str = "./data/decomposition"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.examples: List[DecompositionExample] = []

    def collect_example(
        self,
        problem: Dict[str, Any],
        decomposition_result: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> DecompositionExample:
        """
        Collect a training example from a problem decomposition.

        Args:
            problem: Original problem
            decomposition_result: Result of decomposition
            outcome: Final outcome

        Returns:
            DecompositionExample
        """
        # Extract features
        features = self._extract_features(problem)

        # Determine optimal depth
        optimal_depth = self._determine_optimal_depth(decomposition_result, outcome)

        # Create example
        example = DecompositionExample(
            example_id=self._generate_example_id(),
            problem_features=features,
            optimal_depth=optimal_depth,
            actual_depth_used=decomposition_result.get('depth', 0),
            success=outcome.get('success', False),
            score=outcome.get('score', 0),
            execution_time=outcome.get('execution_time', 0),
            metadata={
                'collected_at': datetime.utcnow().isoformat(),
                'problem_statement': problem.get('statement', ''),
            }
        )

        self.examples.append(example)

        # Periodically save to disk
        if len(self.examples) % 10 == 0:
            self._save_examples()

        return example

    def _extract_features(self, problem: Dict[str, Any]) -> ProblemFeatures:
        """Extract features from problem"""
        statement = problem.get('statement', '')

        # Text features
        statement_length = len(statement)
        words = statement.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        unique_words = set(words.lower())
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

        # Complexity features
        requirements = problem.get('requirements', [])
        requirements_count = len(requirements) if isinstance(requirements, list) else 0

        subproblems = problem.get('subproblems', [])
        subproblem_count = len(subproblems) if isinstance(subproblems, list) else 0

        max_nesting_depth = self._calculate_nesting_depth(subproblems)

        has_constraints = bool(problem.get('constraints'))
        has_dependencies = bool(problem.get('dependencies'))

        # Domain features
        statement_lower = statement.lower()
        domain_features = {
            'domain_web_dev': 1.0 if any(w in statement_lower for w in ['web', 'api', 'http']) else 0.0,
            'domain_ml': 1.0 if any(w in statement_lower for w in ['ml', 'model', 'ai', 'predict']) else 0.0,
            'domain_data_processing': 1.0 if any(w in statement_lower for w in ['data', 'etl', 'pipeline']) else 0.0,
            'domain_security': 1.0 if any(w in statement_lower for w in ['security', 'auth', 'encrypt']) else 0.0,
            'domain_general': 1.0,  # Default
        }

        return ProblemFeatures(
            problem_id=problem.get('id', 'unknown'),
            statement_length=statement_length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            unique_word_ratio=unique_word_ratio,
            requirements_count=requirements_count,
            subproblem_count=subproblem_count,
            max_nesting_depth=max_nesting_depth,
            has_constraints=has_constraints,
            has_dependencies=has_dependencies,
            **domain_features
        )

    def _calculate_nesting_depth(self, subproblems: List) -> int:
        """Calculate maximum nesting depth"""
        if not subproblems:
            return 0

        max_depth = 0
        for sp in subproblems:
            if isinstance(sp, dict):
                sub_sub = sp.get('subproblems', [])
                if sub_sub:
                    depth = 1 + self._calculate_nesting_depth(sub_sub)
                max_depth = max(max_depth, depth)

        return max_depth

    def _determine_optimal_depth(
        self,
        decomposition_result: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> int:
        """Determine the optimal decomposition depth"""
        # Use outcome to determine optimal depth
        success = outcome.get('success', False)
        score = outcome.get('score', 0)

        # If successful and high score, actual depth was optimal
        if success and score >= 0.8:
            return decomposition_result.get('depth', 0)

        # If failed or low score, optimal might be shallower or deeper
        # This is simplified - a real model would learn this from data
        actual_depth = decomposition_result.get('depth', 0)

        # Heuristic: if success, keep same depth; if failed, try shallower
        if success:
            return actual_depth
        else:
            return max(0, actual_depth - 1)

    def _generate_example_id(self) -> str:
        """Generate unique example ID"""
        return f"example_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    def _save_examples(self):
        """Save examples to disk"""
        output_file = self.storage_path / "examples.jsonl"

        examples_data = [
            {
                'problem_features': e.problem_features.__dict__,
                'optimal_depth': e.optimal_depth,
                'actual_depth_used': e.actual_depth_used,
                'success': e.success,
                'score': e.score,
                'execution_time': e.execution_time,
                'metadata': e.metadata,
            }
            for e in self.examples
        ]

        with open(output_file, 'w') as f:
            json.dump(examples_data, f, indent=2)

        logger.info(f"Saved {len(examples_data)} examples to {output_file}")

    def load_examples(self, filepath: str = None) -> List[DecompositionExample]:
        """Load examples from disk"""
        if filepath is None:
            filepath = self.storage_path / "examples.jsonl"

        with open(filepath, 'r') as f:
            examples_data = json.load(f)

        examples = []
        for ex_data in examples_data:
            features = ProblemFeatures(**ex_data['problem_features'])
            example = DecompositionExample(
                example_id=ex_data['example_id'],
                problem_features=features,
                optimal_depth=ex_data['optimal_depth'],
                actual_depth_used=ex_data['actual_depth_used'],
                success=ex_data['success'],
                score=ex_data['score'],
                execution_time=ex_data['execution_time'],
                metadata=ex_data['metadata']
            )
            examples.append(example)

        self.examples = examples
        return examples

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the collected dataset"""
        if not self.examples:
            return {
                'total_examples': 0,
                'avg_depth': 0,
                'success_rate': 0.0,
            }

        depths = [e.optimal_depth for e in self.examples]
        successes = [e.success for e in self.examples]
        scores = [e.score for e in self.examples]

        return {
            'total_examples': len(self.examples),
            'avg_optimal_depth': sum(depths) / len(depths),
            'depth_distribution': {
                'depth_0': sum(1 for d in depths if d == 0),
                'depth_1': sum(1 for d in depths if d == 1),
                'depth_2': sum(1 for d in depths if d == 2),
                'depth_3': sum(1 for d in depths if d == 3),
                'depth_4+': sum(1 for d in depths if d >= 4),
            },
            'success_rate': sum(successes) / len(successes),
            'avg_score': sum(scores) / len(scores),
        }


class DepthPredictionModel:
    """
    Model for predicting optimal decomposition depth.

    Simplified rule-based model that can be replaced with
    trained ML models.
    """

    def __init__(self):
        # Simple rule-based model
        self.rules = {
            'very_simple': (0, 1),  # Very simple: depth 0-1
            'simple': (1, 2),  # Simple: depth 1-2
            'moderate': (2, 3),  # Moderate: depth 2-3
            'complex': (3, 4),  # Complex: depth 3-4
            'very_complex': (4, 5),  # Very complex: depth 4-5
        }

    def predict_depth(self, features: ProblemFeatures) -> Tuple[int, float]:
        """
        Predict optimal decomposition depth.

        Args:
            features: Extracted problem features

        Returns:
            Tuple of (predicted_depth, confidence)
        """
        # Calculate complexity score
        complexity_score = (
            features.requirements_count * 0.3 +
            features.subproblem_count * 0.2 +
            features.max_nesting_depth * 0.3 +
            features.has_constraints * 0.1
        )

        # Normalize to 0-1
        complexity_score = min(1.0, complexity_score / 10)

        # Determine depth range based on complexity
        if complexity_score < 0.2:
            depth_range = self.rules['very_simple']
        elif complexity_score < 0.4:
            depth_range = self.rules['simple']
        elif complexity_score < 0.6:
            depth_range = self.rules['moderate']
        elif complexity_score < 0.8:
            depth_range = self.rules['complex']
        else:
            depth_range = self.rules['very_complex']

        # Adjust based on historical success
        if features.avg_success_rate > 0.8:
            # Successful history -> try shallower
            depth_range = (depth_range[0], max(0, depth_range[1] - 1))
        elif features.avg_success_rate < 0.3:
            # Poor history -> go deeper for better decomposition
            depth_range = (depth_range[0], depth_range[1] + 1)

        # Predict depth as weighted average
        predicted_depth = int(round(
            depth_range[0] * 0.3 + depth_range[1] * 0.7
        ))

        # Calculate confidence
        confidence = 0.5 + (features.avg_success_rate * 0.3)

        return (predicted_depth, min(1.0, confidence))


class DecompositionPredictor:
    """
    Main interface for decomposition prediction.
    """

    def __init__(
        self,
        model: DepthPredictionModel = None,
        data_collector: DecompositionDataCollector = None
    ):
        self.model = model or DepthPredictionModel()
        self.data_collector = data_collector or DecompositionDataCollector()

    def collect_data(
        self,
        problem: Dict[str, Any],
        decomposition_result: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> DecompositionExample:
        """Collect training data from problem decomposition"""
        return self.data_collector.collect_example(
            problem,
            decomposition_result,
            outcome
        )

    def predict_depth(
        self,
        problem: Dict[str, Any],
        historical_data: Dict[str, Any] = None
    ) -> Tuple[int, float]:
        """
        Predict optimal decomposition depth for a problem.

        Args:
            problem: Problem to analyze
            historical_data: Historical performance data

        Returns:
            Tuple of (predicted_depth, confidence)
        """
        # Extract features
        features = self._extract_features(problem, historical_data)

        # Make prediction
        depth, confidence = self.model.predict_depth(features)

        return (depth, confidence)

    def _extract_features(
        self,
        problem: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> ProblemFeatures:
        """Extract features from problem"""
        # Simplified feature extraction
        statement = problem.get('statement', '')

        # Text features
        words = statement.split()
        word_count = len(words)
        unique_words = set(words.lower())
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

        # Complexity features
        requirements = problem.get('requirements', [])
        requirements_count = len(requirements) if isinstance(requirements, list) else 0

        subproblems = problem.get('subproblems', [])
        subproblem_count = len(subproblems) if isinstance(subproblems, list) else 0

        # Domain features
        statement_lower = statement.lower()
        domain_features = {
            'domain_web_dev': 1.0 if 'web' in statement_lower else 0.0,
            'domain_ml': 1.0 if 'ml' in statement_lower or 'ai' in statement_lower else 0.0,
            'domain_data_processing': 1.0 if 'data' in statement_lower else 0.0,
            'domain_security': 1.0 if 'security' in statement_lower else 0.0,
            'domain_general': 1.0,
        }

        # Historical features
        historical_success_rate = historical_data.get('baseline_success_rate', 0.5) if historical_data else 0.5

        return ProblemFeatures(
            problem_id=problem.get('id', 'unknown'),
            statement_length=len(statement),
            word_count=word_count,
            avg_word_length=sum(len(w) for w in words) / word_count if word_count > 0 else 0,
            unique_word_ratio=unique_word_ratio,
            requirements_count=requirements_count,
            subproblem_count=subproblem_count,
            max_nesting_depth=0,  # Would need full tree analysis
            has_constraints=bool(problem.get('constraints')),
            has_dependencies=bool(problem.get('dependencies')),
            avg_success_rate=historical_success_rate,
            **domain_features
        )

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about training data"""
        return self.data_collector.get_dataset_statistics()


def create_decomposition_predictor(
    data_path: str = "./data/decomposition"
) -> DecompositionPredictor:
    """Factory function to create decomposition predictor"""
    return DecompositionPredictor(
        data_collector=DecompositionDataCollector(data_path)
    )


# Example usage
async def demo_ml_decomposition():
    """Demonstration of ML-based decomposition prediction"""

    predictor = create_decomposition_predictor()

    print("\n" + "=" * 60)
    print("ML-Based Decomposition Prediction Demo")
    print("=" * 60)

    # Example 1: Simple problem
    simple_problem = {
        'id': 'simple_1',
        'statement': 'Create a user login page',
        'requirements': ['email', 'password']
    }

    depth1, conf1 = predictor.predict_depth(simple_problem)
    print(f"\nSimple problem:")
    print(f"  Statement: {simple_problem['statement']}")
    print(f"  Predicted depth: {depth1}")
    print(f"  Confidence: {conf1:.1%}")

    # Example 2: Complex problem
    complex_problem = {
        'id': 'complex_1',
        'statement': 'Build a machine learning pipeline with data ingestion, model training, and deployment',
        'requirements': [
            'data ingestion',
            'model training',
            'model deployment',
            'monitoring',
            'scalability',
            'fault tolerance'
        ],
        'subproblems': [
            {
                'statement': 'Design data model',
                'subproblems': [
                    {'statement': 'Define schema'},
                    {'statement': 'Define relationships'}
                ]
            },
            {
                'statement': 'Implement training',
                'requirements': ['accuracy', 'speed']
            }
        ]
    }

    depth2, conf2 = predictor.predict_depth(complex_problem)
    print(f"\nComplex problem:")
    print(f"  Statement: {complex_problem['statement'][:80]}...")
    print(f"  Predicted depth: {depth2}")
    print(f"  Confidence: {conf2:.1%}")

    # Show training stats
    stats = predictor.get_training_statistics()
    print(f"\nTraining data statistics:")
    print(f"  Examples collected: {stats['total_examples']}")
    print(f"  Average optimal depth: {stats['avg_optimal_depth']:.1f}")
    print(f"  Success rate: {stats['success_rate']:.1%}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_ml_decomposition())
