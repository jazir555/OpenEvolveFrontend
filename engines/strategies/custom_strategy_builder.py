"""
Custom Strategy Builder for Decomposition Engine

This module allows users to create custom decomposition strategies tailored to
their specific needs, with validation, testing, and integration capabilities.

Features:
- Create custom strategies from configuration
- Create strategies from templates
- Validate custom strategies
- Test strategies on sample problems
- Register strategies with DecompositionEngine
- Export/import strategy definitions
"""
from __future__ import annotations



import logging
import json
import importlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import time

from sovereign_data_models import (
    ProblemDefinition, SubProblem, SubProblemType,
    ComplexityScore, DomainContext, ProblemType,
    generate_id
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for custom strategy."""
    strategy_name: str
    description: str

    # Decomposition criteria
    decomposition_criteria: Dict[str, Any] = field(default_factory=dict)

    # LLM prompts
    system_prompt: str = ""
    user_prompt_template: str = ""

    # Sub-problem handling
    sub_problem_ordering: str = "sequential"  # "sequential", "priority", "custom"
    custom_ordering_function: Optional[str] = None

    # Dependencies
    add_dependencies: bool = True
    dependency_rules: List[str] = field(default_factory=list)

    # Quality
    quality_thresholds: Dict[str, float] = field(default_factory=dict)

    # Metadata
    author: str = "Unknown"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate strategy configuration."""
        errors = []

        if not self.strategy_name:
            errors.append("strategy_name is required")

        valid_orderings = ["sequential", "priority", "custom"]
        if self.sub_problem_ordering not in valid_orderings:
            errors.append(f"sub_problem_ordering must be one of {valid_orderings}, got {self.sub_problem_ordering}")

        if self.sub_problem_ordering == "custom" and not self.custom_ordering_function:
            errors.append("custom_ordering_function is required when sub_problem_ordering is 'custom'")

        # Validate quality thresholds
        for key, value in self.quality_thresholds.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                errors.append(f"quality_thresholds.{key} must be between 0.0 and 1.0, got {value}")

        return errors


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_checks: Dict[str, bool] = field(default_factory=dict)

    # Specific checks
    has_decompose_method: bool = False
    returns_valid_subproblems: bool = False
    handles_errors: bool = False
    compatible_with_engine: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StrategyTestResults:
    """Results of testing custom strategy."""
    strategy_name: str

    # Test execution
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0

    # Performance metrics
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_decomposition_time: float = 0.0

    # Quality metrics
    avg_sub_problem_count: float = 0.0
    avg_complexity_score: float = 0.0

    # Error handling
    error_handling_score: float = 0.0

    # Recommendations
    ready_for_production: bool = False
    improvements_needed: List[str] = field(default_factory=list)

    # Detailed test results
    test_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CustomStrategyBuilder:
    """
    Allows users to create custom decomposition strategies.

    Features:
    - Create strategy from configuration
    - Create strategy from template
    - Validate strategy
    - Test strategy on sample problems
    - Register with DecompositionEngine
    - Export/import definitions
    """

    def __init__(self):
        """Initialize builder."""
        self.registered_strategies: Dict[str, 'DecompositionStrategyBase'] = {}
        self.strategy_history: List[Dict[str, Any]] = []
        logger.info("CustomStrategyBuilder initialized")

    def create_strategy(
        self,
        strategy_config: StrategyConfig
    ) -> 'DecompositionStrategyBase':
        """
        Create custom strategy from configuration.

        Args:
            strategy_config: StrategyConfig object

        Returns:
            DecompositionStrategyBase instance
        """
        logger.info(f"Creating custom strategy: {strategy_config.strategy_name}")

        # Validate configuration
        errors = strategy_config.validate()
        if errors:
            raise ValueError(f"Invalid strategy configuration: {errors}")

        # Create strategy class dynamically
        class CustomDecompositionStrategy:
            """Dynamically created decomposition strategy."""

            def __init__(self, config: StrategyConfig):
                self.config = config
                self.strategy_name = config.strategy_name
                self.description = config.description

            def decompose(self, problem: ProblemDefinition, domain_context=None) -> List[SubProblem]:
                """Decompose problem using custom strategy."""
                # This is a simplified implementation
                # In production, this would use LLM-based decomposition

                sub_problems = []

                # Determine number of sub-problems based on complexity
                num_sub_problems = self._determine_subproblem_count(problem)

                for i in range(num_sub_problems):
                    sp_id = generate_id(f"sub_{self.strategy_name}_{i}")

                    # Create sub-problem with custom criteria
                    sub_problem = SubProblem(
                        id=sp_id,
                        title=f"{self.strategy_name.title()} Sub-Problem {i+1}",
                        description=f"Custom decomposition sub-problem {i+1} for {problem.title}",
                        sub_problem_type=SubProblemType.IMPLEMENTATION,
                        parent_problem_id=problem.id,
                        dependencies=[],
                        acceptance_criteria=[f"Criterion {j+1}" for j in range(3)],
                        priority=5,
                        estimated_resources={
                            'time_hours': 16.0,
                            'api_tokens': 50000,
                            'computational_units': 2.0,
                            'human_review_minutes': 30
                        },
                        complexity_score=ComplexityScore(
                            explanation="Custom strategy complexity",
                            cognitive_complexity=5.0,
                            computational_complexity=5.0,
                            domain_complexity=5.0,
                            integration_complexity=5.0,
                            overall_complexity=5.0
                        ),
                        evolution_mode="standard"
                    )

                    sub_problems.append(sub_problem)

                # Apply ordering
                sub_problems = self._order_subproblems(sub_problems)

                # Apply dependencies if configured
                if self.config.add_dependencies:
                    sub_problems = self._apply_dependencies(sub_problems)

                return sub_problems

            def get_strategy_name(self) -> str:
                """Get strategy name."""
                return self.strategy_name

            def _determine_subproblem_count(self, problem: ProblemDefinition) -> int:
                """Determine optimal number of sub-problems."""
                criteria = self.config.decomposition_criteria

                # Use custom count if specified
                if 'sub_problem_count' in criteria:
                    return criteria['sub_problem_count']

                # Otherwise, base on complexity
                complexity = problem.complexity_score.overall_complexity
                if complexity < 3:
                    return 3
                elif complexity < 6:
                    return 4
                elif complexity < 8:
                    return 5
                else:
                    return 6

            def _order_subproblems(self, sub_problems: List[SubProblem]) -> List[SubProblem]:
                """Order sub-problems according to configuration."""
                ordering = self.config.sub_problem_ordering

                if ordering == "priority":
                    # Sort by priority (descending)
                    return sorted(sub_problems, key=lambda sp: sp.priority, reverse=True)
                elif ordering == "custom" and self.config.custom_ordering_function:
                    # Apply custom ordering function
                    try:
                        # In production, this would dynamically load and execute the function
                        logger.warning("Custom ordering functions not yet implemented, using sequential")
                    except (ValueError, TypeError, RuntimeError) as e:
                        logger.error(f"Error applying custom ordering: {e}")

                # Default: sequential (no reordering)
                return sub_problems

            def _apply_dependencies(self, sub_problems: List[SubProblem]) -> List[SubProblem]:
                """Apply dependencies between sub-problems."""
                # Simple sequential dependencies
                for i, sp in enumerate(sub_problems[1:], start=1):
                    sp.dependencies = [sub_problems[i-1].id]

                return sub_problems

        # Create and return strategy instance
        strategy = CustomDecompositionStrategy(strategy_config)

        # Log creation
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "created",
            "strategy_name": strategy_config.strategy_name,
            "config": strategy_config.to_dict()
        })

        logger.info(f"Successfully created strategy: {strategy_config.strategy_name}")
        return strategy

    def create_strategy_from_template(
        self,
        template_name: str,
        customizations: Dict[str, Any]
    ) -> 'DecompositionStrategyBase':
        """
        Create strategy by customizing template.

        Templates:
        - "domain_specific": For specific domains
        - "complexity_based": Focus on complexity
        - "priority_based": Focus on business value
        - "team_based": Focus on team capabilities

        Args:
            template_name: Name of template to use
            customizations: Dict of customizations to apply

        Returns:
            DecompositionStrategyBase instance
        """
        logger.info(f"Creating strategy from template: {template_name}")

        # Import templates
        try:
            from strategy_templates import StrategyTemplates
            templates = StrategyTemplates()
        except ImportError:
            logger.error("strategy_templates module not found")
            raise ImportError("Strategy templates not available")

        # Get template config
        if template_name == "domain_specific":
            domain = customizations.get("domain", "general")
            config = templates.domain_specific_template(domain)
        elif template_name == "priority_based":
            config = templates.priority_based_template()
        elif template_name == "complexity_based":
            config = templates.complexity_based_template()
        elif template_name == "team_based":
            config = templates.team_based_template()
        else:
            raise ValueError(f"Unknown template: {template_name}. Available: domain_specific, priority_based, complexity_based, team_based")

        # Apply customizations
        for key, value in customizations.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Set strategy name if not customized
        if "strategy_name" not in customizations:
            config.strategy_name = f"{template_name}_custom_{int(time.time())}"

        # Create strategy from config
        return self.create_strategy(config)

    def validate_strategy(
        self,
        strategy: 'DecompositionStrategyBase'
    ) -> ValidationResult:
        """
        Validate custom strategy.

        Checks:
        - Has required methods
        - Returns valid SubProblem list
        - Handles errors gracefully
        - Compatible with engine

        Args:
            strategy: Strategy to validate

        Returns:
            ValidationResult object
        """
        logger.info(f"Validating strategy: {strategy.get_strategy_name()}")

        result = ValidationResult(
            is_valid=True,
            validation_checks={}
        )

        # Check 1: Has decompose method
        if hasattr(strategy, 'decompose') and callable(strategy.decompose):
            result.has_decompose_method = True
            result.validation_checks['has_decompose_method'] = True
        else:
            result.errors.append("Strategy must have a 'decompose' method")
            result.validation_checks['has_decompose_method'] = False
            result.is_valid = False

        # Check 2: Returns valid SubProblem list
        if result.has_decompose_method:
            try:
                # Create test problem
                test_problem = self._create_test_problem()

                # Test decomposition
                sub_problems = strategy.decompose(test_problem)

                if isinstance(sub_problems, list) and len(sub_problems) > 0:
                    # Validate each sub-problem
                    all_valid = True
                    for sp in sub_problems[:3]:  # Check first 3
                        if not isinstance(sp, SubProblem):
                            all_valid = False
                            break

                    if all_valid:
                        result.returns_valid_subproblems = True
                        result.validation_checks['returns_valid_subproblems'] = True
                    else:
                        result.errors.append("Strategy must return list of SubProblem objects")
                        result.validation_checks['returns_valid_subproblems'] = False
                        result.is_valid = False
                else:
                    result.errors.append("Strategy must return non-empty list")
                    result.validation_checks['returns_valid_subproblems'] = False
                    result.is_valid = False

            except (RuntimeError, ValueError, TypeError) as e:
                result.errors.append(f"decompose() raised exception: {e}")
                result.validation_checks['returns_valid_subproblems'] = False
                result.is_valid = False

        # Check 3: Handles errors gracefully
        try:
            # Test with invalid input
            test_problem = self._create_test_problem()
            test_problem.title = ""  # Invalid

            try:
                sub_problems = strategy.decompose(test_problem)
                # If it didn't raise, check if it handled gracefully
                if isinstance(sub_problems, list) and len(sub_problems) == 0:
                    result.handles_errors = True
                    result.validation_checks['handles_errors'] = True
                else:
                    result.warnings.append("Strategy may not handle invalid input gracefully")
                    result.validation_checks['handles_errors'] = False
            except (RuntimeError, ValueError, TypeError):
                # It raised, but was it graceful?
                result.validation_checks['handles_errors'] = False

        except (RuntimeError, ValueError, TypeError) as e:
            result.warnings.append(f"Error handling test failed: {e}")
            result.validation_checks['handles_errors'] = False

        # Check 4: Compatible with engine
        if result.has_decompose_method and hasattr(strategy, 'get_strategy_name'):
            result.compatible_with_engine = True
            result.validation_checks['compatible_with_engine'] = True
        else:
            result.errors.append("Strategy must have get_strategy_name() method")
            result.validation_checks['compatible_with_engine'] = False
            result.is_valid = False

        logger.info(f"Validation complete. Valid: {result.is_valid}, Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
        return result

    def test_strategy(
        self,
        strategy: 'DecompositionStrategyBase',
        test_problems: List[ProblemDefinition],
        sample_size: int = 5
    ) -> StrategyTestResults:
        """
        Test custom strategy on sample problems.

        Measures:
        - Success rate (produces valid decompositions)
        - Quality of results
        - Performance (time)
        - Error handling

        Args:
            strategy: Strategy to test
            test_problems: List of test problems
            sample_size: Number of problems to test (max)

        Returns:
            StrategyTestResults object
        """
        logger.info(f"Testing strategy: {strategy.get_strategy_name()}")

        results = StrategyTestResults(
            strategy_name=strategy.get_strategy_name()
        )

        # Limit test sample size
        test_problems = test_problems[:sample_size]
        results.tests_run = len(test_problems)

        sub_problem_counts = []
        complexity_scores = []
        decomposition_times = []
        quality_scores = []

        for problem in test_problems:
            test_detail = {
                "problem_id": problem.id,
                "success": False,
                "sub_problem_count": 0,
                "decomposition_time": 0.0,
                "errors": []
            }

            try:
                # Measure decomposition time
                start_time = time.time()
                sub_problems = strategy.decompose(problem)
                end_time = time.time()

                decomposition_time = end_time - start_time
                test_detail["decomposition_time"] = decomposition_time
                decomposition_times.append(decomposition_time)

                # Validate results
                if isinstance(sub_problems, list) and len(sub_problems) > 0:
                    test_detail["success"] = True
                    test_detail["sub_problem_count"] = len(sub_problems)
                    results.tests_passed += 1

                    sub_problem_counts.append(len(sub_problems))

                    # Calculate quality metrics
                    if sub_problems:
                        avg_complexity = sum(
                            sp.complexity_score.overall_complexity
                            for sp in sub_problems
                        ) / len(sub_problems)
                        complexity_scores.append(avg_complexity)

                        # Simple quality score: balanced complexity + reasonable count
                        quality = 1.0 - abs(len(sub_problems) - 5) / 10.0  # Optimal: 5 sub-problems
                        quality_scores.append(max(0.0, quality))
                else:
                    results.tests_failed += 1
                    test_detail["errors"].append("Empty or invalid result")

            except (RuntimeError, ValueError, TypeError) as e:
                results.tests_failed += 1
                test_detail["errors"].append(str(e))
                logger.error(f"Strategy test error: {e}", exc_info=True)

            results.test_details.append(test_detail)

        # Calculate aggregate metrics
        results.success_rate = results.tests_passed / results.tests_run if results.tests_run > 0 else 0.0
        results.avg_sub_problem_count = sum(sub_problem_counts) / len(sub_problem_counts) if sub_problem_counts else 0.0
        results.avg_complexity_score = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
        results.avg_decomposition_time = sum(decomposition_times) / len(decomposition_times) if decomposition_times else 0.0
        results.avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        results.error_handling_score = 1.0 - (results.tests_failed / results.tests_run) if results.tests_run > 0 else 0.0

        # Determine production readiness
        results.ready_for_production = (
            results.success_rate >= 0.8 and
            results.avg_quality_score >= 0.6 and
            results.error_handling_score >= 0.7
        )

        # Generate recommendations
        if results.success_rate < 0.8:
            results.improvements_needed.append("Improve success rate (currently {:.0%})".format(results.success_rate))

        if results.avg_quality_score < 0.6:
            results.improvements_needed.append("Improve decomposition quality (currently {:.2f})".format(results.avg_quality_score))

        if results.avg_decomposition_time > 30:
            results.improvements_needed.append("Optimize decomposition time (currently {:.1f}s)".format(results.avg_decomposition_time))

        if not results.ready_for_production:
            results.improvements_needed.append("Address production readiness issues before deployment")

        logger.info(f"Testing complete. Success rate: {results.success_rate:.1%}, Ready for production: {results.ready_for_production}")
        return results

    def register_strategy(
        self,
        strategy: 'DecompositionStrategyBase',
        engine=None
    ) -> str:
        """
        Register custom strategy with DecompositionEngine.

        Args:
            strategy: Strategy to register
            engine: Optional DecompositionEngine instance

        Returns:
            Strategy ID
        """
        strategy_name = strategy.get_strategy_name()

        # Store in local registry
        self.registered_strategies[strategy_name] = strategy

        # Register with engine if provided
        if engine:
            try:
                engine.register_custom_strategy(strategy)
                logger.info(f"Registered strategy '{strategy_name}' with engine")
            except AttributeError:
                logger.warning("Engine does not support custom strategy registration")
        else:
            logger.info(f"Registered strategy '{strategy_name}' locally (no engine provided)")

        # Log registration
        self.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "registered",
            "strategy_name": strategy_name
        })

        return strategy_name

    def export_strategy(
        self,
        strategy: 'DecompositionStrategyBase',
        output_path: str
    ):
        """
        Export strategy definition for sharing.

        Args:
            strategy: Strategy to export
            output_path: Path to save strategy
        """
        logger.info(f"Exporting strategy: {strategy.get_strategy_name()}")

        # Extract strategy configuration
        if hasattr(strategy, 'config'):
            config = strategy.config
        else:
            # Create basic config from strategy
            config = StrategyConfig(
                strategy_name=strategy.get_strategy_name(),
                description=f"Exported strategy: {strategy.get_strategy_name()}",
                decomposition_criteria={},
                tags=["exported"]
            )

        # Create export data
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "strategy_config": config.to_dict(),
            "strategy_type": "custom"
        }

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Strategy exported to: {output_path}")

    def import_strategy(
        self,
        strategy_path: str
    ) -> 'DecompositionStrategyBase':
        """
        Import strategy definition.

        Args:
            strategy_path: Path to strategy file

        Returns:
            DecompositionStrategyBase instance
        """
        logger.info(f"Importing strategy from: {strategy_path}")

        # Read strategy file
        with open(strategy_path, 'r') as f:
            import_data = json.load(f)

        # Extract config
        config_data = import_data.get("strategy_config", {})
        config = StrategyConfig.from_dict(config_data)

        # Create strategy
        strategy = self.create_strategy(config)

        logger.info(f"Strategy imported: {strategy.get_strategy_name()}")
        return strategy

    def list_registered_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self.registered_strategies.keys())

    def get_strategy(self, strategy_name: str) -> Optional['DecompositionStrategyBase']:
        """Get registered strategy by name."""
        return self.registered_strategies.get(strategy_name)

    def _create_test_problem(self) -> ProblemDefinition:
        """Create a test problem for validation."""
        from sovereign_data_models import ComplexityScore, DomainContext, ProblemType

        return ProblemDefinition(
            id=generate_id("test_problem"),
            title="Test Problem for Strategy Validation",
            description="This is a test problem used to validate custom decomposition strategies.",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(
                domain="testing",
                subdomain="validation"
            ),
            complexity_score=ComplexityScore(
                explanation="Test problem complexity",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )

    def get_strategy_history(self) -> List[Dict[str, Any]]:
        """Get history of strategy operations."""
        return self.strategy_history.copy()
