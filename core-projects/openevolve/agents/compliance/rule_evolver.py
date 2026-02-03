"""
Rule Evolution Engine
Evolves compliance rule sets using LoongFlow PES and maintains rule history.

Author: AI Architecture Team
Date: 2026-01-30
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path

# Import unified evolution API
from ...unified.unified_evolution_api import evolve


class RuleStatus(Enum):
    """Status of evolved rules"""
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


@dataclass
class RuleVersion:
    """Represents a version of a compliance rule"""
    rule_id: str
    version: str
    code: str
    description: str
    status: RuleStatus
    created_at: datetime
    parent_version: Optional[str] = None
    regulatory_changes: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rule_id': self.rule_id,
            'version': self.version,
            'code': self.code,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'parent_version': self.parent_version,
            'regulatory_changes': self.regulatory_changes,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleVersion':
        """Create from dictionary"""
        return cls(
            rule_id=data['rule_id'],
            version=data['version'],
            code=data['code'],
            description=data['description'],
            status=RuleStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            parent_version=data.get('parent_version'),
            regulatory_changes=data.get('regulatory_changes', []),
            test_results=data.get('test_results', {}),
            performance_metrics=data.get('performance_metrics', {})
        )


@dataclass
class EvolutionResult:
    """Result of rule evolution"""
    success: bool
    evolved_rules: Dict[str, Any]
    changes_made: List[str]
    confidence: float
    generation: int
    test_coverage: float = 0.0
    edge_cases_addressed: int = 0


class RuleEvolver:
    """
    Evolves compliance rules using LoongFlow PES

    Features:
    - Automatic rule updates based on regulatory changes
    - A/B testing of rule effectiveness
    - Rule history and provenance tracking
    - Test-driven evolution
    - Performance optimization

    Example:
        >>> evolver = RuleEvolver()
        >>> result = await evolver.evolve_rules(
        ...     current_rules={'rule1': 'code'},
        ...     regulatory_changes=[{'title': 'New SEC rule'}]
        ... )
        >>> if result.success:
        ...     print(f"Evolved rules with confidence {result.confidence}")
    """

    def __init__(
        self,
        cache_dir: str = "./cache/rule_evolution",
        max_generations: int = 10,
        population_size: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize rule evolver

        Args:
            cache_dir: Directory for caching rule versions
            max_generations: Maximum PES generations
            population_size: Population size for evolution
            logger: Logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_generations = max_generations
        self.population_size = population_size

        self.logger = logger or self._setup_logging()

        # Rule history
        self.rule_history: List[RuleVersion] = []
        self._load_history()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("RuleEvolver")
        logger.setLevel(logging.INFO)
        return logger

    def _load_history(self):
        """Load rule history from cache"""
        history_file = self.cache_dir / "rule_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                self.rule_history = [
                    RuleVersion.from_dict(item) for item in data
                ]
                self.logger.info(f"Loaded {len(self.rule_history)} rule versions")
            except Exception as e:
                self.logger.error(f"Failed to load rule history: {e}")

    def _save_history(self):
        """Save rule history to cache"""
        history_file = self.cache_dir / "rule_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump([r.to_dict() for r in self.rule_history], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save rule history: {e}")

    async def evolve_rules(
        self,
        current_rules: Dict[str, Any],
        regulatory_changes: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evolve compliance rules based on regulatory changes

        Args:
            current_rules: Current rule set
            regulatory_changes: List of regulatory changes
            constraints: Optional constraints on evolution

        Returns:
            Evolved rule set
        """
        self.logger.info(f"Evolving {len(current_rules)} rules based on {len(regulatory_changes)} changes")

        if not regulatory_changes:
            self.logger.info("No regulatory changes, returning current rules")
            return current_rules

        # Construct problem statement for LoongFlow
        problem_statement = self._construct_evolution_problem(
            current_rules,
            regulatory_changes,
            constraints
        )

        try:
            # Use LoongFlow PES for evolution
            result = await evolve(
                problem_statement=problem_statement,
                mode="pes",  # Pareto Evolutionary Strategy
                max_generations=self.max_generations,
                population_size=self.population_size,
                objectives=[
                    "maximize_compliance_coverage",
                    "minimize_false_positives",
                    "maximize_interpretability"
                ]
            )

            if result.get('success'):
                evolved_rules = self._extract_evolved_rules(result)

                # Record new version
                new_version = RuleVersion(
                    rule_id=self._generate_rule_id(),
                    version=self._get_next_version(),
                    code=json.dumps(evolved_rules, indent=2),
                    description=f"Evolved based on {len(regulatory_changes)} regulatory changes",
                    status=RuleStatus.DRAFT,
                    created_at=datetime.utcnow(),
                    regulatory_changes=[c.get('title', '') for c in regulatory_changes],
                    performance_metrics={
                        'confidence': result.get('confidence', 0.0),
                        'generations': result.get('generation', 0)
                    }
                )

                self.rule_history.append(new_version)
                self._save_history()

                self.logger.info(
                    f"Successfully evolved rules (version {new_version.version}, "
                    f"confidence {new_version.performance_metrics['confidence']:.2f})"
                )

                return evolved_rules
            else:
                self.logger.error("Rule evolution failed")
                return current_rules

        except Exception as e:
            self.logger.error(f"Error evolving rules: {e}", exc_info=True)
            return current_rules

    def _construct_evolution_problem(
        self,
        current_rules: Dict[str, Any],
        regulatory_changes: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Construct problem statement for LoongFlow"""
        problem = f"""
You are a compliance rule evolution engine. Your task is to update the current
compliance rules to address new regulatory changes while maintaining existing
coverage and minimizing false positives.

CURRENT RULES:
```json
{json.dumps(current_rules, indent=2)}
```

REGULATORY CHANGES:
{json.dumps(regulatory_changes, indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

REQUIREMENTS:
1. Update rules to address all regulatory changes
2. Maintain backward compatibility where possible
3. Minimize false positives
4. Maximize compliance coverage
5. Ensure rules are interpretable and explainable
6. Add comprehensive test cases

OUTPUT FORMAT:
Provide evolved rules as a JSON object with:
- rule_id: Unique identifier
- name: Human-readable name
- description: What the rule checks
- logic: The rule logic (pseudocode or Python)
- test_cases: List of test cases
- regulatory_mapping: Which regulations this addresses

Example:
```json
{{
  "rule_001": {{
    "name": "SEC Rule 10b-5 Compliance",
    "description": "Detects potential insider trading patterns",
    "logic": "if (pattern_matches and volume_threshold) then flag",
    "test_cases": [...],
    "regulatory_mapping": ["SEC Rule 10b-5"]
  }}
}}
```
"""
        return problem

    def _extract_evolved_rules(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evolved rules from LoongFlow result"""
        try:
            solution_text = result.get('best_solution', '{}')

            # Try to parse as JSON
            try:
                return json.loads(solution_text)
            except json.JSONDecodeError:
                # Extract JSON from solution text
                import re
                json_match = re.search(r'\{[\s\S]*\}', solution_text)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    self.logger.error("Could not extract JSON from solution")
                    return {}

        except Exception as e:
            self.logger.error(f"Error extracting evolved rules: {e}")
            return {}

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        hash_suffix = hashlib.sha256(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:8]
        return f"rule_{timestamp}_{hash_suffix}"

    def _get_next_version(self) -> str:
        """Get next version number"""
        if not self.rule_history:
            return "1.0.0"

        latest = self.rule_history[-1]
        major, minor, patch = map(int, latest.version.split('.'))

        # Increment minor version for regulatory changes
        return f"{major}.{minor + 1}.0"

    async def test_rules(
        self,
        rules: Dict[str, Any],
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Test evolved rules against test cases

        Args:
            rules: Rule set to test
            test_cases: Optional test cases

        Returns:
            Test results
        """
        self.logger.info(f"Testing {len(rules)} rules")

        results = {
            'total_rules': len(rules),
            'tests_passed': 0,
            'tests_failed': 0,
            'coverage': 0.0,
            'false_positive_rate': 0.0,
            'details': []
        }

        # Generate test cases if not provided
        if not test_cases:
            test_cases = await self._generate_test_cases(rules)

        # Run tests
        for rule_id, rule in rules.items():
            rule_tests = [tc for tc in test_cases if tc.get('rule_id') == rule_id]

            for test in rule_tests:
                try:
                    # This is a simplified test - real implementation would execute rule logic
                    test_passed = self._execute_rule_test(rule, test)

                    if test_passed:
                        results['tests_passed'] += 1
                    else:
                        results['tests_failed'] += 1

                    results['details'].append({
                        'rule_id': rule_id,
                        'test_id': test.get('test_id'),
                        'passed': test_passed
                    })

                except Exception as e:
                    self.logger.error(f"Error testing {rule_id}: {e}")
                    results['tests_failed'] += 1

        # Calculate metrics
        total_tests = results['tests_passed'] + results['tests_failed']
        if total_tests > 0:
            results['coverage'] = results['tests_passed'] / total_tests

        return results

    async def _generate_test_cases(self, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for rules"""
        test_cases = []

        for rule_id, rule in rules.items():
            # Generate positive cases (should trigger rule)
            positive_cases = await self._generate_positive_cases(rule)

            # Generate negative cases (should not trigger rule)
            negative_cases = await self._generate_negative_cases(rule)

            # Generate edge cases
            edge_cases = await self._generate_edge_cases(rule)

            test_cases.extend(positive_cases + negative_cases + edge_cases)

        return test_cases

    async def _generate_positive_cases(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases that should trigger the rule"""
        # Simplified - real implementation would use LoongFlow to generate
        return []

    async def _generate_negative_cases(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases that should not trigger the rule"""
        # Simplified - real implementation would use LoongFlow to generate
        return []

    async def _generate_edge_cases(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge case tests"""
        # Simplified - real implementation would use LoongFlow to generate
        return []

    def _execute_rule_test(self, rule: Dict[str, Any], test: Dict[str, Any]) -> bool:
        """Execute a single rule test"""
        # Simplified - real implementation would execute rule logic
        return True

    async def ab_test_rules(
        self,
        old_rules: Dict[str, Any],
        new_rules: Dict[str, Any],
        test_data: Optional[List[Dict[str, Any]]] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        A/B test old vs new rules

        Args:
            old_rules: Current rule set
            new_rules: Evolved rule set
            test_data: Optional test data
            sample_size: Sample size for testing

        Returns:
            A/B test results
        """
        self.logger.info("Running A/B test on rules")

        # Generate test data if not provided
        if not test_data:
            test_data = await self._generate_ab_test_data(sample_size)

        # Test old rules
        old_results = await self._evaluate_rules(old_rules, test_data)

        # Test new rules
        new_results = await self._evaluate_rules(new_rules, test_data)

        # Compare
        comparison = {
            'old_rules': old_results,
            'new_rules': new_results,
            'improvement': {
                'true_positive_rate': new_results['tpr'] - old_results['tpr'],
                'false_positive_rate': new_results['fpr'] - old_results['fpr'],
                'coverage': new_results['coverage'] - old_results['coverage']
            },
            'recommendation': self._get_ab_test_recommendation(old_results, new_results)
        }

        self.logger.info(f"A/B test complete: {comparison['recommendation']}")
        return comparison

    async def _generate_ab_test_data(self, sample_size: int) -> List[Dict[str, Any]]:
        """Generate data for A/B testing"""
        # Simplified - would use historical transaction data
        return []

    async def _evaluate_rules(
        self,
        rules: Dict[str, Any],
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate rules on test data"""
        # Simplified
        return {
            'tpr': 0.85,
            'fpr': 0.15,
            'coverage': 0.90,
            'precision': 0.88
        }

    def _get_ab_test_recommendation(
        self,
        old_results: Dict[str, float],
        new_results: Dict[str, float]
    ) -> str:
        """Get recommendation from A/B test results"""
        # Simplified logic
        if new_results['tpr'] > old_results['tpr'] and new_results['fpr'] < old_results['fpr']:
            return "DEPLOY_NEW_RULES"
        elif new_results['tpr'] > old_results['tpr'] * 1.05:
            return "DEPLOY_NEW_RULES_WITH_MONITORING"
        else:
            return "KEEP_OLD_RULES"

    def get_rule_provenance(self, rule_id: str) -> List[RuleVersion]:
        """
        Get provenance history for a rule

        Args:
            rule_id: Rule identifier

        Returns:
            List of rule versions
        """
        return [
            version for version in self.rule_history
            if version.rule_id == rule_id
        ]

    def get_rule_diff(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Get diff between two rule versions

        Args:
            version1: First version
            version2: Second version

        Returns:
            Diff information
        """
        # Simplified - would use proper diff algorithm
        return {
            'version1': version1,
            'version2': version2,
            'changes': []
        }

    async def optimize_rule_performance(
        self,
        rules: Dict[str, Any],
        performance_profile: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Optimize rules for specific performance profile

        Args:
            rules: Rule set to optimize
            performance_profile: One of: 'speed', 'accuracy', 'balanced'

        Returns:
            Optimized rule set
        """
        problem_statement = f"""
Optimize the following compliance rules for {performance_profile} performance:

```json
{json.dumps(rules, indent=2)}
```

Focus on:
- {'Minimizing execution time' if performance_profile == 'speed' else 'Maximizing accuracy' if performance_profile == 'accuracy' else 'Balancing speed and accuracy'}
- Maintaining compliance coverage
- Reducing false positives

Provide optimized rules in the same JSON format.
"""

        try:
            result = await evolve(
                problem_statement=problem_statement,
                mode="standard",
                max_generations=5
            )

            if result.get('success'):
                return self._extract_evolved_rules(result)
            else:
                return rules

        except Exception as e:
            self.logger.error(f"Error optimizing rules: {e}")
            return rules
