"""
Edge Case Discovery Module
Discovers edge cases and coverage gaps using adversarial testing and fuzzing.

Author: AI Architecture Team
Date: 2026-01-30
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import random
from pathlib import Path

# Import unified evolution API
from ...unified.unified_evolution_api import evolve


class EdgeCaseType(Enum):
    """Types of edge cases"""
    BOUNDARY = "boundary"  # Boundary conditions
    COMBINATORIAL = "combinatorial"  # Unexpected combinations
    ADVERSARIAL = "adversarial"  # Malicious inputs
    DATA_EDGE = "data_edge"  # Unusual data patterns
    TIMING = "timing"  # Race conditions, timing issues
    SCALE = "scale"  # Large volumes, extreme values
    REGULATORY = "regulatory"  # Regulatory gray areas
    LOGICAL = "logical"  # Logic gaps


@dataclass
class EdgeCase:
    """Represents an edge case"""
    case_id: str
    case_type: EdgeCaseType
    description: str
    scenario: Dict[str, Any]
    expected_behavior: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    affected_rules: List[str] = field(default_factory=list)
    addressed: bool = False
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    mitigation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'case_id': self.case_id,
            'case_type': self.case_type.value,
            'description': self.description,
            'scenario': self.scenario,
            'expected_behavior': self.expected_behavior,
            'severity': self.severity,
            'affected_rules': self.affected_rules,
            'addressed': self.addressed,
            'discovered_at': self.discovered_at.isoformat(),
            'mitigation': self.mitigation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeCase':
        """Create from dictionary"""
        return cls(
            case_id=data['case_id'],
            case_type=EdgeCaseType(data['case_type']),
            description=data['description'],
            scenario=data['scenario'],
            expected_behavior=data['expected_behavior'],
            severity=data['severity'],
            affected_rules=data.get('affected_rules', []),
            addressed=data.get('addressed', False),
            discovered_at=datetime.fromisoformat(data['discovered_at']),
            mitigation=data.get('mitigation')
        )


@dataclass
class CoverageReport:
    """Coverage analysis report"""
    total_rules: int
    covered_scenarios: int
    total_scenarios: int
    coverage_percentage: float
    edge_cases_found: int
    critical_gaps: List[str]
    recommendations: List[str]


class EdgeCaseDiscovery:
    """
    Discovers edge cases and coverage gaps in compliance rules

    Methods:
    - Adversarial testing: Try to break rules with malicious inputs
    - Fuzz testing: Random inputs to find edge cases
    - Combinatorial testing: Test combinations of conditions
    - Boundary testing: Test limits and thresholds
    - Scenario generation: Generate realistic edge cases

    Example:
        >>> discovery = EdgeCaseDiscovery()
        >>> cases = await discovery.discover_cases(rules={'rule1': {...}})
        >>> print(f"Found {len(cases)} edge cases")
        >>> coverage = await discovery.analyze_coverage(rules)
        >>> print(f"Coverage: {coverage.coverage_percentage}%")
    """

    def __init__(
        self,
        cache_dir: str = "./cache/edge_cases",
        max_adversarial_iterations: int = 100,
        max_fuzz_iterations: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize edge case discovery

        Args:
            cache_dir: Directory for caching edge cases
            max_adversarial_iterations: Max iterations for adversarial testing
            max_fuzz_iterations: Max iterations for fuzz testing
            logger: Logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_adversarial_iterations = max_adversarial_iterations
        self.max_fuzz_iterations = max_fuzz_iterations

        self.logger = logger or self._setup_logging()

        # Edge case repository
        self.edge_cases: List[EdgeCase] = []
        self._load_edge_cases()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("EdgeCaseDiscovery")
        logger.setLevel(logging.INFO)
        return logger

    def _load_edge_cases(self):
        """Load edge cases from cache"""
        cache_file = self.cache_dir / "edge_cases.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                self.edge_cases = [EdgeCase.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.edge_cases)} edge cases")
            except Exception as e:
                self.logger.error(f"Failed to load edge cases: {e}")

    def _save_edge_cases(self):
        """Save edge cases to cache"""
        cache_file = self.cache_dir / "edge_cases.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump([c.to_dict() for c in self.edge_cases], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save edge cases: {e}")

    async def discover_cases(
        self,
        rules: Dict[str, Any],
        discovery_methods: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover edge cases using multiple methods

        Args:
            rules: Compliance rules to test
            discovery_methods: Methods to use (default: all)

        Returns:
            List of discovered edge cases
        """
        self.logger.info(f"Discovering edge cases for {len(rules)} rules")

        if not discovery_methods:
            discovery_methods = [
                'adversarial',
                'fuzz',
                'boundary',
                'combinatorial',
                'scenario'
            ]

        all_cases = []

        # Adversarial testing
        if 'adversarial' in discovery_methods:
            adversarial_cases = await self._adversarial_testing(rules)
            all_cases.extend(adversarial_cases)
            self.logger.info(f"Adversarial testing found {len(adversarial_cases)} cases")

        # Fuzz testing
        if 'fuzz' in discovery_methods:
            fuzz_cases = await self._fuzz_testing(rules)
            all_cases.extend(fuzz_cases)
            self.logger.info(f"Fuzz testing found {len(fuzz_cases)} cases")

        # Boundary testing
        if 'boundary' in discovery_methods:
            boundary_cases = await self._boundary_testing(rules)
            all_cases.extend(boundary_cases)
            self.logger.info(f"Boundary testing found {len(boundary_cases)} cases")

        # Combinatorial testing
        if 'combinatorial' in discovery_methods:
            combinatorial_cases = await self._combinatorial_testing(rules)
            all_cases.extend(combinatorial_cases)
            self.logger.info(f"Combinatorial testing found {len(combinatorial_cases)} cases")

        # Scenario generation
        if 'scenario' in discovery_methods:
            scenario_cases = await self._scenario_generation(rules)
            all_cases.extend(scenario_cases)
            self.logger.info(f"Scenario generation found {len(scenario_cases)} cases")

        # Remove duplicates and save
        unique_cases = self._deduplicate_cases(all_cases)
        self.edge_cases.extend(unique_cases)
        self._save_edge_cases()

        self.logger.info(f"Total unique edge cases discovered: {len(unique_cases)}")

        return [case.to_dict() for case in unique_cases]

    async def _adversarial_testing(self, rules: Dict[str, Any]) -> List[EdgeCase]:
        """
        Adversarial testing - try to break rules with malicious inputs

        Uses LoongFlow to generate adversarial scenarios
        """
        cases = []

        problem_statement = f"""
You are an adversarial tester trying to find vulnerabilities in compliance rules.
Generate adversarial test scenarios that might bypass or break these rules:

```json
{json.dumps(rules, indent=2)}
```

For each scenario, provide:
1. Description of the adversarial approach
2. Input data (simulated malicious activity)
3. Expected behavior (what the rule should catch)
4. Why it might bypass the rule

Generate {self.max_adversarial_iterations} adversarial scenarios.

Focus on:
- Regulatory arbitrage opportunities
- Timing-based exploits
- Obfuscation techniques
- Boundary exploitation
- Logic gaps

Output as JSON array.
"""

        try:
            result = await evolve(
                problem_statement=problem_statement,
                mode="standard",
                max_generations=3
            )

            if result.get('success'):
                scenarios = self._extract_scenarios(result.get('best_solution', ''))

                for scenario in scenarios[:self.max_adversarial_iterations]:
                    case = EdgeCase(
                        case_id=f"adv_{len(self.edge_cases) + len(cases)}",
                        case_type=EdgeCaseType.ADVERSARIAL,
                        description=scenario.get('description', ''),
                        scenario=scenario.get('input', {}),
                        expected_behavior=scenario.get('expected', ''),
                        severity='critical',
                        affected_rules=scenario.get('affected_rules', [])
                    )
                    cases.append(case)

        except Exception as e:
            self.logger.error(f"Error in adversarial testing: {e}")

        return cases

    async def _fuzz_testing(self, rules: Dict[str, Any]) -> List[EdgeCase]:
        """
        Fuzz testing - random inputs to find unexpected behavior
        """
        cases = []

        # Generate random test scenarios
        for i in range(self.max_fuzz_iterations):
            scenario = self._generate_random_scenario(rules)

            # Check if it triggers unexpected behavior
            is_edge_case = await self._test_scenario(scenario, rules)

            if is_edge_case:
                case = EdgeCase(
                    case_id=f"fuzz_{len(self.edge_cases) + len(cases)}",
                    case_type=EdgeCaseType.DATA_EDGE,
                    description=f"Random fuzz test #{i}",
                    scenario=scenario,
                    expected_behavior="Should handle gracefully",
                    severity='low',
                    affected_rules=[]
                )
                cases.append(case)

        return cases

    async def _boundary_testing(self, rules: Dict[str, Any]) -> List[EdgeCase]:
        """
        Boundary testing - test limits and thresholds
        """
        cases = []

        # Extract thresholds from rules
        thresholds = self._extract_thresholds(rules)

        for threshold in thresholds:
            # Test at boundary
            case_at = EdgeCase(
                case_id=f"bound_at_{len(self.edge_cases) + len(cases)}",
                case_type=EdgeCaseType.BOUNDARY,
                description=f"Value at threshold {threshold['name']}",
                scenario={'value': threshold['value']},
                expected_behavior=f"Should trigger rule at {threshold['value']}",
                severity='medium',
                affected_rules=[threshold['rule_id']]
            )
            cases.append(case_at)

            # Test just above boundary
            case_above = EdgeCase(
                case_id=f"bound_above_{len(self.edge_cases) + len(cases)}",
                case_type=EdgeCaseType.BOUNDARY,
                description=f"Value just above threshold {threshold['name']}",
                scenario={'value': threshold['value'] * 1.001},
                expected_behavior=f"Should trigger rule above {threshold['value']}",
                severity='high',
                affected_rules=[threshold['rule_id']]
            )
            cases.append(case_above)

            # Test just below boundary
            case_below = EdgeCase(
                case_id=f"bound_below_{len(self.edge_cases) + len(cases)}",
                case_type=EdgeCaseType.BOUNDARY,
                description=f"Value just below threshold {threshold['name']}",
                scenario={'value': threshold['value'] * 0.999},
                expected_behavior=f"Should not trigger rule below {threshold['value']}",
                severity='high',
                affected_rules=[threshold['rule_id']]
            )
            cases.append(case_below)

        return cases

    async def _combinatorial_testing(self, rules: Dict[str, Any]) -> List[EdgeCase]:
        """
        Combinatorial testing - test combinations of conditions
        """
        cases = []

        # Get rule conditions
        conditions = self._extract_conditions(rules)

        # Generate combinations
        for i, combination in enumerate(self._generate_combinations(conditions)):
            if i >= 50:  # Limit combinations
                break

            case = EdgeCase(
                case_id=f"comb_{len(self.edge_cases) + len(cases)}",
                case_type=EdgeCaseType.COMBINATORIAL,
                description=f"Combination #{i}: {combination['description']}",
                scenario=combination['scenario'],
                expected_behavior="Should handle this combination correctly",
                severity='medium',
                affected_rules=combination['affected_rules']
            )
            cases.append(case)

        return cases

    async def _scenario_generation(self, rules: Dict[str, Any]) -> List[EdgeCase]:
        """
        Scenario generation - generate realistic edge case scenarios
        """
        cases = []

        problem_statement = f"""
Generate realistic edge case scenarios for these compliance rules:

```json
{json.dumps(rules, indent=2)}
```

Generate scenarios that are:
- Plausible in real-world operations
- Unusual but not impossible
- Potentially missed by standard testing
- Representative of "gray areas" in regulations

For each scenario, provide:
1. Description
2. Context/motivation
3. Specific inputs
4. Expected compliance determination

Generate 10-20 scenarios. Output as JSON array.
"""

        try:
            result = await evolve(
                problem_statement=problem_statement,
                mode="standard",
                max_generations=3
            )

            if result.get('success'):
                scenarios = self._extract_scenarios(result.get('best_solution', ''))

                for scenario in scenarios:
                    case = EdgeCase(
                        case_id=f"scenario_{len(self.edge_cases) + len(cases)}",
                        case_type=EdgeCaseType.REGULATORY,
                        description=scenario.get('description', ''),
                        scenario=scenario,
                        expected_behavior=scenario.get('expected', ''),
                        severity='medium',
                        affected_rules=scenario.get('affected_rules', [])
                    )
                    cases.append(case)

        except Exception as e:
            self.logger.error(f"Error in scenario generation: {e}")

        return cases

    def _extract_thresholds(self, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract numeric thresholds from rules"""
        thresholds = []

        for rule_id, rule in rules.items():
            # Look for common threshold patterns
            description = str(rule.get('description', '')) + str(rule.get('logic', ''))

            # Pattern matching for thresholds
            import re
            patterns = [
                (r'(\d+)%', 'percent'),
                (r'\$(\d+)', 'dollars'),
                (r'(\d+) days?', 'days'),
                (r'(\d+) trades?', 'trades'),
            ]

            for pattern, unit in patterns:
                matches = re.findall(pattern, description)
                for match in matches:
                    thresholds.append({
                        'rule_id': rule_id,
                        'name': f"{rule_id}_{unit}_{match}",
                        'value': float(match),
                        'unit': unit
                    })

        return thresholds

    def _extract_conditions(self, rules: Dict[str, Any]) -> List[str]:
        """Extract conditions from rules"""
        conditions = []

        for rule_id, rule in rules.items():
            logic = str(rule.get('logic', ''))
            # Simplified - would extract actual conditions
            conditions.append(f"{rule_id}: {logic[:100]}")

        return conditions

    def _generate_combinations(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate combinations of conditions"""
        combinations = []

        # Simplified - would use proper combinatorial algorithm
        for i in range(min(50, len(conditions) * 2)):
            # Randomly select 2-3 conditions
            selected = random.sample(conditions, min(3, len(conditions)))
            combinations.append({
                'description': f"Combination of {len(selected)} conditions",
                'scenario': {'conditions': selected},
                'affected_rules': []
            })

        return combinations

    def _generate_random_scenario(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random test scenario"""
        return {
            'transaction_type': random.choice(['buy', 'sell', 'transfer']),
            'amount': random.uniform(1, 1000000),
            'timestamp': datetime.utcnow().isoformat(),
            'random_field': random.random()
        }

    async def _test_scenario(
        self,
        scenario: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> bool:
        """Test if scenario triggers unexpected behavior"""
        # Simplified - real implementation would execute rules
        return random.random() < 0.1  # 10% chance of being edge case

    def _extract_scenarios(self, solution_text: str) -> List[Dict[str, Any]]:
        """Extract scenarios from LoongFlow solution"""
        try:
            # Try to parse as JSON
            try:
                return json.loads(solution_text)
            except json.JSONDecodeError:
                # Extract JSON from text
                import re
                json_match = re.search(r'\[[\s\S]*\]', solution_text)
                if json_match:
                    return json.loads(json_match.group())
                return []
        except Exception:
            return []

    def _deduplicate_cases(self, cases: List[EdgeCase]) -> List[EdgeCase]:
        """Remove duplicate edge cases"""
        seen = set()
        unique = []

        for case in cases:
            # Create signature from description and type
            signature = f"{case.case_type.value}:{case.description[:100]}"
            if signature not in seen:
                seen.add(signature)
                unique.append(case)

        return unique

    async def analyze_coverage(
        self,
        rules: Dict[str, Any]
    ) -> CoverageReport:
        """
        Analyze test coverage of compliance rules

        Args:
            rules: Compliance rules

        Returns:
            Coverage report
        """
        self.logger.info("Analyzing rule coverage")

        # Get relevant edge cases
        relevant_cases = [
            case for case in self.edge_cases
            if any(rule in case.affected_rules for rule in rules.keys())
        ]

        # Count covered scenarios
        covered_scenarios = sum(
            1 for case in relevant_cases
            if case.addressed
        )

        # Find critical gaps
        critical_gaps = [
            case.case_id for case in relevant_cases
            if case.severity in ['critical', 'high'] and not case.addressed
        ]

        # Generate recommendations
        recommendations = []
        if critical_gaps:
            recommendations.append(
                f"Address {len(critical_gaps)} critical/high severity gaps"
            )

        coverage_percentage = (
            covered_scenarios / len(relevant_cases) * 100
            if relevant_cases else 100.0
        )

        return CoverageReport(
            total_rules=len(rules),
            covered_scenarios=covered_scenarios,
            total_scenarios=len(relevant_cases),
            coverage_percentage=coverage_percentage,
            edge_cases_found=len(relevant_cases),
            critical_gaps=critical_gaps,
            recommendations=recommendations
        )

    def get_unaddressed_cases(self, severity: Optional[str] = None) -> List[EdgeCase]:
        """
        Get unaddressed edge cases

        Args:
            severity: Optional severity filter

        Returns:
            List of unaddressed edge cases
        """
        cases = [c for c in self.edge_cases if not c.addressed]

        if severity:
            cases = [c for c in cases if c.severity == severity]

        return cases

    def mark_case_addressed(self, case_id: str, mitigation: str):
        """
        Mark an edge case as addressed

        Args:
            case_id: Case identifier
            mitigation: Mitigation strategy
        """
        for case in self.edge_cases:
            if case.case_id == case_id:
                case.addressed = True
                case.mitigation = mitigation
                self.logger.info(f"Marked case {case_id} as addressed")
                break

        self._save_edge_cases()
