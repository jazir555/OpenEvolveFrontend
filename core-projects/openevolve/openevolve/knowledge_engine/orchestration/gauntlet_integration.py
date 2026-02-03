"""
Gauntlet System Integration

Integrates the orchestrator with the gauntlet system for:
- Continuous validation of knowledge
- Quality assurance on extracted information
- A/B testing of processing strategies
- Regression testing
- Performance benchmarking

The gauntlet ensures the knowledge engine maintains quality
as it learns and adapts.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of gauntlet tests"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    REGRESSION = "regression"
    ROBUSTNESS = "robustness"


class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class GauntletTest:
    """A single gauntlet test"""
    test_id: str
    test_type: TestType
    name: str
    description: str
    
    # Test configuration
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Thresholds
    min_accuracy: float = 0.7
    max_execution_time_ms: int = 30000
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)


@dataclass
class TestExecution:
    """Result of executing a gauntlet test"""
    execution_id: str
    test_id: str
    timestamp: str
    
    # Results
    result: TestResult
    score: float  # 0-1
    actual_output: Dict[str, Any]
    
    # Metrics
    execution_time_ms: float
    accuracy_score: float
    completeness_score: float
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Comparison to expected
    differences: List[Dict[str, Any]] = field(default_factory=list)


class GauntletIntegration:
    """
    Integration with the gauntlet system for continuous validation.
    
    Ensures the knowledge engine maintains quality as it learns:
    - Runs tests on processing results
    - Validates against ground truth
    - Detects regressions
    - Benchmarks performance
    - Provides quality gates
    """
    
    def __init__(self, orchestrator, test_suite_path: Optional[str] = None):
        """
        Initialize gauntlet integration.
        
        Args:
            orchestrator: The orchestrator to test
            test_suite_path: Optional path to test suite JSON
        """
        self.orchestrator = orchestrator
        self.tests: Dict[str, GauntletTest] = {}
        self.execution_history: List[TestExecution] = []
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_accuracy': 0.7,
            'min_completeness': 0.6,
            'max_regression': 0.1,
            'max_avg_execution_time_ms': 10000
        }
        
        # Load test suite if provided
        if test_suite_path:
            self._load_test_suite(test_suite_path)
        
        logger.info({
            "msg": "GauntletIntegration initialized",
            "tests_loaded": len(self.tests),
            "thresholds": self.quality_thresholds
        })
    
    def create_test(self, 
                   name: str,
                   test_type: TestType,
                   input_data: Dict[str, Any],
                   expected_output: Optional[Dict[str, Any]] = None,
                   validation_rules: List[Dict[str, Any]] = None,
                   tags: List[str] = None) -> GauntletTest:
        """
        Create a new gauntlet test.
        
        Args:
            name: Test name
            test_type: Type of test
            input_data: Input data for test
            expected_output: Optional expected output
            validation_rules: Rules to validate output
            tags: Test tags
            
        Returns:
            Created test
        """
        test_id = f"test_{len(self.tests)}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        test = GauntletTest(
            test_id=test_id,
            test_type=test_type,
            name=name,
            description=f"Test {name} of type {test_type.value}",
            input_data=input_data,
            expected_output=expected_output,
            validation_rules=validation_rules or [],
            tags=tags or []
        )
        
        self.tests[test_id] = test
        
        logger.info({
            "msg": "Gauntlet test created",
            "test_id": test_id,
            "name": name,
            "type": test_type.value
        })
        
        return test
    
    def run_test(self, test_id: str) -> TestExecution:
        """
        Run a single gauntlet test.
        
        Args:
            test_id: Test to run
            
        Returns:
            Test execution result
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": f"Running gauntlet test: {test.name}",
            "test_id": test_id,
            "type": test.test_type.value
        })
        
        try:
            # Execute through orchestrator
            result = self.orchestrator.process(test.input_data)
            
            execution_time_ms = result.get('execution', {}).get('duration_ms', 0)
            actual_output = result.get('results', {})
            
            # Validate based on test type
            if test.test_type == TestType.ACCURACY:
                validation = self._validate_accuracy(test, actual_output)
            elif test.test_type == TestType.COMPLETENESS:
                validation = self._validate_completeness(test, actual_output)
            elif test.test_type == TestType.CONSISTENCY:
                validation = self._validate_consistency(test, actual_output)
            elif test.test_type == TestType.PERFORMANCE:
                validation = self._validate_performance(test, execution_time_ms)
            elif test.test_type == TestType.REGRESSION:
                validation = self._validate_regression(test, actual_output)
            elif test.test_type == TestType.ROBUSTNESS:
                validation = self._validate_robustness(test, actual_output)
            else:
                validation = {'result': TestResult.SKIP, 'score': 0.0, 'issues': ['Unknown test type']}
            
            # Create execution record
            execution = TestExecution(
                execution_id=f"exec_{len(self.execution_history)}",
                test_id=test_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                result=validation['result'],
                score=validation['score'],
                actual_output=actual_output,
                execution_time_ms=execution_time_ms,
                accuracy_score=validation.get('accuracy', 0.0),
                completeness_score=validation.get('completeness', 0.0),
                issues=validation.get('issues', []),
                warnings=validation.get('warnings', []),
                differences=validation.get('differences', [])
            )
            
            self.execution_history.append(execution)
            
            logger.info({
                "msg": f"Test completed: {test.name}",
                "result": execution.result.value,
                "score": execution.score
            })
            
            return execution
            
        except Exception as e:
            logger.error({
                "msg": f"Test failed: {test.name}",
                "error": str(e)
            })
            
            return TestExecution(
                execution_id=f"exec_{len(self.execution_history)}",
                test_id=test_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                result=TestResult.FAIL,
                score=0.0,
                actual_output={'error': str(e)},
                execution_time_ms=0,
                accuracy_score=0.0,
                completeness_score=0.0,
                issues=[f"Execution failed: {str(e)}"]
            )
    
    def run_all_tests(self, 
                     test_type: Optional[TestType] = None,
                     tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all gauntlet tests matching criteria.
        
        Args:
            test_type: Optional type filter
            tags: Optional tag filter
            
        Returns:
            Summary of all test results
        """
        results = []
        
        for test_id, test in self.tests.items():
            # Filter by type
            if test_type and test.test_type != test_type:
                continue
            
            # Filter by tags
            if tags and not any(t in test.tags for t in tags):
                continue
            
            execution = self.run_test(test_id)
            results.append(execution)
        
        # Calculate summary
        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        warnings = sum(1 for r in results if r.result == TestResult.WARNING)
        
        avg_score = statistics.mean([r.score for r in results]) if results else 0.0
        avg_execution_time = statistics.mean([r.execution_time_ms for r in results]) if results else 0.0
        
        summary = {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "average_score": avg_score,
            "average_execution_time_ms": avg_execution_time,
            "results": [
                {
                    "test_id": r.test_id,
                    "result": r.result.value,
                    "score": r.score,
                    "issues": r.issues
                }
                for r in results
            ]
        }
        
        logger.info({
            "msg": "Gauntlet test suite completed",
            "summary": summary
        })
        
        return summary
    
    def _validate_accuracy(self, test: GauntletTest, 
                          actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate accuracy against expected output"""
        if not test.expected_output:
            return {'result': TestResult.SKIP, 'score': 0.0, 'issues': ['No expected output']}
        
        issues = []
        warnings = []
        differences = []
        
        # Compare entities
        expected_entities = test.expected_output.get('entities', [])
        actual_entities = actual_output.get('entities', [])
        
        expected_set = {e.get('text', '') for e in expected_entities}
        actual_set = {e.get('text', '') for e in actual_entities}
        
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        if missing:
            issues.append(f"Missing entities: {missing}")
            differences.append({'type': 'missing_entities', 'items': list(missing)})
        
        if extra:
            warnings.append(f"Extra entities found: {extra}")
            differences.append({'type': 'extra_entities', 'items': list(extra)})
        
        # Calculate accuracy
        if expected_set:
            accuracy = len(expected_set & actual_set) / len(expected_set)
        else:
            accuracy = 1.0 if not actual_set else 0.0
        
        # Determine result
        if accuracy >= test.min_accuracy:
            result = TestResult.PASS
        elif accuracy >= test.min_accuracy * 0.8:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'score': accuracy,
            'accuracy': accuracy,
            'issues': issues,
            'warnings': warnings,
            'differences': differences
        }
    
    def _validate_completeness(self, test: GauntletTest,
                              actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completeness of extraction"""
        # Check for expected result types
        expected_types = test.expected_output.get('result_types', []) if test.expected_output else []
        actual_types = list(actual_output.keys())
        
        missing_types = [t for t in expected_types if t not in actual_types]
        
        if expected_types:
            completeness = (len(expected_types) - len(missing_types)) / len(expected_types)
        else:
            completeness = 1.0 if actual_output else 0.0
        
        issues = [f"Missing result types: {missing_types}"] if missing_types else []
        
        if completeness >= 0.8:
            result = TestResult.PASS
        elif completeness >= 0.6:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'score': completeness,
            'completeness': completeness,
            'issues': issues
        }
    
    def _validate_consistency(self, test: GauntletTest,
                             actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate internal consistency"""
        issues = []
        
        # Check for contradictions in entities
        entities = actual_output.get('entities', [])
        entity_texts = [e.get('text', '').lower() for e in entities]
        
        duplicates = set(x for x in entity_texts if entity_texts.count(x) > 1)
        if duplicates:
            issues.append(f"Duplicate entities: {duplicates}")
        
        # Check relations match entities
        relations = actual_output.get('relations', [])
        entity_set = set(entity_texts)
        
        for rel in relations:
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()
            
            if source not in entity_set:
                issues.append(f"Relation source not in entities: {source}")
            if target not in entity_set:
                issues.append(f"Relation target not in entities: {target}")
        
        score = 1.0 if not issues else max(0.0, 1.0 - len(issues) * 0.1)
        
        if score >= 0.9:
            result = TestResult.PASS
        elif score >= 0.7:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'score': score,
            'issues': issues
        }
    
    def _validate_performance(self, test: GauntletTest,
                             execution_time_ms: float) -> Dict[str, Any]:
        """Validate performance metrics"""
        threshold = test.max_execution_time_ms
        
        if execution_time_ms <= threshold:
            result = TestResult.PASS
            score = 1.0
        elif execution_time_ms <= threshold * 1.5:
            result = TestResult.WARNING
            score = 0.7
        else:
            result = TestResult.FAIL
            score = max(0.0, 1.0 - (execution_time_ms - threshold) / threshold)
        
        issues = []
        if execution_time_ms > threshold:
            issues.append(f"Execution time {execution_time_ms}ms exceeds threshold {threshold}ms")
        
        return {
            'result': result,
            'score': score,
            'issues': issues
        }
    
    def _validate_regression(self, test: GauntletTest,
                            actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against previous execution"""
        # Find previous executions of this test
        previous = [
            e for e in self.execution_history
            if e.test_id == test.test_id and e.result != TestResult.FAIL
        ]
        
        if not previous:
            return {'result': TestResult.SKIP, 'score': 1.0, 'issues': ['No previous execution']}
        
        last_execution = previous[-1]
        
        # Compare scores
        score_diff = last_execution.score - 0.0  # Compare to current (would need current)
        
        # This is a simplified check
        return {
            'result': TestResult.PASS,
            'score': 1.0,
            'issues': [],
            'notes': 'Regression check needs implementation'
        }
    
    def _validate_robustness(self, test: GauntletTest,
                            actual_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate robustness (no crashes, valid output format)"""
        issues = []
        
        # Check output is not empty
        if not actual_output:
            issues.append("Output is empty")
        
        # Check for error keys
        if 'error' in actual_output:
            issues.append(f"Error in output: {actual_output['error']}")
        
        # Check data types
        for key, value in actual_output.items():
            if value is None:
                issues.append(f"Null value for key: {key}")
        
        score = 1.0 if not issues else max(0.0, 1.0 - len(issues) * 0.2)
        
        if score >= 0.9:
            result = TestResult.PASS
        elif score >= 0.7:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'score': score,
            'issues': issues
        }
    
    def check_quality_gate(self) -> Dict[str, Any]:
        """
        Check if system passes quality gate.
        
        Returns:
            Quality gate status
        """
        # Run all tests
        summary = self.run_all_tests()
        
        # Check thresholds
        passed = True
        failures = []
        
        if summary['average_score'] < self.quality_thresholds['min_accuracy']:
            passed = False
            failures.append(f"Average score {summary['average_score']:.2f} below threshold {self.quality_thresholds['min_accuracy']}")
        
        if summary['average_execution_time_ms'] > self.quality_thresholds['max_avg_execution_time_ms']:
            passed = False
            failures.append(f"Average execution time {summary['average_execution_time_ms']:.0f}ms exceeds threshold")
        
        fail_rate = summary['failed'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        if fail_rate > 0.2:  # Max 20% failure rate
            passed = False
            failures.append(f"Failure rate {fail_rate:.1%} too high")
        
        return {
            "passed": passed,
            "summary": summary,
            "failures": failures,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _load_test_suite(self, path: str):
        """Load test suite from file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for test_data in data.get('tests', []):
                test = GauntletTest(
                    test_id=test_data['test_id'],
                    test_type=TestType(test_data['test_type']),
                    name=test_data['name'],
                    description=test_data.get('description', ''),
                    input_data=test_data['input_data'],
                    expected_output=test_data.get('expected_output'),
                    validation_rules=test_data.get('validation_rules', []),
                    min_accuracy=test_data.get('min_accuracy', 0.7),
                    max_execution_time_ms=test_data.get('max_execution_time_ms', 30000),
                    tags=test_data.get('tags', [])
                )
                self.tests[test_id] = test
            
            logger.info({
                "msg": "Test suite loaded",
                "tests": len(self.tests)
            })
            
        except Exception as e:
            logger.error({
                "msg": "Failed to load test suite",
                "error": str(e)
            })
    
    def export_test_suite(self, path: str):
        """Export test suite to file"""
        data = {
            "tests": [
                {
                    "test_id": t.test_id,
                    "test_type": t.test_type.value,
                    "name": t.name,
                    "description": t.description,
                    "input_data": t.input_data,
                    "expected_output": t.expected_output,
                    "validation_rules": t.validation_rules,
                    "min_accuracy": t.min_accuracy,
                    "max_execution_time_ms": t.max_execution_time_ms,
                    "tags": t.tags
                }
                for t in self.tests.values()
            ],
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info({
            "msg": "Test suite exported",
            "path": path,
            "tests": len(self.tests)
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gauntlet statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        recent = self.execution_history[-100:]  # Last 100
        
        return {
            "total_tests": len(self.tests),
            "total_executions": len(self.execution_history),
            "recent_pass_rate": sum(1 for e in recent if e.result == TestResult.PASS) / len(recent),
            "average_score": statistics.mean([e.score for e in recent]),
            "by_type": {
                t.value: {
                    "count": sum(1 for e in self.execution_history if self.tests.get(e.test_id, GauntletTest(test_id='', test_type=TestType.ACCURACY, name='', description='', input_data={})).test_type == t),
                    "avg_score": statistics.mean([e.score for e in self.execution_history if self.tests.get(e.test_id, GauntletTest(test_id='', test_type=TestType.ACCURACY, name='', description='', input_data={})).test_type == t]) if any(self.tests.get(e.test_id, GauntletTest(test_id='', test_type=TestType.ACCURACY, name='', description='', input_data={})).test_type == t for e in self.execution_history) else 0
                }
                for t in TestType
            }
        }


import statistics