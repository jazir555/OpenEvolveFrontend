"""
Unit Tests for Conflict Detector Module

Tests cover:
- Naming conflict detection
- Logic conflict detection
- Dependency conflict detection
- Severity assessment
- Resolution proposals
- Edge cases

Author: OpenEvolve AI System
Version: 1.0.0
"""

import unittest
from typing import List, Dict
import ast

from conflict_detector import (
    ConflictDetector,
    Conflict,
    ConflictType,
    ConflictSeverity,
    SolutionAnalysis,
    ConflictReporter,
    detect_conflicts,
    analyze_naming_conflicts,
    analyze_logic_conflicts,
    analyze_dependency_conflicts,
    assess_conflict_severity,
    propose_resolution
)


class TestConflictDetector(unittest.TestCase):
    """Test cases for ConflictDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = ConflictDetector()

    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector.strict_mode)
        self.assertIsInstance(self.detector.analyses, dict)

    def test_detect_conflicts_empty_list(self):
        """Test conflict detection with empty solution list"""
        conflicts = self.detector.detect_conflicts([], [])
        self.assertEqual(len(conflicts), 0)

    def test_detect_conflicts_single_solution(self):
        """Test conflict detection with single solution (no conflicts)"""
        solution = "def foo(): return 42"
        conflicts = self.detector.detect_conflicts([solution], [{'id': 'sol1'}])
        # Should have minimal to no conflicts for a single simple solution
        self.assertIsInstance(conflicts, list)

    def test_analyze_solution_valid_code(self):
        """Test analyzing valid Python code"""
        code = """
def calculate(x, y):
    return x + y

result = calculate(1, 2)
"""
        analysis = self.detector._analyze_solution(code, "test_sol")

        self.assertIsInstance(analysis, SolutionAnalysis)
        self.assertEqual(analysis.solution_id, "test_sol")
        self.assertIn("calculate", analysis.names_defined)
        self.assertIn("result", analysis.names_defined)
        self.assertIn("calculate", analysis.names_used)

    def test_analyze_solution_syntax_error(self):
        """Test analyzing code with syntax errors"""
        code = "def foo(\n"  # Incomplete function definition
        analysis = self.detector._analyze_solution(code, "error_sol")

        # Should return minimal analysis without crashing
        self.assertIsInstance(analysis, SolutionAnalysis)
        self.assertEqual(analysis.solution_id, "error_sol")

    def test_analyze_naming_conflicts_duplicate_names(self):
        """Test detection of duplicate names"""
        solution1 = "def process_data(): pass"
        solution2 = "def process_data(): pass"

        conflicts = self.detector.detect_conflicts(
            [solution1, solution2],
            [{'id': 'sol1'}, {'id': 'sol2'}]
        )

        # Should detect duplicate function name
        naming_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.NAMING_CONFLICT]
        self.assertGreater(len(naming_conflicts), 0)

        conflict = naming_conflicts[0]
        self.assertIn('sol1', conflict.affected_solutions)
        self.assertIn('sol2', conflict.affected_solutions)
        self.assertIn('process_data', conflict.description)

    def test_analyze_naming_conflicts_builtin_shadowing(self):
        """Test detection of builtin shadowing"""
        solution = "list = [1, 2, 3]"

        conflicts = self.detector.detect_conflicts(
            [solution],
            [{'id': 'sol1'}]
        )

        # Should detect builtin shadowing
        shadowing_conflicts = [
            c for c in conflicts
            if c.conflict_type == ConflictType.NAMING_CONFLICT
            and 'shadow' in c.description.lower()
        ]
        self.assertGreater(len(shadowing_conflicts), 0)

    def test_analyze_logic_conflicts_contradictory_patterns(self):
        """Test detection of contradictory logic patterns"""
        solution1 = """
def enable_feature():
    return True

def verify_positive():
    return True
"""
        solution2 = """
def disable_feature():
    return False

def verify_negative():
    return False
"""

        conflicts = self.detector.detect_conflicts(
            [solution1, solution2],
            [{'id': 'sol1'}, {'id': 'sol2'}]
        )

        # Should detect contradictory patterns
        logic_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.LOGIC_CONFLICT]
        self.assertGreater(len(logic_conflicts), 0)

    def test_analyze_logic_conflicts_async_sync_mix(self):
        """Test detection of mixed async/sync patterns"""
        solution1 = """
async def fetch_data():
    return await api_call()
"""
        solution2 = """
def fetch_data():
    return api_call()
"""

        conflicts = self.detector.detect_conflicts(
            [solution1, solution2],
            [{'id': 'async_sol'}, {'id': 'sync_sol'}]
        )

        # Should detect async/sync mismatch
        logic_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.LOGIC_CONFLICT]
        self.assertGreater(len(logic_conflicts), 0)

    def test_analyze_dependency_conflicts_api_incompatibility(self):
        """Test detection of incompatible API usage"""
        solution1 = """
import threading
def process():
    thread = threading.Thread(target=work)
    thread.start()
"""
        solution2 = """
import asyncio
async def process():
    await asyncio.sleep(1)
"""

        conflicts = self.detector.detect_conflicts(
            [solution1, solution2],
            [{'id': 'threading_sol'}, {'id': 'asyncio_sol'}]
        )

        # Should detect threading/asyncio incompatibility
        dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_CONFLICT]
        self.assertGreater(len(dep_conflicts), 0)

    def test_analyze_dependency_conflicts_circular_imports(self):
        """Test detection of circular dependencies"""
        solution1 = """
from solution2 import helper2
def helper1():
    return helper2()
"""
        solution2 = """
from solution1 import helper1
def helper2():
    return helper1()
"""

        conflicts = self.detector.detect_conflicts(
            [solution1, solution2],
            [{'id': 'solution1'}, {'id': 'solution2'}]
        )

        # Should detect circular dependency
        dep_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_CONFLICT]
        circular_conflicts = [c for c in dep_conflicts if 'circular' in c.description.lower()]
        self.assertGreater(len(circular_conflicts), 0)

    def test_assess_conflict_severity(self):
        """Test severity assessment"""
        # Create a test conflict
        conflict = Conflict(
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Test conflict",
            affected_solutions=['sol1', 'sol2'],
            source_locations=[],
            suggested_resolution={'strategy': 'rename'}
        )

        severity = self.detector.assess_conflict_severity(conflict)
        self.assertEqual(severity, 'HIGH')

    def test_propose_resolution_naming_conflict(self):
        """Test resolution proposal for naming conflicts"""
        conflict = Conflict(
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Duplicate name 'process'",
            affected_solutions=['sol1', 'sol2'],
            source_locations=[],
            suggested_resolution={
                'strategy': 'rename',
                'suggested_names': ['sol1_process', 'sol2_process']
            }
        )

        resolution = self.detector.propose_resolution(conflict)

        self.assertIn('strategy', resolution)
        self.assertIn('implementation_steps', resolution)
        self.assertEqual(resolution['strategy'], 'rename')
        self.assertGreater(len(resolution['implementation_steps']), 0)

    def test_propose_resolution_logic_conflict(self):
        """Test resolution proposal for logic conflicts"""
        conflict = Conflict(
            conflict_type=ConflictType.LOGIC_CONFLICT,
            severity=ConflictSeverity.CRITICAL,
            description="Contradictory assertions",
            affected_solutions=['sol1', 'sol2'],
            source_locations=[],
            suggested_resolution={
                'strategy': 'arbitrate',
                'options': ['Use sol1', 'Use sol2', 'Add conditional']
            }
        )

        resolution = self.detector.propose_resolution(conflict)

        self.assertIn('strategy', resolution)
        self.assertIn('implementation_steps', resolution)
        self.assertEqual(resolution['strategy'], 'arbitrate')

    def test_propose_resolution_dependency_conflict(self):
        """Test resolution proposal for dependency conflicts"""
        conflict = Conflict(
            conflict_type=ConflictType.DEPENDENCY_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Incompatible APIs",
            affected_solutions=['sol1', 'sol2'],
            source_locations=[],
            suggested_resolution={
                'strategy': 'separate_or_adapter',
                'options': ['Use API1 only', 'Use API2 only', 'Create adapter']
            }
        )

        resolution = self.detector.propose_resolution(conflict)

        self.assertIn('strategy', resolution)
        self.assertIn('implementation_steps', resolution)
        self.assertEqual(resolution['strategy'], 'separate_or_adapter')

    def test_conflict_to_dict(self):
        """Test Conflict serialization to dictionary"""
        conflict = Conflict(
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Test conflict",
            affected_solutions=['sol1', 'sol2'],
            source_locations=[{'solution': 'sol1', 'line': 10}],
            suggested_resolution={'strategy': 'rename'},
            metadata={'test': 'data'},
            confidence=0.95
        )

        conflict_dict = conflict.to_dict()

        self.assertEqual(conflict_dict['conflict_type'], 'naming_conflict')
        self.assertEqual(conflict_dict['severity'], 'HIGH')
        self.assertEqual(conflict_dict['description'], 'Test conflict')
        self.assertEqual(conflict_dict['confidence'], 0.95)
        self.assertIn('test', conflict_dict['metadata'])


class TestConflictReporter(unittest.TestCase):
    """Test cases for ConflictReporter class"""

    def setUp(self):
        """Set up test fixtures"""
        self.conflicts = [
            Conflict(
                conflict_type=ConflictType.NAMING_CONFLICT,
                severity=ConflictSeverity.HIGH,
                description="Duplicate function name",
                affected_solutions=['sol1', 'sol2'],
                source_locations=[],
                suggested_resolution={'strategy': 'rename'}
            ),
            Conflict(
                conflict_type=ConflictType.LOGIC_CONFLICT,
                severity=ConflictSeverity.CRITICAL,
                description="Contradictory logic",
                affected_solutions=['sol1', 'sol2'],
                source_locations=[],
                suggested_resolution={'strategy': 'arbitrate'}
            )
        ]

    def test_generate_text_report(self):
        """Test text report generation"""
        report = ConflictReporter.generate_report(self.conflicts, 'text')

        self.assertIsInstance(report, str)
        self.assertIn('CONFLICT DETECTION REPORT', report)
        self.assertIn('Total conflicts detected: 2', report)
        self.assertIn('HIGH', report)
        self.assertIn('CRITICAL', report)
        self.assertIn('Duplicate function name', report)
        self.assertIn('Contradictory logic', report)

    def test_generate_json_report(self):
        """Test JSON report generation"""
        import json

        report = ConflictReporter.generate_report(self.conflicts, 'json')

        self.assertIsInstance(report, str)
        parsed = json.loads(report)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]['conflict_type'], 'naming_conflict')
        self.assertEqual(parsed[1]['severity'], 'CRITICAL')

    def test_generate_markdown_report(self):
        """Test Markdown report generation"""
        report = ConflictReporter.generate_report(self.conflicts, 'markdown')

        self.assertIsInstance(report, str)
        self.assertIn('# Conflict Detection Report', report)
        self.assertIn('## Summary', report)
        self.assertIn('## Conflicts', report)
        self.assertIn('| Severity | Count |', report)
        self.assertIn('Duplicate function name', report)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""

    def test_detect_conflicts(self):
        """Test detect_conflicts convenience function"""
        solutions = [
            "def process(): pass",
            "def process(): pass"
        ]

        conflicts = detect_conflicts(solutions, [{'id': 's1'}, {'id': 's2'}])

        self.assertIsInstance(conflicts, list)
        self.assertGreater(len(conflicts), 0)

    def test_detect_conflicts_strict_mode(self):
        """Test detect_conflicts with strict mode enabled"""
        solutions = [
            "def foo(): pass",
            "def bar(): pass"
        ]

        conflicts = detect_conflicts(
            solutions,
            [{'id': 's1'}, {'id': 's2'}],
            strict_mode=True
        )

        # Should detect conflicts even with different names
        self.assertIsInstance(conflicts, list)

    def test_analyze_naming_conflicts(self):
        """Test analyze_naming_conflicts convenience function"""
        solutions = [
            "def data_processor(): pass",
            "def data_processor(): pass"
        ]

        conflicts = analyze_naming_conflicts(solutions)

        self.assertIsInstance(conflicts, list)
        # Should find at least the duplicate name
        self.assertGreater(len(conflicts), 0)

    def test_analyze_logic_conflicts(self):
        """Test analyze_logic_conflicts convenience function"""
        solutions = [
            "def enable_feature(): return True",
            "def disable_feature(): return False"
        ]

        conflicts = analyze_logic_conflicts(solutions)

        self.assertIsInstance(conflicts, list)

    def test_analyze_dependency_conflicts(self):
        """Test analyze_dependency_conflicts convenience function"""
        solutions = [
            "import threading\npass",
            "import asyncio\npass"
        ]

        conflicts = analyze_dependency_conflicts(solutions)

        self.assertIsInstance(conflicts, list)
        # Should detect threading/asyncio incompatibility
        self.assertGreater(len(conflicts), 0)

    def test_assess_conflict_severity(self):
        """Test assess_conflict_severity convenience function"""
        conflict = Conflict(
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            description="Test",
            affected_solutions=['s1'],
            source_locations=[],
            suggested_resolution={}
        )

        severity = assess_conflict_severity(conflict)
        self.assertEqual(severity, 'MEDIUM')

    def test_propose_resolution(self):
        """Test propose_resolution convenience function"""
        conflict = Conflict(
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Test",
            affected_solutions=['s1'],
            source_locations=[],
            suggested_resolution={'strategy': 'rename'}
        )

        resolution = propose_resolution(conflict)

        self.assertIsInstance(resolution, dict)
        self.assertIn('strategy', resolution)
        self.assertIn('implementation_steps', resolution)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = ConflictDetector()

    def test_empty_solution_code(self):
        """Test with empty solution code"""
        conflicts = self.detector.detect_conflicts([''], [{'id': 'empty'}])
        self.assertIsInstance(conflicts, list)

    def test_malformed_code(self):
        """Test with malformed Python code"""
        malformed_code = "this is not valid python (((((("
        conflicts = self.detector.detect_conflicts(
            [malformed_code, "def foo(): pass"],
            [{'id': 'bad'}, {'id': 'good'}]
        )
        self.assertIsInstance(conflicts, list)

    def test_very_long_code(self):
        """Test with very long solution code"""
        long_code = "\n".join([f"def func_{i}(): pass" for i in range(1000)])
        conflicts = self.detector.detect_conflicts([long_code], [{'id': 'long'}])
        self.assertIsInstance(conflicts, list)

    def test_unicode_in_code(self):
        """Test with unicode characters in code"""
        unicode_code = """
def café():
    return "coffee"

def 日本語():
    return "Japanese"
"""
        conflicts = self.detector.detect_conflicts([unicode_code], [{'id': 'unicode'}])
        self.assertIsInstance(conflicts, list)

    def test_nested_classes(self):
        """Test with nested class definitions"""
        nested_code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        conflicts = self.detector.detect_conflicts([nested_code], [{'id': 'nested'}])
        self.assertIsInstance(conflicts, list)

    def test_decorators(self):
        """Test with decorated functions"""
        decorated_code = """
@staticmethod
def static_method():
    pass

@property
def property_method(self):
    return self._value
"""
        conflicts = self.detector.detect_conflicts([decorated_code], [{'id': 'decorated'}])
        self.assertIsInstance(conflicts, list)

    def test_lambda_functions(self):
        """Test with lambda functions"""
        lambda_code = """
f = lambda x: x * 2
g = lambda y: y + 1
"""
        conflicts = self.detector.detect_conflicts([lambda_code], [{'id': 'lambda'}])
        self.assertIsInstance(conflicts, list)

    def test_list_comprehensions(self):
        """Test with list comprehensions"""
        comprehension_code = """
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
"""
        conflicts = self.detector.detect_conflicts([comprehension_code], [{'id': 'comp'}])
        self.assertIsInstance(conflicts, list)

    def test_context_managers(self):
        """Test with context managers (with statements)"""
        context_code = """
with open('file.txt') as f:
    data = f.read()
"""
        conflicts = self.detector.detect_conflicts([context_code], [{'id': 'context'}])
        self.assertIsInstance(conflicts, list)

    def test_exception_handling(self):
        """Test with exception handling"""
        exception_code = """
try:
    risky_operation()
except ValueError as e:
    handle_error(e)
except Exception:
    handle_generic()
finally:
    cleanup()
"""
        conflicts = self.detector.detect_conflicts([exception_code], [{'id': 'exception'}])
        self.assertIsInstance(conflicts, list)

    def test_metadata_missing(self):
        """Test with missing metadata"""
        solutions = ["def foo(): pass", "def bar(): pass"]
        conflicts = self.detector.detect_conflicts(solutions, None)
        self.assertIsInstance(conflicts, list)

    def test_metadata_mismatch(self):
        """Test with metadata list length mismatch"""
        solutions = ["def foo(): pass", "def bar(): pass"]
        metadata = [{'id': 's1'}]  # Only one metadata for two solutions

        conflicts = self.detector.detect_conflicts(solutions, metadata)
        self.assertIsInstance(conflicts, list)

    def test_special_characters_in_names(self):
        """Test with special characters (though invalid in Python)"""
        # Test with underscores and numbers
        special_code = """
def _private_function():
    pass

def __dunder_method__():
    pass

def func123():
    pass
"""
        conflicts = self.detector.detect_conflicts([special_code], [{'id': 'special'}])
        self.assertIsInstance(conflicts, list)


class TestIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = ConflictDetector()

    def test_realistic_scenario_1(self):
        """Test realistic scenario: Multiple solutions with different types of conflicts"""
        solutions = [
            """
import json
import os

def process_data(data):
    return json.dumps(data)

def save_file(filename):
    with open(filename, 'w') as f:
        f.write('content')
""",
            """
import json
import threading

def process_data(data):
    return json.loads(data)

def save_file(filename):
    thread = threading.Thread(target=write_file, args=(filename,))
    thread.start()
""",
            """
import asyncio

async def process_data(data):
    await asyncio.sleep(0.1)
    return data

async def save_file(filename):
    await write_async(filename)
"""
        ]

        metadata = [{'id': 'json_solution'}, {'id': 'threading_solution'}, {'id': 'async_solution'}]
        conflicts = self.detector.detect_conflicts(solutions, metadata)

        self.assertGreater(len(conflicts), 0)

        # Check for different conflict types
        conflict_types = set(c.conflict_type for c in conflicts)
        self.assertTrue(
            len(conflict_types) > 0,
            "Should detect at least one type of conflict"
        )

    def test_realistic_scenario_2(self):
        """Test realistic scenario: Class-based solutions"""
        solutions = [
            """
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, item):
        self.data.append(item)
        return item

class DataValidator:
    def validate(self, item):
        return True
""",
            """
class DataProcessor:
    def __init__(self):
        self.data = set()

    def process(self, item):
        self.data.add(item)
        return item * 2

class DataValidator:
    def validate(self, item):
        return len(item) > 0
"""
        ]

        metadata = [{'id': 'list_processor'}, {'id': 'set_processor'}]
        conflicts = self.detector.detect_conflicts(solutions, metadata)

        self.assertGreater(len(conflicts), 0)

        # Should detect duplicate class names
        naming_conflicts = [
            c for c in conflicts
            if c.conflict_type == ConflictType.NAMING_CONFLICT
            and 'DataProcessor' in c.description
        ]
        self.assertGreater(len(naming_conflicts), 0)

    def test_realistic_scenario_3(self):
        """Test realistic scenario: State management conflicts"""
        solutions = [
            """
counter = 0

def increment():
    global counter
    counter += 1
    return counter

def decrement():
    global counter
    counter -= 1
    return counter
""",
            """
counter = 0

def increment():
    global counter
    counter += 2
    return counter

def reset():
    global counter
    counter = 0
    return counter
"""
        ]

        metadata = [{'id': 'counter_v1'}, {'id': 'counter_v2'}]
        conflicts = self.detector.detect_conflicts(solutions, metadata)

        self.assertGreater(len(conflicts), 0)

        # Should detect duplicate global variable
        naming_conflicts = [
            c for c in conflicts
            if c.conflict_type == ConflictType.NAMING_CONFLICT
        ]
        self.assertGreater(len(naming_conflicts), 0)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConflictDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictReporter))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
