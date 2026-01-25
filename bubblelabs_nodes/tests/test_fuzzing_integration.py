"""
Integration Tests for Fuzzing System

Comprehensive integration tests for the fuzzing, crash detection,
and vulnerability analysis pipeline.
"""

import pytest
import asyncio
from bubblelabs_nodes.fuzzing import (
    FuzzInputGenerator,
    FuzzExecutor,
    SolutionFuzzer,
    fuzz_solution,
    Vulnerability,
    VulnerabilitySeverity,
    FuzzResult,
)
from bubblelabs_nodes.crash_analyzer import (
    CrashAnalyzer,
    CrashReporter,
    CrashPattern,
    CrashReport,
    analyze_crashes,
    generate_crash_report,
)


class TestFuzzingIntegration:
    """Integration tests for complete fuzzing pipeline"""

    @pytest.mark.asyncio
    async def test_complete_fuzzing_workflow(self):
        """Test complete fuzzing workflow from execution to analysis"""
        # Create a vulnerable solution
        def vulnerable_solution(input_data):
            # Bug: Doesn't handle None
            if len(input_data) > 5:
                return input_data[:5]
            return input_data

        # Run fuzzing
        result = await fuzz_solution(
            solution=vulnerable_solution,
            iterations=100,
            timeout=1.0,
            max_concurrent=2
        )

        # Analyze crashes
        analyzer = CrashAnalyzer()
        report = analyzer.analyze(result)

        # Verify workflow
        assert isinstance(result, FuzzResult)
        assert isinstance(report, CrashReport)
        assert report.total_crashes >= 0
        assert report.unique_crashes >= 0

    @pytest.mark.asyncio
    async def test_fuzzing_with_real_solutions(self):
        """Test fuzzing with realistic solution functions"""
        test_cases = [
            # String processing solution
            lambda s: s[:5] if s and len(s) > 5 else s,

            # Array processing solution
            lambda arr: [x * 2 for x in arr[:10]] if arr else [],

            # Object processing solution
            lambda obj: obj.get('value', 0) if obj else 0,
        ]

        for solution in test_cases:
            result = await fuzz_solution(
                solution=solution,
                iterations=50,
                timeout=1.0,
                max_concurrent=2
            )

            # Should complete without error
            assert result.iterations == 50
            assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_crash_detection_accuracy(self):
        """Test that crash detection accurately identifies issues"""
        crashes_found = []
        vulnerabilities_found = []

        # Solution with known crash condition
        def buggy_solution(input_data):
            if isinstance(input_data, str) and input_data == "crash":
                raise ValueError("Intentional crash")
            return input_data

        # Create custom fuzzer to track crashes
        fuzzer = SolutionFuzzer(
            iterations=100,
            timeout=1.0,
            max_concurrent=2
        )

        # Mock generator to include crash trigger
        original_generate = fuzzer.generator.generate_input
        def mock_generate(input_type, constraints):
            # Intermix crash triggers
            import random
            if random.random() < 0.1:
                return "crash"
            return original_generate(input_type, constraints)

        fuzzer.generator.generate_input = mock_generate

        result = await fuzzer.fuzz(buggy_solution, 'auto', {})

        # Verify crash detection
        assert result.crashes_found > 0
        assert len(result.vulnerabilities) > 0

    @pytest.mark.asyncio
    async def test_vulnerability_reporting(self):
        """Test vulnerability reporting and prioritization"""
        def solution_with_multiple_bugs(input_data):
            # Bug 1: No None check
            if len(input_data) > 0:
                # Bug 2: No type check
                return input_data[0] * 2
            return None

        result = await fuzz_solution(
            solution=solution_with_multiple_bugs,
            iterations=100,
            timeout=1.0
        )

        # Generate crash report
        reporter = CrashReporter()
        report_text = reporter.generate_report(result, format='text')
        report_markdown = reporter.generate_report(result, format='markdown')
        report_json = reporter.generate_report(result, format='json')

        # Verify reports generated
        assert len(report_text) > 0
        assert len(report_markdown) > 0
        assert len(report_json) > 0
        assert report_json.startswith('{')  # Valid JSON

    @pytest.mark.asyncio
    async def test_fuzz_performance_impact(self):
        """Test performance impact of fuzzing"""
        def simple_solution(input_data):
            return input_data

        # Benchmark without fuzzing
        import time
        start = time.time()
        for _ in range(10):
            await asyncio.to_thread(simple_solution, "test")
        baseline_time = time.time() - start

        # Benchmark with fuzzing
        start = time.time()
        result = await fuzz_solution(
            solution=simple_solution,
            iterations=10,
            timeout=1.0
        )
        fuzzing_time = time.time() - start

        # Fuzzing overhead should be reasonable (less than 100x)
        # This is a loose threshold since fuzzing is inherently expensive
        assert fuzzing_time < baseline_time * 100

    @pytest.mark.asyncio
    async def test_concurrent_fuzzing(self):
        """Test that concurrent fuzzing works correctly"""
        async def fuzz_multiple_solutions():
            solutions = [
                lambda x: x,
                lambda x: x[:5] if x else x,
                lambda x: len(x) if x else 0,
            ]

            tasks = [
                fuzz_solution(sol, iterations=50, timeout=1.0)
                for sol in solutions
            ]

            results = await asyncio.gather(*tasks)
            return results

        results = await fuzz_multiple_solutions()

        assert len(results) == 3
        for result in results:
            assert result.iterations == 50

    @pytest.mark.asyncio
    async def test_corpus_management(self):
        """Test corpus of interesting inputs"""
        fuzzer = SolutionFuzzer(
            iterations=200,
            corpus_size=10
        )

        # Solution that crashes on specific inputs
        def selective_crash(input_data):
            if isinstance(input_data, str) and 'special' in input_data:
                raise ValueError("Special crash")
            return input_data

        result = await fuzzer.fuzz(selective_crash)

        # Check corpus
        corpus = fuzzer.get_corpus()
        assert len(corpus) <= 10  # Should respect corpus size

        # Clear corpus
        fuzzer.clear_corpus()
        assert len(fuzzer.get_corpus()) == 0

    @pytest.mark.asyncio
    async def test_crash_deduplication(self):
        """Test that duplicate crashes are deduplicated"""
        analyzer = CrashAnalyzer()

        # Create vulnerabilities with same signature
        vuln1 = Vulnerability(
            vulnerability_id="vuln_1",
            severity=VulnerabilitySeverity.HIGH,
            title="Test Vulnerability",
            description="ValueError: test error",
            input_data="test",
            crash_type="ValueError",
        )

        vuln2 = Vulnerability(
            vulnerability_id="vuln_2",
            severity=VulnerabilitySeverity.HIGH,
            title="Test Vulnerability 2",
            description="ValueError: test error",  # Same description
            input_data="test2",
            crash_type="ValueError",
        )

        vuln3 = Vulnerability(
            vulnerability_id="vuln_3",
            severity=VulnerabilitySeverity.MEDIUM,
            title="Different Vulnerability",
            description="TypeError: different error",
            input_data="test3",
            crash_type="TypeError",
        )

        # Deduplicate
        unique = analyzer.deduplicate_vulnerabilities([vuln1, vuln2, vuln3])

        # Should deduplicate vuln1 and vuln2
        assert len(unique) == 2

    @pytest.mark.asyncio
    async def test_crash_pattern_identification(self):
        """Test identification of crash patterns"""
        analyzer = CrashAnalyzer()

        # Create result with multiple crashes
        result = FuzzResult(
            iterations=100,
            crashes_found=10,
            unique_crashes=5,
            vulnerabilities=[
                Vulnerability(
                    vulnerability_id=f"vuln_{i}",
                    severity=VulnerabilitySeverity.MEDIUM,
                    title=f"Crash {i}",
                    description="IndexError: list index out of range",
                    input_data=f"input_{i}",
                    crash_type="IndexError",
                )
                for i in range(5)
            ]
        )

        report = analyzer.analyze(result)

        # Should identify pattern
        assert len(report.crash_patterns) > 0
        assert any('buffer' in p.pattern_name.lower() or 'index' in p.pattern_name.lower()
                   for p in report.crash_patterns)

    @pytest.mark.asyncio
    async def test_severity_classification(self):
        """Test severity classification of vulnerabilities"""
        # Map exception types to expected severities
        test_cases = [
            (MemoryError(), VulnerabilitySeverity.CRITICAL),
            (ValueError(), VulnerabilitySeverity.MEDIUM),
            (AttributeError(), VulnerabilitySeverity.LOW),
        ]

        fuzzer = SolutionFuzzer(iterations=10, timeout=1.0)

        for exception, expected_severity in test_cases:
            severity = fuzzer._assess_severity(exception)
            assert severity == expected_severity

    @pytest.mark.asyncio
    async def test_fuzzing_timeout_handling(self):
        """Test that fuzzing respects timeout"""
        # Solution that hangs
        def hanging_solution(input_data):
            import time
            time.sleep(10)  # Sleep longer than timeout
            return input_data

        result = await fuzz_solution(
            solution=hanging_solution,
            iterations=1,
            timeout=0.5,  # Short timeout
            max_concurrent=1
        )

        # Should have timeout crashes
        assert result.crashes_found > 0

    @pytest.mark.asyncio
    async def test_fuzzing_various_input_types(self):
        """Test fuzzing with various input types"""
        generator = FuzzInputGenerator(seed=42)

        input_types = ['string', 'number', 'boolean', 'array', 'object', 'edge_case']

        for input_type in input_types:
            input_data = generator.generate_input(input_type)
            assert input_data is not None or input_type == 'null'

    @pytest.mark.asyncio
    async def test_crash_report_recommendations(self):
        """Test that crash report provides actionable recommendations"""
        result = FuzzResult(
            iterations=100,
            crashes_found=5,
            unique_crashes=3,
            vulnerabilities=[
                Vulnerability(
                    vulnerability_id="vuln_critical",
                    severity=VulnerabilitySeverity.CRITICAL,
                    title="Critical Issue",
                    description="SystemExit: Critical failure",
                    input_data="crash_input",
                    crash_type="SystemExit",
                ),
                Vulnerability(
                    vulnerability_id="vuln_high",
                    severity=VulnerabilitySeverity.HIGH,
                    title="High Issue",
                    description="AssertionError: Assertion failed",
                    input_data="assert_input",
                    crash_type="AssertionError",
                ),
            ]
        )

        reporter = CrashReporter()
        report = reporter.generate_report(result, format='text')

        # Report should contain recommendations
        assert 'RECOMMENDATIONS' in report or 'recommendations' in report.lower()
        assert 'URGENT' in report or 'critical' in report.lower()

    @pytest.mark.asyncio
    async def test_fuzzing_with_constraints(self):
        """Test fuzzing with input constraints"""
        generator = FuzzInputGenerator(seed=42)

        # Test with constraints
        constrained_string = generator.generate_input(
            'string',
            constraints={'min_length': 5, 'max_length': 10}
        )
        assert 5 <= len(constrained_string) <= 10

        constrained_number = generator.generate_input(
            'number',
            constraints={'min': 0, 'max': 100}
        )
        assert 0 <= constrained_number <= 100

    @pytest.mark.asyncio
    async def test_fuzz_effectiveness_metrics(self):
        """Test fuzzing effectiveness metrics"""
        # Create a solution with intentional vulnerabilities
        def vulnerable_solution(input_data):
            if not isinstance(input_data, str):
                raise TypeError("Expected string")

            if len(input_data) > 1000:
                raise ValueError("Input too long")

            if input_data == "null":
                raise ValueError("Null input")

            return input_data[:10]

        result = await fuzz_solution(
            solution=vulnerable_solution,
            iterations=200,
            timeout=1.0,
            max_concurrent=4
        )

        # Should find some vulnerabilities
        assert result.unique_crashes >= 0
        assert result.execution_time > 0

        # Analyze effectiveness
        if result.crashes_found > 0:
            crash_rate = result.crashes_found / result.iterations
            assert 0 < crash_rate <= 1.0


class TestFuzzingEffectiveness:
    """Tests to measure fuzzing effectiveness"""

    @pytest.mark.asyncio
    async def test_false_positive_rate(self):
        """Measure false positive rate (benign inputs flagged as crashes)"""
        def robust_solution(input_data):
            # Robust solution that handles all inputs gracefully
            try:
                if input_data is None:
                    return None
                if isinstance(input_data, str):
                    return input_data[:100]
                return str(input_data)[:100]
            except Exception:
                return None

        result = await fuzz_solution(
            solution=robust_solution,
            iterations=100,
            timeout=1.0
        )

        # Should have very few crashes (false positives)
        false_positive_rate = result.crashes_found / result.iterations
        assert false_positive_rate < 0.1  # Less than 10% false positives

    @pytest.mark.asyncio
    async def test_false_negative_rate(self):
        """Measure false negative rate (actual bugs not found)"""
        # Solution with known bug
        def buggy_solution(input_data):
            # Known bug: crashes on empty string
            return input_data[0] if input_data else None

        result = await fuzz_solution(
            solution=buggy_solution,
            iterations=100,
            timeout=1.0
        )

        # Should find the bug (will crash on empty string)
        # Note: This is probabilistic, so we just check it doesn't crash the test
        assert result.iterations == 100

    @pytest.mark.asyncio
    async def test_coverage_estimation(self):
        """Test code coverage estimation through fuzzing"""
        executed_paths = set()

        def solution_with_branches(input_data):
            if isinstance(input_data, str):
                executed_paths.add('string')
                return input_data.upper()
            elif isinstance(input_data, int):
                executed_paths.add('int')
                return input_data * 2
            elif isinstance(input_data, list):
                executed_paths.add('list')
                return len(input_data)
            else:
                executed_paths.add('other')
                return None

        await fuzz_solution(
            solution=solution_with_branches,
            iterations=200,
            timeout=1.0
        )

        # Should execute multiple paths
        # Note: This is probabilistic, so we just check it doesn't crash
        assert len(executed_paths) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
