"""
Unit Tests for Fuzzing Components

Comprehensive tests for the fuzzing system including input generation,
crash detection, and vulnerability analysis.
"""

import pytest
import asyncio
from bubblelabs_nodes import (
    FuzzInputGenerator,
    SolutionFuzzer,
    CrashDetector,
    get_fuzzer,
)


class TestFuzzInputGenerator:
    """Tests for FuzzInputGenerator"""

    def test_generate_string_input(self):
        """Test string input generation"""
        generator = FuzzInputGenerator()

        input_str = generator.generate_input(input_type='string')

        assert isinstance(input_str, str)
        assert len(input_str) > 0

    def test_generate_integer_input(self):
        """Test integer input generation"""
        generator = FuzzInputGenerator()

        input_int = generator.generate_input(input_type='integer', constraints={'min': 0, 'max': 100})

        assert isinstance(input_int, int)
        assert 0 <= input_int <= 100

    def test_generate_json_input(self):
        """Test JSON input generation"""
        generator = FuzzInputGenerator()

        input_json = generator.generate_input(input_type='json')

        assert isinstance(input_json, dict)

    def test_generate_malformed_input(self):
        """Test malformed input generation"""
        generator = FuzzInputGenerator()

        malformed = generator.generate_input(input_type='malformed')

        assert malformed is not None

    def test_constraints_honor(self):
        """Test that constraints are honored"""
        generator = FuzzInputGenerator()

        input_val = generator.generate_input(
            input_type='integer',
            constraints={'min': 10, 'max': 20}
        )

        assert 10 <= input_val <= 20


class TestSolutionFuzzer:
    """Tests for SolutionFuzzer"""

    @pytest.mark.asyncio
    async def test_fuzz_basic(self):
        """Test basic fuzzing"""
        fuzzer = get_fuzzer()

        def solution_func(input_data):
            return {'result': input_data}

        result = await fuzzer.fuzz(
            solution=solution_func,
            input_type='auto',
            constraints={}
        )

        assert result is not None
        assert hasattr(result, 'iterations')

    @pytest.mark.asyncio
    async def test_fuzz_with_crash_detection(self):
        """Test fuzzing with crash detection"""
        fuzzer = get_fuzzer()

        def crash_prone_func(input_data):
            if isinstance(input_data, str) and len(input_data) > 1000:
                raise ValueError("Buffer overflow!")
            return {'ok': True}

        result = await fuzzer.fuzz(
            solution=crash_prone_func,
            input_type='string',
            constraints={}
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_fuzz_timeout(self):
        """Test fuzzing timeout"""
        fuzzer = get_fuzzer(max_iterations=100, timeout_seconds=1)

        async def slow_func(input_data):
            await asyncio.sleep(2)  # Longer than timeout
            return {'ok': True}

        result = await fuzzer.fuzz(
            solution=slow_func,
            input_type='integer',
            constraints={}
        )

        # Should handle timeout gracefully
        assert result is not None


class TestCrashDetector:
    """Tests for CrashDetector"""

    def test_detect_exception(self):
        """Test exception detection"""
        detector = CrashDetector()

        def crashing_func():
            raise ValueError("Crash!")

        try:
            crashing_func()
        except Exception as e:
            crash_info = detector.detect_crash(e)

            assert crash_info is not None
            assert crash_info['exception_type'] == 'ValueError'

    def test_detect_timeout(self):
        """Test timeout detection"""
        detector = CrashDetector()

        # Simulate timeout
        crash_info = detector.detect_timeout(30.0)

        assert crash_info is not None
        assert 'timeout' in crash_info['crash_type'].lower()

    def test_detect_memory_error(self):
        """Test memory error detection"""
        detector = CrashDetector()

        try:
            # Simulate memory error
            raise MemoryError("Out of memory!")
        except Exception as e:
            crash_info = detector.detect_crash(e)

            assert crash_info is not None
            assert crash_info['exception_type'] == 'MemoryError'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
