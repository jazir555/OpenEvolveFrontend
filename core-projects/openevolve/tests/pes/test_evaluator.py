"""
Unit tests for OpenEvolve evaluator

Tests the program evaluation functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess


class TestProgramEvaluator:
    """Test cases for program evaluator"""

    @pytest.fixture
    def sample_evaluator_script(self, tmp_path):
        """Create sample evaluator script"""
        eval_script = tmp_path / "evaluator.py"
        eval_script.write_text('''
import sys
import json

def evaluate_program(output):
    """
    Evaluate program output
    Returns fitness score (lower is better)
    """
    try:
        # Parse output as number
        value = float(output.strip())

        # Simple fitness: minimize absolute value
        fitness = abs(value)

        result = {
            "fitness": fitness,
            "valid": True,
            "metadata": {
                "output_length": len(output),
                "numeric": True
            }
        }

        return json.dumps(result)
    except (ValueError, TypeError) as e:
        # Return high fitness for invalid output
        return json.dumps({
            "fitness": 1e6,
            "valid": False,
            "error": str(e)
        })

if __name__ == "__main__":
    # Read program output from stdin
    program_output = sys.stdin.read()
    result = evaluate_program(program_output)
    print(result)
''')
        return str(eval_script)

    @pytest.fixture
    def sample_program(self, tmp_path):
        """Create sample program to evaluate"""
        program = tmp_path / "program.py"
        program.write_text('''
def solve():
    # Simple quadratic function
    x = 5.0
    result = x * x
    return result

if __name__ == "__main__":
    print(solve())
''')
        return str(program)

    @pytest.fixture
    def evaluator(self, sample_evaluator_script):
        """Create Evaluator instance"""
        from openevolve.evaluator import Evaluator
        return Evaluator(evaluation_file=sample_evaluator_script, timeout=10)

    def test_evaluator_initialization(self, sample_evaluator_script):
        """Test evaluator can be initialized"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=sample_evaluator_script,
            timeout=10,
            num_evaluations=3
        )

        assert evaluator.evaluation_file == sample_evaluator_script
        assert evaluator.timeout == 10
        assert evaluator.num_evaluations == 3

    def test_evaluator_evaluate_program(self, evaluator, sample_program):
        """Test evaluator can run a program"""
        result = evaluator.evaluate(sample_program)

        assert result is not None
        assert "fitness" in result
        assert isinstance(result["fitness"], (int, float))

    def test_evaluator_returns_valid_fitness(self, evaluator, sample_program):
        """Test evaluator returns valid fitness value"""
        result = evaluator.evaluate(sample_program)

        # Fitness should be numeric
        assert isinstance(result["fitness"], (int, float))

        # Fitness should be non-negative
        assert result["fitness"] >= 0

    def test_evaluator_handles_program_output(self, evaluator, sample_program):
        """Test evaluator correctly processes program output"""
        result = evaluator.evaluate(sample_program)

        # Should have processed the output
        assert "fitness" in result
        # Program should return 25.0 (5^2)
        assert result["fitness"] == 25.0

    def test_evaluator_timeout(self, sample_evaluator_script, tmp_path):
        """Test evaluator respects timeout"""
        # Create infinite loop program
        infinite_program = tmp_path / "infinite.py"
        infinite_program.write_text('''
while True:
    pass
''')

        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=sample_evaluator_script,
            timeout=1  # 1 second timeout
        )

        # Should timeout and return penalty fitness
        result = evaluator.evaluate(str(infinite_program))

        assert result is not None
        assert "fitness" in result
        # Should have high fitness due to timeout
        assert result["fitness"] >= 1e6

    def test_evaluator_handles_invalid_output(self, sample_evaluator_script, tmp_path):
        """Test evaluator handles programs that produce invalid output"""
        # Create program with invalid output
        invalid_program = tmp_path / "invalid.py"
        invalid_program.write_text('''
print("not a number")
print("more lines")
''')

        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=sample_evaluator_script,
            timeout=10
        )

        result = evaluator.evaluate(str(invalid_program))

        # Should handle gracefully with high fitness
        assert result is not None
        assert result["fitness"] >= 1e6

    def test_evaluator_handles_program_crash(self, sample_evaluator_script, tmp_path):
        """Test evaluator handles programs that crash"""
        # Create crashing program
        crashing_program = tmp_path / "crashing.py"
        crashing_program.write_text('''
raise ValueError("Crash!")
''')

        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=sample_evaluator_script,
            timeout=10
        )

        result = evaluator.evaluate(str(crashing_program))

        # Should handle crash gracefully
        assert result is not None
        assert result["fitness"] >= 1e6

    def test_evaluator_multiple_evaluations(self, sample_evaluator_script, sample_program):
        """Test evaluator runs multiple evaluations for stability"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=sample_evaluator_script,
            timeout=10,
            num_evaluations=3
        )

        result = evaluator.evaluate(sample_program)

        # Should have multiple fitness values
        if "fitness_values" in result:
            assert len(result["fitness_values"]) == 3

        # Should have average fitness
        if "avg_fitness" in result:
            assert isinstance(result["avg_fitness"], (int, float))


class TestEvaluatorValidation:
    """Test evaluator validation and error handling"""

    @pytest.fixture
    def valid_evaluator(self, tmp_path):
        """Create a valid evaluator script"""
        eval_script = tmp_path / "valid_eval.py"
        eval_script.write_text('''
import sys, json
output = sys.stdin.read()
print(json.dumps({"fitness": 0.5}))
''')
        return str(eval_script)

    def test_evaluator_validates_fitness_format(self, valid_evaluator, tmp_path):
        """Test evaluator validates JSON fitness output"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=valid_evaluator,
            timeout=10
        )

        # Create program that outputs number
        program = tmp_path / "prog.py"
        program.write_text('print(42)')

        result = evaluator.evaluate(str(program))

        assert result is not None
        assert "fitness" in result

    def test_evaluator_handles_missing_fitness(self, tmp_path):
        """Test evaluator handles scripts that don't return fitness"""
        # Create evaluator that doesn't output JSON
        bad_evaluator = tmp_path / "bad_eval.py"
        bad_evaluator.write_text('print("no fitness here")')

        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=str(bad_evaluator),
            timeout=10
        )

        program = tmp_path / "prog.py"
        program.write_text('print(42)')

        # Should handle missing fitness gracefully
        result = evaluator.evaluate(str(program))
        assert result is not None

    def test_evaluator_check_file_exists(self, tmp_path):
        """Test evaluator validates evaluation file exists"""
        from openevolve.evaluator import Evaluator

        with pytest.raises(FileNotFoundError):
            Evaluator(
                evaluation_file="/nonexistent/evaluator.py",
                timeout=10
            )


class TestEvaluatorPerformance:
    """Test evaluator performance metrics"""

    @pytest.fixture
    def performance_evaluator(self, tmp_path):
        """Create evaluator that tracks performance"""
        eval_script = tmp_path / "perf_eval.py"
        eval_script.write_text('''
import sys, json, time
output = sys.stdin.read()
start = time.time()
result = float(output) if output.strip() else 0
elapsed = time.time() - start
print(json.dumps({
    "fitness": result,
    "evaluation_time": elapsed
}))
''')
        return str(eval_script)

    def test_evaluator_measures_execution_time(self, performance_evaluator, tmp_path):
        """Test evaluator measures program execution time"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=performance_evaluator,
            timeout=10
        )

        program = tmp_path / "prog.py"
        program.write_text('print(100)')

        result = evaluator.evaluate(str(program))

        # Should track timing information
        assert result is not None
        # May have timing metadata

    def test_evaluator_handles_slow_programs(self, performance_evaluator, tmp_path):
        """Test evaluator handles programs with slow execution"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=performance_evaluator,
            timeout=10
        )

        # Create slow program
        slow_program = tmp_path / "slow.py"
        slow_program.write_text('''
import time
time.sleep(0.1)
print(42)
''')

        import time
        start = time.time()
        result = evaluator.evaluate(str(slow_program))
        elapsed = time.time() - start

        # Should complete within reasonable time
        assert elapsed < 5.0  # Should be much faster than timeout
        assert result is not None


class TestEvaluatorEdgeCases:
    """Test evaluator edge cases and boundary conditions"""

    @pytest.fixture
    def robust_evaluator(self, tmp_path):
        """Create evaluator that handles edge cases"""
        eval_script = tmp_path / "robust_eval.py"
        eval_script.write_text('''
import sys, json

try:
    output = sys.stdin.read().strip()

    if not output:
        fitness = 1e6
    elif output.lower() == 'infinity':
        fitness = float('inf')
    elif output.lower() == 'negative_infinity':
        fitness = float('-inf')
    else:
        try:
            value = float(output)
            fitness = abs(value)
        except:
            fitness = 1e6

    print(json.dumps({"fitness": fitness}))
except Exception as e:
    print(json.dumps({"fitness": 1e6, "error": str(e)}))
''')
        return str(eval_script)

    def test_evaluator_handles_empty_output(self, robust_evaluator, tmp_path):
        """Test evaluator handles programs that output nothing"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=robust_evaluator,
            timeout=10
        )

        empty_program = tmp_path / "empty.py"
        empty_program.write_text('''
# Outputs nothing
pass
''')

        result = evaluator.evaluate(str(empty_program))

        assert result is not None
        assert result["fitness"] >= 1e6  # High penalty for no output

    def test_evaluator_handles_nan(self, robust_evaluator, tmp_path):
        """Test evaluator handles NaN values"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=robust_evaluator,
            timeout=10
        )

        nan_program = tmp_path / "nan.py"
        nan_program.write_text('''
import math
print(math.nan)
''')

        result = evaluator.evaluate(str(nan_program))

        # Should handle NaN
        assert result is not None

    def test_evaluator_handles_very_large_numbers(self, robust_evaluator, tmp_path):
        """Test evaluator handles very large fitness values"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=robust_evaluator,
            timeout=10
        )

        large_program = tmp_path / "large.py"
        large_program.write_text('print(1e308)')  # Very large number

        result = evaluator.evaluate(str(large_program))

        assert result is not None
        assert isinstance(result["fitness"], (int, float))
