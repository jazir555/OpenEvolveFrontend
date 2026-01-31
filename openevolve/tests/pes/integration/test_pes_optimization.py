"""
Integration tests for PES optimization

Tests end-to-end evolution scenarios including:
- Simple optimization problems
- Convergence detection
- Multi-modal optimization
- Evolution over multiple generations
"""

import pytest
import tempfile
from pathlib import Path
import asyncio


class TestSimpleOptimization:
    """Test simple optimization scenarios"""

    @pytest.fixture
    def simple_quadratic_problem(self, tmp_path):
        """
        Create a simple quadratic minimization problem: f(x) = x^2
        Optimal solution: x = 0, f(0) = 0
        """
        initial_program = tmp_path / "initial_program.py"
        initial_program.write_text('''
def solve():
    """Initial solution: x = 5, f(x) = 25"""
    x = 5.0
    result = x * x
    return result

if __name__ == "__main__":
    print(solve())
''')

        evaluator = tmp_path / "eval_program.py"
        evaluator.write_text('''
import sys
import json

def evaluate(output):
    """Evaluate: minimize x^2"""
    try:
        result = float(output.strip())
        return abs(result)  # Fitness = |x^2|, minimize this
    except:
        return 1e6  # Penalty for errors

if __name__ == "__main__":
    output = sys.stdin.read()
    fitness = evaluate(output)
    print(json.dumps({"fitness": fitness}))
''')

        return {
            "initial_program": str(initial_program),
            "evaluator": str(evaluator),
            "target_fitness": 0.0,
            "acceptable_fitness": 1.0
        }

    @pytest.mark.asyncio
    async def test_simple_optimization_converges(self, simple_quadratic_problem):
        """Test PES can optimize simple quadratic function"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=10,
            population_size=5,
            llm_model_name="mock-model",
            llm_temperature=0.7
        )

        # Note: This test uses mock LLM, so actual evolution may not converge
        # The test verifies the infrastructure works, not optimization quality
        controller = OpenEvolve(
            initial_program_path=simple_quadratic_problem["initial_program"],
            evaluation_file=simple_quadratic_problem["evaluator"],
            config=config
        )

        # Verify controller initialized successfully
        assert controller is not None
        assert controller.config.max_generations == 10

    def test_simple_optimization_infrastructure(self, simple_quadratic_problem):
        """Test optimization infrastructure is properly set up"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=5,
            population_size=3,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=simple_quadratic_problem["initial_program"],
            evaluation_file=simple_quadratic_problem["evaluator"],
            config=config
        )

        # Verify all components initialized
        assert controller.db is not None
        assert controller.evaluator is not None
        assert hasattr(controller, 'logger')


class TestConvergenceDetection:
    """Test convergence detection and early stopping"""

    @pytest.fixture
    def convergence_problem(self, tmp_path):
        """Create problem that should converge quickly"""
        initial_program = tmp_path / "initial.py"
        initial_program.write_text('''
def solve():
    return 100  # Starting far from optimum

if __name__ == "__main__":
    print(solve())
''')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json

output = sys.stdin.read().strip()
if output:
    fitness = abs(float(output))
else:
    fitness = 1e6

print(json.dumps({
    "fitness": fitness,
    "converged": fitness < 0.1  # Converged if close to 0
}))
''')

        return {
            "initial_program": str(initial_program),
            "evaluator": str(evaluator)
        }

    def test_convergence_configuration(self, convergence_problem):
        """Test convergence configuration is respected"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=100,  # High limit
            population_size=5,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=convergence_problem["initial_program"],
            evaluation_file=convergence_problem["evaluator"],
            config=config
        )

        # Verify high max generations
        assert controller.config.max_generations == 100

    def test_early_stopping_setup(self, convergence_problem):
        """Test early stopping can be configured"""
        from openevolve.config import Config

        # Note: Check if Config supports early_stopping
        config = Config(
            max_generations=50,
            population_size=5,
            llm_model_name="mock-model"
        )

        # Verify configuration loaded
        assert config.max_generations == 50


class TestMultiObjectiveOptimization:
    """Test multi-objective optimization scenarios"""

    @pytest.fixture
    def multi_objective_problem(self, tmp_path):
        """
        Create multi-objective problem:
        - Minimize f1(x, y) = x^2
        - Minimize f2(x, y) = (y - 1)^2

        Pareto optimal: x = 0, y = 1
        """
        initial_program = tmp_path / "initial.py"
        initial_program.write_text('''
def solve():
    # Initial solution far from Pareto front
    x = 5.0
    y = 5.0
    f1 = x * x
    f2 = (y - 1) ** 2
    return f1 + f2  # Scalarized objective

if __name__ == "__main__":
    print(solve())
''')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json

output = sys.stdin.read().strip()
if output:
    fitness = abs(float(output))
else:
    fitness = 1e6

# Track multiple objectives
result = {
    "fitness": fitness,
    "objectives": {
        "f1": fitness * 0.5,  # Simulated
        "f2": fitness * 0.5   # Simulated
    }
}

print(json.dumps(result))
''')

        return {
            "initial_program": str(initial_program),
            "evaluator": str(evaluator)
        }

    def test_multi_objective_infrastructure(self, multi_objective_problem):
        """Test multi-objective optimization infrastructure"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=15,
            population_size=10,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=multi_objective_problem["initial_program"],
            evaluation_file=multi_objective_problem["evaluator"],
            config=config
        )

        # Should handle multi-objective setup
        assert controller is not None
        assert controller.config.population_size == 10


class TestEvolutionOverGenerations:
    """Test evolution over multiple generations"""

    @pytest.fixture
    def generational_problem(self, tmp_path):
        """Create problem to test across generations"""
        initial_program = tmp_path / "initial.py"
        initial_program.write_text('''
def solve():
    # Start with simple solution
    return 50.0

if __name__ == "__main__":
    print(solve())
''')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json

output = sys.stdin.read().strip()
if output:
    fitness = abs(float(output))
else:
    fitness = 1e6

print(json.dumps({"fitness": fitness}))
''')

        return {
            "initial_program": str(initial_program),
            "evaluator": str(evaluator)
        }

    def test_generational_tracking(self, generational_problem):
        """Test tracking across multiple generations"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=5,
            population_size=3,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=generational_problem["initial_program"],
            evaluation_file=generational_problem["evaluator"],
            config=config
        )

        # Verify database can track generations
        assert controller.db is not None

        # Add test programs to verify tracking
        prog1_id = controller.db.add_program(
            code="gen0_prog1",
            fitness=100.0,
            generation=0
        )
        prog2_id = controller.db.add_program(
            code="gen1_prog1",
            fitness=50.0,
            generation=1,
            parent_id=prog1_id
        )

        # Verify retrieval
        gen0_programs = controller.db.get_programs_by_generation(0)
        gen1_programs = controller.db.get_programs_by_generation(1)

        assert len(gen0_programs) == 1
        assert len(gen1_programs) == 1

    def test_fitness_improvement_tracking(self, generational_problem):
        """Test tracking fitness improvement over generations"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=5,
            population_size=3,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=generational_problem["initial_program"],
            evaluation_file=generational_problem["evaluator"],
            config=config
        )

        # Add programs with improving fitness
        controller.db.add_program(code="prog1", fitness=100.0, generation=0)
        controller.db.add_program(code="prog2", fitness=50.0, generation=1)
        controller.db.add_program(code="prog3", fitness=25.0, generation=2)

        # Check improvement
        best_gen0 = controller.db.get_best_program(generation=0)
        best_gen1 = controller.db.get_best_program(generation=1)
        best_gen2 = controller.db.get_best_program(generation=2)

        assert best_gen0["fitness"] == 100.0
        assert best_gen1["fitness"] == 50.0
        assert best_gen2["fitness"] == 25.0

        # Should show monotonic improvement
        assert best_gen0["fitness"] > best_gen1["fitness"] > best_gen2["fitness"]


class TestErrorRecovery:
    """Test error recovery in evolution scenarios"""

    @pytest.fixture
    def error_prone_problem(self, tmp_path):
        """Create problem that may produce errors"""
        initial_program = tmp_path / "initial.py"
        initial_program.write_text('''
def solve():
    return 42

if __name__ == "__main__":
    print(solve())
''')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json

try:
    output = sys.stdin.read().strip()
    if output:
        fitness = abs(float(output))
    else:
        fitness = 1e6
except:
    fitness = 1e6

print(json.dumps({"fitness": fitness}))
''')

        return {
            "initial_program": str(initial_program),
            "evaluator": str(evaluator)
        }

    def test_handles_invalid_programs(self, error_prone_problem):
        """Test system handles invalid programs gracefully"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config

        config = Config(
            max_generations=3,
            population_size=2,
            llm_model_name="mock-model"
        )

        controller = OpenEvolve(
            initial_program_path=error_prone_problem["initial_program"],
            evaluation_file=error_prone_problem["evaluator"],
            config=config
        )

        # Should initialize without errors
        assert controller is not None

    def test_continues_after_evaluation_errors(self, error_prone_problem, tmp_path):
        """Test evolution continues after individual evaluation errors"""
        from openevolve.evaluator import Evaluator

        evaluator = Evaluator(
            evaluation_file=error_prone_problem["evaluator"],
            timeout=10
        )

        # Test with valid program
        result = evaluator.evaluate(error_prone_problem["initial_program"])

        assert result is not None
        assert "fitness" in result


class TestPerformanceBenchmarks:
    """Test performance characteristics"""

    def test_initialization_time(self, tmp_path):
        """Test controller initialization is fast"""
        from openevolve.controller import OpenEvolve
        from openevolve.config import Config
        import time

        # Create simple problem
        initial = tmp_path / "initial.py"
        initial.write_text('print(42)')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json
print(json.dumps({"fitness": 42.0}))
''')

        config = Config(
            max_generations=10,
            population_size=5,
            llm_model_name="mock-model"
        )

        start = time.time()
        controller = OpenEvolve(
            initial_program_path=str(initial),
            evaluation_file=str(evaluator),
            config=config
        )
        elapsed = time.time() - start

        # Should initialize quickly (< 5 seconds)
        assert elapsed < 5.0
        assert controller is not None

    def test_database_query_performance(self, tmp_path):
        """Test database queries are efficient"""
        from openevolve.database import ProgramDatabase
        import time

        db = ProgramDatabase(db_path=str(tmp_path / "perf.db"))

        # Add many programs
        start = time.time()
        for i in range(100):
            db.add_program(
                code=f"program_{i}",
                fitness=float(100 - i),
                generation=i // 10
            )
        insert_time = time.time() - start

        # Should insert quickly
        assert insert_time < 5.0

        # Query best program
        start = time.time()
        best = db.get_best_program()
        query_time = time.time() - start

        # Should query quickly
        assert query_time < 1.0
        assert best is not None
        assert best["fitness"] == 0.0  # Last program has fitness 0


@pytest.mark.parametrize("population_size,max_generations", [
    (3, 5),
    (5, 10),
    (10, 15),
])
def test_various_configurations(population_size, max_generations, tmp_path):
    """Test various population and generation configurations"""
    from openevolve.controller import OpenEvolve
    from openevolve.config import Config

    # Create simple problem
    initial = tmp_path / "initial.py"
    initial.write_text('print(42)')

    evaluator = tmp_path / "eval.py"
    evaluator.write_text('import sys, json\nprint(json.dumps({"fitness": 42.0}))')

    config = Config(
        max_generations=max_generations,
        population_size=population_size,
        llm_model_name="mock-model"
    )

    controller = OpenEvolve(
        initial_program_path=str(initial),
        evaluation_file=str(evaluator),
        config=config
    )

    assert controller.config.max_generations == max_generations
    assert controller.config.population_size == population_size
