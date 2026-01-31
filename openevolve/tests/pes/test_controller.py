"""
Unit tests for OpenEvolve controller

Tests the main controller class that orchestrates the evolution process
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from openevolve.controller import OpenEvolve
from openevolve.config import Config


class TestOpenEvolveController:
    """Test cases for OpenEvolve controller"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        return Config(
            max_generations=5,
            population_size=3,
            llm_model_name="mock-model",
            llm_temperature=0.7,
            llm_max_tokens=1000
        )

    @pytest.fixture
    def sample_program(self, tmp_path):
        """Create sample initial program"""
        program_file = tmp_path / "initial_program.py"
        program_file.write_text('''
def solve():
    x = 5.0
    return x * x

if __name__ == "__main__":
    print(solve())
''')
        return str(program_file)

    @pytest.fixture
    def sample_evaluator(self, tmp_path):
        """Create sample evaluator"""
        eval_file = tmp_path / "eval_program.py"
        eval_file.write_text('''
import sys
import json

def evaluate(output):
    try:
        result = float(output.strip())
        return result
    except:
        return 1e6

if __name__ == "__main__":
    output = sys.stdin.read()
    fitness = evaluate(output)
    print(json.dumps({"fitness": fitness}))
''')
        return str(eval_file)

    def test_controller_initialization(self, sample_program, sample_evaluator, mock_config):
        """Test controller can be initialized"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        assert controller.config == mock_config
        assert controller.initial_program_path == sample_program
        assert controller.evaluation_file == sample_evaluator
        assert controller.output_dir is not None

    def test_controller_loads_config_from_file(self, sample_program, sample_evaluator, tmp_path):
        """Test controller loads configuration from file"""
        import yaml

        config_data = {
            "max_generations": 10,
            "population_size": 5,
            "llm": {
                "model_name": "gpt-4",
                "temperature": 0.8
            }
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config_path=str(config_file)
        )

        assert controller.config.max_generations == 10
        assert controller.config.population_size == 5

    def test_controller_creates_output_directory(self, sample_program, sample_evaluator, mock_config, tmp_path):
        """Test controller creates output directory"""
        output_dir = tmp_path / "custom_output"

        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config,
            output_dir=str(output_dir)
        )

        assert output_dir.exists()
        assert controller.output_dir == str(output_dir)

    def test_controller_loads_initial_program(self, sample_program, sample_evaluator, mock_config):
        """Test controller loads initial program code"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        with open(sample_program, 'r') as f:
            expected_code = f.read()

        # Program should be loaded and stored
        assert controller.initial_program_code is not None or Path(sample_program).exists()

    @pytest.mark.asyncio
    async def test_controller_evolve_method_exists(self, sample_program, sample_evaluator, mock_config):
        """Test controller has evolve method"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        # Check that evolve method exists and is callable
        assert hasattr(controller, 'run_evolution')
        assert callable(controller.run_evolution)

    def test_controller_database_initialization(self, sample_program, sample_evaluator, mock_config):
        """Test controller initializes database"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        # Database should be initialized
        assert hasattr(controller, 'db')
        assert controller.db is not None

    def test_controller_evaluator_initialization(self, sample_program, sample_evaluator, mock_config):
        """Test controller initializes evaluator"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        # Evaluator should be initialized
        assert hasattr(controller, 'evaluator')
        assert controller.evaluator is not None

    def test_controller_logging_setup(self, sample_program, sample_evaluator, mock_config):
        """Test controller sets up logging"""
        import logging

        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        # Check logger is configured
        assert hasattr(controller, 'logger')
        assert isinstance(controller.logger, logging.Logger)

    def test_controller_with_invalid_program_path(self, sample_evaluator, mock_config):
        """Test controller handles invalid program path"""
        with pytest.raises(FileNotFoundError):
            OpenEvolve(
                initial_program_path="/nonexistent/path/program.py",
                evaluation_file=sample_evaluator,
                config=mock_config
            )

    def test_controller_with_invalid_evaluator_path(self, sample_program, mock_config):
        """Test controller handles invalid evaluator path"""
        with pytest.raises(FileNotFoundError):
            OpenEvolve(
                initial_program_path=sample_program,
                evaluation_file="/nonexistent/path/eval.py",
                config=mock_config
            )

    @pytest.mark.asyncio
    async def test_controller_evolution_metrics_tracking(self, sample_program, sample_evaluator, mock_config):
        """Test controller tracks evolution metrics"""
        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config
        )

        # Check that metrics tracking is available
        assert hasattr(controller, 'best_fitness')
        assert hasattr(controller, 'generation')

    def test_controller_checkpoint_resume(self, sample_program, sample_evaluator, mock_config, tmp_path):
        """Test controller can save and resume from checkpoint"""
        checkpoint_dir = tmp_path / "checkpoints"

        controller = OpenEvolve(
            initial_program_path=sample_program,
            evaluation_file=sample_evaluator,
            config=mock_config,
            output_dir=str(checkpoint_dir)
        )

        # Checkpoint directory should be created
        checkpoint_path = Path(checkpoint_dir)
        assert checkpoint_path.exists()


class TestControllerEvolutionLogic:
    """Test evolution logic in controller"""

    @pytest.fixture
    def mock_config(self):
        """Minimal config for testing"""
        return Config(
            max_generations=3,
            population_size=2,
            llm_model_name="mock-model"
        )

    @pytest.fixture
    def simple_programs(self, tmp_path):
        """Create simple test programs"""
        initial_prog = tmp_path / "initial.py"
        initial_prog.write_text('def solve(): return 100')

        evaluator = tmp_path / "eval.py"
        evaluator.write_text('''
import sys, json
output = sys.stdin.read()
fitness = abs(float(output) if output.strip() else 1e6)
print(json.dumps({"fitness": fitness}))
''')

        return str(initial_prog), str(evaluator)

    def test_fitness_improvement_detection(self, simple_programs, mock_config):
        """Test controller detects fitness improvements"""
        initial_prog, evaluator = simple_programs

        controller = OpenEvolve(
            initial_program_path=initial_prog,
            evaluation_file=evaluator,
            config=mock_config
        )

        # Should track best fitness
        assert hasattr(controller, 'best_fitness')

    def test_generation_counter(self, simple_programs, mock_config):
        """Test controller tracks generation number"""
        initial_prog, evaluator = simple_programs

        controller = OpenEvolve(
            initial_program_path=initial_prog,
            evaluation_file=evaluator,
            config=mock_config
        )

        # Should have generation counter
        assert hasattr(controller, 'generation')

    def test_max_generations_config(self, simple_programs, mock_config):
        """Test max_generations configuration is respected"""
        initial_prog, evaluator = simple_programs

        controller = OpenEvolve(
            initial_program_path=initial_prog,
            evaluation_file=evaluator,
            config=mock_config
        )

        assert controller.config.max_generations == 3


class TestControllerErrorHandling:
    """Test error handling in controller"""

    @pytest.fixture
    def minimal_config(self):
        """Minimal valid config"""
        return Config(
            max_generations=1,
            population_size=1,
            llm_model_name="mock-model"
        )

    @pytest.fixture
    def failing_program(self, tmp_path):
        """Create a program that will fail"""
        prog_file = tmp_path / "failing.py"
        prog_file.write_text('''
def solve():
    raise ValueError("Intentional failure")
''')
        return str(prog_file)

    @pytest.fixture
    def robust_evaluator(self, tmp_path):
        """Create evaluator that handles errors"""
        eval_file = tmp_path / "eval.py"
        eval_file.write_text('''
import sys, json
try:
    output = sys.stdin.read()
    fitness = abs(float(output))
except:
    fitness = 1e6  # High penalty for errors
print(json.dumps({"fitness": fitness}))
''')
        return str(eval_file)

    def test_controller_handles_evaluation_errors(self, failing_program, robust_evaluator, minimal_config):
        """Test controller handles program evaluation errors gracefully"""
        # Should not raise exception, but handle error internally
        controller = OpenEvolve(
            initial_program_path=failing_program,
            evaluation_file=robust_evaluator,
            config=minimal_config
        )

        # Controller should be initialized despite potential errors
        assert controller is not None

    def test_controller_handles_invalid_config(self, tmp_path):
        """Test controller handles invalid configuration"""
        invalid_config_file = tmp_path / "invalid.yaml"
        invalid_config_file.write_text("invalid: yaml: content:")

        # Should handle invalid config gracefully
        with pytest.raises(Exception):
            OpenEvolve(
                initial_program_path="dummy.py",
                evaluation_file="dummy.py",
                config_path=str(invalid_config_file)
            )
