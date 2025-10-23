"""
Test Suite for OpenEvolve Integration

This module provides comprehensive tests for the OpenEvolve integration,
including parameter management, metrics collection, and team integration.
"""

import pytest
import json
from typing import Dict, Any, Tuple, List
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from parameter_manager import ParameterManager, ParameterValidator, PresetManager
from metrics_collector import MetricsCollector, MetricsStore
from openevolve_client import OpenEvolveClient
from fallback_handler import FallbackHandler


class TestParameterManager:
    """Test suite for ParameterManager"""
    
    def test_parameter_manager_initialization(self):
        """Test ParameterManager initializes correctly"""
        pm = ParameterManager()
        assert pm is not None
        assert hasattr(pm, 'validator')
        assert hasattr(pm, 'preset_manager')
    
    def test_parameter_validation_valid_params(self):
        """Test parameter validation with valid parameters"""
        pm = ParameterManager()
        
        params = {
            'max_iterations': 10,
            'population_size': 20,
            'temperature': 0.7,
            'elite_ratio': 0.1,
            'exploration_ratio': 0.4,
            'exploitation_ratio': 0.5
        }
        
        is_valid, errors = pm.validate_parameters(params)
        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0
    
    def test_parameter_validation_invalid_range(self):
        """Test parameter validation with out-of-range values"""
        pm = ParameterManager()
        
        params = {
            'max_iterations': -1,  # Invalid: must be positive
            'temperature': 3.0,     # Invalid: max is 2.0
        }
        
        is_valid, errors = pm.validate_parameters(params)
        assert not is_valid
        assert len(errors) > 0
    
    def test_parameter_validation_ratio_sum(self):
        """Test parameter validation for ratio sum"""
        pm = ParameterManager()
        
        params = {
            'elite_ratio': 0.2,
            'exploration_ratio': 0.3,
            'exploitation_ratio': 0.6  # Sum = 1.1, should fail
        }
        
        is_valid, errors = pm.validate_parameters(params)
        assert not is_valid
        assert any('ratio' in err.lower() for err in errors)
    
    def test_preset_loading(self):
        """Test loading presets"""
        pm = ParameterManager()
        
        presets = ['fast', 'balanced', 'thorough', 'research']
        
        for preset_name in presets:
            preset = pm.get_preset(preset_name)
            assert preset is not None
            assert 'max_iterations' in preset
            assert 'population_size' in preset
    
    def test_parameter_persistence(self):
        """Test saving and loading parameters"""
        pm = ParameterManager()
        
        params = {
            'max_iterations': 50,
            'population_size': 30,
            'temperature': 0.8
        }
        
        # Save parameters
        pm.save_parameters(params, 'test_config')
        
        # Load parameters
        loaded_params = pm.load_parameters('test_config')
        
        assert loaded_params is not None
        assert loaded_params['max_iterations'] == 50
        assert loaded_params['population_size'] == 30
        assert loaded_params['temperature'] == 0.8


class TestMetricsCollector:
    """Test suite for MetricsCollector"""
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initializes correctly"""
        mc = MetricsCollector()
        assert mc is not None
        assert hasattr(mc, 'store')
    
    def test_record_operation(self):
        """Test recording an operation"""
        mc = MetricsCollector()
        
        operation_id = mc.start_operation(
            evolution_mode='standard',
            parameters={'max_iterations': 10}
        )
        
        assert operation_id is not None
        assert len(operation_id) > 0
    
    def test_update_operation_metrics(self):
        """Test updating operation metrics"""
        mc = MetricsCollector()
        
        operation_id = mc.start_operation(
            evolution_mode='standard',
            parameters={'max_iterations': 10}
        )
        
        mc.update_operation(
            operation_id,
            iteration=5,
            best_fitness=0.75,
            avg_fitness=0.65
        )
        
        operation = mc.get_operation(operation_id)
        assert operation is not None
        assert operation.get('iteration') == 5
        assert operation.get('best_fitness') == 0.75
    
    def test_complete_operation(self):
        """Test completing an operation"""
        mc = MetricsCollector()
        
        operation_id = mc.start_operation(
            evolution_mode='standard',
            parameters={'max_iterations': 10}
        )
        
        mc.complete_operation(
            operation_id,
            success=True,
            final_fitness=0.85
        )
        
        operation = mc.get_operation(operation_id)
        assert operation is not None
        assert operation.get('success') is True
        assert operation.get('final_fitness') == 0.85
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation"""
        mc = MetricsCollector()
        
        # Record multiple operations
        for i in range(5):
            op_id = mc.start_operation(
                evolution_mode='standard',
                parameters={'max_iterations': 10}
            )
            mc.complete_operation(
                op_id,
                success=True,
                final_fitness=0.7 + i * 0.05
            )
        
        # Get aggregated metrics
        aggregated = mc.aggregate_metrics()
        
        assert aggregated is not None
        assert aggregated.get('total_operations') == 5
        assert aggregated.get('success_rate') == 1.0
    
    def test_metrics_export_json(self):
        """Test exporting metrics to JSON"""
        mc = MetricsCollector()
        
        op_id = mc.start_operation(
            evolution_mode='standard',
            parameters={'max_iterations': 10}
        )
        mc.complete_operation(op_id, success=True, final_fitness=0.8)
        
        json_data = mc.export_to_json()
        
        assert json_data is not None
        assert isinstance(json_data, str)
        
        # Verify it's valid JSON
        parsed = json.loads(json_data)
        assert 'operations' in parsed


class TestOpenEvolveClient:
    """Test suite for OpenEvolveClient"""
    
    @patch('openevolve_client.run_evolution')
    def test_client_initialization(self, mock_run_evolution):
        """Test OpenEvolveClient initializes correctly"""
        client = OpenEvolveClient(
            api_key='test_key',
            base_url='https://api.test.com'
        )
        
        assert client is not None
        assert client.api_key == 'test_key'
        assert client.base_url == 'https://api.test.com'
    
    @patch('openevolve_client.run_evolution')
    def test_evolve_with_valid_params(self, mock_run_evolution):
        """Test evolve method with valid parameters"""
        mock_run_evolution.return_value = Mock(
            best_fitness=0.85,
            best_solution='test solution',
            iterations_completed=10
        )
        
        client = OpenEvolveClient(api_key='test_key')
        
        result = client.evolve(
            initial_content='test content',
            evolution_mode='standard',
            max_iterations=10,
            population_size=20
        )
        
        assert result is not None
        assert 'best_fitness' in result or result.best_fitness is not None
    
    def test_validate_parameters(self):
        """Test parameter validation in client"""
        client = OpenEvolveClient(api_key='test_key')
        
        valid_params = {
            'max_iterations': 10,
            'population_size': 20,
            'temperature': 0.7
        }
        
        is_valid, errors = client.validate_parameters(valid_params)
        assert is_valid
        assert len(errors) == 0
    
    def test_get_metrics(self):
        """Test getting metrics from client"""
        client = OpenEvolveClient(api_key='test_key')
        
        metrics = client.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)


class TestFallbackHandler:
    """Test suite for FallbackHandler"""
    
    def test_fallback_handler_initialization(self):
        """Test FallbackHandler initializes correctly"""
        fh = FallbackHandler()
        assert fh is not None
    
    def test_fallback_on_error(self):
        """Test fallback behavior on error"""
        fh = FallbackHandler()
        
        # Simulate an error
        def failing_operation():
            raise Exception("Test error")
        
        result = fh.execute_with_fallback(
            failing_operation,
            fallback_value="fallback result"
        )
        
        assert result == "fallback result"
    
    def test_cache_functionality(self):
        """Test caching in fallback handler"""
        fh = FallbackHandler(enable_cache=True)
        
        call_count = 0
        
        def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - should execute
        result1 = fh.execute_with_cache(
            expensive_operation,
            cache_key='test_key',
            args=(5,)
        )
        
        # Second call - should use cache
        result2 = fh.execute_with_cache(
            expensive_operation,
            cache_key='test_key',
            args=(5,)
        )
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Should only be called once


class TestTeamIntegration:
    """Test suite for team integration with OpenEvolve"""
    
    @patch('blue_team.OpenEvolveClient')
    def test_blue_team_openevolve_integration(self, mock_client):
        """Test Blue Team integration with OpenEvolve"""
        from blue_team import BlueTeam
        from workflow_structures import Team, ModelConfig
        
        # Create mock team
        team = Team(
            name='test_blue_team',
            members=[ModelConfig(model_id='gpt-4', temperature=0.7)],
            system_prompt='Test prompt'
        )
        
        blue_team = BlueTeam(team=team, api_key='test_key')
        
        # Test that OpenEvolve methods exist
        assert hasattr(blue_team, 'generate_solution_with_openevolve')
        assert hasattr(blue_team, 'fix_with_openevolve')
    
    @patch('red_team.OpenEvolveClient')
    def test_red_team_openevolve_integration(self, mock_client):
        """Test Red Team integration with OpenEvolve"""
        from red_team import RedTeam
        from workflow_structures import Team, ModelConfig
        
        # Create mock team
        team = Team(
            name='test_red_team',
            members=[ModelConfig(model_id='gpt-4', temperature=0.7)],
            system_prompt='Test prompt'
        )
        
        red_team = RedTeam(team=team, api_key='test_key')
        
        # Test that OpenEvolve methods exist
        assert hasattr(red_team, 'critique_with_quality_diversity')
    
    @patch('evaluator_team.OpenEvolveClient')
    def test_evaluator_team_openevolve_integration(self, mock_client):
        """Test Evaluator Team integration with OpenEvolve"""
        from evaluator_team import EvaluatorTeam
        from workflow_structures import Team, ModelConfig
        
        # Create mock team
        team = Team(
            name='test_evaluator_team',
            members=[ModelConfig(model_id='gpt-4', temperature=0.7)],
            system_prompt='Test prompt'
        )
        
        evaluator_team = EvaluatorTeam(team=team, api_key='test_key')
        
        # Test that OpenEvolve methods exist
        assert hasattr(evaluator_team, 'evaluate_with_ensemble')


class TestWorkflowIntegration:
    """Test suite for workflow integration with OpenEvolve"""
    
    def test_workflow_engine_openevolve_methods(self):
        """Test that workflow engine has OpenEvolve methods"""
        from workflow_engine import WorkflowEngine
        
        engine = WorkflowEngine()
        
        # Test that OpenEvolve methods exist
        assert hasattr(engine, 'run_content_analysis_with_openevolve')
        assert hasattr(engine, 'run_decomposition_with_openevolve')
        assert hasattr(engine, 'run_assembly_with_openevolve')
    
    def test_workflow_structures_openevolve_fields(self):
        """Test that workflow structures have OpenEvolve metric fields"""
        from workflow_structures import Team, SubProblem, SolutionAttempt, WorkflowState
        
        # Test Team has openevolve_metrics
        team = Team(name='test', members=[], system_prompt='test')
        assert hasattr(team, 'openevolve_metrics')
        
        # Test SubProblem has openevolve_metrics
        sub_problem = SubProblem(
            id='test',
            description='test',
            dependencies=[],
            solver_team_name='test',
            gold_team_gauntlet_name='test'
        )
        assert hasattr(sub_problem, 'openevolve_metrics')


class TestResourceManagement:
    """Test suite for resource management"""
    
    def test_resource_manager_tracking(self):
        """Test resource tracking"""
        from resource_manager import ResourceManager, ResourceLimits
        
        limits = ResourceLimits(
            max_api_calls=100,
            max_cost=10.0
        )
        
        rm = ResourceManager(limits=limits)
        
        # Track some API calls
        rm.track_api_call(
            component='test',
            model='gpt-4',
            tokens=1000,
            execution_time=1.0
        )
        
        usage = rm.get_usage_summary()
        
        assert usage['api_calls'] == 1
        assert usage['tokens_used'] == 1000
    
    def test_resource_limit_enforcement(self):
        """Test resource limit enforcement"""
        from resource_manager import ResourceManager, ResourceLimits, ResourceLimitExceeded
        
        limits = ResourceLimits(max_api_calls=2)
        rm = ResourceManager(limits=limits)
        
        # Track calls up to limit
        rm.track_api_call('test', 'gpt-4', 100, 0.1)
        rm.track_api_call('test', 'gpt-4', 100, 0.1)
        
        # This should exceed the limit
        rm.track_api_call('test', 'gpt-4', 100, 0.1)
        
        within_limits, violations = rm.check_limits()
        assert not within_limits
        assert len(violations) > 0


class TestTemplateManagement:
    """Test suite for template management"""
    
    def test_template_manager_presets(self):
        """Test template manager presets"""
        from template_manager import TemplateManager
        
        tm = TemplateManager()
        tm.add_openevolve_preset_templates()
        
        # Test that presets exist
        fast_preset = tm.get_openevolve_template('fast')
        assert fast_preset is not None
        
        balanced_preset = tm.get_openevolve_template('balanced')
        assert balanced_preset is not None
    
    def test_template_validation(self):
        """Test template validation"""
        from template_manager import TemplateManager
        
        tm = TemplateManager()
        
        valid_config = {
            'max_iterations': 10,
            'population_size': 20,
            'temperature': 0.7,
            'elite_ratio': 0.1,
            'exploration_ratio': 0.4,
            'exploitation_ratio': 0.5
        }
        
        is_valid, errors = tm.validate_openevolve_config(valid_config)
        assert is_valid
        assert len(errors) == 0


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
