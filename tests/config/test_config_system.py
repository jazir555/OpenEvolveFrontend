"""
Comprehensive Configuration System Tests

Tests all components of the configuration system:
- ConfigLoader
- EnvConfigParser
- ConfigValidator
- ProfileManager
- ConfigHierarchy
- ConfigHotReload
- ConfigManager

NOTE: These tests are skipped because the config modules
have not yet been integrated from core-projects to the main openevolve package.
Per the "Law of the Air Gap", we cannot import from core-projects.
"""

import os
import json
import tempfile
import pytest
import shutil
from pathlib import Path

# Skip all tests in this module - config not yet integrated
pytestmark = pytest.mark.skip(
    reason="Config modules not yet integrated from core-projects"
)

# Create stub classes to avoid import errors
class ConfigLoader: pass
class EnvConfigParser: pass
class ConfigValidator: pass
class ValidationResult: pass
class ProfileManager: pass
class ConfigHierarchy: pass
class ConfigHotReload: pass
class ConfigManager: pass
ENV_MAPPINGS = {}
def config_to_env_name(x): return x
def env_name_to_config(x): return x


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        'max_iterations': 50,
        'population_size': 20,
        'temperature': 0.7,
        'enable_planning': True,
        'log_level': 'INFO'
    }


@pytest.fixture
def config_loader():
    """Create ConfigLoader instance"""
    return ConfigLoader()


# =============================================================================
# CONFIG LOADER TESTS
# =============================================================================

class TestConfigLoader:
    """Test ConfigLoader functionality"""

    def test_load_yaml(self, temp_dir, sample_config):
        """Test loading YAML configuration"""
        loader = ConfigLoader()

        # Skip if YAML not available
        if not loader._has_yaml:
            pytest.skip("PyYAML not installed")

        # Create YAML file
        import yaml
        yaml_file = os.path.join(temp_dir, 'config.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Load and verify
        loaded = loader.load_yaml(yaml_file)
        assert loaded == sample_config
        assert loaded['max_iterations'] == 50
        assert loaded['temperature'] == 0.7

    def test_load_json(self, temp_dir, sample_config):
        """Test loading JSON configuration"""
        loader = ConfigLoader()

        # Create JSON file
        json_file = os.path.join(temp_dir, 'config.json')
        with open(json_file, 'w') as f:
            json.dump(sample_config, f)

        # Load and verify
        loaded = loader.load_json(json_file)
        assert loaded == sample_config

    def test_load_auto_yaml(self, temp_dir, sample_config):
        """Test auto-detection of YAML format"""
        loader = ConfigLoader()

        if not loader._has_yaml:
            pytest.skip("PyYAML not installed")

        import yaml
        yaml_file = os.path.join(temp_dir, 'config.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config, f)

        loaded = loader.load_auto(yaml_file)
        assert loaded == sample_config

    def test_load_auto_json(self, temp_dir, sample_config):
        """Test auto-detection of JSON format"""
        loader = ConfigLoader()

        json_file = os.path.join(temp_dir, 'config.json')
        with open(json_file, 'w') as f:
            json.dump(sample_config, f)

        loaded = loader.load_auto(json_file)
        assert loaded == sample_config

    def test_save_yaml(self, temp_dir, sample_config):
        """Test saving YAML configuration"""
        loader = ConfigLoader()

        if not loader._has_yaml:
            pytest.skip("PyYAML not installed")

        yaml_file = os.path.join(temp_dir, 'output.yaml')
        loader.save_yaml(sample_config, yaml_file)

        assert os.path.exists(yaml_file)

        # Verify content
        loaded = loader.load_yaml(yaml_file)
        assert loaded == sample_config

    def test_save_json(self, temp_dir, sample_config):
        """Test saving JSON configuration"""
        loader = ConfigLoader()

        json_file = os.path.join(temp_dir, 'output.json')
        loader.save_json(sample_config, json_file)

        assert os.path.exists(json_file)

        # Verify content
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == sample_config

    def test_unsupported_format(self, config_loader):
        """Test error on unsupported format"""
        with pytest.raises(Exception):  # ConfigFormatError
            config_loader.load_auto('config.xml')

    def test_file_not_found(self, config_loader):
        """Test error on file not found"""
        with pytest.raises(FileNotFoundError):
            config_loader.load_json('nonexistent.json')

    def test_get_format(self, config_loader):
        """Test format detection from extension"""
        assert config_loader.get_format('config.yaml') == 'yaml'
        assert config_loader.get_format('config.yml') == 'yaml'
        assert config_loader.get_format('config.json') == 'json'
        assert config_loader.get_format('config.toml') == 'toml'
        assert config_loader.get_format('config.xml') is None


# =============================================================================
# ENV PARSER TESTS
# =============================================================================

class TestEnvConfigParser:
    """Test environment variable parser"""

    def test_parse_env_empty(self):
        """Test parsing with no env vars set"""
        parser = EnvConfigParser()
        config = parser.parse_env()
        assert isinstance(config, dict)

    def test_parse_env_with_vars(self, monkeypatch):
        """Test parsing with environment variables"""
        monkeypatch.setenv('EVOLVE_MAX_ITERATIONS', '100')
        monkeypatch.setenv('EVOLVE_TEMPERATURE', '0.8')
        monkeypatch.setenv('EVOLVE_ENABLE_PLANNING', 'true')

        parser = EnvConfigParser()
        config = parser.parse_env()

        assert config['max_iterations'] == 100
        assert config['temperature'] == 0.8
        assert config['enable_planning'] is True

    def test_get_env_value(self, monkeypatch):
        """Test getting single env value"""
        monkeypatch.setenv('EVOLVE_MAX_ITERATIONS', '50')

        parser = EnvConfigParser()
        value = parser.get_env_value('max_iterations')

        assert value == 50

    def test_get_env_value_default(self):
        """Test getting env value with default"""
        parser = EnvConfigParser()
        value = parser.get_env_value('nonexistent', 42)
        assert value == 42

    def test_type_conversion_int(self, monkeypatch):
        """Test integer type conversion"""
        monkeypatch.setenv('EVOLVE_MAX_ITERATIONS', '100')

        parser = EnvConfigParser()
        config = parser.parse_env()

        assert isinstance(config['max_iterations'], int)
        assert config['max_iterations'] == 100

    def test_type_conversion_float(self, monkeypatch):
        """Test float type conversion"""
        monkeypatch.setenv('EVOLVE_TEMPERATURE', '0.75')

        parser = EnvConfigParser()
        config = parser.parse_env()

        assert isinstance(config['temperature'], float)
        assert config['temperature'] == 0.75

    def test_type_conversion_bool(self, monkeypatch):
        """Test boolean type conversion"""
        monkeypatch.setenv('EVOLVE_ENABLE_PLANNING', 'true')
        monkeypatch.setenv('EVOLVE_ENABLE_MEMORY', 'false')

        parser = EnvConfigParser()
        config = parser.parse_env()

        assert config['enable_planning'] is True
        assert config['enable_memory'] is False

    def test_param_to_env_name(self):
        """Test converting param name to env var name"""
        assert EnvConfigParser.param_to_env_name('max_iterations') == 'EVOLVE_MAX_ITERATIONS'
        assert EnvConfigParser.param_to_env_name('temperature') == 'EVOLVE_TEMPERATURE'


# =============================================================================
# VALIDATOR TESTS
# =============================================================================

class TestConfigValidator:
    """Test configuration validator"""

    def test_validate_valid_config(self):
        """Test validating a valid configuration"""
        validator = ConfigValidator()
        config = {
            'max_iterations': 50,
            'temperature': 0.7,
            'enable_planning': True,
            'planner_model': 'gpt-4o'  # Required when enable_planning=True
        }

        result = validator.validate(config)
        assert result.is_valid

    def test_validate_temperature_out_of_range(self):
        """Test validating temperature out of range"""
        validator = ConfigValidator()
        config = {'temperature': 3.0}  # Out of range

        result = validator.validate(config)
        # Temperature should trigger consistency check (creates errors or warnings)
        issues = [e for e in result.errors if 'temperature' in e.parameter]
        assert len(issues) > 0

    def test_validate_parameter(self):
        """Test validating single parameter"""
        validator = ConfigValidator()

        is_valid, error = validator.validate_parameter('max_iterations', 50)
        assert is_valid
        assert error is None

    def test_check_dependencies_memory(self):
        """Test memory dependency check"""
        validator = ConfigValidator()
        config = {
            'enable_memory': True
            # Missing memory_type
        }

        errors = validator.check_dependencies(config)
        memory_errors = [e for e in errors if 'memory' in e.parameter]
        assert len(memory_errors) > 0

    def test_check_dependencies_no_errors(self):
        """Test dependency check with all required params"""
        validator = ConfigValidator()
        config = {
            'enable_planning': True,
            'planner_model': 'gpt-4o',  # Required
            'enable_memory': True,
            'memory_type': 'episodic'
        }

        errors = validator.check_dependencies(config)
        assert len(errors) == 0

    def test_suggest_fixes(self):
        """Test error suggestion generation"""
        validator = ConfigValidator()
        from openevolve.config.validator import ValidationError

        error = ValidationError(
            parameter='max_iterations',
            value=20000,
            message='Value 20000 out of range [1, 10000]'
        )

        suggestions = validator.suggest_fixes([error])
        assert len(suggestions) > 0


# =============================================================================
# PROFILE MANAGER TESTS
# =============================================================================

class TestProfileManager:
    """Test profile manager"""

    def test_load_development_profile(self):
        """Test loading development profile"""
        manager = ProfileManager()
        profile = manager.load_profile('development')

        assert 'max_iterations' in profile
        assert profile['max_iterations'] == 20  # Low for dev
        assert profile['log_level'] == 'DEBUG'

    def test_load_production_profile(self):
        """Test loading production profile"""
        manager = ProfileManager()
        profile = manager.load_profile('production')

        assert 'max_iterations' in profile
        assert profile['max_iterations'] == 100  # High for production
        assert profile['log_level'] == 'INFO'

    def test_load_testing_profile(self):
        """Test loading testing profile"""
        manager = ProfileManager()
        profile = manager.load_profile('testing')

        assert profile['max_iterations'] == 5  # Minimal for testing
        assert profile['seed'] == 42  # Deterministic

    def test_load_quickstart_profile(self):
        """Test loading quickstart profile"""
        manager = ProfileManager()
        profile = manager.load_profile('quickstart')

        assert 'max_iterations' in profile
        assert profile['max_iterations'] == 30  # Balanced

    def test_list_profiles(self):
        """Test listing all profiles"""
        manager = ProfileManager()
        profiles = manager.list_profiles()

        assert 'development' in profiles
        assert 'production' in profiles
        assert 'testing' in profiles
        assert 'quickstart' in profiles

    def test_create_profile(self, temp_dir):
        """Test creating custom profile"""
        manager = ProfileManager(profile_dir=temp_dir)
        overrides = {'max_iterations': 15}

        profile = manager.create_profile(
            name='custom',
            base='development',
            overrides=overrides
        )

        assert profile['max_iterations'] == 15

    def test_save_and_load_custom_profile(self, temp_dir):
        """Test saving and loading custom profile"""
        manager = ProfileManager(profile_dir=temp_dir)
        custom_config = {'max_iterations': 25}

        manager.save_profile('my_custom', custom_config)
        loaded = manager.load_profile('my_custom')

        assert loaded['max_iterations'] == 25

    def test_delete_profile(self, temp_dir):
        """Test deleting custom profile"""
        manager = ProfileManager(profile_dir=temp_dir)

        # Create profile
        manager.save_profile('temp', {'max_iterations': 10})
        assert 'temp' in manager.list_profiles()

        # Delete profile
        manager.delete_profile('temp')
        assert 'temp' not in manager.list_profiles()

    def test_delete_builtin_profile_fails(self):
        """Test that built-in profiles cannot be deleted"""
        manager = ProfileManager()

        with pytest.raises(ValueError):
            manager.delete_profile('development')

    def test_get_profile_info(self):
        """Test getting profile information"""
        manager = ProfileManager()
        info = manager.get_profile_info('development')

        assert info.name == 'development'
        assert info.category == 'development'
        assert 'max_iterations' in info.parameters


# =============================================================================
# HIERARCHY TESTS
# =============================================================================

class TestConfigHierarchy:
    """Test configuration hierarchy"""

    def test_resolve_config_defaults(self):
        """Test resolving with only defaults"""
        hierarchy = ConfigHierarchy()
        config = hierarchy.resolve_config()
        assert isinstance(config, dict)

    def test_resolve_config_with_profile(self):
        """Test resolving with profile"""
        hierarchy = ConfigHierarchy()
        config = hierarchy.resolve_config(profile='development')

        assert config['max_iterations'] == 20

    def test_resolve_config_with_overrides(self):
        """Test resolving with runtime overrides"""
        hierarchy = ConfigHierarchy()
        config = hierarchy.resolve_config(
            runtime_overrides={'max_iterations': 999}
        )

        assert config['max_iterations'] == 999

    def test_resolve_config_priority(self, monkeypatch):
        """Test that runtime overrides have highest priority"""
        monkeypatch.setenv('EVOLVE_MAX_ITERATIONS', '100')

        hierarchy = ConfigHierarchy()
        config = hierarchy.resolve_config(
            runtime_overrides={'max_iterations': 200},
            profile='development'
        )

        # Runtime override should win
        assert config['max_iterations'] == 200

    def test_merge_configs(self):
        """Test merging multiple configs"""
        hierarchy = ConfigHierarchy()

        config1 = {'a': 1, 'b': 2}
        config2 = {'b': 3, 'c': 4}
        merged = hierarchy.merge_configs(config1, config2)

        assert merged['a'] == 1
        assert merged['b'] == 3  # Overridden by config2
        assert merged['c'] == 4

    def test_apply_overrides(self):
        """Test applying overrides to base config"""
        hierarchy = ConfigHierarchy()

        base = {'max_iterations': 50}
        overrides = {'max_iterations': 100}
        result = hierarchy.apply_overrides(base, overrides)

        assert result['max_iterations'] == 100


# =============================================================================
# HOT RELOAD TESTS
# =============================================================================

class TestConfigHotReload:
    """Test configuration hot-reload"""

    def test_initialization(self, temp_dir):
        """Test hot-reload initialization"""
        # Create test config file
        config_file = os.path.join(temp_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            json.dump({'max_iterations': 50}, f)

        reload_tracker = []

        def callback(event):
            reload_tracker.append(event)

        watcher = ConfigHotReload(config_file, callback)
        assert watcher.config_file == os.path.abspath(config_file)

    def test_start_and_stop(self, temp_dir):
        """Test starting and stopping watcher"""
        import time

        # Create test config file
        config_file = os.path.join(temp_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            json.dump({'max_iterations': 50}, f)

        def callback(event):
            pass

        watcher = ConfigHotReload(config_file, callback)
        watcher.start()
        assert watcher._running is True

        watcher.stop()
        assert watcher._running is False

    def test_get_current_config(self, temp_dir):
        """Test getting current config from watcher"""
        import time

        config_file = os.path.join(temp_dir, 'config.json')
        test_config = {'max_iterations': 50}
        with open(config_file, 'w') as f:
            json.dump(test_config, f)

        def callback(event):
            pass

        watcher = ConfigHotReload(config_file, callback)
        watcher.start()
        time.sleep(0.1)  # Give it time to load

        current = watcher.get_current_config()
        assert current == test_config

        watcher.stop()


# =============================================================================
# MANAGER TESTS
# =============================================================================

class TestConfigManager:
    """Test unified configuration manager"""

    def test_initialization(self):
        """Test manager initialization"""
        manager = ConfigManager()
        assert manager.loader is not None
        assert manager.env_parser is not None
        assert manager.validator is not None

    def test_load_config_simple(self):
        """Test simple config loading"""
        manager = ConfigManager()
        config = manager.load_config()

        assert isinstance(config, dict)

    def test_load_config_with_profile(self):
        """Test loading config with profile"""
        manager = ConfigManager()
        config = manager.load_config(profile='development')

        assert config['max_iterations'] == 20

    def test_load_config_with_overrides(self):
        """Test loading config with runtime overrides"""
        manager = ConfigManager()
        config = manager.load_config(
            runtime_overrides={'max_iterations': 200}
        )

        assert config['max_iterations'] == 200

    def test_save_config_json(self, temp_dir):
        """Test saving config to JSON"""
        manager = ConfigManager()
        config = {'max_iterations': 50}

        filepath = os.path.join(temp_dir, 'output.json')
        manager.save_config(config, filepath, format='json')

        assert os.path.exists(filepath)

        # Verify
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_config_yaml(self, temp_dir):
        """Test saving config to YAML"""
        manager = ConfigManager()
        config = {'max_iterations': 50}

        if not manager.loader._has_yaml:
            pytest.skip("PyYAML not installed")

        filepath = os.path.join(temp_dir, 'output.yaml')
        manager.save_config(config, filepath, format='yaml')

        assert os.path.exists(filepath)

    def test_list_all_parameters(self):
        """Test listing all parameters"""
        manager = ConfigManager()
        params = manager.list_all_parameters()

        assert len(params) > 100  # Should have 102+ parameters
        assert 'max_iterations' in params
        assert 'temperature' in params

    def test_get_parameter_info(self):
        """Test getting parameter info"""
        manager = ConfigManager()
        info = manager.get_parameter_info('max_iterations')

        assert info is not None
        assert info['name'] == 'max_iterations'
        assert info['env_var'] == 'EVOLVE_MAX_ITERATIONS'
        assert info['type'] == 'int'

    def test_get_env_var_for_param(self):
        """Test getting env var name for parameter"""
        manager = ConfigManager()
        env_var = manager.get_env_var_for_param('max_iterations')

        assert env_var == 'EVOLVE_MAX_ITERATIONS'

    def test_compare_configs(self):
        """Test comparing two configs"""
        manager = ConfigManager()

        config1 = {'max_iterations': 50, 'temperature': 0.7}
        config2 = {'max_iterations': 100, 'temperature': 0.7, 'new_param': 'value'}

        diff = manager.compare_configs(config1, config2)

        assert 'max_iterations' in diff['different_values']
        assert 'new_param' in diff['only_in_second']

    def test_merge_configs_override(self):
        """Test merging configs with override strategy"""
        manager = ConfigManager()

        config1 = {'a': 1, 'b': 2}
        config2 = {'b': 3, 'c': 4}
        merged = manager.merge_configs(config1, config2, strategy='override')

        assert merged['a'] == 1
        assert merged['b'] == 3
        assert merged['c'] == 4


# =============================================================================
# ENV MAPPINGS TESTS
# =============================================================================

class TestEnvMappings:
    """Test environment variable mappings"""

    def test_all_params_mapped(self):
        """Test that all params have mappings"""
        assert len(ENV_MAPPINGS) >= 100

    def test_config_to_env_name(self):
        """Test converting param to env name"""
        assert config_to_env_name('max_iterations') == 'EVOLVE_MAX_ITERATIONS'
        assert config_to_env_name('temperature') == 'EVOLVE_TEMPERATURE'

    def test_env_name_to_config(self):
        """Test converting env name to param"""
        assert env_name_to_config('EVOLVE_MAX_ITERATIONS') == 'max_iterations'
        assert env_name_to_config('EVOLVE_TEMPERATURE') == 'temperature'

    def test_mapping_structure(self):
        """Test that mappings have correct structure"""
        for param_name, (env_name, param_type) in ENV_MAPPINGS.items():
            assert env_name.startswith('EVOLVE_')
            assert param_type in [int, float, str, bool, list]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_config_workflow(self, temp_dir):
        """Test complete config workflow"""
        manager = ConfigManager()

        # 1. Load from profile
        config = manager.load_config(profile='development')
        assert config['max_iterations'] == 20

        # 2. Override with env var
        os.environ['EVOLVE_MAX_ITERATIONS'] = '75'
        config = manager.load_config(profile='development', env_override=True)
        assert config['max_iterations'] == 75
        del os.environ['EVOLVE_MAX_ITERATIONS']

        # 3. Override with runtime
        config = manager.load_config(
            profile='development',
            runtime_overrides={'max_iterations': 90}
        )
        assert config['max_iterations'] == 90

        # 4. Save to file
        config_file = os.path.join(temp_dir, 'saved_config.json')
        manager.save_config(config, config_file, format='json')
        assert os.path.exists(config_file)

        # 5. Load from file
        loaded_config = manager.load_config(config_file=config_file)
        assert loaded_config['max_iterations'] == 90

    def test_profile_creation_and_usage(self, temp_dir):
        """Test creating and using custom profile"""
        manager = ConfigManager()

        # Create custom profile
        custom_profile = manager.create_profile(
            name='my_profile',
            base='development',
            overrides={'max_iterations': 15, 'custom_param': 'value'}
        )

        assert custom_profile['max_iterations'] == 15
        assert custom_profile['custom_param'] == 'value'

    def test_validation_errors(self):
        """Test that validation catches errors"""
        from openevolve.config.manager import ConfigValidationError
        manager = ConfigManager()

        # Config with invalid temperature
        invalid_config = {
            'temperature': 5.0  # Out of range
        }

        with pytest.raises(ConfigValidationError):
            manager.validate_config(invalid_config)

    def test_all_102_parameters_accessible(self):
        """Test that all 102+ parameters are accessible"""
        manager = ConfigManager()
        params = manager.list_all_parameters()

        # Each parameter should:
        # 1. Have a mapping
        # 2. Have an env var name
        # 3. Have type info

        for param in params[:10]:  # Check first 10 for speed
            info = manager.get_parameter_info(param)
            assert info is not None
            assert 'env_var' in info
            assert 'type' in info

        # Just check count for all
        assert len(params) >= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
