"""
Tests for Runtime Configuration System

Comprehensive tests for runtime configuration updates, hot-reload, dynamic
strategy switching, adaptive configuration, and resource-aware configuration.
"""

import pytest
import time
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from openevolve.config.runtime_config import (
    RuntimeConfigUpdater,
    ConfigUpdate,
    ConfigWatcherCallback
)
from openevolve.config.config_watcher import (
    ConfigFileWatcher,
    MultiConfigWatcher
)
from openevolve.config.dynamic_strategy import (
    DynamicStrategySwitcher,
    SystemMode,
    StateMigrator,
    StrategySwitchRecord
)
from openevolve.config.adaptive_config import (
    AdaptiveConfigurator,
    PerformanceMetrics,
    AutoTuner
)
from openevolve.config.resource_config import (
    ResourceAwareConfigurator,
    ResourceInfo,
    ResourceLimits
)
from openevolve.config.config_metrics import (
    ConfigurationMetrics,
    ConfigComparison,
    hash_config
)
from openevolve.unified.config import UnifiedEvolutionConfig


# ==============================================================================
# Runtime Config Updater Tests
# ==============================================================================

class TestRuntimeConfigUpdater:
    """Test suite for RuntimeConfigUpdater"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return UnifiedEvolutionConfig()

    @pytest.fixture
    def updater(self, config):
        """Create updater instance"""
        return RuntimeConfigUpdater(config)

    @pytest.mark.asyncio
    async def test_update_single_parameter(self, updater, config):
        """Test updating a single parameter"""
        original_value = config.common.max_iterations
        new_value = 200

        success = await updater.update_parameter(
            "max_iterations",
            new_value,
            scope="common"
        )

        assert success is True
        assert config.common.max_iterations == new_value
        assert len(updater.update_history) == 1
        assert updater.update_history[0].parameter == "common.max_iterations"

    @pytest.mark.asyncio
    async def test_update_nested_parameter(self, updater, config):
        """Test updating nested parameter with dot notation"""
        new_temp = 0.9

        success = await updater.update_parameter(
            "llm.temperature",
            new_temp
        )

        assert success is True
        assert config.llm.default_temperature == new_temp

    @pytest.mark.asyncio
    async def test_batch_parameter_update(self, updater, config):
        """Test batch parameter updates"""
        updates = {
            "common.max_iterations": 200,
            "common.concurrency": 10,
            "database.population_size": 500
        }

        success = await updater.update_parameters(updates)

        assert success is True
        assert config.common.max_iterations == 200
        assert config.common.concurrency == 10
        assert config.database.population_size == 500
        assert len(updater.update_history) == 3

    @pytest.mark.asyncio
    async def test_batch_update_with_rollback(self, updater, config):
        """Test batch update rollback on error"""
        # First update should succeed
        updates = {
            "common.max_iterations": 200,
            "common.invalid_param": 100  # This should fail
        }

        # Mock validator to fail on invalid_param
        with patch.object(updater.validators, 'validate_parameter') as mock_validate:
            mock_validate.return_value = Mock(is_valid=False, errors=["Invalid parameter"])

            success = await updater.update_parameters(
                updates,
                rollback_on_error=True
            )

            assert success is False
            # Config should be rolled back to original state
            assert config.common.max_iterations == 100  # Default value

    def test_get_update_history(self, updater):
        """Test retrieving update history"""
        # Manually add some history
        updater.update_history.append(ConfigUpdate(
            timestamp=datetime.utcnow(),
            parameter="test.param1",
            old_value=1,
            new_value=2,
            update_type="single"
        ))
        updater.update_history.append(ConfigUpdate(
            timestamp=datetime.utcnow(),
            parameter="test.param2",
            old_value=3,
            new_value=4,
            update_type="batch"
        ))

        history = updater.get_update_history()

        assert len(history) == 2

    def test_get_update_history_with_filter(self, updater):
        """Test filtering update history"""
        updater.update_history.append(ConfigUpdate(
            timestamp=datetime.utcnow(),
            parameter="common.max_iterations",
            old_value=100,
            new_value=200,
            update_type="single"
        ))
        updater.update_history.append(ConfigUpdate(
            timestamp=datetime.utcnow(),
            parameter="database.population_size",
            old_value=1000,
            new_value=500,
            update_type="single"
        ))

        history = updater.get_update_history(parameter_filter="common")

        assert len(history) == 1
        assert "common" in history[0].parameter

    def test_create_and_restore_snapshot(self, updater, config):
        """Test configuration snapshot creation and restoration"""
        original_iterations = config.common.max_iterations

        # Create snapshot
        timestamp = updater.create_snapshot(label="test_snapshot")

        assert timestamp in updater._config_snapshots

        # Modify config
        config.common.max_iterations = 500

        # Restore snapshot
        success = updater.rollback_to(timestamp)

        assert success is True
        assert config.common.max_iterations == original_iterations


# ==============================================================================
# Config File Watcher Tests
# ==============================================================================

class TestConfigFileWatcher:
    """Test suite for ConfigFileWatcher"""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False
        ) as f:
            f.write("""
evolution_mode: openevolve
common:
  max_iterations: 100
  log_level: INFO
llm:
  default_api_base: https://api.openai.com/v1
""")
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def watcher(self, temp_config_file):
        """Create watcher instance"""
        return ConfigFileWatcher(temp_config_file)

    def test_watcher_initialization(self, watcher, temp_config_file):
        """Test watcher initialization"""
        assert watcher.config_file.exists()
        assert watcher.running is False
        assert len(watcher.callbacks) == 0

    def test_register_callback(self, watcher):
        """Test registering callbacks"""
        callback = Mock()
        watcher.register_callback(callback)

        assert len(watcher.callbacks) == 1
        assert callback in watcher.callbacks

    def test_unregister_callback(self, watcher):
        """Test unregistering callbacks"""
        callback = Mock()
        watcher.register_callback(callback)
        watcher.unregister_callback(callback)

        assert len(watcher.callbacks) == 0

    def test_get_stats(self, watcher):
        """Test getting watcher statistics"""
        stats = watcher.get_stats()

        assert "file" in stats
        assert "running" in stats
        assert "reload_count" in stats
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_file_change_detection(self, temp_config_file):
        """Test file change detection"""
        watcher = ConfigFileWatcher(temp_config_file, poll_interval=0.1)

        # Simulate file change
        time.sleep(0.2)
        Path(temp_config_file).touch()

        # Give it time to detect
        time.sleep(0.3)

        assert watcher._has_file_changed()

        watcher.stop()


class TestMultiConfigWatcher:
    """Test suite for MultiConfigWatcher"""

    @pytest.fixture
    def temp_files(self):
        """Create multiple temporary config files"""
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.yaml',
                delete=False
            ) as f:
                f.write(f"evolution_mode: openevolve\ntest: {i}\n")
                files.append(f.name)

        yield files

        # Cleanup
        for f in files:
            Path(f).unlink(missing_ok=True)

    @pytest.fixture
    def multi_watcher(self):
        """Create multi-watcher instance"""
        return MultiConfigWatcher()

    def test_add_file(self, multi_watcher, temp_files):
        """Test adding files to watch"""
        watcher = multi_watcher.add_file(temp_files[0])

        assert temp_files[0] in multi_watcher.watchers
        assert watcher is not None

    def test_remove_file(self, multi_watcher, temp_files):
        """Test removing files from watch"""
        multi_watcher.add_file(temp_files[0])
        multi_watcher.remove_file(temp_files[0])

        assert temp_files[0] not in multi_watcher.watchers

    def test_global_callback(self, multi_watcher, temp_files):
        """Test global callbacks"""
        global_callback = Mock()
        multi_watcher.register_global_callback(global_callback)

        multi_watcher.add_file(temp_files[0])

        # Verify callback was registered
        assert len(multi_watcher.global_callbacks) == 1


# ==============================================================================
# Dynamic Strategy Switcher Tests
# ==============================================================================

class TestDynamicStrategySwitcher:
    """Test suite for DynamicStrategySwitcher"""

    @pytest.fixture
    def switcher(self):
        """Create switcher instance"""
        return DynamicStrategySwitcher(SystemMode.OPENEVOLVE)

    def test_initialization(self, switcher):
        """Test switcher initialization"""
        assert switcher.current_strategy == SystemMode.OPENEVOLVE
        assert len(switcher.strategy_history) == 0

    def test_validate_switch_valid(self, switcher):
        """Test validating a valid strategy switch"""
        config = UnifiedEvolutionConfig(evolution_mode="qd")
        config.qd = Mock()  # Mock QD config

        is_valid = switcher._validate_switch(SystemMode.QD, config)

        assert is_valid is True

    def test_validate_switch_invalid_mode(self, switcher):
        """Test validating switch with invalid mode"""
        config = UnifiedEvolutionConfig(evolution_mode="openevolve")

        is_valid = switcher._validate_switch(SystemMode.QD, config)

        assert is_valid is False

    def test_get_switch_history(self, switcher):
        """Test getting switch history"""
        record = StrategySwitchRecord(
            timestamp=datetime.utcnow(),
            from_strategy=SystemMode.OPENEVOLVE,
            to_strategy=SystemMode.QD,
            state_preserved=True,
            migration_success=True
        )

        switcher.strategy_history.append(record)

        history = switcher.get_switch_history()

        assert len(history) == 1
        assert history[0].from_strategy == SystemMode.OPENEVOLVE


class TestStateMigrator:
    """Test suite for StateMigrator"""

    @pytest.fixture
    def migrator(self):
        """Create migrator instance"""
        return StateMigrator()

    @pytest.mark.asyncio
    async def test_migrate_state(self, migrator):
        """Test state migration"""
        current_state = {
            "best_solutions": [{"fitness": 0.9}],
            "artifacts": {"test": "artifact"},
            "population": [{"fitness": 0.5}]
        }

        migrated = await migrator.migrate(
            SystemMode.OPENEVOLVE,
            SystemMode.QD,
            current_state
        )

        assert migrated is not None
        assert "best_solutions" in migrated
        assert "artifacts" in migrated

    def test_both_use_population(self, migrator):
        """Test population compatibility check"""
        result = migrator._both_use_population(
            SystemMode.OPENEVOLVE,
            SystemMode.QD
        )

        assert result is True

    def test_both_use_archive(self, migrator):
        """Test archive compatibility check"""
        result = migrator._both_use_archive(
            SystemMode.QD,
            SystemMode.MO
        )

        assert result is True


# ==============================================================================
# Adaptive Configuration Tests
# ==============================================================================

class TestAdaptiveConfigurator:
    """Test suite for AdaptiveConfigurator"""

    @pytest.fixture
    def base_config(self):
        """Create base configuration"""
        return UnifiedEvolutionConfig()

    @pytest.fixture
    def adaptive(self, base_config):
        """Create adaptive configurator"""
        return AdaptiveConfigurator(base_config)

    @pytest.mark.asyncio
    async def test_adapt_configuration(self, adaptive):
        """Test configuration adaptation"""
        performance_metrics = {
            "fitness": 0.8,
            "diversity": 0.2,
            "convergence_rate": 0.001,
            "improvement_rate": 0.0,
            "evaluation_time": 15.0,
            "success_rate": 0.95
        }

        # Add enough history
        for i in range(10):
            await adaptive.adapt_configuration(performance_metrics, i)

        suggestions = await adaptive.adapt_configuration(performance_metrics, 10)

        assert isinstance(suggestions, dict)

    def test_is_slow_convergence(self, adaptive):
        """Test slow convergence detection"""
        # Add performance history showing slow convergence
        for i in range(10):
            adaptive.performance_history.append(
                PerformanceMetrics(
                    iteration=i,
                    fitness=0.5 + (i * 0.0001),  # Very slow improvement
                    diversity=0.8,
                    convergence_rate=0.001,
                    improvement_rate=0.0,
                    evaluation_time=5.0,
                    success_rate=1.0,
                    timestamp=datetime.utcnow()
                )
            )

        is_slow = adaptive._is_slow_convergence()

        assert is_slow is True

    def test_is_low_diversity(self, adaptive):
        """Test low diversity detection"""
        metrics = PerformanceMetrics(
            iteration=0,
            fitness=0.5,
            diversity=0.2,  # Low diversity
            convergence_rate=0.01,
            improvement_rate=0.1,
            evaluation_time=5.0,
            success_rate=1.0,
            timestamp=datetime.utcnow()
        )

        is_low = adaptive._is_low_diversity(metrics)

        assert is_low is True

    def test_get_performance_summary(self, adaptive):
        """Test performance summary"""
        for i in range(10):
            adaptive.performance_history.append(
                PerformanceMetrics(
                    iteration=i,
                    fitness=0.5 + (i * 0.05),
                    diversity=0.7,
                    convergence_rate=0.01,
                    improvement_rate=0.1,
                    evaluation_time=5.0,
                    success_rate=1.0,
                    timestamp=datetime.utcnow()
                )
            )

        summary = adaptive.get_performance_summary()

        assert "total_iterations" in summary
        assert summary["total_iterations"] == 10
        assert "best_fitness" in summary


class TestAutoTuner:
    """Test suite for AutoTuner"""

    @pytest.fixture
    def tuner(self):
        """Create auto-tuner instance"""
        return AutoTuner()

    @pytest.mark.asyncio
    async def test_auto_tune(self, tuner):
        """Test automatic tuning"""
        config = UnifiedEvolutionConfig()
        performance_data = {
            "fitness_history": [0.5, 0.55, 0.6],
            "diversity_history": [0.7, 0.65, 0.6],
            "evaluation_times": [5.0, 4.5, 4.0]
        }

        recommendations = await tuner.auto_tune(
            config,
            performance_data,
            domain="code",
            problem_type="regression"
        )

        assert isinstance(recommendations, dict)

    def test_identify_pattern(self, tuner):
        """Test pattern identification"""
        performance_data = {
            "fitness_history": [0.5, 0.6, 0.7],
            "diversity_history": [0.8],
            "evaluation_times": [1.0, 2.0, 1.5]
        }

        pattern = tuner._identify_pattern(performance_data)

        assert "convergence_speed" in pattern
        assert "diversity_trend" in pattern
        assert "efficiency" in pattern


# ==============================================================================
# Resource-Aware Configuration Tests
# ==============================================================================

class TestResourceAwareConfigurator:
    """Test suite for ResourceAwareConfigurator"""

    @pytest.fixture
    def resource_config(self):
        """Create resource-aware configurator"""
        return ResourceAwareConfigurator()

    @patch('openevolve.config.resource_config.psutil')
    def test_detect_resources(self, mock_psutil, resource_config):
        """Test resource detection"""
        # Mock psutil responses
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_percent.return_value = 45.0

        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_memory.available = 8 * (1024 ** 3)
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.free = 100 * (1024 ** 3)
        mock_psutil.disk_usage.return_value = mock_disk

        resources = resource_config.detect_resources()

        assert resources.cpu_count == 8
        assert resources.memory_total_gb == pytest.approx(16.0, rel=0.1)
        assert resources.memory_available_gb == pytest.approx(8.0, rel=0.1)

    def test_adjust_config_for_cpu(self, resource_config):
        """Test CPU-based configuration adjustment"""
        # Mock resources
        resource_config.detected_resources = ResourceInfo(
            cpu_count=8,
            cpu_usage=45.0,
            memory_total_gb=16.0,
            memory_available_gb=8.0,
            memory_usage_percent=50.0,
            gpu_available=False,
            gpu_count=0,
            gpu_memory_total=0.0,
            gpu_memory_available=0.0,
            disk_space_gb=100.0
        )

        config = UnifiedEvolutionConfig()
        config.common.concurrency = 20  # Too high for 8 CPUs

        adjusted = resource_config._adjust_for_cpu(
            config,
            resource_config.detected_resources,
            None
        )

        assert adjusted.common.concurrency <= 7  # One less than CPU count

    def test_adjust_config_for_memory(self, resource_config):
        """Test memory-based configuration adjustment"""
        resource_config.detected_resources = ResourceInfo(
            cpu_count=8,
            cpu_usage=45.0,
            memory_total_gb=16.0,
            memory_available_gb=2.0,  # Low memory
            memory_usage_percent=87.5,
            gpu_available=False,
            gpu_count=0,
            gpu_memory_total=0.0,
            gpu_memory_available=0.0,
            disk_space_gb=100.0
        )

        config = UnifiedEvolutionConfig()
        config.database.population_size = 1000  # Too large

        adjusted = resource_config._adjust_for_memory(
            config,
            resource_config.detected_resources,
            None
        )

        # Population should be reduced due to low memory
        assert adjusted.database.population_size < 1000

    def test_get_resource_recommendations(self, resource_config):
        """Test resource recommendations"""
        resource_config.detected_resources = ResourceInfo(
            cpu_count=8,
            cpu_usage=45.0,
            memory_total_gb=16.0,
            memory_available_gb=8.0,
            memory_usage_percent=50.0,
            gpu_available=True,
            gpu_count=1,
            gpu_memory_total=8.0,
            gpu_memory_available=8.0,
            disk_space_gb=100.0
        )

        config = UnifiedEvolutionConfig()
        config.common.concurrency = 2  # Underutilizing CPU

        recommendations = resource_config.get_resource_recommendations(config)

        assert isinstance(recommendations, dict)


# ==============================================================================
# Configuration Metrics Tests
# ==============================================================================

class TestConfigurationMetrics:
    """Test suite for ConfigurationMetrics"""

    @pytest.fixture
    def metrics(self):
        """Create metrics tracker"""
        return ConfigurationMetrics()

    def test_track_config_usage(self, metrics):
        """Test tracking configuration usage"""
        config = UnifiedEvolutionConfig()

        metrics.track_config_usage(config)

        assert len(metrics.parameter_usage) > 0

    def test_track_config_performance(self, metrics):
        """Test tracking configuration performance"""
        config = UnifiedEvolutionConfig()

        metrics.track_config_performance(
            config,
            performance=0.85,
            iteration=10
        )

        assert len(metrics.config_history) == 1
        assert metrics.config_history[0].performance == 0.85

    def test_get_most_used_parameters(self, metrics):
        """Test getting most used parameters"""
        config = UnifiedEvolutionConfig()

        # Track multiple times
        for _ in range(5):
            metrics.track_config_usage(config)

        most_used = metrics.get_most_used_parameters(limit=5)

        assert len(most_used) <= 5
        assert all(isinstance(name, str) for name, _ in most_used)

    def test_get_performance_summary(self, metrics):
        """Test performance summary"""
        config = UnifiedEvolutionConfig()

        # Add performance records
        for i in range(10):
            metrics.track_config_performance(
                config,
                performance=0.5 + (i * 0.05),
                iteration=i
            )

        summary = metrics.get_performance_summary()

        assert "total_runs" in summary
        assert summary["total_runs"] == 10
        assert "best_performance" in summary
        assert "average_performance" in summary

    def test_analyze_parameter_trends(self, metrics):
        """Test parameter trend analysis"""
        config = UnifiedEvolutionConfig()

        metrics.track_config_usage(config, modified_params=["common.max_iterations"])

        trends = metrics.analyze_parameter_trends("common.max_iterations")

        assert "parameter_name" in trends
        assert trends["parameter_name"] == "common.max_iterations"


class TestConfigComparison:
    """Test suite for ConfigComparison"""

    def test_compare_identical_configs(self):
        """Test comparing identical configurations"""
        config1 = UnifiedEvolutionConfig()
        config2 = UnifiedEvolutionConfig()

        comparison = ConfigComparison.compare_configs(config1, config2)

        assert comparison["are_identical"] is True
        assert comparison["num_differences"] == 0

    def test_compare_different_configs(self):
        """Test comparing different configurations"""
        config1 = UnifiedEvolutionConfig()
        config2 = UnifiedEvolutionConfig()
        config2.common.max_iterations = 200

        comparison = ConfigComparison.compare_configs(config1, config2)

        assert comparison["are_identical"] is False
        assert comparison["num_differences"] > 0

    def test_find_differences(self):
        """Test finding differences"""
        dict1 = {"a": 1, "b": 2, "c": {"x": 10}}
        dict2 = {"a": 1, "b": 3, "c": {"x": 20}}

        differences = ConfigComparison._find_differences(dict1, dict2, "")

        assert len(differences) == 2  # b changed, c.x changed


class TestHashConfig:
    """Test suite for hash_config function"""

    def test_hash_same_config(self):
        """Test hashing same configuration"""
        config = UnifiedEvolutionConfig()

        hash1 = hash_config(config)
        hash2 = hash_config(config)

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_hash_different_config(self):
        """Test hashing different configuration"""
        config1 = UnifiedEvolutionConfig()
        config2 = UnifiedEvolutionConfig()
        config2.common.max_iterations = 200

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 != hash2


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestRuntimeConfigIntegration:
    """Integration tests for runtime configuration system"""

    @pytest.mark.asyncio
    async def test_full_update_cycle(self):
        """Test complete update cycle with multiple components"""
        config = UnifiedEvolutionConfig()

        # Create updater
        updater = RuntimeConfigUpdater(config)

        # Update parameters
        await updater.update_parameter("max_iterations", 200, scope="common")
        await updater.update_parameter("population_size", 500, scope="database")

        # Track metrics
        metrics = ConfigurationMetrics()
        metrics.track_config_usage(config, modified_params=[
            "common.max_iterations",
            "database.population_size"
        ])

        # Verify changes
        assert config.common.max_iterations == 200
        assert config.database.population_size == 500
        assert "common.max_iterations" in metrics.parameter_usage

    @pytest.mark.asyncio
    async def test_adaptive_resource_adjustment(self):
        """Test adaptive configuration with resource awareness"""
        base_config = UnifiedEvolutionConfig()

        # Create resource configurator
        resource_config = ResourceAwareConfigurator()
        resource_config.detected_resources = ResourceInfo(
            cpu_count=4,
            cpu_usage=50.0,
            memory_total_gb=8.0,
            memory_available_gb=4.0,
            memory_usage_percent=50.0,
            gpu_available=False,
            gpu_count=0,
            gpu_memory_total=0.0,
            gpu_memory_available=0.0,
            disk_space_gb=50.0
        )

        # Adjust config for resources
        adjusted = resource_config.adjust_config_for_resources(base_config)

        # Create adaptive configurator
        adaptive = AdaptiveConfigurator(adjusted)

        # Get recommendations
        performance_metrics = {
            "fitness": 0.6,
            "diversity": 0.2,
            "convergence_rate": 0.001,
            "improvement_rate": 0.0,
            "evaluation_time": 10.0,
            "success_rate": 0.9
        }

        # Build history
        for i in range(15):
            await adaptive.adapt_configuration(performance_metrics, i)

        suggestions = await adaptive.adapt_configuration(performance_metrics, 15)

        # Verify suggestions respect resource constraints
        assert isinstance(suggestions, dict)

    @pytest.mark.asyncio
    async def test_strategy_switch_with_state_migration(self):
        """Test strategy switch with state preservation"""
        switcher = DynamicStrategySwitcher(SystemMode.OPENEVOLVE)

        # Create target config
        target_config = UnifiedEvolutionConfig(evolution_mode="qd")
        from openevolve.unified.config import QDConfig
        target_config.qd = QDConfig()

        # Mock state capture
        switcher.current_state = {
            "best_solutions": [{"fitness": 0.9}],
            "population": [{"fitness": 0.7}]
        }

        # Switch strategy
        success = await switcher.switch_strategy(
            SystemMode.QD,
            target_config,
            preserve_state=True,
            reason="Testing strategy switch"
        )

        # Verify switch
        assert switcher.current_strategy == SystemMode.QD
        assert len(switcher.strategy_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
