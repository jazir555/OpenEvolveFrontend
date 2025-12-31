"""
Comprehensive Test Suite for Hybrid MAKER Configuration System

Tests configuration validation, serialization, presets, and utility functions.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from hybrid_maker_config import (
    HybridMakerConfig,
    HybridStrategyProfile,
    LeanAideConfig,
    MakerConfig,
    MCTSConfig,
    EvolutionConfig,
    MDAPConfig,
    AdaptiveConfig,
    PerformanceThresholds,
    HybridMakerConfigPreset,
    StrategyType,
    create_config_from_preset,
    merge_configs,
    validate_and_create_config,
    get_available_presets,
    compare_configs,
    export_config_summary,
)


class TestLeanAideConfig(unittest.TestCase):
    """Test LeanAide configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = LeanAideConfig()
        self.assertEqual(config.server_url, "http://localhost:8080")
        self.assertEqual(config.timeout, 30)
        self.assertTrue(config.verify_tactics)
        self.assertTrue(config.cache_verification_results)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = LeanAideConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validation_invalid_timeout(self):
        """Test validation catches invalid timeout"""
        config = LeanAideConfig(timeout=-1)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("timeout must be positive", errors)

    def test_validation_invalid_retries(self):
        """Test validation catches invalid retries"""
        config = LeanAideConfig(max_retries=-1)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("max_retries cannot be negative", errors)

    def test_validation_invalid_parallel(self):
        """Test validation catches invalid parallel count"""
        config = LeanAideConfig(parallel_verifications=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("parallel_verifications must be at least 1", errors)


class TestMakerConfig(unittest.TestCase):
    """Test MAKER configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = MakerConfig()
        self.assertEqual(config.k_min, 2)
        self.assertEqual(config.k_max, 8)
        self.assertEqual(config.max_votes_per_step, 50)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = MakerConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_validation_k_max_less_than_min(self):
        """Test validation catches k_max < k_min"""
        config = MakerConfig(k_min=5, k_max=3)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("k_max must be >= k_min", errors)

    def test_validation_invalid_confidence(self):
        """Test validation catches invalid confidence"""
        config = MakerConfig(min_confidence=1.5)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("min_confidence must be between 0 and 1", errors)


class TestMCTSConfig(unittest.TestCase):
    """Test MCTS configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = MCTSConfig()
        self.assertEqual(config.num_simulations, 1000)
        self.assertAlmostEqual(config.exploration_constant, 1.414)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = MCTSConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_validation_invalid_simulations(self):
        """Test validation catches invalid simulations"""
        config = MCTSConfig(num_simulations=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("num_simulations must be at least 1", errors)

    def test_validation_invalid_exploration_constant(self):
        """Test validation catches negative exploration constant"""
        config = MCTSConfig(exploration_constant=-1.0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("exploration_constant cannot be negative", errors)

    def test_validation_invalid_discount_factor(self):
        """Test validation catches discount factor out of range"""
        config = MCTSConfig(discount_factor=1.5)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("discount_factor must be between 0 and 1", errors)


class TestEvolutionConfig(unittest.TestCase):
    """Test Evolution configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = EvolutionConfig()
        self.assertEqual(config.population_size, 100)
        self.assertEqual(config.generations, 100)
        self.assertEqual(config.mutation_rate, 0.1)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = EvolutionConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_validation_small_population(self):
        """Test validation catches too small population"""
        config = EvolutionConfig(population_size=5)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("population_size must be at least 10", errors)

    def test_validation_invalid_mutation_rate(self):
        """Test validation catches invalid mutation rate"""
        config = EvolutionConfig(mutation_rate=1.5)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("mutation_rate must be between 0 and 1", errors)

    def test_validation_invalid_islands(self):
        """Test validation catches invalid island count"""
        config = EvolutionConfig(num_islands=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("num_islands must be at least 1", errors)


class TestMDAPConfig(unittest.TestCase):
    """Test MDAP configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = MDAPConfig()
        self.assertEqual(config.decomposition_depth, 3)
        self.assertEqual(config.agent_count, 5)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = MDAPConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_validation_invalid_depth(self):
        """Test validation catches invalid depth"""
        config = MDAPConfig(decomposition_depth=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("decomposition_depth must be at least 1", errors)

    def test_validation_max_less_than_min(self):
        """Test validation catches max_size < min_size"""
        config = MDAPConfig(min_subproblem_size=50, max_subproblem_size=10)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("max_subproblem_size must be >= min_subproblem_size", errors)


class TestHybridStrategyProfile(unittest.TestCase):
    """Test hybrid strategy profile"""

    def test_default_profile(self):
        """Test default profile creation"""
        profile = HybridStrategyProfile(strategy_type=StrategyType.MAKER)
        self.assertTrue(profile.enabled)
        self.assertEqual(profile.performance_weight, 1.0)
        self.assertEqual(profile.priority, 0)

    def test_validation_valid(self):
        """Test validation of valid profile"""
        profile = HybridStrategyProfile(strategy_type=StrategyType.MCTS)
        valid, errors = profile.validate()
        self.assertTrue(valid)

    def test_validation_invalid_weight(self):
        """Test validation catches negative weight"""
        profile = HybridStrategyProfile(strategy_type=StrategyType.EVOLUTION, performance_weight=-1.0)
        valid, errors = profile.validate()
        self.assertFalse(valid)
        self.assertIn("performance_weight cannot be negative", errors)

    def test_validation_invalid_cpu_allocation(self):
        """Test validation catches invalid CPU allocation"""
        profile = HybridStrategyProfile(strategy_type=StrategyType.MDAP, cpu_allocation=1.5)
        valid, errors = profile.validate()
        self.assertFalse(valid)
        self.assertIn("cpu_allocation must be between 0 and 1", errors)


class TestAdaptiveConfig(unittest.TestCase):
    """Test adaptive configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = AdaptiveConfig()
        self.assertTrue(config.enable_adaptive_selection)
        self.assertEqual(config.adaptation_interval, 10)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = AdaptiveConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_validation_invalid_interval(self):
        """Test validation catches invalid interval"""
        config = AdaptiveConfig(adaptation_interval=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("adaptation_interval must be at least 1", errors)

    def test_validation_invalid_exploration_rate(self):
        """Test validation catches exploration rate out of range"""
        config = AdaptiveConfig(exploration_rate=1.5)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("exploration_rate must be between 0 and 1", errors)


class TestPerformanceThresholds(unittest.TestCase):
    """Test performance thresholds"""

    def test_default_thresholds(self):
        """Test default threshold creation"""
        thresholds = PerformanceThresholds()
        self.assertEqual(thresholds.fast_time_threshold, 60)
        self.assertEqual(thresholds.balanced_time_threshold, 300)
        self.assertEqual(thresholds.thorough_time_threshold, 1800)

    def test_validation_valid(self):
        """Test validation of valid thresholds"""
        thresholds = PerformanceThresholds()
        valid, errors = thresholds.validate()
        self.assertTrue(valid)

    def test_validation_invalid_ordering(self):
        """Test validation catches incorrect threshold ordering"""
        thresholds = PerformanceThresholds(
            fast_time_threshold=100,
            balanced_time_threshold=50,
            thorough_time_threshold=200
        )
        valid, errors = thresholds.validate()
        self.assertFalse(valid)
        self.assertIn("balanced_time_threshold must be > fast_time_threshold", errors)


class TestHybridMakerConfig(unittest.TestCase):
    """Test main hybrid configuration"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = HybridMakerConfig()
        self.assertEqual(config.config_name, "default")
        self.assertEqual(config.default_strategy, StrategyType.MAKER)
        self.assertTrue(config.enable_parallel_strategies)

    def test_default_profiles_initialized(self):
        """Test that default strategy profiles are initialized"""
        config = HybridMakerConfig()
        self.assertEqual(len(config.strategy_profiles), 5)
        self.assertIn("leanaide", config.strategy_profiles)
        self.assertIn("maker", config.strategy_profiles)
        self.assertIn("mcts", config.strategy_profiles)
        self.assertIn("evolution", config.strategy_profiles)
        self.assertIn("mdap", config.strategy_profiles)

    def test_validation_valid(self):
        """Test validation of valid configuration"""
        config = HybridMakerConfig()
        valid, errors = config.validate()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_estimate_runtime(self):
        """Test runtime estimation"""
        config = HybridMakerConfig()
        runtime = config.estimate_runtime()

        self.assertIn("leanaide", runtime)
        self.assertIn("maker", runtime)
        self.assertIn("mcts", runtime)
        self.assertIn("evolution", runtime)
        self.assertIn("mdap", runtime)

        # All runtimes should be positive
        for strategy, time in runtime.items():
            self.assertGreater(time, 0)

    def test_estimate_runtime_single_strategy(self):
        """Test runtime estimation for single strategy"""
        config = HybridMakerConfig()
        runtime = config.estimate_runtime(StrategyType.MAKER)

        self.assertIn("maker", runtime)
        self.assertNotIn("leanaide", runtime)
        self.assertNotIn("mcts", runtime)
        self.assertNotIn("evolution", runtime)
        self.assertNotIn("mdap", runtime)

    def test_estimate_resource_usage(self):
        """Test resource usage estimation"""
        config = HybridMakerConfig()
        usage = config.estimate_resource_usage()

        for strategy in ["leanaide", "maker", "mcts", "evolution", "mdap"]:
            self.assertIn(strategy, usage)
            self.assertIn("cpu", usage[strategy])
            self.assertIn("memory_mb", usage[strategy])
            self.assertGreater(usage[strategy]["cpu"], 0)
            self.assertGreater(usage[strategy]["memory_mb"], 0)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = HybridMakerConfig()
        data = config.to_dict()

        self.assertIsInstance(data, dict)
        self.assertIn("config_name", data)
        self.assertIn("leanaide_config", data)
        self.assertIn("maker_config", data)
        self.assertIn("strategy_profiles", data)

    def test_from_dict(self):
        """Test creation from dictionary"""
        config1 = HybridMakerConfig()
        data = config1.to_dict()
        config2 = HybridMakerConfig.from_dict(data)

        self.assertEqual(config1.config_name, config2.config_name)
        self.assertEqual(config1.default_strategy, config2.default_strategy)

    def test_roundtrip_serialization(self):
        """Test round-trip serialization"""
        config1 = HybridMakerConfig()
        data = config1.to_dict()
        config2 = HybridMakerConfig.from_dict(data)

        # Check that all sub-configs match
        self.assertEqual(config1.leanaide_config.timeout, config2.leanaide_config.timeout)
        self.assertEqual(config1.maker_config.k_min, config2.maker_config.k_min)
        self.assertEqual(config1.mcts_config.num_simulations, config2.mcts_config.num_simulations)

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML file"""
        config1 = HybridMakerConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.yaml"

            # Save
            success = config1.save_to_file(filepath, format="yaml")
            self.assertTrue(success)
            self.assertTrue(filepath.exists())

            # Load
            config2 = HybridMakerConfig.load_from_file(filepath)
            self.assertIsNotNone(config2)
            self.assertEqual(config1.config_name, config2.config_name)

    def test_save_and_load_json(self):
        """Test saving and loading JSON file"""
        config1 = HybridMakerConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.json"

            # Save
            success = config1.save_to_file(filepath, format="json")
            self.assertTrue(success)
            self.assertTrue(filepath.exists())

            # Load
            config2 = HybridMakerConfig.load_from_file(filepath)
            self.assertIsNotNone(config2)
            self.assertEqual(config1.config_name, config2.config_name)

    def test_merge_configs(self):
        """Test merging two configurations"""
        config1 = HybridMakerConfig()
        config2 = HybridMakerConfig(config_name="merged")

        config2.maker_config.k_min = 5

        merged = config1.merge_with(config2)

        # Check that non-default values are merged
        self.assertEqual(merged.maker_config.k_min, 5)
        self.assertEqual(merged.config_name, "merged")

    def test_invalid_global_timeout(self):
        """Test validation catches invalid timeout"""
        config = HybridMakerConfig(global_timeout=-1)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("global_timeout must be positive", errors)

    def test_invalid_max_parallel(self):
        """Test validation catches invalid max parallel"""
        config = HybridMakerConfig(max_parallel_strategies=0)
        valid, errors = config.validate()
        self.assertFalse(valid)
        self.assertIn("max_parallel_strategies must be at least 1", errors)


class TestHybridMakerConfigPreset(unittest.TestCase):
    """Test configuration presets"""

    def test_fast_preset(self):
        """Test fast preset"""
        config = HybridMakerConfigPreset.fast()
        self.assertEqual(config.config_name, "fast")
        self.assertLess(config.maker_config.max_votes_per_step, 20)
        self.assertLess(config.mcts_config.num_simulations, 200)

    def test_balanced_preset(self):
        """Test balanced preset"""
        config = HybridMakerConfigPreset.balanced()
        self.assertEqual(config.config_name, "balanced")
        self.assertGreater(config.maker_config.max_votes_per_step, 20)
        self.assertGreater(config.mcts_config.num_simulations, 200)

    def test_thorough_preset(self):
        """Test thorough preset"""
        config = HybridMakerConfigPreset.thorough()
        self.assertEqual(config.config_name, "thorough")
        self.assertGreater(config.maker_config.max_votes_per_step, 80)
        self.assertGreater(config.mcts_config.num_simulations, 1000)

    def test_leanaide_focused_preset(self):
        """Test LeanAide focused preset"""
        config = HybridMakerConfigPreset.leanaide_focused()
        self.assertEqual(config.default_strategy, StrategyType.LEANAIDE)
        self.assertFalse(config.strategy_profiles["maker"].enabled)
        self.assertFalse(config.strategy_profiles["mcts"].enabled)

    def test_maker_focused_preset(self):
        """Test MAKER focused preset"""
        config = HybridMakerConfigPreset.maker_focused()
        self.assertEqual(config.default_strategy, StrategyType.MAKER)
        self.assertFalse(config.strategy_profiles["leanaide"].enabled)
        self.assertTrue(config.strategy_profiles["mdap"].enabled)

    def test_adaptive_preset(self):
        """Test adaptive preset"""
        config = HybridMakerConfigPreset.adaptive()
        self.assertEqual(config.default_strategy, StrategyType.ADAPTIVE)
        self.assertTrue(config.adaptive_config.enable_adaptive_selection)
        self.assertTrue(config.enable_parallel_strategies)

    def test_research_preset(self):
        """Test research preset"""
        config = HybridMakerConfigPreset.research()
        self.assertEqual(config.log_level, "DEBUG")
        self.assertTrue(config.enable_metrics)
        self.assertTrue(config.checkpoint_enabled)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_create_config_from_preset(self):
        """Test creating config from preset name"""
        config = create_config_from_preset("fast")
        self.assertIsNotNone(config)
        self.assertEqual(config.config_name, "fast")

    def test_create_config_from_invalid_preset(self):
        """Test creating config from invalid preset name"""
        config = create_config_from_preset("nonexistent")
        self.assertIsNone(config)

    def test_merge_multiple_configs(self):
        """Test merging multiple configurations"""
        config1 = HybridMakerConfig()
        config2 = HybridMakerConfig(config_name="config2")
        config3 = HybridMakerConfig(config_name="config3")

        merged = merge_configs(config1, config2, config3)
        self.assertEqual(merged.config_name, "config3")

    def test_merge_empty_configs(self):
        """Test merging empty list of configs"""
        merged = merge_configs()
        self.assertIsInstance(merged, HybridMakerConfig)

    def test_validate_and_create_valid(self):
        """Test validate and create with valid data"""
        config = HybridMakerConfig()
        data = config.to_dict()

        result, errors = validate_and_create_config(data)
        self.assertIsNotNone(result)
        self.assertEqual(len(errors), 0)

    def test_validate_and_create_invalid(self):
        """Test validate and create with invalid data"""
        data = {"maker_config": {"k_min": 10, "k_max": 5}}

        result, errors = validate_and_create_config(data)
        self.assertIsNone(result)
        self.assertGreater(len(errors), 0)

    def test_get_available_presets(self):
        """Test getting available presets"""
        presets = get_available_presets()

        self.assertIsInstance(presets, list)
        self.assertIn("fast", presets)
        self.assertIn("balanced", presets)
        self.assertIn("thorough", presets)
        self.assertIn("adaptive", presets)

    def test_compare_configs(self):
        """Test comparing two configurations"""
        config1 = HybridMakerConfig()
        config2 = HybridMakerConfig()

        config2.maker_config.k_min = 5
        config2.strategy_profiles["leanaide"].enabled = False

        diff = compare_configs(config1, config2)

        self.assertIn("changed", diff)
        self.assertIn("unchanged", diff)
        self.assertGreater(len(diff["changed"]), 0)

    def test_export_config_summary(self):
        """Test exporting configuration summary"""
        config = HybridMakerConfig(config_name="test", description="Test config")
        summary = export_config_summary(config)

        self.assertIsInstance(summary, str)
        self.assertIn("test", summary)
        self.assertIn("Test config", summary)
        self.assertIn("Strategy Configuration", summary)
        self.assertIn("Resource Estimates", summary)


class TestConfigurationValidation(unittest.TestCase):
    """Test comprehensive configuration validation"""

    def test_all_strategies_enabled(self):
        """Test configuration with all strategies enabled"""
        config = HybridMakerConfig()

        for profile in config.strategy_profiles.values():
            profile.enabled = True

        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_all_strategies_disabled(self):
        """Test configuration with all strategies disabled"""
        config = HybridMakerConfig()

        for profile in config.strategy_profiles.values():
            profile.enabled = False

        valid, errors = config.validate()
        self.assertTrue(valid)  # This is valid, just not useful

    def test_extreme_values(self):
        """Test configuration with extreme values"""
        config = HybridMakerConfig()

        # Set extreme but valid values
        config.maker_config.k_max = 100
        config.mcts_config.num_simulations = 100000
        config.evolution_config.population_size = 10000
        config.global_timeout = 86400  # 24 hours

        valid, errors = config.validate()
        self.assertTrue(valid)

    def test_boundary_values(self):
        """Test configuration at boundary values"""
        config = HybridMakerConfig()

        # Set minimum valid values
        config.maker_config.k_min = 1
        config.maker_config.max_votes_per_step = 1
        config.maker_config.min_agents = 1
        config.checkpoint_interval = 1

        valid, errors = config.validate()
        self.assertTrue(valid)


class TestConfigurationFileHandling(unittest.TestCase):
    """Test file handling for configurations"""

    def setUp(self):
        """Set up temporary directory"""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_to_nonexistent_directory(self):
        """Test saving to non-existent directory"""
        config = HybridMakerConfig()
        filepath = Path(self.tmpdir) / "subdir" / "config.yaml"

        success = config.save_to_file(filepath)
        self.assertTrue(success)
        self.assertTrue(filepath.exists())

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        filepath = Path(self.tmpdir) / "nonexistent.yaml"
        config = HybridMakerConfig.load_from_file(filepath)
        self.assertIsNone(config)

    def test_save_with_invalid_format(self):
        """Test saving with invalid format"""
        config = HybridMakerConfig()
        filepath = Path(self.tmpdir) / "config.txt"

        success = config.save_to_file(filepath, format="invalid")
        self.assertFalse(success)

    def test_save_and_load_preserves_data(self):
        """Test that save/load preserves all data"""
        config1 = HybridMakerConfig(
            config_name="test",
            description="Test description",
            tags=["tag1", "tag2"]
        )
        config1.maker_config.k_min = 7
        config1.strategy_profiles["leanaide"].priority = 10

        filepath = Path(self.tmpdir) / "config.yaml"
        config1.save_to_file(filepath)
        config2 = HybridMakerConfig.load_from_file(filepath)

        self.assertEqual(config2.config_name, "test")
        self.assertEqual(config2.description, "Test description")
        self.assertEqual(config2.tags, ["tag1", "tag2"])
        self.assertEqual(config2.maker_config.k_min, 7)
        self.assertEqual(config2.strategy_profiles["leanaide"].priority, 10)


class TestRuntimeEstimates(unittest.TestCase):
    """Test runtime and resource estimation"""

    def test_fast_preset_runtime(self):
        """Test runtime estimates for fast preset"""
        config = HybridMakerConfigPreset.fast()
        runtime = config.estimate_runtime()

        # Fast preset should have lower runtimes than balanced
        self.assertLess(runtime["maker"], 10000)
        self.assertLess(runtime["mcts"], 1000)
        self.assertLess(runtime["evolution"], 1000)

        # Verify it's faster than balanced preset
        balanced_config = HybridMakerConfigPreset.balanced()
        balanced_runtime = balanced_config.estimate_runtime()
        self.assertLess(runtime["maker"], balanced_runtime["maker"])

    def test_thorough_preset_runtime(self):
        """Test runtime estimates for thorough preset"""
        config = HybridMakerConfigPreset.thorough()
        runtime = config.estimate_runtime()

        # Thorough preset should have high runtimes
        self.assertGreater(runtime["maker"], 1000)
        self.assertGreater(runtime["mcts"], 1000)

    def test_resource_estimates_scales_with_params(self):
        """Test that resource estimates scale with parameters"""
        config1 = HybridMakerConfig()
        config2 = HybridMakerConfig()

        config2.maker_config.min_agents = 10
        config2.maker_config.max_agents = 20

        usage1 = config1.estimate_resource_usage()
        usage2 = config2.estimate_resource_usage()

        # config2 should require more resources
        self.assertGreater(usage2["maker"]["cpu"], usage1["maker"]["cpu"])
        self.assertGreater(usage2["maker"]["memory_mb"], usage1["maker"]["memory_mb"])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
