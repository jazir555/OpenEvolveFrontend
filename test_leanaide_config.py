"""
Test Suite for LeanAide Configuration Module

Comprehensive tests for:
- Configuration loading from multiple sources
- Configuration validation
- Environment variable overrides
- Default values
- Configuration migration
- Edge cases and error handling
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from dataclasses import asdict

from leanaide_config import (
    LeanAideConfig,
    LeanAideServerConfig,
    LeanAideVerificationConfig,
    LeanAideCacheConfig,
    LeanAideWorkflowConfig,
    LeanAideLean4Config,
    LeanAideLoggingConfig,
    LeanAideSecurityConfig,
    LeanAidePerformanceConfig,
    LeanAideConfigLoader,
    LeanAideConfigMigrator,
    load_leanaide_config,
    get_leanaide_config,
    reload_leanaide_config,
    get_leanaide_config_summary,
    ValidationError,
)


class TestLeanAideConfigDataclasses:
    """Test configuration dataclasses."""

    def test_server_config_defaults(self):
        """Test LeanAideServerConfig default values."""
        config = LeanAideServerConfig()
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.use_ssl is False
        assert config.api_version == "v1"

    def test_server_config_base_url(self):
        """Test get_base_url method."""
        # HTTP
        config = LeanAideServerConfig(host="example.com", port=8080, use_ssl=False)
        assert config.get_base_url() == "http://example.com:8080/v1"

        # HTTPS
        config = LeanAideServerConfig(host="example.com", port=443, use_ssl=True)
        assert config.get_base_url() == "https://example.com:443/v1"

    def test_verification_config_defaults(self):
        """Test LeanAideVerificationConfig default values."""
        config = LeanAideVerificationConfig()
        assert config.enable_auto is True
        assert config.complexity_threshold == 50
        assert config.domains == ["mathlib"]
        assert config.timeout_per_proof == 120.0
        assert config.parallel_verifications == 4
        assert config.verification_strategy == "adaptive"

    def test_cache_config_defaults(self):
        """Test LeanAideCacheConfig default values."""
        config = LeanAideCacheConfig()
        assert config.enable is True
        assert config.ttl == 86400
        assert config.cache_dir == "./leanaide_cache"
        assert config.max_cache_size_mb == 500
        assert config.compression_enabled is True

    def test_workflow_config_defaults(self):
        """Test LeanAideWorkflowConfig default values."""
        config = LeanAideWorkflowConfig()
        assert config.stage_3c_enabled is True
        assert config.stage_5_enabled is True
        assert config.stage_3c_priority == 7
        assert config.stage_5_priority == 8
        assert config.async_verification is True
        assert config.failure_action == "warn"

    def test_full_config_defaults(self):
        """Test LeanAideConfig with all defaults."""
        config = LeanAideConfig()
        assert config.enabled is True
        assert config.environment == "development"
        assert config.server.host == "localhost"
        assert config.verification.enable_auto is True
        assert config.cache.enable is True


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_configuration(self):
        """Test validation passes for valid configuration."""
        config = LeanAideConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_server_port(self):
        """Test validation fails for invalid port."""
        config = LeanAideConfig()
        config.server.port = 70000  # Invalid port
        errors = config.validate()
        assert len(errors) > 0
        assert any("port" in e.lower() for e in errors)

    def test_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        config = LeanAideConfig()
        config.server.timeout = -1.0  # Invalid timeout
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_complexity_threshold(self):
        """Test validation fails for invalid complexity threshold."""
        config = LeanAideConfig()
        config.verification.complexity_threshold = 150  # Out of range
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_verification_strategy(self):
        """Test validation fails for invalid verification strategy."""
        config = LeanAideConfig()
        config.verification.verification_strategy = "invalid"
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_failure_action(self):
        """Test validation fails for invalid failure action."""
        config = LeanAideConfig()
        config.workflow.failure_action = "invalid"
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_priority(self):
        """Test validation fails for invalid priority values."""
        config = LeanAideConfig()
        config.workflow.stage_3c_priority = 15  # Out of range
        errors = config.validate()
        assert len(errors) > 0


class TestConfigurationLoader:
    """Test configuration loading from files."""

    def test_load_with_defaults(self):
        """Test loading configuration with default values."""
        loader = LeanAideConfigLoader(config_dir=Path.cwd())
        config = loader.load()
        assert isinstance(config, LeanAideConfig)
        assert config.server.host == "localhost"
        assert config.verification.enable_auto is True

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "leanaide_config.yaml"
            config_data = {
                "server": {
                    "host": "test.example.com",
                    "port": 9090
                },
                "verification": {
                    "enable_auto": False,
                    "complexity_threshold": 75
                }
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            loader = LeanAideConfigLoader(config_dir=Path(tmpdir))
            config = loader.load()

            assert config.server.host == "test.example.com"
            assert config.server.port == 9090
            assert config.verification.enable_auto is False
            assert config.verification.complexity_threshold == 75

    def test_load_with_overrides(self):
        """Test loading configuration with Python API overrides."""
        loader = LeanAideConfigLoader(config_dir=Path.cwd())
        config = loader.load(
            server_host="override.example.com",
            server_port=9999,
            verification_enable_auto=False
        )

        assert config.server.host == "override.example.com"
        assert config.server.port == 9999
        assert config.verification.enable_auto is False

    def test_precedence_order(self):
        """Test configuration source precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create YAML config
            config_path = Path(tmpdir) / "leanaide_config.yaml"
            config_data = {
                "server": {
                    "host": "yaml.example.com",
                    "port": 8080
                }
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Set environment variable
            os.environ["LEANAIDE_SERVER_PORT"] = "9090"

            try:
                loader = LeanAideConfigLoader(config_dir=Path(tmpdir))
                config = loader.load(server_host="python.example.com")

                # Python API override should win
                assert config.server.host == "python.example.com"

                # Environment variable should override YAML
                assert config.server.port == 9090
            finally:
                del os.environ["LEANAIDE_SERVER_PORT"]


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_server_host_env_var(self):
        """Test LEANAIDE_SERVER_HOST environment variable."""
        os.environ["LEANAIDE_SERVER_HOST"] = "env.example.com"

        try:
            config = load_leanaide_config(force_reload=True)
            assert config.server.host == "env.example.com"
        finally:
            del os.environ["LEANAIDE_SERVER_HOST"]

    def test_verification_threshold_env_var(self):
        """Test LEANAIDE_VERIFICATION_COMPLEXITY_THRESHOLD environment variable."""
        os.environ["LEANAIDE_VERIFICATION_COMPLEXITY_THRESHOLD"] = "85"

        try:
            config = load_leanaide_config(force_reload=True)
            assert config.verification.complexity_threshold == 85
        finally:
            del os.environ["LEANAIDE_VERIFICATION_COMPLEXITY_THRESHOLD"]

    def test_cache_enable_env_var(self):
        """Test LEANAIDE_CACHE_ENABLE environment variable."""
        os.environ["LEANAIDE_CACHE_ENABLE"] = "false"

        try:
            config = load_leanaide_config(force_reload=True)
            assert config.cache.enable is False
        finally:
            del os.environ["LEANAIDE_CACHE_ENABLE"]

    def test_workflow_failure_action_env_var(self):
        """Test LEANAIDE_WORKFLOW_FAILURE_ACTION environment variable."""
        os.environ["LEANAIDE_WORKFLOW_FAILURE_ACTION"] = "error"

        try:
            config = load_leanaide_config(force_reload=True)
            assert config.workflow.failure_action == "error"
        finally:
            del os.environ["LEANAIDE_WORKFLOW_FAILURE_ACTION"]

    def test_boolean_env_var_variations(self):
        """Test various boolean environment variable formats."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ]

        for value, expected in test_cases:
            os.environ["LEANAIDE_CACHE_ENABLE"] = value
            try:
                config = load_leanaide_config(force_reload=True)
                assert config.cache.enable == expected
            finally:
                del os.environ["LEANAIDE_CACHE_ENABLE"]


class TestConfigurationMigration:
    """Test configuration migration support."""

    def test_current_version_no_migration(self):
        """Test that current version doesn't need migration."""
        config_data = {
            "server": {"host": "localhost"},
            "verification": {"enable_auto": True}
        }

        migrated = LeanAideConfigMigrator.migrate(
            config_data,
            LeanAideConfigMigrator.CURRENT_VERSION
        )

        assert migrated == config_data

    def test_migration_from_090(self):
        """Test migration from version 0.9.0."""
        config_data = {
            "server": {
                "host": "localhost",
                "port": 8080
            },
            "verification": {
                "enable_auto": True
            },
            "cache": {
                "enable": True
            }
        }

        migrated = LeanAideConfigMigrator.migrate(config_data, "0.9.0")

        # Check new fields added
        assert "health_check_interval" in migrated["server"]
        assert migrated["server"]["health_check_interval"] == 60
        assert "use_external_prover" in migrated["verification"]
        assert migrated["verification"]["use_external_prover"] is False
        assert "invalidate_on_proof_change" in migrated["cache"]
        assert migrated["cache"]["invalidate_on_proof_change"] is True

    def test_migration_invalid_version(self):
        """Test migration with invalid version."""
        config_data = {"server": {"host": "localhost"}}

        with pytest.raises(ValidationError):
            LeanAideConfigMigrator.migrate(config_data, "0.1.0")


class TestGlobalConfigInstance:
    """Test global configuration instance."""

    def test_load_and_get_config(self):
        """Test load_leanaide_config and get_leanaide_config."""
        # Clear any existing config
        import leanaide_config
        leanaide_config._leanaide_config = None

        config1 = load_leanaide_config()
        config2 = get_leanaide_config()

        # Should be same instance
        assert config1 is config2

    def test_reload_config(self):
        """Test reload_leanaide_config."""
        import leanaide_config
        leanaide_config._leanaide_config = None

        config1 = load_leanaide_config()
        config2 = reload_leanaide_config(server_port=9999)

        # Should be different instance
        assert config1 is not config2
        assert config2.server.port == 9999

    def test_config_summary(self):
        """Test get_leanaide_config_summary."""
        import leanaide_config
        leanaide_config._leanaide_config = None

        config = load_leanaide_config()
        summary = get_leanaide_config_summary()

        assert isinstance(summary, dict)
        assert "enabled" in summary
        assert "server" in summary
        assert "verification" in summary
        assert "cache" in summary
        assert "workflow" in summary

        # Check no sensitive data in summary
        assert summary["server"]["base_url"] == config.server.get_base_url()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_yaml_file(self):
        """Test loading from empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "leanaide_config.yaml"
            config_path.write_text("")

            loader = LeanAideConfigLoader(config_dir=Path(tmpdir))
            config = loader.load()

            # Should use defaults
            assert config.server.host == "localhost"

    def test_invalid_yaml_file(self):
        """Test loading from invalid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "leanaide_config.yaml"
            config_path.write_text("invalid: yaml: content:")

            # Should log warning and use defaults
            loader = LeanAideConfigLoader(config_dir=Path(tmpdir))
            config = loader.load()

            assert config.server.host == "localhost"

    def test_partial_configuration(self):
        """Test loading partial configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "leanaide_config.yaml"
            config_data = {
                "server": {"host": "partial.example.com"}
                # Other sections missing
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            loader = LeanAideConfigLoader(config_dir=Path(tmpdir))
            config = loader.load()

            # Should use override for server, defaults for rest
            assert config.server.host == "partial.example.com"
            assert config.verification.enable_auto is True  # Default

    def test_list_from_string(self):
        """Test parsing list from environment variable."""
        os.environ["LEANAIDE_VERIFICATION_DOMAINS"] = "mathlib,std,analysis"

        try:
            config = load_leanaide_config(force_reload=True)
            assert "mathlib" in config.verification.domains
            assert "std" in config.verification.domains
            assert "analysis" in config.verification.domains
        finally:
            del os.environ["LEANAIDE_VERIFICATION_DOMAINS"]

    def test_invalid_int_env_var(self):
        """Test invalid integer in environment variable."""
        os.environ["LEANAIDE_SERVER_PORT"] = "invalid"

        try:
            with pytest.raises(ValidationError):
                load_leanaide_config(force_reload=True)
        finally:
            del os.environ["LEANAIDE_SERVER_PORT"]

    def test_out_of_range_int_env_var(self):
        """Test out-of-range integer in environment variable."""
        os.environ["LEANAIDE_SERVER_PORT"] = "100000"

        try:
            with pytest.raises(ValidationError):
                load_leanaide_config(force_reload=True)
        finally:
            del os.environ["LEANAIDE_SERVER_PORT"]


class TestConfigToDict:
    """Test configuration serialization."""

    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = LeanAideConfig(
            server=LeanAideServerConfig(host="test.example.com", port=9090)
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "server" in config_dict
        assert config_dict["server"]["host"] == "test.example.com"
        assert config_dict["server"]["port"] == 9090


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_development_setup(self):
        """Test typical development configuration."""
        # Clear global config first
        import leanaide_config
        leanaide_config._leanaide_config = None

        config = load_leanaide_config(
            environment="development",
            server__host="localhost",
            server__port=8080,
            verification__strict_mode=False,
            logging__level="DEBUG"
        )

        assert config.environment == "development"
        assert config.server.host == "localhost"
        assert config.verification.strict_mode is False
        assert config.logging.level == "DEBUG"

    def test_production_setup(self):
        """Test typical production configuration."""
        # Clear global config first
        import leanaide_config
        leanaide_config._leanaide_config = None

        config = load_leanaide_config(
            environment="production",
            server__host="leanaide.prod.example.com",
            server__use_ssl=True,
            verification__strict_mode=True,
            logging__level="WARNING",
            security__enable_sandboxing=True
        )

        assert config.environment == "production"
        assert config.server.use_ssl is True
        assert config.verification.strict_mode is True
        assert config.security.enable_sandboxing is True

    def test_workflow_integration(self):
        """Test workflow integration configuration."""
        # Clear global config first
        import leanaide_config
        leanaide_config._leanaide_config = None

        config = load_leanaide_config(
            workflow__stage_3c_enabled=True,
            workflow__stage_5_enabled=True,
            workflow__async_verification=True,
            workflow__failure_action="error",
            workflow__inject_proof_hints=True
        )

        assert config.workflow.stage_3c_enabled is True
        assert config.workflow.stage_5_enabled is True
        assert config.workflow.async_verification is True
        assert config.workflow.failure_action == "error"
        assert config.workflow.inject_proof_hints is True

    def test_custom_lean4_paths(self):
        """Test custom Lean 4 installation paths."""
        # Clear global config first
        import leanaide_config
        leanaide_config._leanaide_config = None

        config = load_leanaide_config(
            lean4__lean_path="/usr/local/bin/lean",
            lean4__lake_path="/usr/local/bin/lake",
            lean4__mathlib_path="/opt/lean/mathlib",
            lean4__project_root="/opt/lean/projects"
        )

        assert config.lean4.lean_path == "/usr/local/bin/lean"
        assert config.lean4.lake_path == "/usr/local/bin/lake"
        assert config.lean4.mathlib_path == "/opt/lean/mathlib"
        assert config.lean4.project_root == "/opt/lean/projects"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
