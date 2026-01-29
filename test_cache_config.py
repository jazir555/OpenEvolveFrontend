"""
Test cache configuration system.

Validates that:
- Configuration can be loaded from environment
- Configuration validation works correctly
- Invalid configurations are rejected
"""
import os
import sys

# Run from package root
sys.path.insert(0, '.')

from bubblelabs_nodes.gauntlet_config import CacheConfig, CacheType, GauntletConfig


def test_default_config():
    """Test default configuration values."""
    print("\n[TEST 1] Default configuration")

    config = CacheConfig()

    assert config.enabled == True, "Cache should be enabled by default"
    assert config.cache_type == CacheType.MEMORY, "Default cache type should be MEMORY"
    assert config.ttl_seconds == 3600, "Default TTL should be 3600 seconds"
    assert config.max_size == 1000, "Default max size should be 1000"
    assert config.redis_url is None, "Redis URL should be None by default"

    print("  [PASS] All defaults correct")


def test_config_validation():
    """Test configuration validation."""
    print("\n[TEST 2] Configuration validation")

    # Valid config
    config = CacheConfig(
        enabled=True,
        cache_type=CacheType.MEMORY,
        ttl_seconds=7200,
        max_size=5000
    )

    valid, errors = config.validate()
    assert valid, f"Valid config should pass validation: {errors}"
    print("  [PASS] Valid configuration accepted")

    # Invalid: negative TTL
    config_invalid = CacheConfig(ttl_seconds=-1)
    valid, errors = config_invalid.validate()
    assert not valid, "Negative TTL should be rejected"
    assert "TTL must be non-negative" in errors[0]
    print("  [PASS] Negative TTL rejected")

    # Invalid: negative max_size
    config_invalid = CacheConfig(max_size=-100)
    valid, errors = config_invalid.validate()
    assert not valid, "Negative max_size should be rejected"
    assert "Max size must be non-negative" in errors[0]
    print("  [PASS] Negative max_size rejected")

    # Invalid: Redis without URL
    config_invalid = CacheConfig(cache_type=CacheType.REDIS, redis_url=None)
    valid, errors = config_invalid.validate()
    assert not valid, "Redis cache without URL should be rejected"
    assert "Redis URL required" in errors[0]
    print("  [PASS] Redis without URL rejected")


def test_env_config_loading():
    """Test loading configuration from environment."""
    print("\n[TEST 3] Environment variable loading")

    # Set environment variables
    os.environ['CACHE_ENABLED'] = 'false'
    os.environ['CACHE_TYPE'] = 'none'
    os.environ['CACHE_TTL_SECONDS'] = '7200'
    os.environ['CACHE_MAX_SIZE'] = '5000'
    os.environ['CACHE_REDIS_URL'] = 'redis://localhost:6379'

    try:
        # Load from environment
        config = GauntletConfig.from_env()

        assert config.cache.enabled == False, "CACHE_ENABLED should be false"
        assert config.cache.cache_type == CacheType.NONE, "CACHE_TYPE should be none"
        assert config.cache.ttl_seconds == 7200, "CACHE_TTL_SECONDS should be 7200"
        assert config.cache.max_size == 5000, "CACHE_MAX_SIZE should be 5000"
        assert config.cache.redis_url == 'redis://localhost:6379', "CACHE_REDIS_URL should be set"

        print("  [PASS] Environment variables loaded correctly")

    finally:
        # Clean up
        del os.environ['CACHE_ENABLED']
        del os.environ['CACHE_TYPE']
        del os.environ['CACHE_TTL_SECONDS']
        del os.environ['CACHE_MAX_SIZE']
        del os.environ['CACHE_REDIS_URL']


def test_full_config_validation():
    """Test full GauntletConfig validation."""
    print("\n[TEST 4] Full configuration validation")

    # Valid config
    config = GauntletConfig.from_env()
    valid, errors = config.validate()

    assert valid, f"Default config should be valid: {errors}"
    print("  [PASS] Default GauntletConfig is valid")


def main():
    print("=" * 60)
    print("CACHE CONFIGURATION TESTS")
    print("=" * 60)

    test_default_config()
    test_config_validation()
    test_env_config_loading()
    test_full_config_validation()

    print("\n" + "=" * 60)
    print("[SUCCESS] All configuration tests passed!")
    print("=" * 60)

    print("\n[CONFIGURATION OPTIONS]")
    print("  CACHE_ENABLED: Enable/disable caching (default: true)")
    print("  CACHE_TYPE: Cache backend - memory, redis, none (default: memory)")
    print("  CACHE_TTL_SECONDS: TTL in seconds (default: 3600)")
    print("  CACHE_MAX_SIZE: Maximum cache size (default: 1000)")
    print("  CACHE_REDIS_URL: Redis connection URL (required if type=redis)")


if __name__ == '__main__':
    main()
