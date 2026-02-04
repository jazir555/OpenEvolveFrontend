"""
Test script for configuration validation.

This script demonstrates the configuration validation system and can be used
to test environment variable configuration.
"""

import os
import sys

# Ensure proper module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_missing_required():
    """Test validation with missing required variables."""
    print("\n" + "="*80)
    print("TEST 1: Missing Required Variables")
    print("="*80)

    # Temporarily unset required variables
    saved_env = {}
    required_vars = ["GRAPHITI_URI", "GRAPHITI_USER", "GRAPHITI_PASSWORD", "OPENAI_API_KEY"]

    for var in required_vars:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    try:
        from knowledge_engine.config_validation import validate_config, ConfigError

        try:
            result = validate_config(silent=True)
            print(f"[FAIL] Test FAILED: Should have raised ConfigError")
            return False
        except ConfigError as e:
            print(f"[PASS] Test PASSED: ConfigError raised as expected")
            print(f"  Error message: {str(e)[:100]}...")
            return True
    finally:
        # Restore environment
        for var, value in saved_env.items():
            os.environ[var] = value


def test_with_valid_config():
    """Test validation with valid configuration."""
    print("\n" + "="*80)
    print("TEST 2: Valid Configuration")
    print("="*80)

    # Set minimal required configuration
    test_config = {
        "GRAPHITI_URI": "bolt://localhost:7687",
        "GRAPHITI_USER": "neo4j",
        "GRAPHITI_PASSWORD": "test_password",
        "OPENAI_API_KEY": "sk-test-key-1234567890",
    }

    # Save and set test config
    saved_env = {}
    for var, value in test_config.items():
        if var in os.environ:
            saved_env[var] = os.environ[var]
        os.environ[var] = value

    try:
        from knowledge_engine.config_validation import validate_config, ConfigError

        try:
            result = validate_config(silent=True)
            if result.is_valid:
                print(f"[PASS] Test PASSED: Configuration is valid")
                print(f"  Configured variables: {len(result.present_optional)}")
                print(f"  Warnings: {len(result.warnings)}")
                return True
            else:
                print(f"[FAIL] Configuration should be valid")
                print(f"  Errors: {result.errors}")
                return False
        except ConfigError as e:
            print(f"[FAIL] ConfigError raised unexpectedly")
            print(f"  Error: {e}")
            return False
    finally:
        # Restore environment
        for var in test_config:
            if var in saved_env:
                os.environ[var] = saved_env[var]
            else:
                del os.environ[var]


def test_cloud_storage_validation():
    """Test cloud storage credential validation."""
    print("\n" + "="*80)
    print("TEST 3: Cloud Storage Validation")
    print("="*80)

    from knowledge_engine.cloud_storage_backends import (
        S3Credentials, GCSCredentials, AzureCredentials, SFTPCredentials
    )

    # Test S3 credentials with missing values
    print("\n  Testing S3 credentials (missing required)...")
    try:
        S3Credentials.from_env()
        print("  [FAIL] Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  [OK] Correctly raised ValueError: {str(e)[:80]}...")

    # Test S3 credentials with values
    print("\n  Testing S3 credentials (with values)...")
    os.environ["AWS_ACCESS_KEY_ID"] = "test_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret"

    try:
        creds = S3Credentials.from_env()
        print(f"  [OK] Successfully created credentials")
        print(f"    Access Key ID: {creds.access_key_id[:4]}...")
        print(f"    Region: {creds.region}")
    except Exception as e:
        print(f"  [FAIL] Failed to create credentials: {e}")
        return False
    finally:
        # Clean up
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            if var in os.environ:
                del os.environ[var]

    print("\n  [OK] Test PASSED")
    return True


def test_template_generation():
    """Test configuration template generation."""
    print("\n" + "="*80)
    print("TEST 4: Template Generation")
    print("="*80)

    from knowledge_engine.config_validation import get_config_template

    try:
        template = get_config_template()

        # Check template contains expected content
        # Note: Template doesn't include all config vars, just representative ones
        required_strings = [
            "# Knowledge Engine Configuration Template",
            "OPENAI_API_KEY",
            "Optional",
        ]

        for string in required_strings:
            if string not in template:
                print(f"[FAIL] Template missing '{string}'")
                return False

        print(f"[OK] Test PASSED: Template generated successfully")
        print(f"  Template length: {len(template)} characters")
        print(f"  Contains all required sections")

        return True
    except Exception as e:
        print(f"[FAIL] Test FAILED: {e}")
        return False


def main():
    """Run all configuration validation tests."""
    print("\n" + "="*80)
    print("KNOWLEDGE ENGINE CONFIGURATION VALIDATION TESTS")
    print("="*80)

    tests = [
        test_missing_required,
        test_with_valid_config,
        test_cloud_storage_validation,
        test_template_generation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n[PASS] ALL TESTS PASSED")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
