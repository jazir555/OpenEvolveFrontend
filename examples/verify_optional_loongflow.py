"""
Optional LoongFlow Verification Script

This script verifies that all optional LoongFlow features are working correctly.

Author: AI Architecture Team
Date: 2026-01-30
"""

import sys
sys.path.insert(0, 'openevolve')

from unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig
)


def test_feature(name: str, test_func):
    """Run a test and report results"""
    try:
        test_func()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"   Error: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("OPTIONAL LOONGFLOW VERIFICATION")
    print("="*80 + "\n")

    results = []

    # Test 1: Default Configuration
    def test_default_config():
        config = UnifiedEvolutionConfig()
        assert config.enable_loongflow is True
        assert config.loongflow_fallback_enabled is True
        assert config.require_loongflow is False

    results.append(test_feature("Default configuration has correct values", test_default_config))

    # Test 2: Disable LoongFlow
    def test_disable_loongflow():
        config = UnifiedEvolutionConfig(enable_loongflow=False)
        assert config.enable_loongflow is False
        assert config.should_use_loongflow() is False

    results.append(test_feature("Can disable LoongFlow", test_disable_loongflow))

    # Test 3: OpenEvolve Only
    def test_openevolve_only():
        config = UnifiedEvolutionConfig.openevolve_only()
        assert config.enable_loongflow is False
        assert config.loongflow_fallback_enabled is False
        assert config.require_loongflow is False

    results.append(test_feature("OpenEvolve-only configuration works", test_openevolve_only))

    # Test 4: LoongFlow Required
    def test_loongflow_required():
        config = UnifiedEvolutionConfig.loongflow_required()
        assert config.enable_loongflow is True
        assert config.require_loongflow is True
        assert config.loongflow_fallback_enabled is False

    results.append(test_feature("LoongFlow-required configuration works", test_loongflow_required))

    # Test 5: Validation - Contradictory Settings
    def test_validation_error():
        try:
            config = UnifiedEvolutionConfig(
                enable_loongflow=False,
                require_loongflow=True
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass  # Expected

    results.append(test_feature("Validation catches contradictory settings", test_validation_error))

    # Test 6: is_loongflow_enabled
    def test_is_enabled():
        config1 = UnifiedEvolutionConfig(enable_loongflow=True)
        config2 = UnifiedEvolutionConfig(enable_loongflow=False)
        assert config1.is_loongflow_enabled() is True
        assert config2.is_loongflow_enabled() is False

    results.append(test_feature("is_loongflow_enabled() works correctly", test_is_enabled))

    # Test 7: should_use_loongflow when disabled
    def test_should_use_disabled():
        config = UnifiedEvolutionConfig(enable_loongflow=False)
        assert config.should_use_loongflow() is False

    results.append(test_feature("should_use_loongflow() returns False when disabled", test_should_use_disabled))

    # Test 8: should_use_loongflow when enabled
    def test_should_use_enabled():
        config = UnifiedEvolutionConfig(enable_loongflow=True)
        result = config.should_use_loongflow()
        # Should return bool (True if available, False if not)
        assert isinstance(result, bool)

    results.append(test_feature("should_use_loongflow() returns bool when enabled", test_should_use_enabled))

    # Test 9: Availability check
    def test_availability_check():
        config = UnifiedEvolutionConfig()
        result = config._check_loongflow_availability()
        assert isinstance(result, bool)

    results.append(test_feature("Availability check returns bool", test_availability_check))

    # Test 10: QD mode with LoongFlow disabled
    def test_qd_mode():
        config = UnifiedEvolutionConfig.openevolve_only(
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(enabled=True)
        )
        assert config.evolution_mode == EvolutionMode.QD
        assert config.qd.enabled is True
        assert config.enable_loongflow is False

    results.append(test_feature("QD mode works without LoongFlow", test_qd_mode))

    # Test 11: PES mode with LoongFlow required
    def test_pes_mode():
        config = UnifiedEvolutionConfig.loongflow_required(
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True)
        )
        assert config.evolution_mode == EvolutionMode.PES
        assert config.pes.enabled is True
        assert config.require_loongflow is True

    results.append(test_feature("PES mode requires LoongFlow", test_pes_mode))

    # Test 12: Domain configuration
    def test_domain_config():
        config = UnifiedEvolutionConfig.openevolve_only(domain=DomainType.FINANCE)
        assert config.domain == DomainType.FINANCE
        assert config.enable_loongflow is False

    results.append(test_feature("Domain configuration works", test_domain_config))

    # Test 13: Custom parameters
    def test_custom_params():
        config = UnifiedEvolutionConfig.openevolve_only(
            max_iterations=5000,
            random_seed=123
        )
        assert config.max_iterations == 5000
        assert config.random_seed == 123

    results.append(test_feature("Custom parameters work", test_custom_params))

    # Test 14: Fallback enabled
    def test_fallback_enabled():
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True
        )
        assert config.loongflow_fallback_enabled is True

    results.append(test_feature("Fallback can be enabled", test_fallback_enabled))

    # Test 15: Fallback disabled
    def test_fallback_disabled():
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=False
        )
        assert config.loongflow_fallback_enabled is False

    results.append(test_feature("Fallback can be disabled", test_fallback_disabled))

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n*** All tests passed! Optional LoongFlow is working correctly. ***")
        return 0
    else:
        print(f"\n*** {failed} test(s) failed. Please review the errors above. ***")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
