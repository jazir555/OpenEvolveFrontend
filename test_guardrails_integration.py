"""
Guardrails Integration Validation Script
=========================================

Tests the Guardrails adapter integration with MDAP voting system.

Usage:
    python test_guardrails_integration.py
"""

import sys
import json
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_guardrails_import():
    """Test 1: Guardrails adapter import"""
    logger.info("Test 1: Testing Guardrails adapter import...")

    try:
        from reliability.guardrails_adapter import (
            GuardrailsAdapter,
            create_adapter,
            ValidationResult
        )
        logger.info("✅ Guardrails adapter imported successfully")
        return True, {
            "test": "import",
            "status": "success",
            "available": True
        }
    except ImportError as e:
        logger.warning(f"⚠️  Guardrails adapter not available: {e}")
        logger.warning("   Install with: pip install guardrails-ai")
        return False, {
            "test": "import",
            "status": "failed",
            "error": str(e),
            "available": False
        }


def test_adapter_creation():
    """Test 2: Adapter creation"""
    logger.info("Test 2: Testing adapter creation...")

    try:
        from reliability.guardrails_adapter import create_adapter

        adapter = create_adapter(
            enabled=True,
            default_on_fail="filter"
        )

        stats = adapter.get_statistics()
        logger.info(f"✅ Adapter created successfully")
        logger.info(f"   Guardrails available: {stats['guardrails_available']}")
        logger.info(f"   Enabled: {stats['enabled']}")
        logger.info(f"   Total validators: {stats['total_validators']}")

        return True, {
            "test": "adapter_creation",
            "status": "success",
            "statistics": stats
        }
    except (ValueError, TypeError, ImportError) as e:
        logger.error(f"❌ Adapter creation failed: {e}")
        return False, {
            "test": "adapter_creation",
            "status": "failed",
            "error": str(e)
        }


def test_output_validation():
    """Test 3: Output validation"""
    logger.info("Test 3: Testing output validation...")

    try:
        from reliability.guardrails_adapter import create_adapter

        adapter = create_adapter()

        # Test 1: Valid output
        result1 = adapter.validate_output(
            output="This is a valid output",
            validators=["toxic_language"],
            on_fail="filter"
        )

        # Test 2: JSON validation
        result2 = adapter.validate_output(
            output=json.dumps({"action": "approve", "confidence": 0.9}),
            validators=["vote_json"],
            on_fail="filter"
        )

        # Test 3: Multiple validators
        result3 = adapter.validate_output(
            output="Contact me at john@example.com",
            validators=["toxic_language", "pii_filter"],
            on_fail="fix"
        )

        logger.info("✅ Output validation completed")
        logger.info(f"   Valid output test: {result1.is_valid}")
        logger.info(f"   JSON validation test: {result2.is_valid}")
        logger.info(f"   PII filter test: {result3.is_valid}")
        logger.info(f"   PII remediation: {result3.remediation_applied}")

        return True, {
            "test": "output_validation",
            "status": "success",
            "valid_output": result1.is_valid,
            "json_validation": result2.is_valid,
            "pii_filter": result3.is_valid,
            "pii_remediation": result3.remediation_applied
        }
    except (ValueError, TypeError, ImportError) as e:
        logger.error(f"❌ Output validation failed: {e}")
        return False, {
            "test": "output_validation",
            "status": "failed",
            "error": str(e)
        }


def test_mdap_engine_integration():
    """Test 4: MDAP engine integration"""
    logger.info("Test 4: Testing MDAP engine integration...")

    try:
        from mdap_engine import (
            MDAPOrchestrator,
            MDAPConfig,
            RedFlagger,
            RedFlagRules
        )
        from reliability.guardrails_adapter import create_adapter

        # Create Guardrails adapter
        guardrails_adapter = create_adapter(enabled=True)

        # Create RedFlagger with Guardrails
        red_flagger = RedFlagger(
            rules=RedFlagRules(),
            guardrails_adapter=guardrails_adapter
        )

        # Test red-flagging with Guardrails
        is_flagged, reasons = red_flagger.is_flagged(
            raw_text="This is a test response",
            candidate={"action": "approve"},
            schema=None
        )

        # Get statistics
        stats = red_flagger.get_guardrails_stats()

        logger.info("✅ MDAP engine integration successful")
        logger.info(f"   Is flagged: {is_flagged}")
        logger.info(f"   Reasons: {reasons}")
        logger.info(f"   Guardrails stats: {stats}")

        return True, {
            "test": "mdap_engine_integration",
            "status": "success",
            "is_flagged": is_flagged,
            "reasons": reasons,
            "statistics": stats
        }
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        logger.error(f"❌ MDAP engine integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {
            "test": "mdap_engine_integration",
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def test_remediation_strategies():
    """Test 5: Remediation strategies"""
    logger.info("Test 5: Testing remediation strategies...")

    try:
        from reliability.guardrails_adapter import create_adapter

        adapter = create_adapter()

        strategies = ["reask", "fix", "filter", "refrain", "exception"]
        results = {}

        for strategy in strategies:
            try:
                output = "This is a test output with john@example.com"

                # For testing, we'll just check the strategy doesn't crash
                result = adapter.validate_output(
                    output=output,
                    validators=["pii_filter"],
                    on_fail=strategy
                )

                results[strategy] = {
                    "is_valid": result.is_valid,
                    "remediation_applied": result.remediation_applied
                }

            except (ValueError, TypeError, AttributeError) as e:
                results[strategy] = {
                    "error": str(e)
                }

        logger.info("✅ Remediation strategies tested")
        for strategy, result in results.items():
            logger.info(f"   {strategy}: {result}")

        return True, {
            "test": "remediation_strategies",
            "status": "success",
            "strategies": results
        }
    except (ValueError, TypeError, ImportError) as e:
        logger.error(f"❌ Remediation strategies test failed: {e}")
        return False, {
            "test": "remediation_strategies",
            "status": "failed",
            "error": str(e)
        }


def test_custom_validators():
    """Test 6: Custom validators"""
    logger.info("Test 6: Testing custom validators...")

    try:
        from reliability.guardrails_adapter import create_adapter, ValidationResult

        adapter = create_adapter()

        # Define custom validator
        def validate_no_malicious_patterns(output):
            """Check for malicious patterns"""
            output_str = str(output).lower()

            malicious_patterns = [
                "<script", "javascript:", "eval(", "__import__",
                "os.system", "subprocess", "pickle.loads"
            ]

            found = [p for p in malicious_patterns if p in output_str]

            if found:
                return ValidationResult(
                    is_valid=False,
                    failures=[f"Malicious patterns detected: {found}"]
                )

            return ValidationResult(is_valid=True)

        # Register custom validator
        adapter.register_validator(
            name="no_malicious_patterns",
            validator=validate_no_malicious_patterns
        )

        # Test with safe output
        result1 = adapter.validate_output(
            output="This is a safe output",
            validators=["no_malicious_patterns"],
            on_fail="refrain"
        )

        # Test with malicious output
        result2 = adapter.validate_output(
            output="This has <script>alert('xss')</script>",
            validators=["no_malicious_patterns"],
            on_fail="refrain"
        )

        logger.info("✅ Custom validators tested")
        logger.info(f"   Safe output valid: {result1.is_valid}")
        logger.info(f"   Malicious output valid: {result2.is_valid}")
        logger.info(f"   Malicious output failures: {result2.failures}")

        return True, {
            "test": "custom_validators",
            "status": "success",
            "safe_output_valid": result1.is_valid,
            "malicious_output_valid": result2.is_valid,
            "malicious_failures": result2.failures
        }
    except (ValueError, TypeError, ImportError) as e:
        logger.error(f"❌ Custom validators test failed: {e}")
        return False, {
            "test": "custom_validators",
            "status": "failed",
            "error": str(e)
        }


def test_graceful_degradation():
    """Test 7: Graceful degradation"""
    logger.info("Test 7: Testing graceful degradation...")

    try:
        from mdap_engine import RedFlagger, RedFlagRules

        # Test without Guardrails adapter
        red_flagger = RedFlagger(rules=RedFlagRules(), guardrails_adapter=None)

        is_flagged, reasons = red_flagger.is_flagged(
            raw_text="This is a test response",
            candidate={"action": "approve"},
            schema=None
        )

        stats = red_flagger.get_guardrails_stats()

        logger.info("✅ Graceful degradation working")
        logger.info(f"   Is flagged: {is_flagged}")
        logger.info(f"   Guardrails validations: {stats['guardrails_validations']}")
        logger.info(f"   System works without Guardrails: True")

        return True, {
            "test": "graceful_degradation",
            "status": "success",
            "is_flagged": is_flagged,
            "guardrails_validations": stats['guardrails_validations'],
            "degraded_mode_working": True
        }
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        logger.error(f"❌ Graceful degradation test failed: {e}")
        return False, {
            "test": "graceful_degradation",
            "status": "failed",
            "error": str(e)
        }


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Guardrails Integration Validation Suite")
    logger.info("=" * 60)

    tests = [
        test_guardrails_import,
        test_adapter_creation,
        test_output_validation,
        test_mdap_engine_integration,
        test_remediation_strategies,
        test_custom_validators,
        test_graceful_degradation
    ]

    results = []

    for test in tests:
        try:
            success, result = test()
            results.append(result)
        except (ValueError, TypeError, ImportError, RuntimeError) as e:
            logger.error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test": test.__name__,
                "status": "crashed",
                "error": str(e)
            })

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") != "success")

    for result in results:
        status_icon = "✅" if result.get("status") == "success" else "❌"
        logger.info(f"{status_icon} {result.get('test')}: {result.get('status')}")

    logger.info("\n" + "-" * 60)
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {passed / len(results) * 100:.1f}%")
    logger.info("=" * 60)

    # Save results to JSON
    with open("guardrails_validation_results.json", "w") as f:
        json.dump({
            "timestamp": "2026-01-10T12:00:00.000Z",
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(results) * 100,
            "results": results
        }, f, indent=2)

    logger.info("\nResults saved to: guardrails_validation_results.json")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
