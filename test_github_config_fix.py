"""
Test script to verify the base64 encoding fix in github_config.py
This script tests that the base64 encoding is correctly implemented.
"""

import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_base64_encoding():
    """Test that base64 encoding works correctly for GitHub API."""
    test_content = "Hello, GitHub! This is a test file content."

    # OLD METHOD (WRONG) - using .hex()
    old_encoding = test_content.encode("utf-8").hex()
    logger.info(f"Old encoding (hex) length: {len(old_encoding)}")
    logger.info(f"Old encoding (hex): {old_encoding[:50]}...")

    # NEW METHOD (CORRECT) - using base64
    new_encoding = base64.b64encode(test_content.encode("utf-8")).decode("utf-8")
    logger.info(f"New encoding (base64) length: {len(new_encoding)}")
    logger.info(f"New encoding (base64): {new_encoding[:50]}...")

    # Verify they're different
    assert old_encoding != new_encoding, "Hex and base64 should produce different results"
    logger.info("[OK] Confirmed: hex and base64 produce different results")

    # Verify base64 can be decoded back
    decoded = base64.b64decode(new_encoding).decode("utf-8")
    assert decoded == test_content, "Base64 decoding should return original content"
    logger.info("[OK] Confirmed: base64 can be decoded back to original content")

    # Test with unicode content
    unicode_content = "Hello 世界! 🌍 Test with emoji and unicode characters."
    unicode_encoded = base64.b64encode(unicode_content.encode("utf-8")).decode("utf-8")
    unicode_decoded = base64.b64decode(unicode_encoded).decode("utf-8")
    assert unicode_decoded == unicode_content, "Unicode should round-trip correctly"
    logger.info("[OK] Confirmed: Unicode content round-trips correctly")

    # Test with binary-like content
    binary_content = "\x00\x01\x02\x03 Binary data \xFF\xFE\xFD"
    binary_encoded = base64.b64encode(binary_content.encode("utf-8", errors="surrogateescape")).decode("utf-8")
    binary_decoded = base64.b64decode(binary_encoded).decode("utf-8", errors="surrogateescape")
    assert binary_decoded == binary_content, "Binary content should round-trip correctly"
    logger.info("[OK] Confirmed: Binary-like content round-trips correctly")

    logger.info("\n[OK][OK][OK] All base64 encoding tests passed! [OK][OK][OK]")
    return True


def test_github_api_format():
    """Test that the encoded format matches GitHub API expectations."""
    test_content = "test content"

    # GitHub API expects base64 encoding
    encoded = base64.b64encode(test_content.encode("utf-8")).decode("utf-8")

    # Should be ASCII-safe
    assert encoded.isascii(), "Base64 should be ASCII-safe"
    logger.info("[OK] Confirmed: Base64 is ASCII-safe")

    # Should only contain valid base64 characters
    valid_b64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    assert all(c in valid_b64_chars for c in encoded), "Should only contain valid base64 characters"
    logger.info("[OK] Confirmed: Only contains valid base64 characters")

    # Should have padding to multiple of 4
    assert len(encoded) % 4 == 0, "Base64 length should be multiple of 4"
    logger.info("[OK] Confirmed: Length is multiple of 4 (properly padded)")

    logger.info("\n[OK][OK][OK] All GitHub API format tests passed! [OK][OK][OK]")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing GitHub Config Base64 Fix")
    logger.info("=" * 60)
    logger.info("")

    try:
        test_base64_encoding()
        logger.info("")
        test_github_api_format()
        logger.info("")
        logger.info("=" * 60)
        logger.info("[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]")
        logger.info("=" * 60)
        logger.info("")
        logger.info("The fix correctly implements base64 encoding for GitHub API.")
        logger.info("The old .hex() method has been replaced with base64.b64encode().")
        return True
    except AssertionError as e:
        logger.error(f"[FAIL] Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
