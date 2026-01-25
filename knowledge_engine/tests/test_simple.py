"""
Simple standalone tests that don't require full module imports.

These tests verify basic functionality without triggering complex import chains.
"""

import pytest
import json
import logging

logger = logging.getLogger(__name__)


class TestBasicFunctionality:
    """Tests that don't require knowledge_engine imports."""

    def test_json_logging(self):
        """Test that JSON logging works correctly."""
        log_data = {
            "msg": "Test message",
            "level": "INFO",
            "test_value": 123
        }

        log_json = json.dumps(log_data)
        assert log_json is not None
        assert "Test message" in log_json

        logger.info(log_json)

    def test_import_validation(self):
        """Test that we can detect import failures."""
        import sys

        # Try importing a module that doesn't exist
        try:
            import nonexistent_module
            imported = True
        except ImportError:
            imported = False

        assert imported is False, "Should not be able to import nonexistent module"

        logger.info(json.dumps({
            "msg": "Import validation works",
            "level": "INFO"
        }))

    def test_string_operations(self):
        """Test basic string operations used in text processing."""
        text = "Artificial Intelligence and Machine Learning"

        # Test entity extraction (simple version)
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 3]

        assert len(entities) > 0
        assert "Artificial" in entities
        assert "Intelligence" in entities

        logger.info(json.dumps({
            "msg": "String operations work",
            "entities_found": len(entities),
            "entities": entities,
            "level": "INFO"
        }))

    def test_data_structures(self):
        """Test data structures used in knowledge graphs."""
        # Entity dictionary
        entities = {
            "AI": {"type": "Concept", "confidence": 0.95},
            "ML": {"type": "Field", "confidence": 0.92}
        }

        # Relationship list
        relationships = [
            {"subject": "ML", "predicate": "subset_of", "object": "AI"}
        ]

        assert len(entities) == 2
        assert len(relationships) == 1
        assert relationships[0]["subject"] == "ML"

        logger.info(json.dumps({
            "msg": "Data structures work",
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "level": "INFO"
        }))

    def test_temporal_data(self):
        """Test temporal data handling."""
        from datetime import datetime, timedelta

        # Create timestamps
        now = datetime.now()
        past = now - timedelta(days=7)

        # Verify order
        assert past < now

        # Test ISO format conversion
        now_iso = now.isoformat()
        assert "T" in now_iso  # ISO format includes time separator

        logger.info(json.dumps({
            "msg": "Temporal data handling works",
            "now": now_iso,
            "past": past.isoformat(),
            "level": "INFO"
        }))

    def test_pii_detection(self):
        """Test PII detection patterns."""
        import re

        # Email detection
        text = "Contact us at support@example.com"
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        assert len(emails) == 1
        assert emails[0] == "support@example.com"

        # Phone detection
        phone_text = "Call 555-123-4567"
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        phones = re.findall(phone_pattern, phone_text)

        assert len(phones) == 1
        assert phones[0] == "555-123-4567"

        logger.info(json.dumps({
            "msg": "PII detection works",
            "emails_found": len(emails),
            "phones_found": len(phones),
            "level": "INFO"
        }))

    def test_input_sanitization(self):
        """Test input sanitization."""
        import re

        # SQL injection attempt
        malicious = "'; DROP TABLE users; --"

        # Remove dangerous patterns
        sanitized = re.sub(r"DROP\s+TABLE", "", malicious, flags=re.IGNORECASE)
        sanitized = sanitized.replace(";", "")
        sanitized = sanitized.replace("--", "")

        assert "DROP TABLE" not in sanitized
        assert ";" not in sanitized

        logger.info(json.dumps({
            "msg": "Input sanitization works",
            "original": malicious,
            "sanitized": sanitized,
            "level": "INFO"
        }))

    def test_rate_limiting(self):
        """Test rate limiting logic."""
        import time

        class SimpleRateLimiter:
            def __init__(self, max_requests: int, window_seconds: int):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = []

            def check_limit(self, user_id: str) -> bool:
                now = time.time()

                # Clean old requests
                self.requests = [
                    r for r in self.requests
                    if now - r["timestamp"] < self.window_seconds
                ]

                # Check limit
                user_requests = [r for r in self.requests if r["user_id"] == user_id]
                if len(user_requests) < self.max_requests:
                    self.requests.append({
                        "user_id": user_id,
                        "timestamp": now
                    })
                    return True
                return False

        limiter = SimpleRateLimiter(max_requests=3, window_seconds=1)

        # First 3 requests should succeed
        assert limiter.check_limit("user1") is True
        assert limiter.check_limit("user1") is True
        assert limiter.check_limit("user1") is True

        # 4th should fail
        assert limiter.check_limit("user1") is False

        # Different user should succeed
        assert limiter.check_limit("user2") is True

        logger.info(json.dumps({
            "msg": "Rate limiting works",
            "level": "INFO"
        }))

    def test_deduplication(self):
        """Test deduplication logic."""
        # List with duplicates
        items = ["AI", "ML", "AI", "DL", "ML", "AI"]

        # Deduplicate
        unique_items = list(set(items))

        assert len(unique_items) == 3  # AI, ML, DL
        assert set(unique_items) == {"AI", "ML", "DL"}

        # Order-preserving deduplication
        seen = set()
        ordered_unique = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered_unique.append(item)

        assert len(ordered_unique) == 3
        assert ordered_unique == ["AI", "ML", "DL"]

        logger.info(json.dumps({
            "msg": "Deduplication works",
            "original_count": len(items),
            "unique_count": len(unique_items),
            "level": "INFO"
        }))

    def test_quality_metrics(self):
        """Test data quality metrics calculation."""
        entities = [
            {"name": "AI", "type": "Concept", "confidence": 0.95},
            {"name": "ML", "type": "Field", "confidence": 0.88},
            {"name": "DL", "type": None, "confidence": 0.75},  # Missing type
        ]

        # Completeness: entities with all required fields
        required_fields = ["name", "type"]
        complete = sum(
            1 for e in entities
            if all(e.get(f) for f in required_fields)
        )
        completeness = complete / len(entities)

        assert completeness == 2/3  # 2 out of 3 complete

        # Average confidence
        avg_confidence = sum(e["confidence"] for e in entities) / len(entities)

        assert avg_confidence == (0.95 + 0.88 + 0.75) / 3

        logger.info(json.dumps({
            "msg": "Quality metrics calculated",
            "completeness": completeness,
            "avg_confidence": avg_confidence,
            "level": "INFO"
        }))

    def test_performance_tracking(self):
        """Test performance tracking."""
        import time

        class PerformanceTracker:
            def __init__(self):
                self.operations = []

            def record(self, operation: str, duration_ms: float):
                self.operations.append({
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "timestamp": time.time()
                })

            def get_stats(self):
                if not self.operations:
                    return {"count": 0, "avg_duration": 0}

                durations = [op["duration_ms"] for op in self.operations]
                return {
                    "count": len(self.operations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations)
                }

        tracker = PerformanceTracker()

        # Record some operations
        tracker.record("op1", 50)
        tracker.record("op2", 75)
        tracker.record("op3", 100)

        stats = tracker.get_stats()

        assert stats["count"] == 3
        assert stats["avg_duration"] == 75
        assert stats["min_duration"] == 50
        assert stats["max_duration"] == 100

        logger.info(json.dumps({
            "msg": "Performance tracking works",
            "stats": stats,
            "level": "INFO"
        }))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
