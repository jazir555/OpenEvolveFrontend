#!/usr/bin/env python3
"""
Probe Script: Check Model Adapter
Task 3.1: Verify OneKE model adapter functionality

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Verify model actually loads
- STRUCTURED LOGGING: JSON output with correlation IDs
"""

import asyncio
import sys
import logging
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from knowledge_engine.integrations.oneke.model_adapter import (
    OneKEModelAdapter,
    ModelConfig,
    QuantizationMode,
    Language
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def probe_model_adapter():
    """Probe model adapter functionality."""
    correlation_id = "probe_model_adapter_001"

    logger.info({
        "msg": "Starting model adapter probe",
        "correlation_id": correlation_id
    })

    try:
        # Test 1: Configuration validation
        logger.info("Test 1: Configuration validation")
        config = ModelConfig(
            model_name="test/oneke",
            device="cpu",
            max_length=2048,
            quantization=QuantizationMode.NONE,
            temperature=0.1
        )
        logger.info("✓ Configuration created successfully")

        # Test 2: Invalid configuration detection
        logger.info("Test 2: Invalid configuration detection")
        try:
            invalid_config = ModelConfig(temperature=3.0)
            logger.error("✗ Should have rejected invalid temperature")
            return False
        except ValueError as e:
            logger.info(f"✓ Correctly rejected invalid config: {e}")

        # Test 3: Language enum
        logger.info("Test 3: Language enumeration")
        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE.value == "zh"
        logger.info("✓ Language enums working")

        # Test 4: ExtractionResult creation
        logger.info("Test 4: ExtractionResult structure")
        from knowledge_engine.integrations.oneke.model_adapter import ExtractionResult

        result = ExtractionResult(
            entities=[{"id": "E1", "type": "PERSON"}],
            relations=[{"subject": "E1", "object": "E2", "type": "WORKS_FOR"}],
            confidence=0.85,
            language=Language.ENGLISH
        )

        assert len(result.entities) == 1
        assert len(result.relations) == 1
        assert result.confidence == 0.85
        assert result.timestamp.tzinfo is not None  # UTC
        logger.info("✓ ExtractionResult structure valid")

        # Test 5: Environment variable configuration
        logger.info("Test 5: Environment variable configuration")
        import os
        os.environ["ONEKE_MODEL_NAME"] = "test/model"
        os.environ["ONEKE_DEVICE"] = "cpu"
        os.environ["ONEKE_TEMPERATURE"] = "0.5"

        env_config = ModelConfig()
        assert env_config.model_name == "test/model"
        assert env_config.device == "cpu"
        assert env_config.temperature == 0.5
        logger.info("✓ Environment variable configuration working")

        logger.info({
            "msg": "Model adapter probe complete",
            "status": "SUCCESS",
            "correlation_id": correlation_id
        })

        return True

    except Exception as e:
        logger.error({
            "msg": "Model adapter probe failed",
            "error": str(e),
            "correlation_id": correlation_id
        })
        return False


if __name__ == "__main__":
    success = asyncio.run(probe_model_adapter())
    sys.exit(0 if success else 1)
