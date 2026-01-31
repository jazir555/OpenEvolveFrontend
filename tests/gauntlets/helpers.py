"""
Helper functions for gauntlet testing

Provides utilities for running tests, generating data, and measuring metrics.
"""

import time
import asyncio
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, UTC
import json
from pathlib import Path


class GauntletTestHelpers:
    """Helper utilities for gauntlet testing"""

    @staticmethod
    def measure_time(func: Callable) -> tuple[Any, float]:
        """Measure execution time of a function"""
        start = time.time()
        result = func()
        elapsed = time.time() - start
        return (result, elapsed)

    @staticmethod
    async def measure_time_async(func: Callable) -> tuple[Any, float]:
        """Measure execution time of an async function"""
        start = time.time()
        result = await func()
        elapsed = time.time() - start
        return (result, elapsed)

    @staticmethod
    def calculate_metrics(
        true_positives: int,
        false_positives: int,
        true_negatives: int,
        false_negatives: int
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy
        }


if __name__ == '__main__':
    print("Gauntlet test helpers module")
