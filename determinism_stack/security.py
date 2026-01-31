"""Security and PII handling utilities."""

from __future__ import annotations

import re
from typing import Optional

from .utils import optional_import


class SecurityLayer:
    def __init__(self):
        presidio_analyzer = optional_import("presidio_analyzer")
        presidio_anonymizer = optional_import("presidio_anonymizer")
        self._analyzer = presidio_analyzer.AnalyzerEngine() if presidio_analyzer else None
        self._anonymizer = presidio_anonymizer.AnonymizerEngine() if presidio_anonymizer else None

    def sanitize_input(self, text: str) -> str:
        if not text:
            return text
        if self._analyzer and self._anonymizer:
            results = self._analyzer.analyze(
                text=text,
                entities=["EMAIL", "SSN", "CREDIT_CARD", "PHONE_NUMBER"],
                language="en",
            )
            return self._anonymizer.anonymize(text=text, analyzer_results=results).text
        return self._simple_mask(text)

    def _simple_mask(self, text: str) -> str:
        text = re.sub(r"[\\w\\.-]+@[\\w\\.-]+", "[EMAIL]", text)
        text = re.sub(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN]", text)
        text = re.sub(r"\\b\\d{13,16}\\b", "[CARD]", text)
        text = re.sub(r"\\b\\d{3}[-\\.\\s]?\\d{3}[-\\.\\s]?\\d{4}\\b", "[PHONE]", text)
        return text

