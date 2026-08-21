"""
Input Parser Module

Real, dependency-light input parsing with schema-driven format validation.

Public names preserved: InputParserConfig, InputParser, create_input_parser.
New: parse_json, parse_key_value, validate_against_schema.
"""
from __future__ import annotations


import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InputParserConfig:
    """Configuration for input parser"""
    encoding: str = "utf-8"
    max_size: int = 1000000


class InputParser:
    """Input Parser class"""

    def __init__(self, config: Optional[InputParserConfig] = None):
        self.config = config or InputParserConfig()
        logger.info("Input Parser initialized")

    def parse(self, raw_input: str) -> Dict[str, Any]:
        """
        Parse raw input. Tries JSON first, then falls back to key=value lines.
        Returns a structured result with a 'data' payload and metadata.
        """
        if not isinstance(raw_input, str):
            raw_input = str(raw_input)
        if len(raw_input.encode(self.config.encoding, errors="ignore")) > self.config.max_size:
            return {"parsed": False, "error": "input exceeds max_size", "data": {}}

        try:
            data = self.parse_json(raw_input)
        except ValueError:
            data = self.parse_key_value(raw_input)

        return {"parsed": True, "data": data, "format": "json" if data is not None and self._looks_json(raw_input) else "kv"}

    def _looks_json(self, raw: str) -> bool:
        s = raw.strip()
        return s[:1] in ("{", "[") and s[-1:] in ("}", "]")

    def parse_json(self, raw_input: str) -> Any:
        """Parse a JSON document. Raises ValueError on failure."""
        return json.loads(raw_input)

    def parse_key_value(self, raw_input: str) -> Dict[str, str]:
        """Parse simple ``key=value`` and ``key: value`` lines into a dict."""
        result: Dict[str, str] = {}
        for line in raw_input.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
            elif ":" in line:
                key, _, value = line.partition(":")
            else:
                continue
            result[key.strip()] = value.strip()
        return result

    def validate_format(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structural well-formedness of a parsed payload:
        must be a mapping of string keys to serializable values.
        """
        if not isinstance(data, dict):
            return False
        for k, v in data.items():
            if not isinstance(k, str):
                return False
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                return False
        return True

    def validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate data against a lightweight schema:
          schema = {
            "required": ["name"],
            "fields": {
              "name": {"type": str, "min_length": 1, "max_length": 100},
              "age":  {"type": int, "min": 0, "max": 150},
            },
            "types": {"flag": bool},
          }
        Returns a list of human-readable error strings (empty == valid).
        """
        errors: List[str] = []
        for field in schema.get("required", []):
            if field not in data or data[field] in (None, ""):
                errors.append(f"Missing required field: {field}")

        for field, spec in schema.get("fields", {}).items():
            if field not in data:
                continue
            value = data[field]
            expected = spec.get("type")
            if expected is not None and not isinstance(value, expected):
                errors.append(f"Field '{field}' must be of type {expected.__name__}")
                continue
            if isinstance(value, str):
                if "min_length" in spec and len(value) < spec["min_length"]:
                    errors.append(f"Field '{field}' shorter than {spec['min_length']}")
                if "max_length" in spec and len(value) > spec["max_length"]:
                    errors.append(f"Field '{field}' longer than {spec['max_length']}")
            elif isinstance(value, (int, float)):
                if "min" in spec and value < spec["min"]:
                    errors.append(f"Field '{field}' below minimum {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    errors.append(f"Field '{field}' above maximum {spec['max']}")

        for field, typ in schema.get("types", {}).items():
            if field in data and not isinstance(data[field], typ):
                errors.append(f"Field '{field}' must be of type {typ.__name__}")

        return errors


def create_input_parser(config: Optional[InputParserConfig] = None) -> InputParser:
    """Factory function to create input parser instance"""
    return InputParser(config)
