"""
OpenEvolve validation router (mounted at ``/api/validation``).

Implements the content-validation contract the BubbleLab UI expects:
in-memory validation rules plus a ``/validation/run`` endpoint that checks a
piece of content against one or more rules. This is a real schema/content check
(no external service required).

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET    /validation/rules                  -> { rules, rule_names }
    GET    /validation/rules/{rule_name}      -> { name, rule }
    POST   /validation/rules                  -> { created, rule_name, rule }
    PUT    /validation/rules/{rule_name}      -> { updated, rule_name, rule }
    DELETE /validation/rules/{rule_name}      -> { deleted, rule_name }
    POST   /validation/run                    -> ValidationRunResult
    POST   /validation/compliance             -> ComplianceCheckResult
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.validation")

router = APIRouter()

# In-memory rule store: rule_name -> ValidationRule dict.
_RULES: Dict[str, Dict[str, Any]] = {}
_RULES_LOCK = threading.Lock()

# A couple of sensible default rules so the UI is not empty on first load.
_RULES["no_secrets"] = {
    "max_length": None,
    "min_length": None,
    "required_keywords": [],
    "forbidden_patterns": ["api_key\\s*=\\s*[\"'][^\"']+[\"']", "sk-[A-Za-z0-9]{20,}"],
    "required_sections": [],
}
_RULES["min_documentation"] = {
    "max_length": None,
    "min_length": 20,
    "required_keywords": [],
    "forbidden_patterns": [],
    "required_sections": ["\"\"\"", "def "],
}


def _normalize_rule(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_length": payload.get("max_length"),
        "min_length": payload.get("min_length"),
        "required_keywords": list(payload.get("required_keywords") or []),
        "forbidden_patterns": list(payload.get("forbidden_patterns") or []),
        "required_sections": list(payload.get("required_sections") or []),
    }


def _apply_rule(content: str, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
    import re

    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    min_length = rule.get("min_length")
    if isinstance(min_length, int) and len(content) < min_length:
        errors.append(
            f"Content length {len(content)} is below required minimum {min_length}"
        )
    max_length = rule.get("max_length")
    if isinstance(max_length, int) and len(content) > max_length:
        errors.append(
            f"Content length {len(content)} exceeds maximum {max_length}"
        )

    for keyword in rule.get("required_keywords") or []:
        if keyword not in content:
            errors.append(f"Missing required keyword: {keyword!r}")

    for pattern in rule.get("forbidden_patterns") or []:
        if re.search(pattern, content):
            errors.append(f"Forbidden pattern matched: {pattern!r}")

    for section in rule.get("required_sections") or []:
        if section not in content:
            warnings.append(f"Recommended section missing: {section!r}")
            suggestions.append(f"Add a '{section}' section to the content")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "rule_name": rule_name,
        "rule_config": _normalize_rule(rule),
    }


@router.get("/validation/rules")
async def list_validation_rules() -> JSONResponse:
    with _RULES_LOCK:
        rules = {name: _normalize_rule(rule) for name, rule in _RULES.items()}
    return JSONResponse(
        {"rules": rules, "rule_names": list(rules.keys())},
        headers={"Content-Type": "application/json"},
    )


@router.get("/validation/rules/{rule_name}")
async def get_validation_rule(rule_name: str) -> JSONResponse:
    with _RULES_LOCK:
        rule = _RULES.get(rule_name)
    if rule is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Validation rule not found: {rule_name}"},
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(
        {"name": rule_name, "rule": _normalize_rule(rule)},
        headers={"Content-Type": "application/json"},
    )


@router.post("/validation/rules")
async def create_validation_rule(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Body must be a JSON object"},
            headers={"Content-Type": "application/json"},
        )
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "'name' (str) is required"},
            headers={"Content-Type": "application/json"},
        )
    rule = _normalize_rule(body)
    with _RULES_LOCK:
        created = name not in _RULES
        _RULES[name] = rule
    return JSONResponse(
        {"created": created, "rule_name": name, "rule": rule},
        headers={"Content-Type": "application/json"},
    )


@router.put("/validation/rules/{rule_name}")
async def update_validation_rule(rule_name: str, request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    rule = _normalize_rule(body)
    with _RULES_LOCK:
        existed = rule_name in _RULES
        _RULES[rule_name] = rule
    return JSONResponse(
        {"updated": existed, "rule_name": rule_name, "rule": rule},
        headers={"Content-Type": "application/json"},
    )


@router.delete("/validation/rules/{rule_name}")
async def delete_validation_rule(rule_name: str) -> JSONResponse:
    with _RULES_LOCK:
        existed = rule_name in _RULES
        if existed:
            del _RULES[rule_name]
    return JSONResponse(
        {"deleted": existed, "rule_name": rule_name},
        headers={"Content-Type": "application/json"},
    )


@router.post("/validation/run")
async def run_validation(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Body must be a JSON object"},
            headers={"Content-Type": "application/json"},
        )
    content = body.get("content")
    if not isinstance(content, str):
        return JSONResponse(
            status_code=400,
            content={"error": "'content' (str) is required"},
            headers={"Content-Type": "application/json"},
        )
    rule_names = body.get("rule_names") or []
    if not isinstance(rule_names, list):
        rule_names = [rule_names]

    with _RULES_LOCK:
        available = dict(_RULES)

    validations: Dict[str, Any] = {}
    error_count = 0
    warning_count = 0
    suggestion_count = 0
    for name in rule_names:
        rule = available.get(name)
        if rule is None:
            validations[name] = {
                "valid": False,
                "errors": [f"Unknown rule: {name}"],
                "warnings": [],
                "suggestions": [],
                "rule_name": name,
                "rule_config": {},
            }
            error_count += 1
            continue
        result = _apply_rule(content, rule, name)
        validations[name] = result
        error_count += len(result["errors"])
        warning_count += len(result["warnings"])
        suggestion_count += len(result["suggestions"])

    overall = all(v["valid"] for v in validations.values()) if validations else True
    return JSONResponse(
        {
            "content_length": len(content),
            "validations": validations,
            "overall_result": overall,
            "error_count": error_count,
            "warning_count": warning_count,
            "suggestion_count": suggestion_count,
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/validation/compliance")
async def run_compliance_check(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    content = body.get("content")
    if not isinstance(content, str):
        return JSONResponse(
            status_code=400,
            content={"error": "'content' (str) is required"},
            headers={"Content-Type": "application/json"},
        )
    # Compliance runs every registered rule; framework is accepted but currently
    # drives the same checks (a real deployment would branch per framework).
    framework = body.get("framework") or "default"
    with _RULES_LOCK:
        available = dict(_RULES)

    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    for name, rule in available.items():
        result = _apply_rule(content, rule, name)
        errors.extend(result["errors"])
        warnings.extend(result["warnings"])
        suggestions.extend(result["suggestions"])

    return JSONResponse(
        {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "rule_name": f"compliance:{framework}",
            "rule_config": {},
        },
        headers={"Content-Type": "application/json"},
    )
