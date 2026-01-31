"""Utility helpers for deterministic stack integrations."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import importlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


def optional_import(module_name: str):
    try:
        from .deps import ensure_local_dependencies
        ensure_local_dependencies()
    except Exception:
        pass
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def optional_attr(module_name: str, attr: str):
    module = optional_import(module_name)
    if module is None:
        return None
    return getattr(module, attr, None)


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _schema_default(schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type", "string")
    if schema_type == "object":
        props = schema.get("properties", {})
        return {key: _schema_default(val) for key, val in props.items()}
    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_schema_default(item_schema)]
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    return ""


def build_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not schema:
        return {}
    if schema.get("type") != "object":
        return {"value": _schema_default(schema)}
    return _schema_default(schema)


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Iterable[str]]:
    if not schema:
        return True, []
    jsonschema = optional_import("jsonschema")
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, []
        except Exception as exc:
            return False, [str(exc)]
    required = schema.get("required", [])
    missing = [key for key in required if key not in data]
    return len(missing) == 0, [f"missing: {key}" for key in missing]


def safe_eval(expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
    if not expression:
        return True
    safe_module = optional_import("knowledge_engine.orchestration.safe_eval")
    if safe_module and hasattr(safe_module, "safe_eval"):
        return safe_module.safe_eval(expression, context or {})
    import ast
    import operator as op

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if context and node.id in context:
                return context[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op_node, comp in zip(node.ops, node.comparators):
                if not allowed_ops[type(op_node)](left, _eval(comp)):
                    return False
            return True
        raise ValueError("Unsafe expression")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


def safe_z3_eval(expression: str, context: Dict[str, Any]):
    """Safely parse a constraint expression into a Z3 expression."""
    if not expression or not expression.strip():
        return None
    z3 = optional_import("z3")
    if z3 is None:
        raise ValueError("z3 not available")
    import ast

    ops = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Mod: lambda a, b: a % b,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return z3.Not(_eval(node.operand))
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*values)
            if isinstance(node.op, ast.Or):
                return z3.Or(*values)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            comps = []
            for op_node, comp in zip(node.ops, node.comparators):
                comps.append(ops[type(op_node)](left, _eval(comp)))
                left = _eval(comp)
            return z3.And(*comps) if len(comps) > 1 else comps[0]
        raise ValueError("Unsupported constraint expression")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


def is_valid_json_prefix(text: str) -> bool:
    if not text:
        return True
    open_braces = 0
    in_string = False
    escape = False
    for ch in text:
        if ch == "\\" and not escape:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
                if open_braces < 0:
                    return False
        escape = False
    return open_braces >= 0


@dataclass
class SeedSnapshot:
    random_state: object
    numpy_state: Optional[object]
    torch_state: Optional[object]
    torch_cuda_state: Optional[object]


@contextlib.contextmanager
def deterministic_seed(seed: int) -> Iterator[None]:
    numpy = optional_import("numpy")
    torch = optional_import("torch")
    snapshot = SeedSnapshot(
        random_state=random.getstate(),
        numpy_state=numpy.random.get_state() if numpy else None,
        torch_state=torch.random.get_rng_state() if torch else None,
        torch_cuda_state=torch.cuda.get_rng_state_all() if torch and torch.cuda.is_available() else None,
    )
    random.seed(seed)
    if numpy:
        numpy.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(snapshot.random_state)
        if numpy and snapshot.numpy_state is not None:
            numpy.random.set_state(snapshot.numpy_state)
        if torch and snapshot.torch_state is not None:
            torch.random.set_rng_state(snapshot.torch_state)
        if torch and snapshot.torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(snapshot.torch_cuda_state)
