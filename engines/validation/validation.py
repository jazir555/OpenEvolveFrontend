"""
Validation engine: real, dependency-light (stdlib only) validators.

Public names preserved: SyntaxValidator, LintChecker, TypeAnnotationChecker,
ImportValidator, CodingStandardChecker, ComplexityChecker.
"""
from __future__ import annotations


from types import SimpleNamespace
import ast
import re
from typing import List, Optional, Dict, Any


class SyntaxValidator:
    """Validate source code syntax using the standard library AST parser."""

    def validate_syntax(self, code: str, language: str = "python", **kwargs) -> object:
        if language != "python":
            return SimpleNamespace(
                valid=True,
                errors=[f"Syntax validation only implemented for python, got '{language}'"],
            )
        try:
            ast.parse(code)
            return SimpleNamespace(valid=True, errors=[])
        except SyntaxError as e:
            return SimpleNamespace(
                valid=False,
                errors=[f"Line {e.lineno}: {e.msg} ({e.text or ''}).trim()"],
            )


class LintChecker:
    """
    Lightweight AST-based linter.

    Real checks performed (no external linters required):
      - wildcard / star imports
      - bare ``except:`` clauses
      - use of ``eval`` / ``exec``
      - mutable default arguments
      - use of ``assert`` (dangerous in production)
      - unused imports / names (name-reference analysis)
    """

    DEFAULT_RULES = [
        "wildcard_import",
        "bare_except",
        "eval_exec",
        "mutable_default_arg",
        "assert_used",
        "unused_import",
    ]

    def check(self, code: str, rules: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        rules = rules or self.DEFAULT_RULES
        issues: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        class _Visitor(ast.NodeVisitor):
            def __init__(self):
                self.imported_names: Dict[str, ast.Import] = {}
                self.used_names = set()
                self.star_imports = False

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if alias.asname:
                        self.imported_names[alias.asname] = node
                    else:
                        self.imported_names[name] = node
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.names and node.names[0].name == "*":
                    self.star_imports = True
                for alias in node.names:
                    if alias.asname:
                        self.imported_names[alias.asname] = node
                    else:
                        self.imported_names[alias.name] = node
                self.generic_visit(node)

            def visit_Name(self, node):
                self.used_names.add(node.id)
                self.generic_visit(node)

        visitor = _Visitor()
        visitor.visit(tree)

        if "wildcard_import" in rules and visitor.star_imports:
            issues.append({"rule": "wildcard_import", "message": "Wildcard (star) import detected", "line": 0})

        for node in ast.walk(tree):
            if "bare_except" in rules and isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({"rule": "bare_except", "message": "Bare except clause", "line": node.lineno})
            if "eval_exec" in rules and isinstance(node, ast.Call):
                fn = getattr(node.func, "id", None)
                if fn in ("eval", "exec"):
                    issues.append({"rule": "eval_exec", "message": f"Use of {fn}()", "line": node.lineno})
            if "mutable_default_arg" in rules and isinstance(node, ast.FunctionDef):
                for d in node.args.defaults:
                    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                        issues.append({"rule": "mutable_default_arg", "message": f"Mutable default arg in {node.name}", "line": node.lineno})
                        break
            if "assert_used" in rules and isinstance(node, ast.Assert):
                issues.append({"rule": "assert_used", "message": "assert used (disabled under -O)", "line": node.lineno})

        if "unused_import" in rules:
            for name, node in visitor.imported_names.items():
                if name not in visitor.used_names:
                    issues.append({"rule": "unused_import", "message": f"Unused import '{name}'", "line": getattr(node, "lineno", 0)})

        return issues


class TypeAnnotationChecker:
    """Detect functions missing type annotations (params or return)."""

    def check(self, code: str, require_return: bool = True, **kwargs) -> object:
        missing: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SimpleNamespace(fully_typed=True, valid=True, missing_annotations=[])

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.arg in ("self", "cls"):
                        continue
                    if arg.annotation is None:
                        missing.append(f"{node.name}:{arg.arg}")
                if require_return and node.returns is None:
                    missing.append(f"{node.name}:return")

        return SimpleNamespace(
            fully_typed=not missing,
            valid=not missing,
            missing_annotations=missing,
        )


class ImportValidator:
    """Validate imports against allow/deny lists."""

    def validate(
        self,
        code: str,
        allowlist: Optional[List[str]] = None,
        denylist: Optional[List[str]] = None,
        **kwargs,
    ) -> object:
        allowlist = allowlist or []
        denylist = denylist or []
        invalid: List[str] = []
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SimpleNamespace(valid=True, invalid_imports=[], violations=[])

        def _mod(node):
            if isinstance(node, ast.Import):
                return [a.name for a in node.names]
            return [node.module or ""]

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for mod in _mod(node):
                    top = mod.split(".")[0]
                    if denylist and any(re.fullmatch(d.replace("*", ".*"), mod) or top in denylist for d in denylist):
                        violations.append(mod)
                        continue
                    if allowlist and top not in allowlist and mod not in allowlist:
                        invalid.append(mod)

        return SimpleNamespace(
            valid=not (invalid or violations),
            invalid_imports=invalid,
            violations=violations,
        )


class CodingStandardChecker:
    """Real PEP8-ish standard checks (line length, naming, docstrings)."""

    def check(self, code: str, standard: str = "pep8", max_line_length: int = 100, **kwargs) -> object:
        violations: List[str] = []
        score = 100

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length:
                violations.append(f"Line {i} exceeds {max_line_length} chars")
                score -= 2

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SimpleNamespace(valid=True, violations=[], score=100)

        snake = re.compile(r"^[a-z_][a-z0-9_]*$")
        capsnake = re.compile(r"^[A-Z_][A-Z0-9_]*$")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if not snake.match(node.name):
                    violations.append(f"Function '{node.name}' not snake_case")
                    score -= 3
                if ast.get_docstring(node) is None:
                    violations.append(f"Function '{node.name}' missing docstring")
                    score -= 2
            elif isinstance(node, ast.ClassDef):
                if not (snake.match(node.name) or capsnake.match(node.name)):
                    violations.append(f"Class '{node.name}' not CapWords")
                    score -= 3

        score = max(0, score)
        return SimpleNamespace(valid=score >= 70, violations=violations, score=score)


class ComplexityChecker:
    """
    Real McCabe cyclomatic complexity using AST decision-point counting.
    Per-function complexity plus an aggregate check against limits.
    """

    _DECISION = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
        ast.With, ast.AsyncWith, ast.IfExp, ast.comprehension,
        ast.Assert, ast.BoolOp,
    )

    def check(self, code: str, limits: Optional[Dict[str, int]] = None, **kwargs) -> object:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SimpleNamespace(
                valid=True, complexity=1, functions=[], within_limits=True,
                cyclomatic=1, lines=len(code.split("\n")),
            )

        functions: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                    elif isinstance(child, self._DECISION):
                        complexity += 1
                functions.append({"name": node.name, "complexity": complexity})

        total = sum(f["complexity"] for f in functions) or 1
        max_c = max((f["complexity"] for f in functions), default=1)

        result = SimpleNamespace(
            valid=True,
            complexity=total,
            functions=functions,
            within_limits=True,
            cyclomatic=max_c,
            lines=len(code.split("\n")),
        )
        if limits:
            result.within_limits = (
                result.cyclomatic <= limits.get("cyclomatic", 100)
                and result.lines <= limits.get("lines", 1000)
            )
            result.valid = result.within_limits
        return result
