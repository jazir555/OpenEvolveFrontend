"""
Knowledge Graph Query Parser, Validator, Normalizer and fluent QueryBuilder.

Implements a Cypher-subset AST parser (no external grammar dependency),
a security/complexity validator, a normalizer (canonicalization + constant
folding), and a fluent ``QueryBuilder`` API that emits parameterized Cypher.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re
import ast as _ast
import hashlib
from typing import List, Optional, Any, Dict, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class QueryError(Exception):
    """Base class for query processing errors."""


class QueryParseError(QueryError):
    """Raised when a query string cannot be parsed into a valid AST."""


class QueryValidationError(QueryError):
    """Raised when a parsed query fails validation (security/complexity)."""


class QueryNormalizeError(QueryError):
    """Raised when a query cannot be normalized."""


# --------------------------------------------------------------------------- #
# Pattern data structures
# --------------------------------------------------------------------------- #
@dataclass
class NodePattern:
    variable: str = ""
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    is_anonymous: bool = False


@dataclass
class RelationshipPattern:
    variable: str = ""
    types: List[str] = field(default_factory=list)
    direction: str = "out"  # out | in | both
    min_hops: Optional[int] = None
    max_hops: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    is_anonymous: bool = False


@dataclass
class PathPattern:
    elements: List[Union[NodePattern, RelationshipPattern]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Expression tree (for WHERE / filters)
# --------------------------------------------------------------------------- #
class ExprNode:
    """Base expression node."""

    def to_cypher(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Literal(ExprNode):
    value: Any

    def to_cypher(self) -> str:
        return _literal_to_cypher(self.value)


@dataclass
class Parameter(ExprNode):
    name: str

    def to_cypher(self) -> str:
        return f"${self.name}"


@dataclass
class Variable(ExprNode):
    name: str
    field: Optional[str] = None  # dotted access var.field

    def to_cypher(self) -> str:
        if self.field:
            return f"{self.name}.{self.field}"
        return self.name


@dataclass
class FunctionCall(ExprNode):
    name: str
    args: List[ExprNode] = field(default_factory=list)

    def to_cypher(self) -> str:
        return f"{self.name}(" + ", ".join(a.to_cypher() for a in self.args) + ")"


@dataclass
class UnaryExpr(ExprNode):
    op: str
    operand: ExprNode

    def to_cypher(self) -> str:
        return f"{self.op} {self.operand.to_cypher()}"


@dataclass
class BinaryExpr(ExprNode):
    op: str  # AND, OR, =, <>, <, <=, >, >=, =~, CONTAINS, STARTS WITH, ENDS WITH, IN
    left: ExprNode
    right: ExprNode

    def to_cypher(self) -> str:
        return f"({self.left.to_cypher()} {self.op} {self.right.to_cypher()})"


# --------------------------------------------------------------------------- #
# Clauses
# --------------------------------------------------------------------------- #
@dataclass
class MatchClause:
    patterns: List[PathPattern] = field(default_factory=list)
    optional: bool = False


@dataclass
class WhereClause:
    expression: Optional[ExprNode] = None


@dataclass
class ReturnItem:
    expression: ExprNode
    alias: Optional[str] = None


@dataclass
class ReturnClause:
    items: List[ReturnItem] = field(default_factory=list)
    distinct: bool = False


@dataclass
class WithClause:
    items: List[ReturnItem] = field(default_factory=list)
    where: Optional[ExprNode] = None


@dataclass
class OrderByItem:
    expression: ExprNode
    descending: bool = False


@dataclass
class OrderByClause:
    items: List[OrderByItem] = field(default_factory=list)


@dataclass
class LimitClause:
    count: ExprNode = None


@dataclass
class SkipClause:
    count: ExprNode = None


@dataclass
class CallClause:
    procedure: str
    args: List[ExprNode] = field(default_factory=list)
    yield_fields: List[str] = field(default_factory=list)


@dataclass
class CypherAst:
    clauses: List[Any] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    query_type: str = "read"
    complexity: int = 0
    original: str = ""

    def get_clauses(self, cls):
        return [c for c in self.clauses if isinstance(c, cls)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _literal_to_cypher(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal_to_cypher(v) for v in value) + "]"
    return repr(value)


_CYPHER_OPS = {
    "=", "!=", "<>", "<", "<=", ">", ">=", "=~", "CONTAINS",
    "STARTS WITH", "ENDS WITH", "IN", "AND", "OR", "XOR", "NOT",
}


# --------------------------------------------------------------------------- #
# Expression parser (recursive descent)
# --------------------------------------------------------------------------- #
class _ExpressionParser:
    """A small tokenizer + recursive-descent parser for Cypher expressions."""

    _TOKEN_RE = re.compile(
        r"""
        (?P<ws>\s+)
      | (?P<op>(?:<>|<=|>=|<|>|=~|=|\!=))
      | (?P<word>(?:AND|OR|XOR|NOT|CONTAINS|STARTS\s+WITH|ENDS\s+WITH|IN)\b)
      | (?P<param>\$[A-Za-z_][A-Za-z0-9_]*)
      | (?P<num>-?\d+\.\d+|-?\d+)
      | (?P<str>'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<star>\*)
      | (?P<punct>[()\[\].,])
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            m = self._TOKEN_RE.match(text, i)
            if not m:
                raise QueryParseError(f"Unexpected character at position {i}: {text[i]!r}")
            i = m.end()
            kind = m.lastgroup
            if kind == "ws":
                continue
            if kind == "word":
                tokens.append(("WORD", m.group().upper()))
            elif kind == "op":
                op = m.group()
                tokens.append(("OP", op if op != "!=" else "<>"))
            elif kind == "param":
                tokens.append(("PARAM", m.group()[1:]))
            elif kind == "num":
                val = float(m.group()) if "." in m.group() else int(m.group())
                tokens.append(("NUM", val))
            elif kind == "str":
                raw = m.group()[1:-1]
                raw = raw.encode().decode("unicode_escape")
                tokens.append(("STR", raw))
            elif kind == "ident":
                tokens.append(("IDENT", m.group()))
            elif kind == "punct":
                tokens.append(("PUNCT", m.group()))
        tokens.append(("EOF", None))
        return tokens

    def peek(self):
        return self.tokens[self.pos]

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind, value=None):
        tok = self.next()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise QueryParseError(f"Expected {kind} {value}, got {tok}")
        return tok

    # Grammar: or_expr -> and_expr (OR and_expr)*
    def parse(self) -> ExprNode:
        node = self._parse_or()
        if self.peek()[0] != "EOF":
            raise QueryParseError(f"Trailing tokens: {self.peek()}")
        return node

    def _parse_or(self):
        node = self._parse_and()
        while self.peek() == ("WORD", "OR"):
            self.next()
            rhs = self._parse_and()
            node = BinaryExpr("OR", node, rhs)
        return node

    def _parse_and(self):
        node = self._parse_not()
        while self.peek() == ("WORD", "AND"):
            self.next()
            rhs = self._parse_not()
            node = BinaryExpr("AND", node, rhs)
        return node

    def _parse_not(self):
        if self.peek() == ("WORD", "NOT"):
            self.next()
            return UnaryExpr("NOT", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_primary()
        tok = self.peek()
        if tok[0] == "OP" or (tok[0] == "WORD" and tok[1] in (
            "CONTAINS", "STARTS WITH", "ENDS WITH", "IN", "XOR")):
            op = tok[1]
            self.next()
            right = self._parse_primary()
            return BinaryExpr(op, left, right)
        return left

    def _parse_primary(self):
        tok = self.peek()
        if tok[0] == "PUNCT" and tok[1] == "(":
            self.next()
            node = self._parse_or()
            self.expect("PUNCT", ")")
            return node
        if tok[0] == "PARAM":
            self.next()
            return Parameter(tok[1])
        if tok[0] == "NUM":
            self.next()
            return Literal(tok[1])
        if tok[0] == "STR":
            self.next()
            return Literal(tok[1])
        if tok[0] == "WORD" and tok[1] in ("TRUE", "FALSE", "NULL"):
            self.next()
            if tok[1] == "TRUE":
                return Literal(True)
            if tok[1] == "FALSE":
                return Literal(False)
            return Literal(None)
        if tok[0] == "STAR":
            self.next()
            return Literal("*")
        if tok[0] == "IDENT":
            self.next()
            name = tok[1]
            # function call?
            if self.peek() == ("PUNCT", "("):
                self.next()
                args = []
                if self.peek() != ("PUNCT", ")"):
                    args.append(self._parse_or())
                    while self.peek() == ("PUNCT", ","):
                        self.next()
                        args.append(self._parse_or())
                self.expect("PUNCT", ")")
                return FunctionCall(name, args)
            # dotted variable.field
            field_name = None
            if self.peek() == ("PUNCT", "."):
                self.next()
                fld = self.expect("IDENT")
                field_name = fld[1]
            return Variable(name, field_name)
        raise QueryParseError(f"Unexpected token in expression: {tok}")


# --------------------------------------------------------------------------- #
# Pattern parser
# --------------------------------------------------------------------------- #
def _parse_node(text: str) -> NodePattern:
    inner = text.strip()
    if not inner:
        return NodePattern(variable="", is_anonymous=True)
    variable = ""
    labels: List[str] = []
    properties: Dict[str, Any] = {}
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)?", inner)
    if m and m.group(1):
        variable = m.group(1)
        inner = inner[m.end():]
    # labels
    while inner.startswith(":"):
        rest = inner[1:]
        lm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", rest)
        if not lm:
            break
        labels.append(lm.group(1))
        inner = rest[lm.end():]
    inner = inner.strip()
    if inner.startswith("{"):
        props_text = _balanced_brace(inner)
        properties = _parse_properties(props_text)
        inner = inner[len(props_text) + 2:].strip()
    if not variable:
        variable = ""
        is_anon = True
    else:
        is_anon = False
    return NodePattern(variable=variable, labels=labels, properties=properties,
                       is_anonymous=(variable == ""))


def _balanced_brace(text: str) -> str:
    """Return the content between the first '{' and its matching '}'."""
    assert text.startswith("{")
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[1:i]
    raise QueryParseError("Unbalanced braces in pattern")


def _parse_properties(text: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if not text.strip():
        return props
    # tokenize key: value pairs
    tokens = re.split(r",(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)", text)
    for part in tokens:
        if not part.strip():
            continue
        if ":" not in part:
            raise QueryParseError(f"Invalid property: {part}")
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        # strip wrapping quotes if string literal
        if (val.startswith("'") and val.endswith("'")) or (
                val.startswith('"') and val.endswith('"')):
            val = val[1:-1]
        elif val.startswith("$"):
            props[key] = Parameter(val[1:])
        else:
            try:
                props[key] = _ast.literal_eval(val)
            except Exception:
                props[key] = val
    return props


def _parse_relationship(inner: str) -> RelationshipPattern:
    inner = inner.strip()
    variable = ""
    types: List[str] = []
    properties: Dict[str, Any] = {}
    min_hops = None
    max_hops = None
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)?", inner)
    if m and m.group(1):
        variable = m.group(1)
        inner = inner[m.end():]
    while inner.startswith(":"):
        rest = inner[1:]
        tm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", rest)
        if not tm:
            break
        types.append(tm.group(1))
        inner = rest[tm.end():]
    # hop range *min..max
    hm = re.match(r"^\*(\d*)\.\.(\d*)", inner)
    if hm:
        lo = hm.group(1)
        hi = hm.group(2)
        min_hops = int(lo) if lo else 1
        max_hops = int(hi) if hi else None
        inner = inner[hm.end():]
    elif re.match(r"^\*\d+", inner):
        sm = re.match(r"^\*(\d+)", inner)
        min_hops = max_hops = int(sm.group(1))
        inner = inner[sm.end():]
    inner = inner.strip()
    if inner.startswith("{"):
        props_text = _balanced_brace(inner)
        properties = _parse_properties(props_text)
    return RelationshipPattern(
        variable=variable, types=types, min_hops=min_hops, max_hops=max_hops,
        properties=properties, is_anonymous=(variable == ""))


def _parse_path(pattern: str) -> PathPattern:
    pattern = pattern.strip()
    elements: List[Union[NodePattern, RelationshipPattern]] = []
    # iterate over node/rel segments
    i = 0
    n = len(pattern)
    while i < n:
        if pattern[i] == "(":
            end = pattern.find(")", i)
            if end == -1:
                raise QueryParseError("Unbalanced parenthesis in pattern")
            elements.append(_parse_node(pattern[i + 1:end]))
            i = end + 1
        elif pattern[i] in "<>-":
            # relationship segment (may be preceded/followed by - or <)
            rel_text = ""
            direction = "out"
            if pattern[i] == "<":
                direction = "in"
                i += 1
                if pattern[i] != "-":
                    raise QueryParseError("Expected '-' after '<'")
                i += 1
            elif pattern[i] == "-":
                i += 1
            # optional typed relationship [...]
            if pattern[i] == "[":
                end = pattern.find("]", i)
                if end == -1:
                    raise QueryParseError("Unbalanced bracket in relationship")
                rel_text = pattern[i + 1:end]
                i = end + 1
            # trailing direction: optional '-' then optional '>'
            if i < n and pattern[i] == "-":
                i += 1
            if i < n and pattern[i] == ">":
                direction = "out" if direction == "out" else "in"
                i += 1
            elif direction == "in":
                pass
            else:
                direction = "both"
            rel = _parse_relationship(rel_text) if rel_text else RelationshipPattern()
            rel.direction = direction
            elements.append(rel)
        elif pattern[i] == " ":
            i += 1
        else:
            raise QueryParseError(f"Unexpected char in pattern: {pattern[i]!r}")
    return PathPattern(elements=elements)


# --------------------------------------------------------------------------- #
# QueryParser
# --------------------------------------------------------------------------- #
_CLAUSE_SPLIT = re.compile(
    r"\b(MATCH|OPTIONAL\s+MATCH|WHERE|WITH|RETURN|ORDER\s+BY|SKIP|LIMIT|UNION|UNWIND|CALL)\b",
    re.IGNORECASE,
)


class QueryParser:
    """Parse a Cypher-subset query string into a :class:`CypherAst`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = QueryValidator(self.config.get("validation", {}))
        self.normalizer = QueryNormalizer(self.config.get("normalization", {}))

    def parse(self, query_string: str) -> CypherAst:
        """Parse, validate and normalize a query string."""
        ast = self._parse_to_ast(query_string)
        validation = self.validator.validate(ast)
        if not validation.valid:
            raise QueryValidationError("; ".join(validation.errors))
        normalized = self.normalizer.normalize(ast)
        return normalized

    async def parse_async(self, query_string: str) -> CypherAst:
        return self.parse(query_string)

    # -- internal ---------------------------------------------------------- #
    def _parse_to_ast(self, query_string: str) -> CypherAst:
        original = query_string
        query_string = query_string.strip()
        if not query_string:
            raise QueryParseError("Empty query")

        # Split into top-level clauses
        matches = list(_CLAUSE_SPLIT.finditer(query_string))
        if not matches:
            raise QueryParseError("Query must start with a known clause")

        ast = CypherAst(original=original)
        prev_end = 0
        for idx, m in enumerate(matches):
            keyword = m.group(1).upper().replace(" ", "_")
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(query_string)
            body = query_string[m.end():end].strip()

            if keyword == "MATCH":
                optional = False
                if body.upper().startswith("OPTIONAL"):
                    optional = True
                    body = body[len("OPTIONAL"):].strip()
                ast.clauses.append(self._parse_match(body, optional))
            elif keyword == "WHERE":
                ast.clauses.append(self._parse_where(body))
            elif keyword == "RETURN":
                ast.clauses.append(self._parse_return(body))
            elif keyword == "WITH":
                ast.clauses.append(self._parse_with(body))
            elif keyword == "ORDER_BY":
                ast.clauses.append(self._parse_order_by(body))
            elif keyword == "LIMIT":
                ast.clauses.append(LimitClause(count=Literal(int(body.strip()))))
            elif keyword == "SKIP":
                ast.clauses.append(SkipClause(count=Literal(int(body.strip()))))
            elif keyword == "UNION":
                ast.clauses.append(("UNION", body))
            elif keyword == "UNWIND":
                ast.clauses.append(self._parse_unwind(body))
            elif keyword == "CALL":
                ast.clauses.append(self._parse_call(body))
            prev_end = end

        ast.query_type = self._determine_query_type(ast)
        ast.complexity = self._estimate_complexity(ast)
        return ast

    def _parse_match(self, body: str, optional: bool) -> MatchClause:
        # Split comma-separated patterns
        patterns_raw = self._split_top_level(body, ",")
        patterns = [_parse_path(p) for p in patterns_raw]
        return MatchClause(patterns=patterns, optional=optional)

    def _parse_where(self, body: str) -> WhereClause:
        expr = _ExpressionParser(body).parse()
        return WhereClause(expression=expr)

    def _parse_return(self, body: str) -> ReturnClause:
        distinct = False
        if body.upper().startswith("DISTINCT"):
            distinct = True
            body = body[len("DISTINCT"):].strip()
        items_raw = self._split_top_level(body, ",")
        items = []
        for raw in items_raw:
            raw = raw.strip()
            # alias: "expr AS alias"
            asm = re.match(r"^(.*?)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", raw, re.IGNORECASE | re.DOTALL)
            if asm:
                expr = _ExpressionParser(asm.group(1).strip()).parse()
                items.append(ReturnItem(expression=expr, alias=asm.group(2)))
            else:
                items.append(ReturnItem(expression=_ExpressionParser(raw).parse()))
        return ReturnClause(items=items, distinct=distinct)

    def _parse_with(self, body: str) -> WithClause:
        where = None
        if " WHERE " in body.upper():
            bidx = body.upper().rindex(" WHERE ")
            where_expr_text = body[bidx + len(" WHERE "):].strip()
            where = _ExpressionParser(where_expr_text).parse()
            body = body[:bidx].strip()
        items_raw = self._split_top_level(body, ",")
        items = []
        for raw in items_raw:
            raw = raw.strip()
            asm = re.match(r"^(.*?)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", raw, re.IGNORECASE | re.DOTALL)
            if asm:
                expr = _ExpressionParser(asm.group(1).strip()).parse()
                items.append(ReturnItem(expression=expr, alias=asm.group(2)))
            else:
                items.append(ReturnItem(expression=_ExpressionParser(raw).parse()))
        return WithClause(items=items, where=where)

    def _parse_order_by(self, body: str) -> OrderByClause:
        items_raw = self._split_top_level(body, ",")
        items = []
        for raw in items_raw:
            raw = raw.strip()
            descending = False
            m = re.search(r"\b(ASC|DESC)\b", raw, re.IGNORECASE)
            if m:
                descending = m.group(1).upper() == "DESC"
                raw = raw[:m.start()].strip()
            expr = _ExpressionParser(raw).parse()
            items.append(OrderByItem(expression=expr, descending=descending))
        return OrderByClause(items=items)

    def _parse_unwind(self, body: str):
        return ("UNWIND", body)

    def _parse_call(self, body: str):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*?)\)(?:\s+YIELD\s+(.+))?$", body,
                    re.IGNORECASE | re.DOTALL)
        if not m:
            raise QueryParseError(f"Could not parse CALL clause: {body}")
        procedure = m.group(1)
        args_text = m.group(2).strip()
        args = []
        if args_text:
            for a in self._split_top_level(args_text, ","):
                args.append(_ExpressionParser(a.strip()).parse())
        yields = []
        if m.group(3):
            for y in self._split_top_level(m.group(3), ","):
                yields.append(y.strip())
        return CallClause(procedure=procedure, args=args, yield_fields=yields)

    @staticmethod
    def _split_top_level(text: str, sep: str) -> List[str]:
        parts = []
        depth = 0
        cur = ""
        in_str = None
        for ch in text:
            if in_str:
                cur += ch
                if ch == in_str:
                    in_str = None
                continue
            if ch in "'\"":
                in_str = ch
                cur += ch
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == sep and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur)
        return parts

    def _determine_query_type(self, ast: CypherAst) -> str:
        text = ast.original.upper()
        if any(isinstance(c, CallClause) for c in ast.clauses):
            return "procedure"
        if "CREATE" in text or "MERGE" in text or "DELETE" in text or "SET " in text or "DETACH" in text:
            return "write"
        if text.strip().startswith(("CREATE", "DROP", "INDEX", "CONSTRAINT")):
            return "schema"
        return "read"

    def _estimate_complexity(self, ast: CypherAst) -> int:
        complexity = 0
        complexity += 10 * sum(len(m.patterns) for m in ast.get_clauses(MatchClause))
        complexity += 5 * (1 if ast.get_clauses(WhereClause) else 0)
        complexity += 15 * len(ast.get_clauses(ReturnClause))
        complexity += 20 * sum(
            1 for c in ast.clauses if isinstance(c, CallClause))
        return complexity


# --------------------------------------------------------------------------- #
# QueryValidator
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    blocking: bool = False


_INJECTION_PATTERNS = [
    re.compile(r";\s*(DROP|DELETE|CREATE|MERGE|SET|ALTER)", re.IGNORECASE),
    re.compile(r"--"),
    re.compile(r"/\*"),
    re.compile(r"\bUNION\b.*\bSELECT\b", re.IGNORECASE),
]


class QueryValidator:
    """Validate query ASTs for syntax, complexity limits and injection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_nodes_scanned = self.config.get("max_nodes_scanned", 100000)
        self.max_rels_scanned = self.config.get("max_relationships_scanned", 100000)
        self.max_depth = self.config.get("max_depth", 10)
        self.max_conditions = self.config.get("max_conditions", 50)

    def validate(self, ast: CypherAst) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # Must have a MATCH or CALL for graph reads
        if not ast.get_clauses(MatchClause) and not ast.get_clauses(CallClause):
            errors.append("Query must contain at least one MATCH or CALL clause")

        # Must have a RETURN (for read) unless procedure
        if ast.query_type == "read" and not ast.get_clauses(ReturnClause):
            errors.append("Read query must contain a RETURN clause")

        # Complexity / scan limits
        nodes_est = self._estimate_nodes_scanned(ast)
        if nodes_est > self.max_nodes_scanned:
            errors.append(
                f"Query would scan ~{nodes_est} nodes, exceeding limit "
                f"{self.max_nodes_scanned}")
            suggestions.append("Add more specific filters or use indexes")

        rels_est = self._estimate_rels_scanned(ast)
        if rels_est > self.max_rels_scanned:
            errors.append(
                f"Query would scan ~{rels_est} relationships, exceeding limit "
                f"{self.max_rels_scanned}")

        depth = self._calculate_depth(ast)
        if depth > self.max_depth:
            errors.append(
                f"Query traversal depth {depth} exceeds maximum {self.max_depth}")
            suggestions.append("Reduce variable-length pattern depth")

        # Injection scan on raw text
        for pat in _INJECTION_PATTERNS:
            if pat.search(ast.original):
                errors.append("Potential injection pattern detected in query")
                break

        return ValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            blocking=bool(errors),
        )

    def _estimate_nodes_scanned(self, ast: CypherAst) -> int:
        total = 0
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                for el in p.elements:
                    if isinstance(el, NodePattern):
                        total += 1000 if not el.labels and not el.properties else 100
        return max(total, 1)

    def _estimate_rels_scanned(self, ast: CypherAst) -> int:
        total = 0
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                for el in p.elements:
                    if isinstance(el, RelationshipPattern):
                        span = (el.max_hops or 1) * 100
                        total += span
        return max(total, 1)

    def _calculate_depth(self, ast: CypherAst) -> int:
        depth = 0
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                rels = [e for e in p.elements if isinstance(e, RelationshipPattern)]
                path_len = sum((e.max_hops or 1) for e in rels)
                depth = max(depth, path_len)
        return depth


# --------------------------------------------------------------------------- #
# QueryNormalizer
# --------------------------------------------------------------------------- #
class QueryNormalizer:
    """Canonicalize AST and perform constant folding."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def normalize(self, ast: CypherAst) -> CypherAst:
        # Constant folding in WHERE / WITH / RETURN expressions
        for clause in ast.clauses:
            if isinstance(clause, WhereClause) and clause.expression:
                clause.expression = self._fold(clause.expression)
            elif isinstance(clause, ReturnClause):
                for item in clause.items:
                    item.expression = self._fold(item.expression)
            elif isinstance(clause, WithClause):
                for item in clause.items:
                    item.expression = self._fold(item.expression)
                if clause.where:
                    clause.where = self._fold(clause.where)
            elif isinstance(clause, OrderByClause):
                for item in clause.items:
                    item.expression = self._fold(item.expression)
        return ast

    def _fold(self, node: ExprNode) -> ExprNode:
        if isinstance(node, BinaryExpr):
            left = self._fold(node.left)
            right = self._fold(node.right)
            # fold numeric/string arithmetic where both sides are literals
            if isinstance(left, Literal) and isinstance(right, Literal):
                val = self._eval_binary(node.op, left.value, right.value)
                if val is not None:
                    return Literal(val)
            return BinaryExpr(node.op, left, right)
        if isinstance(node, UnaryExpr):
            operand = self._fold(node.operand)
            if node.op == "NOT" and isinstance(operand, Literal):
                return Literal(not operand.value)
            return UnaryExpr(node.op, operand)
        if isinstance(node, FunctionCall):
            args = [self._fold(a) for a in node.args]
            return FunctionCall(node.name, args)
        return node

    @staticmethod
    def _eval_binary(op: str, l: Any, r: Any) -> Optional[Any]:
        try:
            if op == "=":
                return l == r
            if op in ("<>", "!="):
                return l != r
            if op == "<":
                return l < r
            if op == "<=":
                return l <= r
            if op == ">":
                return l > r
            if op == ">=":
                return l >= r
            if op == "AND":
                return bool(l) and bool(r)
            if op == "OR":
                return bool(l) or bool(r)
        except Exception:
            return None
        return None


# --------------------------------------------------------------------------- #
# Fluent QueryBuilder
# --------------------------------------------------------------------------- #
class QueryBuilder:
    """Fluent API that builds parameterized Cypher queries."""

    def __init__(self):
        self.query_parts: List[str] = []
        self.parameters: Dict[str, Any] = {}
        self.query_type: Optional[str] = None
        self._param_counter = 0

    def _new_param(self, value: Any) -> str:
        name = f"p{self._param_counter}"
        self._param_counter += 1
        self.parameters[name] = value
        return name

    def match(self, pattern: str) -> "QueryBuilder":
        self.query_parts.append(f"MATCH {pattern}")
        self.query_type = "read"
        return self

    def optional_match(self, pattern: str) -> "QueryBuilder":
        self.query_parts.append(f"OPTIONAL MATCH {pattern}")
        return self

    def where(self, condition: str, **params: Any) -> "QueryBuilder":
        self.query_parts.append(f"WHERE {condition}")
        self.parameters.update(params)
        return self

    def where_equals(self, field: str, value: Any) -> "QueryBuilder":
        p = self._new_param(value)
        return self.where(f"{field} = ${p}")

    def where_in(self, field: str, values: List[Any]) -> "QueryBuilder":
        p = self._new_param(values)
        return self.where(f"{field} IN ${p}")

    def with_clause(self, variables: str) -> "QueryBuilder":
        self.query_parts.append(f"WITH {variables}")
        return self

    def return_clause(self, variables: str, distinct: bool = False) -> "QueryBuilder":
        prefix = "RETURN DISTINCT " if distinct else "RETURN "
        self.query_parts.append(prefix + variables)
        self.query_type = self.query_type or "read"
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "QueryBuilder":
        direction = direction.upper()
        self.query_parts.append(f"ORDER BY {field} {direction}")
        return self

    def limit(self, count: int) -> "QueryBuilder":
        self.query_parts.append(f"LIMIT {int(count)}")
        return self

    def skip(self, count: int) -> "QueryBuilder":
        self.query_parts.append(f"SKIP {int(count)}")
        return self

    def call(self, procedure: str, *args: Any) -> "QueryBuilder":
        arg_str = ", ".join(str(a) for a in args)
        self.query_parts.append(f"CALL {procedure}({arg_str})")
        return self

    def union(self, other: "QueryBuilder") -> "QueryBuilder":
        self.query_parts.append("UNION")
        self.query_parts.extend(other.query_parts)
        self.parameters.update(other.parameters)
        return self

    def add_parameter(self, name: str, value: Any) -> "QueryBuilder":
        self.parameters[name] = value
        return self

    def build(self) -> Dict[str, Any]:
        query = " ".join(self.query_parts)
        return {
            "query": query,
            "parameters": self.parameters,
            "query_type": self.query_type or "read",
        }

    # -- convenience for CypherAst construction -------------------------- #
    def to_ast(self) -> CypherAst:
        """Build the query then parse it into a normalized AST."""
        built = self.build()
        parser = QueryParser()
        ast = parser.parse(built["query"])
        ast.parameters.update(built["parameters"])
        return ast


def ast_to_cypher(ast: CypherAst) -> str:
    """Render a :class:`CypherAst` back to a canonical Cypher string."""
    parts: List[str] = []
    for clause in ast.clauses:
        if isinstance(clause, MatchClause):
            head = "OPTIONAL MATCH " if clause.optional else "MATCH "
            pats = []
            for p in clause.patterns:
                segs = []
                for el in p.elements:
                    if isinstance(el, NodePattern):
                        segs.append(_render_node(el))
                    else:
                        segs.append(_render_rel(el))
                pats.append("".join(segs))
            parts.append(head + ", ".join(pats))
        elif isinstance(clause, WhereClause):
            if clause.expression:
                parts.append("WHERE " + clause.expression.to_cypher())
        elif isinstance(clause, ReturnClause):
            items = ", ".join(
                (i.expression.to_cypher() + (f" AS {i.alias}" if i.alias else ""))
                for i in clause.items)
            parts.append(("RETURN DISTINCT " if clause.distinct else "RETURN ") + items)
        elif isinstance(clause, WithClause):
            items = ", ".join(
                (i.expression.to_cypher() + (f" AS {i.alias}" if i.alias else ""))
                for i in clause.items)
            line = "WITH " + items
            if clause.where:
                line += " WHERE " + clause.where.to_cypher()
            parts.append(line)
        elif isinstance(clause, OrderByClause):
            items = ", ".join(
                (i.expression.to_cypher() + (" DESC" if i.descending else " ASC"))
                for i in clause.items)
            parts.append("ORDER BY " + items)
        elif isinstance(clause, LimitClause):
            parts.append(f"LIMIT {clause.count.to_cypher()}")
        elif isinstance(clause, SkipClause):
            parts.append(f"SKIP {clause.count.to_cypher()}")
        elif isinstance(clause, CallClause):
            arg_str = ", ".join(a.to_cypher() for a in clause.args)
            line = f"CALL {clause.procedure}({arg_str})"
            if clause.yield_fields:
                line += " YIELD " + ", ".join(clause.yield_fields)
            parts.append(line)
        elif isinstance(clause, tuple) and clause[0] == "UNION":
            parts.append("UNION")
    return " ".join(parts)


def _render_node(n: NodePattern) -> str:
    label_str = ":" + ":".join(n.labels) if n.labels else ""
    if n.properties:
        props = "{" + ", ".join(
            f"{k}: {_render_prop_value(v)}" for k, v in n.properties.items()) + "}"
        return f"({n.variable}{label_str} {props})"
    return f"({n.variable}{label_str})"


def _render_prop_value(v: Any) -> str:
    if isinstance(v, Parameter):
        return f"${v.name}"
    if isinstance(v, ExprNode):
        return v.to_cypher()
    return _literal_to_cypher(v)


def _render_rel(r: RelationshipPattern) -> str:
    type_str = ":" + ":".join(r.types) if r.types else ""
    hop = ""
    if r.min_hops is not None or r.max_hops is not None:
        lo = r.min_hops if r.min_hops is not None else ""
        hi = r.max_hops if r.max_hops is not None else ""
        hop = f"*{lo}..{hi}"
    if r.properties:
        props = "{" + ", ".join(
            f"{k}: {_render_prop_value(v)}" for k, v in r.properties.items()) + "}"
        inner = f"{r.variable}{type_str}{hop} {props}"
    else:
        inner = f"{r.variable}{type_str}{hop}"
    if r.direction == "in":
        return f"<-[{inner}]-"
    if r.direction == "out":
        return f"-[{inner}]->"
    return f"-[{inner}]-"


__all__ = [
    "QueryError", "QueryParseError", "QueryValidationError", "QueryNormalizeError",
    "NodePattern", "RelationshipPattern", "PathPattern", "ExprNode", "Literal",
    "Parameter", "Variable", "FunctionCall", "UnaryExpr", "BinaryExpr",
    "MatchClause", "WhereClause", "ReturnItem", "ReturnClause", "WithClause",
    "OrderByItem", "OrderByClause", "LimitClause", "SkipClause", "CallClause",
    "CypherAst", "ValidationResult", "QueryParser", "QueryValidator",
    "QueryNormalizer", "QueryBuilder", "ast_to_cypher",
]
