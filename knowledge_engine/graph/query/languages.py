"""
Multi-query-language support for the Knowledge Graph engine.

Gremlin, SPARQL, GraphQL and a Custom DSL are treated as first-class query
languages. Each translator lowers the source text to a parameterized Cypher
string (or, for SPARQL, to a SPARQL string) that the execution engine already
understands, so all optimization, caching and analytics apply uniformly.

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
from typing import Dict, List, Optional, Tuple, Any


class TranslationError(Exception):
    """Raised when a query in a non-Cypher language cannot be translated."""


class GremlinTranslator:
    """Translate a useful subset of Apache TinkerPop Gremlin to Cypher."""

    _HASVAL_RE = re.compile(r"\.has\('([^']+)'\s*,\s*(gt|gte|lt|lte|eq|neq)\(([^)]*)\)\)")
    _HAS_RE = re.compile(r"\.has\('([^']+)'\s*,\s*([^)]+)\)")
    _HASLABEL_RE = re.compile(r"\.hasLabel\('([^']+)'\)")
    _OUT_RE = re.compile(r"\.out\('([^']+)'\)")
    _IN_RE = re.compile(r"\.in\('([^']+)'\)")
    _BOTH_RE = re.compile(r"\.both\('([^']+)'\)")
    _LIMIT_RE = re.compile(r"\.limit\((\d+)\)")
    _COUNT_RE = re.compile(r"\.count\(\)")
    _VALUES_RE = re.compile(r"\.values\('([^']+)'\)")
    _AS_RE = re.compile(r"\.as\('([^']+)'\)")

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text.startswith("g."):
            raise TranslationError("Gremlin queries must start with 'g.'")

        # Strip the leading g.V() / g.V
        m = re.match(r"g\.V\(\)", text)
        if not m:
            raise TranslationError("Only g.V() traversals are supported")
        body = text[m.end():]

        nodes: List[Dict[str, Any]] = [{"var": "a", "labels": [], "wheres": []}]
        edges: List[Dict[str, Any]] = []
        ret: Optional[str] = None
        limit: Optional[int] = None
        do_count = False

        i = 0
        cur = nodes[0]
        while i < len(body):
            if body[i] != ".":
                i += 1
                continue
            if (mm := self._HASLABEL_RE.match(body, i)):
                cur["labels"].append(mm.group(1))
                i = mm.end()
            elif (mm := self._HASVAL_RE.match(body, i)):
                field, op, val = mm.group(1), mm.group(2), mm.group(3).strip().strip("'\"")
                cur["wheres"].append(self._cmp(field, op, val))
                i = mm.end()
            elif (mm := self._HAS_RE.match(body, i)):
                field, val = mm.group(1), mm.group(2).strip().strip("'\"")
                cur["wheres"].append(f"{cur['var']}.{field} = {self._lit(val)}")
                i = mm.end()
            elif (mm := self._OUT_RE.match(body, i)):
                nxt = self._new_node(nodes)
                edges.append({"src": cur["var"], "tgt": nxt["var"],
                              "type": mm.group(1), "dir": "out"})
                cur = nxt
                i = mm.end()
            elif (mm := self._IN_RE.match(body, i)):
                nxt = self._new_node(nodes)
                edges.append({"src": cur["var"], "tgt": nxt["var"],
                              "type": mm.group(1), "dir": "in"})
                cur = nxt
                i = mm.end()
            elif (mm := self._BOTH_RE.match(body, i)):
                nxt = self._new_node(nodes)
                edges.append({"src": cur["var"], "tgt": nxt["var"],
                              "type": mm.group(1), "dir": "both"})
                cur = nxt
                i = mm.end()
            elif (mm := self._LIMIT_RE.match(body, i)):
                limit = int(mm.group(1))
                i = mm.end()
            elif (mm := self._COUNT_RE.match(body, i)):
                do_count = True
                i = mm.end()
            elif (mm := self._VALUES_RE.match(body, i)):
                ret = f"{cur['var']}.{mm.group(1)}"
                i = mm.end()
            elif (mm := self._AS_RE.match(body, i)):
                cur["alias"] = mm.group(1)
                i = mm.end()
            else:
                i += 1

        return self._build_cypher(nodes, edges, ret, limit, do_count)

    @staticmethod
    def _new_node(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = {"var": chr(ord("a") + len(nodes)), "labels": [], "wheres": []}
        nodes.append(n)
        return n

    @staticmethod
    def _cmp(field: str, op: str, val: str) -> str:
        mapping = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=",
                   "neq": "<>"}
        return f"<NODE>.{field} {mapping.get(op, '=')} {GremlinTranslator._lit(val)}"

    @staticmethod
    def _lit(val: str) -> str:
        if val.startswith("'") or val.startswith('"'):
            return val
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}'"

    def _build_cypher(self, nodes, edges, ret, limit, do_count) -> str:
        node_by_var = {n["var"]: n for n in nodes}
        segs = [self._node_pat(nodes[0])]
        for e in edges:
            tgt = node_by_var[e["tgt"]]
            if e["dir"] == "out":
                segs.append(f"-[:{e['type']}]->")
            elif e["dir"] == "in":
                segs.append(f"<-[:{e['type']}]-")
            else:
                segs.append(f"-[:{e['type']}]-")
            segs.append(self._node_pat(tgt))
        cypher = "MATCH " + "".join(segs)
        wheres = []
        for n in nodes:
            for w in n["wheres"]:
                wheres.append(w.replace("<NODE>", n["var"]))
        if wheres:
            cypher += " WHERE " + " AND ".join(wheres)
        if do_count:
            cypher += " RETURN count(*)"
        else:
            cypher += f" RETURN {ret or 'a'}"
        if limit:
            cypher += f" LIMIT {limit}"
        return cypher

    @staticmethod
    def _node_pat(n) -> str:
        labels = ":".join(n["labels"])
        return f"({n['var']}{(':' + labels) if labels else ''})"


class SparqlTranslator:
    """Translate a basic SPARQL SELECT query into Cypher (or SPARQL)."""

    def translate(self, text: str, to_cypher: bool = True) -> str:
        text = re.sub(r"#.*", "", text)
        m = re.search(r"SELECT\s+(.+?)\s+WHERE\s*\{(.+?)\}", text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise TranslationError("Only SELECT ... WHERE { ... } is supported")
        select = m.group(1).strip()
        where = m.group(2)
        triples = self._parse_triples(where)
        if not to_cypher:
            return text.strip()

        var_map: Dict[str, str] = {}

        def vname(v: str) -> str:
            if v.startswith("?"):
                if v not in var_map:
                    var_map[v] = chr(ord("a") + len(var_map))
                return var_map[v]
            return v

        segs: List[str] = []
        first = True
        for (s, p, o) in triples:
            pred = p.strip("<>").split("/")[-1].split("#")[-1]
            s_v = vname(s)
            o_v = vname(o)
            if first:
                segs.append(f"({s_v})")
                first = False
            else:
                if s_v != segs[-1].strip("()"):
                    segs.append(f"({s_v})")
                else:
                    segs.pop()  # collapse duplicate anchor
            segs.append(f"-[:{pred}]->")
            segs.append(f"({o_v})")
        cypher = "MATCH " + " ".join(segs)
        select_vars = re.findall(r"\?(\w+)", select)
        if select_vars:
            ret = ", ".join(var_map.get("?" + v, v) for v in select_vars)
        else:
            ret = "*"
        cypher += f" RETURN {ret}"
        return cypher

    @staticmethod
    def _parse_triples(where: str) -> List[Tuple[str, str, str]]:
        triples = []
        for m in re.finditer(
                r"(\??\w+|\<[^>]+\>)\s+(\<[^>]+\>|[^\s]+)\s+(\??\w+|\<[^>]+\>|\"[^\"]*\")\s*\.?",
                where):
            triples.append((m.group(1), m.group(2), m.group(3)))
        return triples


class GraphQLTranslator:
    """Translate a minimal GraphQL-style graph query into Cypher."""

    _QUERY_RE = re.compile(
        r"\{\s*(\w+)\s*(?:\((.*?)\))?\s*\{([^}]*)\}\s*\}", re.DOTALL)

    def translate(self, text: str) -> str:
        m = self._QUERY_RE.match(text.strip())
        if not m:
            raise TranslationError("Unsupported GraphQL shape")
        entity = m.group(1)
        args = m.group(2)
        fields = m.group(3).strip()
        where = ""
        if args:
            conds = []
            for am in re.finditer(r"(\w+)\s*:\s*\"?([^\",]+)\"?", args):
                conds.append(f"n.{am.group(1)} = {self._lit(am.group(2))}")
            if conds:
                where = " WHERE " + " AND ".join(conds)
        field_list = [f.strip() for f in re.split(r"\s+", fields.strip()) if f.strip()]
        ret = ", ".join(f"n.{f}" for f in field_list) if field_list else "n"
        return f"MATCH (n:{entity}){where} RETURN {ret}"

    @staticmethod
    def _lit(val: str) -> str:
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}'"


class CustomDSLTranslator:
    """Translate a small Custom DSL into Cypher.

    Grammar (subset)::

        FIND <Type> [WHERE <field> <op> <value> (AND ...)] [RETURN <fields>] [LIMIT n]
    """

    _RE = re.compile(
        r"FIND\s+(\w+)\s*"
        r"(?:WHERE\s+(.+?))?\s*"
        r"(?:RETURN\s+(.+?))?\s*"
        r"(?:LIMIT\s+(\d+))?\s*$", re.IGNORECASE | re.DOTALL)

    def translate(self, text: str) -> str:
        m = self._RE.match(text.strip())
        if not m:
            raise TranslationError("Custom DSL must start with FIND <Type>")
        typ = m.group(1)
        where = m.group(2)
        ret = m.group(3)
        limit = m.group(4)
        cypher = f"MATCH (n:{typ})"
        if where:
            conds = []
            for part in re.split(r"\bAND\b", where, flags=re.IGNORECASE):
                part = part.strip()
                mm = re.match(r"(\w+)\s*(>=|<=|<>|!=|=|>|<|CONTAINS)\s*(.+)", part, re.IGNORECASE)
                if not mm:
                    continue
                field, op, val = mm.group(1), mm.group(2), mm.group(3).strip()
                val = val.strip("'\"")
                if op.upper() == "CONTAINS":
                    conds.append(f"n.{field} CONTAINS '{val}'")
                else:
                    conds.append(f"n.{field} {op} {self._lit(val)}")
            if conds:
                cypher += " WHERE " + " AND ".join(conds)
        if ret:
            fields = [f.strip() for f in ret.split(",")]
            cypher += " RETURN " + ", ".join(
                f"n.{f}" if not f.startswith("n.") else f for f in fields)
        else:
            cypher += " RETURN n"
        if limit:
            cypher += f" LIMIT {limit}"
        return cypher

    @staticmethod
    def _lit(val: str) -> str:
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}'"


class MultiLanguageTranslator:
    """Router that translates any supported language into Cypher."""

    SUPPORTED = ("cypher", "gremlin", "sparql", "graphql", "dsl", "custom")

    def __init__(self):
        self.gremlin = GremlinTranslator()
        self.sparql = SparqlTranslator()
        self.graphql = GraphQLTranslator()
        self.dsl = CustomDSLTranslator()

    def translate(self, language: str, text: str) -> Tuple[str, str]:
        """Return (cypher_or_sparql_string, target_backend)."""
        lang = (language or "cypher").lower()
        if lang in ("cypher", "openCypher"):
            return text, "memory"
        if lang == "gremlin":
            return self.gremlin.translate(text), "memory"
        if lang == "sparql":
            return self.sparql.translate(text, to_cypher=True), "memory"
        if lang == "graphql":
            return self.graphql.translate(text), "memory"
        if lang in ("dsl", "custom"):
            return self.dsl.translate(text), "memory"
        raise TranslationError(f"Unsupported language: {language}")


__all__ = [
    "TranslationError", "GremlinTranslator", "SparqlTranslator",
    "GraphQLTranslator", "CustomDSLTranslator", "MultiLanguageTranslator",
]
