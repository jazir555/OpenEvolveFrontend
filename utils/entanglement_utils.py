"""
Entanglement Matrix Utilities

Normalization and serialization helpers for fractal entanglement matrices.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Set, List, Tuple

from utils.symbolic_analyzer import SymbolicAnalyzer


def normalize_entanglement_matrix(
    matrix: Optional[Mapping[str, Any]],
    allowed_ids: Optional[Iterable[str]] = None,
    enforce_symmetry: bool = True,
    strict: bool = False,
) -> Dict[str, Set[str]]:
    """
    Normalize entanglement matrices to a Dict[str, Set[str]].

    - Filters unknown ids when allowed_ids is provided
    - Removes self-entanglement
    - Optionally enforces symmetry (A->B implies B->A)
    - Ensures all allowed_ids exist as keys (empty set if none)
    """
    allowed_list = list(allowed_ids or [])
    allowed_set = set(allowed_list)
    raw_map: Dict[str, Set[str]] = {}

    if matrix and not isinstance(matrix, Mapping) and isinstance(matrix, (list, tuple)) and allowed_list:
        # Support adjacency-style matrices when allowed_ids provides ordering.
        for row_idx, row in enumerate(matrix):
            if row_idx >= len(allowed_list):
                break
            key = allowed_list[row_idx]
            if allowed_set and key not in allowed_set:
                continue
            if not isinstance(row, (list, tuple)):
                continue
            for col_idx, value in enumerate(row):
                if col_idx >= len(allowed_list):
                    break
                if not value:
                    continue
                partner = allowed_list[col_idx]
                if partner == key:
                    if strict:
                        raise ValueError(f"Self-entanglement detected for {key}")
                    continue
                raw_map.setdefault(key, set()).add(partner)

    if matrix and isinstance(matrix, Mapping):
        for key, value in matrix.items():
            if allowed_set and key not in allowed_set:
                if strict:
                    raise ValueError(f"Entanglement matrix key not allowed: {key}")
                continue
            if isinstance(value, (set, list, tuple)):
                items = value
            elif value is None:
                items = []
            else:
                items = [value]

            raw_set: Set[str] = set()
            for item in items:
                if item is None:
                    continue
                if item == key:
                    if strict:
                        raise ValueError(f"Self-entanglement detected for {key}")
                    continue
                if allowed_set and item not in allowed_set:
                    if strict:
                        raise ValueError(f"Entanglement partner not allowed: {item}")
                    continue
                raw_set.add(item)
            raw_map[key] = raw_set

    if strict and enforce_symmetry:
        for key, partners in raw_map.items():
            for partner in partners:
                if partner not in raw_map or key not in raw_map.get(partner, set()):
                    raise ValueError(f"Entanglement matrix not symmetric: {key} <-> {partner}")

    if not allowed_set:
        allowed_set = set(raw_map.keys())

    normalized: Dict[str, Set[str]] = {key: set() for key in allowed_set}
    for key, partners in raw_map.items():
        if allowed_set and key not in allowed_set:
            continue
        normalized.setdefault(key, set()).update(partners)

    if enforce_symmetry:
        for key, partners in list(normalized.items()):
            for partner in list(partners):
                if allowed_set and partner not in allowed_set:
                    continue
                normalized.setdefault(partner, set()).add(key)

    for key in normalized:
        normalized[key].discard(key)

    if allowed_set:
        for key in allowed_set:
            normalized.setdefault(key, set())

    return normalized


def serialize_entanglement_matrix(matrix: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
    """Serialize a normalized entanglement matrix into JSON-safe lists."""
    return {key: sorted(list(value)) for key, value in matrix.items()}


def build_symbolic_entanglement_matrix(
    sub_problems: Iterable[Any],
    allowed_ids: Optional[Iterable[str]] = None,
    enforce_symmetry: bool = True,
    strict: bool = False,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build a symbolic entanglement matrix from sub-problem content.

    Returns:
        (matrix, symbols_by_id)
    """
    analyzer = SymbolicAnalyzer()
    ids: List[str] = []
    symbols_by_id: Dict[str, Set[str]] = {}
    symbol_map: Dict[str, Set[str]] = {}

    for sp in sub_problems:
        sp_id = getattr(sp, "id", None)
        if not sp_id and isinstance(sp, dict):
            sp_id = sp.get("id")
        if not sp_id:
            continue
        ids.append(sp_id)

        metadata = getattr(sp, "metadata", None)
        if metadata is None and isinstance(sp, dict):
            metadata = sp.get("metadata") or {}
        metadata = metadata or {}

        symbols: Set[str] = set()
        for key in ("shared_symbols", "interface_symbols", "entanglement_symbols", "interface_contracts"):
            values = metadata.get(key)
            if isinstance(values, (list, tuple, set)):
                symbols.update(str(v) for v in values if v)
            elif isinstance(values, str):
                symbols.add(values)

        title = getattr(sp, "title", None)
        description = getattr(sp, "description", None)
        if title is None and isinstance(sp, dict):
            title = sp.get("title")
        if description is None and isinstance(sp, dict):
            description = sp.get("description")

        text = " ".join([t for t in [title, description] if t])
        if text:
            symbols.update(analyzer.analyze(text).symbols)

        symbols = {s for s in symbols if s}
        symbols_by_id[sp_id] = symbols

        for sym in symbols:
            symbol_map.setdefault(sym, set()).add(sp_id)

    matrix: Dict[str, Set[str]] = {sp_id: set() for sp_id in ids}
    for _, components in symbol_map.items():
        if len(components) < 2:
            continue
        for comp in components:
            matrix[comp].update({c for c in components if c != comp})

    normalized = normalize_entanglement_matrix(
        matrix,
        allowed_ids=allowed_ids or ids,
        enforce_symmetry=enforce_symmetry,
        strict=strict,
    )
    return normalized, symbols_by_id
