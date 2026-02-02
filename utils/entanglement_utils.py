"""
Entanglement Matrix Utilities

Normalization and serialization helpers for fractal entanglement matrices.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Set, List


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
    allowed_set = set(allowed_ids or [])
    raw_map: Dict[str, Set[str]] = {}

    if matrix:
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
