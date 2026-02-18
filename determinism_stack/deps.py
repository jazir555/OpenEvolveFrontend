"""Local dependency discovery and sys.path wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


LOCAL_DEP_PATHS: Dict[str, List[Path]] = {
    "lagrange-mapper": [Path("lagrange-mapper")],
    "roma": [Path("ROMA") / "src"],
    "lmql": [Path("lmql") / "src"],
    "jsonformer": [Path("jsonformer")],
    "outlines": [Path("outlines")],
    "guardrails": [Path("guardrails")],
    "steer": [Path("steer") / "steer" / "src"],
    "dspy": [Path("dspy")],
    "detllm": [Path("detllm")],
    "knowledge_engine": [Path("knowledge_engine")],
    "agentic-context-engine": [Path("agentic-context-engine")],
    "adaptive_mdap": [Path("adaptive_mdap")],
    "mdap_maker": [Path("mdap_maker")],
}


def ensure_local_dependencies(extra_paths: Iterable[Path] | None = None) -> List[Path]:
    root = repo_root()
    added: List[Path] = []
    for paths in LOCAL_DEP_PATHS.values():
        for rel in paths:
            candidate = (root / rel).resolve()
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                    added.append(candidate)
    if extra_paths:
        for path in extra_paths:
            candidate = path.resolve()
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                    added.append(candidate)
    return added


def available_local_dependencies() -> Dict[str, bool]:
    root = repo_root()
    return {name: any((root / rel).exists() for rel in paths) for name, paths in LOCAL_DEP_PATHS.items()}


def matryoshka_cli_path() -> Path:
    return (repo_root() / "Matryoshka" / "src" / "index.ts").resolve()

