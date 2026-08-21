"""
Knowledge Storage Module

Real, dependency-light persistence layer for knowledge artifacts. Provides
CRUD, tag/type based querying and optional JSONL file persistence so the
knowledge engine can store and retrieve artifacts without any external service.

Author: OpenEvolve Team
Date: 2026-08
"""
from __future__ import annotations


import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeStorage:
    """
    Knowledge storage.

    Stores knowledge artifacts (dicts) keyed by id. Supports in-memory operation
    and optional append-only JSONL persistence. Safe to use without any database.
    """
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._items: Dict[str, Dict[str, Any]] = {}
        if persist_path:
            self._load()

    def add(self, artifact: Dict[str, Any]) -> str:
        aid = artifact.get("id") or str(uuid.uuid4())
        artifact = dict(artifact)
        artifact["id"] = aid
        artifact.setdefault("created_at", _now())
        artifact["updated_at"] = _now()
        self._items[aid] = artifact
        self._append(artifact)
        return aid

    def get(self, aid: str) -> Optional[Dict[str, Any]]:
        return self._items.get(aid)

    def update(self, aid: str, patch: Dict[str, Any]) -> bool:
        if aid not in self._items:
            return False
        self._items[aid].update(patch)
        self._items[aid]["updated_at"] = _now()
        self._rewrite()
        return True

    def delete(self, aid: str) -> bool:
        if aid in self._items:
            del self._items[aid]
            self._rewrite()
            return True
        return False

    def list(self, artifact_type: Optional[str] = None,
             tags: Optional[List[str]] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for art in self._items.values():
            if artifact_type and art.get("type") != artifact_type:
                continue
            if tags:
                have = set(art.get("tags", []))
                if not have.issuperset(set(tags)):
                    continue
            out.append(art)
            if len(out) >= limit:
                break
        return out

    def count(self) -> int:
        return len(self._items)

    def _append(self, artifact: Dict[str, Any]) -> None:
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(artifact) + "\n")
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("KnowledgeStorage append failed: %s", exc)

    def _rewrite(self) -> None:
        if not self.persist_path:
            return
        try:
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for art in self._items.values():
                    fh.write(json.dumps(art) + "\n")
            Path(tmp).replace(self.persist_path)
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("KnowledgeStorage rewrite failed: %s", exc)

    def _load(self) -> None:
        try:
            p = Path(self.persist_path)
            if not p.exists():
                return
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                art = json.loads(line)
                self._items[art["id"]] = art
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            logger.warning("KnowledgeStorage load failed: %s", exc)


class KnowledgeStore(KnowledgeStorage):
    """Knowledge store (alias kept for backwards-compatible naming)."""
    pass
