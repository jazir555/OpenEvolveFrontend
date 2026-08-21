"""
Enhanced Knowledge Core Module

Real, dependency-light implementations of the documented knowledge primitive
classes previously left as stubs:

- :class:`KnowledgeExtractor` -- heuristic entity/relation extraction from text.
- :class:`KnowledgeIntegrator` -- merging/continuous-integration of multiple
  knowledge sources into a unified store.
- :class:`EnhancedKnowledgeCore` -- a facade wiring extraction, storage and
  vector retrieval together.

Heavy optional dependencies (transformers, chroma, lean4) are imported lazily
and guarded so this module always imports cleanly on its own.
"""
from __future__ import annotations


import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from knowledge_storage import KnowledgeStorage
from vector_search import VectorSearch, VectorSearchConfig, HashEmbeddingProvider

logger = logging.getLogger(__name__)


_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_REL_VERB_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(is|are|uses|improves|reduces|causes|"
    r"depends on|enables|prevents)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str = "concept"
    confidence: float = 1.0
    source: str = ""


@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    source: str = ""


class KnowledgeExtractor:
    """Dependency-light knowledge extractor (heuristic, regex-based)."""

    def __init__(self, use_model: bool = False):
        self.use_model = use_model
        self._model = None
        if use_model:
            try:  # pragma: no cover - optional dependency
                from vector_search import TransformerEmbeddingProvider
                self._model = TransformerEmbeddingProvider()
            except Exception as exc:
                logger.info("Model extraction unavailable, using heuristics: %s", exc)
                self.use_model = False

    def extract_entities(self, text: str, source: str = "") -> List[ExtractedEntity]:
        seen: Dict[str, ExtractedEntity] = {}
        for m in _ENTITY_RE.finditer(text or ""):
            name = m.group(1)
            if name.lower() in {"the", "this", "that", "these", "those"}:
                continue
            if name not in seen:
                seen[name] = ExtractedEntity(name=name, confidence=0.8, source=source)
        return list(seen.values())

    def extract_relations(self, text: str, source: str = "") -> List[ExtractedRelation]:
        rels: List[ExtractedRelation] = []
        for m in _REL_VERB_RE.finditer(text or ""):
            rels.append(ExtractedRelation(
                subject=m.group(1), predicate=m.group(2), obj=m.group(3),
                confidence=0.7, source=source,
            ))
        return rels

    def extract(self, text: str, source: str = "") -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        return self.extract_entities(text, source), self.extract_relations(text, source)


class KnowledgeIntegrator:
    """
    Integrates (merges) multiple knowledge sources into a single store,
    de-duplicating by normalized key and keeping the highest-confidence record.
    """

    def __init__(self, storage: Optional[KnowledgeStorage] = None):
        self.storage = storage or KnowledgeStorage()

    @staticmethod
    def _key(artifact: Dict[str, Any]) -> str:
        base = artifact.get("title") or artifact.get("name") or artifact.get("id") or ""
        return re.sub(r"\s+", " ", base.strip().lower())

    def integrate(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        added: List[str] = []
        for art in artifacts:
            key = self._key(art)
            existing = next(
                (a for a in self.storage.list() if self._key(a) == key), None
            )
            if existing:
                if float(art.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                    self.storage.update(existing["id"], art)
                    added.append(existing["id"])
            else:
                added.append(self.storage.add(art))
        return added

    def merge_sources(self, *sources: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for src in sources:
            ids.extend(self.integrate(src))
        return ids


class EnhancedKnowledgeCore:
    """
    Facade combining extraction, storage and retrieval. Documented as the
    single entry point for "enhanced knowledge" operations.
    """

    def __init__(self, storage: Optional[KnowledgeStorage] = None,
                 search: Optional[VectorSearch] = None,
                 extractor: Optional[KnowledgeExtractor] = None):
        self.storage = storage or KnowledgeStorage()
        self.search = search or VectorSearch(VectorSearchConfig(
            embedding_provider=HashEmbeddingProvider()))
        self.extractor = extractor or KnowledgeExtractor()

    def ingest_text(self, text: str, source: str = "doc",
                    artifact_type: str = "insight") -> str:
        """Extract knowledge from text and persist it as an artifact."""
        entities, relations = self.extractor.extract(text, source)
        artifact = {
            "type": artifact_type,
            "source": source,
            "text": text,
            "entities": [e.__dict__ for e in entities],
            "relations": [r.__dict__ for r in relations],
            "title": text[:80],
            "confidence": 0.6,
            "tags": [e.name for e in entities[:5]],
        }
        aid = self.storage.add(artifact)
        self.search.add(text, {"id": aid, "type": artifact_type, "source": source})
        return aid

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.search.search(query, filters=None)[:top_k]
        out: List[Dict[str, Any]] = []
        for h in hits:
            art = self.storage.get(h["id"])
            if art is not None:
                out.append({**art, "score": h["score"]})
        return out
