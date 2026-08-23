"""
RAGBits API Routes for OpenEvolve (mounted at ``/openevolve/ragbits``).

Lightweight in-memory RAG surface. The actual ragbits library is NOT imported
(heavy dependency) — requests are accepted, recorded, and answered with a
structured, deterministic JSON response so the API always boots.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.ragbits")

router = APIRouter()

# In-memory document/embedding store (no real vectors; we keep metadata).
_DOCUMENTS: Dict[str, Dict[str, Any]] = []
_SEARCH_LOG: List[Dict[str, Any]] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SearchRequest(BaseModel):
    query: str = ""
    top_k: int = Field(default=5, ge=1, le=100)
    collection: str = "default"
    filters: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    collection: str = "default"
    text: str = ""


@router.post("/search")
async def ragbits_search(payload: SearchRequest) -> Dict[str, Any]:
    """Return a (placeholder) retrieval result for a query."""
    query = payload.query
    # Simple in-memory lexical match against ingested document text.
    hits = []
    for doc in _DOCUMENTS:
        haystack = (doc.get("text", "") + " " + doc.get("title", "")).lower()
        if query.lower() in haystack:
            hits.append(
                {
                    "id": doc.get("id"),
                    "score": 0.5,
                    "title": doc.get("title"),
                    "snippet": doc.get("text", "")[:200],
                }
            )
        if len(hits) >= payload.top_k:
            break
    _SEARCH_LOG.append({"query": query, "hits": len(hits), "at": _now_iso()})
    logger.info("ragbits_search", query=query, hits=len(hits))
    return {
        "success": True,
        "status": "ok",
        "query": query,
        "collection": payload.collection,
        "results": hits,
        "count": len(hits),
        "note": "ragbits not bundled; lexical fallback over ingested docs.",
    }


@router.post("/ingest")
async def ragbits_ingest(payload: IngestRequest) -> Dict[str, Any]:
    """Ingest documents (metadata only) into the in-memory store."""
    docs = list(payload.documents)
    if payload.text:
        docs.append({"id": f"doc_{uuid.uuid4().hex[:12]}", "title": "inline", "text": payload.text})
    ingested = []
    for d in docs:
        doc_id = d.get("id") or f"doc_{uuid.uuid4().hex[:12]}"
        record = {
            "id": doc_id,
            "title": d.get("title", doc_id),
            "text": d.get("text", ""),
            "collection": payload.collection,
            "ingested_at": _now_iso(),
        }
        _DOCUMENTS.append(record)
        ingested.append(doc_id)
    logger.info("ragbits_ingest", count=len(ingested))
    return {
        "success": True,
        "status": "accepted",
        "ingested": ingested,
        "count": len(ingested),
        "collection": payload.collection,
    }


@router.get("/stats")
async def ragbits_stats() -> Dict[str, Any]:
    """Return RAGBits usage statistics."""
    return {
        "success": True,
        "status": "ok",
        "documents": len(_DOCUMENTS),
        "searches": len(_SEARCH_LOG),
        "collections": sorted({d.get("collection", "default") for d in _DOCUMENTS}),
        "generated_at": _now_iso(),
    }
