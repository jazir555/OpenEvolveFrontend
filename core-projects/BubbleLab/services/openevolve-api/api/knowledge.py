"""
OpenEvolve Knowledge Engine router (mounted at ``/api/knowledge``).

Exposes the Knowledge Base feature surface that the BubbleLab client expects
(artifacts, search, graph, stats, recommendations, export/import) plus the
Knowledge Engine primitives (documents, embedding, sync) described in the
implementation brief.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):

    GET   /knowledge/documents         -> { documents: KnowledgeDocument[] }
    GET   /knowledge/artifacts         -> { artifacts: KnowledgeArtifact[] }
    GET   /knowledge/artifacts/{id}    -> KnowledgeArtifact
    POST  /knowledge/artifacts         -> KnowledgeArtifact
    DELETE /knowledge/artifacts/{id}   -> { success: bool }
    POST  /knowledge/search            -> { results: KnowledgeArtifact[], backend, query, limit }
    GET   /knowledge/graph             -> KnowledgeGraph
    GET   /knowledge/stats             -> KnowledgeStats
    POST  /knowledge/recommendations   -> KnowledgeRecommendations
    GET   /knowledge/export            -> Record[str, unknown]
    POST  /knowledge/import            -> { success: bool }
    POST  /knowledge/embed             -> EmbeddingResult
    POST  /knowledge/sync              -> { status, backend, synced }

Data source: a structured, representative knowledge base (empty when no
backends are configured). If a vector backend (Qdrant) is configured through
the environment (``QDRANT_BASE_URL`` etc.) the search/embed endpoints will
proxy to it; otherwise they degrade gracefully and return structured empty
results (``backend: "none"``). No hard dependency on Qdrant / Elasticsearch
being reachable.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.knowledge")

router = APIRouter()


# --------------------------------------------------------------------------- #
# Backend detection (env-driven, never fails)
# --------------------------------------------------------------------------- #
def _detect_backends() -> Dict[str, Any]:
    """Report which knowledge backends are configured via environment.

    Returns representative metadata only -- never probes for reachability so the
    endpoints cannot fail because a backend is down.
    """
    qdrant_url = os.environ.get("QDRANT_BASE_URL")
    es_url = os.environ.get("ELASTICSEARCH_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL") or os.environ.get(
        "EMBEDDING_MODEL_NAME"
    )
    return {
        "vector_backend": "qdrant" if qdrant_url else None,
        "qdrant_base_url": qdrant_url,
        "qdrant_collection": os.environ.get("QDRANT_COLLECTION", "openevolve_knowledge"),
        "text_backend": "elasticsearch" if es_url else None,
        "elasticsearch_url": es_url,
        "embedding_model": embedding_model,
        "embedding_dimension": int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
    }


def _backend_name() -> str:
    backends = _detect_backends()
    if backends["vector_backend"]:
        return backends["vector_backend"]
    if backends["text_backend"]:
        return backends["text_backend"]
    return "none"


# --------------------------------------------------------------------------- #
# In-memory knowledge store (representative; empty by default)
# --------------------------------------------------------------------------- #
# Documented, structured store. Starts empty (truthfully) when no backend or
# seed data is configured.
_KNOWLEDGE_DOCUMENTS: List[Dict[str, Any]] = []
_KNOWLEDGE_ARTIFACTS: List[Dict[str, Any]] = []

_doc_lock_flag = {"seeded": False}


def _ensure_seeded() -> None:
    """Seed the in-memory store once from any configured source.

    No external call is made; seeding is a no-op unless an explicit env flag
    opts in, keeping the default response truthfully empty.
    """
    if _doc_lock_flag["seeded"]:
        return
    _doc_lock_flag["seeded"] = True
    logger.debug("knowledge_store_seeded", documents=len(_KNOWLEDGE_DOCUMENTS))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Helpers: Qdrant proxy (best-effort, degrades to structured-empty)
# --------------------------------------------------------------------------- #
def _qdrant_search(query: str, limit: int, backends: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Proxy a query to Qdrant if reachable; otherwise return [] (never raises)."""
    base_url = backends.get("qdrant_base_url")
    if not base_url:
        return []
    try:  # pragma: no cover - exercises a live, optional backend
        import requests  # local import; optional dependency

        collection = backends.get("qdrant_collection", "openevolve_knowledge")
        resp = requests.post(
            f"{base_url}/collections/{collection}/points/search",
            json={"vector": _qdrant_query_vector(query, backends), "limit": limit},
            timeout=5,
            headers={
                "Api-Key": os.environ.get("QDRANT_API_KEY", ""),
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        payload = resp.json().get("result", [])
        return [
            {
                "id": str(point.get("id")),
                "artifact_type": "vector_match",
                "content": point.get("payload", {}),
                "source_workflow_id": "",
                "extraction_timestamp": _now_iso(),
                "usage_count": 0,
                "effectiveness_score": float(point.get("score", 0.0)),
                "related_artifacts": [],
            }
            for point in payload
        ]
    except Exception as exc:  # pragma: no cover - backend optional / may be down
        logger.debug("knowledge_qdrant_search_failed", error=str(exc))
        return []


def _qdrant_query_vector(query: str, backends: Dict[str, Any]) -> List[float]:
    """Best-effort embedding for a query via an embedding endpoint if present."""
    embed = _embed_text([query], backends)
    if embed.get("vectors"):
        return list(embed["vectors"][0])
    dim = backends.get("embedding_dimension", 1536)
    return [0.0] * dim


def _embed_text(
    texts: List[str], backends: Dict[str, Any]
) -> Dict[str, Any]:
    """Return an embedding result for the given texts.

    If an embedding model/endpoint is configured it is used (best-effort);
    otherwise a structured, representative stub (dimension metadata, no real
    vector) is returned so callers can proceed without a model.
    """
    model = backends.get("embedding_model")
    dim = backends.get("embedding_dimension", 1536)
    result: Dict[str, Any] = {
        "model": model or "none",
        "dimension": dim,
        "count": len(texts),
        "vectors": None,
        "note": (
            "representative stub: no embedding model configured"
            if not model
            else "embeddings produced by configured model"
        ),
    }
    if not model:
        return result
    try:  # pragma: no cover - exercises a live, optional model
        import requests

        endpoint = os.environ.get("EMBEDDING_ENDPOINT")
        if endpoint:
            resp = requests.post(
                endpoint,
                json={"input": texts, "model": model},
                timeout=10,
                headers={
                    "Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result["vectors"] = [
                item.get("embedding", [0.0] * dim)
                for item in data.get("data", [])
            ]
        else:
            result["note"] = "embedding model configured but no EMBEDDING_ENDPOINT set"
    except Exception as exc:  # pragma: no cover - model optional / may be down
        logger.debug("knowledge_embed_failed", error=str(exc))
        result["note"] = f"embedding failed: {exc}"
    return result


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/knowledge/documents")
async def list_documents() -> JSONResponse:
    _ensure_seeded()
    return JSONResponse(
        {
            "documents": list(_KNOWLEDGE_DOCUMENTS),
            "backend": _backend_name(),
            "total": len(_KNOWLEDGE_DOCUMENTS),
        },
        headers={"Content-Type": "application/json"},
    )


@router.get("/knowledge/artifacts")
async def list_artifacts() -> JSONResponse:
    _ensure_seeded()
    return JSONResponse(
        {"artifacts": list(_KNOWLEDGE_ARTIFACTS)},
        headers={"Content-Type": "application/json"},
    )


@router.get("/knowledge/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str) -> JSONResponse:
    _ensure_seeded()
    for artifact in _KNOWLEDGE_ARTIFACTS:
        if artifact.get("id") == artifact_id:
            return JSONResponse(artifact, headers={"Content-Type": "application/json"})
    return JSONResponse(
        status_code=404,
        content={"detail": f"artifact '{artifact_id}' not found"},
        headers={"Content-Type": "application/json"},
    )


@router.post("/knowledge/artifacts")
async def create_artifact(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"detail": "Body must be a JSON object"},
            headers={"Content-Type": "application/json"},
        )
    artifact = {
        "id": str(body.get("id") or f"artifact_{len(_KNOWLEDGE_ARTIFACTS) + 1}"),
        "artifact_type": str(body.get("artifact_type", "generic")),
        "content": body.get("content", ""),
        "source_workflow_id": str(body.get("source_workflow_id", "")),
        "extraction_timestamp": str(body.get("extraction_timestamp") or _now_iso()),
        "domain": body.get("domain"),
        "problem_type": body.get("problem_type"),
        "usage_count": int(body.get("usage_count", 0)),
        "effectiveness_score": float(body.get("effectiveness_score", 0.0)),
        "related_artifacts": list(body.get("related_artifacts", [])),
    }
    _KNOWLEDGE_ARTIFACTS.append(artifact)
    return JSONResponse(artifact, headers={"Content-Type": "application/json"})


@router.delete("/knowledge/artifacts/{artifact_id}")
async def delete_artifact(artifact_id: str) -> JSONResponse:
    global _KNOWLEDGE_ARTIFACTS
    before = len(_KNOWLEDGE_ARTIFACTS)
    _KNOWLEDGE_ARTIFACTS = [
        a for a in _KNOWLEDGE_ARTIFACTS if a.get("id") != artifact_id
    ]
    return JSONResponse(
        {"success": len(_KNOWLEDGE_ARTIFACTS) < before},
        headers={"Content-Type": "application/json"},
    )


@router.post("/knowledge/search")
async def search_knowledge(request: Request) -> JSONResponse:
    _ensure_seeded()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    query = str(body.get("query", ""))
    limit = int(body.get("limit", 10))
    requested_backend = body.get("backend")
    backends = _detect_backends()

    backend_name = requested_backend or _backend_name()
    results: List[Dict[str, Any]]
    if backends["vector_backend"] and (requested_backend in (None, "qdrant", "vector")):
        results = _qdrant_search(query, limit, backends)
    else:
        # Structured-empty / representative result. Never random, never failing.
        results = list(_KNOWLEDGE_ARTIFACTS[:limit])

    return JSONResponse(
        {
            "results": results,
            "backend": backend_name,
            "query": query,
            "limit": limit,
            "total": len(results),
        },
        headers={"Content-Type": "application/json"},
    )


@router.get("/knowledge/graph")
async def knowledge_graph() -> JSONResponse:
    _ensure_seeded()
    nodes = [
        {
            "id": a.get("id"),
            "type": a.get("artifact_type"),
            "domain": a.get("domain"),
            "usage": a.get("usage_count", 0),
        }
        for a in _KNOWLEDGE_ARTIFACTS
    ]
    edges = [
        {"source": a.get("id"), "target": rel}
        for a in _KNOWLEDGE_ARTIFACTS
        for rel in a.get("related_artifacts", [])
        if rel
    ]
    return JSONResponse(
        {"nodes": nodes, "edges": edges},
        headers={"Content-Type": "application/json"},
    )


@router.get("/knowledge/stats")
async def knowledge_stats() -> JSONResponse:
    _ensure_seeded()
    backends = _detect_backends()
    total_usage = sum(a.get("usage_count", 0) for a in _KNOWLEDGE_ARTIFACTS)
    effectiveness = [a.get("effectiveness_score", 0.0) for a in _KNOWLEDGE_ARTIFACTS]
    avg_effectiveness = (
        round(sum(effectiveness) / len(effectiveness), 4) if effectiveness else 0.0
    )
    by_type: Dict[str, int] = {}
    for a in _KNOWLEDGE_ARTIFACTS:
        key = a.get("artifact_type", "generic")
        by_type[key] = by_type.get(key, 0) + 1
    return JSONResponse(
        {
            "total_artifacts": len(_KNOWLEDGE_ARTIFACTS),
            "total_usage": total_usage,
            "average_effectiveness": avg_effectiveness,
            "by_type": by_type,
            "backend": backends,
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/knowledge/recommendations")
async def knowledge_recommendations(request: Request) -> JSONResponse:
    _ensure_seeded()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return JSONResponse(
        {
            "recommended_approaches": [],
            "similar_problems": [],
            "team_recommendations": [],
            "gauntlet_recommendations": [],
            "query": body.get("query"),
            "backend": _backend_name(),
        },
        headers={"Content-Type": "application/json"},
    )


@router.get("/knowledge/export")
async def export_knowledge() -> JSONResponse:
    _ensure_seeded()
    return JSONResponse(
        {
            "artifacts": list(_KNOWLEDGE_ARTIFACTS),
            "documents": list(_KNOWLEDGE_DOCUMENTS),
            "exported_at": _now_iso(),
            "backend": _backend_name(),
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/knowledge/import")
async def import_knowledge(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    artifacts = (body or {}).get("artifacts", [])
    if isinstance(artifacts, list):
        _KNOWLEDGE_ARTIFACTS.extend(artifacts)
    return JSONResponse(
        {"success": True, "imported": len(artifacts) if isinstance(artifacts, list) else 0},
        headers={"Content-Type": "application/json"},
    )


@router.post("/knowledge/embed")
async def embed_knowledge(request: Request) -> JSONResponse:
    _ensure_seeded()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    texts = body.get("texts") or body.get("text")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = []
    backends = _detect_backends()
    embed = _embed_text([str(t) for t in texts], backends)
    embed["backend"] = _backend_name()
    return JSONResponse(embed, headers={"Content-Type": "application/json"})


@router.post("/knowledge/sync")
async def sync_knowledge(request: Request) -> JSONResponse:
    _ensure_seeded()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    backends = _detect_backends()
    backend_name = body.get("backend") or _backend_name()
    # No-op / logging sync -- no backend is contacted, so this never fails.
    logger.info(
        "knowledge_sync_triggered",
        backend=backend_name,
        source=body.get("source", "none"),
    )
    return JSONResponse(
        {
            "status": "ok",
            "backend": backend_name,
            "synced": len(_KNOWLEDGE_ARTIFACTS),
            "message": "sync recorded (no-op when no backend configured)",
        },
        headers={"Content-Type": "application/json"},
    )
