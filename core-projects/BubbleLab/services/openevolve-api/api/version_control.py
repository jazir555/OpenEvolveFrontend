"""
OpenEvolve protocol version-control router (mounted at ``/api/version-control``).

Implements the version-control surface the BubbleLab client expects
(``src/services/openevolveApi.ts`` -> ``listVersions`` / ``getVersion`` /
``getCurrentVersion`` / ``createVersion`` / ``loadVersion`` / ``branchVersion`` /
``compareVersions`` / ``deleteVersion``).

The "current" version is derived for real from the repository's git state
(current branch, HEAD commit hash, dirty flag) when available; the rest of the
catalog is an in-memory, representative store seeded with that derived version so
the UI renders an accurate baseline rather than 404ing. No random values are
generated.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET    /version-control/versions                 -> { versions, current_version_id }
    GET    /version-control/versions/{version_id}   -> VersionEntry
    GET    /version-control/current                  -> { current }
    POST   /version-control/versions                 -> { version_id, version }
    POST   /version-control/versions/{version_id}/load   -> { loaded, current }
    POST   /version-control/versions/{version_id}/branch -> { version_id, version }
    POST   /version-control/compare                  -> VersionCompareResult
    DELETE /version-control/versions/{version_id}   -> { deleted }

Data source: real git metadata (best-effort) + in-memory store.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.version_control")

router = APIRouter()

_DEFAULT_VERSION_ID = "v1-initial"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_info() -> Dict[str, Optional[str]]:
    """Best-effort git metadata for the repo root (two levels up from this file)."""
    info: Dict[str, Optional[str]] = {
        "branch": None,
        "commit": None,
        "dirty": None,
    }
    try:
        import os

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for key, args in (
            ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("dirty", ["git", "status", "--porcelain"]),
        ):
            try:
                out = subprocess.run(
                    args,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if key == "dirty":
                    info["dirty"] = "true" if out.stdout.strip() else "false"
                else:
                    info[key] = out.stdout.strip() or None
            except Exception:
                continue
    except Exception:
        pass
    return info


def _build_initial_version() -> Dict[str, Any]:
    git = _git_info()
    branch = git.get("branch") or "main"
    commit = git.get("commit") or "unknown"
    return {
        "id": _DEFAULT_VERSION_ID,
        "name": f"initial@{branch}:{commit[:8] if commit != 'unknown' else 'unknown'}",
        "timestamp": _now_iso(),
        "protocol_text": (
            "# OpenEvolve protocol (initial version)\n"
            "# Derived from current git state.\n"
            f"branch: {branch}\n"
            f"commit: {commit}\n"
            f"dirty: {git.get('dirty')}\n"
        ),
        "comment": "Auto-seeded baseline version from repository git state.",
        "author": "openevolve-api",
        "complexity_metrics": {},
        "structure_analysis": {},
        "branch_from": None,
        "branch_name": None,
    }


_VERSIONS: Dict[str, Dict[str, Any]] = {_DEFAULT_VERSION_ID: _build_initial_version()}
_CURRENT_VERSION_ID: str = _DEFAULT_VERSION_ID


def _find_version(version_id: str) -> Dict[str, Any]:
    if version_id not in _VERSIONS:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    return _VERSIONS[version_id]


@router.get("/version-control/versions")
async def list_versions() -> Dict[str, Any]:
    return {
        "versions": list(_VERSIONS.values()),
        "current_version_id": _CURRENT_VERSION_ID,
    }


@router.get("/version-control/versions/{version_id}")
async def get_version(version_id: str) -> Dict[str, Any]:
    return _find_version(version_id)


@router.get("/version-control/current")
async def get_current_version() -> Dict[str, Any]:
    return {"current": _VERSIONS.get(_CURRENT_VERSION_ID)}


@router.post("/version-control/versions")
async def create_version(payload: Dict[str, Any]) -> Dict[str, Any]:
    protocol_text = payload.get("protocol_text", "")
    version_name = payload.get("version_name") or f"v{len(_VERSIONS) + 1}-{_now_iso()}"
    version_id = version_name
    version = {
        "id": version_id,
        "name": version_name,
        "timestamp": _now_iso(),
        "protocol_text": protocol_text,
        "comment": payload.get("comment"),
        "author": payload.get("author", "openevolve-api"),
        "complexity_metrics": {},
        "structure_analysis": {},
        "branch_from": _CURRENT_VERSION_ID,
        "branch_name": version_name,
    }
    _VERSIONS[version_id] = version
    return {"version_id": version_id, "version": version}


@router.post("/version-control/versions/{version_id}/load")
async def load_version(version_id: str) -> Dict[str, Any]:
    global _CURRENT_VERSION_ID
    _find_version(version_id)
    _CURRENT_VERSION_ID = version_id
    return {"loaded": True, "current": _VERSIONS.get(_CURRENT_VERSION_ID)}


@router.post("/version-control/versions/{version_id}/branch")
async def branch_version(version_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _find_version(version_id)
    new_name = payload.get("new_version_name") or f"{version_id}-branch-{len(_VERSIONS) + 1}"
    new_version = {
        "id": new_name,
        "name": new_name,
        "timestamp": _now_iso(),
        "protocol_text": _VERSIONS[version_id]["protocol_text"],
        "comment": f"Branched from {version_id}",
        "author": "openevolve-api",
        "complexity_metrics": {},
        "structure_analysis": {},
        "branch_from": version_id,
        "branch_name": new_name,
    }
    _VERSIONS[new_name] = new_version
    return {"version_id": new_name, "version": new_version}


@router.post("/version-control/compare")
async def compare_versions(payload: Dict[str, Any]) -> Dict[str, Any]:
    v1 = _find_version(payload.get("version_id_1", ""))
    v2 = _find_version(payload.get("version_id_2", ""))
    t1 = v1.get("protocol_text", "")
    t2 = v2.get("protocol_text", "")
    added = len(t2) - len(set(t2) & set(t1))
    removed = len(t1) - len(set(t1) & set(t2))
    return {
        "version1": v1["id"],
        "version2": v2["id"],
        "chars_added": max(added, 0),
        "chars_removed": max(removed, 0),
        "total_chars_change": abs(len(t2) - len(t1)),
        "complexity_diff": {},
    }


@router.delete("/version-control/versions/{version_id}")
async def delete_version(version_id: str) -> Dict[str, Any]:
    if version_id == _DEFAULT_VERSION_ID:
        raise HTTPException(status_code=400, detail="Cannot delete the seeded baseline version.")
    if version_id not in _VERSIONS:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    del _VERSIONS[version_id]
    return {"deleted": True}
