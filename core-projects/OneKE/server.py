"""
OneKE FastAPI backend wrapper.

Wraps the OneKE CLI / pipeline (core-projects/OneKE/src/run.py) so the
BubbleLab TS UI can drive knowledge extraction over HTTP.

Design:
- This module only depends on FastAPI + Pydantic + stdlib, so it stays
  importable and runnable even when the heavy OneKE dependencies
  (torch, langchain, sentence-transformers, ...) are not installed.
- Extraction is delegated to ``python src/run.py --config <yaml>`` via a
  subprocess. This reuses the real pipeline entrypoint unchanged (we do
  NOT modify existing src/ logic) and isolates OneKE's import side effects
  from this server process.
- All endpoints are wrapped in error handling: bad input or a failing
  extraction returns a structured error payload instead of crashing.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ONEKE_DIR = Path(__file__).resolve().parent
SRC_DIR = ONEKE_DIR / "src"
RESULTS_DIR = SRC_DIR / "examples" / "results"
SCHEMA_REPO = SRC_DIR / "modules" / "knowledge_base" / "schema_repository.py"
CASE_REPO = SRC_DIR / "modules" / "knowledge_base" / "case_repository.json"

app = FastAPI(title="OneKE API", version="1.0.0")

# In-memory store of past results, keyed by extraction id.
_results: Dict[str, Dict[str, Any]] = {}

VALID_TASKS = ["NER", "RE", "EE", "Triple", "Base"]


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ExtractRequest(BaseModel):
    task: str = Field(default="NER", description="NER / RE / EE / Triple / Base")
    mode: str = Field(default="quick", description="quick / agent / customized")
    config_yaml: Optional[str] = Field(
        default=None, description="Full OneKE YAML config (overrides generated config)"
    )
    text: Optional[str] = Field(default=None, description="Raw input text")
    file_ref: Optional[str] = Field(default=None, description="Path to an input file")
    instruction: Optional[str] = Field(default=None, description="Task instruction")
    constraint: Optional[str] = Field(default=None, description="Schema constraint")
    model: Optional[Dict[str, Any]] = Field(
        default=None, description="OneKE model config block"
    )
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    construct: Optional[Dict[str, Any]] = Field(
        default=None, description="Knowledge-graph construct config (Neo4j)"
    )


class ExtractResult(BaseModel):
    id: str
    answer_json: Any = None
    schema: str = ""
    triples: List[Any] = Field(default_factory=list)
    status: str
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _list_schema_names() -> List[str]:
    """Read schema class names from src schema_repository.py (no import)."""
    if not SCHEMA_REPO.exists():
        return []
    try:
        tree = ast.parse(SCHEMA_REPO.read_text(encoding="utf-8"))
        names: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = (
                        base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                    )
                    if base_name == "BaseModel":
                        names.append(node.name)
                        break
        return names
    except Exception:
        return []


def _build_config(req: ExtractRequest) -> Dict[str, Any]:
    if req.config_yaml:
        import yaml  # local import; only needed when a YAML config is supplied

        return yaml.safe_load(req.config_yaml)

    model_cfg = req.model or {
        "vllm_serve": False,
        "category": "openai",
        "model_name_or_path": req.model_name
        or os.environ.get("ONEKE_MODEL", "gpt-4o-mini"),
        "api_key": req.api_key or os.environ.get("ONEKE_API_KEY", ""),
        "base_url": req.base_url or os.environ.get("ONEKE_BASE_URL", ""),
    }

    extraction = {
        "task": req.task if req.task in VALID_TASKS else "NER",
        "instruction": req.instruction or "",
        "text": req.text or "",
        "output_schema": "",
        "constraint": req.constraint or "",
        "use_file": bool(req.file_ref),
        "file_path": req.file_ref or "",
        "truth": "",
        "mode": req.mode or "quick",
        "update_case": False,
        "show_trajectory": False,
    }

    config: Dict[str, Any] = {"model": model_cfg, "extraction": extraction}
    if req.construct:
        config["construct"] = req.construct
    return config


def _extract_schema_and_answer(stdout: str, result_file: Optional[Path]):
    schema_str = ""
    answer_json: Any = None

    # Best-effort parse of the saved JSON result file.
    if result_file and result_file.exists():
        try:
            answer_json = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            answer_json = None

    # Fallback: parse the "Extraction Result:" block printed to stdout.
    if answer_json is None:
        marker = "Extraction Result:"
        idx = stdout.find(marker)
        if idx != -1:
            block = stdout[idx + len(marker):].strip()
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("{") or line.startswith("["):
                    try:
                        answer_json = json.loads(line)
                        break
                    except Exception:
                        continue

    # Capture the generated schema printed to stdout.
    sidx = stdout.find("Schema:")
    if sidx != -1:
        tail = stdout[sidx + len("Schema:"):].strip()
        schema_str = tail.splitlines()[0] if tail else ""

    return schema_str, answer_json


def _extract_triples(task: str, answer_json: Any) -> List[Any]:
    if not isinstance(answer_json, dict):
        return []
    if task == "Triple":
        return answer_json.get("triple_list", []) or []
    return []


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schemas")
def list_schemas() -> List[str]:
    return _list_schema_names()


@app.get("/cases")
def list_cases() -> List[str]:
    if CASE_REPO.exists():
        try:
            data = json.loads(CASE_REPO.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return list(data.keys())
        except Exception:
            return []
    return []


@app.post("/extract", response_model=ExtractResult)
def extract(req: ExtractRequest) -> ExtractResult:
    result_id = uuid.uuid4().hex
    try:
        config = _build_config(req)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", dir=str(ONEKE_DIR), delete=False, encoding="utf-8"
        ) as fh:
            import yaml  # local import (already imported above if config_yaml set)

            yaml.safe_dump(config, fh, allow_unicode=True)
            cfg_path = fh.name

        try:
            proc = subprocess.run(
                [sys.executable, "src/run.py", "--config", cfg_path],
                cwd=str(ONEKE_DIR),
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("ONEKE_TIMEOUT", "1800")),
            )
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass

        if proc.returncode != 0:
            payload = ExtractResult(
                id=result_id,
                status="error",
                error=(proc.stderr or proc.stdout or "Extraction failed").strip()[-2000:],
            )
            _results[result_id] = payload.model_dump()
            return payload

        base = Path(cfg_path).stem
        result_file = RESULTS_DIR / f"{base}.json"
        schema_str, answer_json = _extract_schema_and_answer(proc.stdout, result_file)
        triples = _extract_triples(req.task, answer_json)

        payload = ExtractResult(
            id=result_id,
            answer_json=answer_json,
            schema=schema_str,
            triples=triples,
            status="success",
        )
        _results[result_id] = payload.model_dump()
        return payload

    except Exception as exc:  # never crash the server
        payload = ExtractResult(id=result_id, status="error", error=str(exc))
        _results[result_id] = payload.model_dump()
        return payload


@app.get("/result/{result_id}", response_model=ExtractResult)
def get_result(result_id: str) -> ExtractResult:
    item = _results.get(result_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return ExtractResult(**item)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
