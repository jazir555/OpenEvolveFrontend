#!/usr/bin/env python3
"""
Generic Knowledge Extraction Tool (GKET) — FastAPI wrapper.

Thin HTTP surface over the existing library. Nothing under `ai/`, `core/`,
`parsers/`, `extraction/` is modified: every entry point is imported lazily so a
missing optional dependency (docling, torch, pandas...) degrades to a JSON
`note` instead of taking the server down.

Endpoints
    GET  /healthz                 -> {"status": "ok"}
    POST /parse                   -> {file_ref, parser: "fast"|"docling"}
    POST /generate-models         -> {text_description}
    POST /extract                 -> {case: 0|1|2, llm, text_or_file_ref, ...}
    GET  /export/{id}?format=...  -> stored /extract result as json|csv|xlsx

Run:  python server.py           (uvicorn on :8766)

KNOWN LIBRARY QUIRKS (worked around here, never by editing the library):
  1. `parsers/__init__.py` imports docling (needs langchain), and
     `extraction/__init__.py` / `utils/__init__.py` import names that do not
     exist (`Case1Classifier`, `MessagingSystem`). We register namespace stubs
     for those packages so the real submodules import cleanly.
  2. An injected module-level `try/except ImportError` sits inside several class
     bodies, so their methods ended up defined at module scope instead of on the
     class (e.g. `OpenAIExtractor.extract_batch`). `_bind_orphan_methods()`
     re-attaches them at runtime.
  3. `core/model_generator.py` is missing, but `extraction/case1_classifier.py`
     imports it. A minimal local `ModelGenerator` is registered under that name.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import io
import json
import logging
import os
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gket.server")

app = FastAPI(title="GKET API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result store: id -> {"id", "status", "records", "raw", "notes", ...}
RESULTS: Dict[str, Dict[str, Any]] = {}


# ==================== library compatibility layer ====================

# Packages whose real `__init__.py` cannot be executed (see quirk 1).
_PKG_STUBS: Dict[str, Path] = {
    "parsers": ROOT / "parsers",
    "extraction": ROOT / "extraction",
    "extraction.hierarchical": ROOT / "extraction" / "hierarchical",
    "utils": ROOT / "utils",
}

# Classes whose methods were orphaned to module scope (see quirk 2).
_COMPAT_TARGETS: List[Tuple[str, str]] = [
    ("ai.clients.openai_client", "OpenAIClient"),
    ("ai.clients.claude_client", "ClaudeClient"),
    ("ai.extractors.openai_extractor", "OpenAIExtractor"),
    ("ai.extractors.claude_extractor", "ClaudeExtractor"),
    ("parsers.document_parser", "DocumentParser"),
    ("core.text_description_client", "OpenAITextDescriptionClient"),
]

_COMPAT_NOTES: List[str] = []
_COMPAT_READY = False


def _install_package_stubs() -> None:
    """Expose real submodules without running the broken package `__init__`."""
    for name, path in _PKG_STUBS.items():
        if name in sys.modules or not path.is_dir():
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]  # type: ignore[attr-defined]
        stub.__package__ = name
        sys.modules[name] = stub
        parent, _, leaf = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], leaf, stub)


def _bind_orphan_methods(module: types.ModuleType, cls: type) -> int:
    """Re-attach module-level `self`-first functions onto their intended class."""
    bound = 0
    for name, func in list(vars(module).items()):
        if not inspect.isfunction(func) or name in cls.__dict__:
            continue
        try:
            params = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            continue
        if params[:1] != ["self"]:
            continue
        setattr(cls, name, func)
        bound += 1
    return bound


def _load(module_name: str, class_name: str) -> type:
    """Import a library class, repairing orphaned methods first."""
    _install_package_stubs()
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    _bind_orphan_methods(module, cls)
    return cls


def _register_model_generator() -> None:
    """Register a minimal `core.model_generator` when the real one is absent."""
    if "core.model_generator" in sys.modules:
        return
    try:
        importlib.import_module("core.model_generator")
        return
    except Exception:  # noqa: BLE001 - expected: the module is missing
        pass

    module = types.ModuleType("core.model_generator")
    module.ModelGenerator = _ModelGenerator  # type: ignore[attr-defined]
    sys.modules["core.model_generator"] = module
    try:
        setattr(importlib.import_module("core"), "model_generator", module)
    except Exception as exc:  # noqa: BLE001
        _COMPAT_NOTES.append(f"could not attach model_generator to core: {exc}")
    _COMPAT_NOTES.append(
        "core/model_generator.py is missing; using the server's local ModelGenerator shim"
    )


def _prepare_library() -> List[str]:
    """Idempotently apply every workaround; returns accumulated notes."""
    global _COMPAT_READY
    if _COMPAT_READY:
        return list(_COMPAT_NOTES)

    _install_package_stubs()
    for module_name, class_name in _COMPAT_TARGETS:
        try:
            _load(module_name, class_name)
        except Exception as exc:  # noqa: BLE001
            _COMPAT_NOTES.append(f"{module_name}.{class_name} unavailable: {exc}")
    _register_model_generator()
    _COMPAT_READY = True
    return list(_COMPAT_NOTES)


# ==================== request models ====================


class ParseRequest(BaseModel):
    file_ref: str
    parser: str = Field(default="fast", description='"fast" (PyMuPDF/docx) or "docling"')
    use_markdown: bool = True


class GenerateModelsRequest(BaseModel):
    text_description: str
    use_case: str = ""
    context: str = ""
    llm: str = "openai"


class ExtractRequest(BaseModel):
    # `model_schema` is part of the documented wire contract; opt out of
    # pydantic's protected `model_` namespace so it stays spelled that way.
    model_config = ConfigDict(protected_namespaces=())

    case: int = 0
    llm: str = "openai"
    text_or_file_ref: str = ""
    model_schema: Optional[Dict[str, Any]] = None
    instruction: Optional[str] = None
    use_case: str = "GketApiRun"


# ==================== helpers ====================


def _fail(message: str, detail: str = "", status: int = 200) -> JSONResponse:
    """Never raise out of a handler; always answer with a readable payload."""
    return JSONResponse(
        status_code=status,
        content={"status": "error", "error": message, "detail": detail},
    )


def _resolve_path(file_ref: str) -> Optional[Path]:
    """Resolve a file reference against CWD and the tool root."""
    if not file_ref:
        return None
    candidates = [Path(file_ref), ROOT / file_ref, ROOT / "data" / file_ref]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_file(file_ref: str, parser: str, use_markdown: bool = True) -> Dict[str, Any]:
    """Parse a document with DocumentParser (fast) or DoclingParser."""
    _prepare_library()
    path = _resolve_path(file_ref)
    if path is None:
        raise FileNotFoundError(f"File not found: {file_ref}")

    if parser == "docling":
        docling_parser = _load("parsers.docling_parser", "DoclingParser")
        return docling_parser().parse_document(str(path), use_markdown=use_markdown)

    document_parser = _load("parsers.document_parser", "DocumentParser")
    return document_parser().parse_document(str(path), use_markdown=use_markdown)


def _as_documents(text_or_file_ref: str, parser: str = "fast") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build the document dicts the extractors expect (file if it exists, else raw text)."""
    notes: List[str] = []
    path = _resolve_path(text_or_file_ref)
    if path is not None:
        try:
            return [_parse_file(str(path), parser)], notes
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            notes.append(f"parse failed ({exc}); falling back to raw text")

    text = text_or_file_ref or ""
    return (
        [
            {
                "file_path": "inline",
                "file_name": "inline.txt",
                "file_extension": ".txt",
                "text_content": text,
                "content_length": len(text),
                "word_count": len(text.split()),
                "parsing_method": "inline",
                "format_used": "text",
            }
        ],
        notes,
    )


_TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list[str]": List[str],
    "list": List[str],
    "array": List[str],
    "enum": str,
    "list[enum]": List[str],
    "object": Dict[str, Any],
}


def _fields_from_schema(model_schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept the shapes this repo produces: {fields}, {parsed_fields}, {extraction_config}, JSON Schema."""
    if not model_schema:
        return []
    schema = model_schema.get("extraction_config", model_schema)
    for key in ("fields", "parsed_fields"):
        if isinstance(schema.get(key), list):
            return schema[key]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        required = set(schema.get("required") or [])
        return [
            {
                "field_name": name,
                "field_type": (spec or {}).get("type", "str"),
                "description": (spec or {}).get("description", name),
                "required": name in required,
            }
            for name, spec in properties.items()
        ]
    return []


def _build_model(fields: List[Dict[str, Any]], model_name: str = "ExtractedData") -> type[BaseModel]:
    """Build a Pydantic model from parsed field configs (mirrors ParsedField semantics)."""
    definitions: Dict[str, Any] = {}
    for entry in fields:
        name = str(entry.get("field_name") or "").strip()
        if not name:
            continue
        annotation = _TYPE_MAP.get(str(entry.get("field_type", "str")).lower(), str)
        description = entry.get("description") or name
        if entry.get("required", True):
            definitions[name] = (annotation, Field(..., description=description))
        else:
            definitions[name] = (Optional[annotation], Field(default=None, description=description))

    if not definitions:
        definitions["extracted_text"] = (str, Field(default="", description="Free-form extraction result"))

    return create_model(model_name or "ExtractedData", **definitions)  # type: ignore[call-overload]


def _build_prompt(config: Dict[str, Any], fields: List[Dict[str, Any]]) -> str:
    """Build the extraction prompt the extractors expect (mirrors the templates/ prompts)."""
    lines = [
        "EXTRACTION TASK",
        f"Use case: {config.get('use_case', 'Document extraction')}",
        f"Description: {config.get('description', '')}",
        "",
        "EXTRACTION RULES — return one JSON object with exactly these keys:",
    ]
    for entry in fields:
        name = entry.get("field_name", "")
        detail = f"- {name} ({entry.get('field_type', 'str')}): {entry.get('description', '')}"
        if entry.get("enum_values"):
            detail += f" | allowed string values: {entry['enum_values']}"
        if not entry.get("required", True):
            detail += " | optional, use null when absent"
        lines.append(detail)
    lines.append("")
    lines.append("Enum fields must return the exact lowercase string value, not the enum member name.")
    return "\n".join(lines)


def _model_source(model_name: str, fields: List[Dict[str, Any]]) -> str:
    """Render simple Pydantic source for the generated model (persisted by case 1)."""
    lines = [
        "from pydantic import BaseModel, Field",
        "from typing import List, Optional",
        "",
        f"class {model_name}(BaseModel):",
        '    """Generated by the GKET server ModelGenerator shim."""',
    ]
    if not fields:
        lines.append('    extracted_text: str = Field(default="", description="Free-form extraction result")')
    for entry in fields:
        name = entry.get("field_name", "")
        type_name = str(entry.get("field_type", "str")).lower()
        annotation = {"list[str]": "List[str]", "list[enum]": "List[str]", "enum": "str"}.get(type_name, type_name)
        if annotation not in ("str", "int", "float", "bool", "List[str]"):
            annotation = "str"
        description = str(entry.get("description", name)).replace('"', "'")
        if entry.get("required", True):
            lines.append(f'    {name}: {annotation} = Field(..., description="{description}")')
        else:
            lines.append(f'    {name}: Optional[{annotation}] = Field(default=None, description="{description}")')
    return "\n".join(lines) + "\n"


class _ModelGenerator:
    """Local stand-in for the missing `core/model_generator.py`.

    Implements only the surface `extraction/case1_classifier.py` uses:
    `generate_models_from_config_data`, `save_generated_models`,
    `save_extraction_prompt`, `get_extraction_prompt`, `load_models_and_prompt`.
    Models are built deterministically from the parsed field config instead of
    being generated by an LLM.
    """

    def __init__(self, model_selection: str = "", api_config: Optional[Dict[str, Any]] = None):
        self.model_selection = model_selection
        self.api_config = api_config or {}
        self.extraction_prompt = ""

    def generate_models_from_config_data(self, config: Dict[str, Any]):
        extraction_config = config.get("extraction_config", config)
        fields = _fields_from_schema(config)
        model_name = extraction_config.get("main_model_name", "ExtractedData")
        model_name = "".join(ch for ch in str(model_name) if ch.isalnum()) or "ExtractedData"
        self.extraction_prompt = _build_prompt(extraction_config, fields)
        return _build_model(fields, model_name), _model_source(model_name, fields)

    def get_extraction_prompt(self) -> str:
        return self.extraction_prompt

    def save_generated_models(self, file_path: str, model_code: str) -> None:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        Path(file_path).write_text(model_code, encoding="utf-8")

    def save_extraction_prompt(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        Path(file_path).write_text(
            f'extraction_prompt = """{self.extraction_prompt}"""\n', encoding="utf-8"
        )

    def load_models_and_prompt(self, model_path: str, prompt_path: str):
        namespace: Dict[str, Any] = {}
        exec(Path(model_path).read_text(encoding="utf-8"), namespace)  # noqa: S102 - trusted local file
        model_class = next(
            (
                obj
                for name, obj in namespace.items()
                if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
            ),
            _build_model([]),
        )
        prompt_namespace: Dict[str, Any] = {}
        exec(Path(prompt_path).read_text(encoding="utf-8"), prompt_namespace)  # noqa: S102
        self.extraction_prompt = prompt_namespace.get("extraction_prompt", "")
        return model_class, self.extraction_prompt


def _make_extractor(llm: str):
    """Instantiate the requested extractor from `ai/extractors`."""
    _prepare_library()
    if str(llm).lower().startswith("claude") or str(llm).lower() == "anthropic":
        return _load("ai.extractors.claude_extractor", "ClaudeExtractor")()
    return _load("ai.extractors.openai_extractor", "OpenAIExtractor")()


def _flatten(record: Any) -> Dict[str, Any]:
    """Flatten one record one level deep so csv/xlsx export stays tabular."""
    if not isinstance(record, dict):
        return {"value": record}
    row: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                row[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (list, tuple)):
            row[key] = json.dumps(list(value), default=str)
        else:
            row[key] = value
    return row


def _to_records(payload: Any) -> List[Dict[str, Any]]:
    """Normalize the very different case 0/1/2 return shapes into a record list."""
    if isinstance(payload, list):
        return [r if isinstance(r, dict) else {"value": r} for r in payload]
    if not isinstance(payload, dict):
        return [{"value": payload}]

    # Case 1: {'results': {doc_type: {'results': [...]}}}
    results = payload.get("results")
    if isinstance(results, dict):
        records: List[Dict[str, Any]] = []
        for doc_type, bundle in results.items():
            rows = bundle.get("results") if isinstance(bundle, dict) else None
            for row in rows or []:
                records.append({"document_type": doc_type, **(row if isinstance(row, dict) else {"value": row})})
            if not rows and isinstance(bundle, dict) and bundle.get("error"):
                records.append({"document_type": doc_type, "error": bundle["error"]})
        return records
    if isinstance(results, list):
        return _to_records(results)

    # Case 2: consolidated results
    if isinstance(payload.get("consolidated_results"), list):
        return _to_records(payload["consolidated_results"])

    return [payload]


def _store(record_id: str, status: str, records: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    entry = {
        "id": record_id,
        "status": status,
        "records": records,
        "created_at": time.time(),
        **extra,
    }
    RESULTS[record_id] = entry
    return entry


# ==================== extraction cases ====================


def _extract_case0(req: ExtractRequest, documents: List[Dict[str, Any]], notes: List[str]) -> Any:
    """Single-type extraction: build a model, then extract_batch()."""
    extractor = _make_extractor(req.llm)
    fields = _fields_from_schema(req.model_schema)
    if not fields:
        notes.append("no model_schema supplied; using a generic single-field model")
    model_class = _build_model(fields, (req.model_schema or {}).get("main_model_name", "ExtractedData"))
    prompt = req.instruction or "Extract the fields defined by the provided schema from the document."
    return extractor.extract_batch(
        documents=documents,
        extraction_prompt=prompt,
        model_class=model_class,
        additional_instructions="",
    )


def _extract_case1(req: ExtractRequest, documents: List[Dict[str, Any]], notes: List[str]) -> Any:
    """Multi-type classification + routing via extraction/case1_classifier.py."""
    extractor = _make_extractor(req.llm)
    case1_extractor = _load("extraction.case1_classifier", "Case1Extractor")
    return case1_extractor(ai_client=getattr(extractor, "client", None)).extract_from_documents(
        documents=documents,
        extraction_description=req.instruction or "Extract the relevant fields for each document type.",
        use_case_name=req.use_case,
    )


def _extract_case2(req: ExtractRequest, documents: List[Dict[str, Any]], notes: List[str]) -> Any:
    """Hierarchical PO -> BOM extraction via extraction/hierarchical/case2_*.py."""
    adapter = _load("extraction.hierarchical.case2_ai_adapter", "Case2AIAdapter")
    orchestrator_class = _load("extraction.hierarchical.case2_main", "Case2Orchestrator")

    orchestrator = orchestrator_class(ai_client=adapter(_make_extractor(req.llm)))
    use_case_path = str(ROOT / "templates" / req.use_case)

    if not os.path.exists(os.path.join(use_case_path, "config.json")):
        created = orchestrator.create_new_use_case(
            description=req.instruction or "Extract purchase order items and their BOM details.",
            use_case_name=req.use_case,
            use_case_path=use_case_path,
        )
        if not created.get("success"):
            notes.append(f"case 2 use-case creation failed: {created.get('error')}")
            return created

    result = orchestrator.extract_documents(documents=documents, use_case_path=use_case_path)
    return {
        "use_case_name": getattr(result, "use_case_name", req.use_case),
        "consolidated_results": getattr(result, "consolidated_results", []),
        "stage_results": getattr(result, "stage_results", {}),
        "processing_metadata": getattr(result, "processing_metadata", {}),
    }


# ==================== endpoints ====================


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
def parse_document(req: ParseRequest):
    notes = _prepare_library()
    parser = req.parser if req.parser in ("fast", "docling") else "fast"
    try:
        parsed = _parse_file(req.file_ref, parser, req.use_markdown)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Parsing failed for {req.file_ref}", str(exc))

    return {
        "status": "ok",
        "parser": parser,
        "file_name": parsed.get("file_name"),
        "text": parsed.get("text_content", ""),
        "content_length": parsed.get("content_length", 0),
        "word_count": parsed.get("word_count", 0),
        "metadata": {k: v for k, v in parsed.items() if k != "text_content"},
        "notes": notes,
    }


@app.post("/generate-models")
def generate_models(req: GenerateModelsRequest):
    """text description -> field config -> Pydantic model schema (+ generated code when available)."""
    notes: List[str] = _prepare_library()
    try:
        parser_class = _load("core.text_description_parser", "TextDescriptionParser")
        model_selection = "claude-sonnet-4-20250514" if "claude" in req.llm.lower() else "gpt-4.1-2025-04-14"
        config = parser_class(model_selection=model_selection).parse_extraction_description(
            description=req.text_description,
            use_case=req.use_case,
            context=req.context,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("Text-description parsing failed", str(exc))

    extraction_config = config.get("extraction_config", config)
    fields = _fields_from_schema(config)
    model_name = extraction_config.get("main_model_name", "ExtractedData")

    json_schema: Dict[str, Any] = {}
    try:
        json_schema = _build_model(fields, model_name).model_json_schema()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"could not build Pydantic model: {exc}")

    model_code = ""
    try:
        generator_class = _load("core.model_generator", "ModelGenerator")
        _, model_code = generator_class(model_selection=req.llm).generate_models_from_config_data(config)
    except Exception as generator_error:  # noqa: BLE001
        notes.append(f"ModelGenerator unavailable ({generator_error}); using ai/clients fallback")
        try:
            client_class = _load("ai.clients.openai_client", "OpenAIClient")
            model_code = client_class().generate_pydantic_models(extraction_config)
        except Exception as client_error:  # noqa: BLE001
            notes.append(f"generate_pydantic_models failed: {client_error}")

    return {
        "status": "ok",
        "model_name": model_name,
        "fields": fields,
        "json_schema": json_schema,
        "model_code": model_code,
        "config": config,
        "notes": notes,
    }


@app.post("/extract")
def extract(req: ExtractRequest):
    """Run case 0 (single-type), 1 (classification routing) or 2 (hierarchical)."""
    record_id = uuid.uuid4().hex[:12]
    notes = _prepare_library()
    documents, parse_notes = _as_documents(req.text_or_file_ref)
    notes = notes + parse_notes

    runners = {0: _extract_case0, 1: _extract_case1, 2: _extract_case2}
    runner = runners.get(req.case)
    if runner is None:
        return _fail(f"Unsupported case: {req.case}", "case must be 0, 1 or 2")

    try:
        raw = runner(req, documents, notes)
        records = _to_records(raw)
        status = "completed"
    except Exception as exc:  # noqa: BLE001
        raw = {"error": str(exc)}
        records = []
        status = "failed"
        notes.append(f"case {req.case} extraction failed: {exc}")

    entry = _store(
        record_id,
        status,
        records,
        case=req.case,
        llm=req.llm,
        raw=raw if isinstance(raw, (dict, list)) else str(raw),
        notes=notes,
    )
    return {
        "id": entry["id"],
        "status": entry["status"],
        "records": entry["records"],
        "case": req.case,
        "notes": notes,
    }


@app.get("/export/{result_id}")
def export_result(result_id: str, format: str = Query("json", pattern="^(csv|json|xlsx)$")):
    """Return a stored /extract result, optionally serialized to csv or xlsx."""
    entry = RESULTS.get(result_id)
    if entry is None:
        return _fail(f"Unknown result id: {result_id}", "run POST /extract first", status=404)

    records: List[Dict[str, Any]] = entry.get("records") or []
    rows = [_flatten(record) for record in records]

    if format == "json":
        return {
            "id": entry["id"],
            "status": entry["status"],
            "case": entry.get("case"),
            "records": records,
            "raw": entry.get("raw"),
            "notes": entry.get("notes", []),
        }

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns or ["value"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in (columns or ["value"])})
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{result_id}.csv"'},
        )

    try:
        import pandas as pd

        buffer = io.BytesIO()
        pd.DataFrame(rows, columns=columns or None).to_excel(buffer, index=False)
        return Response(
            content=buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{result_id}.xlsx"'},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("xlsx export unavailable (pandas/openpyxl required)", str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766)
