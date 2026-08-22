"""
BubbleLab Components Bridge

Wires the Python "BubbleLab UI" application to the TypeScript
``@openevolve/bubblelab-components`` package so every configuration knob listed
in the component manifest is reachable from Python.

This is the concrete integration point called out as "not implemented" in
``docs/Architecture/BubbleLab/BUBBLELABS_INTEGRATION.md``. The TS package ships a
single source of truth, ``src/component-manifest.json`` (compiled into its
``dist``), describing every knob. This bridge:

  * ``discover()``  - locate the TS package and read its ``package.json`` and
                      ``component-manifest.json``.
  * ``get_manifest()`` / ``get_all_knobs()`` - expose the components and every
                      config knob to the Python side.
  * ``build()``     - run the TS ``tsc`` build (``build:components``) via npm.
  * ``serve()``     - serve the built ``dist/`` over HTTP, or a fallback notice
                      page when the package is not built.
  * ``status()``    - degraded-mode report when node/npm/dist are missing.

Graceful degradation: every method tolerates a missing TS toolchain or an
unbuilt package and reports the situation instead of raising.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["BubbleLabComponentsBridge", "get_bridge"]

_PACKAGE_NAME = "@openevolve/bubblelab-components"
_DEFAULT_PACKAGE_REL = Path("glue") / "adapters" / "bubblelab"


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the repo root that contains the TS package."""
    for parent in [start, *start.parents]:
        if (parent / _DEFAULT_PACKAGE_REL / "package.json").is_file():
            return parent
        if (parent / ".git").is_dir() and (parent / "glue").is_dir():
            return parent
    # Fall back to two levels up (repo root when colocated under engines/other).
    return start.parents[2] if len(start.parents) > 2 else start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


class _ManifestHandler(BaseHTTPRequestHandler):
    """Serve the built ``dist/`` directory, or a fallback notice page."""

    server_version = "BubbleLabComponentsBridge/1.0"
    # Set per-server by ComponentsServer.
    root: Path = Path(".")
    built: bool = False

    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self.built:
            html = _FALLBACK_HTML.format(
                package=self.server.package_name,
                port=self.server.server_address[1],
            ).encode("utf-8")
            self._send(200, html)
            return

        rel = self.path.lstrip("/").split("?", 1)[0] or "index.html"
        candidate = (self.root / rel).resolve()
        # Prevent path traversal outside the served root.
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            self._send(403, b"Forbidden")
            return

        if candidate.is_dir():
            candidate = candidate / "index.html"
        if candidate.is_file():
            data = candidate.read_bytes()
            ctype = _guess_content_type(candidate)
            self._send(200, data, ctype)
        else:
            self._send(404, b"Not found")

    def log_message(self, *args: Any) -> None:  # silence default stderr logging
        return


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        "html": "text/html; charset=utf-8",
        "js": "text/javascript; charset=utf-8",
        "mjs": "text/javascript; charset=utf-8",
        "css": "text/css; charset=utf-8",
        "json": "application/json; charset=utf-8",
        "map": "application/json; charset=utf-8",
        "svg": "image/svg+xml",
        "ico": "image/x-icon",
    }.get(suffix.lstrip("."), "application/octet-stream")


_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{package}</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 4rem auto;">
  <h1>{package}</h1>
  <p>The TypeScript components package is reachable from Python but has not been
     built yet, so there is no <code>dist/</code> to serve.</p>
  <p>To build and serve the components from the BubbleLab UI:</p>
  <pre>bridge.build()   # runs `npm run build:components` (tsc)
bridge.serve(port={port})</pre>
  <p>Until then, the BubbleLab UI still exposes every configuration knob via the
     component manifest (<code>get_all_knobs()</code>).</p>
</body></html>"""


class BubbleLabComponentsBridge:
    """Locate, inspect, build, and serve the TS ``@openevolve/bubblelab-components`` package."""

    def __init__(self, package_dir: Optional[Any] = None) -> None:
        self.package_path: Optional[Path] = self._resolve_package_dir(package_dir)
        self.manifest: Optional[Dict[str, Any]] = None
        self.package_meta: Optional[Dict[str, Any]] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        if self.package_path is not None:
            self.discover()

    # ------------------------------------------------------------------ locate
    @staticmethod
    def _resolve_package_dir(package_dir: Optional[Any]) -> Optional[Path]:
        env = os.environ.get("BUBBLELAB_COMPONENTS_DIR")
        candidates: List[Path] = []
        if package_dir is not None:
            candidates.append(Path(package_dir))
        if env:
            candidates.append(Path(env))
        candidates.append(_REPO_ROOT / _DEFAULT_PACKAGE_REL)
        for cand in candidates:
            cand = cand.resolve()
            if cand.is_dir() and (cand / "package.json").is_file():
                return cand
        # Fall back to the last candidate (may not exist) so callers can still
        # report a useful degraded status.
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------ inspect
    def is_available(self) -> bool:
        if self.package_path is None:
            return False
        if not (self.package_path / "package.json").is_file():
            return False
        meta = self.package_meta or self._read_json(self.package_path / "package.json")
        return bool(meta and meta.get("name") == _PACKAGE_NAME)

    def has_manifest(self) -> bool:
        return self.manifest is not None

    def has_dist(self) -> bool:
        return self.package_path is not None and (self.package_path / "dist").is_dir()

    def discover(self) -> Dict[str, Any]:
        """Read ``package.json`` and ``component-manifest.json`` (best effort)."""
        self.package_meta = None
        self.manifest = None
        if self.package_path is None:
            return {}
        pkg_file = self.package_path / "package.json"
        manifest_file = self.package_path / "src" / "component-manifest.json"
        if pkg_file.is_file():
            self.package_meta = self._read_json(pkg_file)
        # Prefer the source manifest (always present after this wiring); fall back
        # to a copy emitted into dist by the build.
        if manifest_file.is_file():
            self.manifest = self._read_json(manifest_file)
        elif self.package_path.joinpath("dist", "component-manifest.json").is_file():
            self.manifest = self._read_json(self.package_path / "dist" / "component-manifest.json")
        return {
            "available": self.is_available(),
            "has_manifest": self.has_manifest(),
            "has_dist": self.has_dist(),
            "package": self.package_meta,
            "manifest": self.manifest,
        }

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------- manifest API
    def get_manifest(self) -> Optional[Dict[str, Any]]:
        return self.manifest

    def get_components(self) -> List[Dict[str, Any]]:
        if not self.manifest:
            return []
        return list(self.manifest.get("components", []))

    def get_all_knobs(self) -> List[Dict[str, Any]]:
        """Every config knob across all components, de-duplicated by id."""
        if not self.manifest:
            return []
        by_id: Dict[str, Dict[str, Any]] = {}
        for component in self.manifest.get("components", []):
            for knob in component.get("knobs", []):
                by_id[knob["id"]] = knob
        return list(by_id.values())

    def get_knob(self, knob_id: str) -> Optional[Dict[str, Any]]:
        return next((k for k in self.get_all_knobs() if k["id"] == knob_id), None)

    def get_component_knob_ids(self, component_id: str) -> List[str]:
        for component in self.get_components():
            if component.get("id") == component_id:
                return [k["id"] for k in component.get("knobs", [])]
        return []

    # ------------------------------------------------------------------ build
    def build(self, script: str = "build:components") -> Dict[str, Any]:
        """
        Run the TS ``tsc`` build via npm (best effort).

        Returns a status dict; never raises on missing tooling.
        """
        if self.package_path is None:
            return {"success": False, "reason": "package directory not located"}
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm is None:
            return {
                "success": False,
                "reason": "npm not found on PATH",
                "degraded": True,
            }
        try:
            proc = subprocess.run(
                [npm, "run", script],
                cwd=str(self.package_path),
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "reason": "build timed out", "degraded": True}
        except OSError as exc:
            return {"success": False, "reason": f"build failed to start: {exc}", "degraded": True}
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    # ------------------------------------------------------------------ serve
    def serve(self, host: str = "127.0.0.1", port: int = 4174, daemon: bool = True) -> Dict[str, Any]:
        """
        Serve the built TS components over HTTP.

        Serves ``dist/`` when present; otherwise serves a fallback notice page so
        the BubbleLab UI always has a working endpoint. Returns a status dict with
        the URL; call :meth:`stop` to shut the server down.
        """
        if self.package_path is None:
            return {"success": False, "reason": "package directory not located"}
        root = self.package_path / "dist"
        built = root.is_dir()
        server = ThreadingHTTPServer((host, port), _ManifestHandler)
        server.root = root
        server.built = built
        server.package_name = _PACKAGE_NAME
        thread = threading.Thread(target=server.serve_forever, daemon=daemon)
        thread.start()
        self._server = server
        self._server_thread = thread
        return {
            "success": True,
            "url": f"http://{host}:{port}",
            "built": built,
            "root": str(root),
            "degraded": not built,
            "mode": "dist" if built else "fallback-notice",
        }

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._server_thread = None

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        scripts = (self.package_meta or {}).get("scripts", {})
        return {
            "package_name": _PACKAGE_NAME,
            "package_dir": str(self.package_path) if self.package_path else None,
            "available": self.is_available(),
            "has_manifest": self.has_manifest(),
            "has_dist": self.has_dist(),
            "component_count": len(self.get_components()),
            "knob_count": len(self.get_all_knobs()),
            "serving": self._server is not None,
            "build_script_configured": bool("build:components" in scripts),
        }


_bridge_singleton: Optional[BubbleLabComponentsBridge] = None


def get_bridge() -> BubbleLabComponentsBridge:
    """Return a process-wide :class:`BubbleLabComponentsBridge` singleton."""
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = BubbleLabComponentsBridge()
    return _bridge_singleton


if __name__ == "__main__":
    b = BubbleLabComponentsBridge()
    print(json.dumps(b.status(), indent=2))
    if b.has_manifest():
        print(f"\nComponents: {len(b.get_components())}  Knobs: {len(b.get_all_knobs())}")
