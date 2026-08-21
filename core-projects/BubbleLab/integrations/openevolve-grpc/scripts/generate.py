#!/usr/bin/env python3
"""
OpenEvolve gRPC Python Code Generation

Cross-platform replacement for the Python half of ``generate.sh`` (which is bash
only and additionally requires ``protoc``/``npm`` for the TypeScript half).

This script only needs ``grpcio-tools`` (``python -m grpc_tools.protoc`` ships its
own protoc), so it runs on Windows without WSL/Git Bash.

Usage:
    python scripts/generate.py
    python -m scripts.generate --check   # fail if stubs are missing/stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = PROJECT_ROOT / "proto"
PYTHON_OUT = PROJECT_ROOT / "python" / "generated"

# Generated `*_pb2.py` files do `import common_pb2 as ...` which only resolves if
# the output dir is on sys.path. Rewrite to package-relative imports instead.
_ABS_IMPORT = re.compile(r"^import (\w+_pb2)( as (\w+))?$", re.MULTILINE)

PACKAGE_DOCSTRING = '''"""
Generated protobuf/gRPC stubs for the OpenEvolve gRPC integration.

Do not edit by hand -- regenerate with::

    python scripts/generate.py

Absolute ``import xxx_pb2`` statements emitted by protoc are rewritten to
relative imports so this directory works as a real Python package.
"""
'''


def proto_files() -> list[Path]:
    return sorted(PROTO_DIR.glob("*.proto"))


def fix_imports(out_dir: Path) -> int:
    """Rewrite protoc's absolute sibling imports to relative ones."""
    fixed = 0
    for path in sorted(out_dir.glob("*_pb2*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = _ABS_IMPORT.sub(
            lambda m: f"from . import {m.group(1)}"
            + (f" as {m.group(3)}" if m.group(3) else ""),
            text,
        )
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            fixed += 1
    return fixed


def write_package_init(out_dir: Path) -> None:
    modules = sorted(p.stem for p in out_dir.glob("*_pb2*.py"))
    lines = [PACKAGE_DOCSTRING, ""]
    for module in modules:
        lines.append(f"from . import {module}  # noqa: F401")
    lines.append("")
    lines.append("__all__ = [")
    for module in modules:
        lines.append(f"    {module!r},")
    lines.append("]")
    lines.append("")
    (out_dir / "__init__.py").write_text("\n".join(lines), encoding="utf-8")


def generate() -> int:
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError:
        print(
            "ERROR: grpcio-tools is not installed.\n"
            "       pip install grpcio grpcio-tools protobuf",
            file=sys.stderr,
        )
        return 1

    protos = proto_files()
    if not protos:
        print(f"ERROR: no .proto files found in {PROTO_DIR}", file=sys.stderr)
        return 1

    PYTHON_OUT.mkdir(parents=True, exist_ok=True)

    # grpcio-tools bundles the well-known types (timestamp/struct/any.proto);
    # they must be on the proto path or common.proto fails to resolve.
    wellknown = Path(grpc_tools.__file__).parent / "_proto"

    args = [
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--proto_path={wellknown}",
        f"--python_out={PYTHON_OUT}",
        f"--grpc_python_out={PYTHON_OUT}",
        f"--pyi_out={PYTHON_OUT}",
        *[str(p) for p in protos],
    ]

    print("Generating Python stubs for:")
    for p in protos:
        print(f"  {p.name}")

    rc = protoc.main(args)
    if rc != 0:
        print(f"ERROR: protoc exited {rc}", file=sys.stderr)
        return rc

    fixed = fix_imports(PYTHON_OUT)
    write_package_init(PYTHON_OUT)

    print(f"Rewrote imports in {fixed} file(s)")
    print(f"Python code generated in {PYTHON_OUT}")
    return 0


def check() -> int:
    """Verify stubs exist for every proto (used by tests / CI)."""
    missing = []
    for proto in proto_files():
        stem = proto.stem
        for suffix in ("_pb2.py", "_pb2_grpc.py"):
            if not (PYTHON_OUT / f"{stem}{suffix}").exists():
                missing.append(f"{stem}{suffix}")
    if missing:
        print("Missing generated stubs: " + ", ".join(missing), file=sys.stderr)
        print("Run: python scripts/generate.py", file=sys.stderr)
        return 1
    print("All Python stubs present.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that generated stubs exist",
    )
    ns = parser.parse_args()
    return check() if ns.check else generate()


if __name__ == "__main__":
    raise SystemExit(main())
