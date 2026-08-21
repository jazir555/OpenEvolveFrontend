# Actual Status — openevolve-grpc

**Assessed:** 2026-08-20 (independent verification)
**Updated:** 2026-08-20 (Python server/client implemented — see "Python gRPC pass" below)
**Verdict:** ⚠️ **Not production-ready, but no longer a skeleton.** Proto contracts,
the TypeScript client shell, and now the **Python server + client work end-to-end**:
the NodeRegistry servicer is registered and all 7 RPCs return real data over the
wire. Remaining gaps are security/packaging/perf, not "nothing is wired up".

This document supersedes `CERTIFICATION.md`, `VERIFICATION_COMPLETE.md`,
`FINAL_VERIFICATION_REPORT.md`, `PRODUCTION_READY_VERIFICATION.md`,
`PRODUCTION_VERIFICATION_REPORT.md`, `DEEP_VERIFICATION_REPORT.md` and
`VERIFICATION_REPORT.md`, all of which claim "CERTIFIED FOR PRODUCTION" /
"deploy immediately". Those documents verified only **syntax** (`py_compile`,
JSON parse validity) and never compiled the TypeScript, ran a test, or exercised a
single RPC. Their "Sign-off" table is signed by an "Automated Verification System"
with no named reviewer.

---

## What was actually verified (reproducible)

| Check | Command | Result |
|---|---|---|
| TS dependency install | `npm ci` (in `typescript/`) | ✅ 416 packages, clean |
| TS typecheck (strict) | `npx tsc --noEmit` | ✅ exit 0, 0 errors |
| TS build | `npx tsc` | ✅ emits `dist/client.js` |
| TS unit tests | `npm test` | ✅ 10/10 pass |
| Proto load at runtime | client construction | ✅ all 7 protos load |
| Proto ↔ client contract | manual diff | ✅ RPC names match `nodes.proto` |
| Python syntax | `py_compile` × 6 | ✅ all valid |
| Python runtime import | `import client` | ✅ imports |
| **Python stub generation** | `python scripts/generate.py` | ✅ 7 protos → `python/generated/` |
| **Live Python e2e** | `python -m pytest` | ✅ **39 passed, 1 skipped** |
| **Live server + CLI client** | `python server.py` + `python client.py` | ✅ lists 3 nodes, streams progress, returns result |

`tsconfig.json` is genuinely strict (`strict`, `noUnusedLocals`, `noUnusedParameters`,
`noImplicitAny`, `strictNullChecks`) and `client.ts` passes it with zero errors.

---

## Bugs found and fixed during this pass

1. **Client was unusable when consumed as a package (critical).**
   `loadProto()` resolved `path.join(__dirname, '..', 'proto')`. That is correct when
   running from source (`typescript/`) but wrong from the compiled output
   (`typescript/dist/`), which is exactly what `package.json` ships as `main`.
   Every `new OpenEvolveGRPCClient()` threw
   `ENOENT ... typescript/proto/common.proto` **in the constructor**, before any
   network activity. Fixed with `resolveProtoDir()`, which probes both layouts and
   accepts an explicit `protoDir` config override, and throws a descriptive error
   instead of a bare ENOENT. Regression-tested.

2. **Opaque failure when calling an RPC before `connect()`.**
   `getChannel()` indexed an empty pool and computed `index % 0` → `NaN`, producing
   `TypeError: cannot read property of undefined`. Now throws
   `"No gRPC channel available: call connect() before issuing requests."` Tested.

3. **Zero tests despite "integration tests included" / "unit test structure in place".**
   `npm test` ran `jest` against 0 test files and **exited 1**. Added `jest.config.js`
   (ts-jest) and `client.test.ts` with 10 offline tests covering proto discovery,
   stub extraction, config defaulting, and the two fixes above.

No application logic was rewritten, and no fake server/mock was introduced to make
anything "pass".

---

## Python gRPC pass (gaps 1–4 below are now closed)

The four blockers that made every application RPC return `UNIMPLEMENTED` are fixed.
The registry is real, in-memory, and served over a real channel.

1. **Generated stubs now exist.** Added `scripts/generate.py`, a cross-platform
   Python-only generator (`grpcio-tools` ships its own `protoc`, so no WSL/Git Bash,
   no `npm`, no global `grpc-tools`). It also puts grpcio-tools' bundled well-known
   types on the proto path and rewrites protoc's `import common_pb2` to
   `from . import common_pb2`, so `python/generated/` is a real package.
   `scripts/generate.sh` is left in place for the TypeScript half.

2. **Two protos did not compile.** `nodes.proto` and `gauntlet.proto` both used
   `google.protobuf.Timestamp` **without importing `timestamp.proto`**. `protoc`
   rejects this; the earlier reports called `proto/` "solid" because nothing ever
   compiled it. Added the missing imports. (TS still typechecks and passes 10/10.)

3. **Servicer is registered.** `add_NodeRegistryServicer_to_server` is no longer
   commented out. `OpenEvolveServicer` now subclasses the generated
   `NodeRegistryServicer` and **returns protobuf messages instead of dicts** — the
   old methods returned plain dicts, which would have failed to serialize even if
   registration had been enabled. All 7 RPCs are implemented, including
   `ExecuteBatch`, which the servicer was missing entirely.

4. **`bubblelabs_nodes` is wired up.** The old `sys.path` hack pointed at
   `core-projects/bubblelabs_nodes` (wrong depth, and it appended the package dir
   rather than its parent). The real package lives at the monorepo root; the path is
   now computed correctly and `OPENEVOLVE_USE_REAL_NODES=1` merges its registry
   (verified: adds the real `causal_analysis` node). Because that import pulls in the
   whole `openevolve` stack (~20 s, and only one node is registered in this
   checkout), it is **opt-in**; `python/local_nodes.py` provides dependency-free seed
   nodes (`echo`, `decomposition`, `semantic_search`) implementing the same node
   contract so the server always serves real data offline.

5. **Python client is implemented.** All 17 placeholder returns are gone.
   `list_nodes()` issues a real `ListNodes` RPC and maps `NodeInfo` messages;
   `execute_node`, `execute_node_streaming`, `execute_batch`, `cancel_execution`,
   `get_execution_status` and `check_health` all call real stubs. `_wait_for_ready()`
   uses `grpc.channel_ready_future()` instead of `asyncio.sleep(0.1)`. Blocking stub
   calls are dispatched to an executor and streaming responses are pumped from a
   worker thread, so the existing `async` API (used by `rest_bridge.py`) is preserved.

6. **Other real defects fixed while wiring this up.**
   - `python/__init__.py` imported `Client`, `RestBridge`, `Server`, `ServiceMesh`,
     `TestIntegration` — **none of which exist**, so the package could never be
     imported. Now exports the real names, with `rest_bridge` guarded (fastapi/uvicorn
     are optional).
   - `HealthServicer.Check` returned `status="SERVING"` (a string) into an enum field.
     It now uses this repo's own generated `grpc.health.v1` stubs, which also avoids
     needing `grpcio-health-checking` and avoids registering two copies of
     `grpc.health.v1` in the descriptor pool.
   - `ExecutionManager` used an `asyncio.Lock` reached through a `_run_async()` helper
     that spawned a fresh event loop (or a thread) per call, from a **thread-based**
     sync servicer. Converted to `threading.RLock`, and completed executions are kept
     in a bounded history so `GetExecutionStatus` still answers after completion.
   - Several error paths ended in a bare `raise` with no active exception (a
     guaranteed `RuntimeError`). Replaced with `context.abort(...)`.
   - `ExecuteNodeStreaming` nested async generators inside a manually driven event
     loop and used `context.add_callback()` as a liveness probe. Rewritten as a plain
     sync generator with a worker thread and a `queue.Queue`, which is what the sync
     gRPC server expects.
   - Server can now bind port 0 (OS-assigned) and `start(block=False)` returns the
     bound port, which is what makes an in-process e2e test possible.

### New test

`python/test_grpc_e2e.py` starts a real server on an ephemeral port in a background
thread and drives it through the real client: `ListNodes` (plus category filter),
`GetNodeSchema`, `ExecuteNode` (success, validation failure, unknown node →
`NOT_FOUND`), `ExecuteNodeStreaming` (including intermediate progress),
`ExecuteBatch`, `GetExecutionStatus`, `CancelExecution` and the health check. It is
fully offline and needs no `pytest-asyncio`. One test asserts `ListNodes` returns a
non-empty list specifically to catch a regression back to the commented-out
registration.

`pytest.ini` was added because pytest otherwise walks up into the unrelated, broken
`core-projects/BubbleLab/__init__.py` package chain; it also disables an unrelated
broken global plugin (`web3`'s `pytest_ethereum`) that aborts pytest startup in this
environment.

### Reproduce

```bash
cd core-projects/BubbleLab/integrations/openevolve-grpc
pip install grpcio grpcio-tools protobuf     # only grpcio-tools was missing here
python scripts/generate.py                   # writes python/generated/
python -m pytest                             # 39 passed, 1 skipped

# or run it for real:
python python/server.py                      # then, in another shell:
python python/client.py                      # lists 3 nodes, streams progress
```

Environment note: installing `grpcio-tools` unpinned pulls `protobuf` 7.x, which
breaks other packages in this shared interpreter (streamlit, opentelemetry, google-*).
Versions were pinned back to the pre-existing `grpcio==1.67.1` / `protobuf==5.29.5`;
only `grpcio-tools` was added.

---

## Remaining gaps (still open)

1. **`grpcio-reflection` is not installed**, so server reflection is skipped at
   startup (logged as a warning, not a failure). `grpcio-health-checking` is
   deliberately *not* required — health is served from this repo's `health.proto`.

2. **TypeScript ↔ Python wire compatibility is still unproven.** The Python client
   and server are now verified against each other, and `client.ts` typechecks and
   passes unit tests, but no TS→Python call has been exercised. The enum mapping the
   Python side uses (`"decomposition"` ↔ `NODE_TYPE_DECOMPOSITION`, centralised in
   `python/proto_mapping.py`) is the most likely place for a mismatch.

3. **Packaging bug (will break publish).** `package.json` has
   `files: ["dist", "generated"]`, which excludes the `.proto` files the client loads
   at runtime. `proto/` also sits *outside* the package root, so it cannot be added
   via `files`. A published tarball would reproduce bug #1. Fixing this needs a
   `prepack` copy step or a repo restructure — a build-system decision, so it was
   left alone rather than guessed at.

4. **`typescript/generated/` is still absent.** Only the Python half of code
   generation was run; the TS client uses runtime proto loading, so it does not need
   the generated stubs today.

5. **Unverified performance claims.** "5-10x faster latency", "10x throughput",
   "1000+ req/s" appear as fact in the reports; the docs' own tables mark them
   "⚠️ Pending test". Nothing was benchmarked.

6. **Security items still open** (the reports concede this): TLS/mTLS,
   authentication, and CORS are all "add for production". The server binds
   `add_insecure_port` only.

7. **Node coverage is thin.** The proto enumerates ~60 `NodeType` values; the real
   `bubblelabs_nodes` registry registers exactly **one** (`causal_analysis`), and the
   local seed module adds three. `ListNodes` is real but short.

---

## Honest readiness assessment

- **`proto/` — solid, and now actually compiles.** 7 proto3 schemas; service/RPC
  names match what both clients call. Two missing well-known-type imports were fixed.
- **`typescript/client.ts` — compiles, builds, unit-tested, and is genuinely wired**
  (it calls real stubs). Not yet confirmed against a live server.
- **`python/` — implemented and e2e-tested.** Server registers the NodeRegistry
  servicer and returns real protobuf messages; the client makes real RPCs. Verified
  in-process (pytest) and out-of-process (`server.py` + `client.py`).

**Bottom line:** the integration is now a **working gRPC NodeRegistry**, suitable for
development and for wiring the TS client against. It is still not production-ready:
no TLS, no auth, unproven cross-language interop, an unresolved npm packaging bug,
and no benchmarks behind the performance claims.

