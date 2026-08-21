# Actual Status — openevolve-grpc

**Assessed:** 2026-08-20 (independent verification)
**Verdict:** ⚠️ **NOT production-ready.** Proto contracts + TypeScript client shell are real
and now build/test cleanly. The **Python server never registers its servicer** and the
**Python client is stubbed**, so no application RPC works end-to-end.

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
| **Live e2e** | — | ❌ **not possible, see below** |

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

## Remaining gaps (NOT fixed — these are real blockers)

1. **The gRPC server serves no application RPCs.** In `python/server.py` `start()`,
   the servicer registration is commented out:
   ```python
   # add_NodeRegistryServicer_to_server(self.servicer, self.server)
   ```
   Only health and reflection are registered. Any `ListNodes` / `ExecuteNode` call
   returns `UNIMPLEMENTED`. This alone invalidates the production claim.

2. **The Python client is a stub, not an implementation.** `python/client.py` has the
   protobuf imports commented out (`# from generated import nodes_pb2...`), never
   constructs a stub (`# self.stub = nodes_pb2_grpc.NodeRegistryStub(...)`),
   `list_nodes()` returns `[]`, and `_wait_for_ready()` is `await asyncio.sleep(0.1)`.
   17 placeholder markers across the file.

3. **Generated protobuf stubs do not exist.** Neither `python/generated/` nor
   `typescript/generated/` is present; `scripts/generate.sh` has never been run here.
   `tsconfig.json` already includes `generated/**/*.ts`, so that path is inert.
   `generate.sh` is bash and requires `protoc`, `python3`, `npm`, plus global
   `grpc-tools` — it will not run on Windows without WSL/Git Bash.

4. **Upstream dependency is absent.** `server.py` imports
   `from bubblelabs_nodes import NodeRegistry`. No `bubblelabs_nodes` module exists
   anywhere in this repo, so the node adapter cannot load real nodes.

5. **Packaging bug (will break publish).** `package.json` has
   `files: ["dist", "generated"]`, which excludes the `.proto` files the client loads
   at runtime. `proto/` also sits *outside* the package root, so it cannot be added
   via `files`. A published tarball would reproduce bug #1. Fixing this needs a
   `prepack` copy step or a repo restructure — a build-system decision, so it was
   left alone rather than guessed at.

6. **Unverified performance claims.** "5-10x faster latency", "10x throughput",
   "1000+ req/s" appear as fact in the reports; the docs' own tables mark them
   "⚠️ Pending test". Nothing was benchmarked.

7. **Security items still open** (the reports concede this): TLS/mTLS,
   authentication, and CORS are all "add for production".

---

## Honest readiness assessment

- **`proto/` — solid.** 7 valid proto3 schemas; service/RPC names match what the TS
  client calls. This is the strongest part of the integration.
- **`typescript/client.ts` — compiles, builds, unit-tested, and is genuinely wired**
  (it calls real stubs, unlike the Python client). Cannot be confirmed against a live
  server here. Treat as *plausible but unproven at the wire level*.
- **`python/` — scaffolding.** Syntactically valid, architecturally reasonable, but the
  server registers no servicer and the client returns canned values.

**Bottom line:** this is a well-structured **skeleton at roughly design/prototype
stage**, not a production-ready integration. Reaching e2e requires, at minimum:
running `generate.sh`, uncommenting and wiring
`add_NodeRegistryServicer_to_server`, implementing the Python client's real stub
calls, and supplying `bubblelabs_nodes`. Only then is a live-server e2e meaningful.

### Reproduce this assessment

```bash
cd core-projects/BubbleLab/integrations/openevolve-grpc/typescript
npm ci
npx tsc --noEmit   # exit 0
npm test           # 10/10 pass
```
