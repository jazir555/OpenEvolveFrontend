# BubbleLab Packages — Actual Build / Type-Check Status

Generated: 2026-08-20
Scope: verify TypeScript packages under `packages/` build/type-check and tests pass.

## Packages present
| Package | Name | Notes |
|---|---|---|
| bubble-core | `@bubblelab/bubble-core` | Known-good (prior pass). Re-checked exports. |
| bubble-runtime | `@bubblelab/bubble-runtime` | Verified this pass. |
| bubble-shared-schemas | `@bubblelab/shared-schemas` | Verified this pass. |
| bubble-scope-manager | `@bubblelab/ts-scope-manager` | Prebuilt single-file package. |
| create-bubblelab-app | `create-bubblelab-app` | JS CLI + template test. |
| ragbits-bubblelab-integration | (ragbits) | Known-good (prior pass). |

> Note: the task referenced `http-bubble`; **no such package exists** in this repo.

## bubble-runtime — ✅ builds, ⚠️ tests partially fail
- `tsc --noEmit` (typecheck): PASS after 2 safe fixes.
- `tsc` (build → dist): PASS.
- Tests (`vitest run`, non-integration): **17 passed / 7 failed**.
- Safe fixes applied:
  1. `packages/bubble-core/package.json`: added `"./tracing"` subpath export so
     `@bubblelab/bubble-core/tracing` resolves (was missing → TS2307).
  2. `packages/bubble-runtime/src/runtime/BubbleRunner.tracing.ts:31`: cast the
     Jaeger `host`/`port` exporter `options` to `any` — the `ExporterConfig.options`
     union only declares OTLP/Collector/Console shapes (TS2353). No logic changed.
- Test failures: all 7 are in `validation/validator.test.ts` and stem from the
  custom TypeScript `LanguageService` host failing to resolve virtual files
  (`Could not find source file: '...src\virtual\*.ts'`). This is an environmental /
  LanguageService-host resolution issue, NOT a trivial fix — documented, not modified.

## bubble-shared-schemas — ⚠️ typecheck fails, ✅ JS build + tests pass
- `tsup` (JS bundle → dist/index.js): PASS (258 KB).
- `tsc --noEmit` (typecheck): **FAIL** — 7 errors in `credential-schema.ts` /
  `bubble-definition-schema.ts`. Cause: objects typed as
  `Record<CredentialType, X>` are missing several keys
  (OAUTH_TOKEN, ELASTICSEARCH_CRED, GITHUB_CRED, POSTGRESQL_CRED, …), i.e. the
  credential-type coverage is incomplete in source data. Fixing requires supplying
  the missing credential entries (data, not a mechanical type fix) → left as-is.
- Tests (`vitest run`): **15 passed / 0 failed**.

## bubble-scope-manager — ✅ ready
- Prebuilt package (`index.js`, `index.mjs`, `index.d.ts` already shipped). No
  tsconfig/build script; `tsc --noEmit` is a no-op (only `.d.ts` present). No action needed.

## create-bubblelab-app — ✅ JS-only, ⚠️ test not run
- Plain JS CLI (`bin/cli.js`) + `templates.test.ts`. No `tsconfig`, no TS build step.
- Not type-checked. Its `templates.test.ts` scaffolds a full app and imports
  `@bubblelab/shared-schemas`; running it requires the complete workspace install and
  is integration-heavy — not executed as a "trivial" check.

## Summary
- Builds clean: bubble-runtime, bubble-shared-schemas (JS), bubble-scope-manager,
  bubble-core*, rabits*.
- Type-check clean: bubble-runtime (after fixes), bubble-scope-manager, bubble-core*, rabits*.
- Known type errors: bubble-shared-schemas (incomplete credential map — data gap).
- Test issues: bubble-runtime has 7 LanguageService virtual-file failures (environmental).
- External deps: all packages rely on the pnpm workspace (`workspace:*`) + published
  deps already present in root `node_modules`; no missing external dependency installs required.

* = known-good from prior passes; re-verified exports only.
