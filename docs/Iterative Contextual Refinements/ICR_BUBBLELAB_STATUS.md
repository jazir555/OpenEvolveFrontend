# BubbleLab ICR Integration Status

Date: 2026-02-01

## Completed
- Added `openevolve-icr` service bubble with refinement events, reward calibration, and heatmap snapshot operations.
- Added `openevolve-determinism` bubble and determinism API routes for deterministic generation + reproducibility checks.
- Added `openevolve-decomposition` bubble and decomposition API route for problem analysis + plan generation.
- Added `openevolve-decomposition-workflow` bubble for sovereign decomposition workflows with MDAP/MAKER toggles.
- Registered `openevolve-icr` in BubbleLab bubble factory and code generator list, and exported it from bubble-core.
- Added ICR event bridge API (`/icr/*`) to BubbleLab OpenEvolve service with in-memory queues.
- Added determinism endpoints (`/determinism/*`) to BubbleLab OpenEvolve service.
- Added decomposition endpoint (`/api/decomposition/plan`) to BubbleLab OpenEvolve service.
- Added persistent ICR settings endpoints (`/api/settings/icr`) with defaults and validation.
- Added Bubble Studio ICR settings panel, including toggles and thresholds for auto-refine, reward calibration, and heatmap analysis.
- Added ICR config state to Bubble Studio store and API client helpers.

## Remaining / Follow-ups
- Regenerate `apps/bubble-studio/public/bubbles.json` (via `bubble-core/scripts/bubble-metadata-bundler.ts`) if you want the new bubble to appear in the Bubble Studio list UI.
- Optionally extend `@bubblelab/shared-schemas` BubbleName + credential mappings to include all OpenEvolve bubbles (including `openevolve-icr`) for stricter typing.
- Wire heatmap/ICR settings into any runtime polling bridges if you want them to gate event polling or calibration requests.

## Files Touched
- `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/openevolve-icr-bubble.ts`
- `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/openevolve-determinism-bubble.ts`
- `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/openevolve-decomposition-bubble.ts`
- `BubbleLab/packages/bubble-core/src/bubbles/service-bubble/openevolve-decomposition-workflow-bubble.ts`
- `BubbleLab/packages/bubble-core/src/bubble-factory.ts`
- `BubbleLab/packages/bubble-core/src/index.ts`
- `BubbleLab/services/openevolve-api/api/icr.py`
- `BubbleLab/services/openevolve-api/api/determinism.py`
- `BubbleLab/services/openevolve-api/api/decomposition.py`
- `BubbleLab/services/openevolve-api/api/settings.py`
- `BubbleLab/services/openevolve-api/api/__init__.py`
- `BubbleLab/services/openevolve-api/main.py`
- `BubbleLab/services/openevolve-api/models/__init__.py`
- `BubbleLab/apps/bubble-studio/src/types/api.ts`
- `BubbleLab/apps/bubble-studio/src/lib/api-client.ts`
- `BubbleLab/apps/bubble-studio/src/stores/configStore.ts`
- `BubbleLab/apps/bubble-studio/src/components/settings/SettingsPanel.tsx`
