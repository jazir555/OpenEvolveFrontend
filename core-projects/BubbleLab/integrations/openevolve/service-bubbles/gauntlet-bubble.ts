// ALIAS/DEPRECATION NOTE: Canonical implementation is
// packages/bubble-core/src/bubbles/service-bubble/openevolve-gauntlet-bubble.ts.
// This file only re-exports the real bubble from the built @bubblelab/bubble-core
// (single source of truth). The deep `dist/` specifier pulls ONLY this submodule,
// avoiding the package's monolithic barrel (which would require langchain/aws-sdk
// at load time) — same approach as `types/bubble-core.ts`.
export {
  OpenEvolveGauntletBubble as GauntletBubble,
  OpenEvolveGauntletBubble as default,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-gauntlet-bubble.js';
