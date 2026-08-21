// ALIAS/DEPRECATION NOTE: Canonical implementation is
// packages/bubble-core/src/bubbles/service-bubble/apify/apify.ts.
// This file previously contained only the literal token `placeholder` (not valid TS);
// it now re-exports the real bubble from the built @bubblelab/bubble-core (single
// source of truth). The deep `dist/` specifier pulls ONLY this submodule, avoiding
// the package's monolithic barrel (which would require langchain/aws-sdk at load
// time) — same approach as `types/bubble-core.ts`.
export {
  ApifyBubble,
  ApifyBubble as default,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/apify/apify.js';
