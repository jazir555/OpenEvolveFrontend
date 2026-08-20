/**
 * Supplemental shims for this isolated package. The `ioredis` ambient module
 * lives in `types/ioredis.d.ts` (a script file) so it is treated as a true
 * ambient declaration rather than a module augmentation.
 */

declare global {
  interface ImportMeta {
    readonly env?: any;
  }
}

export {};
