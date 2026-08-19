/**
 * Ambient module shims for external BubbleLab schema modules.
 *
 * The Z3 adapter contract tests reference schema modules that live outside
 * this adapter's source tree (in the BubbleLab monorepo). Those modules are
 * not part of this adapter's build, so we declare them as ambient modules to
 * keep the tests type-checking without pulling in external sources.
 *
 * The imports are mapped to these non-relative names (see contract.test.ts)
 * because TS only applies ambient `declare module` to non-relative specifiers.
 */

declare module 'bubblelab-canonical-models';
declare module 'bubblelab-z3';
