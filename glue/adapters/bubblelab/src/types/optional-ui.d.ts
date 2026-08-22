/**
 * Optional UI ambient declarations.
 *
 * The BubbleLab React components import a shared shadcn-style UI kit via the
 * `@/components/ui/*` path alias and icons from `lucide-react`. Those modules are
 * provided by the host application that consumes this package, not by the package
 * itself, so we declare them as ambient `any` modules. This lets the `tsc` build
 * type-check the component sources in isolation (see `tsconfig.components.json`)
 * without requiring the UI kit to be installed in this package.
 *
 * A real DOM build/render still needs the host app to supply the concrete
 * `@/components/ui/*` and `lucide-react` implementations (mapped via the
 * `@/*` -> `src/*` path alias at the consumer's bundler).
 */

declare module "@/components/ui/*";
declare module "lucide-react";

// Modules that live outside this package (shared orchestration layer supplied by
// the host). Declared as ambient `any` so the isolated component build resolves
// them without the broader monorepo present.
declare module "*workflow-orchestrator";
declare module "*plugin-registry";
declare module "*workflow-templates";

