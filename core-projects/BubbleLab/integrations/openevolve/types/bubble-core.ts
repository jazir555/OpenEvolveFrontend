/**
 * Runtime + type shim for `@bubblelab/bubble-core`.
 *
 * The integration IS now part of the pnpm workspace (core-projects/BubbleLab/
 * pnpm-workspace.yaml globs `integrations/*`), so the real bubble-core package
 * is linked via node_modules and resolved at runtime. This module still provides
 * BOTH:
 *   - the type declarations the integration bubbles need (mirrors
 *     `types/bubble-core.d.ts`), and
 *   - minimal runtime class bodies so the bubbles can actually be instantiated
 *     and executed under `tsx` / Node (the `.d.ts` form carries no runtime).
 *
 * The bubbles only ever *construct* `HttpBubble`/`ServiceBubble` for the
 * OpenEvolve HTTP contract; they perform the real HTTP calls via `fetch`
 * themselves, so the shim implementations are intentionally inert.
 *
 * The canonical OpenEvolve service bubbles are re-exported from the REAL built
 * `@bubblelab/bubble-core` package (packages/bubble-core/dist/.../openevolve-*-bubble.js),
 * whose source of truth lives at
 * packages/bubble-core/src/bubbles/service-bubble/openevolve-*-bubble.ts.
 */

export interface BubbleContext {
  logger?: {
    log?: (...args: unknown[]) => void;
    error?: (...args: unknown[]) => void;
    warn?: (...args: unknown[]) => void;
    debug?: (...args: unknown[]) => void;
    info?: (...args: unknown[]) => void;
  };
  variableId?: number;
  invocationCallSiteKey?: string;
  dependencyGraph?: unknown;
  currentUniqueId?: string;
  __uniqueIdCounters__?: Record<string, number>;
  executionMeta?: unknown;
  [key: string]: unknown;
}

export type BubbleOperationResult = Record<string, unknown>;

export interface BubbleResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  executionId?: string;
  timestamp?: Date;
}

/** Minimal base class matching how the integration bubbles extend it. */
export abstract class ServiceBubble<
  TParams = unknown,
  TResult = any
> {
  public params: TParams;
  public context?: BubbleContext;
  public previousResult: BubbleResult<TResult> | undefined;
  constructor(params: unknown, context?: BubbleContext) {
    this.params = params as TParams;
    this.context = context;
    this.previousResult = undefined;
  }
  async action(): Promise<any> {
    return { success: true };
  }
  async performAction(_context?: BubbleContext): Promise<TResult> {
    return (await this.action()) as TResult;
  }
  saveResult<R extends BubbleOperationResult>(_result: BubbleResult<R>): void {
    /* inert shim */
  }
  clearSavedResult(): void {
    /* inert shim */
  }
  generateMockResult(): BubbleResult<TResult> {
    return { success: true };
  }
  generateMockResultWithSeed(_seed: number): BubbleResult<TResult> {
    return { success: true };
  }
}

export interface HttpBubbleParams {
  url: string;
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  [key: string]: unknown;
}

export class HttpBubble {
  public params: HttpBubbleParams;
  public context?: BubbleContext;
  constructor(params: HttpBubbleParams, context?: BubbleContext) {
    this.params = params;
    this.context = context;
  }
  async action(): Promise<any> {
    return { success: true };
  }
}

export interface PostgreSQLBubbleParams {
  query?: string;
  params?: unknown[];
  connectionPool?: { max: number; idleTimeoutMillis: number };
  [key: string]: unknown;
}

export class PostgreSQLBubble {
  constructor(_params: PostgreSQLBubbleParams, _context?: BubbleContext) {}
  async action(): Promise<{
    success: boolean;
    data?: { rows?: any[]; rowCount?: number };
    error?: string;
  }> {
    return { success: true, data: { rows: [], rowCount: 0 } };
  }
  async query(
    _sql: string,
    _params?: unknown[]
  ): Promise<{ rows?: any[]; rowCount?: number }> {
    return { rows: [], rowCount: 0 };
  }
}

export interface AIAgentBubbleParams {
  model?: { model?: string };
  systemPrompt?: string;
  [key: string]: unknown;
}

export class AIAgentBubble {
  constructor(_params: AIAgentBubbleParams, _context?: BubbleContext) {}
  async action(): Promise<any> {
    return { success: true };
  }
}

// ----------------------------------------------------------------------------
// Re-export the canonical OpenEvolve service bubbles from the BUILT
// `@bubblelab/bubble-core` package (single source of truth). These are deep
// imports into the package's compiled output so we pull ONLY the specific
// submodule (and its light runtime deps: zod + shared-schemas), avoiding the
// package's monolithic barrel (which would otherwise require langchain,
// aws-sdk, etc. at load time).
// ----------------------------------------------------------------------------
export {
  OpenEvolveKnowledgeEngineBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-knowledge-engine-bubble.js';
export {
  OpenEvolveWorkflowOrchestratorBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-workflow-orchestrator-bubble.js';
export {
  OpenEvolveCrewAIBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-crewai-bubble.js';
export {
  OpenEvolveLeanAideBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-leanaide-bubble.js';
export {
  OpenEvolveZ3ProverBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-z3prover-bubble.js';
export {
  OpenEvolveAceToolsBubble,
} from '../node_modules/@bubblelab/bubble-core/dist/bubbles/service-bubble/openevolve-ace-tools-bubble.js';
