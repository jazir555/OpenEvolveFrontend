/**
 * Ambient type shim for `@bubblelab/bubble-core`.
 *
 * This integration is not yet part of the pnpm workspace, so the real
 * bubble-core package cannot be resolved here. This minimal declaration
 * exposes ONLY the symbols the OpenEvolve integration bubbles use, mapped
 * via tsconfig `paths` so the real package is never consulted (no dupes).
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
export declare abstract class ServiceBubble<
  TParams = unknown,
  TResult = any
> {
  public params: TParams;
  public context?: BubbleContext;
  public previousResult: BubbleResult<TResult> | undefined;
  constructor(params: unknown, context?: BubbleContext);
  action(): Promise<any>;
  performAction(context?: BubbleContext): Promise<TResult>;
  saveResult<R extends BubbleOperationResult>(result: BubbleResult<R>): void;
  clearSavedResult(): void;
  generateMockResult(): BubbleResult<TResult>;
  generateMockResultWithSeed(seed: number): BubbleResult<TResult>;
}

export interface HttpBubbleParams {
  url: string;
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  [key: string]: unknown;
}

export declare class HttpBubble {
  constructor(params: HttpBubbleParams, context?: BubbleContext);
  action(): Promise<any>;
}

export interface PostgreSQLBubbleParams {
  query?: string;
  params?: unknown[];
  connectionPool?: { max: number; idleTimeoutMillis: number };
  [key: string]: unknown;
}

export declare class PostgreSQLBubble {
  constructor(params: PostgreSQLBubbleParams, context?: BubbleContext);
  action(): Promise<{
    success: boolean;
    data?: { rows?: any[]; rowCount?: number };
    error?: string;
  }>;
  query(
    sql: string,
    params?: unknown[]
  ): Promise<{ rows?: any[]; rowCount?: number }>;
}

export interface AIAgentBubbleParams {
  model?: { model?: string };
  systemPrompt?: string;
  [key: string]: unknown;
}

export declare class AIAgentBubble {
  constructor(params: AIAgentBubbleParams, context?: BubbleContext);
  action(): Promise<any>;
}
