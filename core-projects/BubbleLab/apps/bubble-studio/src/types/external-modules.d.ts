declare module '@bubblelab/bubble-core' {
  export interface BubbleContext {
    [key: string]: unknown;
  }

  export class WorkflowBubble<TInput = unknown, TOutput = unknown> {
    protected params: TInput;
    protected context?: BubbleContext;
    constructor(params: TInput, context?: BubbleContext);
    protected performAction(context?: BubbleContext): Promise<TOutput>;
    action(context?: BubbleContext): Promise<TOutput>;
  }

  export class BubbleFlow<TTrigger extends string = string> {
    constructor(...args: unknown[]);
  }

  export type BubbleTriggerEventRegistry = Record<string, unknown>;

  export class StorageBubble {
    constructor(params: Record<string, unknown>);
    action(): Promise<Record<string, unknown>>;
  }

  export class PDFOcrWorkflow {
    constructor(params: Record<string, unknown>);
    action(): Promise<Record<string, unknown>>;
  }

  export class ParseDocumentWorkflow {
    constructor(params: Record<string, unknown>);
    action(): Promise<Record<string, unknown>>;
  }

  export class GenerateDocumentWorkflow {
    constructor(params: Record<string, unknown>);
    action(): Promise<Record<string, unknown>>;
  }
}

declare module 'html-to-image' {
  export function toPng(
    node: HTMLElement,
    options?: Record<string, unknown>
  ): Promise<string>;
}
