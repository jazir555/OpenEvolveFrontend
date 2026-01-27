/**
 * Shared types for streaming log events between backend and frontend
 */
import type { CoffeeClarificationEvent, CoffeeContextEvent, CoffeePlanEvent, CoffeeRequestExternalContextEvent } from './coffee.js';
export interface StreamingLogEvent {
    type: 'log_line' | 'bubble_instantiation' | 'bubble_execution' | 'bubble_start' | 'bubble_complete' | 'execution_complete' | 'error' | 'stream_complete' | 'info' | 'warn' | 'debug' | 'bubble_execution_complete' | 'trace' | 'fatal' | 'bubble_parameters_update' | 'tool_call_start' | 'tool_call_complete' | 'function_call_start' | 'function_call_complete';
    timestamp: string;
    lineNumber?: number;
    variableId?: number;
    message: string;
    bubbleId?: string;
    bubbleName?: string;
    variableName?: string;
    additionalData?: Record<string, unknown>;
    executionTime?: number;
    memoryUsage?: number;
    logLevel?: string;
    bubbleParameters?: Record<number, import('./bubble-definition-schema').ParsedBubbleWithInfo>;
    toolCallId?: string;
    toolName?: string;
    toolInput?: unknown;
    toolOutput?: unknown;
    toolDuration?: number;
    functionName?: string;
    functionInput?: unknown;
    functionOutput?: unknown;
    functionDuration?: number;
    tokenUsage?: {
        inputTokens: number;
        outputTokens: number;
        totalTokens: number;
    };
    cumulativeTokenUsage?: {
        inputTokens: number;
        outputTokens: number;
        totalTokens: number;
    };
}
export type StreamingEvent = {
    type: 'start';
    data: {
        message: string;
        maxIterations: number;
        timestamp: string;
    };
} | {
    type: 'llm_start';
    data: {
        model: string;
        temperature: number;
    };
} | {
    type: 'token';
    data: {
        content: string;
        messageId: string;
    };
} | {
    type: 'think';
    data: {
        content: string;
        messageId: string;
    };
} | {
    type: 'llm_complete';
    data: {
        messageId: string;
        totalTokens?: number;
        content?: string;
    };
} | {
    type: 'tool_start';
    data: {
        tool: string;
        input: unknown;
        callId: string;
    };
} | {
    type: 'tool_complete';
    data: {
        tool: string;
        input: {
            input: string;
        };
        output: unknown;
        duration: number;
        callId: string;
    };
} | {
    type: 'iteration_start';
    data: {
        iteration: number;
    };
} | {
    type: 'iteration_complete';
    data: {
        iteration: number;
        hasToolCalls: boolean;
    };
} | {
    type: 'error';
    data: {
        error: string;
        recoverable: boolean;
    };
} | {
    type: 'complete';
    data: {
        result: unknown;
        totalDuration: number;
    };
} | {
    type: 'coffee_clarification';
    data: CoffeeClarificationEvent;
} | {
    type: 'coffee_context_gathering';
    data: CoffeeContextEvent;
} | {
    type: 'coffee_request_context';
    data: CoffeeRequestExternalContextEvent;
} | {
    type: 'coffee_plan';
    data: CoffeePlanEvent;
} | {
    type: 'coffee_complete';
    data: {
        success: boolean;
        message?: string;
    };
};
export type StreamCallback = (event: StreamingLogEvent) => void | Promise<void>;
//# sourceMappingURL=streaming-events.d.ts.map