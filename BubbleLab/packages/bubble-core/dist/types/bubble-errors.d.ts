/**
 * Custom error classes for bubble operations
 * These errors carry metadata like variableId to enable better error tracking and logging
 */
/**
 * Base error class for all bubble-related errors
 * Includes variableId and bubbleName for context tracking
 */
export declare class BubbleError extends Error {
    readonly variableId?: number;
    readonly bubbleName?: string;
    constructor(message: string, options?: {
        variableId?: number;
        bubbleName?: string;
        cause?: Error;
    });
}
/**
 * Thrown when bubble parameter validation fails
 * Used in BaseBubble constructor when schema.parse() fails
 */
export declare class BubbleValidationError extends BubbleError {
    readonly validationErrors?: string[];
    constructor(message: string, options?: {
        variableId?: number;
        bubbleName?: string;
        validationErrors?: string[];
        cause?: Error;
    });
}
/**
 * Thrown when bubble execution fails during performAction
 * Used in BaseBubble.action() when the operation fails
 */
export declare class BubbleExecutionError extends BubbleError {
    readonly executionPhase?: 'instantiation' | 'execution' | 'validation';
    constructor(message: string, options?: {
        variableId?: number;
        bubbleName?: string;
        executionPhase?: 'instantiation' | 'execution' | 'validation';
        cause?: Error;
    });
}
//# sourceMappingURL=bubble-errors.d.ts.map