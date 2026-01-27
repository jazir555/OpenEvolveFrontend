import { BubbleLogger, LogLevel, } from './BubbleLogger.js';
import { sanitizeErrorMessage, sanitizeErrorStack, } from '../utils/error-sanitizer.js';
/**
 * Streaming version of BubbleLogger that emits real-time events
 * Extends BubbleLogger to maintain all existing functionality
 */
export class StreamingBubbleLogger extends BubbleLogger {
    streamCallback;
    constructor(flowName, options = { pricingTable: {} }) {
        const { streamCallback, ...loggerConfig } = options;
        super(flowName, loggerConfig);
        this.streamCallback = streamCallback;
    }
    /**
     * Override logLine to emit streaming events
     */
    logLine(lineNumber, message, additionalData) {
        // Check if this line should be logged before emitting
        const shouldLog = this.shouldLogLine(lineNumber);
        // Call parent method to maintain existing functionality and get the message
        const logMessage = super.logLine(lineNumber, message, additionalData);
        // Only emit streaming event if the line was actually logged
        if (shouldLog) {
            this.emitStreamEvent({
                type: 'log_line',
                timestamp: new Date().toISOString(),
                lineNumber,
                message: logMessage,
                additionalData,
                executionTime: this.getCurrentExecutionTime(),
                memoryUsage: this.getCurrentMemoryUsage(),
            });
        }
        return logMessage;
    }
    /**
     * Override logBubbleInstantiation to emit streaming events
     */
    logBubbleInstantiation(variableId, bubbleName, variableName, parameters) {
        // Call parent method and use the returned message
        const logMessage = super.logBubbleInstantiation(variableId, bubbleName, variableName, parameters);
        this.emitStreamEvent({
            type: 'bubble_instantiation',
            timestamp: new Date().toISOString(),
            variableId,
            message: logMessage,
            bubbleName,
            variableName,
            additionalData: { parameters, variableId },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
        });
        return logMessage;
    }
    /**
     * Override logBubbleExecution to emit streaming events
     */
    logBubbleExecution(variableId, bubbleName, variableName, parameters) {
        // Call parent method and use the returned message
        const logMessage = super.logBubbleExecution(variableId, bubbleName, variableName, parameters);
        this.emitStreamEvent({
            type: 'bubble_execution',
            timestamp: new Date().toISOString(),
            variableId,
            message: logMessage,
            bubbleName,
            variableName,
            additionalData: { parameters, variableId },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
        });
        return logMessage;
    }
    logBubbleExecutionComplete(variableId, bubbleName, variableName, result) {
        // Get individual bubble execution time BEFORE calling parent method
        // (parent method will clean up the start time)
        const individualExecutionTime = this.getBubbleExecutionTime(variableId);
        // Call parent method and use the returned message
        const logMessage = super.logBubbleExecutionComplete(variableId, bubbleName, variableName, result);
        this.emitStreamEvent({
            type: 'bubble_execution_complete',
            timestamp: new Date().toISOString(),
            message: logMessage,
            variableId,
            bubbleName,
            variableName,
            additionalData: { result, variableId },
            executionTime: individualExecutionTime,
            memoryUsage: this.getCurrentMemoryUsage(),
        });
        return logMessage;
    }
    /**
     * Override logFunctionCallStart to emit streaming events
     */
    logFunctionCallStart(variableId, functionName, functionInput, lineNumber) {
        // Call parent method and use the returned message
        const logMessage = super.logFunctionCallStart(variableId, functionName, functionInput, lineNumber);
        this.emitStreamEvent({
            type: 'function_call_start',
            timestamp: new Date().toISOString(),
            variableId,
            lineNumber,
            message: logMessage,
            functionName,
            functionInput,
            additionalData: { variableId, functionInput },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
        });
        return logMessage;
    }
    /**
     * Override logFunctionCallComplete to emit streaming events
     */
    logFunctionCallComplete(variableId, functionName, functionOutput, duration, lineNumber) {
        // Get individual function call execution time BEFORE calling parent method
        // (parent method will clean up the start time)
        const individualExecutionTime = this.getFunctionCallExecutionTime(variableId);
        const actualDuration = individualExecutionTime > 0 ? individualExecutionTime : duration;
        // Call parent method and use the returned message
        const logMessage = super.logFunctionCallComplete(variableId, functionName, functionOutput, actualDuration, lineNumber);
        this.emitStreamEvent({
            type: 'function_call_complete',
            timestamp: new Date().toISOString(),
            variableId,
            lineNumber,
            message: logMessage,
            functionName,
            functionOutput,
            functionDuration: actualDuration,
            additionalData: { variableId, functionOutput, duration: actualDuration },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
        });
        return logMessage;
    }
    /**
     * Log execution completion
     */
    logExecutionComplete(success, finalResult, error) {
        const message = success
            ? 'Execution completed successfully in ' +
                (this.getCurrentExecutionTime() / 1000).toFixed(2) +
                's.' +
                ' Total cost: $' +
                this.getExecutionSummary()
                    ?.serviceUsage?.reduce((acc, curr) => acc + curr.totalCost, 0)
                    .toFixed(6)
            : `Execution failed: ${error || 'Unknown error'}`;
        this.logLine(0, message, {
            success,
            finalResult,
            error,
        });
        this.emitStreamEvent({
            type: 'execution_complete',
            timestamp: new Date().toISOString(),
            message,
            additionalData: {
                success,
                finalResult,
                error,
                summary: this.getExecutionSummary(),
            },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
        });
    }
    /**
     * Override trace method to emit streaming events
     */
    trace(message, metadata) {
        super.trace(message, metadata);
        this.emitStreamEvent({
            type: 'trace',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: metadata?.bubbleName,
            variableName: metadata?.variableName,
            additionalData: metadata?.additionalData,
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.TRACE],
        });
    }
    /**
     * Override debug method to emit streaming events
     */
    debug(message, metadata) {
        super.debug(message, metadata);
        this.emitStreamEvent({
            type: 'debug',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: metadata?.bubbleName,
            variableName: metadata?.variableName,
            additionalData: metadata?.additionalData,
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.DEBUG],
        });
    }
    /**
     * Override info method to emit streaming events
     */
    info(message, metadata) {
        super.info(message, metadata);
        this.emitStreamEvent({
            type: 'info',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: metadata?.bubbleName,
            variableName: metadata?.variableName,
            additionalData: metadata?.additionalData,
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.INFO],
        });
    }
    /**
     * Override warn method to emit streaming events
     */
    warn(message, metadata) {
        super.warn(message, metadata);
        this.emitStreamEvent({
            type: 'warn',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: metadata?.bubbleName,
            variableName: metadata?.variableName,
            additionalData: metadata?.additionalData,
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.WARN],
        });
    }
    /**
     * Override error method to emit streaming events
     */
    error(message, error, metadata) {
        super.error(message, error, metadata);
        this.emitStreamEvent({
            type: 'error',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: error?.bubbleName,
            variableId: error?.variableId,
            additionalData: {
                ...metadata?.additionalData,
                error: error
                    ? {
                        message: sanitizeErrorMessage(error.message),
                        stack: sanitizeErrorStack(error),
                        name: error.name,
                    }
                    : undefined,
            },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.ERROR],
        });
    }
    /**
     * Override fatal method to emit streaming events
     */
    fatal(message, error, metadata) {
        super.fatal(message, error, metadata);
        this.emitStreamEvent({
            type: 'fatal',
            timestamp: new Date().toISOString(),
            message,
            lineNumber: metadata?.lineNumber,
            bubbleName: error?.bubbleName,
            variableId: error?.variableId,
            additionalData: {
                ...metadata?.additionalData,
                error: error
                    ? {
                        message: sanitizeErrorMessage(error.message),
                        stack: sanitizeErrorStack(error),
                        name: error.name,
                    }
                    : undefined,
            },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.FATAL],
        });
    }
    /**
     * Override logToolCallStart to add streaming events
     */
    logToolCallStart(toolCallId, toolName, toolInput, message) {
        // Call parent method to maintain existing functionality and get the message
        const logMessage = super.logToolCallStart(toolCallId, toolName, toolInput, message);
        // Emit streaming event using the returned message
        this.emitStreamEvent({
            type: 'tool_call_start',
            timestamp: new Date().toISOString(),
            message: logMessage,
            bubbleName: 'ai-agent',
            toolCallId,
            toolName,
            toolInput,
            additionalData: { toolCallId, toolName, toolInput },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.DEBUG],
        });
        return logMessage;
    }
    /**
     * Override logToolCallComplete to add streaming events
     */
    logToolCallComplete(toolCallId, toolName, toolInput, toolOutput, duration, message) {
        // Call parent method to maintain existing functionality and get the message
        const logMessage = super.logToolCallComplete(toolCallId, toolName, toolInput, toolOutput, duration, message);
        // Emit streaming event using the returned message
        this.emitStreamEvent({
            type: 'tool_call_complete',
            timestamp: new Date().toISOString(),
            message: logMessage,
            bubbleName: 'ai-agent',
            toolCallId,
            toolName,
            toolInput,
            toolOutput,
            toolDuration: duration,
            additionalData: { toolCallId, toolName, toolInput, toolOutput, duration },
            executionTime: this.getCurrentExecutionTime(),
            memoryUsage: this.getCurrentMemoryUsage(),
            logLevel: LogLevel[LogLevel.DEBUG],
        });
        return logMessage;
    }
    /**
     * Set or update the stream callback
     */
    setStreamCallback(callback) {
        this.streamCallback = callback;
    }
    /**
     * Clear the stream callback
     */
    clearStreamCallback() {
        this.streamCallback = undefined;
    }
    /**
     * Emit a streaming event if callback is set
     */
    emitStreamEvent(event) {
        if (this.streamCallback) {
            try {
                // Handle both sync and async callbacks
                const result = this.streamCallback(event);
                if (result instanceof Promise) {
                    result.catch((error) => {
                        console.error('Stream callback error:', error);
                    });
                }
            }
            catch (error) {
                console.error('Stream callback error:', error);
            }
        }
    }
    /**
     * Get current execution time in milliseconds
     */
    getCurrentExecutionTime() {
        const summary = this.getExecutionSummary();
        return summary.totalDuration;
    }
    /**
     * Get current memory usage in bytes
     */
    getCurrentMemoryUsage() {
        if (typeof process !== 'undefined' && process.memoryUsage) {
            return process.memoryUsage().heapUsed;
        }
        return 0;
    }
}
//# sourceMappingURL=StreamingBubbleLogger.js.map