import { z } from 'zod';
import { randomUUID } from 'crypto';
import { MockDataGenerator } from '@bubblelab/shared-schemas';
import { BubbleValidationError, BubbleExecutionError, } from './bubble-errors.js';
import { sanitizeParams } from '@bubblelab/shared-schemas';
/**
 * Abstract base class for all bubble types
 * Implements common properties and methods defined in IBubble interface
 */
export class BaseBubble {
    name;
    schema;
    resultSchema;
    shortDescription;
    longDescription;
    alias;
    params;
    context;
    previousResult;
    instanceId;
    constructor(params, context, instanceId) {
        // Use static properties from the class - typed as required static metadata
        const ctor = this.constructor;
        this.name = ctor.bubbleName;
        this.schema = ctor.schema;
        this.resultSchema = ctor.resultSchema;
        this.shortDescription = ctor.shortDescription;
        this.longDescription = ctor.longDescription;
        this.alias = ctor.alias;
        this.instanceId = instanceId;
        try {
            this.params = this.schema.parse(params);
            const normalizedContext = context;
            // Enrich context with child variableId/currentUniqueId if dependencyGraph is provided
            if (normalizedContext &&
                normalizedContext.dependencyGraph &&
                normalizedContext.currentUniqueId) {
                const next = this.computeChildContext(normalizedContext);
                this.context = next;
                console.debug('[BaseBubble] Computed child context unique id:', this.context?.currentUniqueId);
                //Prnt the var id of the computed child context
                console.debug('[BaseBubble] Computed child context variable id:', this.context?.variableId);
            }
            else {
                this.context = normalizedContext;
            }
        }
        catch (error) {
            const errorMessage = error instanceof z.ZodError
                ? `Input Schema validation failed: ${error.errors.map((e) => `${e.path.join('.')}: ${e.message}`).join(', ')}`
                : `Input Schema validation failed: ${error instanceof Error ? error.message : 'Unknown validation error'}`;
            throw new BubbleValidationError(errorMessage, {
                variableId: context?.variableId,
                bubbleName: ctor.bubbleName,
                cause: error instanceof Error ? error : undefined,
            });
        }
    }
    /**
     * Compute child context based on dependency graph and current unique id.
     * Finds the node matching currentUniqueId, then determines this child's unique id as:
     * - If instanceId is provided: `${currentUniqueId}.${this.name}#${instanceId}`
     * - Otherwise: `${currentUniqueId}.${this.name}#k` for the next ordinal k
     * Assigns the variableId from the dependency graph if present, otherwise keeps parent's variableId.
     */
    computeChildContext(parentContext) {
        const graph = parentContext.dependencyGraph;
        const currentId = parentContext.currentUniqueId || '';
        if (!graph)
            return parentContext;
        // Depth-first search to find node by uniqueId
        const findByUniqueId = (node, target) => {
            if (node.uniqueId === target)
                return node;
            for (const child of node.dependencies || []) {
                const found = findByUniqueId(child, target);
                if (found)
                    return found;
            }
            return null;
        };
        console.log('Current ID:', this.name);
        console.log('Current varid:', this.context?.variableId);
        console.log('Finding parent node by uniqueId:', currentId);
        const parentNode = currentId ? findByUniqueId(graph, currentId) : graph;
        // If the current bubble matches the node at currentUniqueId, don't advance; keep IDs from that node
        if (parentNode && parentNode.name === this.name) {
            const sameNodeVarId = parentContext.variableId ??
                parentNode.variableId ??
                parentContext.variableId;
            return {
                ...parentContext,
                variableId: sameNodeVarId,
                currentUniqueId: currentId,
                __uniqueIdCounters__: { ...(parentContext.__uniqueIdCounters__ || {}) },
            };
        }
        // Determine this bubble's identifier under the parent
        const children = parentNode?.dependencies || [];
        const counters = { ...(parentContext.__uniqueIdCounters__ || {}) };
        let selectedChild = undefined;
        // Use ordinal counter as before
        const counterKey = `${currentId || 'ROOT'}|${this.name}`;
        const ordinal = (counters[counterKey] || 0) + 1;
        const suffix = `#${ordinal}`;
        counters[counterKey] = ordinal;
        // Try to select the nth child by name for an exact uniqueId match
        const sameNameChildren = children.filter((c) => c.name === this.name);
        selectedChild = sameNameChildren[ordinal - 1];
        const childUniqueId = selectedChild?.uniqueId ||
            (currentId
                ? `${currentId}.${this.name}${suffix}`
                : `${this.name}${suffix}`);
        // Try to find a matching child node to get variableId; fallback to parent's
        let matchingChild = children.find((c) => c.variableName === this.instanceId);
        console.log(`[BaseBubble] ${this.name}.computeChildContext: Matching child by variableName:`, matchingChild);
        // if no match is found fallback to || c.uniqueId === childUniqueId || c.name === this.name
        if (!matchingChild) {
            matchingChild = children.find((c) => c.uniqueId === childUniqueId || c.name === this.name);
            console.log(`[BaseBubble] ${this.name}.computeChildContext: Matching child by uniqueId:`, matchingChild);
        }
        const childVariableId = (matchingChild && typeof matchingChild.variableId === 'number'
            ? matchingChild.variableId
            : parentContext.variableId) || parentContext.variableId;
        return {
            ...parentContext,
            variableId: childVariableId,
            currentUniqueId: childUniqueId,
            __uniqueIdCounters__: counters,
        };
    }
    saveResult(result) {
        this.previousResult = result;
    }
    clearSavedResult() {
        this.previousResult = undefined;
    }
    /**
     * Override toJSON to prevent credential leaking via JSON.stringify or console.log
     * Only exposes safe metadata, never params which may contain credentials
     */
    toJSON() {
        return {
            name: this.name,
            type: this.type,
            shortDescription: this.shortDescription,
            alias: this.alias,
            // Explicitly exclude params, context, and previousResult
            // These may contain sensitive credentials
        };
    }
    /**
     * Execute the bubble - just runs the action
     */
    async action() {
        const logger = this.context?.logger;
        logger?.logBubbleExecution(this.context?.variableId ?? -999, this.name, this.name, sanitizeParams(this.params));
        // If we have a saved result, return it instead of executing
        if (this.previousResult) {
            logger?.debug(`[BubbleClass - ${this.name}] Returning saved result`);
            // Narrow saved base result to current TResult by keeping metadata and
            // treating data as unknown (caller side should only read known fields)
            const savedResult = this.previousResult;
            // Log bubble execution completion for saved result
            logger?.logBubbleExecutionComplete(this.context?.variableId ?? -999, this.name, this.name, savedResult);
            return savedResult;
        }
        let result;
        try {
            result = await this.performAction(this.context);
        }
        catch (error) {
            console.error('Error executing bubble:', error);
            this.context?.logger?.logBubbleExecutionComplete(this.context?.variableId ?? -999, this.name, this.name, {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                executionId: randomUUID(),
                timestamp: new Date(),
            });
            this.context?.logger?.error(`[${this.name}] Unexpected error when performing action: ${error instanceof Error ? error.message : 'Unknown error'}`);
            throw new BubbleExecutionError(error instanceof Error ? error.message : 'Unknown error', {
                variableId: this.context?.variableId,
                bubbleName: this.name,
                executionPhase: 'execution',
                cause: error instanceof Error ? error : undefined,
            });
        }
        // Validate result if schema is provided
        if (this.resultSchema) {
            try {
                const validatedResult = this.resultSchema.parse(result);
                const finalResult = {
                    success: result.success,
                    data: result,
                    executionId: randomUUID(),
                    error: validatedResult.error || '',
                    timestamp: new Date(),
                };
                // Log bubble execution completion
                logger?.logBubbleExecutionComplete(this.context?.variableId ?? -999, this.name, this.name, finalResult);
                if (!finalResult.success) {
                    logger?.warn(`[${this.name}] Execution did not succeed: ${finalResult.error}. The flow will continue to run unless you manually catch and handle the error.`);
                }
                return finalResult;
            }
            catch (validationError) {
                // Validation error for result validation failures
                const errorMessage = validationError instanceof z.ZodError
                    ? `Result schema validation failed: ${validationError.errors.map((e) => `${e.path.join('.')}: ${e.message}`).join(', ')}`
                    : `Result validation failed: ${validationError instanceof Error ? validationError.message : 'Unknown validation error'}`;
                throw new BubbleValidationError(errorMessage, {
                    variableId: this.context?.variableId,
                    bubbleName: this.name,
                    cause: validationError instanceof Error ? validationError : undefined,
                });
            }
        }
        // No result schema defined - proceed without validation
        const finalResult = {
            success: result.success,
            // For data we strip out any excessive fields
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            data: (({ ...rest }) => rest)(result),
            error: result.error || '',
            executionId: randomUUID(),
            timestamp: new Date(),
        };
        if (!result.success) {
            logger?.error(`[${this.name}] Execution error when performing action: ${result.error}`);
        }
        // Log bubble execution completion
        logger?.logBubbleExecutionComplete(this.context?.variableId ?? -999, this.name, this.name, finalResult);
        return finalResult;
    }
    /**
     * Generate mock result data based on the result schema
     * Useful for testing and development when you need sample data
     */
    generateMockResult() {
        return MockDataGenerator.generateMockResult(this.resultSchema);
    }
    /**
     * Generate mock result with a specific seed for reproducible results
     * Useful for consistent testing scenarios
     */
    generateMockResultWithSeed(seed) {
        const mockResult = MockDataGenerator.generateMockWithSeed(this.resultSchema, seed);
        // Override executionId to use randomUUID() instead of seeded value
        // This ensures executionId is always unique even with the same seed
        return {
            ...mockResult,
            executionId: randomUUID(),
        };
    }
}
//# sourceMappingURL=base-bubble-class.js.map