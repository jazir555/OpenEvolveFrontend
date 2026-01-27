/**
 * BUBBLEFLOW VALIDATION TOOL
 *
 * A tool bubble that validates BubbleFlow TypeScript code for syntax, type errors,
 * and bubble structure. Takes TypeScript bubbleflow code as input and outputs
 * validation results including any errors and information about detected bubbles.
 *
 * Features:
 * - TypeScript syntax validation
 * - BubbleFlow class structure validation
 * - Bubble instantiation parsing and analysis
 * - Detailed error reporting with line numbers
 * - Bubble count and type information
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { BubbleFactory } from '../../bubble-factory.js';
import { validateBubbleFlow, } from '../../utils/bubbleflow-validation.js';
import { parseBubbleFlow } from '../../utils/bubbleflow-parser.js';
/**
 * Define the parameters schema using Zod
 * This schema validates and types the input parameters for the validation tool
 */
const BubbleFlowValidationToolParamsSchema = z.object({
    // The TypeScript code to validate
    code: z.string().describe('TypeScript code to validate'),
    // Optional validation options
    options: z
        .object({
        includeDetails: z
            .boolean()
            .default(true)
            .describe('Include detailed bubble analysis'),
        strictMode: z
            .boolean()
            .default(true)
            .describe('Enable strict TypeScript validation'),
    })
        .optional()
        .describe('Validation configuration options'),
    // Credentials (not needed for this tool but following template)
    credentials: z
        .record(z.string(), z.string())
        .optional()
        .describe('Credentials (HIDDEN from AI - injected at runtime)'),
    // Optional configuration
    config: z
        .record(z.string(), z.unknown())
        .optional()
        .describe('Configuration for the validation tool (HIDDEN from AI - injected at runtime)'),
});
/**
 * Define the result schema
 * This schema defines what the validation tool returns
 */
const BubbleFlowValidationToolResultSchema = z.object({
    // Validation results
    valid: z.boolean().describe('Whether the code is valid'),
    errors: z
        .array(z.string())
        .optional()
        .describe('List of validation errors if any'),
    // Bubble analysis (when validation succeeds)
    bubbleCount: z
        .number()
        .optional()
        .describe('Number of bubbles found in the code'),
    bubbles: z
        .array(z.object({
        variableName: z
            .string()
            .describe('Variable name assigned to the bubble'),
        bubbleName: z
            .string()
            .describe('Type of bubble (e.g., postgresql, slack)'),
        className: z
            .string()
            .describe('Bubble class name (e.g., PostgreSQLBubble)'),
        hasAwait: z.boolean().describe('Whether the bubble call is awaited'),
        hasActionCall: z.boolean().describe('Whether .action() is called'),
        parameterCount: z
            .number()
            .describe('Number of parameters passed to the bubble'),
    }))
        .optional(),
    variableTypes: z
        .array(z.object({
        name: z.string().describe('Variable name'),
        type: z.string().describe('Variable type'),
        line: z.number().describe('Line number'),
        column: z.number().describe('Column number'),
    }))
        .optional()
        .describe('Details about each bubble found'),
    // Metadata
    metadata: z.object({
        validatedAt: z.string().describe('Timestamp when validation was performed'),
        codeLength: z.number().describe('Length of the code in characters'),
        strictMode: z.boolean().describe('Whether strict mode was used'),
    }),
    // Standard result fields
    success: z
        .boolean()
        .describe('Whether the validation operation was successful'),
    error: z.string().describe('Error message if validation failed'),
});
/**
 * BubbleFlow Validation Tool
 * Validates TypeScript BubbleFlow code and provides detailed analysis
 */
export class BubbleFlowValidationTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    // Bubble type - always 'tool' for tool bubbles
    static type = 'tool';
    // Unique identifier for the tool
    static bubbleName = 'bubbleflow-validation-tool';
    // Schemas for validation
    static schema = BubbleFlowValidationToolParamsSchema;
    static resultSchema = BubbleFlowValidationToolResultSchema;
    // Short description
    static shortDescription = 'Validates BubbleFlow TypeScript code for syntax and structure';
    // Long description with detailed information
    static longDescription = `
    A comprehensive validation tool for BubbleFlow TypeScript code.
    
    What it does:
    - Validates TypeScript syntax and compilation
    - Checks BubbleFlow class structure and requirements
    - Parses and analyzes bubble instantiations
    - Provides detailed error reporting with line numbers
    - Returns metadata about detected bubbles
    
    How it works:
    - Uses TypeScript compiler API for syntax validation
    - Validates that code extends BubbleFlow and has handle method
    - Parses bubble instantiations using AST analysis
    - Maps bubble class names to registered bubble types
    
    Use cases:
    - When an AI agent needs to validate user-provided BubbleFlow code
    - When checking code before execution or deployment
    - When providing feedback on BubbleFlow implementation
    - When analyzing bubble usage patterns in code
  `;
    // Short alias for the tool
    static alias = 'validate-bubbleflow';
    bubbleFactory;
    initializationPromise = null;
    /**
     * Constructor
     */
    constructor(params, context) {
        super(params, context);
        this.bubbleFactory = new BubbleFactory();
        // SECURITY FIX: Store the initialization promise to prevent race condition
        // This ensures registerDefaults completes before any validation
        this.initializationPromise = this.initializeBubbleFactory();
    }
    async initializeBubbleFactory() {
        await this.bubbleFactory.registerDefaults();
    }
    /**
     * Main action method - performs BubbleFlow validation
     */
    async performAction(context) {
        // Context is available but not used in this tool
        void context;
        try {
            // SECURITY FIX: Ensure initialization completes before validation
            // This prevents race condition where validation runs before factory is ready
            if (this.initializationPromise) {
                await this.initializationPromise;
            }
            // Extract parameters
            const { code, options } = this.params;
            const includeDetails = options?.includeDetails ?? true;
            const strictMode = options?.strictMode ?? true;
            if (!code || code.trim().length === 0) {
                return {
                    valid: false,
                    errors: ['Code cannot be empty'],
                    variableTypes: [],
                    metadata: {
                        validatedAt: new Date().toISOString(),
                        codeLength: 0,
                        strictMode,
                    },
                    success: false,
                    error: 'Code cannot be empty',
                };
            }
            // Perform TypeScript validation
            const validationResult = await validateBubbleFlow(code, this.bubbleFactory);
            if (!validationResult.valid) {
                return {
                    valid: false,
                    errors: validationResult.errors || ['Unknown validation error'],
                    variableTypes: validationResult.variableTypes || [],
                    metadata: {
                        validatedAt: new Date().toISOString(),
                        codeLength: code.length,
                        strictMode,
                    },
                    success: false,
                    error: `Validation failed: ${validationResult.errors?.[0] || 'Unknown error'}`,
                };
            }
            // If validation passes and details are requested, parse bubbles
            let bubbleDetails;
            let bubbleCount = 0;
            if (includeDetails) {
                const parseResult = parseBubbleFlow(code, this.bubbleFactory);
                if (!parseResult.success) {
                    return {
                        valid: false,
                        errors: parseResult.errors || ['Failed to parse bubble details'],
                        variableTypes: validationResult.variableTypes || [],
                        metadata: {
                            validatedAt: new Date().toISOString(),
                            codeLength: code.length,
                            strictMode,
                        },
                        success: false,
                        error: `Bubble parsing failed: ${parseResult.errors?.[0] || 'Unknown error'}`,
                    };
                }
                bubbleCount = Object.keys(parseResult.bubbles).length;
                // Transform parsed bubbles to result format
                bubbleDetails = Object.values(parseResult.bubbles).map((bubble) => ({
                    variableName: bubble.variableName,
                    bubbleName: bubble.bubbleName,
                    className: bubble.className,
                    hasAwait: bubble.hasAwait,
                    hasActionCall: bubble.hasActionCall,
                    parameterCount: bubble.parameters.length,
                }));
            }
            // Return successful result
            return {
                valid: true,
                bubbleCount,
                bubbles: bubbleDetails,
                variableTypes: validationResult.variableTypes || [],
                metadata: {
                    validatedAt: new Date().toISOString(),
                    codeLength: code.length,
                    strictMode,
                },
                success: true,
                error: '',
            };
        }
        catch (error) {
            // Handle unexpected errors gracefully
            return {
                valid: false,
                errors: [
                    error instanceof Error ? error.message : 'Unknown validation error',
                ],
                metadata: {
                    validatedAt: new Date().toISOString(),
                    codeLength: this.params.code?.length || 0,
                    strictMode: this.params.options?.strictMode ?? true,
                },
                success: false,
                error: error instanceof Error ? error.message : 'Unknown validation error',
            };
        }
    }
}
/**
 * REGISTRATION AND USAGE
 *
 * 1. Register the tool in the BubbleFactory:
 * ```typescript
 * import { BubbleFlowValidationTool } from './services/bubbleflow-validation-tool.js';
 *
 * // In registerDefaults() method:
 * this.register('bubbleflow-validation', BubbleFlowValidationTool as BubbleClassWithMetadata);
 * ```
 *
 * 2. Direct usage:
 * ```typescript
 * const tool = new BubbleFlowValidationTool({
 *   code: 'export class TestFlow extends BubbleFlow<"webhook/http"> { ... }',
 *   options: { includeDetails: true }
 * });
 * const result = await tool.action();
 * ```
 *
 * 3. AI Agent usage:
 * ```typescript
 * // AI agent calls with:
 * {
 *   "code": "export class TestFlow extends BubbleFlow<'webhook/http'> { ... }",
 *   "options": { "includeDetails": true, "strictMode": true }
 * }
 * ```
 */
//# sourceMappingURL=bubbleflow-validation-tool.js.map