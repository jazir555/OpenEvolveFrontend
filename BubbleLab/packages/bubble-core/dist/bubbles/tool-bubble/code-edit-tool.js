/**
 * EDIT BUBBLEFLOW TOOL
 *
 * A tool bubble that applies code edits to BubbleFlow files using Morph Fast Apply.
 * This tool uses the Morph API via OpenRouter to intelligently merge code changes
 * specified by an AI agent into existing BubbleFlow code, following the Fast Apply
 * pattern used in Cursor.
 *
 * Features:
 * - Intelligent code merging using Morph Fast Apply model
 * - Support for lazy edits with "// ... existing code ..." markers
 * - Minimal context repetition for efficient edits
 * - Automatic validation after edits
 * - Detailed diff reporting
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import { AIAgentBubble } from '../service-bubble/ai-agent.js';
/**
 * Define the parameters schema using Zod
 * This schema validates and types the input parameters for the edit tool
 */
const EditBubbleFlowToolParamsSchema = z.object({
    // The original BubbleFlow code
    initialCode: z
        .string()
        .max(500000, 'Code exceeds maximum allowed size of 500KB') // SECURITY: Limit size to prevent DoS
        .describe('The original code to be edited')
        .refine((code) => {
        // SECURITY FIX: Validate that code doesn't contain malicious patterns
        // This is a basic safety check, not a full parser
        // Block obvious malicious patterns
        const maliciousPatterns = [
            /eval\s*\(/i, // Code execution
            /Function\s*\(/i, // Dynamic function creation
            /require\s*\(\s*['"]child_process['"]\)/i, // Process spawning
            /require\s*\(\s*['"]fs['"]\)/i, // File system access (should use APIs)
            /\.exec\s*\(/i, // Command execution
            /\.spawn\s*\(/i, // Process spawning
            /\.fork\s*\(/i, // Process forking
            /import\s*\(/i, // Dynamic imports (can be safe but risky)
            /new\s+Function\s*\(/i, // Dynamic function
            /__proto__/i, // Prototype pollution
            /constructor\s*\[/i, // Constructor pollution
        ];
        return !maliciousPatterns.some((pattern) => pattern.test(code));
    }, 'Code contains potentially malicious patterns'),
    // The edit instructions
    instructions: z
        .string()
        .max(10000, 'Instructions exceed maximum allowed size of 10KB') // SECURITY: Limit size
        .describe('A single sentence instruction in first person describing what changes are being made (e.g., "I am adding error handling to the user auth and removing the old auth functions"). Used to help disambiguate uncertainty in the edit.'),
    // The code edit with markers
    codeEdit: z
        .string()
        .max(200000, 'Code edit exceeds maximum allowed size of 200KB') // SECURITY: Limit size
        .describe('The code changes to apply. Specify ONLY the precise lines of code that you wish to edit. Use "// ... existing code ..." to represent unchanged sections. DO NOT omit spans of pre-existing code without using the marker, as this may cause inadvertent deletion.')
        .refine((code) => {
        // SECURITY FIX: Validate that code edit doesn't contain malicious patterns
        const maliciousPatterns = [
            /eval\s*\(/i,
            /Function\s*\(/i,
            /require\s*\(\s*['"]child_process['"]\)/i,
            /require\s*\(\s*['"]fs['"]\)/i,
            /\.exec\s*\(/i,
            /\.spawn\s*\(/i,
            /\.fork\s*\(/i,
            /import\s*\(/i,
            /new\s+Function\s*\(/i,
            /__proto__/i,
            /constructor\s*\[/i,
        ];
        return !maliciousPatterns.some((pattern) => pattern.test(code));
    }, 'Code edit contains potentially malicious patterns'),
    // Optional Morph model configuration
    morphModel: z
        .string()
        .default('morph/morph-v3-large')
        .optional()
        .describe('Morph model to use for applying edits via OpenRouter'),
    // Credentials (injected at runtime)
    credentials: z
        .record(z.string(), z.string())
        .optional()
        .describe('Credentials (HIDDEN from AI - injected at runtime)'),
    // Optional configuration
    config: z
        .record(z.string(), z.unknown())
        .optional()
        .describe('Configuration for the edit tool (HIDDEN from AI - injected at runtime)'),
});
/**
 * Define the result schema
 * This schema defines what the edit tool returns
 */
const EditBubbleFlowToolResultSchema = z.object({
    // The final merged code
    mergedCode: z.string().describe('The final code after applying edits'),
    // Success indicator
    applied: z.boolean().describe('Whether the edit was successfully applied'),
    // Optional diff information
    diff: z
        .string()
        .optional()
        .describe('Unified diff showing the changes made (if available)'),
    // Metadata
    metadata: z.object({
        editedAt: z.string().describe('Timestamp when edit was performed'),
        originalLength: z
            .number()
            .describe('Length of original code in characters'),
        finalLength: z.number().describe('Length of final code in characters'),
        morphModel: z.string().describe('Morph model used for the edit'),
    }),
    // Standard result fields
    success: z.boolean().describe('Whether the edit operation was successful'),
    error: z.string().describe('Error message if edit failed'),
});
/**
 * Edit BubbleFlow Tool
 * Applies code edits using Morph Fast Apply API via AIAgent
 */
export class EditBubbleFlowTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    // Bubble type - always 'tool' for tool bubbles
    static type = 'tool';
    // Unique identifier for the tool
    static bubbleName = 'code-edit-tool';
    // Schemas for validation
    static schema = EditBubbleFlowToolParamsSchema;
    static resultSchema = EditBubbleFlowToolResultSchema;
    // Short description
    static shortDescription = 'Applies code edits to BubbleFlow files using Morph Fast Apply';
    // Long description with detailed information
    static longDescription = `
    A tool for intelligently applying code edits to BubbleFlow TypeScript files.
    Uses the Morph Fast Apply API via OpenRouter to merge lazy code edits into existing code.

    What it does:
    - Merges code edits specified with "// ... existing code ..." markers
    - Uses Morph's apply model for intelligent code merging
    - Minimizes context repetition for efficient edits
    - Returns the final merged code

    How it works:
    - Takes original code, edit instructions, and code edit as input
    - Sends to Morph API via OpenRouter using HttpBubble
    - Receives merged code from Morph's apply model
    - Returns the final code ready to be written to file

    Use cases:
    - When an AI agent needs to make edits to BubbleFlow code
    - When applying multiple distinct edits to a file at once
    - When making targeted changes without rewriting entire files
    - When following the Cursor Fast Apply pattern for code edits

    Important:
    - The codeEdit parameter should use "// ... existing code ..." to mark unchanged sections
    - The instructions parameter should be generated by the model in first person
    - Requires OPENROUTER_CRED credential for Morph API access via OpenRouter
  `;
    // Short alias for the tool
    static alias = 'code-edit';
    /**
     * Main action method - performs code edit merging
     */
    async performAction() {
        try {
            // Extract parameters
            const { initialCode, instructions, codeEdit, morphModel } = this.params;
            // Validate inputs
            if (!initialCode || initialCode.trim().length === 0) {
                return {
                    mergedCode: '',
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: 0,
                        finalLength: 0,
                        morphModel: morphModel ?? 'morph/morph-v3-large',
                    },
                    success: false,
                    error: 'Initial code cannot be empty',
                };
            }
            if (!instructions || instructions.trim().length === 0) {
                return {
                    mergedCode: initialCode,
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: initialCode.length,
                        finalLength: initialCode.length,
                        morphModel: morphModel ?? 'morph/morph-v3-large',
                    },
                    success: false,
                    error: 'Instructions cannot be empty',
                };
            }
            if (!codeEdit || codeEdit.trim().length === 0) {
                return {
                    mergedCode: initialCode,
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: initialCode.length,
                        finalLength: initialCode.length,
                        morphModel: morphModel ?? 'morph/morph-v3-large',
                    },
                    success: false,
                    error: 'Code edit cannot be empty',
                };
            }
            // SECURITY FIX: Check both API keys first before logging warnings
            const apiKey = this.params.credentials?.[CredentialType.OPENROUTER_CRED];
            const geminiKey = this.params.credentials?.[CredentialType.GOOGLE_GEMINI_CRED];
            // If neither key is available, return clear error
            if (!apiKey && !geminiKey) {
                return {
                    mergedCode: initialCode,
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: initialCode.length,
                        finalLength: initialCode.length,
                        morphModel: 'none',
                    },
                    success: false,
                    error: 'Code editing requires either OPENROUTER_CRED (recommended) or GOOGLE_GEMINI_CRED credential. Please provide at least one.',
                };
            }
            // If OpenRouter key not available, use Google Gemini as fallback
            if (!apiKey) {
                console.warn('[CodeEditTool] OpenRouter API key not found, using Google Gemini fallback');
                // geminiKey is guaranteed to exist here due to the check above
                try {
                    // Use AIAgentBubble with Google Gemini for code editing
                    const geminiPrompt = `You are a code editing assistant. Your task is to merge code edits into existing code following the instruction.

Original Code:
\`\`\`typescript
${initialCode}
\`\`\`

Instruction: ${instructions}

Edit to apply:
\`\`\`typescript
${codeEdit}
\`\`\`

IMPORTANT: Merge the edit into the original code and return ONLY the final merged code. Do not include any explanations, markdown formatting, or code blocks. Just return the raw code.`;
                    const aiAgent = new AIAgentBubble({
                        name: 'Code Editor (Gemini)',
                        message: geminiPrompt,
                        model: {
                            model: 'google/gemini-3-flash-preview',
                            temperature: 0.2,
                        },
                        credentials: {
                            [CredentialType.GOOGLE_GEMINI_CRED]: geminiKey,
                        },
                    });
                    const aiResult = await aiAgent.action();
                    if (!aiResult.success || !aiResult.data?.response) {
                        return {
                            mergedCode: initialCode,
                            applied: false,
                            metadata: {
                                editedAt: new Date().toISOString(),
                                originalLength: initialCode.length,
                                finalLength: initialCode.length,
                                morphModel: 'google/gemini-3-flash-preview (fallback)',
                            },
                            success: false,
                            error: aiResult.error || 'Gemini code editing failed',
                        };
                    }
                    // Extract code from response (remove markdown code blocks if present)
                    let mergedCode = aiResult.data.response.trim();
                    // Remove ```typescript and ``` markers if present
                    mergedCode = mergedCode.replace(/^```typescript\n/, '').replace(/^```\n/, '').replace(/\n```$/, '');
                    return {
                        mergedCode,
                        applied: true,
                        metadata: {
                            editedAt: new Date().toISOString(),
                            originalLength: initialCode.length,
                            finalLength: mergedCode.length,
                            morphModel: 'google/gemini-3-flash-preview (fallback)',
                        },
                        success: true,
                        error: '',
                    };
                }
                catch (fallbackError) {
                    return {
                        mergedCode: initialCode,
                        applied: false,
                        metadata: {
                            editedAt: new Date().toISOString(),
                            originalLength: initialCode.length,
                            finalLength: initialCode.length,
                            morphModel: 'google/gemini-3-flash-preview (fallback)',
                        },
                        success: false,
                        error: `Fallback to Gemini failed: ${fallbackError instanceof Error ? fallbackError.message : 'Unknown error'}`,
                    };
                }
            }
            // Construct the message for Morph API following the recommended format
            const morphPrompt = `<instruction>${instructions}</instruction>\n<code>${initialCode}</code>\n<update>${codeEdit}</update>`;
            // Use HttpBubble to call OpenRouter API
            const httpBubble = new HttpBubble({
                url: 'https://openrouter.ai/api/v1/chat/completions',
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: {
                    model: 'morph/morph-v3-large',
                    messages: [
                        {
                            role: 'user',
                            content: morphPrompt,
                        },
                    ],
                },
                timeout: 60000, // 60 second timeout for code generation
            }, this.context);
            // Execute the HTTP request
            const result = await httpBubble.action();
            if (!result.success || !result.data?.json) {
                return {
                    mergedCode: initialCode,
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: initialCode.length,
                        finalLength: initialCode.length,
                        morphModel: morphModel ?? 'morph/morph-v3-large',
                    },
                    success: false,
                    error: result.error || 'Morph API returned empty response',
                };
            }
            const responseData = result.data.json;
            // Extract token usage - OpenRouter uses 'usage' field (OpenAI format)
            // Fallback to usage_metadata if present (some providers use this)
            let inputTokens = 0;
            let outputTokens = 0;
            let totalTokens = 0;
            if (responseData.usage) {
                // OpenAI format: usage.prompt_tokens, usage.completion_tokens
                inputTokens = responseData.usage.prompt_tokens ?? 0;
                outputTokens = responseData.usage.completion_tokens ?? 0;
                totalTokens = responseData.usage.total_tokens ?? 0;
            }
            else if (responseData.usage_metadata) {
                // Alternative format: usage_metadata.input_tokens, usage_metadata.output_tokens
                inputTokens = responseData.usage_metadata.input_tokens ?? 0;
                outputTokens = responseData.usage_metadata.output_tokens ?? 0;
                totalTokens = responseData.usage_metadata.total_tokens ?? 0;
            }
            const tokenUsage = {
                inputTokens,
                outputTokens,
                totalTokens,
                modelName: morphModel ?? 'morph/morph-v3-large',
            };
            if (this.context?.logger) {
                this.context.logger.logTokenUsage({
                    usage: tokenUsage.inputTokens,
                    service: CredentialType.OPENROUTER_CRED,
                    unit: 'input_tokens',
                    subService: morphModel ?? 'morph/morph-v3-large',
                }, `LLM completion: ${tokenUsage.inputTokens} input`, {
                    bubbleName: 'code-edit-tool',
                    variableId: this.context?.variableId,
                    operationType: 'bubble_execution',
                });
                this.context.logger.logTokenUsage({
                    usage: tokenUsage.outputTokens,
                    service: CredentialType.OPENROUTER_CRED,
                    unit: 'output_tokens',
                    subService: morphModel ?? 'morph/morph-v3-large',
                }, `LLM completion: ${tokenUsage.outputTokens} output`, {
                    bubbleName: 'code-edit-tool',
                    variableId: this.context?.variableId,
                    operationType: 'bubble_execution',
                });
            }
            const mergedCode = responseData.choices?.[0]?.message?.content;
            if (!mergedCode) {
                return {
                    mergedCode: initialCode,
                    applied: false,
                    metadata: {
                        editedAt: new Date().toISOString(),
                        originalLength: initialCode.length,
                        finalLength: initialCode.length,
                        morphModel: morphModel ?? 'morph/morph-v3-large',
                    },
                    success: false,
                    error: 'Morph API returned empty content',
                };
            }
            // Return successful result
            return {
                mergedCode,
                applied: true,
                metadata: {
                    editedAt: new Date().toISOString(),
                    originalLength: initialCode.length,
                    finalLength: mergedCode.length,
                    morphModel: morphModel ?? 'openrouter/morph/morph-v3-large',
                },
                success: true,
                error: '',
            };
        }
        catch (error) {
            // Handle unexpected errors gracefully
            return {
                mergedCode: this.params.initialCode || '',
                applied: false,
                metadata: {
                    editedAt: new Date().toISOString(),
                    originalLength: this.params.initialCode?.length || 0,
                    finalLength: this.params.initialCode?.length || 0,
                    morphModel: this.params.morphModel ?? 'openrouter/morph/morph-v3-large',
                },
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
}
//# sourceMappingURL=code-edit-tool.js.map