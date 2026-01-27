import type { MessageContent } from '@langchain/core/messages';
import type { LLMResult, Generation } from '@langchain/core/outputs';
/**
 * Convert LangChain Generation[] to MessageContent for use with formatFinalResponse.
 * Handles both simple Generation (with text) and ChatGeneration (with message).
 *
 * This is useful when processing LLMResult from callbacks like handleLLMEnd,
 * where you have generations but want to use the unified formatFinalResponse.
 *
 * @param generations - Array of Generation objects (typically from LLMResult.generations.flat())
 * @returns MessageContent that can be passed to formatFinalResponse
 */
export declare function generationsToMessageContent(generations: Generation[]): MessageContent;
/**
 * Extract and stream thinking tokens from different model providers
 */
export declare function extractAndStreamThinkingTokens(output: LLMResult): string | undefined;
/**
 * Format final response with special handling for Gemini image models and JSON mode
 */
export declare function formatFinalResponse(response: MessageContent, modelName: string, jsonMode?: boolean): {
    response: string;
    error?: string;
};
/**
 * Convert Gemini's inlineData format to LangChain-compatible data URI format
 */
export declare function formatGeminiImageResponse(response: string | unknown): string;
//# sourceMappingURL=agent-formatter.d.ts.map