import { ChatGoogleGenerativeAI, type GoogleGenerativeAIChatInput } from '@langchain/google-genai';
import type { BaseMessage } from '@langchain/core/messages';
import { AIMessage } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
/**
 * SafeGeminiChat: A wrapper around ChatGoogleGenerativeAI that gracefully handles
 * the candidateContent.parts.reduce error when Gemini returns blocked/empty content.
 *
 * Issue: When Gemini blocks content (safety filters, policy violations, etc.), it sometimes
 * returns candidates without a `content` field, causing `candidateContent.parts.reduce` to crash.
 *
 * Solution: This wrapper catches these errors and returns a structured error response instead
 * of crashing the entire agent workflow.
 */
export declare class SafeGeminiChat extends ChatGoogleGenerativeAI {
    constructor(fields: GoogleGenerativeAIChatInput);
    /**
     * Override the _generate method to add null-safety checks
     */
    _generate(messages: BaseMessage[], options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): Promise<ChatResult>;
}
/**
 * Utility function to check if an AIMessage contains a Gemini error
 */
export declare function isGeminiErrorResponse(message: AIMessage): boolean;
/**
 * Utility function to extract the original error from a Gemini error response
 */
export declare function getGeminiErrorDetails(message: AIMessage): string | undefined;
//# sourceMappingURL=safe-gemini-chat.d.ts.map