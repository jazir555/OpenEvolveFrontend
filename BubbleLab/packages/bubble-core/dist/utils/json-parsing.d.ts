/**
 * JSON parsing utilities extracted from AI Agent for reusability and testing
 */
/**
 * Extract JSON from mixed text/JSON responses with robust cleanup
 */
export declare function extractAndCleanJSON(input: string): string | null;
/**
 * Post-process potentially malformed JSON string to fix common issues
 */
export declare function postProcessJSON(jsonString: string): string;
/**
 * Multi-stage JSON parsing approach with various fallback strategies
 */
export declare function parseJsonWithFallbacks(finalResponse: string): {
    response: string;
    parsed: object | null;
    success: boolean;
    error?: string;
};
//# sourceMappingURL=json-parsing.d.ts.map