/**
 * Sanitizes user script code by blocking access to process.env.
 * Transforms the code to replace process.env references with a throw statement.
 *
 * This prevents users from accessing sensitive environment variables
 * without affecting the global process.env used by the API server.
 *
 * @param code - The code to sanitize
 */
export declare function sanitizeScript(code: string): string;
//# sourceMappingURL=sanitize-script.d.ts.map