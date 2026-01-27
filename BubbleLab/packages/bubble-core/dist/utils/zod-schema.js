import { zodToJsonSchema } from 'zod-to-json-schema';
/**
 * Checks if a value is a Zod schema object
 * Detects Zod schemas by checking for the internal _def property with typeName
 */
export function isZodSchema(value) {
    return (typeof value === 'object' &&
        value !== null &&
        '_def' in value &&
        typeof value._def === 'object' &&
        value._def?.typeName !== undefined);
}
/**
 * Converts a Zod schema or JSON schema string to a JSON schema string
 * If already a string, returns as-is. If Zod schema, converts to JSON schema.
 *
 * @param schema - Either a Zod schema object or a JSON schema string
 * @param schemaName - Optional name for the schema (default: 'OutputSchema')
 * @returns JSON schema as a string
 */
export function zodSchemaToJsonString(schema, schemaName = 'OutputSchema') {
    if (typeof schema === 'string') {
        return schema;
    }
    if (isZodSchema(schema)) {
        return JSON.stringify(zodToJsonSchema(schema, schemaName));
    }
    // Fallback: try to stringify if it's an object
    return JSON.stringify(schema);
}
/**
 * Builds a system prompt instruction for JSON output with a specific schema
 * This standardizes how we tell LLMs to output structured JSON
 *
 * @param schemaString - The JSON schema as a string
 * @returns Instruction text to append to system prompt
 */
export function buildJsonSchemaInstruction(schemaString) {
    return `
OUTPUT FORMAT REQUIREMENTS:
You MUST return your response as valid JSON that matches this exact schema:
${schemaString}

CRITICAL JSON RULES:
- Return ONLY valid JSON - no markdown code blocks, no explanations, no prefixes
- Start your response directly with { or [ (the JSON structure)
- End your response with } or ] (closing the JSON structure)
- Use double quotes for all strings and property names
- No trailing commas
- No single quotes
- No unescaped newlines in strings
- Properly escape special characters in strings
- Ensure all required schema fields are present
`.trim();
}
//# sourceMappingURL=zod-schema.js.map