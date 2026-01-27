import type { ZodTypeAny } from 'zod';
/**
 * Checks if a value is a Zod schema object
 * Detects Zod schemas by checking for the internal _def property with typeName
 */
export declare function isZodSchema(value: unknown): value is ZodTypeAny;
/**
 * Converts a Zod schema or JSON schema string to a JSON schema string
 * If already a string, returns as-is. If Zod schema, converts to JSON schema.
 *
 * @param schema - Either a Zod schema object or a JSON schema string
 * @param schemaName - Optional name for the schema (default: 'OutputSchema')
 * @returns JSON schema as a string
 */
export declare function zodSchemaToJsonString(schema: ZodTypeAny | string, schemaName?: string): string;
/**
 * Builds a system prompt instruction for JSON output with a specific schema
 * This standardizes how we tell LLMs to output structured JSON
 *
 * @param schemaString - The JSON schema as a string
 * @returns Instruction text to append to system prompt
 */
export declare function buildJsonSchemaInstruction(schemaString: string): string;
//# sourceMappingURL=zod-schema.d.ts.map