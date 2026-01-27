import { z } from 'zod';
/**
 * Represents the difference between actual data and a Zod schema
 */
export interface SchemaDifference {
    /** Fields present in data but not defined in schema */
    extraFields: FieldInfo[];
    /** Fields defined in schema but missing from data (and are required) */
    missingRequiredFields: FieldInfo[];
    /** Fields defined in schema but missing from data (optional fields) */
    missingOptionalFields: FieldInfo[];
    /** Fields with type mismatches */
    typeMismatches: TypeMismatch[];
    /** Nested object differences */
    nestedDifferences: Record<string, SchemaDifference>;
    /** Summary statistics */
    summary: {
        totalExtraFields: number;
        totalMissingRequired: number;
        totalMissingOptional: number;
        totalTypeMismatches: number;
        isCompatible: boolean;
    };
}
export interface FieldInfo {
    path: string;
    value?: unknown;
    type?: string;
}
export interface TypeMismatch {
    path: string;
    expectedType: string;
    actualType: string;
    actualValue: unknown;
}
/**
 * Compare data against a Zod schema and find all differences
 *
 * @param schema - The Zod schema to compare against
 * @param data - The actual data to compare
 * @param basePath - Base path for nested field names (used internally)
 * @returns Detailed comparison results showing all differences
 *
 * @example
 * ```typescript
 * const schema = z.object({
 *   name: z.string(),
 *   age: z.number().optional(),
 * });
 *
 * const data = {
 *   name: 'John',
 *   extraField: 'unexpected',
 * };
 *
 * const diff = compareWithSchema(schema, data);
 * // diff.extraFields = [{ path: 'extraField', value: 'unexpected', type: 'string' }]
 * // diff.missingOptionalFields = [{ path: 'age' }]
 * ```
 */
export declare function compareWithSchema(schema: z.ZodTypeAny, data: unknown, basePath?: string): SchemaDifference;
/**
 * Format schema differences as a human-readable string
 */
export declare function formatSchemaDifference(diff: SchemaDifference, indent?: number): string;
export interface FieldWithIndex {
    fieldName: string;
    index: number;
}
/**
 * Compare multiple data items against a schema and aggregate differences
 * Useful for analyzing API responses with multiple items
 */
export declare function compareMultipleWithSchema(schema: z.ZodTypeAny, items: unknown[], maxItems?: number): {
    itemCount: number;
    status: 'PASS' | 'FAIL';
    allExtraFields: FieldWithIndex[];
    allMissingRequired: FieldWithIndex[];
    allMissingOptional: FieldWithIndex[];
    allTypeMismatches: Map<string, Set<string>>;
    sampleDifferences: SchemaDifference[];
    summary: string;
};
/**
 * Format aggregated differences from multiple items in a git-diff style
 */
export declare function formatAggregatedDifferences(result: ReturnType<typeof compareMultipleWithSchema>): string;
/**
 * Format aggregated differences with per-item breakdown
 */
export declare function formatAggregatedDifferencesDetailed(result: ReturnType<typeof compareMultipleWithSchema>, schema: z.ZodTypeAny): string;
//# sourceMappingURL=schema-comparison.d.ts.map