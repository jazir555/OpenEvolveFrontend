import { z } from 'zod';
export interface BubbleOperationResult {
    success: boolean;
    error: string;
}
export interface BubbleResult<T> extends BubbleOperationResult {
    data: T;
    executionId: string;
    timestamp: Date;
}
/**
 * Utility class for generating mock data from Zod schemas
 * Useful for testing, development, and creating sample data
 */
export declare class MockDataGenerator {
    /**
     * Generate a complete mock BubbleResult from a result schema
     */
    static generateMockResult<TResult extends BubbleOperationResult>(resultSchema: z.ZodObject<z.ZodRawShape>): BubbleResult<TResult>;
    /**
     * Generate mock data from JSON Schema
     * Converts JSON Schema to mock data with realistic values
     */
    static generateMockFromJsonSchema(jsonSchema: Record<string, unknown>): Record<string, unknown>;
    /**
     * Generate a mock value for a specific JSON Schema property
     */
    static generateMockValueFromJsonSchema(schema: Record<string, unknown>): unknown;
    /**
     * Generate mock data object from a Zod schema
     * Recursively handles nested objects, arrays, and primitive types
     */
    static generateMockFromSchema(schema: z.ZodObject<z.ZodRawShape>): Record<string, unknown>;
    /**
     * Generate a mock value for a specific Zod type
     */
    static generateMockValue(zodType: z.ZodTypeAny): unknown;
    /**
     * Generate mock string values with format-specific handling
     */
    private static generateMockString;
    /**
     * Generate mock number values respecting constraints
     */
    private static generateMockNumber;
    /**
     * Generate mock data with custom seed for reproducible results
     */
    static generateMockWithSeed<TResult extends BubbleOperationResult>(resultSchema: z.ZodObject<z.ZodRawShape>, seed: number): BubbleResult<TResult>;
}
//# sourceMappingURL=mock-data-generator.d.ts.map