import { z } from 'zod';
/**
 * Common validation schemas
 */
export declare const validationSchemas: {
    workflow: z.ZodObject<{
        name: z.ZodString;
        type: z.ZodString;
        config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        config?: Record<string, unknown>;
        name?: string;
        type?: string;
    }, {
        config?: Record<string, unknown>;
        name?: string;
        type?: string;
    }>;
    artifact: z.ZodObject<{
        type: z.ZodString;
        name: z.ZodString;
        content: z.ZodUnknown;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        name?: string;
        content?: unknown;
        tags?: string[];
        type?: string;
    }, {
        name?: string;
        content?: unknown;
        tags?: string[];
        type?: string;
    }>;
    leanProof: z.ZodObject<{
        theorem: z.ZodString;
        model: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        theorem?: string;
        model?: string;
    }, {
        theorem?: string;
        model?: string;
    }>;
    searchQuery: z.ZodObject<{
        query: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        page: z.ZodOptional<z.ZodNumber>;
        pageSize: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        tags?: string[];
        query?: string;
        type?: string;
        page?: number;
        pageSize?: number;
    }, {
        tags?: string[];
        query?: string;
        type?: string;
        page?: number;
        pageSize?: number;
    }>;
};
/**
 * Validate data against a schema
 */
export declare function validateData<T>(schema: z.ZodSchema<T>, data: unknown): {
    success: true;
    data: T;
} | {
    success: false;
    error: string;
};
