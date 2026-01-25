import { z } from 'zod';

/**
 * Common validation schemas
 */
export const validationSchemas = {
  // Workflow validation
  workflow: z.object({
    name: z.string().min(1, 'Workflow name is required'),
    type: z.string().min(1, 'Workflow type is required'),
    config: z.record(z.unknown()),
  }),

  // Artifact validation
  artifact: z.object({
    type: z.string().min(1, 'Artifact type is required'),
    name: z.string().min(1, 'Artifact name is required'),
    content: z.unknown(),
    tags: z.array(z.string()).optional(),
  }),

  // Lean proof validation
  leanProof: z.object({
    theorem: z.string().min(1, 'Theorem is required'),
    model: z.string().min(1, 'Model is required'),
  }),

  // Knowledge search validation
  searchQuery: z.object({
    query: z.string().min(1, 'Search query is required'),
    type: z.string().optional(),
    tags: z.array(z.string()).optional(),
    page: z.number().int().positive().optional(),
    pageSize: z.number().int().positive().optional(),
  }),
};

/**
 * Validate data against a schema
 */
export function validateData<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; error: string } {
  try {
    const validated = schema.parse(data);
    return { success: true, data: validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { success: false, error: error.errors[0].message };
    }
    return { success: false, error: 'Validation failed' };
  }
}
