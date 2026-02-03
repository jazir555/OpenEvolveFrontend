import { createRoute, z } from '@hono/zod-openapi';
import { errorResponseSchema } from './index.js';

const designSchema = z.object({
  html: z.string(),
  css: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

const mutationRequestSchema = z.object({
  design: designSchema,
  mutation_types: z.array(z.string()).optional(),
  constraints: z.record(z.string(), z.unknown()).optional(),
});

const mutationResultSchema = z.object({
  html: z.string(),
  css: z.string().optional(),
  changes: z.array(z.string()),
});

const mutationBatchRequestSchema = z.object({
  items: z.array(mutationRequestSchema),
  max_concurrency: z.number().min(1).max(50).optional(),
});

export const mutateRoute = createRoute({
  method: 'post',
  path: '/mutate',
  request: {
    body: {
      content: {
        'application/json': {
          schema: mutationRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: mutationResultSchema,
        },
      },
      description: 'Mutate a single design',
    },
    400: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Invalid request',
    },
  },
  tags: ['EvolutionMutation'],
});

export const mutateBatchRoute = createRoute({
  method: 'post',
  path: '/mutate/batch',
  request: {
    body: {
      content: {
        'application/json': {
          schema: mutationBatchRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: z.object({ results: z.array(mutationResultSchema) }),
        },
      },
      description: 'Mutate a batch of designs',
    },
    400: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Invalid request',
    },
  },
  tags: ['EvolutionMutation'],
});
