import { createRoute, z } from '@hono/zod-openapi';
import { errorResponseSchema, successMessageResponseSchema } from './index.js';

export const evolutionRunSchema = z.object({
  id: z.number(),
  evolutionId: z.string(),
  status: z.string(),
  name: z.string().optional(),
  config: z.record(z.string(), z.unknown()).nullable().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

export const evolutionNodeSchema = z.object({
  id: z.number(),
  runId: z.number(),
  nodeId: z.string(),
  parentNodeId: z.string().nullable().optional(),
  generation: z.number(),
  status: z.string(),
  fitness: z.number().nullable().optional(),
  score: z.number().nullable().optional(),
  label: z.string().nullable().optional(),
  htmlAssetId: z.number().nullable().optional(),
  thumbnailAssetId: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

export const evolutionAssetSchema = z.object({
  id: z.number(),
  url: z.string(),
  contentType: z.string(),
  size: z.number(),
});

export const evolutionUsageSchema = z.object({
  totalBytes: z.number(),
  totalAssets: z.number(),
  htmlBytes: z.number(),
  htmlCount: z.number(),
  thumbnailBytes: z.number(),
  thumbnailCount: z.number(),
});

export const evolutionThumbnailCleanupSchema = z.object({
  message: z.string(),
  removedCount: z.number(),
  freedBytes: z.number(),
});

const createEvolutionRunSchema = z.object({
  evolutionId: z.string(),
  status: z.string().optional(),
  name: z.string().optional(),
  config: z.record(z.string(), z.unknown()).nullable().optional(),
  idempotencyKey: z.string().optional(), // BUG #14 FIX: Add idempotency key support
});

const upsertEvolutionNodeSchema = z.object({
  runId: z.number(),
  nodeId: z.string(),
  parentNodeId: z.string().nullable().optional(),
  generation: z.number(),
  status: z.string(),
  fitness: z.number().nullable().optional(),
  score: z.number().nullable().optional(),
  label: z.string().nullable().optional(),
  htmlAssetId: z.number().nullable().optional(),
  thumbnailAssetId: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
});

const createEvolutionAssetSchema = z.object({
  runId: z.number(),
  kind: z.enum(['html', 'thumbnail']),
  contentType: z.string(),
  dataBase64: z.string(),
  filename: z.string().optional(),
});

export const createEvolutionRunRoute = createRoute({
  method: 'post',
  path: '/runs',
  request: {
    body: {
      content: {
        'application/json': {
          schema: createEvolutionRunSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: evolutionRunSchema,
        },
      },
      description: 'Evolution run created',
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
  tags: ['EvolutionGraph'],
});

export const listEvolutionRunsRoute = createRoute({
  method: 'get',
  path: '/runs',
  responses: {
    200: {
      content: {
        'application/json': {
          schema: z.array(evolutionRunSchema),
        },
      },
      description: 'List evolution runs',
    },
    500: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Server error',
    },
  },
  tags: ['EvolutionGraph'],
});

export const listEvolutionNodesRoute = createRoute({
  method: 'get',
  path: '/runs/{runId}/nodes',
  request: {
    params: z.object({
      runId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: z.array(evolutionNodeSchema),
        },
      },
      description: 'List evolution nodes',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Run not found',
    },
  },
  tags: ['EvolutionGraph'],
});

export const getEvolutionUsageRoute = createRoute({
  method: 'get',
  path: '/runs/{runId}/usage',
  request: {
    params: z.object({
      runId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: evolutionUsageSchema,
        },
      },
      description: 'Evolution storage usage',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Run not found',
    },
  },
  tags: ['EvolutionGraph'],
});

export const clearEvolutionNodesRoute = createRoute({
  method: 'delete',
  path: '/runs/{runId}/nodes',
  request: {
    params: z.object({
      runId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: successMessageResponseSchema,
        },
      },
      description: 'Evolution nodes cleared',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Run not found',
    },
  },
  tags: ['EvolutionGraph'],
});

export const deleteEvolutionRunRoute = createRoute({
  method: 'delete',
  path: '/runs/{runId}',
  request: {
    params: z.object({
      runId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: successMessageResponseSchema,
        },
      },
      description: 'Evolution run deleted',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Run not found',
    },
  },
  tags: ['EvolutionGraph'],
});

export const clearEvolutionThumbnailsRoute = createRoute({
  method: 'delete',
  path: '/runs/{runId}/thumbnails',
  request: {
    params: z.object({
      runId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: evolutionThumbnailCleanupSchema,
        },
      },
      description: 'Evolution thumbnails cleared',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Run not found',
    },
  },
  tags: ['EvolutionGraph'],
});

export const upsertEvolutionNodeRoute = createRoute({
  method: 'post',
  path: '/nodes',
  request: {
    body: {
      content: {
        'application/json': {
          schema: upsertEvolutionNodeSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: evolutionNodeSchema,
        },
      },
      description: 'Evolution node upserted',
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
  tags: ['EvolutionGraph'],
});

export const createEvolutionAssetRoute = createRoute({
  method: 'post',
  path: '/assets',
  request: {
    body: {
      content: {
        'application/json': {
          schema: createEvolutionAssetSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: evolutionAssetSchema,
        },
      },
      description: 'Evolution asset stored',
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
  tags: ['EvolutionGraph'],
});

export const getEvolutionAssetRoute = createRoute({
  method: 'get',
  path: '/assets/{assetId}',
  request: {
    params: z.object({
      assetId: z.string().regex(/^[0-9]+$/),
    }),
  },
  responses: {
    200: {
      description: 'Evolution asset file',
    },
    404: {
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
      description: 'Asset not found',
    },
  },
  tags: ['EvolutionGraph'],
});
