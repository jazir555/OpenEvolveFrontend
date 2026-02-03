import { createRoute, z } from '@hono/zod-openapi';
import { errorResponseSchema } from './index.js';

const base64ImageSchema = z.object({
  type: z.literal('base64'),
  data: z.string().min(1),
  mimeType: z.string().optional(),
  description: z.string().optional(),
});

const urlImageSchema = z.object({
  type: z.literal('url'),
  url: z.string().url(),
  description: z.string().optional(),
});

const judgeImageSchema = z.discriminatedUnion('type', [
  base64ImageSchema,
  urlImageSchema,
]);

export const judgeInputSchema = z.object({
  image: judgeImageSchema,
  criteria: z.string().optional(),
  html: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

const judgeRequestSchema = z.object({
  input: judgeInputSchema,
  weights: z.record(z.string(), z.number()).optional(),
});

const judgeBatchRequestSchema = z.object({
  inputs: z.array(judgeInputSchema),
  weights: z.record(z.string(), z.number()).optional(),
  maxConcurrency: z.number().min(1).max(10).optional(),
});

const judgeEvaluationSchema = z.object({
  agent: z.string(),
  provider: z.string(),
  score: z.number(),
  reasoning: z.string(),
  highlights: z.array(z.string()),
  issues: z.array(z.string()),
  recommendations: z.array(z.string()),
  rawResponse: z.string(),
  costUsd: z.number(),
});

const judgeAggregateSchema = z.object({
  score: z.number(),
  weights: z.record(z.string(), z.number()),
  agents: z.array(judgeEvaluationSchema),
});

export const judgeRoute = createRoute({
  method: 'post',
  path: '/judge',
  request: {
    body: {
      content: {
        'application/json': {
          schema: judgeRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: judgeAggregateSchema,
        },
      },
      description: 'Evaluate a single design',
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
  tags: ['EvolutionJudge'],
});

export const judgeBatchRoute = createRoute({
  method: 'post',
  path: '/judge/batch',
  request: {
    body: {
      content: {
        'application/json': {
          schema: judgeBatchRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      content: {
        'application/json': {
          schema: z.array(judgeAggregateSchema),
        },
      },
      description: 'Evaluate a batch of designs',
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
  tags: ['EvolutionJudge'],
});
