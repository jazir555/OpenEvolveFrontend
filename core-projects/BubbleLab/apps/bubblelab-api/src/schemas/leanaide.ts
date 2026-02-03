import { createRoute, z } from '@hono/zod-openapi';
import { errorResponseSchema } from './index.js';

// LeanAide Request/Response Schemas
export const LeanAideGenerateRequestSchema = z.object({
  theorem: z.string().openapi({
    description: 'The theorem statement to prove',
    example: ' theorem add_comm (a b : Nat) : a + b = b + a',
  }),
  proof_attempt: z.string().optional().openapi({
    description: 'Optional existing proof attempt to improve',
    example: ' by induction b',
  }),
  model: z.string().openapi({
    description: 'LLM model to use for proof generation',
    example: 'gpt-4',
  }),
  temperature: z.number().min(0).max(2).openapi({
    description: 'Temperature for generation (0-2)',
    example: 0.7,
  }),
});

export const LeanCodeOutputSchema = z.object({
  success: z.boolean(),
  lean_code: z.string().optional(),
  proof: z.string().optional(),
  error: z.string().optional(),
  logs: z.string().optional(),
  response_time: z.number().optional(),
});

export const LeanAideVerifyRequestSchema = z.object({
  code: z.string().openapi({
    description: 'Lean 4 code to verify',
    example: ' theorem add_comm (a b : Nat) : a + b = b + a := by induction b',
  }),
});

export const VerificationResultSchema = z.object({
  valid: z.boolean(),
  errors: z.array(z.string()).optional(),
  warnings: z.array(z.string()).optional(),
  tactic_state: z.string().optional(),
  goals: z.array(z.string()).optional(),
});

export const ModelInfoSchema = z.object({
  provider: z.string(),
  models: z.array(z.string()),
});

export const LeanAideBenchmarkRequestSchema = z.object({
  dataset: z.array(z.any()).openapi({
    description: 'Array of theorem statements to benchmark',
  }),
  model: z.string().openapi({
    description: 'Model to use for benchmarking',
  }),
  evaluator: z.string().openapi({
    description: 'Evaluation method',
  }),
});

export const BenchmarkStartResponseSchema = z.object({
  benchmark_id: z.string(),
  status: z.string(),
  message: z.string().optional(),
});

export const BenchmarkResultSchema = z.object({
  benchmark_id: z.string(),
  status: z.string(),
  results: z.array(z.any()).optional(),
  total: z.number().optional(),
  successful: z.number().optional(),
  failed: z.number().optional(),
  avg_time: z.number().optional(),
});

// API Routes
export const generateProofRoute = createRoute({
  method: 'post',
  path: '/generate',
  summary: 'Generate Lean 4 Proof',
  description: 'Generate or improve a Lean 4 proof for a given theorem',
  request: {
    body: {
      content: {
        'application/json': {
          schema: LeanAideGenerateRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Proof generated successfully',
      content: {
        'application/json': {
          schema: LeanCodeOutputSchema,
        },
      },
    },
    400: {
      description: 'Bad request',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
    500: {
      description: 'Internal server error',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
    504: {
      description: 'Request timeout',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
  },
  tags: ['LeanAide'],
});

export const verifyProofRoute = createRoute({
  method: 'post',
  path: '/verify',
  summary: 'Verify Lean 4 Proof',
  description: 'Verify Lean 4 code against the Lean 4 kernel',
  request: {
    body: {
      content: {
        'application/json': {
          schema: LeanAideVerifyRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Proof verified',
      content: {
        'application/json': {
          schema: VerificationResultSchema,
        },
      },
    },
    400: {
      description: 'Bad request',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
    500: {
      description: 'Internal server error',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
  },
  tags: ['LeanAide'],
});

export const getModelsRoute = createRoute({
  method: 'get',
  path: '/models',
  summary: 'Get Available Models',
  description: 'Get list of available LLM models for LeanAide',
  responses: {
    200: {
      description: 'Available models',
      content: {
        'application/json': {
          schema: z.array(ModelInfoSchema),
        },
      },
    },
    500: {
      description: 'Internal server error',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
  },
  tags: ['LeanAide'],
});

export const runBenchmarkRoute = createRoute({
  method: 'post',
  path: '/benchmark/start',
  summary: 'Start Benchmark',
  description: 'Start a new benchmark run on a dataset of theorems',
  request: {
    body: {
      content: {
        'application/json': {
          schema: LeanAideBenchmarkRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Benchmark started',
      content: {
        'application/json': {
          schema: BenchmarkStartResponseSchema,
        },
      },
    },
    400: {
      description: 'Bad request',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
    500: {
      description: 'Internal server error',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
  },
  tags: ['LeanAide'],
});

export const getBenchmarkResultsRoute = createRoute({
  method: 'get',
  path: '/benchmark/{benchmarkId}/results',
  summary: 'Get Benchmark Results',
  description: 'Get results for a completed or running benchmark',
  request: {
    params: z.object({
      benchmarkId: z.string().openapi({
        description: 'Benchmark ID',
        example: 'bm_1234567890',
      }),
    }),
  },
  responses: {
    200: {
      description: 'Benchmark results',
      content: {
        'application/json': {
          schema: BenchmarkResultSchema,
        },
      },
    },
    404: {
      description: 'Benchmark not found',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
    500: {
      description: 'Internal server error',
      content: {
        'application/json': {
          schema: errorResponseSchema,
        },
      },
    },
  },
  tags: ['LeanAide'],
});
