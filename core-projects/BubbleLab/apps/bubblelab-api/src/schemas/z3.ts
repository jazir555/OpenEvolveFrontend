import { createRoute, z } from '@hono/zod-openapi';
import { errorResponseSchema } from './index.js';

// ============================================================================
// Z3 SOLVER SCHEMAS
// ============================================================================

// SMT Solve Request
export const Z3SolveRequestSchema = z.object({
  smtlib2: z.string().describe('SMTLIB2 commands to execute'),
  timeout: z.number().min(1000).max(600000).default(30000).describe('Timeout in milliseconds (default: 30000)'),
  logic: z.string().optional().describe('Optional logic specification (e.g., QF_BV, LIA, AUFLIA)'),
});

export const Z3SolveResponseSchema = z.object({
  result: z.enum(['sat', 'unsat', 'unknown']).describe('Satisfiability result'),
  model: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])).optional().describe('Model (if sat)'),
  statistics: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional().describe('Solver statistics'),
  error: z.string().optional().describe('Error message if failed'),
  timing: z.number().optional().describe('Execution time in milliseconds'),
});

// Optimization Request
export const Z3OptimizeObjectiveSchema = z.object({
  expression: z.string().describe('Expression to optimize (e.g., "x")'),
  type: z.enum(['maximize', 'minimize']).describe('Optimization direction'),
});

export const Z3OptimizeRequestSchema = z.object({
  objectives: z.array(Z3OptimizeObjectiveSchema).min(1).describe('Optimization objectives'),
  constraints: z.array(z.string()).optional().describe('Optional SMTLIB2 constraint strings'),
  timeout: z.number().min(1000).max(600000).default(30000).describe('Timeout in milliseconds'),
});

export const Z3OptimizeResponseSchema = z.object({
  status: z.enum(['optimal', 'unsat', 'unknown']).describe('Optimization status'),
  model: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])).optional().describe('Model (if sat)'),
  objective_values: z.record(z.string(), z.number()).optional().describe('Objective values (if optimal)'),
  error: z.string().optional().describe('Error message if failed'),
  timing: z.number().optional().describe('Execution time in milliseconds'),
});

// Simplify Request
export const Z3SimplifyRequestSchema = z.object({
  expression: z.string().describe('Expression to simplify'),
  assumptions: z.array(z.string()).optional().describe('Optional assumption strings'),
  timeout: z.number().min(1000).max(600000).default(10000).describe('Timeout in milliseconds'),
});

export const Z3SimplifyResponseSchema = z.object({
  result: z.string().describe('Simplified expression'),
  error: z.string().optional().describe('Error message if failed'),
  timing: z.number().optional().describe('Execution time in milliseconds'),
});

// Tactic Request
export const Z3TacticRequestSchema = z.object({
  goal: z.string().describe('Goal expression to apply tactic to'),
  tactic: z.string().describe('Tactic name (e.g., "simplify", "sat", "qfnia")'),
  params: z.record(z.unknown()).optional().describe('Optional tactic parameters'),
  timeout: z.number().min(1000).max(600000).default(30000).describe('Timeout in milliseconds'),
});

export const Z3TacticResponseSchema = z.object({
  status: z.string().describe('Tactic result status'),
  goals: z.array(z.string()).optional().describe('Subgoals after applying tactic'),
  model: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])).optional().describe('Model (if sat)'),
  error: z.string().optional().describe('Error message if failed'),
  timing: z.number().optional().describe('Execution time in milliseconds'),
});

// Fixedpoint Request
export const Z3FixedpointRequestSchema = z.object({
  rules: z.array(z.string()).optional().describe('Fixedpoint rules'),
  query: z.string().optional().describe('Query to check'),
  timeout: z.number().min(1000).max(600000).default(30000).describe('Timeout in milliseconds'),
});

export const Z3FixedpointResponseSchema = z.object({
  result: z.string().describe('Query result'),
  answer: z.string().optional().describe('Answer if available'),
  error: z.string().optional().describe('Error message if failed'),
  timing: z.number().optional().describe('Execution time in milliseconds'),
});

// ============================================================================
// OPENAPI ROUTES
// ============================================================================

export const solveRoute = createRoute({
  method: 'post',
  path: '/solve',
  summary: 'Solve SMT Problem',
  description: 'Solve an SMT (Satisfiability Modulo Theories) problem using Z3',
  request: {
    body: {
      content: {
        'application/json': {
          schema: Z3SolveRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'SMT problem solved successfully',
      content: {
        'application/json': {
          schema: Z3SolveResponseSchema,
        },
      },
    },
    400: {
      description: 'Bad request or invalid SMTLIB2',
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
  tags: ['Z3'],
});

export const optimizeRoute = createRoute({
  method: 'post',
  path: '/optimize',
  summary: 'Solve Optimization Problem',
  description: 'Solve an optimization problem (maximize/minimize objectives) using Z3',
  request: {
    body: {
      content: {
        'application/json': {
          schema: Z3OptimizeRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Optimization problem solved',
      content: {
        'application/json': {
          schema: Z3OptimizeResponseSchema,
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
  tags: ['Z3'],
});

export const simplifyRoute = createRoute({
  method: 'post',
  path: '/simplify',
  summary: 'Simplify Expression',
  description: 'Simplify a mathematical expression using Z3',
  request: {
    body: {
      content: {
        'application/json': {
          schema: Z3SimplifyRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Expression simplified',
      content: {
        'application/json': {
          schema: Z3SimplifyResponseSchema,
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
  tags: ['Z3'],
});

export const tacticRoute = createRoute({
  method: 'post',
  path: '/tactic',
  summary: 'Apply Tactic',
  description: 'Apply a Z3 tactic to a goal (e.g., simplify, sat, bit-blast)',
  request: {
    body: {
      content: {
        'application/json': {
          schema: Z3TacticRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Tactic applied successfully',
      content: {
        'application/json': {
          schema: Z3TacticResponseSchema,
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
  tags: ['Z3'],
});

export const fixedpointRoute = createRoute({
  method: 'post',
  path: '/fixedpoint',
  summary: 'Fixedpoint Query',
  description: 'Perform fixedpoint computation and query using Z3',
  request: {
    body: {
      content: {
        'application/json': {
          schema: Z3FixedpointRequestSchema,
        },
      },
    },
  },
  responses: {
    200: {
      description: 'Fixedpoint query completed',
      content: {
        'application/json': {
          schema: Z3FixedpointResponseSchema,
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
  tags: ['Z3'],
});

export const getTacticsRoute = createRoute({
  method: 'get',
  path: '/tactics',
  summary: 'Get Available Tactics',
  description: 'Get list of available Z3 tactics',
  responses: {
    200: {
      description: 'List of available tactics',
      content: {
        'application/json': {
          schema: z.array(z.object({
            name: z.string(),
            description: z.string().optional(),
          })),
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
  tags: ['Z3'],
});

export const getLogicsRoute = createRoute({
  method: 'get',
  path: '/logics',
  summary: 'Get Supported Logics',
  description: 'Get list of supported SMT logics',
  responses: {
    200: {
      description: 'List of supported logics',
      content: {
        'application/json': {
          schema: z.array(z.string()),
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
  tags: ['Z3'],
});

export const getVersionRoute = createRoute({
  method: 'get',
  path: '/version',
  summary: 'Get Z3 Version',
  description: 'Get Z3 version information',
  responses: {
    200: {
      description: 'Z3 version information',
      content: {
        'application/json': {
          schema: z.object({
            version: z.string(),
            full_version: z.string().optional(),
          }),
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
  tags: ['Z3'],
});

export const healthRoute = createRoute({
  method: 'get',
  path: '/health',
  summary: 'Health Check',
  description: 'Check if Z3 service is available',
  responses: {
    200: {
      description: 'Z3 service is healthy',
      content: {
        'application/json': {
          schema: z.object({
            status: z.literal('ok'),
            z3_available: z.boolean(),
            version: z.string().optional(),
          }),
        },
      },
    },
    503: {
      description: 'Z3 service is unavailable or unhealthy',
      content: {
        'application/json': {
          schema: z.object({
            status: z.literal('degraded'),
            z3_available: z.literal(false),
            error: z.string().optional(),
          }),
        },
      },
    },
  },
  tags: ['Z3'],
});
