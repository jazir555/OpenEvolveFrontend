/**
 * Arbor Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Arbor
 * (tree-based data structures and algorithms) interactions.
 */

import { z } from 'zod';

/**
 * Tree Type Enum
 */
export const TreeType = z.enum([
  'binary_tree',
  'n_ary_tree',
  'bst',
  'avl',
  'red_black',
  'b_tree',
  'trie',
  'segment_tree',
  'fenwick',
]);

export type TreeType = z.infer<typeof TreeType>;

/**
 * Tree Traversal Enum
 */
export const TraversalOrder = z.enum([
  'pre_order',
  'in_order',
  'post_order',
  'level_order',
]);

export type TraversalOrder = z.infer<typeof TraversalOrder>;

/**
 * Tree Node Schema
 */
export const TreeNode = z.object({
  node_id: z.string().describe("Unique node identifier"),

  value: z.any().describe("Node value"),

  children: z.array(z.string()).optional().describe("Child node IDs"),

  parent: z.string().optional().describe("Parent node ID"),

  metadata: z.object({
    depth: z.number().int().min(0).optional().describe("Node depth"),
    height: z.number().int().min(0).optional().describe("Node height"),
    size: z.number().int().positive().optional().describe("Subtree size"),
    balance_factor: z.number().optional().describe("Balance factor (AVL)"),
    color: z.enum(['red', 'black']).optional().describe("Node color (Red-Black)"),
  }).optional().describe("Node metadata"),

  properties: z.record(z.any()).optional().describe("Custom properties"),
});

export type TreeNode = z.infer<typeof TreeNode>;

/**
 * Tree Schema
 */
export const Tree = z.object({
  tree_id: z.string().describe("Tree identifier"),

  tree_type: TreeType.describe("Type of tree"),

  root_id: z.string().optional().describe("Root node ID"),

  nodes: z.record(z.any()).optional().describe("All nodes in tree"),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    updated_at: z.string().datetime().optional(),
    node_count: z.number().optional(),
    height: z.number().optional(),
    max_degree: z.number().optional().describe("Maximum children per node"),
  }).optional().describe("Tree metadata"),
});

export type Tree = z.infer<typeof Tree>;

/**
 * Tree Operation Schema
 */
export const TreeOperation = z.object({
  operation: z.enum([
    'insert',
    'delete',
    'search',
    'update',
    'traverse',
    'query',
    'balance',
    'merge',
    'split',
  ]).describe("Operation to perform"),

  target: z.object({
    node_id: z.string().optional().describe("Target node ID"),
    value: z.any().optional().describe("Target value"),
    path: z.array(z.string()).optional().describe("Path from root"),
  }).optional().describe("Operation target"),

  parameters: z.object({
    position: z.enum(['left', 'right', 'root']).optional().describe("Insert position"),
    traversal_order: TraversalOrder.optional().describe("Traversal order"),
    range: z.tuple([z.any(), z.any()]).optional().describe("Range query"),
    comparison_key: z.string().optional().describe("Key for comparisons"),
  }).optional().describe("Operation parameters"),

  timeout_ms: z.number().int().positive().max(30000).optional(),
});

export type TreeOperation = z.infer<typeof TreeOperation>;

/**
 * Arbor Request Schema
 */
export const ArborRequest = z.object({
  tree_id: z.string().optional().describe("Tree identifier (optional for creation)"),

  action: z.enum([
    'create_tree',
    'destroy_tree',
    'execute_operation',
    'batch_operations',
    'clone_tree',
    'export_tree',
    'import_tree',
  ]).describe("Action to perform"),

  tree_config: z.object({
    tree_type: TreeType.optional(),
    comparison_key: z.string().optional().describe("Key for ordered trees"),
    max_children: z.number().int().positive().optional().describe("Max children (n-ary)"),
    order: z.number().int().positive().optional().describe("Tree order (B-tree)"),
    auto_balance: z.boolean().optional().describe("Auto-balance on insert/delete"),
  }).optional().describe("Tree configuration"),

  node: TreeNode.optional().describe("Node to insert"),

  operation: TreeOperation.optional().describe("Operation to execute"),

  operations: z.array(TreeOperation).optional().describe("Batch operations"),

  timeout_ms: z.number()
    .int().positive().max(60000)
    .describe("Request timeout (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),

  metadata: z.record(z.any()).optional(),
});

export type ArborRequest = z.infer<typeof ArborRequest>;

/**
 * Arbor Response Schema
 */
export const ArborResponse = z.object({
  tree_id: z.string().describe("Tree identifier"),

  action: z.enum([
    'create_tree',
    'destroy_tree',
    'execute_operation',
    'batch_operations',
    'clone_tree',
    'export_tree',
    'import_tree',
  ]).describe("Action performed"),

  status: z.enum([
    'success',
    'failed',
    'timeout',
  ]).describe("Action status"),

  result: z.object({
    tree: Tree.optional().describe("Resulting tree"),
    node_id: z.string().optional().describe("Created/updated node ID"),
    nodes: z.array(TreeNode).optional().describe("Traversal/query results"),
    found: z.boolean().optional().describe("Search result"),
    value: z.any().optional().describe("Retrieved value"),
    traversal: z.array(z.any()).optional().describe("Traversal result"),
    cloned_tree_id: z.string().optional().describe("Cloned tree ID"),
    export_data: z.record(z.any()).optional().describe("Exported tree data"),
    operations_completed: z.number().optional().describe("Batch operations count"),
    stats: z.object({
      node_count: z.number().optional(),
      height: z.number().optional(),
      balance_factor: z.number().optional(),
      operation_time_ms: z.number().optional(),
    }).optional().describe("Tree statistics"),
  }).optional().describe("Action result"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional(),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    updated_at: z.string().datetime().optional(),
    processing_time_ms: z.number().optional(),
  }).optional(),

  correlation_id: z.string().uuid().optional(),

  timestamp: z.string().datetime(),
});

export type ArborResponse = z.infer<typeof ArborResponse>;

/**
 * Error Model
 */
export const ArborError = z.object({
  code: z.enum([
    'TREE_NOT_FOUND',
    'NODE_NOT_FOUND',
    'INVALID_TREE_TYPE',
    'INVALID_OPERATION',
    'DUPLICATE_NODE',
    'TREE_VIOLATION',
    'BALANCE_ERROR',
    'VALIDATION_ERROR',
    'UNKNOWN_ERROR',
  ]),
  message: z.string(),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type ArborError = z.infer<typeof ArborError>;

/**
 * Validation Functions
 */
export function validateArborRequest(data: unknown): {
  success: boolean;
  data?: ArborRequest;
  errors?: string[];
} {
  const result = ArborRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function isArborRequest(data: unknown): data is ArborRequest {
  return typeof data === 'object' && data !== null &&
    'action' in data;
}

/**
 * Examples
 */
export const ArborExamples = {
  validCreateTree: {
    action: "create_tree" as const,
    tree_config: {
      tree_type: "bst" as const,
      comparison_key: "value",
      auto_balance: true,
    },
    timeout_ms: 5000,
  } as ArborRequest,

  validInsert: {
    tree_id: "tree_001",
    action: "execute_operation" as const,
    node: {
      node_id: "node_001",
      value: 42,
    },
    operation: {
      operation: "insert" as const,
    },
    timeout_ms: 5000,
  } as ArborRequest,

  validTraversal: {
    tree_id: "tree_001",
    action: "execute_operation" as const,
    operation: {
      operation: "traverse" as const,
      parameters: {
        traversal_order: "in_order" as const,
      },
    },
    timeout_ms: 5000,
  } as ArborRequest,
};
