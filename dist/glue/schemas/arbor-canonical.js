"use strict";
/**
 * Arbor Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Arbor
 * (tree-based data structures and algorithms) interactions.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ArborExamples = exports.ArborError = exports.ArborResponse = exports.ArborRequest = exports.TreeOperation = exports.Tree = exports.TreeNode = exports.TraversalOrder = exports.TreeType = void 0;
exports.validateArborRequest = validateArborRequest;
exports.isArborRequest = isArborRequest;
const zod_1 = require("zod");
/**
 * Tree Type Enum
 */
exports.TreeType = zod_1.z.enum([
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
/**
 * Tree Traversal Enum
 */
exports.TraversalOrder = zod_1.z.enum([
    'pre_order',
    'in_order',
    'post_order',
    'level_order',
]);
/**
 * Tree Node Schema
 */
exports.TreeNode = zod_1.z.object({
    node_id: zod_1.z.string().describe("Unique node identifier"),
    value: zod_1.z.any().describe("Node value"),
    children: zod_1.z.array(zod_1.z.string()).optional().describe("Child node IDs"),
    parent: zod_1.z.string().optional().describe("Parent node ID"),
    metadata: zod_1.z.object({
        depth: zod_1.z.number().int().min(0).optional().describe("Node depth"),
        height: zod_1.z.number().int().min(0).optional().describe("Node height"),
        size: zod_1.z.number().int().positive().optional().describe("Subtree size"),
        balance_factor: zod_1.z.number().optional().describe("Balance factor (AVL)"),
        color: zod_1.z.enum(['red', 'black']).optional().describe("Node color (Red-Black)"),
    }).optional().describe("Node metadata"),
    properties: zod_1.z.record(zod_1.z.any()).optional().describe("Custom properties"),
});
/**
 * Tree Schema
 */
exports.Tree = zod_1.z.object({
    tree_id: zod_1.z.string().describe("Tree identifier"),
    tree_type: exports.TreeType.describe("Type of tree"),
    root_id: zod_1.z.string().optional().describe("Root node ID"),
    nodes: zod_1.z.record(zod_1.z.any()).optional().describe("All nodes in tree"),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        updated_at: zod_1.z.string().datetime().optional(),
        node_count: zod_1.z.number().optional(),
        height: zod_1.z.number().optional(),
        max_degree: zod_1.z.number().optional().describe("Maximum children per node"),
    }).optional().describe("Tree metadata"),
});
/**
 * Tree Operation Schema
 */
exports.TreeOperation = zod_1.z.object({
    operation: zod_1.z.enum([
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
    target: zod_1.z.object({
        node_id: zod_1.z.string().optional().describe("Target node ID"),
        value: zod_1.z.any().optional().describe("Target value"),
        path: zod_1.z.array(zod_1.z.string()).optional().describe("Path from root"),
    }).optional().describe("Operation target"),
    parameters: zod_1.z.object({
        position: zod_1.z.enum(['left', 'right', 'root']).optional().describe("Insert position"),
        traversal_order: exports.TraversalOrder.optional().describe("Traversal order"),
        range: zod_1.z.tuple([zod_1.z.any(), zod_1.z.any()]).optional().describe("Range query"),
        comparison_key: zod_1.z.string().optional().describe("Key for comparisons"),
    }).optional().describe("Operation parameters"),
    timeout_ms: zod_1.z.number().int().positive().max(30000).optional(),
});
/**
 * Arbor Request Schema
 */
exports.ArborRequest = zod_1.z.object({
    tree_id: zod_1.z.string().optional().describe("Tree identifier (optional for creation)"),
    action: zod_1.z.enum([
        'create_tree',
        'destroy_tree',
        'execute_operation',
        'batch_operations',
        'clone_tree',
        'export_tree',
        'import_tree',
    ]).describe("Action to perform"),
    tree_config: zod_1.z.object({
        tree_type: exports.TreeType.optional(),
        comparison_key: zod_1.z.string().optional().describe("Key for ordered trees"),
        max_children: zod_1.z.number().int().positive().optional().describe("Max children (n-ary)"),
        order: zod_1.z.number().int().positive().optional().describe("Tree order (B-tree)"),
        auto_balance: zod_1.z.boolean().optional().describe("Auto-balance on insert/delete"),
    }).optional().describe("Tree configuration"),
    node: exports.TreeNode.optional().describe("Node to insert"),
    operation: exports.TreeOperation.optional().describe("Operation to execute"),
    operations: zod_1.z.array(exports.TreeOperation).optional().describe("Batch operations"),
    timeout_ms: zod_1.z.number()
        .int().positive().max(60000)
        .describe("Request timeout (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Arbor Response Schema
 */
exports.ArborResponse = zod_1.z.object({
    tree_id: zod_1.z.string().describe("Tree identifier"),
    action: zod_1.z.enum([
        'create_tree',
        'destroy_tree',
        'execute_operation',
        'batch_operations',
        'clone_tree',
        'export_tree',
        'import_tree',
    ]).describe("Action performed"),
    status: zod_1.z.enum([
        'success',
        'failed',
        'timeout',
    ]).describe("Action status"),
    result: zod_1.z.object({
        tree: exports.Tree.optional().describe("Resulting tree"),
        node_id: zod_1.z.string().optional().describe("Created/updated node ID"),
        nodes: zod_1.z.array(exports.TreeNode).optional().describe("Traversal/query results"),
        found: zod_1.z.boolean().optional().describe("Search result"),
        value: zod_1.z.any().optional().describe("Retrieved value"),
        traversal: zod_1.z.array(zod_1.z.any()).optional().describe("Traversal result"),
        cloned_tree_id: zod_1.z.string().optional().describe("Cloned tree ID"),
        export_data: zod_1.z.record(zod_1.z.any()).optional().describe("Exported tree data"),
        operations_completed: zod_1.z.number().optional().describe("Batch operations count"),
        stats: zod_1.z.object({
            node_count: zod_1.z.number().optional(),
            height: zod_1.z.number().optional(),
            balance_factor: zod_1.z.number().optional(),
            operation_time_ms: zod_1.z.number().optional(),
        }).optional().describe("Tree statistics"),
    }).optional().describe("Action result"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional(),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        updated_at: zod_1.z.string().datetime().optional(),
        processing_time_ms: zod_1.z.number().optional(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Error Model
 */
exports.ArborError = zod_1.z.object({
    code: zod_1.z.enum([
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
    message: zod_1.z.string(),
    details: zod_1.z.record(zod_1.z.any()).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
function validateArborRequest(data) {
    const result = exports.ArborRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function isArborRequest(data) {
    return typeof data === 'object' && data !== null &&
        'action' in data;
}
/**
 * Examples
 */
exports.ArborExamples = {
    validCreateTree: {
        action: "create_tree",
        tree_config: {
            tree_type: "bst",
            comparison_key: "value",
            auto_balance: true,
        },
        timeout_ms: 5000,
    },
    validInsert: {
        tree_id: "tree_001",
        action: "execute_operation",
        node: {
            node_id: "node_001",
            value: 42,
        },
        operation: {
            operation: "insert",
        },
        timeout_ms: 5000,
    },
    validTraversal: {
        tree_id: "tree_001",
        action: "execute_operation",
        operation: {
            operation: "traverse",
            parameters: {
                traversal_order: "in_order",
            },
        },
        timeout_ms: 5000,
    },
};
//# sourceMappingURL=arbor-canonical.js.map