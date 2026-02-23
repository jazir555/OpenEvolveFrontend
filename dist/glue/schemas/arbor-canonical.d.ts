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
export declare const TreeType: z.ZodEnum<["binary_tree", "n_ary_tree", "bst", "avl", "red_black", "b_tree", "trie", "segment_tree", "fenwick"]>;
export type TreeType = z.infer<typeof TreeType>;
/**
 * Tree Traversal Enum
 */
export declare const TraversalOrder: z.ZodEnum<["pre_order", "in_order", "post_order", "level_order"]>;
export type TraversalOrder = z.infer<typeof TraversalOrder>;
/**
 * Tree Node Schema
 */
export declare const TreeNode: z.ZodObject<{
    node_id: z.ZodString;
    value: z.ZodAny;
    children: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    parent: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        depth: z.ZodOptional<z.ZodNumber>;
        height: z.ZodOptional<z.ZodNumber>;
        size: z.ZodOptional<z.ZodNumber>;
        balance_factor: z.ZodOptional<z.ZodNumber>;
        color: z.ZodOptional<z.ZodEnum<["red", "black"]>>;
    }, "strip", z.ZodTypeAny, {
        color?: "black" | "red" | undefined;
        depth?: number | undefined;
        size?: number | undefined;
        height?: number | undefined;
        balance_factor?: number | undefined;
    }, {
        color?: "black" | "red" | undefined;
        depth?: number | undefined;
        size?: number | undefined;
        height?: number | undefined;
        balance_factor?: number | undefined;
    }>>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    node_id: string;
    metadata?: {
        color?: "black" | "red" | undefined;
        depth?: number | undefined;
        size?: number | undefined;
        height?: number | undefined;
        balance_factor?: number | undefined;
    } | undefined;
    value?: any;
    children?: string[] | undefined;
    properties?: Record<string, any> | undefined;
    parent?: string | undefined;
}, {
    node_id: string;
    metadata?: {
        color?: "black" | "red" | undefined;
        depth?: number | undefined;
        size?: number | undefined;
        height?: number | undefined;
        balance_factor?: number | undefined;
    } | undefined;
    value?: any;
    children?: string[] | undefined;
    properties?: Record<string, any> | undefined;
    parent?: string | undefined;
}>;
export type TreeNode = z.infer<typeof TreeNode>;
/**
 * Tree Schema
 */
export declare const Tree: z.ZodObject<{
    tree_id: z.ZodString;
    tree_type: z.ZodEnum<["binary_tree", "n_ary_tree", "bst", "avl", "red_black", "b_tree", "trie", "segment_tree", "fenwick"]>;
    root_id: z.ZodOptional<z.ZodString>;
    nodes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        node_count: z.ZodOptional<z.ZodNumber>;
        height: z.ZodOptional<z.ZodNumber>;
        max_degree: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        node_count?: number | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        height?: number | undefined;
        max_degree?: number | undefined;
    }, {
        node_count?: number | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        height?: number | undefined;
        max_degree?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    tree_id: string;
    tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
    metadata?: {
        node_count?: number | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        height?: number | undefined;
        max_degree?: number | undefined;
    } | undefined;
    nodes?: Record<string, any> | undefined;
    root_id?: string | undefined;
}, {
    tree_id: string;
    tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
    metadata?: {
        node_count?: number | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        height?: number | undefined;
        max_degree?: number | undefined;
    } | undefined;
    nodes?: Record<string, any> | undefined;
    root_id?: string | undefined;
}>;
export type Tree = z.infer<typeof Tree>;
/**
 * Tree Operation Schema
 */
export declare const TreeOperation: z.ZodObject<{
    operation: z.ZodEnum<["insert", "delete", "search", "update", "traverse", "query", "balance", "merge", "split"]>;
    target: z.ZodOptional<z.ZodObject<{
        node_id: z.ZodOptional<z.ZodString>;
        value: z.ZodOptional<z.ZodAny>;
        path: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        value?: any;
        path?: string[] | undefined;
        node_id?: string | undefined;
    }, {
        value?: any;
        path?: string[] | undefined;
        node_id?: string | undefined;
    }>>;
    parameters: z.ZodOptional<z.ZodObject<{
        position: z.ZodOptional<z.ZodEnum<["left", "right", "root"]>>;
        traversal_order: z.ZodOptional<z.ZodEnum<["pre_order", "in_order", "post_order", "level_order"]>>;
        range: z.ZodOptional<z.ZodTuple<[z.ZodAny, z.ZodAny], null>>;
        comparison_key: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        range?: [any, any] | undefined;
        position?: "left" | "right" | "root" | undefined;
        traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
        comparison_key?: string | undefined;
    }, {
        range?: [any, any] | undefined;
        position?: "left" | "right" | "root" | undefined;
        traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
        comparison_key?: string | undefined;
    }>>;
    timeout_ms: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
    timeout_ms?: number | undefined;
    parameters?: {
        range?: [any, any] | undefined;
        position?: "left" | "right" | "root" | undefined;
        traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
        comparison_key?: string | undefined;
    } | undefined;
    target?: {
        value?: any;
        path?: string[] | undefined;
        node_id?: string | undefined;
    } | undefined;
}, {
    operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
    timeout_ms?: number | undefined;
    parameters?: {
        range?: [any, any] | undefined;
        position?: "left" | "right" | "root" | undefined;
        traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
        comparison_key?: string | undefined;
    } | undefined;
    target?: {
        value?: any;
        path?: string[] | undefined;
        node_id?: string | undefined;
    } | undefined;
}>;
export type TreeOperation = z.infer<typeof TreeOperation>;
/**
 * Arbor Request Schema
 */
export declare const ArborRequest: z.ZodObject<{
    tree_id: z.ZodOptional<z.ZodString>;
    action: z.ZodEnum<["create_tree", "destroy_tree", "execute_operation", "batch_operations", "clone_tree", "export_tree", "import_tree"]>;
    tree_config: z.ZodOptional<z.ZodObject<{
        tree_type: z.ZodOptional<z.ZodEnum<["binary_tree", "n_ary_tree", "bst", "avl", "red_black", "b_tree", "trie", "segment_tree", "fenwick"]>>;
        comparison_key: z.ZodOptional<z.ZodString>;
        max_children: z.ZodOptional<z.ZodNumber>;
        order: z.ZodOptional<z.ZodNumber>;
        auto_balance: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        order?: number | undefined;
        tree_type?: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick" | undefined;
        comparison_key?: string | undefined;
        max_children?: number | undefined;
        auto_balance?: boolean | undefined;
    }, {
        order?: number | undefined;
        tree_type?: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick" | undefined;
        comparison_key?: string | undefined;
        max_children?: number | undefined;
        auto_balance?: boolean | undefined;
    }>>;
    node: z.ZodOptional<z.ZodObject<{
        node_id: z.ZodString;
        value: z.ZodAny;
        children: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        parent: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodObject<{
            depth: z.ZodOptional<z.ZodNumber>;
            height: z.ZodOptional<z.ZodNumber>;
            size: z.ZodOptional<z.ZodNumber>;
            balance_factor: z.ZodOptional<z.ZodNumber>;
            color: z.ZodOptional<z.ZodEnum<["red", "black"]>>;
        }, "strip", z.ZodTypeAny, {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        }, {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        }>>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        node_id: string;
        metadata?: {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        } | undefined;
        value?: any;
        children?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        parent?: string | undefined;
    }, {
        node_id: string;
        metadata?: {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        } | undefined;
        value?: any;
        children?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        parent?: string | undefined;
    }>>;
    operation: z.ZodOptional<z.ZodObject<{
        operation: z.ZodEnum<["insert", "delete", "search", "update", "traverse", "query", "balance", "merge", "split"]>;
        target: z.ZodOptional<z.ZodObject<{
            node_id: z.ZodOptional<z.ZodString>;
            value: z.ZodOptional<z.ZodAny>;
            path: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        }, {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        }>>;
        parameters: z.ZodOptional<z.ZodObject<{
            position: z.ZodOptional<z.ZodEnum<["left", "right", "root"]>>;
            traversal_order: z.ZodOptional<z.ZodEnum<["pre_order", "in_order", "post_order", "level_order"]>>;
            range: z.ZodOptional<z.ZodTuple<[z.ZodAny, z.ZodAny], null>>;
            comparison_key: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        }, {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        }>>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }, {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }>>;
    operations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        operation: z.ZodEnum<["insert", "delete", "search", "update", "traverse", "query", "balance", "merge", "split"]>;
        target: z.ZodOptional<z.ZodObject<{
            node_id: z.ZodOptional<z.ZodString>;
            value: z.ZodOptional<z.ZodAny>;
            path: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        }, {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        }>>;
        parameters: z.ZodOptional<z.ZodObject<{
            position: z.ZodOptional<z.ZodEnum<["left", "right", "root"]>>;
            traversal_order: z.ZodOptional<z.ZodEnum<["pre_order", "in_order", "post_order", "level_order"]>>;
            range: z.ZodOptional<z.ZodTuple<[z.ZodAny, z.ZodAny], null>>;
            comparison_key: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        }, {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        }>>;
        timeout_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }, {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }>, "many">>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    action: "create_tree" | "destroy_tree" | "execute_operation" | "batch_operations" | "clone_tree" | "export_tree" | "import_tree";
    node?: {
        node_id: string;
        metadata?: {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        } | undefined;
        value?: any;
        children?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        parent?: string | undefined;
    } | undefined;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    operation?: {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    } | undefined;
    operations?: {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }[] | undefined;
    tree_id?: string | undefined;
    tree_config?: {
        order?: number | undefined;
        tree_type?: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick" | undefined;
        comparison_key?: string | undefined;
        max_children?: number | undefined;
        auto_balance?: boolean | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    action: "create_tree" | "destroy_tree" | "execute_operation" | "batch_operations" | "clone_tree" | "export_tree" | "import_tree";
    node?: {
        node_id: string;
        metadata?: {
            color?: "black" | "red" | undefined;
            depth?: number | undefined;
            size?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
        } | undefined;
        value?: any;
        children?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        parent?: string | undefined;
    } | undefined;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    operation?: {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    } | undefined;
    operations?: {
        operation: "search" | "update" | "delete" | "query" | "split" | "merge" | "insert" | "traverse" | "balance";
        timeout_ms?: number | undefined;
        parameters?: {
            range?: [any, any] | undefined;
            position?: "left" | "right" | "root" | undefined;
            traversal_order?: "pre_order" | "in_order" | "post_order" | "level_order" | undefined;
            comparison_key?: string | undefined;
        } | undefined;
        target?: {
            value?: any;
            path?: string[] | undefined;
            node_id?: string | undefined;
        } | undefined;
    }[] | undefined;
    tree_id?: string | undefined;
    tree_config?: {
        order?: number | undefined;
        tree_type?: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick" | undefined;
        comparison_key?: string | undefined;
        max_children?: number | undefined;
        auto_balance?: boolean | undefined;
    } | undefined;
}>;
export type ArborRequest = z.infer<typeof ArborRequest>;
/**
 * Arbor Response Schema
 */
export declare const ArborResponse: z.ZodObject<{
    tree_id: z.ZodString;
    action: z.ZodEnum<["create_tree", "destroy_tree", "execute_operation", "batch_operations", "clone_tree", "export_tree", "import_tree"]>;
    status: z.ZodEnum<["success", "failed", "timeout"]>;
    result: z.ZodOptional<z.ZodObject<{
        tree: z.ZodOptional<z.ZodObject<{
            tree_id: z.ZodString;
            tree_type: z.ZodEnum<["binary_tree", "n_ary_tree", "bst", "avl", "red_black", "b_tree", "trie", "segment_tree", "fenwick"]>;
            root_id: z.ZodOptional<z.ZodString>;
            nodes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            metadata: z.ZodOptional<z.ZodObject<{
                created_at: z.ZodOptional<z.ZodString>;
                updated_at: z.ZodOptional<z.ZodString>;
                node_count: z.ZodOptional<z.ZodNumber>;
                height: z.ZodOptional<z.ZodNumber>;
                max_degree: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            }, {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        }, {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        }>>;
        node_id: z.ZodOptional<z.ZodString>;
        nodes: z.ZodOptional<z.ZodArray<z.ZodObject<{
            node_id: z.ZodString;
            value: z.ZodAny;
            children: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            parent: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodObject<{
                depth: z.ZodOptional<z.ZodNumber>;
                height: z.ZodOptional<z.ZodNumber>;
                size: z.ZodOptional<z.ZodNumber>;
                balance_factor: z.ZodOptional<z.ZodNumber>;
                color: z.ZodOptional<z.ZodEnum<["red", "black"]>>;
            }, "strip", z.ZodTypeAny, {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            }, {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            }>>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }, {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }>, "many">>;
        found: z.ZodOptional<z.ZodBoolean>;
        value: z.ZodOptional<z.ZodAny>;
        traversal: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        cloned_tree_id: z.ZodOptional<z.ZodString>;
        export_data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        operations_completed: z.ZodOptional<z.ZodNumber>;
        stats: z.ZodOptional<z.ZodObject<{
            node_count: z.ZodOptional<z.ZodNumber>;
            height: z.ZodOptional<z.ZodNumber>;
            balance_factor: z.ZodOptional<z.ZodNumber>;
            operation_time_ms: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        }, {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        value?: any;
        stats?: {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        } | undefined;
        nodes?: {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }[] | undefined;
        operations_completed?: number | undefined;
        node_id?: string | undefined;
        export_data?: Record<string, any> | undefined;
        tree?: {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        } | undefined;
        found?: boolean | undefined;
        traversal?: any[] | undefined;
        cloned_tree_id?: string | undefined;
    }, {
        value?: any;
        stats?: {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        } | undefined;
        nodes?: {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }[] | undefined;
        operations_completed?: number | undefined;
        node_id?: string | undefined;
        export_data?: Record<string, any> | undefined;
        tree?: {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        } | undefined;
        found?: boolean | undefined;
        traversal?: any[] | undefined;
        cloned_tree_id?: string | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "success" | "failed" | "timeout";
    action: "create_tree" | "destroy_tree" | "execute_operation" | "batch_operations" | "clone_tree" | "export_tree" | "import_tree";
    tree_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        value?: any;
        stats?: {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        } | undefined;
        nodes?: {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }[] | undefined;
        operations_completed?: number | undefined;
        node_id?: string | undefined;
        export_data?: Record<string, any> | undefined;
        tree?: {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        } | undefined;
        found?: boolean | undefined;
        traversal?: any[] | undefined;
        cloned_tree_id?: string | undefined;
    } | undefined;
}, {
    timestamp: string;
    status: "success" | "failed" | "timeout";
    action: "create_tree" | "destroy_tree" | "execute_operation" | "batch_operations" | "clone_tree" | "export_tree" | "import_tree";
    tree_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        value?: any;
        stats?: {
            node_count?: number | undefined;
            height?: number | undefined;
            balance_factor?: number | undefined;
            operation_time_ms?: number | undefined;
        } | undefined;
        nodes?: {
            node_id: string;
            metadata?: {
                color?: "black" | "red" | undefined;
                depth?: number | undefined;
                size?: number | undefined;
                height?: number | undefined;
                balance_factor?: number | undefined;
            } | undefined;
            value?: any;
            children?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            parent?: string | undefined;
        }[] | undefined;
        operations_completed?: number | undefined;
        node_id?: string | undefined;
        export_data?: Record<string, any> | undefined;
        tree?: {
            tree_id: string;
            tree_type: "binary_tree" | "n_ary_tree" | "bst" | "avl" | "red_black" | "b_tree" | "trie" | "segment_tree" | "fenwick";
            metadata?: {
                node_count?: number | undefined;
                created_at?: string | undefined;
                updated_at?: string | undefined;
                height?: number | undefined;
                max_degree?: number | undefined;
            } | undefined;
            nodes?: Record<string, any> | undefined;
            root_id?: string | undefined;
        } | undefined;
        found?: boolean | undefined;
        traversal?: any[] | undefined;
        cloned_tree_id?: string | undefined;
    } | undefined;
}>;
export type ArborResponse = z.infer<typeof ArborResponse>;
/**
 * Error Model
 */
export declare const ArborError: z.ZodObject<{
    code: z.ZodEnum<["TREE_NOT_FOUND", "NODE_NOT_FOUND", "INVALID_TREE_TYPE", "INVALID_OPERATION", "DUPLICATE_NODE", "TREE_VIOLATION", "BALANCE_ERROR", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "NODE_NOT_FOUND" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "DUPLICATE_NODE" | "TREE_NOT_FOUND" | "INVALID_TREE_TYPE" | "INVALID_OPERATION" | "TREE_VIOLATION" | "BALANCE_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "NODE_NOT_FOUND" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "DUPLICATE_NODE" | "TREE_NOT_FOUND" | "INVALID_TREE_TYPE" | "INVALID_OPERATION" | "TREE_VIOLATION" | "BALANCE_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type ArborError = z.infer<typeof ArborError>;
/**
 * Validation Functions
 */
export declare function validateArborRequest(data: unknown): {
    success: boolean;
    data?: ArborRequest;
    errors?: string[];
};
export declare function isArborRequest(data: unknown): data is ArborRequest;
/**
 * Examples
 */
export declare const ArborExamples: {
    validCreateTree: ArborRequest;
    validInsert: ArborRequest;
    validTraversal: ArborRequest;
};
//# sourceMappingURL=arbor-canonical.d.ts.map