/**
 * DATA TRANSFORMER TOOL
 *
 * A tool bubble for transforming and reshaping data with support for
 * mapping, filtering, sorting, aggregation, and complex transformations.
 *
 * Features:
 * - Map transformations with custom functions
 * - Filter data based on conditions
 * - Sort by multiple fields
 * - Group and aggregate data
 * - Join/merge datasets
 * - Pivot and unpivot operations
 * - Custom transformation expressions
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Data transformer parameters schema
 */
declare const DataTransformerToolParamsSchema: z.ZodObject<{
    inputData: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    operation: z.ZodEnum<["map", "filter", "sort", "groupBy", "join", "pivot", "unpivot", "custom"]>;
    mapOperations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        targetField: z.ZodString;
        sourceField: z.ZodOptional<z.ZodString>;
        transform: z.ZodEnum<["copy", "rename", "calculate", "format", "extract", "lookup"]>;
        expression: z.ZodOptional<z.ZodString>;
        format: z.ZodOptional<z.ZodString>;
        lookupTable: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
        targetField: string;
        format?: string | undefined;
        expression?: string | undefined;
        sourceField?: string | undefined;
        lookupTable?: Record<string, unknown> | undefined;
    }, {
        transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
        targetField: string;
        format?: string | undefined;
        expression?: string | undefined;
        sourceField?: string | undefined;
        lookupTable?: Record<string, unknown> | undefined;
    }>, "many">>;
    filterConditions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        operator: z.ZodEnum<["eq", "ne", "gt", "lt", "gte", "lte", "contains", "startsWith", "endsWith", "in", "isNull"]>;
        value: z.ZodOptional<z.ZodUnknown>;
        values: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    }, "strip", z.ZodTypeAny, {
        field: string;
        operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
        value?: unknown;
        values?: unknown[] | undefined;
    }, {
        field: string;
        operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
        value?: unknown;
        values?: unknown[] | undefined;
    }>, "many">>;
    sortFields: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        order: z.ZodDefault<z.ZodEnum<["asc", "desc"]>>;
    }, "strip", z.ZodTypeAny, {
        field: string;
        order: "asc" | "desc";
    }, {
        field: string;
        order?: "asc" | "desc" | undefined;
    }>, "many">>;
    groupByFields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        operation: z.ZodEnum<["sum", "avg", "min", "max", "count", "first", "last", "concat", "collect"]>;
        alias: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
        field: string;
        alias?: string | undefined;
    }, {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
        field: string;
        alias?: string | undefined;
    }>, "many">>;
    joinData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    joinKey: z.ZodOptional<z.ZodString>;
    joinType: z.ZodOptional<z.ZodDefault<z.ZodEnum<["inner", "left", "right", "outer", "cross"]>>>;
    pivotField: z.ZodOptional<z.ZodString>;
    valueField: z.ZodOptional<z.ZodString>;
    aggregateFunction: z.ZodOptional<z.ZodDefault<z.ZodEnum<["sum", "avg", "min", "max", "count", "first", "last"]>>>;
    customScript: z.ZodOptional<z.ZodString>;
    preserveOriginal: z.ZodDefault<z.ZodBoolean>;
    removeNullFields: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "join" | "sort" | "map" | "filter" | "custom" | "groupBy" | "pivot" | "unpivot";
    inputData: Record<string, unknown>[];
    preserveOriginal: boolean;
    removeNullFields: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    aggregations?: {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
        field: string;
        alias?: string | undefined;
    }[] | undefined;
    mapOperations?: {
        transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
        targetField: string;
        format?: string | undefined;
        expression?: string | undefined;
        sourceField?: string | undefined;
        lookupTable?: Record<string, unknown> | undefined;
    }[] | undefined;
    filterConditions?: {
        field: string;
        operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
        value?: unknown;
        values?: unknown[] | undefined;
    }[] | undefined;
    sortFields?: {
        field: string;
        order: "asc" | "desc";
    }[] | undefined;
    groupByFields?: string[] | undefined;
    joinData?: Record<string, unknown>[] | undefined;
    joinKey?: string | undefined;
    joinType?: "inner" | "left" | "right" | "outer" | "cross" | undefined;
    pivotField?: string | undefined;
    valueField?: string | undefined;
    aggregateFunction?: "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | undefined;
    customScript?: string | undefined;
}, {
    operation: "join" | "sort" | "map" | "filter" | "custom" | "groupBy" | "pivot" | "unpivot";
    inputData: Record<string, unknown>[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    aggregations?: {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
        field: string;
        alias?: string | undefined;
    }[] | undefined;
    mapOperations?: {
        transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
        targetField: string;
        format?: string | undefined;
        expression?: string | undefined;
        sourceField?: string | undefined;
        lookupTable?: Record<string, unknown> | undefined;
    }[] | undefined;
    filterConditions?: {
        field: string;
        operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
        value?: unknown;
        values?: unknown[] | undefined;
    }[] | undefined;
    sortFields?: {
        field: string;
        order?: "asc" | "desc" | undefined;
    }[] | undefined;
    groupByFields?: string[] | undefined;
    joinData?: Record<string, unknown>[] | undefined;
    joinKey?: string | undefined;
    joinType?: "inner" | "left" | "right" | "outer" | "cross" | undefined;
    pivotField?: string | undefined;
    valueField?: string | undefined;
    aggregateFunction?: "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | undefined;
    customScript?: string | undefined;
    preserveOriginal?: boolean | undefined;
    removeNullFields?: boolean | undefined;
}>;
/**
 * Data transformer result schema
 */
declare const DataTransformerToolResultSchema: z.ZodObject<{
    outputData: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    inputRecordCount: z.ZodNumber;
    outputRecordCount: z.ZodNumber;
    fieldsAdded: z.ZodArray<z.ZodString, "many">;
    fieldsRemoved: z.ZodArray<z.ZodString, "many">;
    fieldsModified: z.ZodArray<z.ZodString, "many">;
    transformationStats: z.ZodObject<{
        recordsFiltered: z.ZodOptional<z.ZodNumber>;
        recordsGrouped: z.ZodOptional<z.ZodNumber>;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        recordsFiltered?: number | undefined;
        recordsGrouped?: number | undefined;
    }, {
        processingTime: number;
        recordsFiltered?: number | undefined;
        recordsGrouped?: number | undefined;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    outputData: Record<string, unknown>[];
    inputRecordCount: number;
    outputRecordCount: number;
    fieldsAdded: string[];
    fieldsRemoved: string[];
    fieldsModified: string[];
    transformationStats: {
        processingTime: number;
        recordsFiltered?: number | undefined;
        recordsGrouped?: number | undefined;
    };
}, {
    error: string;
    success: boolean;
    outputData: Record<string, unknown>[];
    inputRecordCount: number;
    outputRecordCount: number;
    fieldsAdded: string[];
    fieldsRemoved: string[];
    fieldsModified: string[];
    transformationStats: {
        processingTime: number;
        recordsFiltered?: number | undefined;
        recordsGrouped?: number | undefined;
    };
}>;
type DataTransformerToolParams = z.output<typeof DataTransformerToolParamsSchema>;
type DataTransformerToolResult = z.output<typeof DataTransformerToolResultSchema>;
type DataTransformerToolParamsInput = z.input<typeof DataTransformerToolParamsSchema>;
/**
 * Data Transformer Tool
 * Transform and reshape data with comprehensive operations
 */
export declare class DataTransformerTool extends ToolBubble<DataTransformerToolParams, DataTransformerToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        inputData: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        operation: z.ZodEnum<["map", "filter", "sort", "groupBy", "join", "pivot", "unpivot", "custom"]>;
        mapOperations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            targetField: z.ZodString;
            sourceField: z.ZodOptional<z.ZodString>;
            transform: z.ZodEnum<["copy", "rename", "calculate", "format", "extract", "lookup"]>;
            expression: z.ZodOptional<z.ZodString>;
            format: z.ZodOptional<z.ZodString>;
            lookupTable: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
            targetField: string;
            format?: string | undefined;
            expression?: string | undefined;
            sourceField?: string | undefined;
            lookupTable?: Record<string, unknown> | undefined;
        }, {
            transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
            targetField: string;
            format?: string | undefined;
            expression?: string | undefined;
            sourceField?: string | undefined;
            lookupTable?: Record<string, unknown> | undefined;
        }>, "many">>;
        filterConditions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            operator: z.ZodEnum<["eq", "ne", "gt", "lt", "gte", "lte", "contains", "startsWith", "endsWith", "in", "isNull"]>;
            value: z.ZodOptional<z.ZodUnknown>;
            values: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        }, "strip", z.ZodTypeAny, {
            field: string;
            operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
            value?: unknown;
            values?: unknown[] | undefined;
        }, {
            field: string;
            operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
            value?: unknown;
            values?: unknown[] | undefined;
        }>, "many">>;
        sortFields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            order: z.ZodDefault<z.ZodEnum<["asc", "desc"]>>;
        }, "strip", z.ZodTypeAny, {
            field: string;
            order: "asc" | "desc";
        }, {
            field: string;
            order?: "asc" | "desc" | undefined;
        }>, "many">>;
        groupByFields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            operation: z.ZodEnum<["sum", "avg", "min", "max", "count", "first", "last", "concat", "collect"]>;
            alias: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
            field: string;
            alias?: string | undefined;
        }, {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
            field: string;
            alias?: string | undefined;
        }>, "many">>;
        joinData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        joinKey: z.ZodOptional<z.ZodString>;
        joinType: z.ZodOptional<z.ZodDefault<z.ZodEnum<["inner", "left", "right", "outer", "cross"]>>>;
        pivotField: z.ZodOptional<z.ZodString>;
        valueField: z.ZodOptional<z.ZodString>;
        aggregateFunction: z.ZodOptional<z.ZodDefault<z.ZodEnum<["sum", "avg", "min", "max", "count", "first", "last"]>>>;
        customScript: z.ZodOptional<z.ZodString>;
        preserveOriginal: z.ZodDefault<z.ZodBoolean>;
        removeNullFields: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "join" | "sort" | "map" | "filter" | "custom" | "groupBy" | "pivot" | "unpivot";
        inputData: Record<string, unknown>[];
        preserveOriginal: boolean;
        removeNullFields: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        aggregations?: {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
            field: string;
            alias?: string | undefined;
        }[] | undefined;
        mapOperations?: {
            transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
            targetField: string;
            format?: string | undefined;
            expression?: string | undefined;
            sourceField?: string | undefined;
            lookupTable?: Record<string, unknown> | undefined;
        }[] | undefined;
        filterConditions?: {
            field: string;
            operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
            value?: unknown;
            values?: unknown[] | undefined;
        }[] | undefined;
        sortFields?: {
            field: string;
            order: "asc" | "desc";
        }[] | undefined;
        groupByFields?: string[] | undefined;
        joinData?: Record<string, unknown>[] | undefined;
        joinKey?: string | undefined;
        joinType?: "inner" | "left" | "right" | "outer" | "cross" | undefined;
        pivotField?: string | undefined;
        valueField?: string | undefined;
        aggregateFunction?: "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | undefined;
        customScript?: string | undefined;
    }, {
        operation: "join" | "sort" | "map" | "filter" | "custom" | "groupBy" | "pivot" | "unpivot";
        inputData: Record<string, unknown>[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        aggregations?: {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | "collect";
            field: string;
            alias?: string | undefined;
        }[] | undefined;
        mapOperations?: {
            transform: "format" | "extract" | "lookup" | "copy" | "rename" | "calculate";
            targetField: string;
            format?: string | undefined;
            expression?: string | undefined;
            sourceField?: string | undefined;
            lookupTable?: Record<string, unknown> | undefined;
        }[] | undefined;
        filterConditions?: {
            field: string;
            operator: "endsWith" | "startsWith" | "in" | "lt" | "ne" | "contains" | "eq" | "gt" | "gte" | "lte" | "isNull";
            value?: unknown;
            values?: unknown[] | undefined;
        }[] | undefined;
        sortFields?: {
            field: string;
            order?: "asc" | "desc" | undefined;
        }[] | undefined;
        groupByFields?: string[] | undefined;
        joinData?: Record<string, unknown>[] | undefined;
        joinKey?: string | undefined;
        joinType?: "inner" | "left" | "right" | "outer" | "cross" | undefined;
        pivotField?: string | undefined;
        valueField?: string | undefined;
        aggregateFunction?: "min" | "max" | "count" | "sum" | "avg" | "first" | "last" | undefined;
        customScript?: string | undefined;
        preserveOriginal?: boolean | undefined;
        removeNullFields?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        outputData: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        inputRecordCount: z.ZodNumber;
        outputRecordCount: z.ZodNumber;
        fieldsAdded: z.ZodArray<z.ZodString, "many">;
        fieldsRemoved: z.ZodArray<z.ZodString, "many">;
        fieldsModified: z.ZodArray<z.ZodString, "many">;
        transformationStats: z.ZodObject<{
            recordsFiltered: z.ZodOptional<z.ZodNumber>;
            recordsGrouped: z.ZodOptional<z.ZodNumber>;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            recordsFiltered?: number | undefined;
            recordsGrouped?: number | undefined;
        }, {
            processingTime: number;
            recordsFiltered?: number | undefined;
            recordsGrouped?: number | undefined;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        outputData: Record<string, unknown>[];
        inputRecordCount: number;
        outputRecordCount: number;
        fieldsAdded: string[];
        fieldsRemoved: string[];
        fieldsModified: string[];
        transformationStats: {
            processingTime: number;
            recordsFiltered?: number | undefined;
            recordsGrouped?: number | undefined;
        };
    }, {
        error: string;
        success: boolean;
        outputData: Record<string, unknown>[];
        inputRecordCount: number;
        outputRecordCount: number;
        fieldsAdded: string[];
        fieldsRemoved: string[];
        fieldsModified: string[];
        transformationStats: {
            processingTime: number;
            recordsFiltered?: number | undefined;
            recordsGrouped?: number | undefined;
        };
    }>;
    static readonly shortDescription = "Transform, filter, sort, and aggregate data arrays";
    static readonly longDescription = "\n    A powerful data transformation tool for reshaping and manipulating arrays of objects.\n\n    Features:\n    - MAP: Copy, rename, calculate, format, extract, or lookup fields\n    - FILTER: Apply complex filtering conditions with multiple operators\n    - SORT: Sort by multiple fields with ascending/descending order\n    - GROUP BY: Group data and apply aggregations (sum, avg, min, max, count, etc.)\n    - JOIN: Join datasets with inner, left, right, outer, or cross joins\n    - PIVOT: Pivot data to create cross-tabulations\n    - UNPIVOT: Unpivot data from wide to long format\n    - CUSTOM: Apply custom JavaScript transformations\n\n    Map Operations:\n    - copy: Copy field value to new field\n    - rename: Rename field\n    - calculate: Calculate value using expression\n    - format: Format value using format string\n    - extract: Extract value using regex\n    - lookup: Lookup value in lookup table\n\n    Filter Operators:\n    - eq/ne: Equal/not equal\n    - gt/lt/gte/lte: Greater/less than comparisons\n    - contains/startsWith/endsWith: String matching\n    - in: Value in array\n    - isNull: Null check\n\n    Aggregation Operations:\n    - sum/avg/min/max: Statistical aggregations\n    - count: Count records\n    - first/last: Get first/last value\n    - concat: Concatenate strings\n    - collect: Collect values into array\n\n    Use cases:\n    - Data preprocessing for analytics\n    - ETL (Extract, Transform, Load) operations\n    - Report generation\n    - Data cleaning and normalization\n    - Feature engineering for ML\n    - API response transformation\n  ";
    static readonly alias = "transform";
    constructor(params: DataTransformerToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs data transformation
     */
    performAction(context?: BubbleContext): Promise<DataTransformerToolResult>;
    /**
     * Apply map operations
     */
    private applyMapOperations;
    /**
     * Evaluate expression safely using mathjs
     * SECURE: Uses mathjs library to prevent code injection attacks
     */
    private evaluateExpression;
    /**
     * Apply filter conditions
     */
    private applyFilter;
    /**
     * Apply sort
     */
    private applySort;
    /**
     * Apply group by with aggregations
     */
    private applyGroupBy;
    /**
     * Apply join
     */
    private applyJoin;
    /**
     * Apply pivot
     */
    private applyPivot;
    /**
     * Apply unpivot
     */
    private applyUnpivot;
    /**
     * Apply custom transformation
     * SECURITY WARNING: This feature is disabled by default due to code injection risks.
     * To enable, set environment variable ALLOW_CUSTOM_TRANSFORMATIONS=true
     *
     * If enabled, the script is subjected to strict validation before execution.
     */
    private applyCustomTransformation;
}
export {};
//# sourceMappingURL=data-transformer-tool.d.ts.map