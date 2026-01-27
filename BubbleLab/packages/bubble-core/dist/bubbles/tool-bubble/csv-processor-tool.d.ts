/**
 * CSV PROCESSOR TOOL
 *
 * A tool bubble for processing CSV files with comprehensive validation,
 * transformation, and analysis capabilities.
 *
 * Features:
 * - Parse CSV files with flexible delimiters
 * - Validate CSV structure and data types
 * - Transform CSV data (filter, map, reduce operations)
 * - Export data to CSV format
 * - Handle large files with streaming
 * - Support for custom delimiters and quote characters
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * CSV operation types
 */
export declare enum CSVOperationType {
    PARSE = "parse",
    VALIDATE = "validate",
    TRANSFORM = "transform",
    FILTER = "filter",
    EXPORT = "export",
    MERGE = "merge",
    AGGREGATE = "aggregate"
}
/**
 * CSV delimiter options
 */
export declare enum CSVDelimiter {
    COMMA = ",",
    SEMICOLON = ";",
    TAB = "\t",
    PIPE = "|",
    COLON = ":"
}
/**
 * CSV processor parameters schema
 */
declare const CSVProcessorToolParamsSchema: z.ZodObject<{
    operation: z.ZodNativeEnum<typeof CSVOperationType>;
    csvData: z.ZodOptional<z.ZodString>;
    csvFilePath: z.ZodOptional<z.ZodString>;
    delimiter: z.ZodDefault<z.ZodNativeEnum<typeof CSVDelimiter>>;
    quoteChar: z.ZodDefault<z.ZodString>;
    escapeChar: z.ZodDefault<z.ZodString>;
    hasHeader: z.ZodDefault<z.ZodBoolean>;
    skipEmptyLines: z.ZodDefault<z.ZodBoolean>;
    trimWhitespace: z.ZodDefault<z.ZodBoolean>;
    validateSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodBoolean, z.ZodNumber]>>>;
    maxRows: z.ZodOptional<z.ZodNumber>;
    transformRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
        column: z.ZodString;
        operation: z.ZodEnum<["upper", "lower", "trim", "replace", "calculate", "format"]>;
        value: z.ZodOptional<z.ZodString>;
        expression: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
        column: string;
        value?: string | undefined;
        expression?: string | undefined;
    }, {
        operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
        column: string;
        value?: string | undefined;
        expression?: string | undefined;
    }>, "many">>;
    filterRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
        column: z.ZodString;
        operator: z.ZodEnum<["equals", "contains", "startsWith", "endsWith", "gt", "lt", "gte", "lte"]>;
        value: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    }, "strip", z.ZodTypeAny, {
        value: string | number;
        column: string;
        operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
    }, {
        value: string | number;
        column: string;
        operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
    }>, "many">>;
    groupBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        column: z.ZodString;
        operation: z.ZodEnum<["sum", "avg", "min", "max", "count", "concat"]>;
        alias: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
        column: string;
        alias?: string | undefined;
    }, {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
        column: string;
        alias?: string | undefined;
    }>, "many">>;
    exportData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: CSVOperationType;
    delimiter: CSVDelimiter;
    quoteChar: string;
    escapeChar: string;
    hasHeader: boolean;
    skipEmptyLines: boolean;
    trimWhitespace: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxRows?: number | undefined;
    csvData?: string | undefined;
    csvFilePath?: string | undefined;
    validateSchema?: Record<string, string | number | boolean> | undefined;
    transformRules?: {
        operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
        column: string;
        value?: string | undefined;
        expression?: string | undefined;
    }[] | undefined;
    filterRules?: {
        value: string | number;
        column: string;
        operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
    }[] | undefined;
    groupBy?: string[] | undefined;
    aggregations?: {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
        column: string;
        alias?: string | undefined;
    }[] | undefined;
    exportData?: Record<string, unknown>[] | undefined;
}, {
    operation: CSVOperationType;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxRows?: number | undefined;
    csvData?: string | undefined;
    csvFilePath?: string | undefined;
    delimiter?: CSVDelimiter | undefined;
    quoteChar?: string | undefined;
    escapeChar?: string | undefined;
    hasHeader?: boolean | undefined;
    skipEmptyLines?: boolean | undefined;
    trimWhitespace?: boolean | undefined;
    validateSchema?: Record<string, string | number | boolean> | undefined;
    transformRules?: {
        operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
        column: string;
        value?: string | undefined;
        expression?: string | undefined;
    }[] | undefined;
    filterRules?: {
        value: string | number;
        column: string;
        operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
    }[] | undefined;
    groupBy?: string[] | undefined;
    aggregations?: {
        operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
        column: string;
        alias?: string | undefined;
    }[] | undefined;
    exportData?: Record<string, unknown>[] | undefined;
}>;
/**
 * CSV processor result schema
 */
declare const CSVProcessorToolResultSchema: z.ZodObject<{
    data: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    rowCount: z.ZodNumber;
    columnCount: z.ZodNumber;
    headers: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    validationErrors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        row: z.ZodNumber;
        column: z.ZodString;
        error: z.ZodString;
        value: z.ZodUnknown;
    }, "strip", z.ZodTypeAny, {
        error: string;
        column: string;
        row: number;
        value?: unknown;
    }, {
        error: string;
        column: string;
        row: number;
        value?: unknown;
    }>, "many">>;
    csvOutput: z.ZodOptional<z.ZodString>;
    statistics: z.ZodOptional<z.ZodObject<{
        totalRows: z.ZodNumber;
        validRows: z.ZodNumber;
        invalidRows: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        totalRows: number;
        validRows: number;
        invalidRows: number;
    }, {
        processingTime: number;
        totalRows: number;
        validRows: number;
        invalidRows: number;
    }>>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    rowCount: number;
    columnCount: number;
    data?: Record<string, unknown>[] | undefined;
    headers?: string[] | undefined;
    validationErrors?: {
        error: string;
        column: string;
        row: number;
        value?: unknown;
    }[] | undefined;
    csvOutput?: string | undefined;
    statistics?: {
        processingTime: number;
        totalRows: number;
        validRows: number;
        invalidRows: number;
    } | undefined;
}, {
    error: string;
    success: boolean;
    rowCount: number;
    columnCount: number;
    data?: Record<string, unknown>[] | undefined;
    headers?: string[] | undefined;
    validationErrors?: {
        error: string;
        column: string;
        row: number;
        value?: unknown;
    }[] | undefined;
    csvOutput?: string | undefined;
    statistics?: {
        processingTime: number;
        totalRows: number;
        validRows: number;
        invalidRows: number;
    } | undefined;
}>;
type CSVProcessorToolParams = z.output<typeof CSVProcessorToolParamsSchema>;
type CSVProcessorToolResult = z.output<typeof CSVProcessorToolResultSchema>;
type CSVProcessorToolParamsInput = z.input<typeof CSVProcessorToolParamsSchema>;
/**
 * CSV Processor Tool
 * Comprehensive CSV file processing with validation and transformation
 */
export declare class CSVProcessorTool extends ToolBubble<CSVProcessorToolParams, CSVProcessorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodNativeEnum<typeof CSVOperationType>;
        csvData: z.ZodOptional<z.ZodString>;
        csvFilePath: z.ZodOptional<z.ZodString>;
        delimiter: z.ZodDefault<z.ZodNativeEnum<typeof CSVDelimiter>>;
        quoteChar: z.ZodDefault<z.ZodString>;
        escapeChar: z.ZodDefault<z.ZodString>;
        hasHeader: z.ZodDefault<z.ZodBoolean>;
        skipEmptyLines: z.ZodDefault<z.ZodBoolean>;
        trimWhitespace: z.ZodDefault<z.ZodBoolean>;
        validateSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodBoolean, z.ZodNumber]>>>;
        maxRows: z.ZodOptional<z.ZodNumber>;
        transformRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            column: z.ZodString;
            operation: z.ZodEnum<["upper", "lower", "trim", "replace", "calculate", "format"]>;
            value: z.ZodOptional<z.ZodString>;
            expression: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
            column: string;
            value?: string | undefined;
            expression?: string | undefined;
        }, {
            operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
            column: string;
            value?: string | undefined;
            expression?: string | undefined;
        }>, "many">>;
        filterRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            column: z.ZodString;
            operator: z.ZodEnum<["equals", "contains", "startsWith", "endsWith", "gt", "lt", "gte", "lte"]>;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        }, "strip", z.ZodTypeAny, {
            value: string | number;
            column: string;
            operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
        }, {
            value: string | number;
            column: string;
            operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
        }>, "many">>;
        groupBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            column: z.ZodString;
            operation: z.ZodEnum<["sum", "avg", "min", "max", "count", "concat"]>;
            alias: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
            column: string;
            alias?: string | undefined;
        }, {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
            column: string;
            alias?: string | undefined;
        }>, "many">>;
        exportData: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: CSVOperationType;
        delimiter: CSVDelimiter;
        quoteChar: string;
        escapeChar: string;
        hasHeader: boolean;
        skipEmptyLines: boolean;
        trimWhitespace: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxRows?: number | undefined;
        csvData?: string | undefined;
        csvFilePath?: string | undefined;
        validateSchema?: Record<string, string | number | boolean> | undefined;
        transformRules?: {
            operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
            column: string;
            value?: string | undefined;
            expression?: string | undefined;
        }[] | undefined;
        filterRules?: {
            value: string | number;
            column: string;
            operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
        }[] | undefined;
        groupBy?: string[] | undefined;
        aggregations?: {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
            column: string;
            alias?: string | undefined;
        }[] | undefined;
        exportData?: Record<string, unknown>[] | undefined;
    }, {
        operation: CSVOperationType;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxRows?: number | undefined;
        csvData?: string | undefined;
        csvFilePath?: string | undefined;
        delimiter?: CSVDelimiter | undefined;
        quoteChar?: string | undefined;
        escapeChar?: string | undefined;
        hasHeader?: boolean | undefined;
        skipEmptyLines?: boolean | undefined;
        trimWhitespace?: boolean | undefined;
        validateSchema?: Record<string, string | number | boolean> | undefined;
        transformRules?: {
            operation: "format" | "replace" | "trim" | "upper" | "lower" | "calculate";
            column: string;
            value?: string | undefined;
            expression?: string | undefined;
        }[] | undefined;
        filterRules?: {
            value: string | number;
            column: string;
            operator: "endsWith" | "startsWith" | "lt" | "contains" | "equals" | "gt" | "gte" | "lte";
        }[] | undefined;
        groupBy?: string[] | undefined;
        aggregations?: {
            operation: "concat" | "min" | "max" | "count" | "sum" | "avg";
            column: string;
            alias?: string | undefined;
        }[] | undefined;
        exportData?: Record<string, unknown>[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        data: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        rowCount: z.ZodNumber;
        columnCount: z.ZodNumber;
        headers: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        validationErrors: z.ZodOptional<z.ZodArray<z.ZodObject<{
            row: z.ZodNumber;
            column: z.ZodString;
            error: z.ZodString;
            value: z.ZodUnknown;
        }, "strip", z.ZodTypeAny, {
            error: string;
            column: string;
            row: number;
            value?: unknown;
        }, {
            error: string;
            column: string;
            row: number;
            value?: unknown;
        }>, "many">>;
        csvOutput: z.ZodOptional<z.ZodString>;
        statistics: z.ZodOptional<z.ZodObject<{
            totalRows: z.ZodNumber;
            validRows: z.ZodNumber;
            invalidRows: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            totalRows: number;
            validRows: number;
            invalidRows: number;
        }, {
            processingTime: number;
            totalRows: number;
            validRows: number;
            invalidRows: number;
        }>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        rowCount: number;
        columnCount: number;
        data?: Record<string, unknown>[] | undefined;
        headers?: string[] | undefined;
        validationErrors?: {
            error: string;
            column: string;
            row: number;
            value?: unknown;
        }[] | undefined;
        csvOutput?: string | undefined;
        statistics?: {
            processingTime: number;
            totalRows: number;
            validRows: number;
            invalidRows: number;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        rowCount: number;
        columnCount: number;
        data?: Record<string, unknown>[] | undefined;
        headers?: string[] | undefined;
        validationErrors?: {
            error: string;
            column: string;
            row: number;
            value?: unknown;
        }[] | undefined;
        csvOutput?: string | undefined;
        statistics?: {
            processingTime: number;
            totalRows: number;
            validRows: number;
            invalidRows: number;
        } | undefined;
    }>;
    static readonly shortDescription = "Process, validate, transform, and analyze CSV files";
    static readonly longDescription = "\n    A comprehensive tool for processing CSV files with support for parsing,\n    validation, transformation, filtering, aggregation, and export operations.\n\n    Features:\n    - Parse CSV files with flexible delimiters (comma, semicolon, tab, pipe)\n    - Validate CSV structure and data types against schemas\n    - Transform data with operations (uppercase, lowercase, trim, replace, calculate)\n    - Filter rows based on conditions\n    - Aggregate data with group by operations (sum, avg, min, max, count)\n    - Export data arrays to CSV format\n    - Handle large files efficiently\n    - Detailed error reporting for validation issues\n\n    Operations:\n    - PARSE: Convert CSV string to array of objects\n    - VALIDATE: Validate CSV data against schema\n    - TRANSFORM: Apply transformations to columns\n    - FILTER: Filter rows based on conditions\n    - EXPORT: Convert array of objects to CSV string\n    - MERGE: Merge multiple CSV files\n    - AGGREGATE: Group and aggregate data\n\n    Use cases:\n    - Data preprocessing for analysis\n    - ETL (Extract, Transform, Load) pipelines\n    - Data validation and cleaning\n    - Report generation from data\n    - Data format conversion\n    - Batch data processing\n  ";
    static readonly alias = "csv";
    constructor(params: CSVProcessorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs CSV operation
     */
    performAction(context?: BubbleContext): Promise<CSVProcessorToolResult>;
    /**
     * Parse CSV line with delimiter, handling quotes and escapes properly
     * Implements RFC 4180 CSV parsing with extended features
     */
    private parseLine;
    /**
     * Infer data type from a string value
     * Attempts to parse as boolean, number, or date before falling back to string
     */
    private inferDataType;
    /**
     * Parse CSV string to array of objects with enhanced error handling
     */
    private parseCSV;
    /**
     * Validate CSV data against schema
     */
    private validateCSV;
    /**
     * Transform CSV data
     */
    private transformCSV;
    /**
     * Filter CSV data
     */
    private filterCSV;
    /**
     * Export data to CSV
     */
    private exportCSV;
    /**
     * Aggregate CSV data
     */
    private aggregateCSV;
}
export {};
//# sourceMappingURL=csv-processor-tool.d.ts.map