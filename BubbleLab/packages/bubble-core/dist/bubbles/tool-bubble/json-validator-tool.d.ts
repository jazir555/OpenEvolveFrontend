/**
 * JSON VALIDATOR TOOL
 *
 * A tool bubble for comprehensive JSON validation, schema checking,
 * and data quality analysis.
 *
 * Features:
 * - Validate JSON syntax and structure with line/column numbers
 * - Validate against JSON Schema with detailed error paths
 * - Deep path querying with wildcards and JSON Pointer
 * - Advanced transformations (conditional, mathematical, string operations)
 * - JSON Patch operations (RFC 6902)
 * - Pretty printing with custom indentation
 * - Data type inference and validation
 * - Custom validation rules (regex, range, length, enum)
 * - Detailed error reporting with context
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * JSON validator parameters schema
 */
declare const JSONValidatorToolParamsSchema: z.ZodObject<{
    jsonData: z.ZodString;
    validateSyntax: z.ZodDefault<z.ZodBoolean>;
    validateSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    checkRequiredFields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    checkDataTypes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodEnum<["string", "number", "boolean", "array", "object", "null"]>>>;
    customRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        rule: z.ZodEnum<["required", "regex", "range", "length", "enum"]>;
        value: z.ZodOptional<z.ZodUnknown>;
        message: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        field: string;
        rule: "length" | "required" | "enum" | "regex" | "range";
        value?: unknown;
    }, {
        message: string;
        field: string;
        rule: "length" | "required" | "enum" | "regex" | "range";
        value?: unknown;
    }>, "many">>;
    queryPath: z.ZodOptional<z.ZodString>;
    transformations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        path: z.ZodString;
        operation: z.ZodEnum<["uppercase", "lowercase", "trim", "replace", "add", "subtract", "multiply", "divide"]>;
        value: z.ZodOptional<z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
        value?: unknown;
    }, {
        path: string;
        operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
        value?: unknown;
    }>, "many">>;
    patches: z.ZodOptional<z.ZodArray<z.ZodObject<{
        op: z.ZodEnum<["add", "remove", "replace", "move", "copy", "test"]>;
        path: z.ZodString;
        value: z.ZodOptional<z.ZodUnknown>;
        from: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        op: "replace" | "remove" | "copy" | "move" | "add" | "test";
        value?: unknown;
        from?: string | undefined;
    }, {
        path: string;
        op: "replace" | "remove" | "copy" | "move" | "add" | "test";
        value?: unknown;
        from?: string | undefined;
    }>, "many">>;
    prettyPrint: z.ZodDefault<z.ZodBoolean>;
    indent: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    jsonData: string;
    validateSyntax: boolean;
    prettyPrint: boolean;
    indent: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    validateSchema?: Record<string, unknown> | undefined;
    checkRequiredFields?: string[] | undefined;
    checkDataTypes?: Record<string, "string" | "number" | "boolean" | "object" | "null" | "array"> | undefined;
    customRules?: {
        message: string;
        field: string;
        rule: "length" | "required" | "enum" | "regex" | "range";
        value?: unknown;
    }[] | undefined;
    queryPath?: string | undefined;
    transformations?: {
        path: string;
        operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
        value?: unknown;
    }[] | undefined;
    patches?: {
        path: string;
        op: "replace" | "remove" | "copy" | "move" | "add" | "test";
        value?: unknown;
        from?: string | undefined;
    }[] | undefined;
}, {
    jsonData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    validateSchema?: Record<string, unknown> | undefined;
    validateSyntax?: boolean | undefined;
    checkRequiredFields?: string[] | undefined;
    checkDataTypes?: Record<string, "string" | "number" | "boolean" | "object" | "null" | "array"> | undefined;
    customRules?: {
        message: string;
        field: string;
        rule: "length" | "required" | "enum" | "regex" | "range";
        value?: unknown;
    }[] | undefined;
    queryPath?: string | undefined;
    transformations?: {
        path: string;
        operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
        value?: unknown;
    }[] | undefined;
    patches?: {
        path: string;
        op: "replace" | "remove" | "copy" | "move" | "add" | "test";
        value?: unknown;
        from?: string | undefined;
    }[] | undefined;
    prettyPrint?: boolean | undefined;
    indent?: number | undefined;
}>;
/**
 * JSON validator result schema
 */
declare const JSONValidatorToolResultSchema: z.ZodObject<{
    isValid: z.ZodBoolean;
    errors: z.ZodArray<z.ZodObject<{
        path: z.ZodString;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        expected: z.ZodOptional<z.ZodUnknown>;
        actual: z.ZodOptional<z.ZodUnknown>;
        severity: z.ZodEnum<["error", "warning"]>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }, {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }>, "many">;
    warnings: z.ZodArray<z.ZodObject<{
        path: z.ZodString;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        expected: z.ZodOptional<z.ZodUnknown>;
        actual: z.ZodOptional<z.ZodUnknown>;
        severity: z.ZodEnum<["error", "warning"]>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }, {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }>, "many">;
    parsedData: z.ZodOptional<z.ZodUnknown>;
    queryResults: z.ZodOptional<z.ZodUnknown>;
    transformedData: z.ZodOptional<z.ZodUnknown>;
    patchedData: z.ZodOptional<z.ZodUnknown>;
    statistics: z.ZodObject<{
        totalErrors: z.ZodNumber;
        totalWarnings: z.ZodNumber;
        validationTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        totalErrors: number;
        totalWarnings: number;
        validationTime: number;
    }, {
        totalErrors: number;
        totalWarnings: number;
        validationTime: number;
    }>;
    formattedJson: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    errors: {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }[];
    isValid: boolean;
    statistics: {
        totalErrors: number;
        totalWarnings: number;
        validationTime: number;
    };
    warnings: {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }[];
    queryResults?: unknown;
    parsedData?: unknown;
    transformedData?: unknown;
    patchedData?: unknown;
    formattedJson?: string | undefined;
}, {
    error: string;
    success: boolean;
    errors: {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }[];
    isValid: boolean;
    statistics: {
        totalErrors: number;
        totalWarnings: number;
        validationTime: number;
    };
    warnings: {
        path: string;
        message: string;
        severity: "error" | "warning";
        expected?: unknown;
        line?: number | undefined;
        column?: number | undefined;
        actual?: unknown;
    }[];
    queryResults?: unknown;
    parsedData?: unknown;
    transformedData?: unknown;
    patchedData?: unknown;
    formattedJson?: string | undefined;
}>;
type JSONValidatorToolParams = z.output<typeof JSONValidatorToolParamsSchema>;
type JSONValidatorToolResult = z.output<typeof JSONValidatorToolResultSchema>;
type JSONValidatorToolParamsInput = z.input<typeof JSONValidatorToolParamsSchema>;
/**
 * JSON Validator Tool
 * Comprehensive JSON validation with schema support
 */
export declare class JSONValidatorTool extends ToolBubble<JSONValidatorToolParams, JSONValidatorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        jsonData: z.ZodString;
        validateSyntax: z.ZodDefault<z.ZodBoolean>;
        validateSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        checkRequiredFields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        checkDataTypes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodEnum<["string", "number", "boolean", "array", "object", "null"]>>>;
        customRules: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            rule: z.ZodEnum<["required", "regex", "range", "length", "enum"]>;
            value: z.ZodOptional<z.ZodUnknown>;
            message: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            field: string;
            rule: "length" | "required" | "enum" | "regex" | "range";
            value?: unknown;
        }, {
            message: string;
            field: string;
            rule: "length" | "required" | "enum" | "regex" | "range";
            value?: unknown;
        }>, "many">>;
        queryPath: z.ZodOptional<z.ZodString>;
        transformations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            path: z.ZodString;
            operation: z.ZodEnum<["uppercase", "lowercase", "trim", "replace", "add", "subtract", "multiply", "divide"]>;
            value: z.ZodOptional<z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
            value?: unknown;
        }, {
            path: string;
            operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
            value?: unknown;
        }>, "many">>;
        patches: z.ZodOptional<z.ZodArray<z.ZodObject<{
            op: z.ZodEnum<["add", "remove", "replace", "move", "copy", "test"]>;
            path: z.ZodString;
            value: z.ZodOptional<z.ZodUnknown>;
            from: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            op: "replace" | "remove" | "copy" | "move" | "add" | "test";
            value?: unknown;
            from?: string | undefined;
        }, {
            path: string;
            op: "replace" | "remove" | "copy" | "move" | "add" | "test";
            value?: unknown;
            from?: string | undefined;
        }>, "many">>;
        prettyPrint: z.ZodDefault<z.ZodBoolean>;
        indent: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        jsonData: string;
        validateSyntax: boolean;
        prettyPrint: boolean;
        indent: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        validateSchema?: Record<string, unknown> | undefined;
        checkRequiredFields?: string[] | undefined;
        checkDataTypes?: Record<string, "string" | "number" | "boolean" | "object" | "null" | "array"> | undefined;
        customRules?: {
            message: string;
            field: string;
            rule: "length" | "required" | "enum" | "regex" | "range";
            value?: unknown;
        }[] | undefined;
        queryPath?: string | undefined;
        transformations?: {
            path: string;
            operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
            value?: unknown;
        }[] | undefined;
        patches?: {
            path: string;
            op: "replace" | "remove" | "copy" | "move" | "add" | "test";
            value?: unknown;
            from?: string | undefined;
        }[] | undefined;
    }, {
        jsonData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        validateSchema?: Record<string, unknown> | undefined;
        validateSyntax?: boolean | undefined;
        checkRequiredFields?: string[] | undefined;
        checkDataTypes?: Record<string, "string" | "number" | "boolean" | "object" | "null" | "array"> | undefined;
        customRules?: {
            message: string;
            field: string;
            rule: "length" | "required" | "enum" | "regex" | "range";
            value?: unknown;
        }[] | undefined;
        queryPath?: string | undefined;
        transformations?: {
            path: string;
            operation: "replace" | "trim" | "add" | "uppercase" | "lowercase" | "subtract" | "multiply" | "divide";
            value?: unknown;
        }[] | undefined;
        patches?: {
            path: string;
            op: "replace" | "remove" | "copy" | "move" | "add" | "test";
            value?: unknown;
            from?: string | undefined;
        }[] | undefined;
        prettyPrint?: boolean | undefined;
        indent?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        isValid: z.ZodBoolean;
        errors: z.ZodArray<z.ZodObject<{
            path: z.ZodString;
            line: z.ZodOptional<z.ZodNumber>;
            column: z.ZodOptional<z.ZodNumber>;
            message: z.ZodString;
            expected: z.ZodOptional<z.ZodUnknown>;
            actual: z.ZodOptional<z.ZodUnknown>;
            severity: z.ZodEnum<["error", "warning"]>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }, {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }>, "many">;
        warnings: z.ZodArray<z.ZodObject<{
            path: z.ZodString;
            line: z.ZodOptional<z.ZodNumber>;
            column: z.ZodOptional<z.ZodNumber>;
            message: z.ZodString;
            expected: z.ZodOptional<z.ZodUnknown>;
            actual: z.ZodOptional<z.ZodUnknown>;
            severity: z.ZodEnum<["error", "warning"]>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }, {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }>, "many">;
        parsedData: z.ZodOptional<z.ZodUnknown>;
        queryResults: z.ZodOptional<z.ZodUnknown>;
        transformedData: z.ZodOptional<z.ZodUnknown>;
        patchedData: z.ZodOptional<z.ZodUnknown>;
        statistics: z.ZodObject<{
            totalErrors: z.ZodNumber;
            totalWarnings: z.ZodNumber;
            validationTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalErrors: number;
            totalWarnings: number;
            validationTime: number;
        }, {
            totalErrors: number;
            totalWarnings: number;
            validationTime: number;
        }>;
        formattedJson: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        errors: {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }[];
        isValid: boolean;
        statistics: {
            totalErrors: number;
            totalWarnings: number;
            validationTime: number;
        };
        warnings: {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }[];
        queryResults?: unknown;
        parsedData?: unknown;
        transformedData?: unknown;
        patchedData?: unknown;
        formattedJson?: string | undefined;
    }, {
        error: string;
        success: boolean;
        errors: {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }[];
        isValid: boolean;
        statistics: {
            totalErrors: number;
            totalWarnings: number;
            validationTime: number;
        };
        warnings: {
            path: string;
            message: string;
            severity: "error" | "warning";
            expected?: unknown;
            line?: number | undefined;
            column?: number | undefined;
            actual?: unknown;
        }[];
        queryResults?: unknown;
        parsedData?: unknown;
        transformedData?: unknown;
        patchedData?: unknown;
        formattedJson?: string | undefined;
    }>;
    static readonly shortDescription = "Validate JSON syntax, schema, and data quality";
    static readonly longDescription = "\n    A comprehensive JSON validation tool that checks syntax, schema compliance,\n    and data quality.\n\n    Features:\n    - Validate JSON syntax and structure\n    - Validate against JSON Schema\n    - Check for required fields\n    - Validate data types for specific fields\n    - Apply custom validation rules (regex, range, length, enum)\n    - Detailed error reporting with JSON paths\n    - Support for nested object validation\n    - Pretty print and format JSON\n\n    Validation Rules:\n    - SYNTAX: Check if JSON is well-formed\n    - SCHEMA: Validate against JSON Schema\n    - REQUIRED_FIELDS: Ensure specific fields exist\n    - DATA_TYPES: Verify field data types\n    - CUSTOM_RULES: Apply custom validation logic\n\n    Custom Rules:\n    - required: Field must be present and non-null\n    - regex: Field must match regex pattern\n    - range: Numeric field must be within range\n    - length: String/array length must be within range\n    - enum: Field must be one of the allowed values\n\n    Use cases:\n    - API response validation\n    - Configuration file validation\n    - Data quality checks\n    - Schema compliance verification\n    - Debugging JSON structure issues\n    - Data pipeline validation\n  ";
    static readonly alias = "json-validate";
    constructor(params: JSONValidatorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs JSON validation and operations
     */
    performAction(context?: BubbleContext): Promise<JSONValidatorToolResult>;
    /**
     * Extract line and column information from JSON parse error
     */
    private extractErrorLocation;
    /**
     * Query JSON data using path syntax with wildcards
     * Supports:
     * - Dot notation: "user.email"
     * - Array indices: "users[0].name"
     * - Wildcards: "users.*.email"
     * - Recursive wildcard: "$..email"
     */
    private queryPath;
    /**
     * Apply transformations to JSON data
     */
    private applyTransformations;
    /**
     * Get parent object for a given path
     */
    private getParentPath;
    /**
     * Get the leaf key from a path
     */
    private getLeafKey;
    /**
     * Apply JSON Patch operations (RFC 6902)
     */
    private applyPatches;
    /**
     * JSON Patch: add operation
     */
    private patchAdd;
    /**
     * JSON Patch: remove operation
     */
    private patchRemove;
    /**
     * JSON Patch: replace operation
     */
    private patchReplace;
    /**
     * Validate data against JSON Schema
     */
    private validateAgainstSchema;
    /**
     * Check for required fields
     */
    private checkRequiredFields;
    /**
     * Check data types for fields
     */
    private checkDataTypes;
    /**
     * Apply custom validation rules
     */
    private applyCustomRules;
}
export {};
//# sourceMappingURL=json-validator-tool.d.ts.map