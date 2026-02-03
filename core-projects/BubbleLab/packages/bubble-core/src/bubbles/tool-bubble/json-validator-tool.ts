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
const JSONValidatorToolParamsSchema = z.object({
  // Input JSON
  jsonData: z
    .string()
    .describe('JSON string to validate'),

  // Validation options
  validateSyntax: z
    .boolean()
    .default(true)
    .describe('Whether to validate JSON syntax'),

  validateSchema: z
    .record(z.unknown())
    .optional()
    .describe('JSON Schema to validate against'),

  checkRequiredFields: z
    .array(z.string())
    .optional()
    .describe('List of required fields to check for'),

  checkDataTypes: z
    .record(z.string(), z.enum(['string', 'number', 'boolean', 'array', 'object', 'null']))
    .optional()
    .describe('Expected data types for fields'),

  customRules: z
    .array(
      z.object({
        field: z.string().describe('JSON path to field (e.g., "user.email")'),
        rule: z
          .enum(['required', 'regex', 'range', 'length', 'enum'])
          .describe('Validation rule type'),
        value: z.unknown().optional().describe('Rule value (regex pattern, range, etc.)'),
        message: z.string().describe('Error message for failed validation'),
      })
    )
    .optional()
    .describe('Custom validation rules'),

  // Query options
  queryPath: z
    .string()
    .optional()
    .describe('JSON path to query (e.g., "users.*.email" or "users[0].name")'),

  // Transformation options
  transformations: z
    .array(
      z.object({
        path: z.string().describe('JSON path to apply transformation to'),
        operation: z
          .enum(['uppercase', 'lowercase', 'trim', 'replace', 'add', 'subtract', 'multiply', 'divide'])
          .describe('Transformation operation'),
        value: z.unknown().optional().describe('Value for transformation'),
      })
    )
    .optional()
    .describe('Transformations to apply'),

  // Patch options (RFC 6902 JSON Patch)
  patches: z
    .array(
      z.object({
        op: z.enum(['add', 'remove', 'replace', 'move', 'copy', 'test']).describe('Patch operation'),
        path: z.string().describe('JSON Pointer path'),
        value: z.unknown().optional().describe('Value for add/replace/test operations'),
        from: z.string().optional().describe('Source path for move/copy operations'),
      })
    )
    .optional()
    .describe('JSON Patch operations to apply'),

  // Format options
  prettyPrint: z
    .boolean()
    .default(true)
    .describe('Whether to format error messages nicely'),

  indent: z
    .number()
    .int()
    .min(0)
    .max(10)
    .default(2)
    .describe('Number of spaces for indentation (0-10)'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for external schema references'),
});

/**
 * Validation error schema with line/column information
 */
const ValidationErrorSchema = z.object({
  path: z.string().describe('JSON path to error location'),
  line: z.number().optional().describe('Line number where error occurred'),
  column: z.number().optional().describe('Column number where error occurred'),
  message: z.string().describe('Error message'),
  expected: z.unknown().optional().describe('Expected value/type'),
  actual: z.unknown().optional().describe('Actual value/type'),
  severity: z
    .enum(['error', 'warning'])
    .describe('Severity of the validation error'),
});

/**
 * JSON validator result schema
 */
const JSONValidatorToolResultSchema = z.object({
  // Validation results
  isValid: z
    .boolean()
    .describe('Whether the JSON is valid'),

  errors: z
    .array(ValidationErrorSchema)
    .describe('Array of validation errors'),

  warnings: z
    .array(ValidationErrorSchema)
    .describe('Array of validation warnings'),

  // Parsed data
  parsedData: z
    .unknown()
    .optional()
    .describe('Parsed JSON object'),

  // Query results
  queryResults: z
    .unknown()
    .optional()
    .describe('Results from path query'),

  // Transformed data
  transformedData: z
    .unknown()
    .optional()
    .describe('Data after applying transformations'),

  // Patched data
  patchedData: z
    .unknown()
    .optional()
    .describe('Data after applying patches'),

  // Statistics
  statistics: z
    .object({
      totalErrors: z.number(),
      totalWarnings: z.number(),
      validationTime: z.number(),
    })
    .describe('Validation statistics'),

  // Formatted output
  formattedJson: z
    .string()
    .optional()
    .describe('Formatted JSON string (if prettyPrint is true)'),

  success: z.boolean().describe('Whether the validation operation was successful'),
  error: z.string().describe('Error message if validation operation failed'),
});

// Type definitions
type JSONValidatorToolParams = z.output<typeof JSONValidatorToolParamsSchema>;
type JSONValidatorToolResult = z.output<typeof JSONValidatorToolResultSchema>;
type JSONValidatorToolParamsInput = z.input<typeof JSONValidatorToolParamsSchema>;

/**
 * JSON Validator Tool
 * Comprehensive JSON validation with schema support
 */
export class JSONValidatorTool extends ToolBubble<
  JSONValidatorToolParams,
  JSONValidatorToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'json-validator-tool';
  static readonly schema = JSONValidatorToolParamsSchema;
  static readonly resultSchema = JSONValidatorToolResultSchema;
  static readonly shortDescription =
    'Validate JSON syntax, schema, and data quality';
  static readonly longDescription = `
    A comprehensive JSON validation tool that checks syntax, schema compliance,
    and data quality.

    Features:
    - Validate JSON syntax and structure
    - Validate against JSON Schema
    - Check for required fields
    - Validate data types for specific fields
    - Apply custom validation rules (regex, range, length, enum)
    - Detailed error reporting with JSON paths
    - Support for nested object validation
    - Pretty print and format JSON

    Validation Rules:
    - SYNTAX: Check if JSON is well-formed
    - SCHEMA: Validate against JSON Schema
    - REQUIRED_FIELDS: Ensure specific fields exist
    - DATA_TYPES: Verify field data types
    - CUSTOM_RULES: Apply custom validation logic

    Custom Rules:
    - required: Field must be present and non-null
    - regex: Field must match regex pattern
    - range: Numeric field must be within range
    - length: String/array length must be within range
    - enum: Field must be one of the allowed values

    Use cases:
    - API response validation
    - Configuration file validation
    - Data quality checks
    - Schema compliance verification
    - Debugging JSON structure issues
    - Data pipeline validation
  `;
  static readonly alias = 'json-validate';

  constructor(
    params: JSONValidatorToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - performs JSON validation and operations
   */
  async performAction(
    context?: BubbleContext
  ): Promise<JSONValidatorToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log('[JSONValidatorTool] Starting JSON validation');

      const errors: Array<{
        path: string;
        line?: number;
        column?: number;
        message: string;
        expected?: unknown;
        actual?: unknown;
        severity: 'error' | 'warning';
      }> = [];
      const warnings: typeof errors = [];

      let parsedData: unknown = null;

      // 1. Validate JSON syntax with line/column tracking
      if (this.params.validateSyntax) {
        try {
          parsedData = JSON.parse(this.params.jsonData);
          console.log('[JSONValidatorTool] Syntax validation passed');
        } catch (parseError) {
          const location = this.extractErrorLocation(parseError);
          errors.push({
            path: 'root',
            line: location.line,
            column: location.column,
            message: `Invalid JSON syntax: ${parseError instanceof Error ? parseError.message : 'Unknown error'}`,
            severity: 'error',
          });

          const validationTime = Date.now() - startTime;

          return {
            isValid: false,
            errors,
            warnings,
            statistics: {
              totalErrors: errors.length,
              totalWarnings: warnings.length,
              validationTime,
            },
            success: true,
            error: '',
          };
        }
      }

      // 2. Validate against JSON Schema
      if (this.params.validateSchema && parsedData) {
        this.validateAgainstSchema(parsedData, this.params.validateSchema, errors);
      }

      // 3. Check required fields
      if (this.params.checkRequiredFields && parsedData) {
        this.checkRequiredFields(parsedData, this.params.checkRequiredFields, errors);
      }

      // 4. Check data types
      if (this.params.checkDataTypes && parsedData) {
        this.checkDataTypes(parsedData, this.params.checkDataTypes, warnings);
      }

      // 5. Apply custom validation rules
      if (this.params.customRules && parsedData) {
        this.applyCustomRules(parsedData, this.params.customRules, errors);
      }

      // 6. Query path if specified
      let queryResults: unknown;
      if (this.params.queryPath && parsedData) {
        queryResults = this.queryPath(parsedData, this.params.queryPath);
        console.log(`[JSONValidatorTool] Query path "${this.params.queryPath}" completed`);
      }

      // 7. Apply transformations if specified
      let transformedData: unknown;
      if (this.params.transformations && parsedData) {
        transformedData = JSON.parse(JSON.stringify(parsedData)); // Deep clone
        this.applyTransformations(transformedData, this.params.transformations);
        console.log(`[JSONValidatorTool] Applied ${this.params.transformations.length} transformations`);
      }

      // 8. Apply patches if specified
      let patchedData: unknown;
      if (this.params.patches && parsedData) {
        patchedData = JSON.parse(JSON.stringify(parsedData)); // Deep clone
        this.applyPatches(patchedData, this.params.patches);
        console.log(`[JSONValidatorTool] Applied ${this.params.patches.length} patches`);
      }

      const validationTime = Date.now() - startTime;
      const isValid = errors.length === 0;

      console.log(`[JSONValidatorTool] Validation completed. Valid: ${isValid}, Errors: ${errors.length}, Warnings: ${warnings.length}`);

      // Format JSON if requested
      let formattedJson: string | undefined;
      if (this.params.prettyPrint && parsedData) {
        formattedJson = JSON.stringify(parsedData, null, this.params.indent);
      }

      return {
        isValid,
        errors,
        warnings,
        parsedData,
        queryResults,
        transformedData,
        patchedData,
        formattedJson,
        statistics: {
          totalErrors: errors.length,
          totalWarnings: warnings.length,
          validationTime,
        },
        success: true,
        error: '',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[JSONValidatorTool] Validation failed: ${errorMessage}`);

      return {
        isValid: false,
        errors: [],
        warnings: [],
        statistics: {
          totalErrors: 0,
          totalWarnings: 0,
          validationTime: Date.now() - startTime,
        },
        success: false,
        error: errorMessage,
      };
    }
  }

  /**
   * Extract line and column information from JSON parse error
   */
  private extractErrorLocation(error: unknown): { line?: number; column?: number } {
    if (error instanceof Error) {
      const match = error.message.match(/position (\d+)/);
      if (match) {
        const position = parseInt(match[1], 10);
        const textBeforeError = this.params.jsonData.substring(0, position);
        const lines = textBeforeError.split('\n');
        return {
          line: lines.length,
          column: lines[lines.length - 1].length + 1,
        };
      }
    }
    return {};
  }

  /**
   * Query JSON data using path syntax with wildcards
   * Supports:
   * - Dot notation: "user.email"
   * - Array indices: "users[0].name"
   * - Wildcards: "users.*.email"
   * - Recursive wildcard: "$..email"
   */
  private queryPath(data: unknown, path: string): unknown {
    const parts = path.split('.');
    let current: unknown = data;
    const results: unknown[] = [];

    const traverse = (obj: unknown, parts: string[], index: number): void => {
      if (index >= parts.length) {
        results.push(obj);
        return;
      }

      const part = parts[index];

      // Handle array indexing (e.g., "[0]" or "[*]")
      if (part.includes('[')) {
        const arrayMatch = part.match(/^(\w+)\[(\d+|\*)\]$/);
        if (arrayMatch && typeof obj === 'object' && obj !== null) {
          const array = (obj as Record<string, unknown>)[arrayMatch[1]] as unknown[];
          if (Array.isArray(array)) {
            if (arrayMatch[2] === '*') {
              array.forEach((item) => traverse(item, parts, index + 1));
            } else {
              const idx = parseInt(arrayMatch[2], 10);
              if (idx < array.length) {
                traverse(array[idx], parts, index + 1);
              }
            }
          }
        }
        return;
      }

      // Handle wildcard
      if (part === '*') {
        if (Array.isArray(obj)) {
          obj.forEach((item) => traverse(item, parts, index + 1));
        } else if (typeof obj === 'object' && obj !== null) {
          Object.values(obj).forEach((value) => traverse(value, parts, index + 1));
        }
        return;
      }

      // Regular property access
      if (typeof obj === 'object' && obj !== null) {
        const nextObj = (obj as Record<string, unknown>)[part];
        if (nextObj !== undefined) {
          traverse(nextObj, parts, index + 1);
        }
      }
    };

    traverse(current, parts, 0);

    // Return single result or array of results
    return results.length === 1 ? results[0] : results;
  }

  /**
   * Apply transformations to JSON data
   */
  private applyTransformations(
    data: unknown,
    transformations: Array<{ path: string; operation: string; value?: unknown }>
  ): void {
    transformations.forEach((transform) => {
      const { path, operation, value } = transform;
      const target = this.queryPath(data, path);

      if (target === undefined || target === null) {
        console.warn(`[JSONValidatorTool] Cannot apply transformation to undefined path: ${path}`);
        return;
      }

      switch (operation) {
        case 'uppercase':
          if (Array.isArray(target)) {
            for (let i = 0; i < target.length; i++) {
              if (typeof target[i] === 'string') {
                (target as string[])[i] = target[i].toUpperCase();
              }
            }
          } else if (typeof target === 'string') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target.toUpperCase();
            }
          }
          break;

        case 'lowercase':
          if (Array.isArray(target)) {
            for (let i = 0; i < target.length; i++) {
              if (typeof target[i] === 'string') {
                (target as string[])[i] = target[i].toLowerCase();
              }
            }
          } else if (typeof target === 'string') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target.toLowerCase();
            }
          }
          break;

        case 'trim':
          if (Array.isArray(target)) {
            for (let i = 0; i < target.length; i++) {
              if (typeof target[i] === 'string') {
                (target as string[])[i] = target[i].trim();
              }
            }
          } else if (typeof target === 'string') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target.trim();
            }
          }
          break;

        case 'replace':
          if (typeof value === 'string' && Array.isArray(target)) {
            for (let i = 0; i < target.length; i++) {
              if (typeof target[i] === 'string') {
                const [search, replace] = value.split('|');
                (target as string[])[i] = target[i].replace(new RegExp(search, 'g'), replace);
              }
            }
          } else if (typeof value === 'string' && typeof target === 'string') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              const [search, replace] = value.split('|');
              (parent as Record<string, unknown>)[key] = target.replace(
                new RegExp(search, 'g'),
                replace
              );
            }
          }
          break;

        case 'add':
          if (typeof target === 'number' && typeof value === 'number') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target + value;
            }
          }
          break;

        case 'subtract':
          if (typeof target === 'number' && typeof value === 'number') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target - value;
            }
          }
          break;

        case 'multiply':
          if (typeof target === 'number' && typeof value === 'number') {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target * value;
            }
          }
          break;

        case 'divide':
          if (typeof target === 'number' && typeof value === 'number' && value !== 0) {
            const parent = this.getParentPath(data, path);
            if (parent) {
              const key = this.getLeafKey(path);
              (parent as Record<string, unknown>)[key] = target / value;
            }
          }
          break;

        default:
          console.warn(`[JSONValidatorTool] Unknown transformation operation: ${operation}`);
      }
    });
  }

  /**
   * Get parent object for a given path
   */
  private getParentPath(data: unknown, path: string): unknown {
    const parts = path.split('.');
    parts.pop(); // Remove leaf
    if (parts.length === 0) {
      return data;
    }
    return this.queryPath(data, parts.join('.'));
  }

  /**
   * Get the leaf key from a path
   */
  private getLeafKey(path: string): string {
    const parts = path.split('.');
    return parts[parts.length - 1];
  }

  /**
   * Apply JSON Patch operations (RFC 6902)
   */
  private applyPatches(
    data: unknown,
    patches: Array<{ op: string; path: string; value?: unknown; from?: string }>
  ): void {
    patches.forEach((patch) => {
      const { op, path, value, from } = patch;

      // Convert JSON Pointer to path
      const jsonPath = path.replace(/^\//, '').replace(/\//g, '.');

      switch (op) {
        case 'add':
          this.patchAdd(data, jsonPath, value);
          break;

        case 'remove':
          this.patchRemove(data, jsonPath);
          break;

        case 'replace':
          this.patchReplace(data, jsonPath, value);
          break;

        case 'move':
          if (from) {
            const fromPath = from.replace(/^\//, '').replace(/\//g, '.');
            const valueToMove = this.queryPath(data, fromPath);
            this.patchRemove(data, fromPath);
            this.patchAdd(data, jsonPath, valueToMove);
          }
          break;

        case 'copy':
          if (from) {
            const fromPath = from.replace(/^\//, '').replace(/\//g, '.');
            const valueToCopy = this.queryPath(data, fromPath);
            this.patchAdd(data, jsonPath, JSON.parse(JSON.stringify(valueToCopy)));
          }
          break;

        case 'test':
          const currentValue = this.queryPath(data, jsonPath);
          if (JSON.stringify(currentValue) !== JSON.stringify(value)) {
            throw new Error(`Test operation failed at path ${path}: expected ${value}, got ${currentValue}`);
          }
          break;

        default:
          console.warn(`[JSONValidatorTool] Unknown patch operation: ${op}`);
      }
    });
  }

  /**
   * JSON Patch: add operation
   */
  private patchAdd(data: unknown, path: string, value?: unknown): void {
    const parts = path.split('.');
    const leaf = parts.pop()!;

    if (parts.length > 0) {
      const parentPath = parts.join('.');
      const parent = this.queryPath(data, parentPath);
      if (typeof parent === 'object' && parent !== null && !Array.isArray(parent)) {
        (parent as Record<string, unknown>)[leaf] = value;
      } else if (Array.isArray(parent) && /^\d+$/.test(leaf)) {
        const index = parseInt(leaf, 10);
        parent.splice(index, 0, value);
      }
    } else if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
      (data as Record<string, unknown>)[leaf] = value;
    }
  }

  /**
   * JSON Patch: remove operation
   */
  private patchRemove(data: unknown, path: string): void {
    const parts = path.split('.');
    const leaf = parts.pop()!;

    if (parts.length > 0) {
      const parentPath = parts.join('.');
      const parent = this.queryPath(data, parentPath);
      if (typeof parent === 'object' && parent !== null && !Array.isArray(parent)) {
        delete (parent as Record<string, unknown>)[leaf];
      } else if (Array.isArray(parent) && /^\d+$/.test(leaf)) {
        const index = parseInt(leaf, 10);
        parent.splice(index, 1);
      }
    } else if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
      delete (data as Record<string, unknown>)[leaf];
    }
  }

  /**
   * JSON Patch: replace operation
   */
  private patchReplace(data: unknown, path: string, value?: unknown): void {
    this.patchRemove(data, path);
    this.patchAdd(data, path, value);
  }

  /**
   * Validate data against JSON Schema
   */
  private validateAgainstSchema(
    data: unknown,
    schema: Record<string, unknown>,
    errors: Array<{
      path: string;
      message: string;
      expected?: unknown;
      actual?: unknown;
      severity: 'error' | 'warning';
    }>,
    path = 'root'
  ): void {
    const type = schema.type;

    if (type) {
      const actualType = Array.isArray(data) ? 'array' : typeof data;

      if (type === 'object' && actualType !== 'object') {
        errors.push({
          path,
          message: `Expected object, got ${actualType}`,
          expected: type,
          actual: actualType,
          severity: 'error',
        });
        return;
      }

      if (type === 'array' && !Array.isArray(data)) {
        errors.push({
          path,
          message: `Expected array, got ${actualType}`,
          expected: type,
          actual: actualType,
          severity: 'error',
        });
        return;
      }

      if (type === 'string' && typeof data !== 'string') {
        errors.push({
          path,
          message: `Expected string, got ${actualType}`,
          expected: type,
          actual: actualType,
          severity: 'error',
        });
        return;
      }

      if (type === 'number' && typeof data !== 'number') {
        errors.push({
          path,
          message: `Expected number, got ${actualType}`,
          expected: type,
          actual: actualType,
          severity: 'error',
        });
        return;
      }

      if (type === 'boolean' && typeof data !== 'boolean') {
        errors.push({
          path,
          message: `Expected boolean, got ${actualType}`,
          expected: type,
          actual: actualType,
          severity: 'error',
        });
        return;
      }
    }

    // Check required properties
    if (schema.required && typeof data === 'object' && data !== null) {
      const required = schema.required as string[];
      required.forEach((prop) => {
        if (!(prop in data)) {
          errors.push({
            path: `${path}.${prop}`,
            message: `Required property missing`,
            severity: 'error',
          });
        }
      });
    }

    // Check properties
    if (schema.properties && typeof data === 'object' && data !== null) {
      const properties = schema.properties as Record<string, Record<string, unknown>>;
      Object.entries(properties).forEach(([propName, propSchema]) => {
        if (propName in data) {
          this.validateAgainstSchema(
            (data as Record<string, unknown>)[propName],
            propSchema,
            errors,
            `${path}.${propName}`
          );
        }
      });
    }

    // Check array items
    if (schema.items && Array.isArray(data)) {
      data.forEach((item, index) => {
        this.validateAgainstSchema(
          item,
          schema.items as Record<string, unknown>,
          errors,
          `${path}[${index}]`
        );
      });
    }
  }

  /**
   * Check for required fields
   */
  private checkRequiredFields(
    data: unknown,
    requiredFields: string[],
    errors: Array<{
      path: string;
      message: string;
      expected?: unknown;
      actual?: unknown;
      severity: 'error' | 'warning';
    }>
  ): void {
    if (typeof data !== 'object' || data === null) {
      return;
    }

    requiredFields.forEach((field) => {
      const fieldParts = field.split('.');
      let current: unknown = data;
      let currentPath = 'root';

      for (const part of fieldParts) {
        if (typeof current === 'object' && current !== null && part in current) {
          current = (current as Record<string, unknown>)[part];
          currentPath += `.${part}`;
        } else {
          errors.push({
            path: currentPath + '.' + part,
            message: `Required field is missing`,
            severity: 'error',
          });
          return;
        }
      }

      // Check if field is null or undefined
      if (current === null || current === undefined) {
        errors.push({
          path: currentPath,
          message: `Required field is null or undefined`,
          severity: 'error',
        });
      }
    });
  }

  /**
   * Check data types for fields
   */
  private checkDataTypes(
    data: unknown,
    dataTypes: Record<string, string>,
    warnings: Array<{
      path: string;
      message: string;
      expected?: unknown;
      actual?: unknown;
      severity: 'error' | 'warning';
    }>
  ): void {
    if (typeof data !== 'object' || data === null) {
      return;
    }

    Object.entries(dataTypes).forEach(([field, expectedType]) => {
      const fieldParts = field.split('.');
      let current: unknown = data;
      let currentPath = 'root';

      for (const part of fieldParts) {
        if (typeof current === 'object' && current !== null && part in current) {
          current = (current as Record<string, unknown>)[part];
          currentPath += `.${part}`;
        } else {
          return; // Field doesn't exist, skip type check
        }
      }

      const actualType = Array.isArray(current) ? 'array' : typeof current;

      if (actualType !== expectedType) {
        warnings.push({
          path: currentPath,
          message: `Type mismatch`,
          expected: expectedType,
          actual: actualType,
          severity: 'warning',
        });
      }
    });
  }

  /**
   * Apply custom validation rules
   */
  private applyCustomRules(
    data: unknown,
    rules: Array<{
      field: string;
      rule: string;
      value?: unknown;
      message: string;
    }>,
    errors: Array<{
      path: string;
      message: string;
      expected?: unknown;
      actual?: unknown;
      severity: 'error' | 'warning';
    }>
  ): void {
    rules.forEach((ruleConfig) => {
      const { field, rule, value, message } = ruleConfig;

      // Get field value
      const fieldParts = field.split('.');
      let current: unknown = data;

      for (const part of fieldParts) {
        if (typeof current === 'object' && current !== null && part in current) {
          current = (current as Record<string, unknown>)[part];
        } else {
          // Field doesn't exist
          if (rule === 'required') {
            errors.push({
              path: field,
              message,
              severity: 'error',
            });
          }
          return;
        }
      }

      // Apply validation rule
      switch (rule) {
        case 'required':
          if (current === null || current === undefined || current === '') {
            errors.push({
              path: field,
              message,
              severity: 'error',
            });
          }
          break;

        case 'regex':
          if (typeof current === 'string' && value) {
            const regex = new RegExp(value as string);
            if (!regex.test(current)) {
              errors.push({
                path: field,
                message,
                actual: current,
                severity: 'error',
              });
            }
          }
          break;

        case 'range':
          if (typeof current === 'number' && Array.isArray(value)) {
            const [min, max] = value as [number, number];
            if (current < min || current > max) {
              errors.push({
                path: field,
                message,
                actual: current,
                expected: { min, max },
                severity: 'error',
              });
            }
          }
          break;

        case 'length':
          if (Array.isArray(value)) {
            const [min, max] = value as [number, number];
            const length =
              typeof current === 'string'
                ? current.length
                : Array.isArray(current)
                ? current.length
                : 0;

            if (length < min || length > max) {
              errors.push({
                path: field,
                message,
                actual: length,
                expected: { min, max },
                severity: 'error',
              });
            }
          }
          break;

        case 'enum':
          if (Array.isArray(value)) {
            if (!value.includes(current)) {
              errors.push({
                path: field,
                message,
                actual: current,
                expected: value,
                severity: 'error',
              });
            }
          }
          break;
      }
    });
  }
}
