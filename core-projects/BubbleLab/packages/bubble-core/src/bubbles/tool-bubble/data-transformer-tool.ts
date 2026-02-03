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
import { evaluate } from 'mathjs';

/**
 * Data transformer parameters schema
 */
const DataTransformerToolParamsSchema = z.object({
  // Input data
  inputData: z
    .array(z.record(z.unknown()))
    .describe('Input data array to transform'),

  // Transformation type
  operation: z
    .enum(['map', 'filter', 'sort', 'groupBy', 'join', 'pivot', 'unpivot', 'custom'])
    .describe('Type of transformation operation'),

  // Map operation
  mapOperations: z
    .array(
      z.object({
        targetField: z.string().describe('Field to create or modify'),
        sourceField: z.string().optional().describe('Source field to read from'),
        transform: z
          .enum(['copy', 'rename', 'calculate', 'format', 'extract', 'lookup'])
          .describe('Transformation type'),
        expression: z.string().optional().describe('Expression for calculation'),
        format: z.string().optional().describe('Format string for formatting'),
        lookupTable: z.record(z.unknown()).optional().describe('Lookup table for lookup transforms'),
      })
    )
    .optional()
    .describe('Map operations to apply'),

  // Filter operation
  filterConditions: z
    .array(
      z.object({
        field: z.string().describe('Field to filter on'),
        operator: z
          .enum(['eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains', 'startsWith', 'endsWith', 'in', 'isNull'])
          .describe('Comparison operator'),
        value: z.unknown().optional().describe('Value to compare against'),
        values: z.array(z.unknown()).optional().describe('Values for "in" operator'),
      })
    )
    .optional()
    .describe('Filter conditions (AND logic)'),

  // Sort operation
  sortFields: z
    .array(
      z.object({
        field: z.string().describe('Field to sort by'),
        order: z.enum(['asc', 'desc']).default('asc').describe('Sort order'),
      })
    )
    .optional()
    .describe('Fields to sort by (priority order)'),

  // Group by operation
  groupByFields: z
    .array(z.string())
    .optional()
    .describe('Fields to group by'),

  aggregations: z
    .array(
      z.object({
        field: z.string().describe('Field to aggregate'),
        operation: z
          .enum(['sum', 'avg', 'min', 'max', 'count', 'first', 'last', 'concat', 'collect'])
          .describe('Aggregation operation'),
        alias: z.string().optional().describe('Alias for aggregated field'),
      })
    )
    .optional()
    .describe('Aggregation operations'),

  // Join operation
  joinData: z
    .array(z.record(z.unknown()))
    .optional()
    .describe('Data to join with'),

  joinKey: z
    .string()
    .optional()
    .describe('Key field to join on'),

  joinType: z
    .enum(['inner', 'left', 'right', 'outer', 'cross'])
    .default('inner')
    .optional()
    .describe('Type of join'),

  // Pivot operation
  pivotField: z
    .string()
    .optional()
    .describe('Field to pivot on'),

  valueField: z
    .string()
    .optional()
    .describe('Field containing values to pivot'),

  aggregateFunction: z
    .enum(['sum', 'avg', 'min', 'max', 'count', 'first', 'last'])
    .default('sum')
    .optional()
    .describe('Aggregation for pivot values'),

  // Custom operation
  customScript: z
    .string()
    .optional()
    .describe('Custom JavaScript code for transformation'),

  // Output options
  preserveOriginal: z
    .boolean()
    .default(false)
    .describe('Whether to preserve original fields'),

  removeNullFields: z
    .boolean()
    .default(false)
    .describe('Whether to remove fields with null values'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for external data sources'),
});

/**
 * Data transformer result schema
 */
const DataTransformerToolResultSchema = z.object({
  // Transformed data
  outputData: z
    .array(z.record(z.unknown()))
    .describe('Transformed output data'),

  // Metadata
  inputRecordCount: z
    .number()
    .describe('Number of records in input'),

  outputRecordCount: z
    .number()
    .describe('Number of records in output'),

  fieldsAdded: z
    .array(z.string())
    .describe('Fields added by transformation'),

  fieldsRemoved: z
    .array(z.string())
    .describe('Fields removed by transformation'),

  fieldsModified: z
    .array(z.string())
    .describe('Fields modified by transformation'),

  // Statistics
  transformationStats: z
    .object({
      recordsFiltered: z.number().optional(),
      recordsGrouped: z.number().optional(),
      processingTime: z.number(),
    })
    .describe('Transformation statistics'),

  success: z.boolean().describe('Whether the transformation was successful'),
  error: z.string().describe('Error message if transformation failed'),
});

// Type definitions
type DataTransformerToolParams = z.output<typeof DataTransformerToolParamsSchema>;
type DataTransformerToolResult = z.output<typeof DataTransformerToolResultSchema>;
type DataTransformerToolParamsInput = z.input<typeof DataTransformerToolParamsSchema>;

/**
 * Data Transformer Tool
 * Transform and reshape data with comprehensive operations
 */
export class DataTransformerTool extends ToolBubble<
  DataTransformerToolParams,
  DataTransformerToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'data-transformer-tool';
  static readonly schema = DataTransformerToolParamsSchema;
  static readonly resultSchema = DataTransformerToolResultSchema;
  static readonly shortDescription =
    'Transform, filter, sort, and aggregate data arrays';
  static readonly longDescription = `
    A powerful data transformation tool for reshaping and manipulating arrays of objects.

    Features:
    - MAP: Copy, rename, calculate, format, extract, or lookup fields
    - FILTER: Apply complex filtering conditions with multiple operators
    - SORT: Sort by multiple fields with ascending/descending order
    - GROUP BY: Group data and apply aggregations (sum, avg, min, max, count, etc.)
    - JOIN: Join datasets with inner, left, right, outer, or cross joins
    - PIVOT: Pivot data to create cross-tabulations
    - UNPIVOT: Unpivot data from wide to long format
    - CUSTOM: Apply custom JavaScript transformations

    Map Operations:
    - copy: Copy field value to new field
    - rename: Rename field
    - calculate: Calculate value using expression
    - format: Format value using format string
    - extract: Extract value using regex
    - lookup: Lookup value in lookup table

    Filter Operators:
    - eq/ne: Equal/not equal
    - gt/lt/gte/lte: Greater/less than comparisons
    - contains/startsWith/endsWith: String matching
    - in: Value in array
    - isNull: Null check

    Aggregation Operations:
    - sum/avg/min/max: Statistical aggregations
    - count: Count records
    - first/last: Get first/last value
    - concat: Concatenate strings
    - collect: Collect values into array

    Use cases:
    - Data preprocessing for analytics
    - ETL (Extract, Transform, Load) operations
    - Report generation
    - Data cleaning and normalization
    - Feature engineering for ML
    - API response transformation
  `;
  static readonly alias = 'transform';

  constructor(
    params: DataTransformerToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - performs data transformation
   */
  async performAction(
    context?: BubbleContext
  ): Promise<DataTransformerToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log(`[DataTransformerTool] Executing operation: ${this.params.operation}`);

      const inputRecordCount = this.params.inputData.length;
      let outputData = [...this.params.inputData];
      const fieldsAdded: string[] = [];
      const fieldsRemoved: string[] = [];
      const fieldsModified: string[] = [];
      let recordsFiltered = 0;
      let recordsGrouped = 0;

      switch (this.params.operation) {
        case 'map':
          outputData = this.applyMapOperations(
            outputData,
            fieldsAdded,
            fieldsModified,
            fieldsRemoved
          );
          break;

        case 'filter':
          const beforeFilter = outputData.length;
          outputData = this.applyFilter(outputData);
          recordsFiltered = beforeFilter - outputData.length;
          break;

        case 'sort':
          outputData = this.applySort(outputData);
          break;

        case 'groupBy':
          const result = this.applyGroupBy(outputData);
          outputData = result.data;
          recordsGrouped = result.groupCount;
          break;

        case 'join':
          const joinResult = this.applyJoin(outputData);
          outputData = joinResult.data;
          fieldsAdded.push(...joinResult.fieldsAdded);
          break;

        case 'pivot':
          const pivotResult = this.applyPivot(outputData);
          outputData = pivotResult.data;
          fieldsAdded.push(...pivotResult.fieldsAdded);
          break;

        case 'unpivot':
          const unpivotResult = this.applyUnpivot(outputData);
          outputData = unpivotResult.data;
          break;

        case 'custom':
          outputData = this.applyCustomTransformation(outputData);
          break;

        default:
          throw new Error(`Unsupported operation: ${this.params.operation}`);
      }

      // Remove null fields if configured
      if (this.params.removeNullFields) {
        outputData = outputData.map((record) => {
          const cleaned: Record<string, unknown> = {};
          Object.entries(record).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
              cleaned[key] = value;
            }
          });
          return cleaned;
        });
      }

      const processingTime = Date.now() - startTime;

      console.log(`[DataTransformerTool] Transformation completed. Input: ${inputRecordCount}, Output: ${outputData.length}, Time: ${processingTime}ms`);

      return {
        outputData,
        inputRecordCount,
        outputRecordCount: outputData.length,
        fieldsAdded,
        fieldsRemoved,
        fieldsModified,
        transformationStats: {
          recordsFiltered,
          recordsGrouped,
          processingTime,
        },
        success: true,
        error: '',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[DataTransformerTool] Transformation failed: ${errorMessage}`);

      return {
        outputData: [],
        inputRecordCount: this.params.inputData.length,
        outputRecordCount: 0,
        fieldsAdded: [],
        fieldsRemoved: [],
        fieldsModified: [],
        transformationStats: {
          processingTime: Date.now() - startTime,
        },
        success: false,
        error: errorMessage,
      };
    }
  }

  /**
   * Apply map operations
   */
  private applyMapOperations(
    data: Record<string, unknown>[],
    fieldsAdded: string[],
    fieldsModified: string[],
    fieldsRemoved: string[]
  ): Record<string, unknown>[] {
    if (!this.params.mapOperations) {
      return data;
    }

    return data.map((record) => {
      const transformed = { ...record };

      this.params.mapOperations!.forEach((op) => {
        const { targetField, sourceField, transform, expression, format, lookupTable } = op;

        let value: unknown;

        switch (transform) {
          case 'copy':
            if (sourceField && sourceField in record) {
              value = record[sourceField];
              fieldsAdded.push(targetField);
            }
            break;

          case 'rename':
            if (sourceField && sourceField in record) {
              value = record[sourceField];
              delete transformed[sourceField];
              fieldsRemoved.push(sourceField);
              fieldsAdded.push(targetField);
            }
            break;

          case 'calculate':
            try {
              // Create a safe evaluation context
              const context = { ...record };
              value = this.evaluateExpression(expression || '', context);
              fieldsModified.push(targetField);
            } catch (e) {
              console.error(`Calculation failed for ${targetField}:`, e);
              value = null;
            }
            break;

          case 'format':
            if (sourceField && sourceField in record) {
              value = format!.replace(/{value}/g, String(record[sourceField]));
              fieldsModified.push(targetField);
            }
            break;

          case 'extract':
            if (sourceField && sourceField in record && expression) {
              const regex = new RegExp(expression);
              const match = String(record[sourceField]).match(regex);
              value = match ? match[1] || match[0] : null;
              fieldsAdded.push(targetField);
            }
            break;

          case 'lookup':
            if (sourceField && sourceField in record && lookupTable) {
              const key = String(record[sourceField]);
              value = lookupTable[key];
              fieldsAdded.push(targetField);
            }
            break;
        }

        if (value !== undefined) {
          transformed[targetField] = value;
        }
      });

      return transformed;
    });
  }

  /**
   * Evaluate expression safely using mathjs
   * SECURE: Uses mathjs library to prevent code injection attacks
   */
  private evaluateExpression(
    expression: string,
    context: Record<string, unknown>
  ): unknown {
    // Validate expression is not empty
    if (!expression || expression.trim().length === 0) {
      throw new Error('Expression cannot be empty');
    }

    // Validate expression length to prevent DoS attacks
    if (expression.length > 1000) {
      throw new Error('Expression too long (max 1000 characters)');
    }

    // Replace field references with actual values
    // Format: {fieldName} becomes actual value from context
    const sanitized = expression.replace(/\{(\w+)\}/g, (_, key) => {
      const value = context[key];
      // Only allow numeric values in expressions
      if (typeof value === 'number' && !isNaN(value)) {
        return String(value);
      }
      return '0';
    });

    // Validate that sanitized expression only contains safe characters
    // This prevents injection of non-mathematical code
    if (!/^[\d\s+\-*/().%]+$/.test(sanitized)) {
      throw new Error(
        `Invalid expression: "${expression}". Only mathematical operations are allowed.`
      );
    }

    try {
      // Use mathjs evaluate for secure math expression evaluation
      // mathjs provides a sandboxed environment that only allows mathematical operations
      const result = evaluate(sanitized);

      // Validate result is a number
      if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
        return result;
      }

      throw new Error(`Expression result is not a valid number: ${result}`);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Failed to evaluate expression "${expression}": ${errorMsg}`);
    }
  }

  /**
   * Apply filter conditions
   */
  private applyFilter(data: Record<string, unknown>[]): Record<string, unknown>[] {
    if (!this.params.filterConditions || this.params.filterConditions.length === 0) {
      return data;
    }

    return data.filter((record) => {
      return this.params.filterConditions!.every((condition) => {
        const { field, operator, value, values } = condition;
        const fieldValue = record[field];

        switch (operator) {
          case 'eq':
            return fieldValue === value;
          case 'ne':
            return fieldValue !== value;
          case 'gt':
            return Number(fieldValue) > Number(value);
          case 'lt':
            return Number(fieldValue) < Number(value);
          case 'gte':
            return Number(fieldValue) >= Number(value);
          case 'lte':
            return Number(fieldValue) <= Number(value);
          case 'contains':
            return String(fieldValue).includes(String(value));
          case 'startsWith':
            return String(fieldValue).startsWith(String(value));
          case 'endsWith':
            return String(fieldValue).endsWith(String(value));
          case 'in':
            return values ? values.includes(fieldValue) : false;
          case 'isNull':
            return fieldValue === null || fieldValue === undefined;
          default:
            return true;
        }
      });
    });
  }

  /**
   * Apply sort
   */
  private applySort(data: Record<string, unknown>[]): Record<string, unknown>[] {
    if (!this.params.sortFields || this.params.sortFields.length === 0) {
      return data;
    }

    return data.sort((a, b) => {
      for (const sortField of this.params.sortFields!) {
        const { field, order } = sortField;
        const aVal = a[field];
        const bVal = b[field];

        let comparison = 0;

        if (typeof aVal === 'number' && typeof bVal === 'number') {
          comparison = aVal - bVal;
        } else {
          comparison = String(aVal).localeCompare(String(bVal));
        }

        if (comparison !== 0) {
          return order === 'desc' ? -comparison : comparison;
        }
      }

      return 0;
    });
  }

  /**
   * Apply group by with aggregations
   */
  private applyGroupBy(
    data: Record<string, unknown>[]
  ): { data: Record<string, unknown>[]; groupCount: number } {
    if (!this.params.groupByFields || !this.params.aggregations) {
      throw new Error('groupByFields and aggregations are required for groupBy operation');
    }

    const groups = new Map<string, Record<string, unknown>[]>();

    // Group records
    data.forEach((record) => {
      const key = this.params
        .groupByFields!.map((field) => String(record[field] ?? ''))
        .join('|');

      if (!groups.has(key)) {
        groups.set(key, []);
      }

      groups.get(key)!.push(record);
    });

    // Apply aggregations
    const result: Record<string, unknown>[] = [];

    groups.forEach((groupRecords) => {
      const aggregated: Record<string, unknown> = {};

      // Add group by fields
      this.params.groupByFields!.forEach((field) => {
        aggregated[field] = groupRecords[0][field];
      });

      // Apply aggregations
      this.params.aggregations!.forEach((agg) => {
        const { field, operation, alias } = agg;
        const values = groupRecords.map((r) => r[field]).filter((v) => v != null);

        let resultValue: unknown;

        switch (operation) {
          case 'sum':
            resultValue = values.reduce((sum: number, v) => sum + Number(v), 0);
            break;
          case 'avg':
            resultValue =
              values.reduce((sum: number, v) => sum + Number(v), 0) / values.length;
            break;
          case 'min':
            resultValue = Math.min(...values.map((v) => Number(v)));
            break;
          case 'max':
            resultValue = Math.max(...values.map((v) => Number(v)));
            break;
          case 'count':
            resultValue = values.length;
            break;
          case 'first':
            resultValue = values[0];
            break;
          case 'last':
            resultValue = values[values.length - 1];
            break;
          case 'concat':
            resultValue = values.join(', ');
            break;
          case 'collect':
            resultValue = values;
            break;
          default:
            resultValue = null;
        }

        aggregated[alias || `${field}_${operation}`] = resultValue;
      });

      result.push(aggregated);
    });

    return { data: result, groupCount: groups.size };
  }

  /**
   * Apply join
   */
  private applyJoin(
    data: Record<string, unknown>[]
  ): { data: Record<string, unknown>[]; fieldsAdded: string[] } {
    if (!this.params.joinData || !this.params.joinKey) {
      throw new Error('joinData and joinKey are required for join operation');
    }

    const joinType = this.params.joinType || 'inner';
    const joinKey = this.params.joinKey;
    const fieldsAdded: string[] = [];

    // Create lookup map for join data
    const joinMap = new Map<string, Record<string, unknown>>();
    this.params.joinData.forEach((record) => {
      const key = String(record[joinKey]);
      joinMap.set(key, record);
    });

    const result: Record<string, unknown>[] = [];

    if (joinType === 'cross') {
      // Cross join: every combination
      data.forEach((leftRecord) => {
        this.params.joinData!.forEach((rightRecord) => {
          result.push({ ...leftRecord, ...rightRecord });
        });
      });
    } else {
      // Other join types
      data.forEach((leftRecord) => {
        const key = String(leftRecord[joinKey]);
        const rightRecord = joinMap.get(key);

        if (rightRecord) {
          // Inner or left join
          const joined = { ...leftRecord };

          // Prefix join fields to avoid collisions
          Object.entries(rightRecord).forEach(([k, v]) => {
            const prefixedKey = `${k}_joined`;
            joined[prefixedKey] = v;
            if (!fieldsAdded.includes(prefixedKey)) {
              fieldsAdded.push(prefixedKey);
            }
          });

          result.push(joined);
        } else if (joinType === 'left' || joinType === 'outer') {
          // Left join: keep left record even if no match
          result.push(leftRecord);
        }
      });

      // Right or outer join: add unmatched records from right
      if (joinType === 'right' || joinType === 'outer') {
        const leftKeys = new Set(data.map((r) => String(r[joinKey])));

        joinMap.forEach((rightRecord, key) => {
          if (!leftKeys.has(key)) {
            const joined: Record<string, unknown> = {};

            // Prefix join fields
            Object.entries(rightRecord).forEach(([k, v]) => {
              const prefixedKey = `${k}_joined`;
              joined[prefixedKey] = v;
            });

            result.push(joined);
          }
        });
      }
    }

    return { data: result, fieldsAdded };
  }

  /**
   * Apply pivot
   */
  private applyPivot(
    data: Record<string, unknown>[]
  ): { data: Record<string, unknown>[]; fieldsAdded: string[] } {
    if (!this.params.pivotField || !this.params.valueField) {
      throw new Error('pivotField and valueField are required for pivot operation');
    }

    const pivotField = this.params.pivotField;
    const valueField = this.params.valueField;
    const aggFunc = this.params.aggregateFunction || 'sum';

    // Group by all fields except pivot and value fields
    const groupByFields = Object.keys(data[0]).filter(
      (f) => f !== pivotField && f !== valueField
    );

    const groups = new Map<string, Record<string, unknown>[]>();

    data.forEach((record) => {
      const groupKey = groupByFields.map((f) => String(record[f])).join('|');

      if (!groups.has(groupKey)) {
        groups.set(groupKey, []);
      }

      groups.get(groupKey)!.push(record);
    });

    const result: Record<string, unknown>[] = [];
    const pivotValues = new Set<string>();
    const fieldsAdded: string[] = [];

    // Collect all pivot values
    data.forEach((record) => {
      pivotValues.add(String(record[pivotField]));
    });

    // Build pivoted records
    groups.forEach((groupRecords) => {
      const pivoted: Record<string, unknown> = {};

      // Add group by fields
      groupByFields.forEach((field, index) => {
        const groupKey = Array.from(groups.keys()).find((k) =>
          groups.get(k) === groupRecords
        );
        pivoted[field] = groupKey?.split('|')[index];
      });

      // Pivot the data
      Array.from(pivotValues).forEach((pivotValue) => {
        const matchingRecords = groupRecords.filter(
          (r) => String(r[pivotField]) === pivotValue
        );

        let aggregatedValue: unknown;

        switch (aggFunc) {
          case 'sum':
            aggregatedValue = matchingRecords.reduce(
              (sum, r) => sum + Number(r[valueField] || 0),
              0
            );
            break;
          case 'avg':
            aggregatedValue =
              matchingRecords.reduce(
                (sum, r) => sum + Number(r[valueField] || 0),
                0
              ) / matchingRecords.length;
            break;
          case 'min':
            aggregatedValue = Math.min(
              ...matchingRecords.map((r) => Number(r[valueField] || 0))
            );
            break;
          case 'max':
            aggregatedValue = Math.max(
              ...matchingRecords.map((r) => Number(r[valueField] || 0))
            );
            break;
          case 'count':
            aggregatedValue = matchingRecords.length;
            break;
          case 'first':
            aggregatedValue = matchingRecords[0]?.[valueField];
            break;
          case 'last':
            aggregatedValue = matchingRecords[matchingRecords.length - 1]?.[valueField];
            break;
          default:
            aggregatedValue = null;
        }

        pivoted[pivotValue] = aggregatedValue;
        if (!fieldsAdded.includes(pivotValue)) {
          fieldsAdded.push(pivotValue);
        }
      });

      result.push(pivoted);
    });

    return { data: result, fieldsAdded };
  }

  /**
   * Apply unpivot
   */
  private applyUnpivot(
    data: Record<string, unknown>[]
  ): { data: Record<string, unknown>[]; fieldsAdded: string[] } {
    if (!this.params.pivotField || !this.params.valueField) {
      throw new Error('pivotField and valueField are required for unpivot operation');
  }

    const pivotField = this.params.pivotField;
    const valueField = this.params.valueField;

    const result: Record<string, unknown>[] = [];
    const identifierFields = Object.keys(data[0]).filter(
      (f) => f !== pivotField && f !== valueField
    );

    data.forEach((record) => {
      const identifier = Object.fromEntries(
        identifierFields.map((f) => [f, record[f]])
      );

      Object.entries(record).forEach(([key, value]) => {
        if (!identifierFields.includes(key)) {
          result.push({
            ...identifier,
            [pivotField]: key,
            [valueField]: value,
          });
        }
      });
    });

    return { data: result, fieldsAdded: [pivotField, valueField] };
  }

  /**
   * Apply custom transformation
   * SECURITY WARNING: This feature is disabled by default due to code injection risks.
   * To enable, set environment variable ALLOW_CUSTOM_TRANSFORMATIONS=true
   *
   * If enabled, the script is subjected to strict validation before execution.
   */
  private applyCustomTransformation(
    data: Record<string, unknown>[]
  ): Record<string, unknown>[] {
    if (!this.params.customScript) {
      return data;
    }

    // Check if custom transformations are allowed
    const allowCustom = process.env.ALLOW_CUSTOM_TRANSFORMATIONS === 'true';

    if (!allowCustom) {
      throw new Error(
        'Custom transformations are disabled for security reasons. ' +
        'To enable, set environment variable ALLOW_CUSTOM_TRANSFORMATIONS=true. ' +
        'Warning: Only enable this if you trust the source of all transformation scripts.'
      );
    }

    // Validate script length to prevent DoS attacks
    if (this.params.customScript.length > 10000) {
      throw new Error('Custom script too long (max 10000 characters)');
    }

    // Strict validation: only allow specific safe patterns
    // This pattern allows: data.map/filter/reduce/sort, return, basic JS syntax
    // It BLOCKS: eval, Function, require, import, fetch, XMLHttpRequest, etc.
    const dangerousPatterns = [
      /\beval\s*\(/,
      /\bFunction\s*\(/,
      /\brequire\s*\(/,
      /\bimport\s+/,
      /\bfetch\s*\(/,
      /\bXMLHttpRequest/,
      /\bprocess\./,
      /\bchild_process/,
      /\bfs\./,
      /\b__dirname/,
      /\b__filename/,
      /\.\.\//,  // path traversal
      /document\./,
      /window\./,
      /localStorage/,
      /sessionStorage/,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(this.params.customScript)) {
        throw new Error(
          `Custom script contains dangerous pattern: ${pattern.source}. ` +
          'This operation is blocked for security reasons.'
        );
      }
    }

    try {
      // Create a sandboxed function with limited scope
      // Note: This is still potentially dangerous, which is why it requires explicit opt-in
      const transformFn = new Function(
        'data',
        '"use strict"; ' +
        'return (' + this.params.customScript + ')(data);'
      );

      const result = transformFn(data) as Record<string, unknown>[];

      // Validate result is an array
      if (!Array.isArray(result)) {
        throw new Error('Custom transformation must return an array');
      }

      // Validate all items are objects
      if (!result.every(item => typeof item === 'object' && item !== null)) {
        throw new Error('Custom transformation must return an array of objects');
      }

      // Log the transformation for audit trail
      console.warn(
        `[DataTransformerTool] Custom transformation executed. ` +
        `Script length: ${this.params.customScript.length}, ` +
        `Input records: ${data.length}, ` +
        `Output records: ${result.length}`
      );

      return result;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Custom transformation failed: ${errorMsg}`);
    }
  }
}
