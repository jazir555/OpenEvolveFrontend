import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * Constants for JSONValidatorTool
 */
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_JSON_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_NESTING_DEPTH = 100;

/**
 * Parameters for JSON validation operation
 */
interface ValidateParams {
  json: string;
  schema?: Record<string, unknown>;
  strictMode?: boolean;
  allowComments?: boolean;
  allowTrailingCommas?: boolean;
}

/**
 * Parameters for JSON transformation operation
 */
interface TransformParams {
  json: string;
  transformations: Array<{
    path: string;
    operation: 'set' | 'delete' | 'rename' | 'map';
    value?: unknown;
  }>;
  format?: boolean;
}

/**
 * Parameters for JSON query operation
 */
interface QueryParams {
  json: string;
  path: string;
  returnType?: 'value' | 'path' | 'exists';
}

/**
 * Validation result interface
 */
interface ValidationResult {
  isValid: boolean;
  errors?: Array<{
    path: string;
    message: string;
    line?: number;
    column?: number;
  }>;
  warnings?: Array<{
    path: string;
    message: string;
  }>;
}

/**
 * Transformation result interface
 */
interface TransformResult {
  transformed: string;
  original: string;
  changes: number;
  appliedTransformations: Array<{
    path: string;
    operation: string;
    success: boolean;
  }>;
}

/**
 * Query result interface
 */
interface QueryResult {
  found: boolean;
  path: string;
  value?: unknown;
  type?: string;
  exists?: boolean;
}

/**
 * Input parameters for JSONValidatorTool
 */
export interface JSONValidatorParams {
  timeout?: number;
  validate?: ValidateParams;
  transform?: TransformParams;
  query?: QueryParams;
}

/**
 * Result of JSONValidatorTool operation
 */
export interface JSONValidatorResult {
  success: boolean;
  result?: ValidationResult | TransformResult | QueryResult;
  error?: string;
}

/**
 * JSONValidatorTool - Performs JSON validation, transformation, and querying
 *
 * This tool provides three main operations:
 * 1. Validate: Validates JSON structure against schema with detailed error reporting
 * 2. Transform: Applies transformations to JSON data with change tracking
 * 3. Query: Queries JSON data using JSONPath or similar query syntax
 *
 * All operations include proper error handling, size validation, and result formatting.
 */
export class JSONValidatorTool extends ToolBubble<JSONValidatorParams, JSONValidatorResult> {
  bubbleName = 'jsonvalidator';
  type = 'tool';
  alias = 'jsonvalidator';

  params = {
    timeout: z.number().int().positive().default(DEFAULT_TIMEOUT_MS)
  };

  /**
   * Executes the JSON validator operation
   * @param input - Operation parameters
   * @returns Promise<JSONValidatorResult> - Result with validation/transform/query data
   */
  async execute(input: JSONValidatorParams): Promise<JSONValidatorResult> {
    try {
      const result = await this.process(input);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'JSON operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Processes the input and routes to appropriate operation
   * @param input - Operation parameters
   * @returns Promise<ValidationResult | TransformResult | QueryResult> - Processed result
   */
  private async process(input: JSONValidatorParams): Promise<ValidationResult | TransformResult | QueryResult> {
    if (input.validate) {
      return await this.validate(input.validate);
    } else if (input.transform) {
      return await this.transform(input.transform);
    } else if (input.query) {
      return await this.query(input.query);
    }
    throw new Error('No valid operation parameters provided');
  }

  /**
   * Validates JSON data against optional schema
   * @param params - Validation parameters
   * @returns Promise<ValidationResult> - Validation result with errors and warnings
   */
  async validate(params: ValidateParams): Promise<ValidationResult> {
    try {
      this.validateJsonSize(params.json);

      const result: ValidationResult = {
        isValid: true,
        errors: [],
        warnings: []
      };

      // Parse JSON
      let jsonData: unknown;
      try {
        jsonData = JSON.parse(params.json);
      } catch (error) {
        const parseError = error instanceof Error ? error.message : 'Unknown parse error';
        result.isValid = false;
        result.errors = [{
          path: 'root',
          message: `Failed to parse JSON: ${parseError}`
        }];
        return result;
      }

      // Validate schema if provided
      if (params.schema) {
        const schemaValidation = await this.client.validate({
          data: jsonData,
          schema: params.schema,
          strictMode: params.strictMode || false
        });

        if (!schemaValidation.isValid) {
          result.isValid = false;
          result.errors = [...(result.errors || []), ...(schemaValidation.errors || [])];
        }
      }

      // Check nesting depth
      const depthCheck = this.checkNestingDepth(jsonData);
      if (!depthCheck.valid) {
        result.isValid = false;
        result.errors = [...(result.errors || []), ...depthCheck.errors];
      }

      // Add warnings for potential issues
      if (params.strictMode) {
        result.warnings = this.analyzeJsonQuality(jsonData);
      }

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Validation failed';
      throw new Error(`Failed to validate JSON: ${errorMessage}`);
    }
  }

  /**
   * Transforms JSON data with specified transformations
   * @param params - Transformation parameters
   * @returns Promise<TransformResult> - Transformation result with change tracking
   */
  async transform(params: TransformParams): Promise<TransformResult> {
    try {
      this.validateJsonSize(params.json);

      // Parse JSON
      let jsonData: unknown;
      try {
        jsonData = JSON.parse(params.json);
      } catch (error) {
        throw new Error('Invalid JSON input');
      }

      const appliedTransformations: Array<{
        path: string;
        operation: string;
        success: boolean;
      }> = [];

      // Apply transformations
      for (const transformation of params.transformations) {
        try {
          await this.client.transform({
            data: jsonData,
            path: transformation.path,
            operation: transformation.operation,
            value: transformation.value
          });
          appliedTransformations.push({
            path: transformation.path,
            operation: transformation.operation,
            success: true
          });
        } catch (error) {
          appliedTransformations.push({
            path: transformation.path,
            operation: transformation.operation,
            success: false
          });
        }
      }

      // Format if requested
      const transformed = params.format
        ? JSON.stringify(jsonData, null, 2)
        : JSON.stringify(jsonData);

      return {
        transformed,
        original: params.json,
        changes: appliedTransformations.filter(t => t.success).length,
        appliedTransformations
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Transformation failed';
      throw new Error(`Failed to transform JSON: ${errorMessage}`);
    }
  }

  /**
   * Queries JSON data using path expressions
   * @param params - Query parameters
   * @returns Promise<QueryResult> - Query result with value and metadata
   */
  async query(params: QueryParams): Promise<QueryResult> {
    try {
      this.validateJsonSize(params.json);

      // Parse JSON
      let jsonData: unknown;
      try {
        jsonData = JSON.parse(params.json);
      } catch (error) {
        throw new Error('Invalid JSON input');
      }

      // Execute query
      const queryResult = await this.client.query({
        data: jsonData,
        path: params.path,
        returnType: params.returnType || 'value'
      });

      return {
        found: queryResult.found,
        path: params.path,
        value: queryResult.value,
        type: queryResult.value !== undefined ? this.getValueType(queryResult.value) : undefined,
        exists: queryResult.found
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Query failed';
      throw new Error(`Failed to query JSON: ${errorMessage}`);
    }
  }

  /**
   * Validates JSON size against maximum allowed size
   * @param json - JSON string to validate
   * @throws Error if size exceeds maximum
   */
  private validateJsonSize(json: string): void {
    const size = new Blob([json]).size;

    if (size > MAX_JSON_SIZE) {
      throw new Error(`JSON size exceeds maximum allowed size of ${MAX_JSON_SIZE} bytes`);
    }
  }

  /**
   * Checks nesting depth of JSON data
   * @param data - Parsed JSON data
   * @returns Validation result with errors if depth exceeds maximum
   */
  private checkNestingDepth(data: unknown, currentDepth: number = 0): {
    valid: boolean;
    errors?: Array<{ path: string; message: string }>;
  } {
    if (currentDepth > MAX_NESTING_DEPTH) {
      return {
        valid: false,
        errors: [{
          path: 'root',
          message: `Nesting depth exceeds maximum of ${MAX_NESTING_DEPTH}`
        }]
      };
    }

    if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
      for (const value of Object.values(data as Record<string, unknown>)) {
        const result = this.checkNestingDepth(value, currentDepth + 1);
        if (!result.valid) {
          return result;
        }
      }
    } else if (Array.isArray(data)) {
      for (const item of data) {
        const result = this.checkNestingDepth(item, currentDepth + 1);
        if (!result.valid) {
          return result;
        }
      }
    }

    return { valid: true };
  }

  /**
   * Analyzes JSON data for quality issues
   * @param data - Parsed JSON data
   * @returns Array of warnings
   */
  private analyzeJsonQuality(data: unknown): Array<{ path: string; message: string }> {
    const warnings: Array<{ path: string; message: string }> = [];

    // Check for empty strings
    const checkEmptyStrings = (obj: unknown, path: string = 'root'): void => {
      if (typeof obj === 'string' && obj.trim().length === 0) {
        warnings.push({ path, message: 'Empty string found' });
      } else if (typeof obj === 'object' && obj !== null) {
        for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
          checkEmptyStrings(value, `${path}.${key}`);
        }
      }
    };

    checkEmptyStrings(data);

    return warnings;
  }

  /**
   * Gets the type of a value
   * @param value - Value to check
   * @returns Type name as string
   */
  private getValueType(value: unknown): string {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';
    return typeof value;
  }
}
