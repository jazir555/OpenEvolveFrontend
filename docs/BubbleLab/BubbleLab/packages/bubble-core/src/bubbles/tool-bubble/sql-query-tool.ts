import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * Constants for SQLQueryTool
 */
const DEFAULT_TIMEOUT_MS = 30000;
const MAX_QUERY_LENGTH = 10000;
const DEFAULT_BATCH_SIZE = 100;

/**
 * Parameters for SQL query operation
 */
interface QueryParams {
  query: string;
  database?: string;
  params?: Record<string, unknown>;
  timeout?: number;
}

/**
 * Parameters for SQL validation operation
 */
interface ValidateParams {
  query: string;
  checkSyntax?: boolean;
  checkPermissions?: boolean;
  dryRun?: boolean;
}

/**
 * Parameters for SQL formatting operation
 */
interface FormatParams {
  query: string;
  language?: 'sql' | 'mysql' | 'postgresql' | 'sqlite';
  uppercase?: boolean;
  indent?: number;
}

/**
 * Query result interface
 */
interface QueryResult {
  rows?: Array<Record<string, unknown>>;
  rowCount?: number;
  fields?: Array<{ name: string; type: string }>;
  executionTime?: number;
}

/**
 * Validation result interface
 */
interface ValidationResult {
  isValid: boolean;
  syntax?: {
    valid: boolean;
    errors?: Array<{ line: number; column: number; message: string }>;
  };
  permissions?: {
    valid: boolean;
    missing?: string[];
  };
  estimatedCost?: number;
}

/**
 * Format result interface
 */
interface FormatResult {
  formatted: string;
  original: string;
  changes: number;
}

/**
 * Input parameters for SQLQueryTool
 */
export interface SQLQueryParams {
  timeout?: number;
  query?: QueryParams;
  validate?: ValidateParams;
  format?: FormatParams;
}

/**
 * Result of SQLQueryTool operation
 */
export interface SQLQueryResult {
  success: boolean;
  result?: QueryResult | ValidationResult | FormatResult;
  error?: string;
}

/**
 * SQLQueryTool - Performs SQL query execution, validation, and formatting
 *
 * This tool provides three main operations:
 * 1. Query: Executes SQL queries with parameter binding and timeout control
 * 2. Validate: Validates SQL syntax and permissions without executing
 * 3. Format: Formats SQL queries for readability
 *
 * All operations include proper error handling, SQL injection protection, and result formatting.
 */
export class SQLQueryTool extends ToolBubble<SQLQueryParams, SQLQueryResult> {
  bubbleName = 'sqlquery';
  type = 'tool';
  alias = 'sqlquery';

  params = {
    timeout: z.number().int().positive().default(DEFAULT_TIMEOUT_MS)
  };

  /**
   * Executes the SQL query operation
   * @param input - Operation parameters
   * @returns Promise<SQLQueryResult> - Result with query data
   */
  async execute(input: SQLQueryParams): Promise<SQLQueryResult> {
    try {
      const result = await this.process(input);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'SQL operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Processes the input and routes to appropriate operation
   * @param input - Operation parameters
   * @returns Promise<QueryResult | ValidationResult | FormatResult> - Processed result
   */
  private async process(input: SQLQueryParams): Promise<QueryResult | ValidationResult | FormatResult> {
    if (input.query) {
      return await this.query(input.query);
    } else if (input.validate) {
      return await this.validate(input.validate);
    } else if (input.format) {
      return await this.format(input.format);
    }
    throw new Error('No valid operation parameters provided');
  }

  /**
   * Executes a SQL query with parameters
   * @param params - Query parameters including SQL and parameter bindings
   * @returns Promise<QueryResult> - Query execution result
   */
  async query(params: QueryParams): Promise<QueryResult> {
    try {
      this.validateQuery(params.query);

      const startTime = Date.now();
      const result = await this.client.query(params);
      const executionTime = Date.now() - startTime;

      return {
        rows: result.rows,
        rowCount: result.rowCount,
        fields: result.fields,
        executionTime
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Query execution failed';
      throw new Error(`Failed to execute query: ${errorMessage}`);
    }
  }

  /**
   * Validates a SQL query without executing it
   * @param params - Validation parameters
   * @returns Promise<ValidationResult> - Validation result with details
   */
  async validate(params: ValidateParams): Promise<ValidationResult> {
    try {
      if (!params.query || params.query.trim().length === 0) {
        throw new Error('Query cannot be empty');
      }

      const result: ValidationResult = {
        isValid: true,
        syntax: { valid: true },
        permissions: { valid: true }
      };

      // Check syntax if requested
      if (params.checkSyntax) {
        const syntaxResult = await this.client.validate({
          query: params.query,
          checkType: 'syntax'
        });
        result.syntax = syntaxResult;
        if (!syntaxResult.valid) {
          result.isValid = false;
        }
      }

      // Check permissions if requested
      if (params.checkPermissions && !params.dryRun) {
        const permissionResult = await this.client.validate({
          query: params.query,
          checkType: 'permissions'
        });
        result.permissions = permissionResult;
        if (!permissionResult.valid) {
          result.isValid = false;
        }
      }

      // Perform dry run if requested
      if (params.dryRun) {
        const dryRunResult = await this.client.query({
          query: params.query,
          dryRun: true
        });
        result.estimatedCost = dryRunResult.estimatedCost;
      }

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Validation failed';
      throw new Error(`Failed to validate query: ${errorMessage}`);
    }
  }

  /**
   * Formats a SQL query for readability
   * @param params - Formatting parameters
   * @returns Promise<FormatResult> - Formatted query result
   */
  async format(params: FormatParams): Promise<FormatResult> {
    try {
      if (!params.query || params.query.trim().length === 0) {
        throw new Error('Query cannot be empty');
      }

      const options = {
        language: params.language || 'sql',
        uppercase: params.uppercase !== false,
        indent: params.indent || 2
      };

      const formatted = await this.client.format({
        query: params.query,
        ...options
      });

      // Count the number of changes made
      const changes = this.countQueryChanges(params.query, formatted);

      return {
        formatted,
        original: params.query,
        changes
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Format operation failed';
      throw new Error(`Failed to format query: ${errorMessage}`);
    }
  }

  /**
   * Validates query string for basic constraints
   * @param query - Query string to validate
   * @throws Error if validation fails
   */
  private validateQuery(query: string): void {
    if (!query || query.trim().length === 0) {
      throw new Error('Query cannot be empty');
    }

    if (query.length > MAX_QUERY_LENGTH) {
      throw new Error(`Query exceeds maximum length of ${MAX_QUERY_LENGTH} characters`);
    }

    // Check for potentially dangerous operations
    const dangerousKeywords = ['DROP\\s+DATABASE', 'DROP\\s+TABLE', 'TRUNCATE', 'DELETE\\s+FROM.+WHERE\\s*1\\s*=\\s*1'];
    const regex = new RegExp(dangerousKeywords.join('|'), 'gi');

    if (regex.test(query)) {
      throw new Error('Query contains potentially dangerous operations');
    }
  }

  /**
   * Counts the number of changes made during formatting
   * @param original - Original query
   * @param formatted - Formatted query
   * @returns Number of changes
   */
  private countQueryChanges(original: string, formatted: string): number {
    const originalLines = original.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    const formattedLines = formatted.split('\n').map(line => line.trim()).filter(line => line.length > 0);

    // Count line differences
    return Math.abs(originalLines.length - formattedLines.length) +
           originalLines.filter(line => !formattedLines.includes(line)).length;
  }
}
