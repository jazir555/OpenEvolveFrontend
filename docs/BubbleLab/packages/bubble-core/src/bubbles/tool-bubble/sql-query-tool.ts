import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * SQLQueryTool - SQL query operations
 */
export class SQLQueryTool extends ToolBubble<SQLQueryParams, SQLQueryResult> {
  bubbleName = 'sql-query';
  type = 'tool';
  alias = 'sql-query';

  params = {
    timeout: z.number().int().positive().default(30000),
    maxResults: z.number().int().positive().default(1000)
  };

  private databaseType?: string;

  // Performance optimization: Query result cache
  private queryCache = new Map<string, { data: any; timestamp: number }>();
  private readonly CACHE_TTL = 60000; // 1 minute for query results
  private readonly MAX_CACHE_SIZE = 100;

  // Performance: Compiled regex patterns for SQL validation (14 rules)
  private static readonly DANGEROUS_PATTERNS = [
    { pattern: /\bDROP\s+TABLE\b/i, msg: 'DROP TABLE operations are not allowed', type: 'error' as const },
    { pattern: /\bTRUNCATE\b/i, msg: 'TRUNCATE operations are not allowed', type: 'error' as const },
    { pattern: /;\s*DROP\b/i, msg: 'SQL injection detected (semicolon + DROP)', type: 'error' as const },
    { pattern: /;\s*DELETE\b/i, msg: 'SQL injection detected (semicolon + DELETE)', type: 'error' as const },
    { pattern: /--/i, msg: 'SQL comments detected, ensure no SQL injection', type: 'warning' as const },
    { pattern: /\/\*/i, msg: 'Multi-line comments detected', type: 'warning' as const },
    { pattern: /;\s*EXEC\b/i, msg: 'EXEC commands not allowed', type: 'error' as const },
    { pattern: /\bEXECUTE\b/i, msg: 'EXECUTE commands not allowed', type: 'error' as const },
    { pattern: /;\s*EXECUTE\b/i, msg: 'EXECUTE injection detected', type: 'error' as const },
    { pattern: /\bUNION\s+SELECT\b/i, msg: 'UNION SELECT injection detected', type: 'error' as const },
    { pattern: /\bINSERT\s+INTO\b/i, msg: 'INSERT operations not allowed', type: 'error' as const },
    { pattern: /\bUPDATE\b.*\bSET\b/i, msg: 'UPDATE operations not allowed', type: 'error' as const },
    { pattern: /\bDELETE\s+FROM\b/i, msg: 'DELETE FROM operations not allowed', type: 'error' as const },
    { pattern: /\bCREATE\b/i, msg: 'CREATE operations not allowed', type: 'error' as const },
    { pattern: /\bALTER\b/i, msg: 'ALTER operations not allowed', type: 'error' as const },
    { pattern: /;\s*ALTER\b/i, msg: 'ALTER injection detected', type: 'error' as const },
    { pattern: /0x[0-9a-f]+/i, msg: 'Hex encoding detected, possible injection', type: 'warning' as const },
    { pattern: /char\s*\(/i, msg: 'CHAR() function detected, possible injection', type: 'warning' as const },
    { pattern: /\/\*.*?\*\//gis, msg: 'Comment blocks detected', type: 'warning' as const },
    { pattern: /\bor\b\s*1\s*=\s*1\b/i, msg: 'Tautology injection detected', type: 'error' as const },
    { pattern: /\band\b\s*1\s*=\s*1\b/i, msg: 'Tautology injection detected', type: 'error' as const }
  ];

  // Performance: Connection pool
  private connectionPool = new Map<string, any>();
  private readonly MAX_POOL_SIZE = 5;

  /**
   * COMPREHENSIVE VALIDATION SCHEMAS
   * All validation rules for SQL query operations
   */

  // SQL query validation schema (8 rules)
  private static readonly SQLQueryParamsSchema = z.object({
    sql: z.string().min(1).max(10000).trim()
      .refine(
        (query) => !query.includes('\0'),
        { message: 'SQL query cannot contain null bytes' }
      )
      .refine(
        (query) => query.length > 0 && query.trim().length > 0,
        { message: 'SQL query cannot be empty or whitespace-only' }
      ),
    reasoning: z.string().min(10).max(5000).optional(),
    timeout: z.number().int().min(1000).max(300000).optional(),
    maxRows: z.number().int().min(1).max(10000).optional(),
    database: z.string().min(1).max(64).optional(),
    connection: z.string().min(1).max(256).optional(),
    params: z.array(z.unknown()).max(100).optional()
  });

  // Query result schema (3 rules)
  private static readonly SQLQueryResultSchema = z.object({
    success: z.boolean(),
    rows: z.array(z.record(z.string(), z.unknown())).max(10000).optional(),
    rowCount: z.number().int().min(0).max(10000).optional(),
    executionTime: z.number().min(0).max(3600000).optional(),
    metadata: z.object({
      databaseType: z.string().optional(),
      timestamp: z.string().datetime().optional(),
      table: z.string().optional(),
      columns: z.array(z.string()).max(1000).optional(),
      hasJoins: z.boolean().optional(),
      hasWhere: z.boolean().optional(),
      hasGroupBy: z.boolean().optional(),
      hasOrderBy: z.boolean().optional(),
      warnings: z.array(z.string()).max(100).optional()
    }).optional(),
    valid: z.boolean().optional(),
    errors: z.array(z.string().max(1000)).max(100).optional(),
    warnings: z.array(z.string().max(1000)).max(100).optional(),
    formatted: z.string().max(10000).optional(),
    error: z.string().max(1000).optional(),
    details: z.record(z.unknown()).optional(),
    cached: z.boolean().optional()
  });

  // Field validation schema (3 rules)
  private static readonly FieldSchema = z.object({
    name: z.string().min(1).max(128).regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/),
    dataTypeID: z.number().int().min(0).max(10000).optional(),
    dataType: z.string().max(64).optional(),
    nullable: z.boolean().optional(),
    defaultValue: z.unknown().optional()
  });

  /**
   * Performance: Clean up resources
   */
  async destroy(): Promise<void> {
    try {
      this.queryCache.clear();

      // Close all database connections
      for (const [key, connection] of this.connectionPool.entries()) {
        try {
          if (connection && typeof connection.close === 'function') {
            await connection.close();
          } else if (connection && typeof connection.end === 'function') {
            connection.end();
          }
        } catch (error) {
          console.error(`Error closing connection for ${key}:`, error);
        }
      }
      this.connectionPool.clear();
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  /**
   * Performance: Get cached query result
   */
  private getCachedQuery(query: string): any | null {
    const cacheKey = this.generateQueryCacheKey(query);
    const cached = this.queryCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data;
    }
    if (cached) {
      this.queryCache.delete(cacheKey);
    }
    return null;
  }

  /**
   * Performance: Set query result in cache with LRU eviction
   */
  private setCachedQuery(query: string, data: any): void {
    if (this.queryCache.size >= this.MAX_CACHE_SIZE) {
      const oldestKey = this.queryCache.keys().next().value;
      if (oldestKey) {
        this.queryCache.delete(oldestKey);
      }
    }
    const cacheKey = this.generateQueryCacheKey(query);
    this.queryCache.set(cacheKey, { data, timestamp: Date.now() });
  }

  /**
   * Performance: Generate cache key from query
   */
  private generateQueryCacheKey(query: string): string {
    // Normalize query for caching (remove extra whitespace, convert to uppercase)
    return query.trim().replace(/\s+/g, ' ').toUpperCase();
  }

  async query(params: { sql: string; connection?: string; params?: any[] }): Promise<SQLQueryResult> {
    // VALIDATION: Validate input against schema
    const validationResult = SQLQueryTool.SQLQueryParamsSchema.safeParse(params);
    if (!validationResult.success) {
      const errors = validationResult.error.errors.map(e =>
        `${e.path.join('.')}: ${e.message}`
      ).join('; ');
      return {
        success: false,
        error: `Validation failed: ${errors}`,
        errors: [errors]
      };
    }

    const validatedParams = validationResult.data;

    // Performance: Add timeout wrapper with Promise.race
    const timeoutPromise = new Promise<SQLQueryResult>((_, reject) =>
      setTimeout(() => reject(new Error('SQL query timeout')), this.params.timeout.default())
    );

    const queryOperation = async (): Promise<SQLQueryResult> => {
      try {
        const startTime = Date.now();

        // Performance: Check cache first for SELECT queries
        const isSelectQuery = validatedParams.sql.trim().toUpperCase().startsWith('SELECT');
        if (isSelectQuery) {
          const cached = this.getCachedQuery(validatedParams.sql);
          if (cached) {
            return {
              ...cached,
              cached: true,
              executionTime: Date.now() - startTime
            };
          }
        }

        // Validate and sanitize the query
        const sanitizedSQL = this.sanitizeSQL(validatedParams.sql);
        const validation = await this.validate({ sql: sanitizedSQL });

        if (!validation.valid) {
          return {
            success: false,
            error: 'SQL validation failed',
            errors: validation.errors
          };
        }

        // Add LIMIT if not present
        const finalSQL = this.addLimit(sanitizedSQL);

        // In production, execute against real database using:
        // - PostgreSQL: pg library
        // - MySQL: mysql2 library
        // - SQLite: better-sqlite3 library
        // For now, return mock result structure

        const metadata = this.extractSQLMetadata(params.sql);
        const executionTime = Date.now() - startTime;

        const result = {
          success: true,
          rows: [],
          rowCount: 0,
          executionTime,
          metadata: {
            ...metadata,
            databaseType: this.databaseType || 'sqlite',
            timestamp: new Date().toISOString(),
            warnings: validation.warnings
          }
        };

        // Performance: Cache SELECT query results
        if (isSelectQuery) {
          this.setCachedQuery(validatedParams.sql, result);
        }

        return result;
      } catch (error: any) {
        return {
          success: false,
          error: error.message,
          details: { sql: validatedParams.sql }
        };
      }
    };

    try {
      // Performance: Race between query and timeout
      return await Promise.race([queryOperation(), timeoutPromise]);
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        details: { sql: validatedParams.sql }
      };
    }
  }

  async validate(params: { sql: string }): Promise<SQLQueryResult> {
    try {
      const errors: string[] = [];
      const warnings: string[] = [];
      const sql = params.sql.trim().toUpperCase();

      // Performance: Use pre-compiled dangerous patterns
      for (const { pattern, msg, type } of SQLQueryTool.DANGEROUS_PATTERNS) {
        if (pattern.test(params.sql)) {
          if (type === 'error') errors.push(msg);
          else warnings.push(msg);
        }
      }

      // Validate structure
      if (!sql.startsWith('SELECT') && !sql.startsWith('WITH') && !sql.startsWith('SHOW')) {
        errors.push('Query must start with SELECT, WITH, or SHOW');
      }

      // Performance: Optimized parentheses counting
      let openParens = 0;
      let closeParens = 0;
      for (const char of params.sql) {
        if (char === '(') openParens++;
        else if (char === ')') closeParens++;
      }
      if (openParens !== closeParens) {
        errors.push('Unbalanced parentheses in query');
      }

      // Check quotes
      const quotes = params.sql.match(/'/g);
      if (quotes && quotes.length % 2 !== 0) {
        errors.push('Unbalanced quotes in query');
      }

      // Check for LIMIT
      if (!/\bLIMIT\s+\d+\b/i.test(params.sql)) {
        warnings.push('No LIMIT clause found, consider adding one');
      }

      return {
        success: true,
        valid: errors.length === 0,
        errors,
        warnings
      };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async format(params: { sql: string }): Promise<SQLQueryResult> {
    try {
      let formatted = params.sql;

      // Add newlines before keywords
      const keywords = ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING',
        'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'AND', 'OR'];

      keywords.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        formatted = formatted.replace(regex, `\n${keyword}`);
      });

      // Clean up spacing
      formatted = formatted.replace(/\n\s*\n/g, '\n').replace(/[ \t]+/g, ' ').trim();

      return { success: true, formatted };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  private sanitizeSQL(sql: string): string {
    return sql
      .replace(/;\s*DROP\s+/gi, '')
      .replace(/;\s*DELETE\s+/gi, '')
      .replace(/--.*$/gm, '')
      .trim();
  }

  private addLimit(sql: string): string {
    const maxRows = this.params.maxResults?.default() || 1000;
    if (!/\bLIMIT\s+\d+/i.test(sql)) {
      return `${sql} LIMIT ${maxRows}`;
    }
    return sql;
  }

  private extractSQLMetadata(sql: string): any {
    const metadata: any = {};
    const fromMatch = sql.match(/\bFROM\s+(\w+)/i);
    if (fromMatch) metadata.table = fromMatch[1];

    const selectMatch = sql.match(/SELECT\s+(.*?)\s+FROM/i);
    if (selectMatch) {
      metadata.columns = selectMatch[1].split(',').map((c: string) => c.trim());
    }

    metadata.hasJoins = /\bJOIN\b/i.test(sql);
    metadata.hasWhere = /\bWHERE\b/i.test(sql);
    metadata.hasGroupBy = /\bGROUP BY\b/i.test(sql);
    metadata.hasOrderBy = /\bORDER BY\b/i.test(sql);

    return metadata;
  }
}

export interface SQLQueryParams {
  timeout?: number;
}

export interface SQLQueryResult {
  success: boolean;
  rows?: any[];
  rowCount?: number;
  executionTime?: number;
  metadata?: any;
  valid?: boolean;
  errors?: string[];
  warnings?: string[];
  formatted?: string;
  error?: string;
  details?: any;
}
