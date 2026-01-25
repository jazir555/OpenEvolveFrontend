import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * JSONValidatorTool - JSON validation and transformation
 */
export class JSONValidatorTool extends ToolBubble<JSONValidatorParams, JSONValidatorResult> {
  bubbleName = 'json-validator';
  type = 'tool';
  alias = 'json-validator';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  // Performance optimization: Validation result cache
  private validationCache = new Map<string, { data: any; timestamp: number }>();
  private readonly CACHE_TTL = 300000; // 5 minutes
  private readonly MAX_CACHE_SIZE = 200;

  // Performance: Pre-compiled JSON path regex
  private static readonly PATH_SPLIT_REGEX = /\./g;
  private static readonly BRACKET_REGEX = /\[(\d+)\]/g;

  /**
   * COMPREHENSIVE VALIDATION SCHEMAS
   * All validation rules for JSON validation operations
   */

  // JSON path validation schema (2 rules)
  private static readonly JSONPathSchema = z.string().min(1).max(1024)
    .regex(/^[a-zA-Z_][a-zA-Z0-9_\[\].*]*$/, 'Invalid JSON path format');

  // Custom validation rule schema (7 rules)
  private static readonly CustomRuleSchema = z.object({
    field: z.string().min(1).max(256)
      .regex(/^[a-zA-Z_][a-zA-Z0-9_.*\[\]]*$/, 'Invalid field path'),
    rule: z.enum(['required', 'regex', 'range', 'length', 'enum', 'type', 'format']),
    value: z.unknown().optional(),
    values: z.array(z.unknown()).max(100).optional(),
    message: z.string().min(1).max(1000)
  }).refine(
    (rule) => {
      if (rule.rule === 'regex') {
        return typeof rule.value === 'string';
      }
      if (rule.rule === 'range') {
        return Array.isArray(rule.value) &&
          rule.value.length === 2 &&
          typeof rule.value[0] === 'number' &&
          typeof rule.value[1] === 'number';
      }
      if (rule.rule === 'length') {
        return Array.isArray(rule.value) &&
          rule.value.length === 2 &&
          typeof rule.value[0] === 'number' &&
          typeof rule.value[1] === 'number';
      }
      if (rule.rule === 'enum') {
        return Array.isArray(rule.value) && rule.value.length <= 100;
      }
      return true;
    },
    { message: 'Rule value does not match rule type' }
  );

  // JSON patch operation schema (5 rules)
  private static readonly JSONPatchSchema = z.object({
    op: z.enum(['add', 'remove', 'replace', 'move', 'copy', 'test']),
    path: z.string().min(1).max(1024),
    value: z.unknown().optional(),
    from: z.string().min(1).max(1024).optional()
  }).refine(
    (patch) => {
      if (['move', 'copy'].includes(patch.op)) {
        return !!patch.from;
      }
      if (['add', 'replace', 'test'].includes(patch.op)) {
        return patch.value !== undefined;
      }
      return true;
    },
    { message: 'Patch operation missing required field' }
  );

  // Main JSON validator parameters schema (7 rules)
  private static readonly JSONValidatorParamsSchema = z.object({
    jsonData: z.string().min(1).max(1e7), // Max 10MB
    schema: z.record(z.string(), z.union([z.string(), z.array(z.string())])).max(100).optional(),
    queryPath: JSONValidatorTool.JSONPathSchema.optional(),
    customRules: z.array(JSONValidatorTool.CustomRuleSchema).max(100).optional(),
    transformations: z.array(z.object({
      type: z.enum(['rename', 'delete', 'add', 'copy', 'move']),
      oldKey: z.string().min(1).max(256).optional(),
      newKey: z.string().min(1).max(256).optional(),
      key: z.string().min(1).max(256).optional(),
      value: z.unknown().optional(),
      from: z.string().min(1).max(256).optional(),
      path: z.string().min(1).max(256).optional()
    })).max(100).optional(),
    patches: z.array(JSONValidatorTool.JSONPatchSchema).max(100).optional(),
    maxDepth: z.number().int().min(1).max(100).default(100),
    timeout: z.number().int().positive().max(300000).default(30000)
  });

  /**
   * Performance: Clean up resources
   */
  async destroy(): Promise<void> {
    try {
      this.validationCache.clear();
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  /**
   * Performance: Get cached validation result
   */
  private getCachedValidation(key: string): any | null {
    const cached = this.validationCache.get(key);
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data;
    }
    if (cached) {
      this.validationCache.delete(key);
    }
    return null;
  }

  /**
   * Performance: Set validation result in cache with LRU eviction
   */
  private setCachedValidation(key: string, data: any): void {
    if (this.validationCache.size >= this.MAX_CACHE_SIZE) {
      const oldestKey = this.validationCache.keys().next().value;
      if (oldestKey) {
        this.validationCache.delete(oldestKey);
      }
    }
    this.validationCache.set(key, { data, timestamp: Date.now() });
  }

  /**
   * Performance: Generate cache key from JSON and schema
   */
  private generateCacheKey(json: any, schema?: any): string {
    try {
      const jsonStr = typeof json === 'string' ? json : JSON.stringify(json);
      const schemaStr = schema ? JSON.stringify(schema) : '';
      return `${jsonStr}-${schemaStr}`;
    } catch {
      return String(Date.now());
    }
  }

  async validate(params: { json: any; schema?: any }): Promise<JSONValidatorResult> {
    // VALIDATION: Check JSON size
    if (typeof params.json === 'string') {
      if (params.json.length > 1e7) { // 10MB
        return {
          success: false,
          error: 'JSON data exceeds maximum size of 10MB'
        };
      }
    }

    // VALIDATION: Validate custom rules if provided
    if (params.customRules) {
      const rulesValidation = z.array(JSONValidatorTool.CustomRuleSchema).max(100).safeParse(params.customRules);
      if (!rulesValidation.success) {
        return {
          success: false,
          error: `Invalid custom rules: ${rulesValidation.error.errors.map(e => e.message).join(', ')}`
        };
      }
    }

    // VALIDATION: Validate JSON depth
    const checkDepth = (obj: any, depth: number = 0): number => {
      if (depth > 100) return depth;
      if (typeof obj === 'object' && obj !== null) {
        let maxDepth = depth;
        for (const value of Object.values(obj)) {
          maxDepth = Math.max(maxDepth, checkDepth(value, depth + 1));
        }
        return maxDepth;
      }
      return depth;
    };

    try {
      const json = typeof params.json === 'string' ? JSON.parse(params.json) : params.json;
      const depth = checkDepth(json);
      if (depth > 100) {
        return {
          success: false,
          error: `JSON depth exceeds maximum of 100 levels (actual: ${depth})`
        };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }

    // Performance: Add timeout wrapper with Promise.race
    const timeoutPromise = new Promise<JSONValidatorResult>((_, reject) =>
      setTimeout(() => reject(new Error('JSON validation timeout')), this.params.timeout.default())
    );

    const validationOperation = async (): Promise<JSONValidatorResult> => {
      try {
        // Performance: Check cache first
        const cacheKey = this.generateCacheKey(params.json, params.schema);
        const cached = this.getCachedValidation(cacheKey);
        if (cached) {
          return { success: true, ...cached, cached: true };
        }

        let isValid = true;
        const errors = [];

        if (typeof params.json === 'string') {
          try {
            JSON.parse(params.json);
          } catch {
            isValid = false;
            errors.push('Invalid JSON syntax');
          }
        }

        if (params.schema) {
          // Performance: Optimized schema validation
          const json = typeof params.json === 'string' ? JSON.parse(params.json) : params.json;

          for (const [key, type] of Object.entries(params.schema)) {
            if (!(key in json)) {
              isValid = false;
              errors.push(`Missing required field: ${key}`);
            }
          }
        }

        const result = { success: true, valid: isValid, errors };

        // Performance: Cache validation result
        this.setCachedValidation(cacheKey, result);

        return result;
      } catch (error: any) {
        return { success: false, error: error.message };
      }
    };

    try {
      // Performance: Race between validation and timeout
      return await Promise.race([validationOperation(), timeoutPromise]);
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transform(params: { json: any; transformations: any[] }): Promise<JSONValidatorResult> {
    try {
      let result = params.json;
      params.transformations.forEach(t => {
        if (t.type === 'rename') {
          result[t.newKey] = result[t.oldKey];
          delete result[t.oldKey];
        } else if (t.type === 'delete') {
          delete result[t.key];
        } else if (t.type === 'add') {
          result[t.key] = t.value;
        }
      });
      return { success: true, transformed: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transform(params: { json: any; transformations: any[] }): Promise<JSONValidatorResult> {
    try {
      // Performance: Deep clone to avoid mutating original
      let result = typeof params.json === 'string' ? JSON.parse(params.json) : JSON.parse(JSON.stringify(params.json));

      // Performance: Optimized transformations loop
      const transformations = params.transformations || [];
      for (const t of transformations) {
        if (t.type === 'rename') {
          if (t.oldKey in result) {
            result[t.newKey] = result[t.oldKey];
            delete result[t.oldKey];
          }
        } else if (t.type === 'delete') {
          delete result[t.key];
        } else if (t.type === 'add') {
          result[t.key] = t.value;
        }
      }

      return { success: true, transformed: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async query(params: { json: any; path: string }): Promise<JSONValidatorResult> {
    try {
      // Performance: Optimized path parsing with support for array indices
      const pathParts: string[] = [];
      const arrayIndices: number[] = [];

      let lastIndex = 0;
      let match;

      // Performance: Use pre-compiled regex for bracket extraction
      JSONValidatorTool.BRACKET_REGEX.lastIndex = 0;

      while ((match = JSONValidatorTool.BRACKET_REGEX.exec(params.path)) !== null) {
        const beforeBracket = params.path.substring(lastIndex, match.index);
        if (beforeBracket) {
          pathParts.push(...beforeBracket.split('.').filter(p => p));
        }
        arrayIndices.push(parseInt(match[1], 10));
        lastIndex = match.index + match[0].length;
      }

      const remaining = params.path.substring(lastIndex);
      if (remaining) {
        pathParts.push(...remaining.split('.').filter(p => p));
      }

      // Performance: Navigate path
      let result = typeof params.json === 'string' ? JSON.parse(params.json) : params.json;
      let arrayIndex = 0;

      for (const key of pathParts) {
        if (result && typeof result === 'object') {
          result = result[key];
        } else {
          return { success: false, error: 'Path not found' };
        }

        // Performance: Handle array indices
        if (Array.isArray(result) && arrayIndex < arrayIndices.length) {
          const idx = arrayIndices[arrayIndex];
          if (idx < result.length) {
            result = result[idx];
            arrayIndex++;
          } else {
            return { success: false, error: 'Array index out of bounds' };
          }
        }
      }

      return { success: true, result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface JSONValidatorParams {
  timeout?: number;
}

export interface JSONValidatorResult {
  success: boolean;
  valid?: boolean;
  errors?: string[];
  transformed?: any;
  result?: any;
  error?: string;
}
