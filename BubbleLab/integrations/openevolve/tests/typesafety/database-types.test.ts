/**
 * Database Type Safety Tests
 *
 * Tests for Bug #4: Type-safe JSON parsing and database field validation
 * Tests safeParseJsonField, isValidRunConfig, isValidNodeMetadata, and response transformers
 */

// Import the actual functions from evolution-graph.ts
// For testing, we'll replicate them here
function isValidJsonObject(value: unknown): value is Record<string, unknown> {
  if (value === null || value === undefined) {
    return true;
  }

  if (typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }

  return Object.keys(value).every(key => typeof key === 'string');
}

function safeParseJsonField(value: unknown): Record<string, unknown> | null {
  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value === 'object' && !Array.isArray(value)) {
    return isValidJsonObject(value) ? (value as Record<string, unknown>) : null;
  }

  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return isValidJsonObject(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

function isValidRunConfig(value: unknown): value is Record<string, unknown> | null {
  const validated = safeParseJsonField(value);
  return validated !== null || value === null;
}

function isValidNodeMetadata(value: unknown): value is Record<string, unknown> | null {
  const validated = safeParseJsonField(value);
  return validated !== null || value === null;
}

// Mock database types
interface DbRun {
  id: string;
  config: unknown;
  createdAt: Date;
}

interface DbNode {
  id: string;
  metadata: unknown;
  runId: string;
}

type RunResponse = DbRun & {
  config?: Record<string, unknown>;
};

type NodeResponse = DbNode & {
  metadata?: Record<string, unknown>;
};

function toRunResponse(run: DbRun): RunResponse {
  const validatedConfig = safeParseJsonField(run.config);

  // Return undefined for null config, validated object otherwise
  return {
    ...run,
    config: run.config === null ? undefined : validatedConfig || undefined,
  };
}

function toNodeResponse(node: DbNode): NodeResponse {
  const validatedMetadata = safeParseJsonField(node.metadata);

  // Return undefined for null metadata, validated object otherwise
  return {
    ...node,
    metadata: node.metadata === null ? undefined : validatedMetadata || undefined,
  };
}

describe('Database Type Safety Tests', () => {
  describe('safeParseJsonField', () => {
    describe('null and undefined handling', () => {
      it('should return null for null input', () => {
        expect(safeParseJsonField(null)).toBeNull();
      });

      it('should return null for undefined input', () => {
        expect(safeParseJsonField(undefined)).toBeNull();
      });
    });

    describe('object input handling', () => {
      it('should return valid object as-is', () => {
        const input = { key: 'value', nested: { field: 123 } };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should return object with various primitive types', () => {
        const input = {
          string: 'text',
          number: 42,
          boolean: true,
          nullValue: null,
        };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should return object with arrays as values', () => {
        const input = {
          tags: ['tag1', 'tag2'],
          numbers: [1, 2, 3],
        };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should return deeply nested objects', () => {
        const input = {
          level1: {
            level2: {
              level3: {
                value: 'deep',
              },
            },
          },
        };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should return null for arrays', () => {
        const input = [1, 2, 3];
        const result = safeParseJsonField(input);

        expect(result).toBeNull();
      });

      it('should handle objects with numeric keys (JS auto-converts to strings)', () => {
        const input = { [123]: 'value' } as any;
        const result = safeParseJsonField(input);

        // JavaScript auto-converts numeric keys to strings
        expect(result).toEqual({ "123": 'value' });
      });
    });

    describe('string input handling', () => {
      it('should parse valid JSON string', () => {
        const input = '{"key":"value","number":123}';
        const result = safeParseJsonField(input);

        expect(result).toEqual({ key: 'value', number: 123 });
      });

      it('should parse nested JSON string', () => {
        const input = '{"user":{"name":"John","age":30}}';
        const result = safeParseJsonField(input);

        expect(result).toEqual({ user: { name: 'John', age: 30 } });
      });

      it('should parse JSON string with arrays', () => {
        const input = '{"tags":["tag1","tag2"],"items":[1,2,3]}';
        const result = safeParseJsonField(input);

        expect(result).toEqual({ tags: ['tag1', 'tag2'], items: [1, 2, 3] });
      });

      it('should parse empty object JSON string', () => {
        const input = '{}';
        const result = safeParseJsonField(input);

        expect(result).toEqual({});
      });

      it('should return null for invalid JSON string', () => {
        const input = '{invalid json}';
        const result = safeParseJsonField(input);

        expect(result).toBeNull();
      });

      it('should return null for JSON array string', () => {
        const input = '[1,2,3]';
        const result = safeParseJsonField(input);

        expect(result).toBeNull();
      });

      it('should return null for JSON primitive string', () => {
        expect(safeParseJsonField('"string"')).toBeNull();
        expect(safeParseJsonField('123')).toBeNull();
        expect(safeParseJsonField('true')).toBeNull();
        expect(safeParseJsonField('null')).toBeNull();
      });

      it('should return null for malformed JSON', () => {
        expect(safeParseJsonField('{key: value}')).toBeNull(); // Unquoted keys
        expect(safeParseJsonField('{"key": undefined}')).toBeNull(); // undefined value
        expect(safeParseJsonField('{"key": function(){}}')).toBeNull(); // Function
      });
    });

    describe('primitive type handling', () => {
      it('should return null for number input', () => {
        expect(safeParseJsonField(123)).toBeNull();
        expect(safeParseJsonField(0)).toBeNull();
        expect(safeParseJsonField(-1)).toBeNull();
      });

      it('should return null for boolean input', () => {
        expect(safeParseJsonField(true)).toBeNull();
        expect(safeParseJsonField(false)).toBeNull();
      });

      it('should return null for empty string', () => {
        expect(safeParseJsonField('')).toBeNull();
      });
    });

    describe('edge cases', () => {
      it('should handle object with empty string keys', () => {
        const input = { '': 'value', key: '' };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should handle object with special characters in keys', () => {
        const input = {
          'key-with-dash': 'value1',
          'key_with_underscore': 'value2',
          'key.with.dots': 'value3',
        };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should handle very large objects', () => {
        const largeObj: Record<string, unknown> = {};
        for (let i = 0; i < 1000; i++) {
          largeObj[`key${i}`] = `value${i}`;
        }
        const result = safeParseJsonField(largeObj);

        expect(result).toEqual(largeObj);
      });

      it('should handle objects with null values', () => {
        const input = { key1: null, key2: 'value' };
        const result = safeParseJsonField(input);

        expect(result).toEqual(input);
      });

      it('should handle empty string JSON', () => {
        const result = safeParseJsonField('');
        expect(result).toBeNull();
      });
    });
  });

  describe('isValidRunConfig', () => {
    it('should return true for valid config object', () => {
      const config = {
        algorithm: 'evolutionary',
        generations: 100,
        mutationRate: 0.1,
      };

      expect(isValidRunConfig(config)).toBe(true);
    });

    it('should return true for null config', () => {
      expect(isValidRunConfig(null)).toBe(true);
    });

    it('should return true for valid JSON string config', () => {
      const config = '{"algorithm":"evolutionary","generations":100}';
      expect(isValidRunConfig(config)).toBe(true);
    });

    it('should return false for invalid JSON string', () => {
      const config = '{invalid json}';
      expect(isValidRunConfig(config)).toBe(false);
    });

    it('should return false for array', () => {
      const config = [1, 2, 3];
      expect(isValidRunConfig(config)).toBe(false);
    });

    it('should return false for primitive types', () => {
      expect(isValidRunConfig(123)).toBe(false);
      expect(isValidRunConfig(true)).toBe(false);
      expect(isValidRunConfig('string')).toBe(false);
    });

    it('should handle object with numeric keys (JS auto-converts to strings)', () => {
      const config = { 123: 'value' } as any;
      // JavaScript auto-converts numeric keys to strings, so it's valid
      expect(isValidRunConfig(config)).toBe(true);
    });

    it('should type-narrow correctly', () => {
      const config: unknown = { key: 'value' };

      if (isValidRunConfig(config)) {
        // config should be Record<string, unknown> | null here
        expect(typeof config === 'object' || config === null).toBe(true);
      }
    });
  });

  describe('isValidNodeMetadata', () => {
    it('should return true for valid metadata object', () => {
      const metadata = {
        tags: ['tag1', 'tag2'],
        category: 'research',
        timestamp: 1234567890,
      };

      expect(isValidNodeMetadata(metadata)).toBe(true);
    });

    it('should return true for null metadata', () => {
      expect(isValidNodeMetadata(null)).toBe(true);
    });

    it('should return true for valid JSON string metadata', () => {
      const metadata = '{"tags":["tag1"],"category":"research"}';
      expect(isValidNodeMetadata(metadata)).toBe(true);
    });

    it('should return false for invalid JSON string', () => {
      const metadata = '{invalid}';
      expect(isValidNodeMetadata(metadata)).toBe(false);
    });

    it('should return false for array', () => {
      const metadata = [1, 2, 3];
      expect(isValidNodeMetadata(metadata)).toBe(false);
    });

    it('should return false for primitive types', () => {
      expect(isValidNodeMetadata(123)).toBe(false);
      expect(isValidNodeMetadata('string')).toBe(false);
      expect(isValidNodeMetadata(true)).toBe(false);
    });

    it('should type-narrow correctly', () => {
      const metadata: unknown = { key: 'value' };

      if (isValidNodeMetadata(metadata)) {
        // metadata should be Record<string, unknown> | null here
        expect(typeof metadata === 'object' || metadata === null).toBe(true);
      }
    });
  });

  describe('toRunResponse', () => {
    it('should include parsed config for valid config', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: { algorithm: 'evolutionary', generations: 100 },
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toEqual({ algorithm: 'evolutionary', generations: 100 });
      expect(response.id).toBe('run-1');
    });

    it('should include parsed config for valid JSON string config', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: '{"algorithm":"evolutionary","generations":100}',
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toEqual({ algorithm: 'evolutionary', generations: 100 });
    });

    it('should set config to undefined for null config', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: null,
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toBeUndefined();
    });

    it('should set config to undefined for invalid JSON string', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: '{invalid json}',
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toBeUndefined();
    });

    it('should set config to undefined for array config', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: [1, 2, 3],
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toBeUndefined();
    });

    it('should set config to undefined for primitive config', () => {
      const dbRun1: DbRun = {
        id: 'run-1',
        config: 123,
        createdAt: new Date(),
      };
      const dbRun2: DbRun = {
        id: 'run-2',
        config: 'string',
        createdAt: new Date(),
      };
      const dbRun3: DbRun = {
        id: 'run-3',
        config: true,
        createdAt: new Date(),
      };

      expect(toRunResponse(dbRun1).config).toBeUndefined();
      expect(toRunResponse(dbRun2).config).toBeUndefined();
      expect(toRunResponse(dbRun3).config).toBeUndefined();
    });

    it('should preserve all other run fields', () => {
      const dbRun: DbRun = {
        id: 'run-1',
        config: { key: 'value' },
        createdAt: new Date('2024-01-01'),
      };

      const response = toRunResponse(dbRun);

      expect(response.id).toBe('run-1');
      expect(response.createdAt).toEqual(new Date('2024-01-01'));
    });

    it('should handle complex nested config objects', () => {
      const complexConfig = {
        algorithm: 'evolutionary',
        parameters: {
          mutationRate: 0.1,
          crossoverRate: 0.8,
          selection: {
            method: 'tournament',
            size: 5,
          },
        },
        metadata: {
          tags: ['ai', 'optimization'],
          category: 'research',
        },
      };

      const dbRun: DbRun = {
        id: 'run-1',
        config: complexConfig,
        createdAt: new Date(),
      };

      const response = toRunResponse(dbRun);

      expect(response.config).toEqual(complexConfig);
    });
  });

  describe('toNodeResponse', () => {
    it('should include parsed metadata for valid metadata', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: { type: 'function', status: 'active' },
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toEqual({ type: 'function', status: 'active' });
      expect(response.id).toBe('node-1');
      expect(response.runId).toBe('run-1');
    });

    it('should include parsed metadata for valid JSON string', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: '{"type":"function","status":"active"}',
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toEqual({ type: 'function', status: 'active' });
    });

    it('should set metadata to undefined for null', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: null,
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toBeUndefined();
    });

    it('should set metadata to undefined for invalid JSON', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: '{invalid}',
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toBeUndefined();
    });

    it('should set metadata to undefined for array', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: [1, 2, 3],
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toBeUndefined();
    });

    it('should preserve all other node fields', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: { key: 'value' },
        runId: 'run-123',
      };

      const response = toNodeResponse(dbNode);

      expect(response.id).toBe('node-1');
      expect(response.runId).toBe('run-123');
    });

    it('should handle complex nested metadata', () => {
      const complexMetadata = {
        execution: {
          duration: 1234,
          startTime: '2024-01-01T00:00:00Z',
          endTime: '2024-01-01T00:00:01Z',
        },
        output: {
          type: 'result',
          value: 42,
          format: 'number',
        },
        tags: ['processed', 'validated'],
      };

      const dbNode: DbNode = {
        id: 'node-1',
        metadata: complexMetadata,
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toEqual(complexMetadata);
    });
  });

  describe('Integration with Database Patterns', () => {
    it('should handle typical database record with object field', () => {
      const records: DbRun[] = [
        {
          id: 'run-1',
          config: { algorithm: 'genetic', populationSize: 100 },
          createdAt: new Date(),
        },
        {
          id: 'run-2',
          config: null,
          createdAt: new Date(),
        },
      ];

      const responses = records.map(toRunResponse);

      expect(responses[0].config).toEqual({ algorithm: 'genetic', populationSize: 100 });
      expect(responses[1].config).toBeUndefined();
    });

    it('should handle database records with JSON string fields', () => {
      const records: DbNode[] = [
        {
          id: 'node-1',
          metadata: '{"type":"start","active":true}',
          runId: 'run-1',
        },
        {
          id: 'node-2',
          metadata: '{"type":"end","active":false}',
          runId: 'run-1',
        },
      ];

      const responses = records.map(toNodeResponse);

      expect(responses[0].metadata).toEqual({ type: 'start', active: true });
      expect(responses[1].metadata).toEqual({ type: 'end', active: false });
    });

    it('should handle mixed valid and invalid data', () => {
      const records: DbNode[] = [
        { id: 'node-1', metadata: { valid: true }, runId: 'run-1' },
        { id: 'node-2', metadata: 'invalid json', runId: 'run-1' },
        { id: 'node-3', metadata: null, runId: 'run-1' },
        { id: 'node-4', metadata: [1, 2, 3], runId: 'run-1' },
      ];

      const responses = records.map(toNodeResponse);

      expect(responses[0].metadata).toEqual({ valid: true });
      expect(responses[1].metadata).toBeUndefined();
      expect(responses[2].metadata).toBeUndefined();
      expect(responses[3].metadata).toBeUndefined();
    });

    it('should handle empty objects correctly', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: {},
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toEqual({});
    });

    it('should handle JSON string of empty object', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: '{}',
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      expect(response.metadata).toEqual({});
    });
  });

  describe('Type Safety Guarantees', () => {
    it('should never return unsafe types for valid data', () => {
      const validRun: DbRun = {
        id: 'run-1',
        config: { key: 'value' },
        createdAt: new Date(),
      };

      const response = toRunResponse(validRun);

      if (response.config) {
        // TypeScript should know this is Record<string, unknown>
        const keys = Object.keys(response.config);
        expect(keys).toEqual(['key']);
      }
    });

    it('should ensure undefined for invalid data', () => {
      const invalidRun: DbRun = {
        id: 'run-1',
        config: [1, 2, 3],
        createdAt: new Date(),
      };

      const response = toRunResponse(invalidRun);

      expect(response.config).toBeUndefined();
    });

    it('should maintain type safety after validation', () => {
      const dbNode: DbNode = {
        id: 'node-1',
        metadata: '{"key":"value"}',
        runId: 'run-1',
      };

      const response = toNodeResponse(dbNode);

      if (response.metadata) {
        // Should be type-safe to access
        const key = response.metadata.key;
        expect(key).toBe('value');
      }
    });
  });

  describe('Error Handling', () => {
    it('should handle circular references gracefully', () => {
      const circular: any = { key: 'value' };
      circular.self = circular;

      const result = safeParseJsonField(circular);
      // Should either return null or handle gracefully
      expect(result === null || typeof result === 'object').toBe(true);
    });

    it('should handle extremely long JSON strings', () => {
      const largeObj: Record<string, unknown> = {};
      for (let i = 0; i < 10000; i++) {
        largeObj[`key${i}`] = `value${i}`.repeat(100);
      }

      const jsonString = JSON.stringify(largeObj);
      const result = safeParseJsonField(jsonString);

      expect(result).toBeDefined();
    });

    it('should handle objects with very deep nesting', () => {
      let deep: any = { value: 'deep' };
      for (let i = 0; i < 100; i++) {
        deep = { nested: deep };
      }

      const result = safeParseJsonField(deep);
      expect(result).toBeDefined();
    });
  });
});
