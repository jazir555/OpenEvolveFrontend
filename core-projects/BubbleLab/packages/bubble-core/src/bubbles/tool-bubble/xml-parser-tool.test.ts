/**
 * XML Parser Tool - unit tests
 * Rewritten to exercise the real XMLParserTool API (performAction).
 * The optional `xml2js` dependency is mocked so no native install is needed.
 */
import { describe, it, expect, vi } from 'vitest';

vi.mock('xml2js', () => {
  class Parser {
    parseStringPromise(_xml: string) {
      return Promise.resolve({ root: { item: 'hello' } });
    }
  }
  class Builder {
    buildObject(_obj: unknown) {
      return '<?xml version="1.0"?><root></root>';
    }
  }
  return { Parser, Builder, default: { Parser, Builder } };
});

import { XMLParserTool } from './xml-parser-tool';

describe('XMLParserTool', () => {
  it('validates well-formed xml', async () => {
    const tool = new XMLParserTool({
      operation: 'validate',
      xmlData: '<root><item>1</item></root>',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.isValid).toBe(true);
  });

  it('detects malformed xml', async () => {
    const tool = new XMLParserTool({
      operation: 'validate',
      xmlData: '<root><item></root>',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.isValid).toBe(false);
  });

  it('parses xml to an object', async () => {
    const tool = new XMLParserTool({
      operation: 'parse',
      xmlData: '<root><item>hello</item></root>',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
    expect(result.stats.nodeCount).toBeGreaterThan(0);
  });

  it('generates xml from a json string', async () => {
    const tool = new XMLParserTool({
      operation: 'generate',
      xmlData: '{"root":{"item":"x"}}',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.xml).toContain('<root>');
  });

  it('queries xml by path', async () => {
    const tool = new XMLParserTool({
      operation: 'query',
      xmlData: '<root><item>hello</item></root>',
      queryPath: 'root.item',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.queryResults).toBeDefined();
  });

  it('requires xmlData for the parse operation', async () => {
    const tool = new XMLParserTool({ operation: 'parse', xmlData: '' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
  });
});
