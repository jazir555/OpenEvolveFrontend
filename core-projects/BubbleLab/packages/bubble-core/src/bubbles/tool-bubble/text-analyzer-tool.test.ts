/**
 * Text Analyzer Tool - unit tests
 * Rewritten to exercise the real TextAnalyzerTool API (performAction).
 */
import { describe, it, expect } from 'vitest';
import { TextAnalyzerTool } from './text-analyzer-tool';

describe('TextAnalyzerTool', () => {
  it('constructs with the required text parameter', () => {
    const tool = new TextAnalyzerTool({ text: 'hello world' });
    expect(tool).toBeDefined();
    expect(tool.params.text).toBe('hello world');
  });

  it('runs sentiment, statistics and keyword analysis', async () => {
    const tool = new TextAnalyzerTool({
      text: 'I love this good product. It is wonderful and amazing!',
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.statistics).toBeDefined();
    expect(result.statistics!.wordCount).toBeGreaterThan(0);
    expect(result.sentiment).toBeDefined();
    expect(result.sentiment!.label).toBe('positive');
    expect(result.keywords).toBeDefined();
  });

  it('handles empty text gracefully', async () => {
    const tool = new TextAnalyzerTool({ text: '    ' });
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Text is required');
  });

  it('supports readability and frequency operations', async () => {
    const tool = new TextAnalyzerTool({
      text: 'The cat sat on the mat. The dog ran fast. A bird flew high.',
      operations: ['readability', 'frequency', 'statistics'],
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.readability).toBeDefined();
    expect(result.frequency).toBeDefined();
  });
});
