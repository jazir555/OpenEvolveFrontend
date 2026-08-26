/**
 * URL Validator Tool - unit tests
 * Rewritten to exercise the real URLValidatorTool API (performAction).
 */
import { describe, it, expect } from 'vitest';
import { URLValidatorTool } from './url-validator-tool';

describe('URLValidatorTool', () => {
  it('constructs with a url', () => {
    const tool = new URLValidatorTool({ url: 'https://example.com' });
    expect(tool).toBeDefined();
  });

  it('validates a well-formed https url', async () => {
    const tool = new URLValidatorTool({ url: 'https://example.com/path?q=1' });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].isValid).toBe(true);
    expect(result.results[0].protocol).toBe('https');
    expect(result.stats.totalURLs).toBe(1);
  });

  it('flags invalid url syntax', async () => {
    const tool = new URLValidatorTool({ url: 'not a url' });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.results[0].isValid).toBe(false);
  });

  it('detects suspicious url shorteners', async () => {
    const tool = new URLValidatorTool({ url: 'https://bit.ly/abc' });
    const result = await tool.performAction();

    expect(result.results[0].isSuspicious).toBe(true);
  });

  it('requires url or urls', async () => {
    const tool = new URLValidatorTool({});
    const result = await tool.performAction();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Either url or urls');
  });

  it('validates a batch of urls', async () => {
    const tool = new URLValidatorTool({
      urls: ['https://example.com', 'https://another.com'],
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.results).toHaveLength(2);
    expect(result.stats.validURLs).toBe(2);
  });
});
