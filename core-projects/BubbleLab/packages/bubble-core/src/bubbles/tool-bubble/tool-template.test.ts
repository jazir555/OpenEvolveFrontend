/**
 * Tool Template - unit tests
 * Rewritten to exercise the real MyCustomTool API (performAction).
 */
import { describe, it, expect } from 'vitest';
import { MyCustomTool } from './tool-template';

describe('MyCustomTool (tool-template)', () => {
  it('constructs with the required inputData parameter', () => {
    const tool = new MyCustomTool({ inputData: 'hello' });
    expect(tool).toBeDefined();
    expect(tool.params.inputData).toBe('hello');
  });

  it('processes data via performAction', async () => {
    const tool = new MyCustomTool({ inputData: 'my data' });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.processedData).toContain('Processed: my data');
    expect(result.metadata.itemsProcessed).toBe(1);
    expect(result.error).toBe('');
  });

  it('honours the options parameter', async () => {
    const tool = new MyCustomTool({
      inputData: 'x',
      options: { includeDetails: true, maxResults: 5 },
    });
    const result = await tool.performAction();

    expect(result.success).toBe(true);
    expect(result.processedData).toContain('details: true');
    expect(result.processedData).toContain('max: 5');
  });
});
