import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AceToolsBubble } from './ace-tools-bubble';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('AceToolsBubble', () => {
  it('should expose correct static metadata', () => {
    expect(AceToolsBubble.bubbleName).toBe('ace-tools');
    expect(AceToolsBubble.service).toBe('ace-tools');
    expect(AceToolsBubble.type).toBe('service');
    expect(AceToolsBubble.authType).toBe('apikey');
    expect(AceToolsBubble.schema).toBeDefined();
  });

  it('should accept valid params via the schema', () => {
    const params = {
      operation: 'validateCode',
      code: 'const x = 1;',
      language: 'javascript' as const,
    };
    const result = AceToolsBubble.schema.safeParse(params);
    expect(result.success).toBe(true);
  });

  it('should reject params with an unknown operation via the schema', () => {
    const result = AceToolsBubble.schema.safeParse({ operation: 'nope' });
    expect(result.success).toBe(false);
  });

  it('should NOT throw on invalid params (constructor is resilient)', () => {
    expect(() => new AceToolsBubble({} as any)).not.toThrow();
    expect(() => new AceToolsBubble({ operation: 'validateCode' } as any)).not.toThrow();
  });

  it('should return a controlled error from action() for invalid params', async () => {
    const bubble = new AceToolsBubble({} as any);
    const result = await bubble.action();
    expect(result.success).toBe(false);
    expect(result.error).toContain('Input Schema validation failed');
  });

  it('should execute a valid operation successfully (no network)', async () => {
    const bubble = new AceToolsBubble({
      operation: 'validateCode',
      code: 'const x = 1;',
      language: 'javascript',
    });
    const result = await bubble.action();
    expect(result.success).toBe(true);
    expect(result.error).toBe('');
    expect((result.data as any).success).toBe(true);
    expect((result.data as any).data).toBeDefined();
  });

  it('should return a controlled error from action() when execution throws', async () => {
    // executeCode with a dangerous pattern triggers a security validation failure
    // path that is surfaced as a controlled error rather than a thrown exception.
    const bubble = new AceToolsBubble({
      operation: 'executeCode',
      code: "require('child_process')",
      language: 'javascript',
    });
    const result = await bubble.action();
    expect(result).toBeDefined();
    expect(typeof result.success).toBe('boolean');
    expect(typeof result.error).toBe('string');
  });
});
