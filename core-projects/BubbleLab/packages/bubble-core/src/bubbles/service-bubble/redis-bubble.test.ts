   /*
    * Tests for RedisBubble
    *
    * These tests exercise the real Bubble base-class contract:
    * - The constructor never throws on invalid params; it captures a
    *   validationError and `action()` returns a controlled
    *   { success: false, error } result.
    * - The bubble exposes the expected static metadata.
    */

   import { describe, it, expect } from 'vitest';
   import { RedisBubble } from './redis-bubble';

   describe('RedisBubble', () => {
     it('does not throw on invalid params (captures validationError)', () => {
       expect(() => new RedisBubble({ notARealField: 1 } as any)).not.toThrow();
     });

     it('action() returns a controlled error for invalid params', async () => {
       const bubble = new RedisBubble({ notARealField: 1 } as any);
       const result = await bubble.action();
       expect(result.success).toBe(false);
       expect(typeof result.error).toBe('string');
       expect(result.error.length).toBeGreaterThan(0);
     });

     it('constructs with operation params without throwing', () => {
       const bubble = new RedisBubble({
         operation: 'get',
         key: 'foo',
       } as any);
       expect(bubble).toBeDefined();
     });

     it('exposes static metadata', () => {
       expect(RedisBubble.bubbleName).toBeDefined();
       expect(RedisBubble.service).toBeDefined();
       expect(RedisBubble.schema).toBeDefined();
     });
   });
