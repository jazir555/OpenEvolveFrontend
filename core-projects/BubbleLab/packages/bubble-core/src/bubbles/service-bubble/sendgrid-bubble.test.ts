   /*
    * Tests for SendGridBubble
    *
    * These tests exercise the real Bubble base-class contract:
    * - The constructor never throws on invalid params; it captures a
    *   validationError and `action()` returns a controlled
    *   { success: false, error } result.
    * - The bubble exposes the expected static metadata.
    */

   import { describe, it, expect } from 'vitest';
   import { SendGridBubble } from './sendgrid-bubble';

   describe('SendGridBubble', () => {
     it('does not throw on invalid params (captures validationError)', () => {
       expect(() => new SendGridBubble({ notARealField: 1 } as any)).not.toThrow();
     });

     it('action() returns a controlled error for invalid params', async () => {
       const bubble = new SendGridBubble({ notARealField: 1 } as any);
       const result = await bubble.action();
       expect(result.success).toBe(false);
       expect(typeof result.error).toBe('string');
       expect(result.error.length).toBeGreaterThan(0);
     });

     it('constructs with operation params without throwing', () => {
       const bubble = new SendGridBubble({
         operation: 'sendEmail',
         to: 'test@example.com',
         from: 'noreply@example.com',
         subject: 'Hi',
         text: 'Hello',
       } as any);
       expect(bubble).toBeDefined();
     });

     it('exposes static metadata', () => {
       expect(SendGridBubble.bubbleName).toBeDefined();
       expect(SendGridBubble.service).toBeDefined();
       expect(SendGridBubble.schema).toBeDefined();
     });
   });
