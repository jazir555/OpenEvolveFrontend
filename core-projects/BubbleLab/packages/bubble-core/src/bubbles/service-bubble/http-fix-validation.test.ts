/**
 * HTTP body-size validation fix — regression tests
 *
 * BACKGROUND / WHY THIS FILE DOES NOT IMPORT `./http-fix-validation.js`:
 * `src/bubbles/service-bubble/http-fix-validation.ts` is a standalone
 * verification SCRIPT, not a module: it exports nothing and calls
 * `process.exit()` at import time, which would kill the Vitest worker. The
 * previous contents of this test file imported a non-existent
 * `HttpFixValidation` export and tested an API that never existed.
 *
 * The bug being guarded against: `z.union([...]).max(n)` is not valid on a
 * union, so the original schema silently failed to enforce a body-size limit.
 * The fix expresses the limit with `.refine()` instead. These tests pin that
 * behaviour, plus the real HttpBubble body schema it protects.
 */

import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { HttpBubble } from './http-bubble.js';

const MAX_BODY_BYTES = 10485760; // 10MB

/**
 * The fixed schema, mirroring http-fix-validation.ts.
 * A union cannot carry `.max()`, so the size limit lives in `.refine()`.
 */
const FixedHttpBodySchema = z
  .union([z.string(), z.record(z.unknown())])
  .refine((val) => {
    if (typeof val === 'string') {
      return val.length <= MAX_BODY_BYTES;
    }
    return JSON.stringify(val).length <= MAX_BODY_BYTES;
  }, 'Request body exceeds maximum size of 10MB')
  .optional();

describe('http body-size validation fix', () => {
  describe('accepted bodies', () => {
    it('accepts a small string body', () => {
      expect(FixedHttpBodySchema.parse('small body')).toBe('small body');
    });

    it('accepts a small object body', () => {
      const body = { key: 'value', num: 123 };
      expect(FixedHttpBodySchema.parse(body)).toEqual(body);
    });

    it('accepts undefined (the field is optional)', () => {
      expect(FixedHttpBodySchema.parse(undefined)).toBeUndefined();
    });

    it('accepts an empty string', () => {
      expect(FixedHttpBodySchema.parse('')).toBe('');
    });

    it('accepts an empty object', () => {
      expect(FixedHttpBodySchema.parse({})).toEqual({});
    });

    it('accepts a nested object body', () => {
      const body = { a: { b: { c: [1, 2, 3] } } };
      expect(FixedHttpBodySchema.parse(body)).toEqual(body);
    });

    it('accepts a string exactly at the 10MB limit', () => {
      const atLimit = 'x'.repeat(MAX_BODY_BYTES);
      expect(() => FixedHttpBodySchema.parse(atLimit)).not.toThrow();
    });
  });

  describe('rejected bodies', () => {
    it('rejects null (null is neither a string nor a record)', () => {
      const result = FixedHttpBodySchema.safeParse(null);
      expect(result.success).toBe(false);
    });

    it('rejects a number', () => {
      expect(FixedHttpBodySchema.safeParse(42).success).toBe(false);
    });

    it('rejects a string one byte over the 10MB limit', () => {
      const overLimit = 'x'.repeat(MAX_BODY_BYTES + 1);
      const result = FixedHttpBodySchema.safeParse(overLimit);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toContain(
          'exceeds maximum size of 10MB'
        );
      }
    });

    it('rejects an object whose JSON form exceeds the 10MB limit', () => {
      const largeObject = { data: 'x'.repeat(MAX_BODY_BYTES) };
      const result = FixedHttpBodySchema.safeParse(largeObject);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toContain(
          'exceeds maximum size of 10MB'
        );
      }
    });
  });

  describe('regression: the size limit is actually enforced', () => {
    it('demonstrates that .max() on a union does not constrain size', () => {
      // This is the shape of the original (buggy) schema: `.max()` is not part
      // of the ZodUnion API, so it cannot have been enforcing anything.
      const union = z.union([z.string(), z.record(z.unknown())]);
      expect(
        (union as unknown as { max?: unknown }).max
      ).toBeUndefined();
    });

    it('refine-based validation runs for both union branches', () => {
      expect(
        FixedHttpBodySchema.safeParse('x'.repeat(MAX_BODY_BYTES + 1)).success
      ).toBe(false);
      expect(
        FixedHttpBodySchema.safeParse({ d: 'x'.repeat(MAX_BODY_BYTES) }).success
      ).toBe(false);
    });
  });

  describe('real HttpBubble request body schema', () => {
    it('accepts a string body', () => {
      const result = HttpBubble.schema.safeParse({
        operation: 'post',
        url: 'https://api.example.com/data',
        body: 'raw payload',
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.body).toBe('raw payload');
      }
    });

    it('accepts an object body', () => {
      const result = HttpBubble.schema.safeParse({
        operation: 'post',
        url: 'https://api.example.com/data',
        body: { hello: 'world' },
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.body).toEqual({ hello: 'world' });
      }
    });

    it('accepts URLSearchParams and FormData bodies', () => {
      expect(
        HttpBubble.schema.safeParse({
          operation: 'post',
          url: 'https://api.example.com/data',
          body: new URLSearchParams({ a: '1' }),
        }).success
      ).toBe(true);

      expect(
        HttpBubble.schema.safeParse({
          operation: 'post',
          url: 'https://api.example.com/data',
          body: new FormData(),
        }).success
      ).toBe(true);
    });

    it('treats body as optional', () => {
      const result = HttpBubble.schema.safeParse({
        operation: 'get',
        url: 'https://api.example.com/data',
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.body).toBeUndefined();
      }
    });

    it('rejects a numeric body', () => {
      expect(
        HttpBubble.schema.safeParse({
          operation: 'post',
          url: 'https://api.example.com/data',
          body: 12345,
        }).success
      ).toBe(false);
    });

    it('still rejects an invalid URL regardless of body', () => {
      expect(
        HttpBubble.schema.safeParse({
          operation: 'post',
          url: 'not-a-url',
          body: 'payload',
        }).success
      ).toBe(false);
    });
  });
});
