/*
 * Test Suite for email-validator-tool
 * Covers the real EmailValidatorTool API (constructor + action()).
 */

import { describe, it, expect } from 'vitest';
import { EmailValidatorTool } from './email-validator-tool';

describe('email-validator-tool', () => {
  describe('Construction', () => {
    it('should construct with valid params (no throw)', () => {
      const tool = new EmailValidatorTool({ email: 'test@example.com' });
      expect(tool).toBeDefined();
      expect(tool.name).toBe('email-validator-tool');
    });

    it('should not throw on invalid params (captures validation error)', () => {
      const tool = new EmailValidatorTool({});
      expect(tool).toBeDefined();
    });
  });

  describe('action()', () => {
    it('should validate a correct email', async () => {
      const tool = new EmailValidatorTool({ email: 'test@gmail.com' });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.results).toHaveLength(1);
      expect(result.data?.results[0].isValid).toBe(true);
      expect(result.data?.results[0].domain).toBe('gmail.com');
      expect(result.data?.stats.totalEmails).toBe(1);
    });

    it('should reject an invalid email syntax', async () => {
      const tool = new EmailValidatorTool({ email: 'notanemail' });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.results[0].isValid).toBe(false);
      expect(result.data?.results[0].syntaxValid).toBe(false);
    });

    it('should detect disposable email domains', async () => {
      const tool = new EmailValidatorTool({
        email: 'spam@mailinator.com',
        checkDisposable: true,
      });
      const result = await tool.action();
      expect(result.success).toBe(true);
      expect(result.data?.results[0].isDisposable).toBe(true);
      expect(result.data?.stats.disposableEmails).toBe(1);
    });

    it('should return a controlled error when no email is provided', async () => {
      const tool = new EmailValidatorTool({});
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('email');
    });

    it('should return a controlled error for invalid params schema', async () => {
      const tool = new EmailValidatorTool({
        // @ts-expect-error intentionally wrong type
        emails: 'not-an-array',
      });
      const result = await tool.action();
      expect(result.success).toBe(false);
      expect(result.error).toContain('validation failed');
    });
  });
});
