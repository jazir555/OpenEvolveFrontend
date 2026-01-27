import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * EmailValidatorTool - Email validation operations
 */
export class EmailValidatorTool extends ToolBubble<EmailValidatorParams, EmailValidatorResult> {
  bubbleName = 'email-validator';
  type = 'tool';
  alias = 'email-validator';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<EmailValidatorResult> {
    try {
      const result = await this.validate(input);
      return { success: true, valid: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validate(params: { email: string }): Promise<EmailValidatorResult> {
    try {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      const isValid = emailRegex.test(params.email);
      return { success: true, valid: isValid, email: params.email };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async format(params: { email: string }): Promise<EmailValidatorResult> {
    try {
      const formatted = params.email.toLowerCase().trim();
      return { success: true, formatted };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async check(params: { email: string; mx?: boolean }): Promise<EmailValidatorResult> {
    try {
      // Placeholder for MX record check
      const hasMX = params.mx ? true : undefined;
      return { success: true, deliverable: true, mx: hasMX };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface EmailValidatorParams {
  timeout?: number;
}

export interface EmailValidatorResult {
  success: boolean;
  valid?: boolean;
  email?: string;
  formatted?: string;
  deliverable?: boolean;
  mx?: boolean;
  error?: string;
}
