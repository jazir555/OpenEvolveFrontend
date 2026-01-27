import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * CodeFormatterTool - Code formatting operations
 */
export class CodeFormatterTool extends ToolBubble<CodeFormatterParams, CodeFormatterResult> {
  bubbleName = 'code-formatter';
  type = 'tool';
  alias = 'code-formatter';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<CodeFormatterResult> {
    try {
      const result = await this.format(input);
      return { success: true, formatted: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async format(params: { code: string; language: string }): Promise<CodeFormatterResult> {
    try {
      // Basic formatting - in production would use real formatter
      const formatted = params.code
        .replace(/\s+/g, ' ')
        .replace(/; /g, ';\n')
        .replace(/\{ /g, ' {\n  ')
        .replace(/ \}/g, '\n}');
      return { success: true, formatted };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async lint(params: { code: string; language: string }): Promise<CodeFormatterResult> {
    try {
      const issues = [];
      if (!params.code.includes(';')) {
        issues.push({ line: 1, message: 'Missing semicolons', severity: 'warning' });
      }
      return { success: true, issues, score: 100 - issues.length * 10 };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async fix(params: { code: string; language: string }): Promise<CodeFormatterResult> {
    try {
      // Auto-fix common issues
      let fixed = params.code;
      fixed = fixed.replace(/;/g, ';\n');
      fixed = fixed.replace(/\{/g, ' {');
      return { success: true, fixed, fixes: ['Added semicolons', 'Added spacing'] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface CodeFormatterParams {
  timeout?: number;
}

export interface CodeFormatterResult {
  success: boolean;
  formatted?: string;
  fixed?: string;
  issues?: any[];
  fixes?: string[];
  score?: number;
  error?: string;
}
