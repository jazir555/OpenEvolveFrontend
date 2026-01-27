import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * CodeEditTool - Secure code editing with command injection prevention
 *
 * Security Features:
 * - Blocks dangerous JavaScript functions (eval, Function, require, etc.)
 * - Blocks child_process and filesystem access
 * - Enforces size limits to prevent DoS
 * - Detects obfuscated and encoded injection attempts
 * - Prevents prototype pollution attacks
 */
export class CodeEditTool extends ToolBubble<CodeEditParams, CodeEditResult> {
  bubbleName = 'code-edit';
  type = 'tool';
  alias = 'code-edit';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  // Size limits (in bytes)
  private readonly MAX_INITIAL_CODE_SIZE = 500 * 1024; // 500KB
  private readonly MAX_EDIT_SIZE = 200 * 1024; // 200KB
  private readonly MAX_INSTRUCTIONS_SIZE = 10 * 1024; // 10KB

  // Dangerous patterns that MUST be blocked
  private readonly DANGEROUS_PATTERNS = [
    // Code execution
    /\beval\s*\(/,
    /\bFunction\s*\(/,
    /\bnew\s+Function\s*\(/,

    // Module imports
    /\brequire\s*\(/,
    /\bimport\s*\(/,
    /\bimport\s+/,

    // Child process
    /child_process/,
    /\.exec\s*\(/,
    /\.execSync\s*\(/,
    /\.spawn\s*\(/,

    // File system
    /fs\./,
    /require\s*\(\s*['"]fs['"]/,

    // Process access
    /\bprocess\./,
    /\bprocess\s*\[/,

    // Prototype pollution
    /__proto__/,
    /__defineGetter__/,
    /__defineSetter__/,
    /\.constructor\s*\[/,
    /\.constructor\s*=/,
  ];

  // Encoded/obfuscated injection patterns
  private readonly OBFUSCATION_PATTERNS = [
    // String concatenation to hide keywords
    /['"]\w*['"]\s*\+\s*['"]\w*['"]\s*\+\s*['"]\w*['"]/,
    /['"]\w*['"]\s*\+\s*\w+\s*\+\s*['"]\w*['"]/,

    // Unicode escapes
    /\\u[0-9a-fA-F]{4}/,
    /\\x[0-9a-fA-F]{2}/,

    // Base64-like strings (potential encoded payloads)
    /['"][A-Za-z0-9+/]{50,}={0,2}['"]/,

    // Char code obfuscation
    /String\.fromCharCode/,
    /String\.prototype\.charCodeAt/,
  ];

  async execute(input: any): Promise<CodeEditResult> {
    try {
      // Validate input structure
      const validation = this.validateInput(input);
      if (!validation.valid) {
        return {
          success: false,
          error: `Validation failed: ${validation.error}`,
          errorType: 'validation'
        };
      }

      // Check size limits
      const sizeCheck = this.checkSizeLimits(input);
      if (!sizeCheck.valid) {
        return {
          success: false,
          error: sizeCheck.error,
          errorType: 'size_limit'
        };
      }

      // Security scan for command injection
      const securityCheck = this.scanForSecurityThreats(input);
      if (!securityCheck.safe) {
        return {
          success: false,
          error: `Security threat detected: ${securityCheck.reason}`,
          errorType: 'security_violation',
          blockedPattern: securityCheck.pattern
        };
      }

      // Perform the code edit
      const result = await this.performEdit(input);

      return {
        success: true,
        editedCode: result.editedCode,
        changes: result.changes,
        stats: result.stats
      };

    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        errorType: 'execution_error'
      };
    }
  }

  /**
   * Validate input structure
   */
  private validateInput(input: any): { valid: boolean; error?: string } {
    if (!input) {
      return { valid: false, error: 'Input is required' };
    }

    if (typeof input.initialCode !== 'string') {
      return { valid: false, error: 'initialCode must be a string' };
    }

    if (!Array.isArray(input.edits)) {
      return { valid: false, error: 'edits must be an array' };
    }

    if (typeof input.instructions !== 'string') {
      return { valid: false, error: 'instructions must be a string' };
    }

    // Validate each edit operation
    for (let i = 0; i < input.edits.length; i++) {
      const edit = input.edits[i];
      if (!edit.oldText || typeof edit.oldText !== 'string') {
        return { valid: false, error: `Edit ${i}: oldText is required` };
      }
      if (!edit.newText || typeof edit.newText !== 'string') {
        return { valid: false, error: `Edit ${i}: newText is required` };
      }
    }

    return { valid: true };
  }

  /**
   * Check size limits to prevent DoS attacks
   */
  private checkSizeLimits(input: any): { valid: boolean; error?: string } {
    const initialCodeSize = Buffer.byteLength(input.initialCode, 'utf8');
    if (initialCodeSize > this.MAX_INITIAL_CODE_SIZE) {
      return {
        valid: false,
        error: `initialCode exceeds ${this.MAX_INITIAL_CODE_SIZE} bytes limit (${initialCodeSize} bytes)`
      };
    }

    for (let i = 0; i < input.edits.length; i++) {
      const editSize = Buffer.byteLength(input.edits[i].newText, 'utf8');
      if (editSize > this.MAX_EDIT_SIZE) {
        return {
          valid: false,
          error: `Edit ${i} newText exceeds ${this.MAX_EDIT_SIZE} bytes limit (${editSize} bytes)`
        };
      }
    }

    const instructionsSize = Buffer.byteLength(input.instructions, 'utf8');
    if (instructionsSize > this.MAX_INSTRUCTIONS_SIZE) {
      return {
        valid: false,
        error: `instructions exceeds ${this.MAX_INSTRUCTIONS_SIZE} bytes limit (${instructionsSize} bytes)`
      };
    }

    return { valid: true };
  }

  /**
   * Scan for security threats including:
   * - Command injection patterns
   * - Obfuscated code
   * - Encoded payloads
   * - Multi-stage attacks
   */
  private scanForSecurityThreats(input: any): { safe: boolean; reason?: string; pattern?: string } {
    // Combine all input for scanning
    const allContent = [
      input.initialCode,
      ...input.edits.map((e: any) => e.newText),
      input.instructions
    ].join('\n');

    // Check dangerous patterns
    for (const pattern of this.DANGEROUS_PATTERNS) {
      if (pattern.test(allContent)) {
        const match = allContent.match(pattern);
        return {
          safe: false,
          reason: 'Dangerous function or pattern detected',
          pattern: match ? match[0] : pattern.toString()
        };
      }
    }

    // Check for obfuscation
    for (const pattern of this.OBFUSCATION_PATTERNS) {
      if (pattern.test(allContent)) {
        const match = allContent.match(pattern);
        return {
          safe: false,
          reason: 'Obfuscated or encoded code detected',
          pattern: match ? match[0] : pattern.toString()
        };
      }
    }

    // Check for multi-stage attacks (multiple suspicious but individually safe patterns)
    constSuspiciousCount = 0;
    if (allContent.includes('atob') || allContent.includes('btoa')) suspiciousCount++;
    if (allContent.includes('setTimeout') || allContent.includes('setInterval')) suspiciousCount++;
    if (allContent.includes('Promise.all') || allContent.includes('async')) suspiciousCount++;
    if (allContent.includes('window.') || allContent.includes('global.')) suspiciousCount++;

    if (suspiciousCount >= 3) {
      return {
        safe: false,
        reason: 'Multi-stage attack pattern detected',
        pattern: 'Multiple suspicious patterns combined'
      };
    }

    return { safe: true };
  }

  /**
   * Perform the actual code edit
   */
  private async performEdit(input: any): Promise<{
    editedCode: string;
    changes: number;
    stats: any;
  }> {
    let editedCode = input.initialCode;
    let changes = 0;

    for (const edit of input.edits) {
      const index = editedCode.indexOf(edit.oldText);
      if (index !== -1) {
        editedCode =
          editedCode.substring(0, index) +
          edit.newText +
          editedCode.substring(index + edit.oldText.length);
        changes++;
      }
    }

    return {
      editedCode,
      changes,
      stats: {
        originalLength: input.initialCode.length,
        editedLength: editedCode.length,
        editsApplied: changes,
        editsAttempted: input.edits.length
      }
    };
  }
}

export interface CodeEditParams {
  timeout?: number;
}

export interface CodeEditResult {
  success: boolean;
  editedCode?: string;
  changes?: number;
  stats?: {
    originalLength: number;
    editedLength: number;
    editsApplied: number;
    editsAttempted: number;
  };
  error?: string;
  errorType?: 'validation' | 'size_limit' | 'security_violation' | 'execution_error';
  blockedPattern?: string;
}
