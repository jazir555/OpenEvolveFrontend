/**
 * CodeEditTool Security Tests
 *
 * Comprehensive security test suite covering:
 * - Command injection prevention (9 tests)
 * - Size limit enforcement (3 tests)
 * - Malicious pattern detection (3 tests)
 *
 * Total: 15 security test cases
 */

import { describe, test, expect } from 'vitest';
import { CodeEditTool } from './code-edit-tool';

describe('CodeEditTool - Security Tests', () => {
  let tool: CodeEditTool;

  beforeEach(() => {
    tool = new CodeEditTool();
  });

  describe('Command Injection Prevention', () => {

    /**
     * TEST 1: Block eval() usage
     *
     * Threat: Direct code execution via eval()
     * Impact: Remote code execution
     */
    test('should block eval usage', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const x = eval("malicious code");'
        }],
        instructions: 'Add eval'
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Security threat detected');
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('eval');
    });

    /**
     * TEST 2: Block Function() constructor
     *
     * Threat: Dynamic function creation for code execution
     * Impact: Remote code execution
     */
    test('should block Function constructor', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const x = new Function("return malicious")();'
        }],
        instructions: 'Use Function constructor'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('Function');
    });

    /**
     * TEST 3: Block require() usage
     *
     * Threat: Module injection for accessing dangerous modules
     * Impact: Access to fs, child_process, etc.
     */
    test('should block require usage', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const fs = require("fs");'
        }],
        instructions: 'Import fs module'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('require');
    });

    /**
     * TEST 4: Block child_process.exec()
     *
     * Threat: Direct command execution
     * Impact: System command injection
     */
    test('should block child_process.exec', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const cp = require("child_process"); cp.exec("rm -rf /");'
        }],
        instructions: 'Execute command'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('child_process');
    });

    /**
     * TEST 5: Block spawn() with dangerous arguments
     *
     * Threat: Process spawning for command execution
     * Impact: System command injection
     */
    test('should block spawn usage', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const { spawn } = require("child_process"); spawn("bash", ["-c", "evil"]);'
        }],
        instructions: 'Spawn process'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('spawn');
    });

    /**
     * TEST 6: Block execSync()
     *
     * Threat: Synchronous command execution
     * Impact: System command injection, DoS
     */
    test('should block execSync usage', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const cp = require("child_process"); cp.execSync("malicious");'
        }],
        instructions: 'Sync exec'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('execSync');
    });

    /**
     * TEST 7: Block dynamic imports
     *
     * Threat: Dynamic module loading
     * Impact: Access to dangerous modules
     */
    test('should block dynamic imports', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const mod = await import("fs");'
        }],
        instructions: 'Dynamic import'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('import');
    });

    /**
     * TEST 8: Block prototype pollution
     *
     * Threat: Modify Object.prototype to affect all objects
     * Impact: Application-wide compromise
     */
    test('should block prototype pollution attempts', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const obj = {}; obj.__proto__.malicious = true;'
        }],
        instructions: 'Prototype pollution'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('__proto__');
    });

    /**
     * TEST 9: Block process access
     *
     * Threat: Access Node.js process object
     * Impact: Environment variable leakage, process manipulation
     */
    test('should block process access', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const env = process.env;'
        }],
        instructions: 'Access process'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toContain('process');
    });
  });

  describe('Size Limit Enforcement', () => {

    /**
     * TEST 10: Enforce 500KB limit for initial code
     *
     * Threat: DoS via oversized inputs
     * Impact: Memory exhaustion, service unavailability
     */
    test('should enforce 500KB limit for initial code', async () => {
      // Create code larger than 500KB
      const largeCode = 'const x = "' + 'a'.repeat(600 * 1024) + '";';

      const result = await tool.execute({
        initialCode: largeCode,
        edits: [],
        instructions: 'Test large code'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('size_limit');
      expect(result.error).toContain('initialCode exceeds');
      expect(result.error).toContain('500');
    });

    /**
     * TEST 11: Enforce 200KB limit for code edits
     *
     * Threat: DoS via large edits
     * Impact: Memory exhaustion during edit application
     */
    test('should enforce 200KB limit for code edits', async () => {
      const largeEdit = 'const x = "' + 'a'.repeat(250 * 1024) + '";';

      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: largeEdit
        }],
        instructions: 'Large edit'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('size_limit');
      expect(result.error).toContain('exceeds');
      expect(result.error).toContain('200');
    });

    /**
     * TEST 12: Enforce 10KB limit for instructions
     *
     * Threat: DoS via massive instructions
     * Impact: Processing overhead, memory exhaustion
     */
    test('should enforce 10KB limit for instructions', async () => {
      const largeInstructions = 'A'.repeat(15 * 1024);

      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [],
        instructions: largeInstructions
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('size_limit');
      expect(result.error).toContain('instructions exceeds');
      expect(result.error).toContain('10');
    });
  });

  describe('Malicious Pattern Detection', () => {

    /**
     * TEST 13: Block obfuscated injection attempts
     *
     * Threat: String concatenation to hide malicious keywords
     * Impact: Bypass static analysis, execute arbitrary code
     */
    test('should block obfuscated injection attempts', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const ev = "e" + "v" + "al"; const x = ev("code");'
        }],
        instructions: 'Obfuscated eval'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.error).toContain('Obfuscated');
      expect(result.blockedPattern).toBeDefined();
    });

    /**
     * TEST 14: Block encoded payloads
     *
     * Threat: Unicode/base64 encoding to hide malicious code
     * Impact: Bypass pattern matching, execute arbitrary code
     */
    test('should block encoded payloads', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const x = "\\u0065\\u0076\\u0061\\u006c";' // Unicode escape for "eval"
        }],
        instructions: 'Unicode payload'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.error).toContain('Obfuscated');
      expect(result.blockedPattern).toBeDefined();
    });

    /**
     * TEST 15: Block multi-stage attacks
     *
     * Threat: Combining multiple suspicious but safe patterns
     * Impact: Sophisticated attacks bypassing single-pattern detection
     */
    test('should block multi-stage attacks', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: `
            // Stage 1: Decode
            const decoded = atob("Y29kZQ==");
            // Stage 2: Async execution
            setTimeout(() => {
              // Stage 3: Global access
              window.location = decoded;
            }, 100);
          `
        }],
        instructions: 'Multi-stage attack'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('security_violation');
      expect(result.error).toContain('Multi-stage');
    });
  });

  describe('Positive Cases - Safe Code', () => {

    /**
     * TEST 16+: Verify safe code patterns are allowed
     */
    test('should allow safe code edits', async () => {
      const result = await tool.execute({
        initialCode: 'function add(a, b) { return a + b; }',
        edits: [{
          oldText: 'return a + b;',
          newText: 'return a + b + 1;'
        }],
        instructions: 'Add 1 to result'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('return a + b + 1');
      expect(result.stats?.editsApplied).toBe(1);
    });

    test('should allow complex but safe refactoring', async () => {
      const result = await tool.execute({
        initialCode: `
          const data = [1, 2, 3];
          const doubled = data.map(x => x * 2);
        `,
        edits: [{
          oldText: 'const doubled = data.map(x => x * 2);',
          newText: 'const doubled = data.map(x => x * 2).filter(x => x > 2);'
        }],
        instructions: 'Add filter to map chain'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('.filter');
      expect(result.stats?.editsApplied).toBe(1);
    });

    test('should allow safe string operations', async () => {
      const result = await tool.execute({
        initialCode: 'const name = "John";',
        edits: [{
          oldText: 'const name = "John";',
          newText: 'const name = "John"; const greeting = `Hello ${name}`;'
        }],
        instructions: 'Add template literal'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('Hello ${name}');
    });

    test('should allow safe array operations', async () => {
      const result = await tool.execute({
        initialCode: 'const arr = [1, 2, 3];',
        edits: [{
          oldText: 'const arr = [1, 2, 3];',
          newText: 'const arr = [1, 2, 3]; const sum = arr.reduce((a, b) => a + b, 0);'
        }],
        instructions: 'Add reduce sum'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('reduce');
    });

    test('should allow safe object operations', async () => {
      const result = await tool.execute({
        initialCode: 'const obj = { a: 1 };',
        edits: [{
          oldText: 'const obj = { a: 1 };',
          newText: 'const obj = { a: 1, b: 2 }; const keys = Object.keys(obj);'
        }],
        instructions: 'Add Object.keys'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('Object.keys');
    });

    test('should allow safe class definitions', async () => {
      const result = await tool.execute({
        initialCode: 'class Calculator {}',
        edits: [{
          oldText: 'class Calculator {}',
          newText: 'class Calculator { add(a, b) { return a + b; } }'
        }],
        instructions: 'Add add method'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('add(a, b)');
    });

    test('should allow safe async/await', async () => {
      const result = await tool.execute({
        initialCode: 'async function fetch() {}',
        edits: [{
          oldText: 'async function fetch() {}',
          newText: 'async function fetch() { const data = await getData(); return data; }'
        }],
        instructions: 'Add await'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('await');
    });

    test('should allow safe destructuring', async () => {
      const result = await tool.execute({
        initialCode: 'const obj = { a: 1, b: 2 };',
        edits: [{
          oldText: 'const obj = { a: 1, b: 2 };',
          newText: 'const obj = { a: 1, b: 2 }; const { a, b } = obj;'
        }],
        instructions: 'Add destructuring'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('const { a, b }');
    });
  });

  describe('Input Validation', () => {

    test('should reject missing initialCode', async () => {
      const result = await tool.execute({
        initialCode: null as any,
        edits: [],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('validation');
      expect(result.error).toContain('initialCode');
    });

    test('should reject non-array edits', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: 'not an array' as any,
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('validation');
      expect(result.error).toContain('edits must be an array');
    });

    test('should reject edit without oldText', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{ newText: 'new' } as any],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('validation');
      expect(result.error).toContain('oldText');
    });

    test('should reject edit without newText', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{ oldText: 'old' } as any],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.errorType).toBe('validation');
      expect(result.error).toContain('newText');
    });
  });

  describe('Edge Cases', () => {

    test('should handle empty edits array', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [],
        instructions: 'No edits'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toBe('const x = 1;');
      expect(result.stats?.editsApplied).toBe(0);
    });

    test('should handle non-matching edit gracefully', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'not found',
          newText: 'replacement'
        }],
        instructions: 'Non-matching edit'
      });

      expect(result.success).toBe(true);
      expect(result.stats?.editsApplied).toBe(0);
      expect(result.stats?.editsAttempted).toBe(1);
    });

    test('should handle multiple edits', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1; const y = 2; const z = 3;',
        edits: [
          { oldText: 'const x = 1;', newText: 'const x = 10;' },
          { oldText: 'const y = 2;', newText: 'const y = 20;' },
          { oldText: 'const z = 3;', newText: 'const z = 30;' }
        ],
        instructions: 'Multiple edits'
      });

      expect(result.success).toBe(true);
      expect(result.stats?.editsApplied).toBe(3);
      expect(result.editedCode).toContain('const x = 10;');
      expect(result.editedCode).toContain('const y = 20;');
      expect(result.editedCode).toContain('const z = 30;');
    });

    test('should provide detailed statistics', async () => {
      const result = await tool.execute({
        initialCode: 'function test() { return 1; }',
        edits: [{
          oldText: 'return 1;',
          newText: 'return 2;'
        }],
        instructions: 'Change return value'
      });

      expect(result.success).toBe(true);
      expect(result.stats).toBeDefined();
      expect(result.stats?.originalLength).toBeGreaterThan(0);
      expect(result.stats?.editedLength).toBeGreaterThan(0);
      expect(result.stats?.editsApplied).toBe(1);
      expect(result.stats?.editsAttempted).toBe(1);
    });

    test('should handle unicode characters safely', async () => {
      const result = await tool.execute({
        initialCode: 'const message = "Hello";',
        edits: [{
          oldText: 'const message = "Hello";',
          newText: 'const message = "你好世界";'
        }],
        instructions: 'Add unicode'
      });

      expect(result.success).toBe(true);
      expect(result.editedCode).toContain('你好世界');
    });
  });

  describe('Error Messages', () => {

    test('should provide clear error for eval blocking', async () => {
      const result = await tool.execute({
        initialCode: 'const x = eval("1");',
        edits: [],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error.length).toBeGreaterThan(0);
      expect(result.blockedPattern).toBeDefined();
    });

    test('should provide detailed error for size limit', async () => {
      const largeCode = 'x'.repeat(600 * 1024);
      const result = await tool.execute({
        initialCode: largeCode,
        edits: [],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('exceeds');
      expect(result.error).toContain('bytes');
      expect(result.errorType).toBe('size_limit');
    });

    test('should provide actionable error messages', async () => {
      const result = await tool.execute({
        initialCode: 'const x = 1;',
        edits: [{
          oldText: 'const x = 1;',
          newText: 'const x = require("fs");'
        }],
        instructions: 'Test'
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Security threat detected');
      expect(result.errorType).toBe('security_violation');
      expect(result.blockedPattern).toBeDefined();
    });
  });
});
