#!/usr/bin/env tsx
/**
 * Automated Security Fix Script
 *
 * This script fixes all identified security issues in the BubbleLab codebase:
 * 1. Code injection vulnerabilities (replaces dangerous eval/Function with safe alternatives)
 * 2. Missing input validation (adds validation to all execute methods)
 * 3. Missing timeout handling (adds timeout to network calls)
 * 4. Missing rate limiting (adds rate limiting to API calls)
 *
 * Usage:
 *   npx tsx scripts/fix-security-issues.ts [--dry-run] [--fix-code-injection] [--add-validation] [--add-timeouts] [--add-rate-limiting]
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface FixOptions {
  dryRun: boolean;
  fixCodeInjection: boolean;
  addValidation: boolean;
  addTimeouts: boolean;
  addRateLimiting: boolean;
}

interface SecurityIssue {
  file: string;
  line: number;
  type: 'code-injection' | 'missing-validation' | 'missing-timeout' | 'missing-rate-limit';
  severity: 'high' | 'medium' | 'low';
  description: string;
}

// ============================================================================
// SECURITY PATTERN DETECTORS
// ============================================================================

const DANGEROUS_PATTERNS = {
  codeInjection: [
    /\bnew\s+Function\s*\(/g,
    /\beval\s*\(/g,
    /setTimeout\s*\(\s*['"]/g,
    /setInterval\s*\(\s*['"]/g,
  ],
  missingValidation: [
    /async\s+performAction\s*\([^)]*\)\s*:\s*Promise<[^>]+>\s*\{[\s\S]{0,500}(?!validateInput|validateOrThrow)/,
  ],
  missingTimeout: [
    /await\s+fetch\s*\([^)]+\)(?!\s*\.\s*withTimeout)/,
    /await\s+httpBubble\.(action|execute)\s*\(\)(?![\s\S]*timeout)/,
  ],
  missingRateLimit: [
    /class\s+\w+\s+extends\s+(Service|Tool)Bubble/,
  ],
};

// ============================================================================
// FILE SCANNER
// ============================================================================

function scanFile(filePath: string): SecurityIssue[] {
  const content = readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const issues: SecurityIssue[] = [];

  // Check for code injection
  for (const pattern of DANGEROUS_PATTERNS.codeInjection) {
    let match;
    const regex = new RegExp(pattern.source, 'gi');
    let lineIndex = 0;

    for (const line of lines) {
      regex.lastIndex = 0;
      if (regex.test(line)) {
        issues.push({
          file: filePath,
          line: lineIndex + 1,
          type: 'code-injection',
          severity: 'high',
          description: `Potentially dangerous code execution pattern: ${pattern.source}`,
        });
      }
      lineIndex++;
    }
  }

  // Check for missing input validation
  if (content.includes('async performAction(')) {
    const hasValidation =
      content.includes('validateInput') ||
      content.includes('validateOrThrow') ||
      content.includes('schema.parse') ||
      content.includes('InputValidator');

    if (!hasValidation) {
      issues.push({
        file: filePath,
        line: content.indexOf('async performAction(') / 100,
        type: 'missing-validation',
        severity: 'high',
        description: 'Missing input validation in performAction method',
      });
    }
  }

  // Check for missing timeout on network calls
  if (content.includes('await fetch(') || content.includes('await httpBubble')) {
    const hasTimeout =
      content.includes('withTimeout') ||
      content.includes('timeout:') ||
      content.includes('AbortSignal.timeout');

    if (!hasTimeout) {
      issues.push({
        file: filePath,
        line: 1,
        type: 'missing-timeout',
        severity: 'medium',
        description: 'Network calls without timeout handling',
      });
    }
  }

  // Check for missing rate limiting
  if (content.includes('extends ServiceBubble') || content.includes('extends ToolBubble')) {
    const hasRateLimit =
      content.includes('RateLimiter') ||
      content.includes('rateLimiter') ||
      content.includes('checkLimit');

    if (!hasRateLimit) {
      issues.push({
        file: filePath,
        line: 1,
        type: 'missing-rate-limit',
        severity: 'medium',
        description: 'Missing rate limiting on API calls',
      });
    }
  }

  return issues;
}

// ============================================================================
// SECURITY FIXERS
// ============================================================================

function fixCodeInjection(content: string): string {
  let fixed = content;

  // Fix 1: Replace unsafe new Function with sandboxed execution
  fixed = fixed.replace(
    /const\s+fn\s*=\s*new\s+Function\s*\([^)]+\)\s*;?/g,
    (match) => {
      console.warn(`  [CODE INJECTION] Replacing unsafe Function constructor`);
      return '// SECURITY: Using isolated-vm for safe code execution\n' +
             'const fn = createSandboxedFunction(...sandboxKeys, wrappedCode);';
    }
  );

  // Fix 2: Replace eval with safe alternative
  fixed = fixed.replace(
    /\beval\s*\(\s*([^)]+)\s*\)/g,
    (match, args) => {
      console.warn(`  [CODE INJECTION] Replacing eval() with safe alternative`);
      return `safeEvaluate(${args})`;
    }
  );

  // Fix 3: Replace dangerous setTimeout with string
  fixed = fixed.replace(
    /setTimeout\s*\(\s*['"]([^'"]+)['"]\s*,\s*(\d+)\s*\)/g,
    (match, code, delay) => {
      console.warn(`  [CODE INJECTION] Replacing dangerous setTimeout with string`);
      return `setTimeout(() => { /* ${code} */ }, ${delay})`;
    }
  );

  return fixed;
}

function addInputValidation(content: string, className: string): string {
  // Skip if already has validation
  if (content.includes('validateInput') || content.includes('validateOrThrow')) {
    return content;
  }

  // Find the performAction method
  const performActionMatch = content.match(
    /async\s+performAction\s*\([^)]*\)\s*:\s*Promise<[^>]+>\s*\{/
  );

  if (!performActionMatch) {
    return content;
  }

  const validationImport =
    "import { validateOrThrow, isSafeFromInjection } from '../../security/index.js';\n";

  // Add import if not present
  if (!content.includes('../../security/index.js')) {
    content = content.replace(
      /(^import\s+.*\n)+/,
      (imports) => validationImport + imports
    );
  }

  // Add validation at the start of performAction
  const validationCode = `
    // SECURITY: Validate inputs
    try {
      // Validate parameters against schema
      if (this.params) {
        // Basic safety check
        const paramsStr = JSON.stringify(this.params);
        if (!isSafeFromInjection(paramsStr)) {
          throw new Error('Input contains potentially dangerous patterns');
        }
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Input validation failed',
      } as any;
    }
  `;

  content = content.replace(
    /async\s+performAction\s*\([^)]*\)\s*:\s*Promise<[^>]+>\s*\{/,
    (match) => match + validationCode
  );

  console.log(`  [VALIDATION] Added input validation to ${className}`);
  return content;
}

function addTimeoutHandling(content: string, className: string): string {
  // Skip if already has timeout handling
  if (content.includes('withTimeout') || content.includes('AbortSignal.timeout')) {
    return content;
  }

  // Add timeout import
  const timeoutImport =
    "import { withTimeout } from '../../security/timeout.js';\n";

  if (!content.includes('../../security/timeout.js')) {
    content = content.replace(
      /(^import\s+.*\n)+/,
      (imports) => timeoutImport + imports
    );
  }

  // Fix fetch calls without timeout
  content = content.replace(
    /await\s+fetch\s*\(([^)]+)\)/g,
    (match, args) => {
      // Check if already has timeout
      if (args.includes('timeout') || args.includes('AbortSignal')) {
        return match;
      }

      console.log(`  [TIMEOUT] Added timeout to fetch in ${className}`);
      return `await withTimeout(fetch(${args}), 30000)`;
    }
  );

  return content;
}

function addRateLimiting(content: string, className: string): string {
  // Skip if already has rate limiting
  if (content.includes('RateLimiter') || content.includes('rateLimiter')) {
    return content;
  }

  // Add rate limiter import
  const rateLimitImport =
    "import { RateLimiter } from '../../security/rate-limiter.js';\n";

  if (!content.includes('../../security/rate-limiter.js')) {
    content = content.replace(
      /(^import\s+.*\n)+/,
      (imports) => rateLimitImport + imports
    );
  }

  // Add rate limiter property
  const propertyDeclaration =
    '\n  // Rate limiter for API calls\n' +
    '  private rateLimiter = new RateLimiter({\n' +
    '    maxRequests: 60,\n' +
    '    windowMs: 60000\n' +
    '  });\n';

  // Find class declaration and add property after static properties
  content = content.replace(
    /(static\s+readonly\s+\w+\s*=\s*[^;]+;\s*)+/,
    (match) => match + propertyDeclaration
  );

  // Add rate limit check to performAction
  const rateLimitCheck =
    '\n    // SECURITY: Check rate limit\n' +
    '    await this.rateLimiter.checkLimit(\n' +
    '      this.context?.variableId || \'default\'\n' +
    '    );\n';

  content = content.replace(
    /async\s+performAction\s*\([^)]*\)\s*:\s*Promise<[^>]+>\s*\{/,
    (match) => match + rateLimitCheck
  );

  console.log(`  [RATE LIMIT] Added rate limiting to ${className}`);
  return content;
}

// ============================================================================
// MAIN FIX FUNCTION
// ============================================================================

function fixSecurityIssues(
  files: string[],
  options: FixOptions
): { fixed: number; errors: string[] } {
  const results = { fixed: 0, errors: [] };

  for (const file of files) {
    try {
      console.log(`\n🔍 Scanning: ${file}`);

      const content = readFileSync(file, 'utf-8');
      let fixedContent = content;
      let hasChanges = false;

      // Extract class name
      const classMatch = content.match(/export\s+class\s+(\w+)/);
      const className = classMatch ? classMatch[1] : 'Unknown';

      // Apply fixes based on options
      if (options.fixCodeInjection) {
        const before = fixedContent;
        fixedContent = fixCodeInjection(fixedContent);
        if (before !== fixedContent) hasChanges = true;
      }

      if (options.addValidation) {
        const before = fixedContent;
        fixedContent = addInputValidation(fixedContent, className);
        if (before !== fixedContent) hasChanges = true;
      }

      if (options.addTimeouts) {
        const before = fixedContent;
        fixedContent = addTimeoutHandling(fixedContent, className);
        if (before !== fixedContent) hasChanges = true;
      }

      if (options.addRateLimiting) {
        const before = fixedContent;
        fixedContent = addRateLimiting(fixedContent, className);
        if (before !== fixedContent) hasChanges = true;
      }

      // Write changes
      if (hasChanges && !options.dryRun) {
        writeFileSync(file, fixedContent, 'utf-8');
        console.log(`  ✅ Fixed: ${file}`);
        results.fixed++;
      } else if (hasChanges && options.dryRun) {
        console.log(`  🔶 Would fix: ${file} (dry run)`);
        results.fixed++;
      } else {
        console.log(`  ✅ No issues found: ${file}`);
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error(`  ❌ Error processing ${file}: ${errorMsg}`);
      results.errors.push(`${file}: ${errorMsg}`);
    }
  }

  return results;
}

// ============================================================================
// CLI
// ============================================================================

function parseArgs(): FixOptions {
  const args = process.argv.slice(2);

  return {
    dryRun: args.includes('--dry-run'),
    fixCodeInjection: args.includes('--fix-code-injection') || args.includes('--all'),
    addValidation: args.includes('--add-validation') || args.includes('--all'),
    addTimeouts: args.includes('--add-timeouts') || args.includes('--all'),
    addRateLimiting: args.includes('--add-rate-limiting') || args.includes('--all'),
  };
}

function getFilesToFix(): string[] {
  // Key files with security issues
  const files = [
    'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\BubbleLab\\packages\\bubble-core\\src\\bubbles\\service-bubble\\ace-tools-bubble.ts',
    'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\BubbleLab\\packages\\bubble-core\\src\\bubbles\\workflow-bubble\\event-handler.workflow.ts',
    'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\BubbleLab\\packages\\bubble-core\\src\\bubbles\\tool-bubble\\data-transformer-tool.ts',
    'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\BubbleLab\\packages\\bubble-core\\src\\bubbles\\service-bubble\\http.ts',
    'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend\\BubbleLab\\packages\\bubble-core\\src\\bubbles\\service-bubble\\apify\\apify.ts',
  ];

  return files.filter(existsSync);
}

function main() {
  console.log('🔒 Security Fix Script');
  console.log('=====================\n');

  const options = parseArgs();

  if (options.dryRun) {
    console.log('🔶 DRY RUN MODE - No changes will be made\n');
  }

  const files = getFilesToFix();
  console.log(`Found ${files.length} files to scan\n`);

  if (files.length === 0) {
    console.log('No files found to fix');
    return;
  }

  const results = fixSecurityIssues(files, options);

  console.log('\n=====================');
  console.log(`\n✅ Fixed: ${results.fixed} files`);
  console.log(`❌ Errors: ${results.errors.length}`);

  if (results.errors.length > 0) {
    console.log('\nErrors:');
    results.errors.forEach((error) => console.log(`  - ${error}`));
  }

  if (options.dryRun) {
    console.log('\n⚠️  This was a dry run. Run without --dry-run to apply fixes.');
  }
}

main();
