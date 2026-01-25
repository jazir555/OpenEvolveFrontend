#!/usr/bin/env tsx

/**
 * Bubble Refactoring Helper Script
 *
 * This script assists in refactoring bubbles to use common utilities.
 * It analyzes bubble files and suggests refactoring changes.
 *
 * Usage:
 *   npx tsx scripts/refactor-bubbles.ts analyze <bubble-file>
 *   npx tsx scripts/refactor-bubbles.ts refactor <bubble-file>
 *   npx tsx scripts/refactor-bubbles.ts stats
 */

import * as fs from 'fs/promises';
import * as path from 'path';

interface RefactorMetrics {
  totalLines: number;
  duplicatedValidation: number;
  duplicatedErrorHandling: number;
  duplicatedRetry: number;
  jsdocCount: number;
  jsdocCoverage: number;
  commonImports: number;
}

class BubbleRefactorer {
  /**
   * Analyze a bubble file for refactoring opportunities
   *
   * @param filePath - Path to bubble file
   * @returns Analysis results
   */
  async analyze(filePath: string): Promise<RefactorMetrics & { suggestions: string[] }> {
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n');

    const suggestions: string[] = [];
    let duplicatedValidation = 0;
    let duplicatedErrorHandling = 0;
    let duplicatedRetry = 0;
    let jsdocCount = 0;

    // Check for custom validation patterns
    const validationPatterns = [
      /if\s*\(!\w+\s*\|\|.*typeof.*!==\s*['"`]string['"`]\)/,
      /throw new Error\(['"`]Invalid.*['"`]\)/,
      /\bvalidate[A-Z]\w*\(/,
      /if\s*\(\w+\.length\s*>\s*\d+\)/,
    ];

    // Check for custom error handling
    const errorPatterns = [
      /catch\s*\(\w+\)\s*\{[\s\S]*?console\.error/,
      /throw new Error\(/,
      /class \w+Error extends Error/,
    ];

    // Check for custom retry logic
    const retryPatterns = [
      /for\s*\(\s*let\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*\w+/,
      /setTimeout\(/,
      /retry[A-Z]\w*\(/,
      /maxAttempts/,
    ];

    // Count JSDoc comments
    const jsdocPattern = /\/\*\*\s*\n([\s\S]*?)\*\//g;
    const jsdocMatches = content.match(jsdocPattern) || [];
    jsdocCount = jsdocMatches.length;

    // Analyze each line
    for (const line of lines) {
      for (const pattern of validationPatterns) {
        if (pattern.test(line)) {
          duplicatedValidation++;
          break;
        }
      }

      for (const pattern of errorPatterns) {
        if (pattern.test(line)) {
          duplicatedErrorHandling++;
          break;
        }
      }

      for (const pattern of retryPatterns) {
        if (pattern.test(line)) {
          duplicatedRetry++;
          break;
        }
      }
    }

    // Generate suggestions
    if (duplicatedValidation > 0) {
      suggestions.push(`Found ${duplicatedValidation} lines of custom validation code that could use common validators`);
    }

    if (duplicatedErrorHandling > 0) {
      suggestions.push(`Found ${duplicatedErrorHandling} lines of custom error handling that could use common error classes`);
    }

    if (duplicatedRetry > 0) {
      suggestions.push(`Found ${duplicatedRetry} lines of custom retry logic that could use common retry utilities`);
    }

    if (jsdocCount === 0) {
      suggestions.push('No JSDoc comments found - add comprehensive documentation');
    }

    const totalLines = lines.length;
    const commonImports = (content.match(/from ['"]\.\.\/common\//g) || []).length;

    // Estimate JSDoc coverage (public methods and class)
    const classMatch = content.match(/export class \w+/);
    const publicMethods = (content.match(/^\s*(public|async|protected)\s+\w+/gm) || []).length;
    const totalDocumentables = (classMatch ? 1 : 0) + publicMethods;
    const jsdocCoverage = totalDocumentables > 0 ? (jsdocCount / totalDocumentables) * 100 : 0;

    return {
      totalLines,
      duplicatedValidation,
      duplicatedErrorHandling,
      duplicatedRetry,
      jsdocCount,
      jsdocCoverage,
      commonImports,
      suggestions,
    };
  }

  /**
   * Generate refactored version of a bubble file
   *
   * @param filePath - Path to bubble file
   * @returns Refactored code
   */
  async refactor(filePath: string): Promise<string> {
    const content = await fs.readFile(filePath, 'utf-8');
    let refactored = content;

    // Add common utility imports if not present
    if (!refactored.includes("from '../common/validators.js'")) {
      const importIndex = refactored.indexOf("from '../../types/");
      if (importIndex !== -1) {
        const insertPoint = refactored.lastIndexOf('\n', importIndex) + 1;
        refactored = [
          refactored.slice(0, insertPoint),
          "import { validateNonEmptyString } from '../common/validators.js';\n",
          "import { NetworkError, wrapError } from '../common/error-handlers.js';\n",
          "import { retryWithBackoff } from '../common/retry.js';\n",
          refactored.slice(insertPoint),
        ].join('');
      }
    }

    // Replace custom validation patterns
    // Example: if (!param || typeof param !== 'string') -> validateNonEmptyString(param, 'param')
    refactored = refactored.replace(
      /if\s*\((!\w+\s*\|\|\s*typeof\s+(\w+)\s*!==\s*['"`]string['"`])\)\s*\{\s*throw new Error\(['"`](\w+) is required['"`]\)\s*\}/g,
      'validateNonEmptyString($2, \'$3\')'
    );

    // Replace custom error handling
    // Example: throw new Error('message') -> throw new NetworkError('message')
    refactored = refactored.replace(
      /throw new Error\(['"`](Network error|Connection failed|Timeout)['"`]\)/g,
      'new NetworkError(\'$1\')'
    );

    // Replace custom retry loops
    // Example: for (let i = 0; i < maxRetries; i++) -> retryWithBackoff(...)
    // This is more complex and would need careful handling

    return refactored;
  }

  /**
   * Calculate refactoring statistics across all bubbles
   *
   * @returns Aggregate statistics
   */
  async stats(): Promise<{
    totalBubbles: number;
    refactoredBubbles: number;
    avgJSDocCoverage: number;
    totalDuplicatedCode: number;
    potentialSavings: number;
  }> {
    const bubblesDir = path.join(process.cwd(), 'src/bubbles');
    const serviceBubbles = path.join(bubblesDir, 'service-bubble');
    const toolBubbles = path.join(bubblesDir, 'tool-bubble');

    const allBubbles = [
      ...(await this.findBubbleFiles(serviceBubbles)),
      ...(await this.findBubbleFiles(toolBubbles)),
    ];

    let totalDuplicatedCode = 0;
    let totalJSDocCoverage = 0;
    let refactoredCount = 0;

    for (const bubble of allBubbles) {
      const metrics = await this.analyze(bubble);
      totalDuplicatedCode +=
        metrics.duplicatedValidation +
        metrics.duplicatedErrorHandling +
        metrics.duplicatedRetry;
      totalJSDocCoverage += metrics.jsdocCoverage;

      if (metrics.commonImports > 0) {
        refactoredCount++;
      }
    }

    return {
      totalBubbles: allBubbles.length,
      refactoredBubbles: refactoredCount,
      avgJSDocCoverage: totalJSDocCoverage / allBubbles.length,
      totalDuplicatedCode,
      potentialSavings: totalDuplicatedCode * 0.6, // Estimate 60% reduction
    };
  }

  /**
   * Find all bubble files in a directory
   *
   * @param dir - Directory to search
   * @returns Array of file paths
   */
  private async findBubbleFiles(dir: string): Promise<string[]> {
    const files: string[] = [];

    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          files.push(...(await this.findBubbleFiles(fullPath)));
        } else if (
          entry.isFile() &&
          (entry.name.endsWith('.ts') || entry.name.endsWith('.js')) &&
          !entry.name.includes('.test.') &&
          !entry.name.includes('.schema.')
        ) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      // Directory might not exist
    }

    return files;
  }
}

// CLI interface
async function main() {
  const [,, command, ...args] = process.argv;
  const refactorer = new BubbleRefactorer();

  switch (command) {
    case 'analyze': {
      const filePath = args[0];
      if (!filePath) {
        console.error('Usage: refactor-bubbles analyze <file-path>');
        process.exit(1);
      }

      const metrics = await refactorer.analyze(filePath);
      console.log('\n📊 Refactoring Analysis\n');
      console.log(`Total Lines: ${metrics.totalLines}`);
      console.log(`Duplicated Validation: ${metrics.duplicatedValidation} lines`);
      console.log(`Duplicated Error Handling: ${metrics.duplicatedErrorHandling} lines`);
      console.log(`Duplicated Retry Logic: ${metrics.duplicatedRetry} lines`);
      console.log(`JSDoc Comments: ${metrics.jsdocCount}`);
      console.log(`JSDoc Coverage: ${metrics.jsdocCoverage.toFixed(1)}%`);
      console.log(`Common Imports: ${metrics.commonImports}`);
      console.log('\n💡 Suggestions:\n');
      metrics.suggestions.forEach((s) => console.log(`  • ${s}`));
      break;
    }

    case 'refactor': {
      const filePath = args[0];
      if (!filePath) {
        console.error('Usage: refactor-bubbles refactor <file-path>');
        process.exit(1);
      }

      const refactored = await refactorer.refactor(filePath);
      console.log('\n✅ Refactored Code:\n');
      console.log(refactored);
      break;
    }

    case 'stats': {
      const stats = await refactorer.stats();
      console.log('\n📈 Aggregate Refactoring Statistics\n');
      console.log(`Total Bubbles: ${stats.totalBubbles}`);
      console.log(`Refactored Bubbles: ${stats.refactoredBubbles}`);
      console.log(`Average JSDoc Coverage: ${stats.avgJSDocCoverage.toFixed(1)}%`);
      console.log(`Total Duplicated Code: ${stats.totalDuplicatedCode} lines`);
      console.log(`Potential Savings: ~${stats.potentialSavings.toFixed(0)} lines`);
      console.log(`Progress: ${((stats.refactoredBubbles / stats.totalBubbles) * 100).toFixed(1)}%`);
      break;
    }

    default:
      console.log(`
Bubble Refactoring Helper

Usage:
  npx tsx scripts/refactor-bubbles.ts analyze <file-path>
    Analyze a bubble file for refactoring opportunities

  npx tsx scripts/refactor-bubbles.ts refactor <file-path>
    Generate refactored version of a bubble file

  npx tsx scripts/refactor-bubbles.ts stats
    Show aggregate refactoring statistics across all bubbles

Examples:
  npx tsx scripts/refactor-bubbles.ts analyze src/bubbles/service-bubble/postgresql.ts
  npx tsx scripts/refactor-bubbles.ts stats
      `);
  }
}

main().catch(console.error);
