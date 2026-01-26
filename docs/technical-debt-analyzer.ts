#!/usr/bin/env node

/**
 * Technical Debt Analyzer for BubbleLab Bubbles
 *
 * Analyzes bubble files for technical debt patterns:
 * 1. Code Duplication
 * 2. Long Methods
 * 3. Magic Numbers/Strings
 * 4. Complex Conditional Logic
 * 5. Poor Naming
 */

import * as fs from 'fs';
import * as path from 'path';
import * as glob from 'glob';

interface DebtIssue {
  file: string;
  line: number;
  type: string;
  severity: 'high' | 'medium' | 'low';
  description: string;
  suggestion: string;
}

interface AnalysisResult {
  file: string;
  lines: number;
  issues: DebtIssue[];
  metrics: {
    longMethods: number;
    magicNumbers: number;
    complexConditionals: number;
    duplicatedPatterns: number;
  };
}

const MAGIC_NUMBER_THRESHOLD = 10;
const LONG_METHOD_THRESHOLD = 50;
const COMPLEX_CONDITIONAL_THRESHOLD = 3;

function analyzeFile(filePath: string): AnalysisResult {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const issues: DebtIssue[] = [];

  let currentFunction: { name: string; startLine: number; lines: number[] } | null = null;
  const metrics = {
    longMethods: 0,
    magicNumbers: 0,
    complexConditionals: 0,
    duplicatedPatterns: 0,
  };

  lines.forEach((line, index) => {
    const lineNum = index + 1;
    const trimmed = line.trim();

    // Track function boundaries
    const functionMatch = trimmed.match(/^(\s*)(?:async\s+)?(?:function\s+)?(\w+)\s*\(/);
    if (functionMatch && !trimmed.includes('//')) {
      if (currentFunction && currentFunction.lines.length > LONG_METHOD_THRESHOLD) {
        metrics.longMethods++;
        issues.push({
          file: filePath,
          line: currentFunction.startLine,
          type: 'long-method',
          severity: currentFunction.lines.length > 100 ? 'high' : 'medium',
          description: `Function '${currentFunction.name}' is ${currentFunction.lines.length} lines long`,
          suggestion: 'Extract logical sections into separate helper functions',
        });
      }
      currentFunction = {
        name: functionMatch[2],
        startLine: lineNum,
        lines: [],
      };
    } else if (currentFunction) {
      currentFunction.lines.push(lineNum);
    }

    // Detect magic numbers (numbers not in variables)
    const numbers = trimmed.match(/\b\d{2,}\b/g);
    if (numbers && !trimmed.includes('//') && !trimmed.includes('const') && !trimmed.includes('type')) {
      numbers.forEach((num) => {
        const numVal = parseInt(num);
        if (numVal >= MAGIC_NUMBER_THRESHOLD) {
          metrics.magicNumbers++;
          issues.push({
            file: filePath,
            line: lineNum,
            type: 'magic-number',
            severity: 'low',
            description: `Magic number: ${num}`,
            suggestion: `Extract to named constant: ${num.toUpperCase()}_VALUE`,
          });
        }
      });
    }

    // Detect complex conditionals
    const ifDepth = (trimmed.match(/\bif\b/g) || []).length;
    const elseCount = (trimmed.match(/\belse\b/g) || []).length;
    const logicalOperators = (trimmed.match(/&&|\|\|/g) || []).length;

    if (logicalOperators >= COMPLEX_CONDITIONAL_THRESHOLD && !trimmed.includes('//')) {
      metrics.complexConditionals++;
      issues.push({
        file: filePath,
        line: lineNum,
        type: 'complex-conditional',
        severity: 'medium',
        description: `Complex conditional with ${logicalOperators} logical operators`,
        suggestion: 'Extract condition to named variable or function',
      });
    }

    // Detect nested conditionals
    const indentMatch = line.match(/^(\s*)/);
    if (indentMatch) {
      const indentLevel = indentMatch[1].length;
      if (indentLevel > 24 && (trimmed.includes('if') || trimmed.includes('else'))) {
        issues.push({
          file: filePath,
          line: lineNum,
          type: 'nested-conditional',
          severity: 'high',
          description: `Deep nesting (${Math.floor(indentLevel / 2)} levels)`,
          suggestion: 'Use early returns or extract to separate function',
        });
      }
    }

    // Detect hardcoded strings that look like URLs or API endpoints
    const stringMatch = trimmed.match(/['"](https?:\/\/[^'"]+)['"]/);
    if (stringMatch && !trimmed.includes('const') && !trimmed.includes('=')) {
      issues.push({
        file: filePath,
        line: lineNum,
        type: 'hardcoded-url',
        severity: 'medium',
        description: `Hardcoded URL: ${stringMatch[1]}`,
        suggestion: 'Extract to configuration constant',
      });
    }

    // Detect poor naming patterns
    const poorVarNames = trimmed.match(/\b(tmp|temp|data|item|obj|val)\b/g);
    if (poorVarNames && trimmed.match(/let\s+\w+/)) {
      issues.push({
        file: filePath,
        line: lineNum,
        type: 'poor-naming',
        severity: 'low',
        description: `Possibly unclear variable name: ${poorVarNames.join(', ')}`,
        suggestion: 'Use more descriptive variable names',
      });
    }
  });

  // Check last function
  if (currentFunction && currentFunction.lines.length > LONG_METHOD_THRESHOLD) {
    metrics.longMethods++;
    issues.push({
      file: filePath,
      line: currentFunction.startLine,
      type: 'long-method',
      severity: currentFunction.lines.length > 100 ? 'high' : 'medium',
      description: `Function '${currentFunction.name}' is ${currentFunction.lines.length} lines long`,
      suggestion: 'Extract logical sections into separate helper functions',
    });
  }

  return {
    file: filePath,
    lines: lines.length,
    issues,
    metrics,
  };
}

function analyzeAllFiles(globPattern: string): AnalysisResult[] {
  const files = glob.sync(globPattern);
  return files.map((file) => analyzeFile(file));
}

function generateReport(results: AnalysisResult[]): void {
  const totalIssues = results.reduce((sum, r) => sum + r.issues.length, 0);
  const highSeverity = results.reduce(
    (sum, r) => sum + r.issues.filter((i) => i.severity === 'high').length,
    0
  );
  const mediumSeverity = results.reduce(
    (sum, r) => sum + r.issues.filter((i) => i.severity === 'medium').length,
    0
  );
  const lowSeverity = results.reduce(
    (sum, r) => sum + r.issues.filter((i) => i.severity === 'low').length,
    0
  );

  console.log('\n=== TECHNICAL DEBT ANALYSIS REPORT ===\n');
  console.log(`Files analyzed: ${results.length}`);
  console.log(`Total issues: ${totalIssues}`);
  console.log(`  High severity: ${highSeverity}`);
  console.log(`  Medium severity: ${mediumSeverity}`);
  console.log(`  Low severity: ${lowSeverity}\n`);

  // Top 10 files with most issues
  console.log('=== TOP 10 FILES BY ISSUE COUNT ===\n');
  const sortedResults = [...results].sort((a, b) => b.issues.length - a.issues.length).slice(0, 10);

  sortedResults.forEach((result, index) => {
    console.log(`${index + 1}. ${path.basename(result.file)} (${result.issues.length} issues, ${result.lines} lines)`);
  });

  // Categorize issues
  console.log('\n=== ISSUES BY CATEGORY ===\n');
  const categories = {
    'long-method': 0,
    'magic-number': 0,
    'complex-conditional': 0,
    'nested-conditional': 0,
    'hardcoded-url': 0,
    'poor-naming': 0,
  };

  results.forEach((result) => {
    result.issues.forEach((issue) => {
      categories[issue.type as keyof typeof categories]++;
    });
  });

  Object.entries(categories).forEach(([category, count]) => {
    console.log(`${category}: ${count}`);
  });

  // Save detailed report
  const reportPath = path.join(__dirname, 'technical-debt-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`\nDetailed report saved to: ${reportPath}`);
}

// Main execution
const bubbleFiles = path.join(
  process.cwd(),
  '../BubbleLab/packages/bubble-core/src/bubbles/**/*.ts'
);
const results = analyzeAllFiles(bubbleFiles.replace(/\\/g, '/'));
generateReport(results);
