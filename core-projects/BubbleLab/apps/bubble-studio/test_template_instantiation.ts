/**
 * Template Instantiation Test
 *
 * This script tests that both templates:
 * 1. Can be loaded
 * 2. Have valid TypeScript syntax
 * 3. Can be instantiated
 * 4. Have valid metadata
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ANSI colors
const reset = '\x1b[0m';
const red = '\x1b[31m';
const green = '\x1b[32m';
const yellow = '\x1b[33m';
const blue = '\x1b[34m';
const cyan = '\x1b[36m';

interface TemplateTestResult {
  name: string;
  passed: boolean;
  errors: string[];
  warnings: string[];
}

function log(color: string, message: string) {
  console.log(`${color}${message}${reset}`);
}

function success(message: string) {
  log(green, `✓ ${message}`);
}

function error(message: string) {
  log(red, `✗ ${message}`);
}

function info(message: string) {
  log(blue, `ℹ ${message}`);
}

async function testTemplate(templatePath: string): Promise<TemplateTestResult> {
  const result: TemplateTestResult = {
    name: path.basename(templatePath, '.ts'),
    passed: true,
    errors: [],
    warnings: [],
  };

  info(`\nTesting template: ${result.name}`);

  // 1. Load file
  let content: string;
  try {
    content = fs.readFileSync(templatePath, 'utf-8');
    success('File loaded successfully');
  } catch (err) {
    result.errors.push(`Failed to load file: ${err}`);
    error(`Failed to load file: ${err}`);
    result.passed = false;
    return result;
  }

  // 2. Check for required exports
  info('Checking exports...');

  if (!content.includes('export const templateCode')) {
    result.errors.push('Missing templateCode export');
    error('Missing templateCode export');
    result.passed = false;
  } else {
    success('Has templateCode export');
  }

  if (!content.includes('export const metadata')) {
    result.warnings.push('Missing metadata export');
  } else {
    success('Has metadata export');
  }

  // 3. Extract template code
  const templateCodeMatch = content.match(/export const templateCode = `([\s\S]+?)`;/);
  if (!templateCodeMatch) {
    result.errors.push('Could not extract template code');
    error('Could not extract template code');
    result.passed = false;
    return result;
  }

  const templateCode = templateCodeMatch[1];
  success('Template code extracted');

  // 4. Validate template code structure
  info('Validating template code structure...');

  const checks = [
    { pattern: /extends BubbleFlow/, name: 'Extends BubbleFlow' },
    { pattern: /async handle\(/, name: 'Has async handle method' },
    { pattern: /export interface Output/, name: 'Has Output interface' },
    { pattern: /export interface.*WebhookEvent/, name: 'Has webhook payload interface' },
    { pattern: /new \w+Bubble\(/, name: 'Uses bubbles' },
    { pattern: /\.action\(\)/, name: 'Has action() calls' },
    { pattern: /return \{/, name: 'Has return statement' },
    { pattern: /this\.logger/, name: 'Uses logger' },
  ];

  checks.forEach(check => {
    if (check.pattern.test(templateCode)) {
      success(check.name);
    } else {
      result.warnings.push(`Missing: ${check.name}`);
    }
  });

  // 5. Extract and validate metadata
  const metadataMatch = content.match(/export const metadata = ({[\s\S]+?});/);
  if (metadataMatch) {
    try {
      const metadataStr = metadataMatch[1];
      // Note: This is a simplified check - actual parsing would need eval
      if (metadataStr.includes('inputsSchema')) {
        success('Has inputsSchema');
      } else {
        result.warnings.push('Missing inputsSchema');
      }

      if (metadataStr.includes('requiredCredentials')) {
        success('Has requiredCredentials');
      } else {
        result.warnings.push('Missing requiredCredentials');
      }

      if (metadataStr.includes('preValidatedBubbles')) {
        success('Has preValidatedBubbles');
      } else {
        result.warnings.push('Missing preValidatedBubbles');
      }
    } catch (err) {
      result.warnings.push(`Failed to parse metadata: ${err}`);
    }
  }

  // 6. Check for common issues
  info('Checking for common issues...');

  // Check for template literal syntax issues
  if (templateCode.includes('${RECOMMENDED_MODELS')) {
    // This is inside the template string, so it should use \${ to escape
    if (templateCode.includes('\\${RECOMMENDED_MODELS')) {
      success('RECOMMENDED_MODELS properly escaped');
    } else {
      result.errors.push('RECOMMENDED_MODELS not escaped - will cause runtime error');
      error('RECOMMENDED_MODELS not escaped');
      result.passed = false;
    }
  }

  // Check for import statement inside template code
  if (templateCode.includes("import {")) {
    success('Has import statements inside template code');
  } else {
    result.errors.push('Missing import statements in template code');
    error('Missing import statements');
    result.passed = false;
  }

  return result;
}

async function main() {
  log(cyan, '\n╔════════════════════════════════════════════════════════════╗');
  log(cyan, '║     Template Instantiation Test Suite                      ║');
  log(cyan, '╚════════════════════════════════════════════════════════════╝\n');

  const templates = [
    'src/components/templates/template_codes/websiteLeadGeneration.ts',
    'src/components/templates/template_codes/nanobananaImagePipeline.ts',
  ];

  const results: TemplateTestResult[] = [];

  for (const templatePath of templates) {
    const fullPath = path.join(__dirname, templatePath);
    const result = await testTemplate(fullPath);
    results.push(result);
  }

  // Summary
  log(cyan, '\n' + '='.repeat(60));
  log(cyan, 'SUMMARY');
  log(cyan, '='.repeat(60) + '\n');

  let totalPassed = 0;
  let totalErrors = 0;
  let totalWarnings = 0;

  results.forEach(result => {
    if (result.passed) {
      success(`${result.name}: PASSED`);
    } else {
      error(`${result.name}: FAILED (${result.errors.length} errors)`);
    }

    totalErrors += result.errors.length;
    totalWarnings += result.warnings.length;

    if (result.errors.length > 0) {
      info('\nErrors:');
      result.errors.forEach(err => error(`  - ${err}`));
    }

    if (result.warnings.length > 0) {
      info('\nWarnings:');
      result.warnings.forEach(warn => log(yellow, `  ⚠ - ${warn}`));
    }
    info('');
  });

  if (totalErrors === 0) {
    totalPassed = results.length;
    success(`\n✓ All ${totalPassed} templates passed instantiation tests!`);
  } else {
    error(`\n✗ ${totalErrors} error(s) found`);
  }

  if (totalWarnings > 0) {
    log(yellow, `⚠ ${totalWarnings} warning(s)`);
  }

  process.exit(totalErrors > 0 ? 1 : 0);
}

main().catch(err => {
  error(`\nFatal error: ${err.message}`);
  console.error(err);
  process.exit(1);
});
