#!/usr/bin/env node
/**
 * Comprehensive Template Validation Script
 *
 * Tests both newly implemented templates:
 * 1. websiteLeadGeneration
 * 2. nanobananaImagePipeline
 *
 * Validation includes:
 * - File existence
 * - Import/export structure
 * - Template code structure
 * - Input schema validity
 * - Credential requirements
 * - Bubble usage validation
 * - Comparison with existing templates
 */

const fs = require('fs');
const path = require('path');

// ANSI colors for output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

function log(color, message) {
  console.log(`${color}${message}${colors.reset}`);
}

function success(message) {
  log(colors.green, `✓ ${message}`);
}

function error(message) {
  log(colors.red, `✗ ${message}`);
}

function info(message) {
  log(colors.blue, `ℹ ${message}`);
}

function warnMsg(message) {
  log(colors.yellow, `⚠ ${message}`);
}

// Template files to test
const TEMPLATES_TO_TEST = [
  {
    id: 'websiteLeadGeneration',
    file: 'src/components/templates/template_codes/websiteLeadGeneration.ts',
    expectedBubbles: ['WebScrapeTool', 'AIAgentBubble', 'GoogleDriveBubble', 'ResendBubble'],
    expectedCredentials: ['web-scrape', 'google-drive', 'resend'],
  },
  {
    id: 'nanobananaImagePipeline',
    file: 'src/components/templates/template_codes/nanobananaImagePipeline.ts',
    expectedBubbles: ['GoogleSheetsBubble', 'GoogleDriveBubble', 'AIAgentBubble'],
    expectedCredentials: ['google-sheets', 'google-drive', 'ai-agent'],
  },
];

// Reference templates for comparison
const REFERENCE_TEMPLATES = [
  {
    id: 'githubScraper',
    file: 'src/components/templates/template_codes/githubScraper.ts',
  },
  {
    id: 'productImageTransformer',
    file: 'src/components/templates/template_codes/productImageTransformer.ts',
  },
  {
    id: 'linkedinLeadGen',
    file: 'src/components/templates/template_codes/linkedinLeadGen.ts',
  },
];

class TemplateValidator {
  constructor(templateConfig) {
    this.config = templateConfig;
    this.filePath = path.join(__dirname, templateConfig.file);
    this.content = null;
    this.exports = null;
    this.errors = [];
    this.warnings = [];
    this.passedTests = 0;
    this.failedTests = 0;
  }

  async load() {
    try {
      this.content = fs.readFileSync(this.filePath, 'utf-8');
      success(`Loaded template: ${this.config.id}`);
      return true;
    } catch (err) {
      error(`Failed to load template: ${this.config.id} - ${err.message}`);
      return false;
    }
  }

  validateStructure() {
    info(`  Validating structure for ${this.config.id}...`);

    // Check for templateCode export
    if (!this.content.includes('export const templateCode')) {
      this.errors.push('Missing templateCode export');
      error('    Missing templateCode export');
    } else {
      success('    Has templateCode export');
      this.passedTests++;
    }

    // Check for metadata export
    if (!this.content.includes('export const metadata')) {
      this.warnings.push('Missing metadata export');
      warnMsg('    Missing metadata export (recommended but not required)');
    } else {
      success('    Has metadata export');
      this.passedTests++;
    }

    // Check template code is wrapped in backticks
    const templateCodeMatch = this.content.match(/export const templateCode = `([^`]+)`/s);
    if (!templateCodeMatch) {
      this.errors.push('templateCode is not properly wrapped in backticks');
      error('    templateCode not wrapped in backticks');
    } else {
      success('    templateCode properly wrapped');
      this.passedTests++;
    }

    // Check for BubbleFlow class
    if (!this.content.includes('extends BubbleFlow')) {
      this.errors.push('Missing BubbleFlow class extension');
      error('    Missing BubbleFlow class');
    } else {
      success('    Has BubbleFlow class');
      this.passedTests++;
    }

    // Check for handle method
    if (!this.content.includes('async handle(')) {
      this.errors.push('Missing async handle method');
      error('    Missing handle method');
    } else {
      success('    Has async handle method');
      this.passedTests++;
    }

    return this.errors.length === 0;
  }

  validateInputSchema() {
    info(`  Validating input schema for ${this.config.id}...`);

    // Extract metadata JSON
    const metadataMatch = this.content.match(/export const metadata = ({[^;]+});/s);
    if (!metadataMatch) {
      this.warnings.push('No metadata found to validate schema');
      return false;
    }

    try {
      // Check for inputsSchema
      if (this.content.includes('inputsSchema')) {
        success('    Has inputsSchema');
        this.passedTests++;

        // Validate JSON structure
        const schemaMatch = this.content.match(/inputsSchema: JSON\.stringify\(({[^}]+})\)/s);
        if (schemaMatch) {
          try {
            const schema = JSON.parse(schemaMatch[1]);
            if (schema.type && schema.properties) {
              success('    Valid JSON schema structure');
              this.passedTests++;
            } else {
              this.errors.push('Invalid schema structure');
              error('    Schema missing type or properties');
            }
          } catch (e) {
            this.errors.push('Invalid JSON in schema');
            error('    Schema JSON is invalid');
          }
        }
      } else {
        this.warnings.push('No inputsSchema defined');
      }
    } catch (e) {
      this.errors.push(`Failed to parse metadata: ${e.message}`);
      error(`    Metadata parse error: ${e.message}`);
    }

    return this.errors.length === 0;
  }

  validateCredentials() {
    info(`  Validating credentials for ${this.config.id}...`);

    const { expectedCredentials } = this.config;

    if (!this.content.includes('requiredCredentials')) {
      this.warnings.push('No requiredCredentials defined');
      return false;
    }

    let allFound = true;
    expectedCredentials.forEach(cred => {
      if (this.content.includes(`'${cred}'`) || this.content.includes(`"${cred}"`)) {
        success(`    ✓ Credential: ${cred}`);
        this.passedTests++;
      } else {
        error(`    ✗ Missing credential: ${cred}`);
        this.errors.push(`Missing credential: ${cred}`);
        allFound = false;
      }
    });

    return allFound;
  }

  validateBubbles() {
    info(`  Validating bubble usage for ${this.config.id}...`);

    const { expectedBubbles } = this.config;

    let allFound = true;
    expectedBubbles.forEach(bubble => {
      // Check if bubble is imported
      const importPattern = new RegExp(`import.*${bubble}`);
      const usagePattern = new RegExp(`new ${bubble}\\(`);

      const hasImport = importPattern.test(this.content);
      const hasUsage = usagePattern.test(this.content);

      if (hasImport && hasUsage) {
        success(`    ✓ Bubble: ${bubble} (imported + used)`);
        this.passedTests++;
      } else if (hasImport) {
        warnMsg(`    ⚠ Bubble: ${bubble} (imported but not used)`);
        this.warnings.push(`Bubble ${bubble} imported but not used`);
      } else if (hasUsage) {
        warnMsg(`    ⚠ Bubble: ${bubble} (used but not imported)`);
        this.warnings.push(`Bubble ${bubble} used but not imported`);
      } else {
        error(`    ✗ Missing bubble: ${bubble}`);
        this.errors.push(`Missing bubble: ${bubble}`);
        allFound = false;
      }
    });

    // Check for proper action() calls
    const actionCalls = (this.content.match(/\.action\(\)/g) || []).length;
    if (actionCalls > 0) {
      success(`    Found ${actionCalls} action() calls`);
      this.passedTests++;
    } else {
      error('    No action() calls found');
      this.errors.push('No action() calls found');
    }

    return allFound;
  }

  validateErrorHandling() {
    info(`  Validating error handling for ${this.config.id}...`);

    // Check for try-catch blocks
    const tryCatchCount = (this.content.match(/try\s*{/g) || []).length;
    if (tryCatchCount > 0) {
      success(`    Found ${tryCatchCount} try-catch blocks`);
      this.passedTests++;
    } else {
      warnMsg('    No try-catch blocks found');
      this.warnings.push('Consider adding error handling with try-catch');
    }

    // Check for error throwing
    const throwCount = (this.content.match(/throw new Error\(/g) || []).length;
    if (throwCount > 0) {
      success(`    Found ${throwCount} error throws`);
      this.passedTests++;
    } else {
      warnMsg('    No error throwing found');
      this.warnings.push('Consider throwing errors for invalid inputs');
    }

    // Check for success/error response handling
    if (this.content.includes('result.success')) {
      success('    Checks result.success');
      this.passedTests++;
    } else {
      warnMsg('    Does not check result.success');
      this.warnings.push('Consider checking result.success');
    }

    return true;
  }

  validateOutput() {
    info(`  Validating output interface for ${this.config.id}...`);

    // Check for Output interface
    if (this.content.includes('export interface Output')) {
      success('    Has Output interface');
      this.passedTests++;
    } else {
      warnMsg('    No Output interface defined');
      this.warnings.push('Consider defining an Output interface');
    }

    // Check for return statement
    if (this.content.includes('return {')) {
      success('    Has return statement');
      this.passedTests++;
    } else {
      error('    No return statement found');
      this.errors.push('Missing return statement');
    }

    return true;
  }

  validateLogging() {
    info(`  Validating logging for ${this.config.id}...`);

    // Check for logger usage
    if (this.content.includes('this.logger')) {
      success('    Uses logger');
      this.passedTests++;
    } else {
      warnMsg('    No logging found');
      this.warnings.push('Consider adding logging for better debugging');
    }

    return true;
  }

  async validate() {
    log(colors.cyan, `\n${'='.repeat(60)}`);
    log(colors.cyan, `Validating Template: ${this.config.id}`);
    log(colors.cyan, `${'='.repeat(60)}`);

    if (!(await this.load())) {
      return false;
    }

    this.validateStructure();
    this.validateInputSchema();
    this.validateCredentials();
    this.validateBubbles();
    this.validateErrorHandling();
    this.validateOutput();
    this.validateLogging();

    return this.errors.length === 0;
  }

  getResults() {
    return {
      id: this.config.id,
      passed: this.passedTests,
      failed: this.failedTests,
      errors: this.errors,
      warnings: this.warnings,
      success: this.errors.length === 0,
    };
  }
}

// Compare with reference templates
function compareWithReference() {
  log(colors.magenta, `\n${'='.repeat(60)}`);
  log(colors.magenta, 'Comparing with Reference Templates');
  log(colors.magenta, `${'='.repeat(60)}`);

  REFERENCE_TEMPLATES.forEach(ref => {
    info(`\nChecking reference template: ${ref.id}`);
    const refPath = path.join(__dirname, ref.file);
    const exists = fs.existsSync(refPath);

    if (exists) {
      success(`  Reference template exists: ${ref.id}`);
      const refContent = fs.readFileSync(refPath, 'utf-8');

      // Check for common patterns
      const hasMetadata = refContent.includes('export const metadata');
      const hasPreValidatedBubbles = refContent.includes('preValidatedBubbles');
      const hasInputsSchema = refContent.includes('inputsSchema');

      info(`  - Has metadata: ${hasMetadata ? '✓' : '✗'}`);
      info(`  - Has preValidatedBubbles: ${hasPreValidatedBubbles ? '✓' : '✗'}`);
      info(`  - Has inputsSchema: ${hasInputsSchema ? '✓' : '✗'}`);
    } else {
      warnMsg(`  Reference template missing: ${ref.id}`);
    }
  });
}

// Main execution
async function main() {
  log(colors.cyan, '\n╔════════════════════════════════════════════════════════════╗');
  log(colors.cyan, '║     Template Validation Test Suite                        ║');
  log(colors.cyan, '╚════════════════════════════════════════════════════════════╝');

  const results = [];

  // Test each template
  for (const templateConfig of TEMPLATES_TO_TEST) {
    const validator = new TemplateValidator(templateConfig);
    const isValid = await validator.validate();
    results.push(validator.getResults());
  }

  // Compare with reference templates
  compareWithReference();

  // Summary
  log(colors.magenta, `\n${'='.repeat(60)}`);
  log(colors.magenta, 'VALIDATION SUMMARY');
  log(colors.magenta, `${'='.repeat(60)}\n`);

  let totalPassed = 0;
  let totalErrors = 0;
  let totalWarnings = 0;

  results.forEach(result => {
    if (result.success) {
      success(`${result.id}: PASSED (${result.passed} checks)`);
    } else {
      error(`${result.id}: FAILED (${result.errors.length} errors)`);
    }

    totalPassed += result.passed;
    totalErrors += result.errors.length;
    totalWarnings += result.warnings.length;

    if (result.errors.length > 0) {
      info('\nErrors:');
      result.errors.forEach(err => error(`  - ${err}`));
    }

    if (result.warnings.length > 0) {
      info('\nWarnings:');
      result.warnings.forEach(w => warnMsg(`  - ${w}`));
    }
    info('');
  });

  log(colors.cyan, `${'='.repeat(60)}`);
  info(`Total Checks Passed: ${totalPassed}`);
  if (totalErrors > 0) {
    error(`Total Errors: ${totalErrors}`);
  }
  if (totalWarnings > 0) {
    warnMsg(`Total Warnings: ${totalWarnings}`);
  }
  log(colors.cyan, `${'='.repeat(60)}`);

  // Exit with appropriate code
  const exitCode = totalErrors > 0 ? 1 : 0;
  process.exit(exitCode);
}

// Run the tests
main().catch(err => {
  error(`\nFatal error: ${err.message}`);
  console.error(err);
  process.exit(1);
});
