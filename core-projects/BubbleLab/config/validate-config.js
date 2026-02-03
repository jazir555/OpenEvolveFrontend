#!/usr/bin/env node

/**
 * OpenEvolve BubbleLab - Configuration Validation Script
 * ==============================================================================
 *
 * This script validates environment configuration before deployment to prevent
 * security issues and configuration errors.
 *
 * SECURITY FEATURES:
 * - Validates required environment variables are set
 * - Checks JWT secret format and length (min 32 characters)
 * - Validates TLS certificate paths exist
 * - Detects example domains that need replacement
 * - Validates database connection strings format
 * - Checks for hardcoded credentials
 *
 * USAGE:
 *   node validate-config.js                    # Validate current environment
 *   node validate-config.js --env staging      # Validate specific environment
 *   node validate-config.js --strict           # Fail on warnings
 *
 * EXIT CODES:
 *   0 - All validations passed
 *   1 - Critical security issues found
 *   2 - Warnings found (only in --strict mode)
 *
 * LAST_UPDATED: 2026-01-17T00:00:00Z
 * VERSION: 1.0.0
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

/**
 * Logger utility with color support
 */
const logger = {
  error: (msg) => console.error(`${colors.red}❌ CRITICAL:${colors.reset} ${msg}`),
  warn: (msg) => console.warn(`${colors.yellow}⚠️  WARNING:${colors.reset} ${msg}`),
  info: (msg) => console.log(`${colors.blue}ℹ️  INFO:${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✅ SUCCESS:${colors.reset} ${msg}`),
  header: (msg) => console.log(`\n${colors.bright}${colors.cyan}${msg}${colors.reset}`),
};

/**
 * Validation results tracker
 */
const validationResults = {
  critical: [],
  warnings: [],
  passed: [],
};

/**
 * Mark a validation check
 */
function markResult(type, message) {
  validationResults[type].push(message);
  if (type === 'critical') {
    logger.error(message);
  } else if (type === 'warnings') {
    logger.warn(message);
  } else {
    logger.success(message);
  }
}

/**
 * Check if environment variable is set and non-empty
 */
function validateRequiredEnvVar(envVar, description = '') {
  const value = process.env[envVar];
  if (!value || value.trim() === '') {
    markResult('critical', `Missing required environment variable: ${envVar}${description ? ` (${description})` : ''}`);
    return false;
  }
  markResult('passed', `Environment variable set: ${envVar}`);
  return true;
}

/**
 * Validate JWT secret format and length
 */
function validateJWTSecret(secret, envVarName = 'JWT_SECRET') {
  if (!secret) {
    markResult('critical', `${envVarName} is not set`);
    return false;
  }

  if (secret.length < 32) {
    markResult('critical', `${envVarName} must be at least 32 characters (current: ${secret.length})`);
    return false;
  }

  // Check for obvious weak secrets
  const weakPatterns = [
    'password', 'secret', 'changeme', 'example', 'test', 'demo',
    'jwt-secret', 'super-secret', 'your-super-secret',
  ];

  const lowerSecret = secret.toLowerCase();
  const hasWeakPattern = weakPatterns.some(pattern => lowerSecret.includes(pattern));

  if (hasWeakPattern) {
    markResult('critical', `${envVarName} contains weak/common patterns`);
    return false;
  }

  // Check entropy (basic check for randomness)
  const uniqueChars = new Set(secret).size;
  const entropyRatio = uniqueChars / secret.length;

  if (entropyRatio < 0.4) {
    markResult('warnings', `${envVarName} has low entropy (${(entropyRatio * 100).toFixed(1)}% unique chars). Use: openssl rand -base64 32`);
    return true;
  }

  markResult('passed', `${envVarName} is strong and properly formatted`);
  return true;
}

/**
 * Validate database connection string format
 */
function validateDatabaseURL(url, envVarName = 'DATABASE_URL') {
  if (!url) {
    markResult('critical', `${envVarName} is not set`);
    return false;
  }

  // Check for postgresql:// prefix
  if (!url.startsWith('postgresql://') && !url.startsWith('postgres://')) {
    markResult('critical', `${envVarName} must use postgresql:// protocol`);
    return false;
  }

  // Check for example credentials
  if (url.includes('password@') || url.includes('changeme@') || url.includes('devpassword@')) {
    markResult('critical', `${envVarName} contains example/default credentials`);
    return false;
  }

  // Parse and validate components
  try {
    const urlObj = new URL(url.startsWith('postgres://') ? url.replace('postgres://', 'postgresql://') : url);

    if (!urlObj.hostname || urlObj.hostname === 'localhost') {
      if (process.env.NODE_ENV === 'production') {
        markResult('critical', `${envVarName} uses localhost in production`);
        return false;
      }
      markResult('warnings', `${envVarName} uses localhost (OK for development)`);
    }

    if (!urlObj.username) {
      markResult('critical', `${envVarName} missing username`);
      return false;
    }

    if (!urlObj.password) {
      markResult('critical', `${envVarName} missing password`);
      return false;
    }

    if (!urlObj.pathname || urlObj.pathname === '/') {
      markResult('critical', `${envVarName} missing database name`);
      return false;
    }

    markResult('passed', `${envVarName} format is valid`);
    return true;
  } catch (error) {
    markResult('critical', `${envVarName} has invalid URL format: ${error.message}`);
    return false;
  }
}

/**
 * Validate TLS certificate files exist
 */
function validateTLSCertificates() {
  const certPath = process.env.TLS_CERT_PATH;
  const keyPath = process.env.TLS_KEY_PATH;
  const caPath = process.env.TLS_CA_PATH;

  if (!certPath && !keyPath) {
    logger.info('TLS not configured (optional for development)');
    return true;
  }

  let allValid = true;

  if (certPath) {
    if (fs.existsSync(certPath)) {
      markResult('passed', `TLS certificate exists: ${certPath}`);
    } else {
      markResult('critical', `TLS certificate not found: ${certPath}`);
      allValid = false;
    }
  }

  if (keyPath) {
    if (fs.existsSync(keyPath)) {
      markResult('passed', `TLS key exists: ${keyPath}`);
    } else {
      markResult('critical', `TLS key not found: ${keyPath}`);
      allValid = false;
    }
  }

  if (caPath) {
    if (fs.existsSync(caPath)) {
      markResult('passed', `TLS CA certificate exists: ${caPath}`);
    } else {
      markResult('critical', `TLS CA certificate not found: ${caPath}`);
      allValid = false;
    }
  }

  return allValid;
}

/**
 * Detect example domains in configuration
 */
function detectExampleDomains() {
  const exampleDomainPatterns = [
    'openevolve.example.com',
    'example.com',
    'staging.bubblelab.example.com',
  ];

  let foundExamples = false;

  // Check common environment variables that might contain domains
  const domainEnvVars = [
    'OPENEVOLVE_API_URL',
    'API_BASE_URL',
    'APP_BASE_URL',
    'LENAIDE_CONTINUOUS_URL',
    'KNOWLEDGE_ENGINE_URL',
    'DECOMPOSITION_ENGINE_URL',
  ];

  domainEnvVars.forEach(envVar => {
    const value = process.env[envVar];
    if (value) {
      exampleDomainPatterns.forEach(pattern => {
        if (value.includes(pattern)) {
          markResult('warnings', `${envVar} contains example domain: ${pattern}`);
          foundExamples = true;
        }
      });
    }
  });

  if (!foundExamples) {
    markResult('passed', 'No example domains found in environment variables');
  }

  return !foundExamples;
}

/**
 * Validate API keys are set (for production)
 */
function validateAPIKeys() {
  const requiredAPIKeys = [
    'ANTHROPIC_API_KEY',
    'OPENAI_API_KEY',
  ];

  const optionalAPIKeys = [
    'GOOGLE_API_KEY',
    'OPENROUTER_API_KEY',
    'DEEPSEEK_API_KEY',
  ];

  let allValid = true;

  requiredAPIKeys.forEach(key => {
    if (!validateRequiredEnvVar(key, 'API Key')) {
      allValid = false;
    }
  });

  optionalAPIKeys.forEach(key => {
    const value = process.env[key];
    if (!value || value.trim() === '') {
      markResult('warnings', `Optional API key not set: ${key}`);
    } else {
      markResult('passed', `API key set: ${key}`);
    }
  });

  return allValid;
}

/**
 * Validate environment-specific settings
 */
function validateEnvironmentSpecific() {
  const environment = process.env.NODE_ENV || process.env.ENVIRONMENT || 'development';

  logger.header(`Validating ${environment.toUpperCase()} environment configuration...`);

  if (environment === 'production') {
    // Production-specific validations
    if (process.env.DEBUG_MODE === 'true') {
      markResult('critical', 'DEBUG_MODE cannot be true in production');
    }

    if (process.env.DB_AUTO_MIGRATION === 'true') {
      markResult('critical', 'DB_AUTO_MIGRATION should be false in production');
    }

    if (process.env.DISABLE_AUTH === 'true') {
      markResult('critical', 'DISABLE_AUTH cannot be true in production');
    }

    if (process.env.TLS_ENABLED !== 'true') {
      markResult('critical', 'TLS_ENABLED must be true in production');
    }
  }

  return true;
}

/**
 * Main validation function
 */
function validateConfiguration() {
  logger.header('OpenEvolve BubbleLab Configuration Validation');
  logger.header('================================================');

  // Get environment from command line or default to NODE_ENV
  const args = process.argv.slice(2);
  const envIndex = args.indexOf('--env');
  const environment = envIndex >= 0 ? args[envIndex + 1] : process.env.NODE_ENV || 'development';
  const strictMode = args.includes('--strict');

  logger.info(`Environment: ${environment.toUpperCase()}`);
  if (strictMode) {
    logger.info('Strict mode: ENABLED (warnings will fail validation)');
  }
  console.log('');

  // Run all validations
  logger.header('1. Validating Required Environment Variables');
  const requiredVars = [
    ['DATABASE_URL', 'Primary database connection'],
    ['REDIS_URL', 'Cache connection'],
    ['CLERK_SECRET_KEY', 'Authentication'],
    ['ANTHROPIC_API_KEY', 'AI service'],
    ['OPENAI_API_KEY', 'AI service'],
  ];

  let hasCritical = false;
  requiredVars.forEach(([varName, description]) => {
    if (!validateRequiredEnvVar(varName, description)) {
      hasCritical = true;
    }
  });

  logger.header('2. Validating Security Secrets');
  if (!validateJWTSecret(process.env.JWT_SECRET, 'JWT_SECRET')) hasCritical = true;
  if (!validateJWTSecret(process.env.SESSION_SECRET, 'SESSION_SECRET')) hasCritical = true;
  if (!validateJWTSecret(process.env.CSRF_SECRET, 'CSRF_SECRET')) hasCritical = true;
  if (!validateJWTSecret(process.env.CREDENTIAL_ENCRYPTION_KEY, 'CREDENTIAL_ENCRYPTION_KEY')) hasCritical = true;

  logger.header('3. Validating Database URLs');
  if (!validateDatabaseURL(process.env.DATABASE_URL, 'DATABASE_URL')) hasCritical = true;
  if (process.env.KNOWLEDGE_GRAPH_DATABASE_URL) {
    validateDatabaseURL(process.env.KNOWLEDGE_GRAPH_DATABASE_URL, 'KNOWLEDGE_GRAPH_DATABASE_URL');
  }
  if (process.env.ANALYTICS_DATABASE_URL) {
    validateDatabaseURL(process.env.ANALYTICS_DATABASE_URL, 'ANALYTICS_DATABASE_URL');
  }

  logger.header('4. Validating TLS Certificates');
  if (!validateTLSCertificates()) hasCritical = true;

  logger.header('5. Detecting Example Domains');
  detectExampleDomains();

  logger.header('6. Validating API Keys');
  if (!validateAPIKeys()) hasCritical = true;

  logger.header('7. Validating Environment-Specific Settings');
  validateEnvironmentSpecific();

  // Print summary
  logger.header('================================================');
  logger.header('VALIDATION SUMMARY');
  logger.header('================================================');

  console.log(`${colors.green}✅ Passed:${colors.reset} ${validationResults.passed.length}`);
  console.log(`${colors.yellow}⚠️  Warnings:${colors.reset} ${validationResults.warnings.length}`);
  console.log(`${colors.red}❌ Critical:${colors.reset} ${validationResults.critical.length}`);
  console.log('');

  // Exit with appropriate code
  if (validationResults.critical.length > 0) {
    logger.error('Configuration validation FAILED with critical security issues');
    logger.info('Please fix the critical issues above before deploying.');
    process.exit(1);
  }

  if (validationResults.warnings.length > 0 && strictMode) {
    logger.error('Configuration validation FAILED due to strict mode');
    logger.info('Please fix the warnings above or run without --strict flag.');
    process.exit(2);
  }

  if (validationResults.warnings.length > 0) {
    logger.warn('Configuration validation passed with warnings');
    logger.info('Recommendation: Review and fix warnings before production deployment');
  } else {
    logger.success('Configuration validation PASSED');
    logger.info('All security checks passed. Ready for deployment.');
  }

  process.exit(0);
}

// Run validation
validateConfiguration();
