"use strict";
/**
 * LoongFlow Contract Test Runner
 *
 * Standalone script to run all contract tests and generate detailed reports.
 * Can be used as:
 * 1. Standalone validation script: `ts-node tests/contract-runner.ts`
 * 2. Docker health check
 * 3. CI/CD gate before deployment
 * 4. Startup validation in adapter initialization
 *
 * Purpose: Phase 2 - The Contract (Defense)
 * Law of Runtime Truth: Validates against actual LoongFlow API
 *
 * Exit codes:
 *   0 - All contracts passed
 *   1 - One or more contracts failed
 *   2 - Configuration error (missing env vars)
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runContractTests = runContractTests;
const contract_test_1 = require("./contract.test");
const axios_1 = __importDefault(require("axios"));
// =============================================================================
// CONFIGURATION
// =============================================================================
const API_URL = process.env.LOONGFLOW_API_URL || 'http://localhost:8000';
const TIMEOUT_MS = parseInt(process.env.LOONGFLOW_TIMEOUT_MS || '30000', 10);
const VERBOSE = process.env.VERBOSE === 'true';
// =============================================================================
// CONTRACT TEST EXECUTION
// =============================================================================
/**
 * Execute health check contract test
 */
async function testHealthCheck() {
    const start = Date.now();
    try {
        const response = await axios_1.default.get(`${API_URL}/health`, { timeout: TIMEOUT_MS });
        if (response.status !== 200) {
            return {
                name: 'Health Check',
                passed: false,
                duration_ms: Date.now() - start,
                error: `Unexpected status code: ${response.status}`,
            };
        }
        if (!response.data?.status || !['healthy', 'ok'].includes(response.data.status)) {
            return {
                name: 'Health Check',
                passed: false,
                duration_ms: Date.now() - start,
                error: `Invalid health status: ${response.data?.status}`,
                details: response.data,
            };
        }
        // Validate timestamp format if present
        if (response.data?.timestamp) {
            const timestamp = response.data.timestamp;
            if (!timestamp.endsWith('Z')) {
                return {
                    name: 'Health Check',
                    passed: false,
                    duration_ms: Date.now() - start,
                    error: `Timestamp not in UTC format: ${timestamp}`,
                    details: response.data,
                };
            }
        }
        return {
            name: 'Health Check',
            passed: true,
            duration_ms: Date.now() - start,
            details: response.data,
        };
    }
    catch (error) {
        return {
            name: 'Health Check',
            passed: false,
            duration_ms: Date.now() - start,
            error: error.message,
            details: {
                code: error.code,
                status: error.response?.status,
            },
        };
    }
}
/**
 * Validate environment configuration
 */
function testEnvironmentConfiguration() {
    const start = Date.now();
    try {
        const errors = [];
        // Check LOONGFLOW_API_URL
        if (!process.env.LOONGFLOW_API_URL) {
            errors.push('LOONGFLOW_API_URL is not set');
        }
        else if (process.env.LOONGFLOW_API_URL === 'http://localhost:8000') {
            errors.push('LOONGFLOW_API_URL is using default value (should be explicitly set)');
        }
        else {
            try {
                new URL(process.env.LOONGFLOW_API_URL);
            }
            catch {
                errors.push('LOONGFLOW_API_URL is not a valid URL');
            }
        }
        // Check LOONGFLOW_TIMEOUT_MS
        if (!process.env.LOONGFLOW_TIMEOUT_MS) {
            errors.push('LOONGFLOW_TIMEOUT_MS is not set');
        }
        else {
            const timeout = parseInt(process.env.LOONGFLOW_TIMEOUT_MS, 10);
            if (isNaN(timeout) || timeout <= 0) {
                errors.push('LOONGFLOW_TIMEOUT_MS must be a positive number');
            }
        }
        if (errors.length > 0) {
            return {
                name: 'Environment Configuration',
                passed: false,
                duration_ms: Date.now() - start,
                error: errors.join('; '),
            };
        }
        return {
            name: 'Environment Configuration',
            passed: true,
            duration_ms: Date.now() - start,
            details: {
                api_url: API_URL,
                timeout_ms: TIMEOUT_MS,
            },
        };
    }
    catch (error) {
        return {
            name: 'Environment Configuration',
            passed: false,
            duration_ms: Date.now() - start,
            error: error.message,
        };
    }
}
/**
 * Validate all fixture contracts (offline tests)
 */
function testFixtureContracts() {
    const start = Date.now();
    try {
        (0, contract_test_1.validateAllContracts)();
        return {
            name: 'Fixture Contracts',
            passed: true,
            duration_ms: Date.now() - start,
            details: {
                message: 'All fixture contracts validated successfully',
            },
        };
    }
    catch (error) {
        return {
            name: 'Fixture Contracts',
            passed: false,
            duration_ms: Date.now() - start,
            error: error.message,
        };
    }
}
// =============================================================================
// MAIN RUNNER
// =============================================================================
/**
 * Run all contract tests and generate report
 */
async function runContractTests() {
    const startTime = Date.now();
    const suites = [];
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║     LoongFlow Adapter - Contract Test Runner              ║');
    console.log('╚════════════════════════════════════════════════════════════╝');
    console.log('');
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`API URL: ${API_URL}`);
    console.log(`Timeout: ${TIMEOUT_MS}ms`);
    console.log('');
    // Suite 1: Environment Configuration
    console.log('📋 Testing Environment Configuration...');
    const envResult = testEnvironmentConfiguration();
    suites.push({
        name: 'Environment Configuration',
        passed: envResult.passed ? 1 : 0,
        failed: envResult.passed ? 0 : 1,
        duration_ms: envResult.duration_ms,
        tests: [envResult],
    });
    console.log(`   ${envResult.passed ? '✅' : '❌'} ${envResult.name}: ${envResult.passed ? 'PASSED' : 'FAILED'}`);
    if (envResult.error) {
        console.log(`   Error: ${envResult.error}`);
    }
    console.log('');
    // Suite 2: Fixture Contracts (offline)
    console.log('📝 Testing Fixture Contracts...');
    const fixtureResult = testFixtureContracts();
    suites.push({
        name: 'Fixture Contracts',
        passed: fixtureResult.passed ? 1 : 0,
        failed: fixtureResult.passed ? 0 : 1,
        duration_ms: fixtureResult.duration_ms,
        tests: [fixtureResult],
    });
    console.log(`   ${fixtureResult.passed ? '✅' : '❌'} ${fixtureResult.name}: ${fixtureResult.passed ? 'PASSED' : 'FAILED'}`);
    if (fixtureResult.error) {
        console.log(`   Error: ${fixtureResult.error}`);
    }
    console.log('');
    // Suite 3: API Health Check (if env is valid)
    if (envResult.passed) {
        console.log('🌐 Testing API Connectivity...');
        const healthResult = await testHealthCheck();
        suites.push({
            name: 'API Connectivity',
            passed: healthResult.passed ? 1 : 0,
            failed: healthResult.passed ? 0 : 1,
            duration_ms: healthResult.duration_ms,
            tests: [healthResult],
        });
        console.log(`   ${healthResult.passed ? '✅' : '❌'} ${healthResult.name}: ${healthResult.passed ? 'PASSED' : 'FAILED'}`);
        if (healthResult.error) {
            console.log(`   Error: ${healthResult.error}`);
        }
        console.log('');
    }
    // Calculate totals
    const totalPassed = suites.reduce((sum, suite) => sum + suite.passed, 0);
    const totalFailed = suites.reduce((sum, suite) => sum + suite.failed, 0);
    const totalDuration = Date.now() - startTime;
    const success = totalFailed === 0;
    // Generate report
    const report = {
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development',
        api_url: API_URL,
        total_suites: suites.length,
        total_passed: totalPassed,
        total_failed: totalFailed,
        duration_ms: totalDuration,
        success,
        suites,
    };
    // Print summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║                     Test Summary                           ║');
    console.log('╚════════════════════════════════════════════════════════════╝');
    console.log('');
    console.log(`Total Suites: ${report.total_suites}`);
    console.log(`Total Passed: ${report.total_passed}`);
    console.log(`Total Failed: ${report.total_failed}`);
    console.log(`Duration: ${report.duration_ms}ms`);
    console.log(`Status: ${success ? '✅ SUCCESS' : '❌ FAILED'}`);
    console.log('');
    if (VERBOSE && !success) {
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║                     Failed Tests                          ║');
        console.log('╚════════════════════════════════════════════════════════════╝');
        console.log('');
        suites.forEach(suite => {
            suite.tests.filter(t => !t.passed).forEach(test => {
                console.log(`❌ ${test.name}`);
                console.log(`   Error: ${test.error}`);
                if (test.details) {
                    console.log(`   Details: ${JSON.stringify(test.details, null, 2)}`);
                }
                console.log('');
            });
        });
    }
    return report;
}
// =============================================================================
// CLI INTERFACE
// =============================================================================
/**
 * Main entry point when run as standalone script
 */
async function main() {
    try {
        const report = await runContractTests();
        // Print JSON report if requested
        if (process.env.JSON_OUTPUT === 'true') {
            console.log('\n📄 JSON Report:');
            console.log(JSON.stringify(report, null, 2));
        }
        // Exit with appropriate code
        if (!report.success) {
            console.error('\n❌ Contract tests failed!');
            console.error('\nAction Required:');
            console.error('1. Verify LoongFlow API is accessible');
            console.error('2. Check environment variables are set correctly');
            console.error('3. Review fixture data matches API responses');
            console.error('4. Update contracts if API has changed');
            process.exit(1);
        }
        console.log('\n✅ All contract tests passed!');
        process.exit(0);
    }
    catch (error) {
        console.error('\n❌ Fatal error running contract tests:', error.message);
        if (VERBOSE) {
            console.error(error.stack);
        }
        process.exit(2);
    }
}
// Run if executed directly
if (require.main === module) {
    main();
}
exports.default = runContractTests;
//# sourceMappingURL=contract-runner.js.map