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
interface TestResult {
    name: string;
    passed: boolean;
    duration_ms: number;
    error?: string;
    details?: any;
}
interface TestSuite {
    name: string;
    passed: number;
    failed: number;
    duration_ms: number;
    tests: TestResult[];
}
interface ContractTestReport {
    timestamp: string;
    environment: string;
    api_url: string;
    total_suites: number;
    total_passed: number;
    total_failed: number;
    duration_ms: number;
    success: boolean;
    suites: TestSuite[];
}
/**
 * Run all contract tests and generate report
 */
export declare function runContractTests(): Promise<ContractTestReport>;
export default runContractTests;
//# sourceMappingURL=contract-runner.d.ts.map