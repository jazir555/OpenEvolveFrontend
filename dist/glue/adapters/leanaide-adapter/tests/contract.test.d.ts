/**
 * LeanAide Contract Tests
 *
 * These tests validate the API contracts between the Glue Layer and LeanAide server.
 * Following the Federation Constitution's "Proof of Work" doctrine:
 *
 * 1. FAIL FAST: If contracts are violated, the adapter refuses to start
 * 2. RUNTIME TRUTH: Tests validate actual API behavior, not documentation
 * 3. ZERO TRUST: Every field and response type is explicitly validated
 *
 * Purpose: Prevent LeanAide API changes from breaking the integration silently.
 *
 * Usage:
 *   - Run on adapter startup: If tests fail, adapter startup is blocked
 *   - Run in CI/CD: Prevent deployments with broken contracts
 *   - Run after LeanAide updates: Verify API compatibility
 *
 * @module LeanAideContractTests
 */
/**
 * LeanAide API Client (Mock for contract testing)
 *
 * In production, this would make actual HTTP calls to the LeanAide server.
 * For contract testing, we mock the responses to validate schema compliance.
 */
declare class LeanAideAPIClient {
    private baseUrl;
    private timeout;
    constructor(baseUrl: string, timeout: number);
    /**
     * Mock POST request to LeanAide server
     */
    post(path: string, data: any, options?: any): Promise<any>;
    /**
     * Mock proof verification response
     */
    private mockVerifyResponse;
    /**
     * Mock compilation response
     */
    private mockCompileResponse;
}
/**
 * Export test runner for container startup
 *
 * This function can be called during adapter startup to validate contracts.
 * If tests fail, the adapter should refuse to start.
 */
export declare function runContractTests(): Promise<boolean>;
export default LeanAideAPIClient;
//# sourceMappingURL=contract.test.d.ts.map