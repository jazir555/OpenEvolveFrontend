/**
 * Datapizza API Contract Tests
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the Datapizza API returns the expected fields.
 * If the contract is violated (Project API changed), the adapter MUST refuse to start.
 *
 * Runs on container startup. If these tests fail, the application should NOT start.
 */
export {};
/**
 * Usage Instructions:
 *
 * 1. Run tests on container startup:
 *    ```bash
 *    npm run test:contract
 *    ```
 *
 * 2. If tests fail, container MUST refuse to start:
 *    ```javascript
 *    try {
 *      await runContractTests();
 *    } catch (error) {
 *      logger.error('Contract tests failed - refusing to start');
 *      process.exit(1);
 *    }
 *    ```
 *
 * 3. Tests verify critical fields that the adapter depends on
 * 4. If API changes, tests fail before corrupting data
 */
//# sourceMappingURL=datapizza-api.test.d.ts.map