/**
 * HTTP Bug Fix Validation Test
 * Tests the fix for the z.union().max() bug
 */
import { z } from 'zod';
// Recreate the fixed schema for testing
const FixedHttpBodySchema = z
    .union([z.string(), z.record(z.unknown())])
    .refine((val) => {
    // Check body size: strings by length, objects by JSON string length
    if (typeof val === 'string') {
        return val.length <= 10485760; // 10MB
    }
    return JSON.stringify(val).length <= 10485760; // 10MB
}, 'Request body exceeds maximum size of 10MB')
    .optional();
function testHttpBodyValidation() {
    console.log('Testing HTTP Body Validation Fix...\n');
    let passed = 0;
    let failed = 0;
    // Test 1: Valid small string
    try {
        const result = FixedHttpBodySchema.parse('small body');
        console.log('✓ Test 1 PASSED: Small string accepted');
        passed++;
    }
    catch (error) {
        console.log('✗ Test 1 FAILED:', error.message);
        failed++;
    }
    // Test 2: Valid small object
    try {
        const result = FixedHttpBodySchema.parse({ key: 'value', num: 123 });
        console.log('✓ Test 2 PASSED: Small object accepted');
        passed++;
    }
    catch (error) {
        console.log('✗ Test 2 FAILED:', error.message);
        failed++;
    }
    // Test 3: Optional (undefined)
    try {
        const result = FixedHttpBodySchema.parse(undefined);
        console.log('✓ Test 3 PASSED: Undefined/optional accepted');
        passed++;
    }
    catch (error) {
        console.log('✗ Test 3 FAILED:', error.message);
        failed++;
    }
    // Test 4: Null (should be rejected by refine)
    try {
        const result = FixedHttpBodySchema.parse(null);
        console.log('✗ Test 4 FAILED: Null should be rejected');
        failed++;
    }
    catch (error) {
        console.log('✓ Test 4 PASSED: Null correctly rejected');
        passed++;
    }
    // Test 5: Large string (> 10MB)
    try {
        const largeString = 'x'.repeat(10485761); // 10MB + 1 byte
        const result = FixedHttpBodySchema.parse(largeString);
        console.log('✗ Test 5 FAILED: Large string should be rejected');
        failed++;
    }
    catch (error) {
        console.log('✓ Test 5 PASSED: Large string correctly rejected');
        passed++;
    }
    // Test 6: Large object (> 10MB when stringified)
    try {
        const largeObject = { data: 'x'.repeat(10485751) };
        const result = FixedHttpBodySchema.parse(largeObject);
        console.log('✗ Test 6 FAILED: Large object should be rejected');
        failed++;
    }
    catch (error) {
        console.log('✓ Test 6 PASSED: Large object correctly rejected');
        passed++;
    }
    // Test 7: Empty string (valid)
    try {
        const result = FixedHttpBodySchema.parse('');
        console.log('✓ Test 7 PASSED: Empty string accepted');
        passed++;
    }
    catch (error) {
        console.log('✗ Test 7 FAILED:', error.message);
        failed++;
    }
    // Test 8: Empty object (valid)
    try {
        const result = FixedHttpBodySchema.parse({});
        console.log('✓ Test 8 PASSED: Empty object accepted');
        passed++;
    }
    catch (error) {
        console.log('✗ Test 8 FAILED:', error.message);
        failed++;
    }
    console.log('\n' + '='.repeat(50));
    console.log(`Test Results: ${passed} passed, ${failed} failed`);
    console.log('='.repeat(50));
    return { passed, failed, total: passed + failed };
}
// Run the tests
const results = testHttpBodyValidation();
// Exit with appropriate code
process.exit(results.failed > 0 ? 1 : 0);
//# sourceMappingURL=http-fix-validation.js.map