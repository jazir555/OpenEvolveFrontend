"use strict";
/**
 * CONTRACT TESTS - Unified Verification Orchestrator
 *
 * Following Federation Constitution Section 4: The Proof of Work
 *
 * These tests verify that:
 * - Z3 API returns expected fields
 * - LeanAide API returns expected fields
 * - Cross-validation workflow works end-to-end
 * - Canonical schemas are correctly enforced
 *
 * If these tests fail, the adapter REFUSES TO START to prevent data corruption.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const axios_1 = __importDefault(require("axios"));
const canonical_1 = require("../src/canonical");
// Configuration from environment
const Z3_URL = process.env.Z3_URL || 'http://localhost:8080';
const LEANAIDE_URL = process.env.LEANAIDE_URL || 'http://localhost:8081';
(0, globals_1.describe)('Contract Tests: Z3 API', () => {
    (0, globals_1.test)('Z3 health endpoint returns 200', async () => {
        const response = await axios_1.default.get(`${Z3_URL}/health`, { timeout: 5000 });
        (0, globals_1.expect)(response.status).toBe(200);
    });
    (0, globals_1.test)('Z3 verify endpoint accepts valid request', async () => {
        const payload = {
            problem: {
                type: 'SMT_CONSTRAINTS',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)',
                description: 'Simple constraint: x > 0'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 10000 });
        // Verify response structure
        (0, globals_1.expect)(response.data).toBeDefined();
        (0, globals_1.expect)(typeof response.data.verified).toBe('boolean');
        (0, globals_1.expect)(typeof response.data.confidence).toBe('number');
        (0, globals_1.expect)(typeof response.data.output).toBe('string');
    });
    (0, globals_1.test)('Z3 response contains all required fields', async () => {
        const payload = {
            problem: {
                type: 'SMT_CONSTRAINTS',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)',
                description: 'Simple constraint: x > 0'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 10000 });
        // Required fields per VerificationResult schema
        (0, globals_1.expect)(response.data).toHaveProperty('verified');
        (0, globals_1.expect)(response.data).toHaveProperty('confidence');
        (0, globals_1.expect)(response.data).toHaveProperty('output');
        (0, globals_1.expect)(response.data).toHaveProperty('metadata');
        // Metadata must have execution time
        (0, globals_1.expect)(response.data.metadata).toHaveProperty('executionTime');
        (0, globals_1.expect)(typeof response.data.metadata.executionTime).toBe('number');
    });
    (0, globals_1.test)('Z3 confidence score is within valid range', async () => {
        const payload = {
            problem: {
                type: 'SMT_CONSTRAINTS',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)',
                description: 'Simple constraint: x > 0'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 10000 });
        (0, globals_1.expect)(response.data.confidence).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(response.data.confidence).toBeLessThanOrEqual(1);
    });
});
(0, globals_1.describe)('Contract Tests: LeanAide API', () => {
    (0, globals_1.test)('LeanAide health endpoint returns 200', async () => {
        const response = await axios_1.default.get(`${LEANAIDE_URL}/health`, { timeout: 5000 });
        (0, globals_1.expect)(response.status).toBe(200);
    });
    (0, globals_1.test)('LeanAide verify endpoint accepts valid request', async () => {
        const payload = {
            problem: {
                type: 'THEOREM_PROVING',
                statement: 'theorem add_zero (n : Nat) : n + 0 = n := by simp',
                description: 'Simple theorem: n + 0 = n'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 10000 });
        // Verify response structure
        (0, globals_1.expect)(response.data).toBeDefined();
        (0, globals_1.expect)(typeof response.data.verified).toBe('boolean');
        (0, globals_1.expect)(typeof response.data.confidence).toBe('number');
        (0, globals_1.expect)(typeof response.data.output).toBe('string');
    });
    (0, globals_1.test)('LeanAide response contains all required fields', async () => {
        const payload = {
            problem: {
                type: 'THEOREM_PROVING',
                statement: 'theorem add_zero (n : Nat) : n + 0 = n := by simp',
                description: 'Simple theorem: n + 0 = n'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 10000 });
        // Required fields per VerificationResult schema
        (0, globals_1.expect)(response.data).toHaveProperty('verified');
        (0, globals_1.expect)(response.data).toHaveProperty('confidence');
        (0, globals_1.expect)(response.data).toHaveProperty('output');
        (0, globals_1.expect)(response.data).toHaveProperty('metadata');
        // Metadata must have execution time
        (0, globals_1.expect)(response.data.metadata).toHaveProperty('executionTime');
        (0, globals_1.expect)(typeof response.data.metadata.executionTime).toBe('number');
    });
    (0, globals_1.test)('LeanAide confidence score is within valid range', async () => {
        const payload = {
            problem: {
                type: 'THEOREM_PROVING',
                statement: 'theorem add_zero (n : Nat) : n + 0 = n := by simp',
                description: 'Simple theorem: n + 0 = n'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const response = await axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 10000 });
        (0, globals_1.expect)(response.data.confidence).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(response.data.confidence).toBeLessThanOrEqual(1);
    });
});
(0, globals_1.describe)('Contract Tests: Canonical Schemas', () => {
    (0, globals_1.test)('VerificationRequest schema validates correctly', () => {
        const validRequest = {
            requestId: '550e8400-e29b-41d4-a716-446655440000',
            problem: {
                id: '550e8400-e29b-41d4-a716-446655440000',
                type: 'SMT_CONSTRAINTS',
                description: 'Test problem',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium',
                allowedSystems: ['both'],
                requiredConfidence: 0.95
            },
            confidenceRequired: 0.95,
            timestamp: new Date().toISOString()
        };
        const result = canonical_1.CanonicalSchemas.VerificationRequest.safeParse(validRequest);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('VerificationResult schema validates correctly', () => {
        const validResult = {
            system: 'z3',
            verified: true,
            confidence: 0.95,
            output: 'sat',
            proof: '(declare-const x Int) ...',
            metadata: {
                executionTime: 1500,
                memoryUsed: 128,
                strategy: 'z3_only',
                timestamp: new Date().toISOString()
            }
        };
        const result = canonical_1.CanonicalSchemas.VerificationResult.safeParse(validResult);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('CrossValidationResult schema validates correctly', () => {
        const validCrossValidation = {
            requestId: '550e8400-e29b-41d4-a716-446655440000',
            verified: true,
            agreement: true,
            agreementType: 'full_agreement',
            confidence: 0.95,
            systemResults: [
                {
                    system: 'z3',
                    verified: true,
                    confidence: 0.95,
                    output: 'sat',
                    metadata: {
                        executionTime: 1500,
                        strategy: 'parallel',
                        timestamp: new Date().toISOString()
                    }
                },
                {
                    system: 'leanaide',
                    verified: true,
                    confidence: 0.90,
                    output: 'proved',
                    proof: 'theorem proof...',
                    metadata: {
                        executionTime: 2000,
                        strategy: 'parallel',
                        timestamp: new Date().toISOString()
                    }
                }
            ],
            conflicts: [],
            resolution: 'verified',
            strategy: 'parallel',
            metadata: {
                correlationId: 'test-correlation-123',
                totalExecutionTime: 3500,
                timestamp: new Date().toISOString()
            }
        };
        const result = canonical_1.CanonicalSchemas.CrossValidationResult.safeParse(validCrossValidation);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('Schema rejects invalid problem types', () => {
        const invalidRequest = {
            requestId: '550e8400-e29b-41d4-a716-446655440000',
            problem: {
                id: '550e8400-e29b-41d4-a716-446655440000',
                type: 'INVALID_TYPE', // Invalid type
                description: 'Test problem',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium',
                allowedSystems: ['both'],
                requiredConfidence: 0.95
            },
            confidenceRequired: 0.95,
            timestamp: new Date().toISOString()
        };
        const result = canonical_1.CanonicalSchemas.VerificationRequest.safeParse(invalidRequest);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('Schema rejects out-of-range confidence values', () => {
        const invalidResult = {
            system: 'z3',
            verified: true,
            confidence: 1.5, // Invalid: must be between 0 and 1
            output: 'sat',
            metadata: {
                executionTime: 1500,
                strategy: 'z3_only',
                timestamp: new Date().toISOString()
            }
        };
        const result = canonical_1.CanonicalSchemas.VerificationResult.safeParse(invalidResult);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
(0, globals_1.describe)('Contract Tests: Cross-Validation Integration', () => {
    (0, globals_1.test)('Both systems respond to same problem type', async () => {
        const payload = {
            problem: {
                type: 'FORMAL_VERIFICATION',
                statement: 'forall x: int, x + 0 = x',
                description: 'Additive identity verification'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        // Query both systems
        const [z3Response, leanaideResponse] = await Promise.all([
            axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 10000 }),
            axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 10000 })
        ]);
        // Both should have required fields
        (0, globals_1.expect)(z3Response.data).toHaveProperty('verified');
        (0, globals_1.expect)(leanaideResponse.data).toHaveProperty('verified');
        (0, globals_1.expect)(z3Response.data).toHaveProperty('confidence');
        (0, globals_1.expect)(leanaideResponse.data).toHaveProperty('confidence');
    });
    (0, globals_1.test)('Cross-validation agreement can be determined', async () => {
        const payload = {
            problem: {
                type: 'FORMAL_VERIFICATION',
                statement: 'forall x: int, x + 0 = x',
                description: 'Additive identity verification'
            },
            constraints: {
                timeout: 5000,
                precision: 'medium'
            }
        };
        const [z3Response, leanaideResponse] = await Promise.all([
            axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 10000 }),
            axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 10000 })
        ]);
        const z3Verified = z3Response.data.verified;
        const leanaideVerified = leanaideResponse.data.verified;
        // We can determine agreement (may or may not agree)
        (0, globals_1.expect)(typeof z3Verified).toBe('boolean');
        (0, globals_1.expect)(typeof leanaideVerified).toBe('boolean');
        const agreement = z3Verified === leanaideVerified;
        (0, globals_1.expect)(typeof agreement).toBe('boolean');
    });
});
(0, globals_1.describe)('Contract Tests: Error Handling', () => {
    (0, globals_1.test)('Z3 handles timeout constraints', async () => {
        const payload = {
            problem: {
                type: 'SMT_CONSTRAINTS',
                statement: '(declare-const x Int) (assert (> x 0)) (check-sat)',
                description: 'Simple constraint'
            },
            constraints: {
                timeout: 1000, // 1 second timeout
                precision: 'medium'
            }
        };
        // Should either succeed or fail gracefully, not hang
        const response = await axios_1.default.post(`${Z3_URL}/verify`, payload, { timeout: 5000 });
        (0, globals_1.expect)(response.data).toBeDefined();
    });
    (0, globals_1.test)('LeanAide handles timeout constraints', async () => {
        const payload = {
            problem: {
                type: 'THEOREM_PROVING',
                statement: 'theorem add_zero (n : Nat) : n + 0 = n := by simp',
                description: 'Simple theorem'
            },
            constraints: {
                timeout: 1000, // 1 second timeout
                precision: 'medium'
            }
        };
        // Should either succeed or fail gracefully, not hang
        const response = await axios_1.default.post(`${LEANAIDE_URL}/verify`, payload, { timeout: 5000 });
        (0, globals_1.expect)(response.data).toBeDefined();
    });
});
//# sourceMappingURL=contract.test.js.map