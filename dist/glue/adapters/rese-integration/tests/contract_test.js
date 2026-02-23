"use strict";
/**
 * RESE Canonical Schema Contract Tests
 *
 * These tests verify that the RESE canonical schemas correctly validate
 * data structures according to the RESE Technical Manual specification.
 *
 * Tests follow CLAUDE.md principles:
 * - Phase 2: The Contract (Defense)
 * - Protects the Mega-Project from API changes
 * - Ensures schema validation is working correctly
 *
 * Run with:
 *   npm test -- contract_test.ts
 *   OR
 *   ts-node glue/adapters/rese-integration/tests/contract_test.ts
 */
Object.defineProperty(exports, "__esModule", { value: true });
const rese_canonical_1 = require("../../../schemas/rese-canonical");
/**
 * Helper function to run a test
 */
function test(name, fn) {
    try {
        fn();
        console.log(`✓ ${name}`);
    }
    catch (error) {
        console.error(`✗ ${name}`);
        console.error(`  Error: ${error.message}`);
        throw error;
    }
}
/**
 * Helper function to assert a condition
 */
function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}
/**
 * Helper function to assert throws
 */
function assertThrows(fn, message) {
    try {
        fn();
        throw new Error(`Expected function to throw, but it didn't: ${message}`);
    }
    catch (error) {
        // Expected - function threw
    }
}
/**
 * ============================================================================
 * PHASE I: EPISTEMIC AUDIT RESULT TESTS
 * ============================================================================
 */
console.log('\n=== PHASE I: EPISTEMIC AUDIT RESULT TESTS ===\n');
// Test 1: Valid EpistemicAuditResult validation
test('Valid EpistemicAuditResult should pass validation', () => {
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(rese_canonical_1.RESEExamples.validEpistemicAuditResult);
    assert(result.success === true, 'Validation should succeed');
    assert(result.data !== undefined, 'Data should be present');
    assert(result.data?.phase === 'phase1_epistemic_audit', 'Phase should be correct');
    assert(result.data?.audit_id === '550e8400-e29b-41d4-a716-446655440000', 'Audit ID should match');
    assert(result.data?.tacit_assumptions?.length === 1, 'Should have 1 tacit assumption');
    assert(result.data?.contradictions?.length === 1, 'Should have 1 contradiction');
    assert(result.data?.falsification_results?.length === 1, 'Should have 1 falsification result');
});
// Test 2: Invalid EpistemicAuditResult - missing required fields
test('Invalid EpistemicAuditResult (missing required fields) should fail validation', () => {
    const invalidData = {
        phase: 'phase1_epistemic_audit',
        // Missing audit_id
        // Missing problem_description
        timestamp: new Date().toISOString(),
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
    assert(result.errors.length > 0, 'Should have validation errors');
});
// Test 3: Invalid EpistemicAuditResult - invalid timestamp format
test('Invalid EpistemicAuditResult (invalid timestamp) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        timestamp: 'not-a-timestamp',
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
});
// Test 4: Invalid EpistemicAuditResult - tacit assumption with invalid confidence
test('Invalid EpistemicAuditResult (confidence > 1) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        tacit_assumptions: [
            {
                ...rese_canonical_1.RESEExamples.validEpistemicAuditResult.tacit_assumptions[0],
                confidence_score: 1.5, // Invalid: > 1
            },
        ],
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
});
// Test 5: Transformation from raw response to canonical
test('Transform raw EpistemicAudit response to canonical format', () => {
    const rawResponse = {
        audit_id: 'test-audit-123',
        problem: 'Test problem description',
        tacit_assumptions: [
            {
                assumption: 'Test assumption',
                confidence: 0.75,
            },
        ],
        contradictions: [
            {
                type: 'circulus_in_probando',
                set_size: 2,
            },
        ],
        falsification_results: [
            {
                hypothesis_id: 'hyp-123',
                falsified: true,
                robustness: 0.5,
            },
        ],
        metrics: {
            total_assumptions: 10,
            contradictions: 2,
            falsified: 5,
            failure_reduction: 0.3,
        },
        metadata: {
            time: 5000,
            epoch: 1,
        },
    };
    const canonical = (0, rese_canonical_1.transformEpistemicAuditToCanonical)(rawResponse, '550e8400-e29b-41d4-a716-446655440000');
    const validation = (0, rese_canonical_1.validateEpistemicAuditResult)(canonical);
    assert(validation.success === true, 'Transformed data should be valid');
    assert(validation.data?.audit_id === 'test-audit-123', 'Audit ID should match');
    assert(validation.data?.problem_description === 'Test problem description', 'Problem description should match');
});
/**
 * ============================================================================
 * PHASE II: ISOMORPHIC MAPPING TESTS
 * ============================================================================
 */
console.log('\n=== PHASE II: ISOMORPHIC MAPPING TESTS ===\n');
// Test 6: Valid IsomorphicMapping validation
test('Valid IsomorphicMapping should pass validation', () => {
    const result = (0, rese_canonical_1.validateIsomorphicMapping)(rese_canonical_1.RESEExamples.validIsomorphicMapping);
    assert(result.success === true, 'Validation should succeed');
    assert(result.data !== undefined, 'Data should be present');
    assert(result.data?.phase === 'phase2_isomorphic_mapping', 'Phase should be correct');
    assert(result.data?.mapping_id === '550e8400-e29b-41d4-a716-446655440002', 'Mapping ID should match');
    assert(result.data?.cross_domain_patterns?.length === 1, 'Should have 1 cross-domain pattern');
    assert(result.data?.inverted_constraints?.length === 1, 'Should have 1 inverted constraint');
});
// Test 7: Invalid IsomorphicMapping - missing required fields
test('Invalid IsomorphicMapping (missing required fields) should fail validation', () => {
    const invalidData = {
        phase: 'phase2_isomorphic_mapping',
        // Missing mapping_id
        // Missing problem_description
        timestamp: new Date().toISOString(),
    };
    const result = (0, rese_canonical_1.validateIsomorphicMapping)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
});
// Test 8: Invalid IsomorphicMapping - isomorphism score out of range
test('Invalid IsomorphicMapping (isomorphism score > 1) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validIsomorphicMapping,
        cross_domain_patterns: [
            {
                ...rese_canonical_1.RESEExamples.validIsomorphicMapping.cross_domain_patterns[0],
                mechanistic_isomorphism_score: 1.5, // Invalid: > 1
            },
        ],
    };
    const result = (0, rese_canonical_1.validateIsomorphicMapping)(invalidData);
    assert(result.success === false, 'Validation should fail');
});
// Test 9: Transformation from raw response to canonical
test('Transform raw IsomorphicMapping response to canonical format', () => {
    const rawResponse = {
        mapping_id: 'test-mapping-456',
        problem: 'Need isolated state computation',
        patterns: [
            {
                source: 'Homomorphic Encryption',
                source_desc: 'Encrypted computation',
                target: 'Lattice Fusion',
                target_desc: 'Isolated nuclear reactions',
                isomorphism_score: 0.85,
            },
        ],
        constraints: [
            {
                original: 'Energy dissipates',
                inverted: 'Energy is contained',
            },
        ],
        metrics: {
            domains_searched: 10,
            patterns: 4,
        },
        metadata: {
            time: 3000,
            epoch: 2,
        },
    };
    const canonical = (0, rese_canonical_1.transformIsomorphicMappingToCanonical)(rawResponse, '550e8400-e29b-41d4-a716-446655440002');
    const validation = (0, rese_canonical_1.validateIsomorphicMapping)(canonical);
    assert(validation.success === true, 'Transformed data should be valid');
    assert(validation.data?.mapping_id === 'test-mapping-456', 'Mapping ID should match');
});
/**
 * ============================================================================
 * PHASE III: MCTS SEARCH RESULT TESTS
 * ============================================================================
 */
console.log('\n=== PHASE III: MCTS SEARCH RESULT TESTS ===\n');
// Test 10: Valid MCTSSearchResult validation
test('Valid MCTSSearchResult should pass validation', () => {
    const result = (0, rese_canonical_1.validateMCTSSearchResult)(rese_canonical_1.RESEExamples.validMCTSSearchResult);
    assert(result.success === true, 'Validation should succeed');
    assert(result.data !== undefined, 'Data should be present');
    assert(result.data?.phase === 'phase3_mcts_refinement', 'Phase should be correct');
    assert(result.data?.search_id === '550e8400-e29b-41d4-a716-446655440003', 'Search ID should match');
    assert(result.data?.search_tree !== undefined, 'Should have search tree');
    assert(result.data?.hypotheses?.length === 1, 'Should have 1 hypothesis');
    assert(result.data?.validation_metrics !== undefined, 'Should have validation metrics');
});
// Test 11: Invalid MCTSSearchResult - missing required fields
test('Invalid MCTSSearchResult (missing required fields) should fail validation', () => {
    const invalidData = {
        phase: 'phase3_mcts_refinement',
        // Missing search_id
        // Missing problem_description
        timestamp: new Date().toISOString(),
    };
    const result = (0, rese_canonical_1.validateMCTSSearchResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
});
// Test 12: Invalid MCTSSearchResult - confidence out of range
test('Invalid MCTSSearchResult (confidence > 1) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validMCTSSearchResult,
        hypotheses: [
            {
                ...rese_canonical_1.RESEExamples.validMCTSSearchResult.hypotheses[0],
                confidence: 1.2, // Invalid: > 1
            },
        ],
    };
    const result = (0, rese_canonical_1.validateMCTSSearchResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
});
// Test 13: Transformation from raw response to canonical
test('Transform raw MCTSSearch response to canonical format', () => {
    const rawResponse = {
        search_id: 'test-search-789',
        problem: 'Find optimal configuration',
        tree: {
            root_id: 'root-123',
            visits: 100,
            value: 0.75,
            nodes: 500,
            depth: 6,
        },
        hypotheses: [
            {
                hypothesis: 'Test hypothesis',
                confidence: 0.8,
                value: 0.75,
                visits: 50,
            },
        ],
        converged: true,
        metrics: {
            simulations: 2000,
            hypotheses_tested: 50,
            time: 10000,
            epoch: 3,
        },
    };
    const canonical = (0, rese_canonical_1.transformMCTSSearchToCanonical)(rawResponse, '550e8400-e29b-41d4-a716-446655440003');
    const validation = (0, rese_canonical_1.validateMCTSSearchResult)(canonical);
    assert(validation.success === true, 'Transformed data should be valid');
    assert(validation.data?.search_id === 'test-search-789', 'Search ID should match');
});
/**
 * ============================================================================
 * PHASE IV: ARCHITECTURE ASSEMBLY TESTS
 * ============================================================================
 */
console.log('\n=== PHASE IV: ARCHITECTURE ASSEMBLY TESTS ===\n');
// Test 14: Valid ArchitectureAssembly validation
test('Valid ArchitectureAssembly should pass validation', () => {
    const result = (0, rese_canonical_1.validateArchitectureAssembly)(rese_canonical_1.RESEExamples.validArchitectureAssembly);
    assert(result.success === true, 'Validation should succeed');
    assert(result.data !== undefined, 'Data should be present');
    assert(result.data?.phase === 'phase4_architecture_assembly', 'Phase should be correct');
    assert(result.data?.assembly_id === '550e8400-e29b-41d4-a716-446655440004', 'Assembly ID should match');
    assert(result.data?.paradigm_shifts?.length === 1, 'Should have 1 paradigm shift');
    assert(result.data?.synthesized_knowledge?.length === 1, 'Should have 1 knowledge item');
    assert(result.data?.validation_results !== undefined, 'Should have validation results');
});
// Test 15: Invalid ArchitectureAssembly - missing required fields
test('Invalid ArchitectureAssembly (missing required fields) should fail validation', () => {
    const invalidData = {
        phase: 'phase4_architecture_assembly',
        // Missing assembly_id
        // Missing problem_description
        timestamp: new Date().toISOString(),
    };
    const result = (0, rese_canonical_1.validateArchitectureAssembly)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
});
// Test 16: Invalid ArchitectureAssembly - empty title
test('Invalid ArchitectureAssembly (empty title) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validArchitectureAssembly,
        synthesized_knowledge: [
            {
                ...rese_canonical_1.RESEExamples.validArchitectureAssembly.synthesized_knowledge[0],
                title: '', // Invalid: empty string
            },
        ],
    };
    const result = (0, rese_canonical_1.validateArchitectureAssembly)(invalidData);
    assert(result.success === false, 'Validation should fail');
});
// Test 17: Transformation from raw response to canonical
test('Transform raw ArchitectureAssembly response to canonical format', () => {
    const rawResponse = {
        assembly_id: 'test-assembly-999',
        problem: 'Complete theoretical framework',
        shifts: [
            {
                incumbent: 'Old paradigm',
                new: 'New paradigm',
                validated: true,
            },
        ],
        knowledge: [
            {
                type: 'theoretical_model',
                title: 'Test Model',
                description: 'Test description',
                verified: true,
            },
        ],
        validation_results: {
            predictive_efficacy: true,
            aci_reduction: 0.6,
            predictions_count: 10,
            predictions_verified: 8,
            success_rate: 0.8,
        },
        metadata: {
            epochs: 5,
            time: 50000,
            converged: true,
        },
    };
    const canonical = (0, rese_canonical_1.transformArchitectureAssemblyToCanonical)(rawResponse, '550e8400-e29b-41d4-a716-446655440004');
    const validation = (0, rese_canonical_1.validateArchitectureAssembly)(canonical);
    assert(validation.success === true, 'Transformed data should be valid');
    assert(validation.data?.assembly_id === 'test-assembly-999', 'Assembly ID should match');
});
/**
 * ============================================================================
 * EDGE CASES AND ERROR HANDLING TESTS
 * ============================================================================
 */
console.log('\n=== EDGE CASES AND ERROR HANDLING TESTS ===\n');
// Test 18: Empty tacit assumptions array (optional field)
test('EpistemicAuditResult with empty tacit assumptions should be valid', () => {
    const validData = {
        ...rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        tacit_assumptions: [],
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(validData);
    assert(result.success === true, 'Validation should succeed with empty array');
});
// Test 19: Optional metadata fields
test('Result without optional metadata should be valid', () => {
    const minimalData = {
        phase: 'phase1_epistemic_audit',
        audit_id: '550e8400-e29b-41d4-a716-446655440000',
        problem_description: 'Test problem',
        timestamp: new Date().toISOString(),
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(minimalData);
    assert(result.success === true, 'Validation should succeed with minimal data');
});
// Test 20: Invalid correlation_id (not a UUID)
test('Invalid correlation_id (not UUID) should fail validation', () => {
    const invalidData = {
        ...rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        correlation_id: 'not-a-uuid',
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(invalidData);
    assert(result.success === false, 'Validation should fail with invalid UUID');
});
// Test 21: Multiple validation errors
test('Multiple validation errors should all be reported', () => {
    const invalidData = {
        phase: 'phase1_epistemic_audit',
        // Missing audit_id
        // Missing problem_description
        timestamp: 'invalid-timestamp',
        correlation_id: 'not-a-uuid',
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(invalidData);
    assert(result.success === false, 'Validation should fail');
    assert(result.errors !== undefined, 'Errors should be present');
    assert(result.errors.length >= 3, 'Should report multiple errors');
});
// Test 22: Null values in optional fields
test('Null values in optional fields should be handled correctly', () => {
    const dataWithNulls = {
        ...rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        tacit_assumptions: null,
        metadata: null,
    };
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(dataWithNulls);
    // Null should fail for arrays (should be undefined if not present)
    assert(result.success === false, 'Null should fail validation for array fields');
});
/**
 * ============================================================================
 * SERIALIZATION/DESERIALIZATION TESTS
 * ============================================================================
 */
console.log('\n=== SERIALIZATION/DESERIALIZATION TESTS ===\n');
// Test 23: JSON serialization and deserialization
test('EpistemicAuditResult should survive JSON round-trip', () => {
    const original = rese_canonical_1.RESEExamples.validEpistemicAuditResult;
    // Serialize to JSON
    const json = JSON.stringify(original);
    // Deserialize from JSON
    const parsed = JSON.parse(json);
    // Validate parsed data
    const result = (0, rese_canonical_1.validateEpistemicAuditResult)(parsed);
    assert(result.success === true, 'Round-tripped data should be valid');
    assert(result.data?.audit_id === original.audit_id, 'Audit ID should match');
});
// Test 24: All phases should survive JSON round-trip
test('All RESE results should survive JSON round-trip', () => {
    const results = [
        rese_canonical_1.RESEExamples.validEpistemicAuditResult,
        rese_canonical_1.RESEExamples.validIsomorphicMapping,
        rese_canonical_1.RESEExamples.validMCTSSearchResult,
        rese_canonical_1.RESEExamples.validArchitectureAssembly,
    ];
    const validators = [
        rese_canonical_1.validateEpistemicAuditResult,
        rese_canonical_1.validateIsomorphicMapping,
        rese_canonical_1.validateMCTSSearchResult,
        rese_canonical_1.validateArchitectureAssembly,
    ];
    results.forEach((result, index) => {
        const json = JSON.stringify(result);
        const parsed = JSON.parse(json);
        const validation = validators[index](parsed);
        assert(validation.success === true, `Phase ${index + 1} should survive round-trip`);
    });
});
/**
 * ============================================================================
 * SUMMARY
 * ============================================================================
 */
console.log('\n=== ALL CONTRACT TESTS PASSED ===\n');
console.log('Summary:');
console.log('  ✓ Phase I (Epistemic Audit): 5 tests passed');
console.log('  ✓ Phase II (Isomorphic Mapping): 4 tests passed');
console.log('  ✓ Phase III (MCTS Search): 4 tests passed');
console.log('  ✓ Phase IV (Architecture Assembly): 4 tests passed');
console.log('  ✓ Edge Cases: 5 tests passed');
console.log('  ✓ Serialization: 2 tests passed');
console.log('  Total: 24 tests passed\n');
//# sourceMappingURL=contract_test.js.map