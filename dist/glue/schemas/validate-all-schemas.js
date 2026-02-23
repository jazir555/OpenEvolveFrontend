"use strict";
/**
 * Schema Validation Test Script
 *
 * This script validates all canonical schemas to ensure they are correctly defined.
 * Run with: npx ts-node glue/schemas/validate-all-schemas.ts
 */
Object.defineProperty(exports, "__esModule", { value: true });
const index_1 = require("./index");
/**
 * Test suite runner
 */
function runTests() {
    console.log('🧪 Running Schema Validation Tests...\n');
    let passed = 0;
    let failed = 0;
    const testResults = [];
    // Test 1: Z3 Solver Request
    console.log('Test 1: Z3 Solver Request');
    const z3Result = (0, index_1.validateSolverRequest)(index_1.Z3Examples.validSolverRequest);
    if (z3Result.success) {
        console.log('  ✅ PASS: Z3 SolverRequest validation');
        passed++;
        testResults.push({ name: 'Z3 SolverRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: Z3 SolverRequest validation');
        console.log('  Errors:', z3Result.errors);
        failed++;
        testResults.push({ name: 'Z3 SolverRequest', passed: false, error: z3Result.errors?.join(', ') });
    }
    // Test 2: LeanAide Proof Verification Request
    console.log('\nTest 2: LeanAide Proof Verification Request');
    const leanAideResult = (0, index_1.validateProofVerificationRequest)({
        proof_code: 'theorem example : Nat := by trivial',
        theorem: '∃ n: Nat, n = n',
        timeout_ms: index_1.DEFAULT_TIMEOUTS.NORMAL,
        correlation_id: (0, index_1.createCorrelationId)(),
    });
    if (leanAideResult.success) {
        console.log('  ✅ PASS: LeanAide ProofVerificationRequest validation');
        passed++;
        testResults.push({ name: 'LeanAide ProofVerificationRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: LeanAide ProofVerificationRequest validation');
        console.log('  Errors:', leanAideResult.errors);
        failed++;
        testResults.push({ name: 'LeanAide ProofVerificationRequest', passed: false, error: leanAideResult.errors?.join(', ') });
    }
    // Test 3: RAGBits RAG Request
    console.log('\nTest 3: RAGBits RAG Request');
    const ragResult = (0, index_1.validateRAGRequest)(index_1.RAGExamples.validRAGRequest);
    if (ragResult.success) {
        console.log('  ✅ PASS: RAGBits RAGRequest validation');
        passed++;
        testResults.push({ name: 'RAGBits RAGRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: RAGBits RAGRequest validation');
        console.log('  Errors:', ragResult.errors);
        failed++;
        testResults.push({ name: 'RAGBits RAGRequest', passed: false, error: ragResult.errors?.join(', ') });
    }
    // Test 4: RAGBits Document Chunk
    console.log('\nTest 4: RAGBits Document Chunk');
    const chunkResult = (0, index_1.validateDocumentChunk)({
        id: (0, index_1.createCorrelationId)(),
        content: 'This is a sample document chunk for testing.',
        source: 'test_document.txt',
        chunk_index: 0,
        timestamp: (0, index_1.createUTCTimestamp)(),
    });
    if (chunkResult.success) {
        console.log('  ✅ PASS: RAGBits DocumentChunk validation');
        passed++;
        testResults.push({ name: 'RAGBits DocumentChunk', passed: true });
    }
    else {
        console.log('  ❌ FAIL: RAGBits DocumentChunk validation');
        console.log('  Errors:', chunkResult.errors);
        failed++;
        testResults.push({ name: 'RAGBits DocumentChunk', passed: false, error: chunkResult.errors?.join(', ') });
    }
    // Test 5: BubbleLab Bubble Request
    console.log('\nTest 5: BubbleLab Bubble Request');
    const bubbleResult = (0, index_1.validateBubbleRequest)(index_1.BubbleLabExamples.validBubbleRequest);
    if (bubbleResult.success) {
        console.log('  ✅ PASS: BubbleLab BubbleRequest validation');
        passed++;
        testResults.push({ name: 'BubbleLab BubbleRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: BubbleLab BubbleRequest validation');
        console.log('  Errors:', bubbleResult.errors);
        failed++;
        testResults.push({ name: 'BubbleLab BubbleRequest', passed: false, error: bubbleResult.errors?.join(', ') });
    }
    // Test 6: BubbleLab Workflow Request
    console.log('\nTest 6: BubbleLab Workflow Request');
    const workflowResult = (0, index_1.validateWorkflowRequest)(index_1.BubbleLabExamples.validWorkflowRequest);
    if (workflowResult.success) {
        console.log('  ✅ PASS: BubbleLab WorkflowRequest validation');
        passed++;
        testResults.push({ name: 'BubbleLab WorkflowRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: BubbleLab WorkflowRequest validation');
        console.log('  Errors:', workflowResult.errors);
        failed++;
        testResults.push({ name: 'BubbleLab WorkflowRequest', passed: false, error: workflowResult.errors?.join(', ') });
    }
    // Test 7: VectorDB Search Request
    console.log('\nTest 7: VectorDB Search Request');
    const vectorSearchResult = (0, index_1.validateVectorSearchRequest)(index_1.VectorDBExamples.validVectorSearchRequest);
    if (vectorSearchResult.success) {
        console.log('  ✅ PASS: VectorDB VectorSearchRequest validation');
        passed++;
        testResults.push({ name: 'VectorDB VectorSearchRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: VectorDB VectorSearchRequest validation');
        console.log('  Errors:', vectorSearchResult.errors);
        failed++;
        testResults.push({ name: 'VectorDB VectorSearchRequest', passed: false, error: vectorSearchResult.errors?.join(', ') });
    }
    // Test 8: VectorDB Collection Info
    console.log('\nTest 8: VectorDB Collection Info');
    const collectionInfoResult = (0, index_1.validateCollectionInfo)(index_1.VectorDBExamples.validCollectionInfo);
    if (collectionInfoResult.success) {
        console.log('  ✅ PASS: VectorDB CollectionInfo validation');
        passed++;
        testResults.push({ name: 'VectorDB CollectionInfo', passed: true });
    }
    else {
        console.log('  ❌ FAIL: VectorDB CollectionInfo validation');
        console.log('  Errors:', collectionInfoResult.errors);
        failed++;
        testResults.push({ name: 'VectorDB CollectionInfo', passed: false, error: collectionInfoResult.errors?.join(', ') });
    }
    // Test 9: Graphiti Entity
    console.log('\nTest 9: Graphiti Entity');
    const entityResult = (0, index_1.validateCanonical)(index_1.CanonicalEntitySchema, {
        id: (0, index_1.createCorrelationId)(),
        name: 'Test Entity',
        labels: ['Person', 'Employee'],
        summary: 'A test entity for validation',
        attributes: { department: 'Engineering' },
        created_at: (0, index_1.createUTCTimestamp)(),
    });
    if (entityResult.success) {
        console.log('  ✅ PASS: Graphiti CanonicalEntity validation');
        passed++;
        testResults.push({ name: 'Graphiti CanonicalEntity', passed: true });
    }
    else {
        console.log('  ❌ FAIL: Graphiti CanonicalEntity validation');
        console.log('  Errors:', entityResult.errors);
        failed++;
        testResults.push({ name: 'Graphiti CanonicalEntity', passed: false, error: entityResult.errors?.join(', ') });
    }
    // Test 10: Graphiti Episode
    console.log('\nTest 10: Graphiti Episode');
    const episodeResult = (0, index_1.validateCanonical)(index_1.CanonicalEpisodeSchema, {
        id: (0, index_1.createCorrelationId)(),
        name: 'Test Episode',
        content: 'This is a test episode for validation purposes.',
        episode_type: 'text',
        valid_at: (0, index_1.createUTCTimestamp)(),
        created_at: (0, index_1.createUTCTimestamp)(),
    });
    if (episodeResult.success) {
        console.log('  ✅ PASS: Graphiti CanonicalEpisode validation');
        passed++;
        testResults.push({ name: 'Graphiti CanonicalEpisode', passed: true });
    }
    else {
        console.log('  ❌ FAIL: Graphiti CanonicalEpisode validation');
        console.log('  Errors:', episodeResult.errors);
        failed++;
        testResults.push({ name: 'Graphiti CanonicalEpisode', passed: false, error: episodeResult.errors?.join(', ') });
    }
    // Test 11: KarateClub Node Embedding Request
    console.log('\nTest 11: KarateClub Node Embedding Request');
    const nodeEmbResult = (0, index_1.validateNodeEmbeddingRequest)({
        algorithm: 'node2vec',
        graph: {
            nodes: [
                { id: 'node1', features: [1.0, 2.0] },
                { id: 'node2', features: [3.0, 4.0] },
            ],
            edges: [
                { source: 'node1', target: 'node2' },
            ],
            directed: false,
        },
        timeout_ms: index_1.DEFAULT_TIMEOUTS.LONG,
        correlation_id: (0, index_1.createCorrelationId)(),
    });
    if (nodeEmbResult.success) {
        console.log('  ✅ PASS: KarateClub NodeEmbeddingRequest validation');
        passed++;
        testResults.push({ name: 'KarateClub NodeEmbeddingRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: KarateClub NodeEmbeddingRequest validation');
        console.log('  Errors:', nodeEmbResult.errors);
        failed++;
        testResults.push({ name: 'KarateClub NodeEmbeddingRequest', passed: false, error: nodeEmbResult.errors?.join(', ') });
    }
    // Test 12: KarateClub Community Detection Request
    console.log('\nTest 12: KarateClub Community Detection Request');
    const communityResult = (0, index_1.validateCommunityDetectionRequest)({
        algorithm: 'label_propagation',
        graph: {
            nodes: [
                { id: 'node1' },
                { id: 'node2' },
                { id: 'node3' },
            ],
            edges: [
                { source: 'node1', target: 'node2' },
                { source: 'node2', target: 'node3' },
            ],
            directed: false,
        },
        timeout_ms: index_1.DEFAULT_TIMEOUTS.LONG,
        correlation_id: (0, index_1.createCorrelationId)(),
    });
    if (communityResult.success) {
        console.log('  ✅ PASS: KarateClub CommunityDetectionRequest validation');
        passed++;
        testResults.push({ name: 'KarateClub CommunityDetectionRequest', passed: true });
    }
    else {
        console.log('  ❌ FAIL: KarateClub CommunityDetectionRequest validation');
        console.log('  Errors:', communityResult.errors);
        failed++;
        testResults.push({ name: 'KarateClub CommunityDetectionRequest', passed: false, error: communityResult.errors?.join(', ') });
    }
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 VALIDATION SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total Tests: ${passed + failed}`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`Success Rate: ${((passed / (passed + failed)) * 100).toFixed(2)}%`);
    console.log('='.repeat(60));
    if (failed === 0) {
        console.log('\n🎉 All schema validations passed successfully!');
    }
    else {
        console.log('\n⚠️  Some validations failed. Please review the errors above.');
        process.exit(1);
    }
}
// Run the tests
runTests();
//# sourceMappingURL=validate-all-schemas.js.map