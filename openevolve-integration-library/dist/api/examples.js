"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.basicInitialization = basicInitialization;
exports.authenticatedClient = authenticatedClient;
exports.quickClient = quickClient;
exports.simpleExecution = simpleExecution;
exports.executionWithOptions = executionWithOptions;
exports.executionWithProgress = executionWithProgress;
exports.streamingExecution = streamingExecution;
exports.streamingWithUI = streamingWithUI;
exports.simpleBatch = simpleBatch;
exports.batchWithStatistics = batchWithStatistics;
exports.basicErrorHandling = basicErrorHandling;
exports.specificErrorHandling = specificErrorHandling;
exports.withRetryLogic = withRetryLogic;
exports.withCancellation = withCancellation;
exports.basicHealthCheck = basicHealthCheck;
exports.detailedHealthMonitoring = detailedHealthMonitoring;
exports.trackMetrics = trackMetrics;
exports.performanceMonitoring = performanceMonitoring;
exports.integrationSpecificMethods = integrationSpecificMethods;
exports.allIntegrations = allIntegrations;
exports.completeWorkflow = completeWorkflow;
const client_1 = require("./client");
const errors_1 = require("./errors");
async function basicInitialization() {
    const client = new client_1.OpenEvolveClient({
        baseUrl: 'http://localhost:8000',
        timeout: 30000,
        retryAttempts: 3,
        enableWebSocket: true,
        debug: true,
    });
    await client.connect();
    console.log('Connected:', client.isConnected());
    return client;
}
async function authenticatedClient() {
    const client = new client_1.OpenEvolveClient({
        baseUrl: 'http://localhost:8000',
        apiKey: 'your-api-key-here',
        headers: {
            'X-Organization': 'MyCompany',
            'X-Project': 'AI-Research',
        },
        debug: false,
    });
    return client;
}
async function quickClient() {
    const client = (0, client_1.createOpenEvolveClient)('http://localhost:8000');
    return client;
}
async function simpleExecution(client) {
    try {
        const result = await client.execute(client_1.IntegrationName.LEANAIDE, {
            type: 'theorem_proving',
            statement: 'theorem simple : ∀ n : Nat, n + 0 = n',
        });
        console.log('Proof generated:', result);
        return result;
    }
    catch (error) {
        console.error('Execution failed:', error);
        throw error;
    }
}
async function executionWithOptions(client) {
    const options = {
        timeout: 60000,
        retries: 5,
        onComplete: (result) => {
            console.log('Execution completed:', result);
        },
        onError: (error) => {
            console.error('Execution error:', error);
        },
    };
    const result = await client.execute(client_1.IntegrationName.EVOLUTION, {
        prompt: 'Evolve a sorting algorithm',
        iterations: 100,
        populationSize: 50,
    }, options);
    return result;
}
async function executionWithProgress(client) {
    const result = await client.execute(client_1.IntegrationName.MAKER, {
        prompt: 'Create a novel renewable energy device',
        domain: 'energy',
    }, {
        onProgress: (update) => {
            console.log(`[${update.timestamp}] ${update.progress}%: ${update.message}`);
            console.log(`  Stage: ${update.stage}`);
            console.log(`  ETA: ${update.eta}ms`);
            if (update.data) {
                console.log(`  Current idea: ${update.data.currentIdea}`);
                console.log(`  Feasibility: ${update.data.feasibility}`);
            }
        },
    });
    return result;
}
async function streamingExecution(client) {
    const result = await client.executeStream(client_1.IntegrationName.EVOLUTION, {
        prompt: 'Evolve a solution for TSP',
        parameters: {
            populationSize: 100,
            generations: 50,
        },
    }, (update) => {
        console.log(`Generation ${update.data.generation}:`);
        console.log(`  Best fitness: ${update.data.bestFitness}`);
        console.log(`  Average fitness: ${update.data.avgFitness}`);
        console.log(`  Progress: ${update.progress}%`);
    });
    console.log('Final solution:', result.solution);
    return result;
}
async function streamingWithUI(client) {
    let progressBar = 0;
    const result = await client.executeStream(client_1.IntegrationName.HEPHAESTUS, {
        task: 'Formal verification of circuit',
        specification: '...',
    }, (update) => {
        progressBar = update.progress;
        updateProgressBar(progressBar);
        updateStatusText(update.message);
        updateCurrentStage(update.stage || 'Unknown');
        console.log(`[${update.stage}] ${update.message}`);
    });
    return result;
    function updateProgressBar(progress) {
    }
    function updateStatusText(message) {
    }
    function updateCurrentStage(stage) {
    }
}
async function simpleBatch(client) {
    const requests = [
        {
            integration: client_1.IntegrationName.LEANAIDE,
            id: 'proof1',
            inputs: { problem: 'Prove theorem 1' },
        },
        {
            integration: client_1.IntegrationName.MAKER,
            id: 'invention1',
            inputs: { prompt: 'Create something new' },
        },
        {
            integration: client_1.IntegrationName.EVOLUTION,
            id: 'evolution1',
            inputs: { prompt: 'Evolve this solution' },
        },
    ];
    const results = await client.executeBatch(requests);
    results.forEach((result) => {
        if (result.success) {
            console.log(`${result.id} succeeded in ${result.executionTime}ms`);
            console.log('Result:', result.result);
        }
        else {
            console.error(`${result.id} failed:`, result.error?.message);
        }
    });
    return results;
}
async function batchWithStatistics(client) {
    const problems = Array.from({ length: 10 }, (_, i) => ({
        integration: client_1.IntegrationName.LEANAIDE,
        id: `problem-${i}`,
        inputs: {
            type: 'theorem_proving',
            statement: `Theorem ${i + 1}`,
        },
    }));
    const results = await client.executeBatch(problems);
    const successCount = results.filter((r) => r.success).length;
    const failureCount = results.filter((r) => !r.success).length;
    const avgTime = results.reduce((sum, r) => sum + r.executionTime, 0) / results.length;
    const minTime = Math.min(...results.map((r) => r.executionTime));
    const maxTime = Math.max(...results.map((r) => r.executionTime));
    console.log('Batch Statistics:');
    console.log(`  Success: ${successCount}/${results.length}`);
    console.log(`  Failures: ${failureCount}`);
    console.log(`  Average time: ${avgTime.toFixed(2)}ms`);
    console.log(`  Min time: ${minTime}ms`);
    console.log(`  Max time: ${maxTime}ms`);
    return results;
}
async function basicErrorHandling(client) {
    try {
        const result = await client.execute(client_1.IntegrationName.LEANAIDE, {
            type: 'theorem_proving',
            statement: '...',
        });
        return result;
    }
    catch (error) {
        if (error instanceof errors_1.IntegrationError) {
            console.error(`Integration error [${error.code}]:`, error.message);
            console.error('Integration:', error.integration);
            console.error('Details:', error.details);
        }
        throw error;
    }
}
async function specificErrorHandling(client) {
    try {
        const result = await client.execute(client_1.IntegrationName.MAKER, {
            prompt: 'Create something',
        });
        return result;
    }
    catch (error) {
        if (error instanceof errors_1.ValidationError) {
            console.error('Validation failed:');
            error.getErrorMessages().forEach((msg) => console.error(`  - ${msg}`));
        }
        else if (error instanceof errors_1.TimeoutError) {
            console.error('Request timed out');
        }
        else if (error instanceof errors_1.RateLimitError) {
            const retryAfter = error.getRetryAfterMs();
            console.log(`Rate limited. Retry after ${retryAfter}ms`);
        }
        else if (error instanceof errors_1.IntegrationError) {
            console.error('Integration error:', error.message);
        }
        throw error;
    }
}
async function withRetryLogic(client) {
    let attempts = 0;
    const maxAttempts = 3;
    while (attempts < maxAttempts) {
        try {
            const result = await client.execute(client_1.IntegrationName.EVOLUTION, {
                prompt: 'Evolve solution',
            });
            return result;
        }
        catch (error) {
            attempts++;
            if ((0, client_1.isRetryableError)(error) && attempts < maxAttempts) {
                console.log(`Attempt ${attempts} failed, retrying...`);
                await new Promise((resolve) => setTimeout(resolve, 1000 * attempts));
            }
            else if ((0, client_1.isCriticalError)(error)) {
                console.error('Critical error, not retrying:', error);
                throw error;
            }
            else {
                console.error('Max attempts reached or non-retryable error');
                throw error;
            }
        }
    }
}
async function withCancellation(client) {
    const controller = new AbortController();
    const promise = client.execute(client_1.IntegrationName.EVOLUTION, { prompt: 'Long running task...' }, { signal: controller.signal });
    setTimeout(() => {
        controller.abort();
        console.log('Request cancelled');
    }, 5000);
    try {
        const result = await promise;
        return result;
    }
    catch (error) {
        if (error.code === 'CANCELLATION_ERROR') {
            console.log('Request was successfully cancelled');
        }
        throw error;
    }
}
async function basicHealthCheck(client) {
    const health = await client.healthCheck();
    console.log('Overall status:', health.status);
    console.log('Backend online:', health.backend.online);
    console.log('Backend version:', health.backend.version);
    console.log('Active connections:', health.backend.activeConnections);
    console.log('Memory usage:', health.backend.memory);
    console.log('CPU usage:', health.backend.cpu);
    return health;
}
async function detailedHealthMonitoring(client) {
    const health = await client.healthCheck();
    if (health.status !== 'healthy') {
        console.warn('System health is', health.status);
    }
    if (!health.backend.online) {
        console.error('Backend is offline!');
        return;
    }
    Object.entries(health.integrations).forEach(([name, status]) => {
        console.log(`\n${name}:`);
        console.log(`  Status: ${status.status}`);
        console.log(`  Response time: ${status.responseTime}ms`);
        if (status.status === 'unavailable') {
            console.error(`  Error: ${status.lastError}`);
        }
        console.log(`  Available endpoints: ${status.endpoints.join(', ')}`);
    });
    return health;
}
async function trackMetrics(client) {
    const executionId = 'my-execution-id';
    const result = await client.execute(client_1.IntegrationName.LEANAIDE, { problem: '...' }, { metadata: { executionId } });
    const metrics = client.getMetrics(executionId);
    if (metrics) {
        console.log('Execution metrics:');
        console.log(`  Duration: ${metrics.duration}ms`);
        console.log(`  Success: ${metrics.success}`);
        console.log(`  Retries: ${metrics.retries}`);
        console.log(`  Status code: ${metrics.statusCode}`);
        if (metrics.error) {
            console.error(`  Error: ${metrics.error}`);
        }
    }
    return result;
}
async function performanceMonitoring(client) {
    const operations = 100;
    const times = [];
    for (let i = 0; i < operations; i++) {
        const start = Date.now();
        await client.execute(client_1.IntegrationName.LEANAIDE, {
            problem: `Problem ${i}`,
        });
        times.push(Date.now() - start);
    }
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    const p50 = times.sort((a, b) => a - b)[Math.floor(times.length / 2)];
    const p95 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];
    const p99 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.99)];
    console.log('Performance Statistics:');
    console.log(`  Average: ${avg.toFixed(2)}ms`);
    console.log(`  Min: ${min}ms`);
    console.log(`  Max: ${max}ms`);
    console.log(`  P50: ${p50}ms`);
    console.log(`  P95: ${p95}ms`);
    console.log(`  P99: ${p99}ms`);
    return { avg, min, max, p50, p95, p99 };
}
async function integrationSpecificMethods(client) {
    const leanaide = client.integrations.leanaide;
    const proof = await leanaide.execute({
        type: 'theorem_proving',
        statement: 'theorem example : ∀ x, x + 0 = x',
    });
    const health = await leanaide.healthCheck();
    console.log('LeanAide health:', health);
    const validation = await leanaide.validate({
        type: 'theorem_proving',
        statement: '...',
    });
    console.log('Validation:', validation);
    return proof;
}
async function allIntegrations(client) {
    const results = await Promise.all([
        client.integrations.leanaide.execute({
            type: 'theorem_proving',
            statement: '...',
        }),
        client.integrations.evolution.execute({
            prompt: 'Evolve solution',
            parameters: { generations: 50 },
        }),
        client.integrations.knowledge.execute({
            query: 'Find relationships',
            options: { depth: 3 },
        }),
        client.integrations.maker.execute({
            prompt: 'Create new invention',
            domain: 'technology',
        }),
        client.integrations.hephaestus.execute({
            task: 'verify',
            specification: '...',
        }),
        client.integrations.decomposition.execute({
            problem: 'Complex problem',
            strategy: 'hierarchical',
        }),
        client.integrations.verification.execute({
            result: '...',
            criteria: ['correctness', 'completeness'],
        }),
        client.integrations.assembly.execute({
            components: ['comp1', 'comp2', 'comp3'],
            strategy: 'optimized',
        }),
    ]);
    return results;
}
async function completeWorkflow() {
    const client = new client_1.OpenEvolveClient({
        baseUrl: 'http://localhost:8000',
        timeout: 60000,
        retryAttempts: 3,
        enableWebSocket: true,
        debug: true,
    });
    try {
        console.log('Checking system health...');
        const health = await client.healthCheck();
        if (health.status !== 'healthy') {
            throw new Error('System is not healthy');
        }
        console.log('System is healthy');
        console.log('Connecting to backend...');
        await client.connect();
        console.log('Connected');
        console.log('Starting evolution task...');
        const result = await client.execute(client_1.IntegrationName.EVOLUTION, {
            prompt: 'Evolve optimal solution',
            parameters: {
                populationSize: 100,
                generations: 100,
            },
        }, {
            onProgress: (update) => {
                console.log(`[${update.progress}%] ${update.message}`);
            },
            onComplete: (result) => {
                console.log('Task completed successfully');
            },
            onError: (error) => {
                console.error('Task error:', error);
            },
        });
        console.log('Processing result...');
        console.log('Solution:', result.solution);
        console.log('Fitness:', result.fitness);
        const allMetrics = client.getAllMetrics();
        console.log(`Total executions: ${allMetrics.size}`);
        console.log('Cleaning up...');
        client.clearMetrics();
        await client.disconnect();
        return result;
    }
    catch (error) {
        console.error('Workflow failed:', error);
        try {
            await client.disconnect();
        }
        catch (e) {
            console.error('Cleanup failed:', e);
        }
        throw error;
    }
}
//# sourceMappingURL=examples.js.map