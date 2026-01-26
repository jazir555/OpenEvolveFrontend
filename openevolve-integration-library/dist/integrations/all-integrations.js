"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SolutionIntegration = exports.AssemblyIntegration = exports.VerificationIntegration = exports.DecompositionIntegration = exports.HephaestusIntegration = exports.MakerIntegration = exports.KnowledgeIntegration = exports.EvolutionIntegration = exports.LeanAideIntegration = void 0;
const base_1 = require("./base");
const errors_1 = require("../api/errors");
class LeanAideIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'leanaide', '1.0.0', 'LeanAide: Formal mathematics theorem proving and verification', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'translate':
                    return await this.executeBackend('/api/v1/leanaide/translate', typeof input === 'string' ? { theorem: input } : input, executionId, options);
                case 'prove':
                    return await this.executeBackend('/api/v1/leanaide/prove', typeof input === 'string' ? { theorem: input, strategy: 'default', tactics: [], context: [] } : input, executionId, options);
                case 'verify':
                    return await this.executeBackend('/api/v1/leanaide/verify', typeof input === 'string' ? { proof: input } : input, executionId, options);
                case 'mcts':
                    return await this.executeBackend('/api/v1/leanaide/mcts', input, executionId, options);
                case 'query':
                    return await this.executeBackend('/api/v1/leanaide/query', typeof input === 'string' ? { question: input } : input, executionId, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['translate', 'prove', 'verify', 'mcts', 'query'],
                },
                input: {
                    type: 'object',
                    description: 'Operation-specific input'
                },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return ['/api/v1/leanaide/translate', '/api/v1/leanaide/prove',
            '/api/v1/leanaide/verify', '/api/v1/leanaide/mcts',
            '/api/v1/leanaide/query'];
    }
    async translateTheorem(theorem, options) {
        return this.executeBackend('/api/v1/leanaide/translate', { theorem }, undefined, options);
    }
    async generateProof(theorem, strategy, options) {
        return this.executeBackend('/api/v1/leanaide/prove', { theorem, strategy, tactics: [], context: [] }, undefined, options);
    }
    async verifyProof(proof, options) {
        return this.executeBackend('/api/v1/leanaide/verify', { proof }, undefined, options);
    }
    async runMCTS(problem, config, options) {
        return this.executeBackend('/api/v1/leanaide/mcts', { problem, config }, undefined, options);
    }
    async queryMath(question, options) {
        return this.executeBackend('/api/v1/leanaide/query', { question }, undefined, options);
    }
}
exports.LeanAideIntegration = LeanAideIntegration;
class EvolutionIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'evolution', '1.0.0', 'Evolution: Evolutionary and adversarial algorithms', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, config } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'evolution':
                    return await this.executeBackend('/api/v1/evolution/evolve', config, executionId, options);
                case 'adversarial':
                    return await this.executeBackend('/api/v1/evolution/adversarial', config, executionId, options);
                case 'coevolution':
                    return await this.executeBackend('/api/v1/evolution/coevolution', config, executionId, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    enum: ['evolution', 'adversarial', 'coevolution'],
                },
                config: { type: 'object', description: 'Evolution configuration' },
            },
            required: ['operation', 'config'],
        };
    }
    getEndpoints() {
        return ['/api/v1/evolution/evolve', '/api/v1/evolution/adversarial',
            '/api/v1/evolution/coevolution'];
    }
    async runEvolution(config, options) {
        return this.executeBackend('/api/v1/evolution/evolve', config, undefined, options);
    }
    async runAdversarial(config, options) {
        return this.executeBackend('/api/v1/evolution/adversarial', config, undefined, options);
    }
    async runCoevolution(config, options) {
        return this.executeBackend('/api/v1/evolution/coevolution', config, undefined, options);
    }
    async getProgress(executionId, options) {
        return this.requestBackend('GET', `/api/v1/evolution/progress/${executionId}`, undefined, options);
    }
}
exports.EvolutionIntegration = EvolutionIntegration;
class KnowledgeIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'knowledge', '1.0.0', 'Knowledge Engine: Knowledge graph management', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'query':
                    return await this.executeBackend('/api/v1/knowledge/query', input, executionId, options);
                case 'extract':
                    return await this.executeBackend('/api/v1/knowledge/extract', input, executionId, options);
                case 'search':
                    return await this.executeBackend('/api/v1/knowledge/search', input, executionId, options);
                case 'stats':
                    return await this.requestBackend('GET', '/api/v1/knowledge/stats', undefined, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['query', 'extract', 'search', 'stats'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation'],
        };
    }
    getEndpoints() {
        return ['/api/v1/knowledge/query', '/api/v1/knowledge/extract',
            '/api/v1/knowledge/search', '/api/v1/knowledge/stats'];
    }
    async queryGraph(query, options) {
        return this.executeBackend('/api/v1/knowledge/query', query, undefined, options);
    }
    async extractKnowledge(document, options) {
        return this.executeBackend('/api/v1/knowledge/extract', { document, documentType: 'text' }, undefined, options);
    }
    async searchKnowledge(query, options) {
        return this.executeBackend('/api/v1/knowledge/search', { query, type: 'semantic' }, undefined, options);
    }
    async getGraphStats(options) {
        return this.requestBackend('GET', '/api/v1/knowledge/stats', undefined, options);
    }
}
exports.KnowledgeIntegration = KnowledgeIntegration;
class MakerIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'maker', '1.0.0', 'Maker Engine: Tool creation and execution', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'create':
                    return await this.executeBackend('/api/v1/maker/create', input, executionId, options);
                case 'execute':
                    return await this.executeBackend('/api/v1/maker/execute', input, executionId, options);
                case 'validate':
                    return await this.executeBackend('/api/v1/maker/validate', input, executionId, options);
                case 'list':
                    return await this.requestBackend('GET', '/api/v1/maker/tools', undefined, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['create', 'execute', 'validate', 'list'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation'],
        };
    }
    getEndpoints() {
        return ['/api/v1/maker/create', '/api/v1/maker/execute',
            '/api/v1/maker/validate', '/api/v1/maker/tools'];
    }
    async createTool(config, options) {
        return this.executeBackend('/api/v1/maker/create', config, undefined, options);
    }
    async executeTool(toolId, input, options) {
        return this.executeBackend('/api/v1/maker/execute', { toolId, parameters: input }, undefined, options);
    }
    async validateTool(toolId, options) {
        return this.executeBackend('/api/v1/maker/validate', { toolId, validationType: 'all' }, undefined, options);
    }
}
exports.MakerIntegration = MakerIntegration;
class HephaestusIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'hephaestus', '1.0.0', 'Hephaestus: Task delegation and orchestration', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'delegate':
                    return await this.executeBackend('/api/v1/hephaestus/delegate', input, executionId, options);
                case 'status': {
                    const ticketId = typeof input === 'string' ? input : input.ticketId;
                    return await this.requestBackend('GET', `/api/v1/hephaestus/tickets/${ticketId}`, undefined, options);
                }
                case 'create':
                    return await this.executeBackend('/api/v1/hephaestus/tickets', input, executionId, options);
                case 'list':
                    return await this.requestBackend('GET', '/api/v1/hephaestus/tickets', undefined, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['delegate', 'status', 'create', 'list'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation'],
        };
    }
    getEndpoints() {
        return ['/api/v1/hephaestus/delegate', '/api/v1/hephaestus/tickets'];
    }
    async delegateTask(task, options) {
        return this.executeBackend('/api/v1/hephaestus/delegate', task, undefined, options);
    }
    async getTicketStatus(ticketId, options) {
        return this.requestBackend('GET', `/api/v1/hephaestus/tickets/${ticketId}`, undefined, options);
    }
    async createTicket(ticket, options) {
        return this.executeBackend('/api/v1/hephaestus/tickets', ticket, undefined, options);
    }
}
exports.HephaestusIntegration = HephaestusIntegration;
class DecompositionIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'decomposition', '1.0.0', 'Decomposition: Problem decomposition', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'decompose':
                    return await this.executeBackend('/api/v1/decomposition/decompose', input, executionId, options);
                case 'subproblems': {
                    const planId = typeof input === 'string' ? input : input.planId;
                    return await this.requestBackend('GET', `/api/v1/decomposition/plans/${planId}/subproblems`, undefined, options);
                }
                case 'dependencies': {
                    const planId = typeof input === 'string' ? input : input.planId;
                    return await this.requestBackend('GET', `/api/v1/decomposition/plans/${planId}/dependencies`, undefined, options);
                }
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    enum: ['decompose', 'subproblems', 'dependencies'],
                },
                input: { type: 'object' },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return ['/api/v1/decomposition/decompose', '/api/v1/decomposition/plans'];
    }
    async decompose(problem, strategy, options) {
        return this.executeBackend('/api/v1/decomposition/decompose', { problem, strategy, options: {} }, undefined, options);
    }
    async getSubProblems(planId, options) {
        return this.requestBackend('GET', `/api/v1/decomposition/plans/${planId}/subproblems`, undefined, options);
    }
    async getDependencyGraph(planId, options) {
        return this.requestBackend('GET', `/api/v1/decomposition/plans/${planId}/dependencies`, undefined, options);
    }
}
exports.DecompositionIntegration = DecompositionIntegration;
class VerificationIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'verification', '1.0.0', 'Verification: Solution verification', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'verify':
                    return await this.executeBackend('/api/v1/verification/verify', input, executionId, options);
                case 'checks':
                    return await this.executeBackend('/api/v1/verification/checks', input, executionId, options);
                case 'validate':
                    return await this.executeBackend('/api/v1/verification/validate', input, executionId, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['verify', 'checks', 'validate'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return ['/api/v1/verification/verify', '/api/v1/verification/checks',
            '/api/v1/verification/validate'];
    }
    async verifySolution(solution, requirements, options) {
        return this.executeBackend('/api/v1/verification/verify', { solution, requirements }, undefined, options);
    }
    async runChecks(solution, options) {
        return this.executeBackend('/api/v1/verification/checks', { solution, checkTypes: [] }, undefined, options);
    }
}
exports.VerificationIntegration = VerificationIntegration;
class AssemblyIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'assembly', '1.0.0', 'Assembly: Solution assembly and integration', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'assemble':
                    return await this.executeBackend('/api/v1/assembly/assemble', input, executionId, options);
                case 'integrate':
                    return await this.executeBackend('/api/v1/assembly/integrate', input, executionId, options);
                case 'optimize':
                    return await this.executeBackend('/api/v1/assembly/optimize', input, executionId, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['assemble', 'integrate', 'optimize'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return ['/api/v1/assembly/assemble', '/api/v1/assembly/integrate',
            '/api/v1/assembly/optimize'];
    }
    async assembleSolutions(solutions, options) {
        return this.executeBackend('/api/v1/assembly/assemble', { solutions, strategy: 'dependency-driven' }, undefined, options);
    }
    async integrateSolution(assembledSolution, targetSystem, options) {
        return this.executeBackend('/api/v1/assembly/integrate', { assembledSolution, targetSystem }, undefined, options);
    }
    async optimizeSolution(solution, objectives, options) {
        return this.executeBackend('/api/v1/assembly/optimize', { solution, objectives }, undefined, options);
    }
}
exports.AssemblyIntegration = AssemblyIntegration;
class SolutionIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client, retryConfig, circuitBreakerConfig) {
        super(client, 'solution', '1.0.0', 'Solution: Solution generation and refinement', retryConfig, circuitBreakerConfig);
    }
    async execute(inputs, options) {
        try {
            const validation = await this.validate(inputs);
            if (!validation.valid) {
                throw new errors_1.ValidationError(this.name, validation.errors);
            }
            const { operation, input } = inputs;
            const executionId = options?.executionId;
            switch (operation) {
                case 'generate':
                    return await this.executeBackend('/api/v1/solution/generate', input, executionId, options);
                case 'optimize':
                    return await this.executeBackend('/api/v1/solution/optimize', input, executionId, options);
                case 'refine':
                    return await this.executeBackend('/api/v1/solution/refine', input, executionId, options);
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
        }
        catch (error) {
            const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
            if (options?.fallback !== undefined) {
                return options.fallback;
            }
            throw integrationError;
        }
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                operation: {
                    type: 'string',
                    description: 'Operation to perform',
                    enum: ['generate', 'optimize', 'refine'],
                },
                input: { type: 'object', description: 'Operation-specific input' },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return ['/api/v1/solution/generate', '/api/v1/solution/optimize',
            '/api/v1/solution/refine'];
    }
}
exports.SolutionIntegration = SolutionIntegration;
//# sourceMappingURL=all-integrations.js.map