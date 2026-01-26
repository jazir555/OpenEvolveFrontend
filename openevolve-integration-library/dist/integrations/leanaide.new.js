"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAideIntegration = void 0;
const base_1 = require("./base");
class LeanAideIntegration extends base_1.BaseIntegrationAdapter {
    constructor(client) {
        super(client, 'leanaide', '1.0.0', 'LeanAide: Formal mathematics theorem proving and verification');
    }
    async execute(inputs, options) {
        const leanAideInputs = inputs;
        const validation = await this.validate(leanAideInputs);
        if (!validation.valid) {
            throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
        }
        switch (leanAideInputs.operation) {
            case 'translate':
                return await this.translateTheorem(leanAideInputs.input.theorem);
            case 'prove':
                const proofInput = leanAideInputs.input;
                return await this.generateProof(proofInput.theorem, proofInput.strategy);
            case 'verify':
                return await this.verifyProof(leanAideInputs.input.proof);
            case 'mcts':
                const mctsInput = leanAideInputs.input;
                return await this.runMCTS(mctsInput.problem, mctsInput.config);
            case 'query':
                const queryInput = leanAideInputs.input;
                return await this.queryMath(queryInput.question);
            default:
                throw new Error(`Unknown operation: ${leanAideInputs.operation}`);
        }
    }
    async validate(inputs) {
        const leanAideInputs = inputs;
        const errors = [];
        const warnings = [];
        const validOperations = ['translate', 'prove', 'verify', 'mcts', 'query'];
        if (!validOperations.includes(leanAideInputs.operation)) {
            errors.push(`Invalid operation: ${leanAideInputs.operation}. Must be one of: ${validOperations.join(', ')}`);
        }
        if (!leanAideInputs.input) {
            errors.push('Input is required');
        }
        switch (leanAideInputs.operation) {
            case 'translate':
                const transInput = leanAideInputs.input;
                if (!transInput.theorem) {
                    errors.push('Theorem is required for translation');
                }
                break;
            case 'prove':
                const proofInput = leanAideInputs.input;
                if (!proofInput.theorem) {
                    errors.push('Theorem is required for proof generation');
                }
                if (!proofInput.strategy) {
                    warnings.push('No strategy specified, using default');
                }
                if (proofInput.timeout && proofInput.timeout < 0) {
                    errors.push('Timeout must be positive');
                }
                break;
            case 'mcts':
                const mctsInput = leanAideInputs.input;
                if (!mctsInput.problem) {
                    errors.push('Problem is required for MCTS');
                }
                if (!mctsInput.config) {
                    errors.push('MCTS config is required');
                }
                else {
                    const config = mctsInput.config;
                    if (config.simulations <= 0) {
                        errors.push('Simulations must be positive');
                    }
                    if (config.explorationConstant && config.explorationConstant < 0) {
                        errors.push('Exploration constant must be non-negative');
                    }
                }
                break;
            case 'query':
                const queryInput = leanAideInputs.input;
                if (!queryInput.question) {
                    errors.push('Question is required for math query');
                }
                break;
        }
        return this.createValidationResult(errors, warnings);
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
                    description: 'Operation-specific input data',
                },
                config: {
                    type: 'object',
                    description: 'Execution configuration',
                    properties: {
                        stream: {
                            type: 'boolean',
                            description: 'Whether to stream progress updates',
                        },
                        timeout: {
                            type: 'number',
                            description: 'Execution timeout in milliseconds',
                        },
                    },
                },
            },
            required: ['operation', 'input'],
        };
    }
    getEndpoints() {
        return [
            '/api/v1/leanaide/translate',
            '/api/v1/leanaide/prove',
            '/api/v1/leanaide/verify',
            '/api/v1/leanaide/mcts',
            '/api/v1/leanaide/query',
            '/api/v1/leanaide/execute',
        ];
    }
    async translateTheorem(input) {
        return this.executeBackend('/api/v1/leanaide/translate', { theorem: input });
    }
    async generateProof(theorem, strategy) {
        return this.executeBackend('/api/v1/leanaide/prove', {
            theorem,
            strategy,
            tactics: [],
            context: [],
        });
    }
    async verifyProof(proof) {
        return this.executeBackend('/api/v1/leanaide/verify', { proof });
    }
    async runMCTS(problem, config) {
        return this.executeBackend('/api/v1/leanaide/mcts', { problem, config });
    }
    async queryMath(question) {
        return this.executeBackend('/api/v1/leanaide/query', { question });
    }
    async streamGenerateProof(theorem, strategy, onProgress, options) {
        return this.streamExecute('/api/v1/leanaide/prove', { theorem, strategy, tactics: [], context: [] }, onProgress, options);
    }
}
exports.LeanAideIntegration = LeanAideIntegration;
//# sourceMappingURL=leanaide.new.js.map