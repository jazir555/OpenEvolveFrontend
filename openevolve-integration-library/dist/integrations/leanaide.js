"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAideIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class LeanAideIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/leanaide');
        this.name = 'leanaide';
        this.version = '1.0.0';
        this.description = 'Formal verification, MCTS, and MDAP';
    }
    async execute(inputs) {
        const validation = this.validate(inputs);
        if (!validation.valid) {
            throw new Error(`Invalid inputs: ${validation.errors.map(e => e.message).join(', ')}`);
        }
        const endpoint = `${this.endpoint}/${inputs.mode}`;
        return this.client.post(endpoint, inputs);
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                mode: {
                    type: 'string',
                    description: 'Operation mode',
                    enum: ['formal_verification', 'mcts', 'mdap'],
                    default: 'formal_verification'
                },
                problem: {
                    type: 'string',
                    description: 'Problem statement or lemma to prove'
                },
                tactics: {
                    type: 'array',
                    description: 'List of tactics to use (for formal_verification)',
                    items: { type: 'string' }
                },
                iterations: {
                    type: 'number',
                    description: 'Number of iterations (for MCTS)',
                    minimum: 1,
                    maximum: 100000,
                    default: 1000
                },
                constraints: {
                    type: 'object',
                    description: 'Constraints for optimization (for MDAP)'
                }
            },
            required: ['mode', 'problem']
        };
    }
    async verify(lemma, tactics) {
        return this.execute({
            mode: 'formal_verification',
            problem: lemma,
            tactics
        });
    }
    async plan(problem, iterations = 1000) {
        return this.execute({
            mode: 'mcts',
            problem,
            iterations
        });
    }
    async optimize(problem, constraints) {
        return this.execute({
            mode: 'mdap',
            problem,
            constraints
        });
    }
}
exports.LeanAideIntegration = LeanAideIntegration;
//# sourceMappingURL=leanaide.js.map