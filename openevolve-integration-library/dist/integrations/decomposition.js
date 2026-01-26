"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DecompositionIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class DecompositionIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/decomposition');
        this.name = 'decomposition';
        this.version = '1.0.0';
        this.description = 'Problem decomposition and analysis';
    }
    async execute(inputs) {
        const validation = this.validate(inputs);
        if (!validation.valid) {
            throw new Error(`Invalid inputs: ${validation.errors.map(e => e.message).join(', ')}`);
        }
        return this.request('POST', inputs);
    }
    getSchema() {
        return {
            type: 'object',
            properties: {
                problem_statement: {
                    type: 'string',
                    description: 'The problem to decompose'
                },
                method: {
                    type: 'string',
                    description: 'Decomposition method',
                    enum: ['hierarchical', 'hybrid', 'lean4'],
                    default: 'hybrid'
                },
                max_depth: {
                    type: 'number',
                    description: 'Maximum depth of decomposition tree',
                    minimum: 1,
                    maximum: 10,
                    default: 3
                },
                constraints: {
                    type: 'object',
                    description: 'Additional constraints for decomposition'
                }
            },
            required: ['problem_statement']
        };
    }
    async analyzeDependencies(subproblems) {
        return this.client.post(`${this.endpoint}/analyze`, {
            subproblems,
            analysis_type: 'dependencies'
        });
    }
    async validateDecomposition(result) {
        return this.client.post(`${this.endpoint}/validate`, result);
    }
}
exports.DecompositionIntegration = DecompositionIntegration;
//# sourceMappingURL=decomposition.js.map