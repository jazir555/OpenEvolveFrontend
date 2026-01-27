"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MakerIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class MakerIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/maker');
        this.name = 'maker';
        this.version = '1.0.0';
        this.description = 'Tool and workflow creation';
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
                    enum: ['create_tool', 'create_workflow', 'execute'],
                    default: 'create_tool'
                },
                specification: {
                    type: 'object',
                    description: 'Tool or workflow specification'
                },
                workflow: {
                    type: 'object',
                    description: 'Workflow specification'
                },
                tool_id: {
                    type: 'string',
                    description: 'Tool ID (for execute mode)'
                },
                inputs: {
                    type: 'object',
                    description: 'Inputs for tool execution'
                }
            },
            required: ['mode']
        };
    }
    async createTool(specification) {
        return this.execute({
            mode: 'create_tool',
            specification
        });
    }
    async createWorkflow(specification) {
        return this.execute({
            mode: 'create_workflow',
            workflow: specification
        });
    }
    async executeTool(toolId, inputs) {
        return this.execute({
            mode: 'execute',
            tool_id: toolId,
            inputs
        });
    }
    async listTools() {
        return this.client.get(`${this.endpoint}/tools`);
    }
    async getTool(toolId) {
        return this.client.get(`${this.endpoint}/tools/${toolId}`);
    }
}
exports.MakerIntegration = MakerIntegration;
//# sourceMappingURL=maker.js.map