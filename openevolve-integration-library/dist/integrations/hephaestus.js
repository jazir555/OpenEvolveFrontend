"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HephaestusIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class HephaestusIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/hephaestus');
        this.name = 'hephaestus';
        this.version = '1.0.0';
        this.description = 'Delegation and orchestration';
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
                    enum: ['delegate', 'orchestrate', 'monitor'],
                    default: 'delegate'
                },
                task: {
                    type: 'string',
                    description: 'Task description or ID'
                },
                agent_type: {
                    type: 'string',
                    description: 'Type of agent to delegate to',
                    enum: ['specialist', 'generalist', 'hybrid']
                },
                constraints: {
                    type: 'object',
                    description: 'Constraints for task execution'
                },
                workflow: {
                    type: 'object',
                    description: 'Workflow specification for orchestration'
                },
                session_id: {
                    type: 'string',
                    description: 'Session ID for monitoring'
                }
            },
            required: ['mode']
        };
    }
    async delegate(task, agentType = 'specialist', constraints) {
        return this.execute({
            mode: 'delegate',
            task,
            agent_type: agentType,
            constraints
        });
    }
    async orchestrate(workflow) {
        return this.execute({
            mode: 'orchestrate',
            workflow
        });
    }
    async monitor(sessionId) {
        return this.execute({
            mode: 'monitor',
            session_id: sessionId
        });
    }
    async cancelTask(taskId) {
        await this.client.delete(`${this.endpoint}/tasks/${taskId}`);
    }
    async getSessionStatus(sessionId) {
        return this.client.get(`${this.endpoint}/sessions/${sessionId}`);
    }
}
exports.HephaestusIntegration = HephaestusIntegration;
//# sourceMappingURL=hephaestus.js.map