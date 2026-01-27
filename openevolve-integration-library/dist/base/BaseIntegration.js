"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseIntegration = void 0;
const helpers_1 = require("../utils/helpers");
class BaseIntegration {
    constructor(client, endpoint) {
        this.client = client;
        this.endpoint = endpoint;
    }
    validate(inputs) {
        const schema = this.getSchema();
        return (0, helpers_1.validateInputs)(inputs, schema);
    }
    async executeStream(inputs, onUpdate) {
        return this.execute(inputs);
    }
    async request(method, data) {
        try {
            switch (method) {
                case 'GET':
                    return this.client.get(this.endpoint, data);
                case 'POST':
                    return this.client.post(this.endpoint, data);
                case 'PUT':
                    return this.client.put(this.endpoint, data);
                case 'DELETE':
                    return this.client.delete(this.endpoint);
                default:
                    throw new Error(`Unsupported method: ${method}`);
            }
        }
        catch (error) {
            this.handleError(error);
        }
    }
    handleError(error) {
        if (error instanceof Error) {
            throw error;
        }
        throw new Error(`Unknown error in ${this.name}: ${error}`);
    }
    getMetadata() {
        return {
            name: this.name,
            version: this.version,
            description: this.description,
            endpoint: this.endpoint
        };
    }
}
exports.BaseIntegration = BaseIntegration;
//# sourceMappingURL=BaseIntegration.js.map