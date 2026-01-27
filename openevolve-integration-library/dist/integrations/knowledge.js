"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.KnowledgeIntegration = void 0;
const BaseIntegration_1 = require("../base/BaseIntegration");
class KnowledgeIntegration extends BaseIntegration_1.BaseIntegration {
    constructor(client) {
        super(client, '/api/v1/knowledge');
        this.name = 'knowledge';
        this.version = '1.0.0';
        this.description = 'Knowledge graphs and information extraction';
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
                    enum: ['extraction', 'query', 'update'],
                    default: 'extraction'
                },
                source: {
                    type: 'object',
                    description: 'Source document or data (for extraction)'
                },
                query: {
                    type: 'string',
                    description: 'Query string (for query mode)'
                },
                graph_id: {
                    type: 'string',
                    description: 'Knowledge graph ID (for query and update)'
                },
                extraction_type: {
                    type: 'string',
                    description: 'Type of extraction to perform',
                    enum: ['entities', 'relations', 'entities_relations']
                },
                additions: {
                    type: 'array',
                    description: 'Items to add (for update)',
                    items: {}
                },
                deletions: {
                    type: 'array',
                    description: 'Items to delete (for update)',
                    items: {}
                }
            },
            required: ['mode']
        };
    }
    async extract(source, extractionType = 'entities_relations') {
        return this.execute({
            mode: 'extraction',
            source,
            extraction_type: extractionType
        });
    }
    async queryGraph(graphId, query) {
        return this.execute({
            mode: 'query',
            graph_id: graphId,
            query
        });
    }
    async updateGraph(graphId, additions, deletions) {
        return this.execute({
            mode: 'update',
            graph_id: graphId,
            additions,
            deletions
        });
    }
}
exports.KnowledgeIntegration = KnowledgeIntegration;
//# sourceMappingURL=knowledge.js.map