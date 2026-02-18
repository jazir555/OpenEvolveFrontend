import { ApiHttpError } from './apiTypes';
export class LeanAideClient {
    constructor(config) {
        this.config = config;
        this.baseURL = config.serverUrl;
    }
    async request(endpoint, method = 'POST', data) {
        const url = `${this.baseURL}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
        };
        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }
        const options = {
            method,
            headers,
        };
        if (data) {
            options.body = JSON.stringify(data);
        }
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new ApiHttpError(response.status, errorData);
            }
            return response.json();
        }
        catch (error) {
            if (error instanceof ApiHttpError) {
                throw error;
            }
            throw new ApiHttpError(500, {
                message: error instanceof Error ? error.message : 'Network error',
            });
        }
    }
    async translateTheorem(theoremStatement, context) {
        return this.request('/translate-thm', 'POST', {
            taskType: 'translate_thm',
            input: theoremStatement,
            context,
        });
    }
    async translateDefinition(definitionStatement, context) {
        return this.request('/translate-def', 'POST', {
            taskType: 'translate_def',
            input: definitionStatement,
            context,
        });
    }
    async verifySolution(problem, solution, context) {
        return this.request('/verify', 'POST', {
            taskType: 'prove_for_formalization',
            input: problem,
            solution,
            context,
        });
    }
    async elaborateCode(leanCode, context) {
        return this.request('/elaborate', 'POST', {
            taskType: 'elaborate',
            input: leanCode,
            context,
        });
    }
    async mathQuery(query, context) {
        return this.request('/math-query', 'POST', {
            taskType: 'math_query',
            input: query,
            context,
        });
    }
}
//# sourceMappingURL=leanaideClient.js.map