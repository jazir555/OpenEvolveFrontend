"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RagbitsClient = void 0;
const apiTypes_1 = require("./apiTypes");
class RagbitsClient {
    constructor(config) {
        this.serverUrl = config.serverUrl;
        this.apiKey = config.apiKey;
    }
    async search(request) {
        return this.request('/search', request);
    }
    async ingest(request) {
        return this.request('/ingest', request);
    }
    async request(path, payload) {
        const url = `${this.serverUrl}${path}`;
        const headers = {
            'Content-Type': 'application/json',
        };
        if (this.apiKey) {
            headers.Authorization = `Bearer ${this.apiKey}`;
        }
        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload),
        });
        let data = null;
        try {
            data = await response.json();
        }
        catch {
            data = { error: 'Invalid JSON response from RAGBits server' };
        }
        if (!response.ok) {
            throw new apiTypes_1.ApiHttpError(response.status, data);
        }
        return data;
    }
}
exports.RagbitsClient = RagbitsClient;
//# sourceMappingURL=ragbitsClient.js.map