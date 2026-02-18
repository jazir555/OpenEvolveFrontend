export class ApiHttpError extends Error {
    constructor(status, data) {
        super(`HTTP ${status}: ${JSON.stringify(data)}`);
        this.name = 'ApiHttpError';
        this.status = status;
        this.data = data;
    }
}
//# sourceMappingURL=apiTypes.js.map