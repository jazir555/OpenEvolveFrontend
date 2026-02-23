"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ApiHttpError = void 0;
class ApiHttpError extends Error {
    constructor(status, data) {
        super(`HTTP ${status}: ${JSON.stringify(data)}`);
        this.name = 'ApiHttpError';
        this.status = status;
        this.data = data;
    }
}
exports.ApiHttpError = ApiHttpError;
//# sourceMappingURL=apiTypes.js.map