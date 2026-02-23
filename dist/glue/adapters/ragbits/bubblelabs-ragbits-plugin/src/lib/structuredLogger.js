"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ragbitsLogger = void 0;
function asPayload(context, error) {
    const payload = context ? { ...context } : {};
    if (error) {
        payload.error = error.message;
        payload.stack = error.stack;
    }
    return payload;
}
exports.ragbitsLogger = {
    info(message, context) {
        console.info(message, asPayload(context));
    },
    warn(message, context) {
        console.warn(message, asPayload(context));
    },
    error(message, error, context) {
        console.error(message, asPayload(context, error));
    },
};
//# sourceMappingURL=structuredLogger.js.map