function toRecord(context) {
    return context ? { ...context } : {};
}
export const logger = {
    info(message, context) {
        console.info(message, toRecord(context));
    },
    warn(message, context) {
        console.warn(message, toRecord(context));
    },
    error(message, error, context) {
        const payload = toRecord(context);
        if (error) {
            payload.error = error.message;
            payload.stack = error.stack;
        }
        console.error(message, payload);
    },
};
//# sourceMappingURL=structuredLogger.js.map