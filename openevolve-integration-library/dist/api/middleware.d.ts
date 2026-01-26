import { Middleware } from './types';
export declare const loggingMiddleware: Middleware;
export declare const createCachingMiddleware: (ttlMs?: number, maxSize?: number) => Middleware & {
    clear: () => void;
};
//# sourceMappingURL=middleware.d.ts.map