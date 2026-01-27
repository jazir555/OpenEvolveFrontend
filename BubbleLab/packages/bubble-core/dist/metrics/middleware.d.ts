/**
 * Metrics Middleware for Express/Fastify
 *
 * Provides middleware for automatic metrics collection in HTTP servers
 */
import { Request, Response, NextFunction } from 'express';
/**
 * Express middleware to track HTTP metrics
 */
export declare function metricsMiddleware(bubbleName: string): (req: Request, res: Response, next: NextFunction) => void;
/**
 * Error handler middleware to track error metrics
 */
export declare function errorHandlerMiddleware(bubbleName: string): (err: Error, req: Request, res: Response, next: NextFunction) => void;
/**
 * Metrics endpoint middleware
 */
export declare function metricsEndpoint(_req: Request, res: Response): Promise<void>;
//# sourceMappingURL=middleware.d.ts.map