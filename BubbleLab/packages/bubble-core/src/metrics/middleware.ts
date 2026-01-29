/**
 * Metrics Middleware for Express/Fastify
 *
 * Provides middleware for automatic metrics collection in HTTP servers
 */

import { Request, Response, NextFunction } from 'express';
import {
  bubbleOperationDuration,
  bubbleOperationTotal,
  bubbleErrorTotal,
  bubbleRequestSizeBytes,
  bubbleResponseSizeBytes,
} from './prometheus.js';

/**
 * Express middleware to track HTTP metrics
 */
export function metricsMiddleware(bubbleName: string) {
  return (req: Request, res: Response, next: NextFunction) => {
    const start = Date.now();

    // Track request size
    const contentLength = req.get('content-length');
    if (contentLength) {
      bubbleRequestSizeBytes.observe(
        { bubble: bubbleName, operation: req.route?.path || req.path },
        parseInt(contentLength, 10)
      );
    }

    // Listen for response finish
    res.on('finish', () => {
      const duration = (Date.now() - start) / 1000;
      const status = res.statusCode < 400 ? 'success' : 'error';

      // Record operation duration
      bubbleOperationDuration.observe(
        {
          bubble: bubbleName,
          operation: req.route?.path || req.path,
          status
        },
        duration
      );

      // Record operation total
      bubbleOperationTotal.inc({
        bubble: bubbleName,
        operation: req.route?.path || req.path,
        status
      });

      // Track response size
      const responseLength = res.get('content-length');
      if (responseLength) {
        bubbleResponseSizeBytes.observe(
          { bubble: bubbleName, operation: req.route?.path || req.path },
          parseInt(responseLength, 10)
        );
      }

      // Track errors
      if (res.statusCode >= 400) {
        bubbleErrorTotal.inc({
          bubble: bubbleName,
          error_type: 'http_error',
          operation: req.route?.path || req.path
        });
      }
    });

    next();
  };
}

/**
 * Error handler middleware to track error metrics
 */
export function errorHandlerMiddleware(bubbleName: string) {
  return (err: Error, req: Request, res: Response, next: NextFunction) => {
    bubbleErrorTotal.inc({
      bubble: bubbleName,
      error_type: err.name || 'unknown_error',
      operation: req.route?.path || req.path
    });

    next(err);
  };
}

/**
 * Metrics endpoint middleware
 */
export async function metricsEndpoint(_req: Request, res: Response): Promise<void> {
  const { getMetrics } = await import('./prometheus.js');
  res.set('Content-Type', 'text/plain');
  res.send(await getMetrics());
}
