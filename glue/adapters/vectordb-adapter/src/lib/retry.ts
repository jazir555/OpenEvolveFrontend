/**
 * Retry with jitter - resilient retries against transient vector DB failures.
 */

import { Logger } from './logger';

export async function retryWithJitter(
  fn: () => Promise<void>,
  maxRetries: number,
  logger?: Logger
): Promise<void> {
  let attempt = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      await fn();
      return;
    } catch (error) {
      attempt += 1;
      if (attempt > maxRetries) {
        throw error;
      }
      const base = Math.min(1000, 25 * 2 ** (attempt - 1));
      const jitter = Math.random() * base;
      logger?.warn('Retrying after transient failure', {
        attempt,
        maxRetries,
        delay_ms: Math.round(base + jitter),
        error: (error as Error)?.message,
      });
      await new Promise((resolve) => setTimeout(resolve, base + jitter));
    }
  }
}
