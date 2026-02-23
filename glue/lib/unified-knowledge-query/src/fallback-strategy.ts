/**
 * Fallback Strategy
 *
 * Implements graceful degradation when systems are unavailable.
 *
 * Federation Constitution Compliance:
 * - Failure Management: Transient failures trigger fallback
 * - Circuit Breaker: System failures trigger circuit opening
 * - Law of Idempotency: Fallback queries are safe to retry
 */

import { Logger, CircuitBreaker } from '@openevolve/glue-lib';

import {
  SystemConfig,
  SystemSource,
  QueryPlan,
  KnowledgeItem,
  SystemHealth,
  HealthStatus,
} from './canonical';
import { RAGBitsClient, GraphitiClient, VectorDBClient } from './clients';

/**
 * Fallback Configuration
 */
interface FallbackConfig {
  maxAttempts: number;
  retryDelayMs: number;
  enableFallback: boolean;
  fallbackPriority: SystemSource[];
}

/**
 * Fallback Execution Result
 */
interface FallbackResult {
  items: KnowledgeItem[];
  systemUsed: SystemSource;
  attempt: number;
  wasFallback: boolean;
}

/**
 * Fallback Strategy Class
 */
export class FallbackStrategy {
  private logger: Logger;
  private config: FallbackConfig;
  private systemHealth: Map<SystemSource, SystemHealth>;

  constructor(config?: Partial<FallbackConfig>) {
    this.logger = new Logger('fallback-strategy');
    this.systemHealth = new Map();

    this.config = {
      maxAttempts: 3,
      retryDelayMs: 1000,
      enableFallback: true,
      fallbackPriority: ['vectordb', 'ragbits', 'graphiti'],
      ...config,
    };
  }

  /**
   * Execute query with fallback strategy
   */
  async executeWithFallback(
    plan: QueryPlan,
    primary: SystemConfig,
    fallbacks: SystemConfig[],
    query: string,
    options: any = {}
  ): Promise<FallbackResult> {
    this.logger.info('Executing with fallback', {
      correlation_id: options.correlationId,
      primary: primary.name,
      fallbacks: fallbacks.map(f => f.name),
    });

    // Try primary first
    try {
      const items = await this.trySystem(primary, query, options);

      return {
        items,
        systemUsed: primary.name,
        attempt: 1,
        wasFallback: false,
      };
    } catch (error) {
      this.logger.warn('Primary system failed', {
        correlation_id: options.correlationId,
        primary: primary.name,
        error: (error as Error).message,
      });

      // Check if we should fallback
      if (!this.shouldFallback(error as Error) || !this.config.enableFallback) {
        throw error;
      }

      // Try fallbacks in priority order
      for (let i = 0; i < fallbacks.length; i++) {
        const fallback = fallbacks[i];

        this.logger.info('Trying fallback system', {
          correlation_id: options.correlationId,
          fallback: fallback.name,
          attempt: i + 2,
        });

        try {
          // Add delay before fallback
          await this.delay(this.config.retryDelayMs * (i + 1));

          const items = await this.trySystem(fallback, query, options);

          this.logger.info('Fallback system succeeded', {
            correlation_id: options.correlationId,
            fallback: fallback.name,
            attempt: i + 2,
          });

          return {
            items,
            systemUsed: fallback.name,
            attempt: i + 2,
            wasFallback: true,
          };
        } catch (fallbackError) {
          this.logger.warn('Fallback system failed', {
            correlation_id: options.correlationId,
            fallback: fallback.name,
            attempt: i + 2,
            error: (fallbackError as Error).message,
          });

          // Continue to next fallback
        }
      }

      // All systems failed
      const allFailedError = new Error(
        `All systems failed. Primary: ${primary.name}, Fallbacks: ${fallbacks.map(f => f.name).join(', ')}`
      );

      this.logger.error('All systems failed', allFailedError, {
        correlation_id: options.correlationId,
      });

      throw allFailedError;
    }
  }

  /**
   * Execute query against multiple systems in parallel
   * Use first successful response
   */
  async executeParallel(
    systems: SystemConfig[],
    query: string,
    options: any = {}
  ): Promise<FallbackResult> {
    this.logger.info('Executing parallel queries', {
      correlation_id: options.correlationId,
      systems: systems.map(s => s.name),
    });

    const promises = systems.map((system, index) =>
      this.trySystem(system, query, options)
        .then(items => ({
          items,
          systemUsed: system.name,
          attempt: index + 1,
          wasFallback: index > 0,
        }))
        .catch(error => {
          this.logger.warn('Parallel query failed', {
            correlation_id: options.correlationId,
            system: system.name,
            error: (error as Error).message,
          });
          throw error;
        })
    );

    try {
      // Use Promise.any to get first successful result
      const result = await Promise.any(promises);

      this.logger.info('Parallel query succeeded', {
        correlation_id: options.correlationId,
        system_used: result.systemUsed,
      });

      return result;
    } catch (error) {
      // All promises failed
      this.logger.error('All parallel queries failed', error as Error, {
        correlation_id: options.correlationId,
      });
      throw error;
    }
  }

  /**
   * Try executing query against a system
   */
  private async trySystem(
    system: SystemConfig,
    query: string,
    options: any
  ): Promise<KnowledgeItem[]> {
    const client = this.createClient(system);

    // Check circuit breaker
    const health = this.systemHealth.get(system.name);
    if (health && health.status === 'unhealthy') {
      throw new Error(`System ${system.name} is unhealthy`);
    }

    // Execute query
    const items = await client.search(query, options);

    return items;
  }

  /**
   * Create client for system
   */
  private createClient(system: SystemConfig): any {
    switch (system.name) {
      case 'ragbits':
        return new RAGBitsClient(system);
      case 'graphiti':
        return new GraphitiClient(system);
      case 'vectordb':
        return new VectorDBClient(system);
      default:
        throw new Error(`Unknown system: ${system.name}`);
    }
  }

  /**
   * Determine if error should trigger fallback
   */
  private shouldFallback(error: Error): boolean {
    // Fallback on network errors, timeouts, and 5xx errors
    const fallbackPatterns = [
      /network/i,
      /timeout/i,
      /ECONNREFUSED/i,
      /ETIMEDOUT/i,
      /502/i,
      /503/i,
      /504/i,
      /circuit.*open/i,
    ];

    for (const pattern of fallbackPatterns) {
      if (pattern.test(error.message)) {
        return true;
      }
    }

    // Fallback on circuit breaker errors
    if (error.message.includes('Circuit breaker is OPEN')) {
      return true;
    }

    return false;
  }

  /**
   * Select best fallback from available systems
   */
  selectFallback(available: SystemConfig[]): SystemConfig {
    // Sort by priority and health
    const sorted = [...available].sort((a, b) => {
      // First by priority (higher first)
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }

      // Then by health status
      const aHealth = this.systemHealth.get(a.name);
      const bHealth = this.systemHealth.get(b.name);

      const aScore = this.getHealthScore(aHealth?.status);
      const bScore = this.getHealthScore(bHealth?.status);

      return bScore - aScore;
    });

    return sorted[0];
  }

  /**
   * Get numeric score for health status
   */
  private getHealthScore(status?: HealthStatus): number {
    switch (status) {
      case 'healthy':
        return 3;
      case 'degraded':
        return 2;
      case 'unknown':
        return 1;
      case 'unhealthy':
        return 0;
      default:
        return 1;
    }
  }

  /**
   * Update system health
   */
  updateSystemHealth(health: SystemHealth[]): void {
    for (const h of health) {
      this.systemHealth.set(h.system, h);
    }

    this.logger.debug('System health updated in fallback strategy', {
      systems: health.map(h => ({
        system: h.system,
        status: h.status,
      })),
    });
  }

  /**
   * Get available fallbacks
   */
  getAvailableFallbacks(allSystems: SystemConfig[]): SystemConfig[] {
    return allSystems
      .filter(system => {
        const health = this.systemHealth.get(system.name);
        return system.enabled && (!health || health.status !== 'unhealthy');
      })
      .sort((a, b) => {
        // Sort by fallback priority
        const aIndex = this.config.fallbackPriority.indexOf(a.name);
        const bIndex = this.config.fallbackPriority.indexOf(b.name);

        // If not in priority list, put at end
        const aPriority = aIndex === -1 ? 999 : aIndex;
        const bPriority = bIndex === -1 ? 999 : bPriority;

        return aPriority - bPriority;
      });
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current config
   */
  getConfig(): FallbackConfig {
    return { ...this.config };
  }

  /**
   * Update config
   */
  updateConfig(updates: Partial<FallbackConfig>): void {
    this.config = {
      ...this.config,
      ...updates,
    };

    this.logger.info('Fallback config updated', updates);
  }

  /**
   * Reset health status
   */
  reset(): void {
    this.systemHealth.clear();
    this.logger.info('Fallback strategy reset');
  }
}

/**
 * Default fallback strategy instance
 */
export const fallbackStrategy = new FallbackStrategy();
