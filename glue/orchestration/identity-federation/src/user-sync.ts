/**
 * Shadow Account Sync Script
 *
 * Federation Constitution - Identity Federation Strategy (ADR-006)
 * Phase 3: Shadow Account Sync (Last Resort)
 *
 * For services that require local user accounts and don't support OIDC,
 * synchronize users from the central IdP to local service accounts.
 *
 * This script is IDEMPOTENT (Law of Idempotency):
 * - Safe to run multiple times
 * - Updates existing accounts instead of creating duplicates
 * - Handles stale data gracefully
 *
 * Usage:
 *   node sync-users.js [--dry-run] [--service=<service>] [--users=<user1,user2>]
 */

import { logger, LoggerContext } from '../../lib/logger';
import { CircuitBreaker } from '../../lib/circuit-breaker';
import { retryWithBackoff } from '../../lib/retry';

export interface CentralUser {
  sub: string; // Unique ID from IdP
  username: string;
  email: string;
  name?: string;
  groups: string[];
  email_verified?: boolean;
  picture?: string;
  locale?: string;
  updated_at: string; // UTC ISO-8601
}

export interface ShadowAccount {
  remote_id: string; // Maps to central user sub
  username: string;
  email: string;
  display_name?: string;
  groups: string[];
  is_active: boolean;
  last_sync: string; // UTC ISO-8601
  created_at: string; // UTC ISO-8601
  updated_at: string; // UTC ISO-8601
}

export interface ServiceAdapter {
  serviceName: string;
  circuitBreaker?: CircuitBreaker;

  // Check if shadow account exists
  getAccountByRemoteId(remoteId: string): Promise<ShadowAccount | null>;

  // Create new shadow account
  createAccount(account: Omit<ShadowAccount, 'created_at' | 'updated_at' | 'last_sync'>): Promise<ShadowAccount>;

  // Update existing shadow account
  updateAccount(remoteId: string, updates: Partial<ShadowAccount>): Promise<ShadowAccount>;

  // List all shadow accounts
  listAccounts(): Promise<ShadowAccount[]>;

  // Delete shadow account (optional, for cleanup)
  deleteAccount?(remoteId: string): Promise<void>;
}

export interface SyncOptions {
  dryRun?: boolean;
  batchSize?: number;
  continueOnError?: boolean;
  syncDisabledAccounts?: boolean;
}

export interface SyncResult {
  service_name: string;
  total_users: number;
  created: number;
  updated: number;
  skipped: number;
  failed: number;
  errors: Array<{ user: string; error: string }>;
  duration_ms: number;
  timestamp: string;
}

export class ShadowAccountSync {
  private serviceAdapters: Map<string, ServiceAdapter> = new Map();
  private loggerContext: LoggerContext;

  constructor() {
    this.loggerContext = {
      correlation_id: `shadow-sync-${Date.now()}`,
      source_service: 'shadow-account-sync',
    };
  }

  /**
   * Register a service adapter
   */
  registerService(serviceName: string, adapter: ServiceAdapter): void {
    this.serviceAdapters.set(serviceName, adapter);

    logger.info('Registered service adapter', {
      ...this.loggerContext,
      service_name: serviceName,
    });
  }

  /**
   * Sync a single user to a service (idempotent)
   */
  async syncUser(
    serviceName: string,
    centralUser: CentralUser,
    options: SyncOptions = {}
  ): Promise<{ created: boolean; account: ShadowAccount }> {
    const adapter = this.serviceAdapters.get(serviceName);

    if (!adapter) {
      throw new Error(`Service adapter not found: ${serviceName}`);
    }

    logger.info('Syncing user to service', {
      ...this.loggerContext,
      service_name: serviceName,
      user_sub: centralUser.sub,
      username: centralUser.username,
      dry_run: options.dryRun,
    });

    try {
      // Check if shadow account exists
      const existing = await retryWithBackoff(
        () => adapter.getAccountByRemoteId!(centralUser.sub),
        { max_retries: 3, base_delay_ms: 1000 }
      );

      if (existing) {
        // Check if update is needed
        const needsUpdate = this.shouldUpdateAccount(existing, centralUser);

        if (needsUpdate) {
          const updates = this.buildAccountUpdates(centralUser);

          if (options.dryRun) {
            logger.info('[DRY RUN] Would update account', {
              ...this.loggerContext,
              service_name: serviceName,
              remote_id: centralUser.sub,
            });
            return { created: false, account: { ...existing, ...updates } };
          }

          const updated = await retryWithBackoff(
            () => adapter.updateAccount(centralUser.sub, updates),
            { max_retries: 3, base_delay_ms: 1000 }
          );

          logger.info('Updated shadow account', {
            ...this.loggerContext,
            service_name: serviceName,
            remote_id: centralUser.sub,
          });

          return { created: false, account: updated };
        } else {
          logger.debug('Account up to date, skipping', {
            ...this.loggerContext,
            service_name: serviceName,
            remote_id: centralUser.sub,
          });

          return { created: false, account: existing };
        }
      } else {
        // Create new shadow account
        const accountData = this.buildAccountData(centralUser);

        if (options.dryRun) {
          logger.info('[DRY RUN] Would create account', {
            ...this.loggerContext,
            service_name: serviceName,
            username: centralUser.username,
          });
          return { created: true, account: accountData as ShadowAccount };
        }

        const created = await retryWithBackoff(
          () => adapter.createAccount(accountData),
          { max_retries: 3, base_delay_ms: 1000 }
        );

        logger.info('Created shadow account', {
          ...this.loggerContext,
          service_name: serviceName,
          remote_id: centralUser.sub,
        });

        return { created: true, account: created };
      }
    } catch (error) {
      logger.error('Failed to sync user', error as Error, {
        ...this.loggerContext,
        service_name: serviceName,
        user_sub: centralUser.sub,
      });

      throw error;
    }
  }

  /**
   * Sync multiple users to a service
   */
  async syncUsers(
    serviceName: string,
    centralUsers: CentralUser[],
    options: SyncOptions = {}
  ): Promise<SyncResult> {
    const startTime = Date.now();
    const adapter = this.serviceAdapters.get(serviceName);

    if (!adapter) {
      throw new Error(`Service adapter not found: ${serviceName}`);
    }

    logger.info('Starting batch user sync', {
      ...this.loggerContext,
      service_name: serviceName,
      user_count: centralUsers.length,
      dry_run: options.dryRun,
    });

    const result: SyncResult = {
      service_name: serviceName,
      total_users: centralUsers.length,
      created: 0,
      updated: 0,
      skipped: 0,
      failed: 0,
      errors: [],
      duration_ms: 0,
      timestamp: new Date().toISOString(),
    };

    const batchSize = options.batchSize || 10;
    const continueOnError = options.continueOnError || false;

    // Process users in batches
    for (let i = 0; i < centralUsers.length; i += batchSize) {
      const batch = centralUsers.slice(i, i + batchSize);

      logger.debug('Processing batch', {
        ...this.loggerContext,
        service_name: serviceName,
        batch_start: i,
        batch_end: i + batchSize,
      });

      for (const user of batch) {
        try {
          const syncResult = await this.syncUser(serviceName, user, options);

          if (syncResult.created) {
            result.created++;
          } else {
            result.updated++;
          }
        } catch (error) {
          result.failed++;
          result.errors.push({
            user: user.sub,
            error: error instanceof Error ? error.message : String(error),
          });

          if (!continueOnError) {
            logger.error('Batch sync failed, aborting', error as Error, {
              ...this.loggerContext,
              service_name: serviceName,
            });
            break;
          }
        }
      }
    }

    result.duration_ms = Date.now() - startTime;

    logger.info('Batch sync completed', {
      ...this.loggerContext,
      service_name: serviceName,
      created: result.created,
      updated: result.updated,
      skipped: result.skipped,
      failed: result.failed,
      duration_ms: result.duration_ms,
    });

    return result;
  }

  /**
   * Sync all users to all services
   */
  async syncAllServices(
    centralUsers: CentralUser[],
    options: SyncOptions = {}
  ): Promise<SyncResult[]> {
    const results: SyncResult[] = [];

    for (const serviceName of this.serviceAdapters.keys()) {
      try {
        const result = await this.syncUsers(serviceName, centralUsers, options);
        results.push(result);
      } catch (error) {
        logger.error('Failed to sync service', error as Error, {
          ...this.loggerContext,
          service_name: serviceName,
        });

        results.push({
          service_name: serviceName,
          total_users: centralUsers.length,
          created: 0,
          updated: 0,
          skipped: 0,
          failed: centralUsers.length,
          errors: [{ user: 'all', error: error instanceof Error ? error.message : String(error) }],
          duration_ms: 0,
          timestamp: new Date().toISOString(),
        });
      }
    }

    return results;
  }

  /**
   * Check if account needs updating
   */
  private shouldUpdateAccount(existing: ShadowAccount, centralUser: CentralUser): boolean {
    // Always update if central user was updated more recently
    const centralUpdatedAt = new Date(centralUser.updated_at).getTime();
    const localUpdatedAt = new Date(existing.updated_at).getTime();

    if (centralUpdatedAt > localUpdatedAt) {
      return true;
    }

    // Check if critical fields differ
    if (existing.username !== centralUser.username) return true;
    if (existing.email !== centralUser.email) return true;

    // Check if groups changed
    const existingGroups = existing.groups.sort().join(',');
    const centralGroups = centralUser.groups.sort().join(',');

    if (existingGroups !== centralGroups) {
      return true;
    }

    return false;
  }

  /**
   * Build account data for creation
   */
  private buildAccountData(centralUser: CentralUser): Omit<ShadowAccount, 'created_at' | 'updated_at' | 'last_sync'> {
    const now = new Date().toISOString();

    return {
      remote_id: centralUser.sub,
      username: centralUser.username,
      email: centralUser.email,
      display_name: centralUser.name || centralUser.username,
      groups: centralUser.groups,
      is_active: true,
      last_sync: now,
    };
  }

  /**
   * Build account updates
   */
  private buildAccountUpdates(centralUser: CentralUser): Partial<ShadowAccount> {
    return {
      username: centralUser.username,
      email: centralUser.email,
      display_name: centralUser.name || centralUser.username,
      groups: centralUser.groups,
      last_sync: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  }

  /**
   * Cleanup stale shadow accounts
   * Removes accounts that haven't been synced in a while
   */
  async cleanupStaleAccounts(
    serviceName: string,
    staleThresholdMs: number = 30 * 24 * 60 * 60 * 1000 // 30 days
  ): Promise<number> {
    const adapter = this.serviceAdapters.get(serviceName);

    if (!adapter) {
      throw new Error(`Service adapter not found: ${serviceName}`);
    }

    logger.info('Cleaning up stale accounts', {
      ...this.loggerContext,
      service_name: serviceName,
      stale_threshold_ms: staleThresholdMs,
    });

    const accounts = await adapter.listAccounts();
    const now = Date.now();
    let cleaned = 0;

    for (const account of accounts) {
      const lastSync = new Date(account.last_sync).getTime();
      const staleAge = now - lastSync;

      if (staleAge > staleThresholdMs && adapter.deleteAccount) {
        try {
          await adapter.deleteAccount(account.remote_id);
          cleaned++;

          logger.info('Deleted stale account', {
            ...this.loggerContext,
            service_name: serviceName,
            remote_id: account.remote_id,
            stale_age_days: Math.floor(staleAge / (24 * 60 * 60 * 1000)),
          });
        } catch (error) {
          logger.error('Failed to delete stale account', error as Error, {
            ...this.loggerContext,
            service_name: serviceName,
            remote_id: account.remote_id,
          });
        }
      }
    }

    logger.info('Cleanup completed', {
      ...this.loggerContext,
      service_name: serviceName,
      cleaned,
    });

    return cleaned;
  }
}

/**
 * Example usage:
 *
 * ```typescript
 * // Create sync instance
 * const sync = new ShadowAccountSync();
 *
 * // Register service adapters
 * sync.registerService('graphiti', new GraphitiAdapter());
 * sync.registerService('bubblelab', new BubbleLabAdapter());
 *
 * // Get users from central IdP
 * const centralUsers = await fetchUsersFromKeycloak();
 *
 * // Sync to specific service
 * const result = await sync.syncUsers('graphiti', centralUsers, {
 *   dryRun: false,
 *   batchSize: 50,
 *   continueOnError: true,
 * });
 *
 * // Or sync to all services
 * const results = await sync.syncAllServices(centralUsers);
 *
 * // Cleanup stale accounts
 * await sync.cleanupStaleAccounts('graphiti', 90 * 24 * 60 * 60 * 1000); // 90 days
 * ```
 */
