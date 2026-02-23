"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShadowAccountSync = void 0;
const logger_1 = require("../../lib/logger");
const retry_1 = require("../../lib/retry");
class ShadowAccountSync {
    constructor() {
        this.serviceAdapters = new Map();
        this.loggerContext = {
            correlation_id: `shadow-sync-${Date.now()}`,
            source_service: 'shadow-account-sync',
        };
    }
    /**
     * Register a service adapter
     */
    registerService(serviceName, adapter) {
        this.serviceAdapters.set(serviceName, adapter);
        logger_1.logger.info('Registered service adapter', {
            ...this.loggerContext,
            service_name: serviceName,
        });
    }
    /**
     * Sync a single user to a service (idempotent)
     */
    async syncUser(serviceName, centralUser, options = {}) {
        const adapter = this.serviceAdapters.get(serviceName);
        if (!adapter) {
            throw new Error(`Service adapter not found: ${serviceName}`);
        }
        logger_1.logger.info('Syncing user to service', {
            ...this.loggerContext,
            service_name: serviceName,
            user_sub: centralUser.sub,
            username: centralUser.username,
            dry_run: options.dryRun,
        });
        try {
            // Check if shadow account exists
            const existing = await (0, retry_1.retryWithBackoff)(() => adapter.getAccountByRemoteId(centralUser.sub), { max_retries: 3, base_delay_ms: 1000 });
            if (existing) {
                // Check if update is needed
                const needsUpdate = this.shouldUpdateAccount(existing, centralUser);
                if (needsUpdate) {
                    const updates = this.buildAccountUpdates(centralUser);
                    if (options.dryRun) {
                        logger_1.logger.info('[DRY RUN] Would update account', {
                            ...this.loggerContext,
                            service_name: serviceName,
                            remote_id: centralUser.sub,
                        });
                        return { created: false, account: { ...existing, ...updates } };
                    }
                    const updated = await (0, retry_1.retryWithBackoff)(() => adapter.updateAccount(centralUser.sub, updates), { max_retries: 3, base_delay_ms: 1000 });
                    logger_1.logger.info('Updated shadow account', {
                        ...this.loggerContext,
                        service_name: serviceName,
                        remote_id: centralUser.sub,
                    });
                    return { created: false, account: updated };
                }
                else {
                    logger_1.logger.debug('Account up to date, skipping', {
                        ...this.loggerContext,
                        service_name: serviceName,
                        remote_id: centralUser.sub,
                    });
                    return { created: false, account: existing };
                }
            }
            else {
                // Create new shadow account
                const accountData = this.buildAccountData(centralUser);
                if (options.dryRun) {
                    logger_1.logger.info('[DRY RUN] Would create account', {
                        ...this.loggerContext,
                        service_name: serviceName,
                        username: centralUser.username,
                    });
                    return { created: true, account: accountData };
                }
                const created = await (0, retry_1.retryWithBackoff)(() => adapter.createAccount(accountData), { max_retries: 3, base_delay_ms: 1000 });
                logger_1.logger.info('Created shadow account', {
                    ...this.loggerContext,
                    service_name: serviceName,
                    remote_id: centralUser.sub,
                });
                return { created: true, account: created };
            }
        }
        catch (error) {
            logger_1.logger.error('Failed to sync user', error, {
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
    async syncUsers(serviceName, centralUsers, options = {}) {
        const startTime = Date.now();
        const adapter = this.serviceAdapters.get(serviceName);
        if (!adapter) {
            throw new Error(`Service adapter not found: ${serviceName}`);
        }
        logger_1.logger.info('Starting batch user sync', {
            ...this.loggerContext,
            service_name: serviceName,
            user_count: centralUsers.length,
            dry_run: options.dryRun,
        });
        const result = {
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
            logger_1.logger.debug('Processing batch', {
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
                    }
                    else {
                        result.updated++;
                    }
                }
                catch (error) {
                    result.failed++;
                    result.errors.push({
                        user: user.sub,
                        error: error instanceof Error ? error.message : String(error),
                    });
                    if (!continueOnError) {
                        logger_1.logger.error('Batch sync failed, aborting', error, {
                            ...this.loggerContext,
                            service_name: serviceName,
                        });
                        break;
                    }
                }
            }
        }
        result.duration_ms = Date.now() - startTime;
        logger_1.logger.info('Batch sync completed', {
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
    async syncAllServices(centralUsers, options = {}) {
        const results = [];
        for (const serviceName of this.serviceAdapters.keys()) {
            try {
                const result = await this.syncUsers(serviceName, centralUsers, options);
                results.push(result);
            }
            catch (error) {
                logger_1.logger.error('Failed to sync service', error, {
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
    shouldUpdateAccount(existing, centralUser) {
        // Always update if central user was updated more recently
        const centralUpdatedAt = new Date(centralUser.updated_at).getTime();
        const localUpdatedAt = new Date(existing.updated_at).getTime();
        if (centralUpdatedAt > localUpdatedAt) {
            return true;
        }
        // Check if critical fields differ
        if (existing.username !== centralUser.username)
            return true;
        if (existing.email !== centralUser.email)
            return true;
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
    buildAccountData(centralUser) {
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
    buildAccountUpdates(centralUser) {
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
    async cleanupStaleAccounts(serviceName, staleThresholdMs = 30 * 24 * 60 * 60 * 1000 // 30 days
    ) {
        const adapter = this.serviceAdapters.get(serviceName);
        if (!adapter) {
            throw new Error(`Service adapter not found: ${serviceName}`);
        }
        logger_1.logger.info('Cleaning up stale accounts', {
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
                    logger_1.logger.info('Deleted stale account', {
                        ...this.loggerContext,
                        service_name: serviceName,
                        remote_id: account.remote_id,
                        stale_age_days: Math.floor(staleAge / (24 * 60 * 60 * 1000)),
                    });
                }
                catch (error) {
                    logger_1.logger.error('Failed to delete stale account', error, {
                        ...this.loggerContext,
                        service_name: serviceName,
                        remote_id: account.remote_id,
                    });
                }
            }
        }
        logger_1.logger.info('Cleanup completed', {
            ...this.loggerContext,
            service_name: serviceName,
            cleaned,
        });
        return cleaned;
    }
}
exports.ShadowAccountSync = ShadowAccountSync;
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
//# sourceMappingURL=user-sync.js.map