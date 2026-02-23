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
import { CircuitBreaker } from '../../lib/circuit-breaker';
export interface CentralUser {
    sub: string;
    username: string;
    email: string;
    name?: string;
    groups: string[];
    email_verified?: boolean;
    picture?: string;
    locale?: string;
    updated_at: string;
}
export interface ShadowAccount {
    remote_id: string;
    username: string;
    email: string;
    display_name?: string;
    groups: string[];
    is_active: boolean;
    last_sync: string;
    created_at: string;
    updated_at: string;
}
export interface ServiceAdapter {
    serviceName: string;
    circuitBreaker?: CircuitBreaker;
    getAccountByRemoteId(remoteId: string): Promise<ShadowAccount | null>;
    createAccount(account: Omit<ShadowAccount, 'created_at' | 'updated_at' | 'last_sync'>): Promise<ShadowAccount>;
    updateAccount(remoteId: string, updates: Partial<ShadowAccount>): Promise<ShadowAccount>;
    listAccounts(): Promise<ShadowAccount[]>;
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
    errors: Array<{
        user: string;
        error: string;
    }>;
    duration_ms: number;
    timestamp: string;
}
export declare class ShadowAccountSync {
    private serviceAdapters;
    private loggerContext;
    constructor();
    /**
     * Register a service adapter
     */
    registerService(serviceName: string, adapter: ServiceAdapter): void;
    /**
     * Sync a single user to a service (idempotent)
     */
    syncUser(serviceName: string, centralUser: CentralUser, options?: SyncOptions): Promise<{
        created: boolean;
        account: ShadowAccount;
    }>;
    /**
     * Sync multiple users to a service
     */
    syncUsers(serviceName: string, centralUsers: CentralUser[], options?: SyncOptions): Promise<SyncResult>;
    /**
     * Sync all users to all services
     */
    syncAllServices(centralUsers: CentralUser[], options?: SyncOptions): Promise<SyncResult[]>;
    /**
     * Check if account needs updating
     */
    private shouldUpdateAccount;
    /**
     * Build account data for creation
     */
    private buildAccountData;
    /**
     * Build account updates
     */
    private buildAccountUpdates;
    /**
     * Cleanup stale shadow accounts
     * Removes accounts that haven't been synced in a while
     */
    cleanupStaleAccounts(serviceName: string, staleThresholdMs?: number): Promise<number>;
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
//# sourceMappingURL=user-sync.d.ts.map