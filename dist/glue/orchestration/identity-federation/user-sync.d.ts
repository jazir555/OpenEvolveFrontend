/**
 * Shadow Account Synchronization Script
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of Idempotency: Safe to run 100 times (check before create)
 * - Law of UTC: All timestamps in UTC
 * - Law of Configuration Explicitness: All values via environment variables
 * - Law of Runtime Truth: Verify operations succeed
 *
 * @module glue/orchestration/identity-federation/user-sync
 */
import { OIDCUserInfo } from './oidc-provider';
/**
 * User database record (Canonical Schema)
 */
export interface LocalUserRecord {
    id: string;
    external_id: string;
    username: string;
    email: string;
    display_name: string;
    email_verified: boolean;
    picture_url?: string;
    created_at: Date;
    updated_at: Date;
    last_login_at: Date;
    metadata?: Record<string, any>;
}
/**
 * Sync result
 */
export interface SyncResult {
    success: boolean;
    action: 'created' | 'updated' | 'exists' | 'failed';
    user_id?: string;
    error?: string;
}
/**
 * Database adapter interface
 * Each core project must implement this interface for user synchronization
 */
export interface UserDatabaseAdapter {
    /**
     * Find user by external ID (OIDC subject)
     * Returns null if user does not exist
     */
    findByExternalId(externalId: string): Promise<LocalUserRecord | null>;
    /**
     * Find user by email
     * Returns null if user does not exist
     */
    findByEmail(email: string): Promise<LocalUserRecord | null>;
    /**
     * Create new user
     * Must be idempotent - check if exists before creating
     */
    create(user: LocalUserRecord): Promise<LocalUserRecord>;
    /**
     * Update existing user
     * Must update updated_at timestamp
     */
    update(id: string, updates: Partial<LocalUserRecord>): Promise<LocalUserRecord>;
    /**
     * Update last login timestamp
     */
    updateLastLogin(id: string, lastLoginAt: Date): Promise<void>;
}
/**
 * Shadow Account Synchronizer
 *
 * Syncs central OIDC users to local project databases
 */
export declare class ShadowAccountSynchronizer {
    private adapter;
    private correlationId;
    constructor(adapter: UserDatabaseAdapter, correlationId?: string);
    /**
     * Synchronize user from OIDC to local database
     *
     * This method is IDEMPOTENT and follows the Law of Idempotency:
     * - Checks if user exists before creating
     * - Updates if exists and data changed
     * - Safe to run 100 times
     *
     * @param oidcUser - User information from OIDC
     * @returns Sync result
     */
    syncUser(oidcUser: OIDCUserInfo): Promise<SyncResult>;
    /**
     * Update user if data has changed
     *
     * @param existingUser - Existing user record
     * @param oidcUser - OIDC user data
     * @param now - Current UTC timestamp
     * @returns Sync result
     */
    private updateUserIfNeeded;
    /**
     * Generate unique user ID
     * Uses timestamp + random for uniqueness
     */
    private generateUserId;
    /**
     * Generate username from OIDC user info
     * Falls back to email local part
     */
    private generateUsername;
    /**
     * Generate correlation ID for tracing
     */
    private generateCorrelationId;
}
/**
 * Example: SQL-based User Database Adapter
 *
 * This is a template implementation for SQL databases.
 * Each core project should adapt this to their specific database schema.
 */
export declare class SQLUserDatabaseAdapter implements UserDatabaseAdapter {
    private pool;
    private tableName;
    private correlationId;
    constructor(pool: any, tableName?: string, correlationId?: string);
    findByExternalId(externalId: string): Promise<LocalUserRecord | null>;
    findByEmail(email: string): Promise<LocalUserRecord | null>;
    create(user: LocalUserRecord): Promise<LocalUserRecord>;
    update(id: string, updates: Partial<LocalUserRecord>): Promise<LocalUserRecord>;
    updateLastLogin(id: string, lastLoginAt: Date): Promise<void>;
    /**
     * Map database row to UserRecord
     */
    private mapRowToUserRecord;
    /**
     * Convert camelCase to snake_case
     */
    private snakeCase;
    /**
     * Generate correlation ID for tracing
     */
    private generateCorrelationId;
}
/**
 * Example: MongoDB-based User Database Adapter
 *
 * This is a template implementation for MongoDB.
 * Each core project should adapt this to their specific database schema.
 */
export declare class MongoUserDatabaseAdapter implements UserDatabaseAdapter {
    private collection;
    private correlationId;
    constructor(collection: any, correlationId?: string);
    findByExternalId(externalId: string): Promise<LocalUserRecord | null>;
    findByEmail(email: string): Promise<LocalUserRecord | null>;
    create(user: LocalUserRecord): Promise<LocalUserRecord>;
    update(id: string, updates: Partial<LocalUserRecord>): Promise<LocalUserRecord>;
    updateLastLogin(id: string, lastLoginAt: Date): Promise<void>;
    /**
     * Mongo document to UserRecord
     */
    private mapDocToUserRecord;
    /**
     * Generate correlation ID for tracing
     */
    private generateCorrelationId;
}
//# sourceMappingURL=user-sync.d.ts.map