"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.MongoUserDatabaseAdapter = exports.SQLUserDatabaseAdapter = exports.ShadowAccountSynchronizer = void 0;
const logger_1 = require("../../lib/logger");
/**
 * Shadow Account Synchronizer
 *
 * Syncs central OIDC users to local project databases
 */
class ShadowAccountSynchronizer {
    constructor(adapter, correlationId) {
        this.adapter = adapter;
        this.correlationId = correlationId || this.generateCorrelationId();
        logger_1.logger.info({
            msg: 'ShadowAccountSynchronizer initialized',
            correlation_id: this.correlationId,
        });
    }
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
    async syncUser(oidcUser) {
        const now = new Date(); // UTC time
        try {
            logger_1.logger.info({
                msg: 'Starting user sync',
                oidc_sub: oidcUser.sub,
                oidc_email: oidcUser.email,
                correlation_id: this.correlationId,
            });
            // Step 1: Check if user exists by external ID (OIDC subject)
            let existingUser = await this.adapter.findByExternalId(oidcUser.sub);
            if (existingUser) {
                // User exists - check if update needed
                return await this.updateUserIfNeeded(existingUser, oidcUser, now);
            }
            // Step 2: Check if user exists by email (migration scenario)
            const userByEmail = await this.adapter.findByEmail(oidcUser.email);
            if (userByEmail) {
                // User exists with different external_id - update it
                logger_1.logger.info({
                    msg: 'Found user by email, linking to OIDC subject',
                    user_id: userByEmail.id,
                    old_external_id: userByEmail.external_id,
                    new_external_id: oidcUser.sub,
                    correlation_id: this.correlationId,
                });
                await this.adapter.update(userByEmail.id, {
                    external_id: oidcUser.sub,
                    updated_at: now,
                });
                await this.adapter.updateLastLogin(userByEmail.id, now);
                return {
                    success: true,
                    action: 'updated',
                    user_id: userByEmail.id,
                };
            }
            // Step 3: Create new user (idempotent check inside adapter)
            logger_1.logger.info({
                msg: 'Creating new shadow account',
                oidc_sub: oidcUser.sub,
                oidc_email: oidcUser.email,
                correlation_id: this.correlationId,
            });
            const newUser = {
                id: this.generateUserId(),
                external_id: oidcUser.sub,
                username: this.generateUsername(oidcUser),
                email: oidcUser.email,
                display_name: oidcUser.name || oidcUser.email.split('@')[0],
                email_verified: oidcUser.email_verified || false,
                picture_url: oidcUser.picture,
                created_at: now,
                updated_at: now,
                last_login_at: now,
                metadata: {
                    sync_source: 'oidc',
                    synced_at: now.toISOString(),
                },
            };
            const createdUser = await this.adapter.create(newUser);
            logger_1.logger.info({
                msg: 'Shadow account created successfully',
                user_id: createdUser.id,
                external_id: createdUser.external_id,
                correlation_id: this.correlationId,
            });
            return {
                success: true,
                action: 'created',
                user_id: createdUser.id,
            };
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'User sync failed',
                oidc_sub: oidcUser.sub,
                error: err.message,
                stack: err.stack,
                correlation_id: this.correlationId,
            });
            return {
                success: false,
                action: 'failed',
                error: err.message,
            };
        }
    }
    /**
     * Update user if data has changed
     *
     * @param existingUser - Existing user record
     * @param oidcUser - OIDC user data
     * @param now - Current UTC timestamp
     * @returns Sync result
     */
    async updateUserIfNeeded(existingUser, oidcUser, now) {
        // Check if update needed
        const needsUpdate = existingUser.email !== oidcUser.email ||
            existingUser.display_name !== oidcUser.name ||
            existingUser.email_verified !== (oidcUser.email_verified || false) ||
            existingUser.picture_url !== oidcUser.picture;
        if (needsUpdate) {
            logger_1.logger.info({
                msg: 'Updating shadow account',
                user_id: existingUser.id,
                correlation_id: this.correlationId,
            });
            const updates = {
                email: oidcUser.email,
                display_name: oidcUser.name || oidcUser.email.split('@')[0],
                email_verified: oidcUser.email_verified || false,
                picture_url: oidcUser.picture,
                updated_at: now,
            };
            await this.adapter.update(existingUser.id, updates);
            await this.adapter.updateLastLogin(existingUser.id, now);
            return {
                success: true,
                action: 'updated',
                user_id: existingUser.id,
            };
        }
        // No update needed, just update last login
        await this.adapter.updateLastLogin(existingUser.id, now);
        logger_1.logger.info({
            msg: 'Shadow account up to date',
            user_id: existingUser.id,
            correlation_id: this.correlationId,
        });
        return {
            success: true,
            action: 'exists',
            user_id: existingUser.id,
        };
    }
    /**
     * Generate unique user ID
     * Uses timestamp + random for uniqueness
     */
    generateUserId() {
        return `user-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    }
    /**
     * Generate username from OIDC user info
     * Falls back to email local part
     */
    generateUsername(oidcUser) {
        // Try to use preferred_username if available
        if (oidcUser.preferred_username) {
            return oidcUser.preferred_username;
        }
        // Use email local part
        const emailLocal = oidcUser.email.split('@')[0];
        return emailLocal;
    }
    /**
     * Generate correlation ID for tracing
     */
    generateCorrelationId() {
        return `user-sync-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    }
}
exports.ShadowAccountSynchronizer = ShadowAccountSynchronizer;
/**
 * Example: SQL-based User Database Adapter
 *
 * This is a template implementation for SQL databases.
 * Each core project should adapt this to their specific database schema.
 */
class SQLUserDatabaseAdapter {
    constructor(pool, tableName = 'users', correlationId) {
        this.pool = pool;
        this.tableName = tableName;
        this.correlationId = correlationId || this.generateCorrelationId();
    }
    async findByExternalId(externalId) {
        try {
            const query = `
        SELECT * FROM ${this.tableName}
        WHERE external_id = $1
        LIMIT 1
      `;
            const result = await this.pool.query(query, [externalId]);
            if (result.rows.length === 0) {
                return null;
            }
            return this.mapRowToUserRecord(result.rows[0]);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to find user by external ID',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async findByEmail(email) {
        try {
            const query = `
        SELECT * FROM ${this.tableName}
        WHERE email = $1
        LIMIT 1
      `;
            const result = await this.pool.query(query, [email]);
            if (result.rows.length === 0) {
                return null;
            }
            return this.mapRowToUserRecord(result.rows[0]);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to find user by email',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async create(user) {
        try {
            // IDEMPOTENCY CHECK: Use INSERT ... ON CONFLICT DO NOTHING
            // This ensures the operation is safe to run 100 times
            const query = `
        INSERT INTO ${this.tableName} (
          id, external_id, username, email, display_name,
          email_verified, picture_url, created_at, updated_at,
          last_login_at, metadata
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        ON CONFLICT (external_id) DO NOTHING
        RETURNING *
      `;
            const values = [
                user.id,
                user.external_id,
                user.username,
                user.email,
                user.display_name,
                user.email_verified,
                user.picture_url,
                user.created_at,
                user.updated_at,
                user.last_login_at,
                JSON.stringify(user.metadata || {}),
            ];
            const result = await this.pool.query(query, values);
            // Check if insert succeeded or was skipped due to conflict
            if (result.rows.length === 0) {
                // Conflict occurred, fetch existing record
                logger_1.logger.info({
                    msg: 'User already exists (idempotent create)',
                    external_id: user.external_id,
                    correlation_id: this.correlationId,
                });
                const existing = await this.findByExternalId(user.external_id);
                if (!existing) {
                    throw new Error('Failed to fetch existing user after conflict');
                }
                return existing;
            }
            return this.mapRowToUserRecord(result.rows[0]);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to create user',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async update(id, updates) {
        try {
            const fields = [];
            const values = [];
            let paramIndex = 1;
            for (const [key, value] of Object.entries(updates)) {
                if (key === 'id' || key === 'external_id')
                    continue; // Don't update these
                fields.push(`${this.snakeCase(key)} = $${paramIndex}`);
                values.push(value);
                paramIndex++;
            }
            if (fields.length === 0) {
                throw new Error('No fields to update');
            }
            values.push(id); // WHERE clause parameter
            const query = `
        UPDATE ${this.tableName}
        SET ${fields.join(', ')}
        WHERE id = $${paramIndex}
        RETURNING *
      `;
            const result = await this.pool.query(query, values);
            if (result.rows.length === 0) {
                throw new Error(`User not found: ${id}`);
            }
            return this.mapRowToUserRecord(result.rows[0]);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to update user',
                user_id: id,
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async updateLastLogin(id, lastLoginAt) {
        try {
            const query = `
        UPDATE ${this.tableName}
        SET last_login_at = $1
        WHERE id = $2
      `;
            await this.pool.query(query, [lastLoginAt, id]);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to update last login',
                user_id: id,
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    /**
     * Map database row to UserRecord
     */
    mapRowToUserRecord(row) {
        return {
            id: row.id,
            external_id: row.external_id,
            username: row.username,
            email: row.email,
            display_name: row.display_name,
            email_verified: row.email_verified,
            picture_url: row.picture_url,
            created_at: new Date(row.created_at),
            updated_at: new Date(row.updated_at),
            last_login_at: new Date(row.last_login_at),
            metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
        };
    }
    /**
     * Convert camelCase to snake_case
     */
    snakeCase(str) {
        return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
    }
    /**
     * Generate correlation ID for tracing
     */
    generateCorrelationId() {
        return `sql-adapter-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    }
}
exports.SQLUserDatabaseAdapter = SQLUserDatabaseAdapter;
/**
 * Example: MongoDB-based User Database Adapter
 *
 * This is a template implementation for MongoDB.
 * Each core project should adapt this to their specific database schema.
 */
class MongoUserDatabaseAdapter {
    constructor(collection, correlationId) {
        this.collection = collection;
        this.correlationId = correlationId || this.generateCorrelationId();
    }
    async findByExternalId(externalId) {
        try {
            const doc = await this.collection.findOne({ external_id: externalId });
            return doc ? this.mapDocToUserRecord(doc) : null;
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to find user by external ID',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async findByEmail(email) {
        try {
            const doc = await this.collection.findOne({ email: email });
            return doc ? this.mapDocToUserRecord(doc) : null;
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to find user by email',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async create(user) {
        try {
            // IDEMPOTENCY CHECK: Use findOne and create if not exists
            const existing = await this.findByExternalId(user.external_id);
            if (existing) {
                logger_1.logger.info({
                    msg: 'User already exists (idempotent create)',
                    external_id: user.external_id,
                    correlation_id: this.correlationId,
                });
                return existing;
            }
            await this.collection.insertOne(user);
            return user;
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to create user',
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async update(id, updates) {
        try {
            const result = await this.collection.findOneAndUpdate({ id: id }, { $set: updates }, { returnDocument: 'after' });
            if (!result) {
                throw new Error(`User not found: ${id}`);
            }
            return this.mapDocToUserRecord(result);
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to update user',
                user_id: id,
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    async updateLastLogin(id, lastLoginAt) {
        try {
            await this.collection.updateOne({ id: id }, { $set: { last_login_at: lastLoginAt } });
        }
        catch (err) {
            logger_1.logger.error({
                msg: 'Failed to update last login',
                user_id: id,
                error: err.message,
                correlation_id: this.correlationId,
            });
            throw err;
        }
    }
    /**
     * Mongo document to UserRecord
     */
    mapDocToUserRecord(doc) {
        return {
            id: doc.id,
            external_id: doc.external_id,
            username: doc.username,
            email: doc.email,
            display_name: doc.display_name,
            email_verified: doc.email_verified,
            picture_url: doc.picture_url,
            created_at: new Date(doc.created_at),
            updated_at: new Date(doc.updated_at),
            last_login_at: new Date(doc.last_login_at),
            metadata: doc.metadata,
        };
    }
    /**
     * Generate correlation ID for tracing
     */
    generateCorrelationId() {
        return `mongo-adapter-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    }
}
exports.MongoUserDatabaseAdapter = MongoUserDatabaseAdapter;
//# sourceMappingURL=user-sync.js.map