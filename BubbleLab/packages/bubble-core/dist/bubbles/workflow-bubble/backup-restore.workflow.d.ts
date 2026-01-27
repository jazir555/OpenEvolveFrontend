/**
 * BACKUP RESTORE WORKFLOW
 *
 * A comprehensive workflow for automated backup and restore operations
 * with support for multiple storage backends and rollback capabilities.
 *
 * This workflow combines:
 * 1. Database backup operations
 * 2. File storage (S3, local filesystem)
 * 3. Backup validation and integrity checking
 * 4. Automated restore with rollback on failure
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for backup restore workflow
 */
declare const BackupRestoreParamsSchema: z.ZodObject<{
    /**
     * Operation to perform
     */
    operation: z.ZodEnum<["create", "restore", "list", "delete", "validate"]>;
    /**
     * Database configuration
     */
    database: z.ZodObject<{
        type: z.ZodEnum<["postgresql", "mysql", "mongodb"]>;
        connectionString: z.ZodString;
        databaseName: z.ZodString;
        tables: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "postgresql" | "mysql" | "mongodb";
        connectionString: string;
        databaseName: string;
        tables?: string[] | undefined;
    }, {
        type: "postgresql" | "mysql" | "mongodb";
        connectionString: string;
        databaseName: string;
        tables?: string[] | undefined;
    }>;
    /**
     * Storage backend configuration
     */
    storage: z.ZodObject<{
        backend: z.ZodEnum<["local", "s3", "gcs", "azure"]>;
        config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        config: Record<string, unknown>;
        backend: "local" | "s3" | "gcs" | "azure";
    }, {
        config: Record<string, unknown>;
        backend: "local" | "s3" | "gcs" | "azure";
    }>;
    /**
     * Backup ID for restore/delete/validate operations
     */
    backupId: z.ZodOptional<z.ZodString>;
    /**
     * Restore options
     */
    restoreOptions: z.ZodOptional<z.ZodObject<{
        createBackupBeforeRestore: z.ZodDefault<z.ZodBoolean>;
        dropExisting: z.ZodDefault<z.ZodBoolean>;
        rollbackOnError: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        createBackupBeforeRestore: boolean;
        dropExisting: boolean;
        rollbackOnError: boolean;
    }, {
        createBackupBeforeRestore?: boolean | undefined;
        dropExisting?: boolean | undefined;
        rollbackOnError?: boolean | undefined;
    }>>;
    /**
     * Backup options
     */
    backupOptions: z.ZodOptional<z.ZodObject<{
        compression: z.ZodDefault<z.ZodEnum<["none", "gzip", "zstd"]>>;
        encryption: z.ZodDefault<z.ZodBoolean>;
        retentionDays: z.ZodDefault<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        compression: "none" | "gzip" | "zstd";
        encryption: boolean;
        retentionDays: number;
        tags?: string[] | undefined;
    }, {
        tags?: string[] | undefined;
        compression?: "none" | "gzip" | "zstd" | undefined;
        encryption?: boolean | undefined;
        retentionDays?: number | undefined;
    }>>;
    /**
     * Credentials
     */
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    storage: {
        config: Record<string, unknown>;
        backend: "local" | "s3" | "gcs" | "azure";
    };
    operation: "create" | "validate" | "list" | "delete" | "restore";
    database: {
        type: "postgresql" | "mysql" | "mongodb";
        connectionString: string;
        databaseName: string;
        tables?: string[] | undefined;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    backupId?: string | undefined;
    restoreOptions?: {
        createBackupBeforeRestore: boolean;
        dropExisting: boolean;
        rollbackOnError: boolean;
    } | undefined;
    backupOptions?: {
        compression: "none" | "gzip" | "zstd";
        encryption: boolean;
        retentionDays: number;
        tags?: string[] | undefined;
    } | undefined;
}, {
    storage: {
        config: Record<string, unknown>;
        backend: "local" | "s3" | "gcs" | "azure";
    };
    operation: "create" | "validate" | "list" | "delete" | "restore";
    database: {
        type: "postgresql" | "mysql" | "mongodb";
        connectionString: string;
        databaseName: string;
        tables?: string[] | undefined;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    backupId?: string | undefined;
    restoreOptions?: {
        createBackupBeforeRestore?: boolean | undefined;
        dropExisting?: boolean | undefined;
        rollbackOnError?: boolean | undefined;
    } | undefined;
    backupOptions?: {
        tags?: string[] | undefined;
        compression?: "none" | "gzip" | "zstd" | undefined;
        encryption?: boolean | undefined;
        retentionDays?: number | undefined;
    } | undefined;
}>;
type BackupRestoreParams = z.input<typeof BackupRestoreParamsSchema>;
/**
 * Result schema for backup restore workflow
 */
declare const BackupRestoreResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * Operation result
     */
    operationResult: z.ZodOptional<z.ZodObject<{
        operation: z.ZodString;
        backupId: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodDate>;
        size: z.ZodOptional<z.ZodNumber>;
        location: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        timestamp?: Date | undefined;
        size?: number | undefined;
        location?: string | undefined;
        backupId?: string | undefined;
    }, {
        operation: string;
        timestamp?: Date | undefined;
        size?: number | undefined;
        location?: string | undefined;
        backupId?: string | undefined;
    }>>;
    /**
     * Validation result
     */
    validationResult: z.ZodOptional<z.ZodObject<{
        isValid: z.ZodBoolean;
        checksum: z.ZodOptional<z.ZodString>;
        issues: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        isValid: boolean;
        issues?: string[] | undefined;
        checksum?: string | undefined;
    }, {
        isValid: boolean;
        issues?: string[] | undefined;
        checksum?: string | undefined;
    }>>;
    /**
     * List of backups
     */
    backups: z.ZodOptional<z.ZodArray<z.ZodObject<{
        backupId: z.ZodString;
        timestamp: z.ZodDate;
        size: z.ZodNumber;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        timestamp: Date;
        size: number;
        backupId: string;
        tags?: string[] | undefined;
    }, {
        timestamp: Date;
        size: number;
        backupId: string;
        tags?: string[] | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operationResult?: {
        operation: string;
        timestamp?: Date | undefined;
        size?: number | undefined;
        location?: string | undefined;
        backupId?: string | undefined;
    } | undefined;
    validationResult?: {
        isValid: boolean;
        issues?: string[] | undefined;
        checksum?: string | undefined;
    } | undefined;
    backups?: {
        timestamp: Date;
        size: number;
        backupId: string;
        tags?: string[] | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operationResult?: {
        operation: string;
        timestamp?: Date | undefined;
        size?: number | undefined;
        location?: string | undefined;
        backupId?: string | undefined;
    } | undefined;
    validationResult?: {
        isValid: boolean;
        issues?: string[] | undefined;
        checksum?: string | undefined;
    } | undefined;
    backups?: {
        timestamp: Date;
        size: number;
        backupId: string;
        tags?: string[] | undefined;
    }[] | undefined;
}>;
type BackupRestoreResult = z.infer<typeof BackupRestoreResultSchema>;
/**
 * Backup Restore Workflow
 *
 * Comprehensive backup and restore operations with multiple storage backends and rollback support.
 */
export declare class BackupRestoreWorkflow extends WorkflowBubble<BackupRestoreParams, BackupRestoreResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        /**
         * Operation to perform
         */
        operation: z.ZodEnum<["create", "restore", "list", "delete", "validate"]>;
        /**
         * Database configuration
         */
        database: z.ZodObject<{
            type: z.ZodEnum<["postgresql", "mysql", "mongodb"]>;
            connectionString: z.ZodString;
            databaseName: z.ZodString;
            tables: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: "postgresql" | "mysql" | "mongodb";
            connectionString: string;
            databaseName: string;
            tables?: string[] | undefined;
        }, {
            type: "postgresql" | "mysql" | "mongodb";
            connectionString: string;
            databaseName: string;
            tables?: string[] | undefined;
        }>;
        /**
         * Storage backend configuration
         */
        storage: z.ZodObject<{
            backend: z.ZodEnum<["local", "s3", "gcs", "azure"]>;
            config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            config: Record<string, unknown>;
            backend: "local" | "s3" | "gcs" | "azure";
        }, {
            config: Record<string, unknown>;
            backend: "local" | "s3" | "gcs" | "azure";
        }>;
        /**
         * Backup ID for restore/delete/validate operations
         */
        backupId: z.ZodOptional<z.ZodString>;
        /**
         * Restore options
         */
        restoreOptions: z.ZodOptional<z.ZodObject<{
            createBackupBeforeRestore: z.ZodDefault<z.ZodBoolean>;
            dropExisting: z.ZodDefault<z.ZodBoolean>;
            rollbackOnError: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            createBackupBeforeRestore: boolean;
            dropExisting: boolean;
            rollbackOnError: boolean;
        }, {
            createBackupBeforeRestore?: boolean | undefined;
            dropExisting?: boolean | undefined;
            rollbackOnError?: boolean | undefined;
        }>>;
        /**
         * Backup options
         */
        backupOptions: z.ZodOptional<z.ZodObject<{
            compression: z.ZodDefault<z.ZodEnum<["none", "gzip", "zstd"]>>;
            encryption: z.ZodDefault<z.ZodBoolean>;
            retentionDays: z.ZodDefault<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            compression: "none" | "gzip" | "zstd";
            encryption: boolean;
            retentionDays: number;
            tags?: string[] | undefined;
        }, {
            tags?: string[] | undefined;
            compression?: "none" | "gzip" | "zstd" | undefined;
            encryption?: boolean | undefined;
            retentionDays?: number | undefined;
        }>>;
        /**
         * Credentials
         */
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        storage: {
            config: Record<string, unknown>;
            backend: "local" | "s3" | "gcs" | "azure";
        };
        operation: "create" | "validate" | "list" | "delete" | "restore";
        database: {
            type: "postgresql" | "mysql" | "mongodb";
            connectionString: string;
            databaseName: string;
            tables?: string[] | undefined;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        backupId?: string | undefined;
        restoreOptions?: {
            createBackupBeforeRestore: boolean;
            dropExisting: boolean;
            rollbackOnError: boolean;
        } | undefined;
        backupOptions?: {
            compression: "none" | "gzip" | "zstd";
            encryption: boolean;
            retentionDays: number;
            tags?: string[] | undefined;
        } | undefined;
    }, {
        storage: {
            config: Record<string, unknown>;
            backend: "local" | "s3" | "gcs" | "azure";
        };
        operation: "create" | "validate" | "list" | "delete" | "restore";
        database: {
            type: "postgresql" | "mysql" | "mongodb";
            connectionString: string;
            databaseName: string;
            tables?: string[] | undefined;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        backupId?: string | undefined;
        restoreOptions?: {
            createBackupBeforeRestore?: boolean | undefined;
            dropExisting?: boolean | undefined;
            rollbackOnError?: boolean | undefined;
        } | undefined;
        backupOptions?: {
            tags?: string[] | undefined;
            compression?: "none" | "gzip" | "zstd" | undefined;
            encryption?: boolean | undefined;
            retentionDays?: number | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * Operation result
         */
        operationResult: z.ZodOptional<z.ZodObject<{
            operation: z.ZodString;
            backupId: z.ZodOptional<z.ZodString>;
            timestamp: z.ZodOptional<z.ZodDate>;
            size: z.ZodOptional<z.ZodNumber>;
            location: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            timestamp?: Date | undefined;
            size?: number | undefined;
            location?: string | undefined;
            backupId?: string | undefined;
        }, {
            operation: string;
            timestamp?: Date | undefined;
            size?: number | undefined;
            location?: string | undefined;
            backupId?: string | undefined;
        }>>;
        /**
         * Validation result
         */
        validationResult: z.ZodOptional<z.ZodObject<{
            isValid: z.ZodBoolean;
            checksum: z.ZodOptional<z.ZodString>;
            issues: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            isValid: boolean;
            issues?: string[] | undefined;
            checksum?: string | undefined;
        }, {
            isValid: boolean;
            issues?: string[] | undefined;
            checksum?: string | undefined;
        }>>;
        /**
         * List of backups
         */
        backups: z.ZodOptional<z.ZodArray<z.ZodObject<{
            backupId: z.ZodString;
            timestamp: z.ZodDate;
            size: z.ZodNumber;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            timestamp: Date;
            size: number;
            backupId: string;
            tags?: string[] | undefined;
        }, {
            timestamp: Date;
            size: number;
            backupId: string;
            tags?: string[] | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operationResult?: {
            operation: string;
            timestamp?: Date | undefined;
            size?: number | undefined;
            location?: string | undefined;
            backupId?: string | undefined;
        } | undefined;
        validationResult?: {
            isValid: boolean;
            issues?: string[] | undefined;
            checksum?: string | undefined;
        } | undefined;
        backups?: {
            timestamp: Date;
            size: number;
            backupId: string;
            tags?: string[] | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operationResult?: {
            operation: string;
            timestamp?: Date | undefined;
            size?: number | undefined;
            location?: string | undefined;
            backupId?: string | undefined;
        } | undefined;
        validationResult?: {
            isValid: boolean;
            issues?: string[] | undefined;
            checksum?: string | undefined;
        } | undefined;
        backups?: {
            timestamp: Date;
            size: number;
            backupId: string;
            tags?: string[] | undefined;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Automated backup and restore with rollback support";
    static readonly longDescription = "\n    Provides comprehensive backup and restore operations for databases.\n\n    Features:\n    - Support for multiple database types (PostgreSQL, MySQL, MongoDB)\n    - Multiple storage backends (local, S3, GCS, Azure)\n    - Backup compression and encryption\n    - Automated restore with rollback on failure\n    - Backup validation with integrity checking\n    - Backup listing with filtering\n    - Tag-based backup organization\n    - Configurable retention policies\n\n    Use cases:\n    - Automated database backups\n    - Disaster recovery\n    - Database migration\n    - Point-in-time recovery\n    - Backup validation and integrity checks\n\n    Operations:\n    - create: Create new backup\n    - restore: Restore from backup (with rollback support)\n    - list: List available backups\n    - delete: Delete a backup\n    - validate: Validate backup integrity\n  ";
    static readonly alias = "backup-restore";
    constructor(params: BackupRestoreParams, context?: BubbleContext);
    protected performAction(): Promise<BackupRestoreResult>;
    /**
     * Create backup
     */
    private createBackup;
    /**
     * Restore backup
     */
    private restoreBackup;
    /**
     * List backups
     */
    private listBackups;
    /**
     * Delete backup
     */
    private deleteBackup;
    /**
     * Validate backup
     */
    private validateBackup;
    /**
     * Dump database
     */
    private dumpDatabase;
    /**
     * Restore database
     */
    private restoreDatabase;
    /**
     * Compress data
     */
    private compressData;
    /**
     * Decompress data
     */
    private decompressData;
    /**
     * Encrypt data
     */
    private encryptData;
    /**
     * Decrypt data
     */
    private decryptData;
    /**
     * Store backup
     */
    private storeBackup;
    /**
     * Load backup
     */
    private loadBackup;
    /**
     * Store metadata
     */
    private storeMetadata;
    /**
     * Load metadata
     */
    private loadMetadata;
    /**
     * Fetch backup list
     */
    private fetchBackupList;
    /**
     * Delete backup data
     */
    private deleteBackupData;
    /**
     * Delete metadata
     */
    private deleteMetadata;
    /**
     * Drop existing data
     */
    private dropExistingData;
    /**
     * Perform rollback
     */
    private performRollback;
    /**
     * Calculate checksum
     */
    private calculateChecksum;
    /**
     * Generate backup ID
     */
    private generateBackupId;
}
export {};
//# sourceMappingURL=backup-restore.workflow.d.ts.map