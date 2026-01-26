/**
 * COMPREHENSIVE VALIDATION FIXES FOR backup-restore-workflow.ts
 *
 * This file contains all validation improvements to be applied to
 * backup-restore-workflow.ts. Add these schemas and validations
 * after the imports section (around line 23).
 */

import { z } from 'zod';

// ============================================================================
// DATABASE CONFIGURATION SCHEMA
// ============================================================================

/**
 * Database configuration with comprehensive validation
 * - Port range: 1-65535 (valid TCP/UDP ports)
 * - Host length: max 253 chars (DNS limit)
 * - Username/password: reasonable length limits
 * - SQLite-specific: requires path instead of host/database
 * - Cross-field validation: ensures required fields based on type
 */
const DatabaseConfigSchema = z.object({
  type: z.enum(['postgresql', 'mysql', 'mongodb', 'sqlite'], {
    requiredError: 'Database type is required',
    invalidTypeError: 'Database type must be one of: postgresql, mysql, mongodb, sqlite'
  }),

  // Host configuration (not required for SQLite)
  host: z.string()
    .max(253, { message: 'Host name cannot exceed 253 characters (DNS limit)' })
    .regex(/^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]$/, {
      message: 'Host name must be a valid hostname'
    })
    .optional()
    .describe('Database host (required for non-SQLite databases)'),

  // Port validation (1-65535, valid TCP/UDP port range)
  port: z.number()
    .int({ message: 'Port must be an integer' })
    .min(1, { message: 'Port must be at least 1' })
    .max(65535, { message: 'Port cannot exceed 65535' })
    .optional()
    .describe('Database port (defaults: 5432/PostgreSQL, 3306/MySQL, 27017/MongoDB)'),

  // Username validation (required for PostgreSQL and MySQL)
  username: z.string()
    .min(1, { message: 'Username cannot be empty' })
    .max(128, { message: 'Username cannot exceed 128 characters' })
    .regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/, {
      message: 'Username must start with letter or underscore, contain only letters, numbers, underscores'
    })
    .optional()
    .describe('Database username (required for PostgreSQL and MySQL)'),

  // Password validation (optional, but validated if provided)
  password: z.string()
    .min(0, { message: 'Password cannot be empty string if provided' })
    .max(256, { message: 'Password cannot exceed 256 characters' })
    .optional()
    .describe('Database password (optional, recommended for production)'),

  // Database name (not required for SQLite)
  database: z.string()
    .min(1, { message: 'Database name cannot be empty' })
    .max(64, { message: 'Database name cannot exceed 64 characters' })
    .regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/, {
      message: 'Database name must start with letter or underscore, contain only letters, numbers, underscores'
    })
    .optional()
    .describe('Database name (required for non-SQLite databases)'),

  // SQLite file path (required only for SQLite)
  path: z.string()
    .min(1, { message: 'SQLite file path cannot be empty' })
    .max(4096, { message: 'File path cannot exceed 4096 characters' })
    .regex(/^[\w\-./\\]+$/, {
      message: 'File path contains invalid characters'
    })
    .optional()
    .describe('SQLite database file path (required for SQLite)'),

  // Tables list (optional, for selective backup)
  tables: z.array(
    z.string()
      .min(1, { message: 'Table name cannot be empty' })
      .max(128, { message: 'Table name cannot exceed 128 characters' })
      .regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/, {
        message: 'Table name must start with letter or underscore, contain only letters, numbers, underscores'
      })
  )
  .max(1000, { message: 'Cannot backup more than 1000 tables at once' })
  .optional()
  .describe('List of tables to backup (optional, defaults to all tables)')
}).refine(
  // Cross-field validation: SQLite requires path, others require host and database
  (data) => {
    if (data.type === 'sqlite') {
      return !!data.path;
    }
    return !!data.host && !!data.database;
  },
  {
    message: 'SQLite requires path; PostgreSQL/MySQL/MongoDB require host and database'
  }
).refine(
  // Cross-field validation: PostgreSQL and MySQL require username
  (data) => {
    if (data.type === 'postgresql' || data.type === 'mysql') {
      return !!data.username;
    }
    return true;
  },
  {
    message: 'PostgreSQL and MySQL require username'
  }
);

// ============================================================================
// S3 CONFIGURATION SCHEMA
// ============================================================================

/**
 * AWS S3 configuration with bucket name validation
 * - Bucket name: 3-63 chars, DNS-compliant, lowercase
 * - Region: valid AWS region format
 * - Credentials: validated length if provided
 */
const S3ConfigSchema = z.object({
  bucket: z.string()
    .min(3, { message: 'S3 bucket name must be at least 3 characters' })
    .max(63, { message: 'S3 bucket name cannot exceed 63 characters' })
    .regex(/^[a-z0-9][a-z0-9.-]*[a-z0-9]$/, {
      message: 'S3 bucket name must be DNS-compliant: lowercase, alphanumeric, hyphens, dots, not starting/ending with hyphen/dot'
    })
    .refine(
      (name) => !name.includes('..'),
      { message: 'S3 bucket name cannot contain consecutive dots' }
    )
    .describe('S3 bucket name'),

  region: z.string()
    .min(1, { message: 'AWS region cannot be empty' })
    .max(32, { message: 'AWS region cannot exceed 32 characters' })
    .regex(/^[a-z]{2}-[a-z]+-\d{1}$/, {
      message: 'AWS region must be in format like us-east-1, eu-west-2'
    })
    .describe('AWS region (e.g., us-east-1, eu-west-2)'),

  accessKeyId: z.string()
    .min(16, { message: 'AWS access key ID must be at least 16 characters' })
    .max(128, { message: 'AWS access key ID cannot exceed 128 characters' })
    .regex(/^[A-Z0-9]+$/, {
      message: 'AWS access key ID must be alphanumeric uppercase'
    })
    .optional()
    .describe('AWS access key ID (optional, can use IAM roles)'),

  secretAccessKey: z.string()
    .min(16, { message: 'AWS secret access key must be at least 16 characters' })
    .max(128, { message: 'AWS secret access key cannot exceed 128 characters' })
    .optional()
    .describe('AWS secret access key (optional, can use IAM roles)')
});

// ============================================================================
// AZURE CONFIGURATION SCHEMA
// ============================================================================

/**
 * Azure Blob Storage configuration
 * - Connection string: validated format
 * - Container: DNS-compliant naming rules
 * - Account: alphanumeric, lowercase
 */
const AzureConfigSchema = z.object({
  connectionString: z.string()
    .min(20, { message: 'Azure connection string must be at least 20 characters' })
    .max(2048, { message: 'Azure connection string cannot exceed 2048 characters' })
    .refine(
      (str) => str.includes('AccountName=') && str.includes('AccountKey='),
      { message: 'Azure connection string must include AccountName and AccountKey' }
    )
    .describe('Azure storage connection string'),

  container: z.string()
    .min(3, { message: 'Azure container name must be at least 3 characters' })
    .max(63, { message: 'Azure container name cannot exceed 63 characters' })
    .regex(/^[a-z0-9][a-z0-9-]*[a-z0-9]$/, {
      message: 'Azure container name must be lowercase alphanumeric with hyphens, not starting/ending with hyphen'
    })
    .describe('Azure blob container name'),

  account: z.string()
    .min(3, { message: 'Storage account name must be at least 3 characters' })
    .max(24, { message: 'Storage account name cannot exceed 24 characters' })
    .regex(/^[a-z0-9]+$/, {
      message: 'Storage account name must be lowercase alphanumeric'
    })
    .optional()
    .describe('Azure storage account name (optional, extracted from connection string if not provided)')
});

// ============================================================================
// GCS CONFIGURATION SCHEMA
// ============================================================================

/**
 * Google Cloud Storage configuration
 * - Bucket: DNS-compliant naming
 * - Project ID: valid GCP format
 * - Key file: path validation
 */
const GCSConfigSchema = z.object({
  bucket: z.string()
    .min(3, { message: 'GCS bucket name must be at least 3 characters' })
    .max(63, { message: 'GCS bucket name cannot exceed 63 characters' })
    .regex(/^[a-z0-9][a-z0-9.-]*[a-z0-9]$/, {
      message: 'GCS bucket name must be DNS-compliant: lowercase, alphanumeric, hyphens, dots'
    })
    .refine(
      (name) => !name.includes('google'),
      { message: 'GCS bucket name cannot contain "google"' }
    )
    .describe('Google Cloud Storage bucket name'),

  keyFilename: z.string()
    .min(1, { message: 'Key filename cannot be empty' })
    .max(4096, { message: 'Key filename cannot exceed 4096 characters' })
    .regex(/^[\w\-./\\]+$/, {
      message: 'Key filename contains invalid characters'
    })
    .optional()
    .describe('Path to service account JSON key file (optional, can use ADC)'),

  projectId: z.string()
    .min(6, { message: 'GCP project ID must be at least 6 characters' })
    .max(30, { message: 'GCP project ID cannot exceed 30 characters' })
    .regex(/^[a-z0-9-]+$/, {
      message: 'GCP project ID must be lowercase alphanumeric with hyphens'
    })
    .optional()
    .describe('Google Cloud project ID (optional, inferred from credentials if not provided)')
});

// ============================================================================
// MAIN PARAMETER SCHEMA
// ============================================================================

/**
 * Comprehensive backup restore parameters schema
 * All inputs validated with appropriate constraints
 */
const BackupRestoreParamsSchema = z.object({
  // Timeout: operation timeout in milliseconds (1 second to 1 hour)
  timeout: z.number()
    .int({ message: 'Timeout must be an integer' })
    .positive({ message: 'Timeout must be positive' })
    .max(3600000, { message: 'Timeout cannot exceed 3600000ms (1 hour)' })
    .default(300000)
    .describe('Operation timeout in milliseconds'),

  // Compression: enable/disable compression
  compression: z.boolean()
    .default(true)
    .describe('Enable gzip compression'),

  // Encryption: enable/disable encryption
  encryption: z.boolean()
    .default(true)
    .describe('Enable AES-256 encryption'),

  // Storage provider: where to store backups
  storageProvider: z.enum(['local', 's3', 'azure', 'gcs'], {
    requiredError: 'Storage provider is required',
    invalidTypeError: 'Storage provider must be one of: local, s3, azure, gcs'
  })
  .default('local')
  .describe('Storage provider for backups'),

  // Backup type: full, incremental, or differential
  backupType: z.enum(['full', 'incremental', 'differential'], {
    requiredError: 'Backup type is required',
    invalidTypeError: 'Backup type must be one of: full, incremental, differential'
  })
  .default('full')
  .describe('Type of backup to perform'),

  // Retention days: how long to keep backups (1 day to 100 years)
  retentionDays: z.number()
    .int({ message: 'Retention days must be an integer' })
    .positive({ message: 'Retention days must be positive' })
    .max(36500, { message: 'Retention days cannot exceed 36500 (~100 years)' })
    .default(30)
    .describe('Number of days to retain backups'),

  // ============================================================================
  // SOURCE CONFIGURATION
  // ============================================================================

  // File system source (for file backups)
  source: z.string()
    .min(1, { message: 'Source path cannot be empty' })
    .max(4096, { message: 'Source path cannot exceed 4096 characters' })
    .regex(/^[\w\-./\\]+$/, {
      message: 'Source path contains invalid characters'
    })
    .optional()
    .describe('Source file/directory path (for file system backups)'),

  // Source size: for validation and planning
  sourceSize: z.number()
    .int({ message: 'Source size must be an integer' })
    .min(0, { message: 'Source size cannot be negative' })
    .max(1e15, { message: 'Source size cannot exceed 1PB' })
    .optional()
    .describe('Source size in bytes (optional, for validation)'),

  // Files count: for file system backups
  filesCount: z.number()
    .int({ message: 'Files count must be an integer' })
    .min(0, { message: 'Files count cannot be negative' })
    .max(1e9, { message: 'Files count cannot exceed 1 billion' })
    .optional()
    .describe('Number of files to backup (optional, for validation)'),

  // Last modified date: for incremental backups
  lastModified: z.string()
    .datetime({ message: 'Last modified date must be a valid ISO 8601 datetime' })
    .optional()
    .describe('Last modification date (ISO 8601 format)'),

  // Database configuration (for database backups)
  database: DatabaseConfigSchema.optional(),

  // ============================================================================
  // STORAGE CONFIGURATIONS
  // ============================================================================

  // S3 configuration (required if storageProvider='s3')
  s3Config: S3ConfigSchema.optional(),

  // Azure configuration (required if storageProvider='azure')
  azureConfig: AzureConfigSchema.optional(),

  // GCS configuration (required if storageProvider='gcs')
  gcsConfig: GCSConfigSchema.optional(),

  // Local path (required if storageProvider='local')
  localPath: z.string()
    .min(1, { message: 'Local path cannot be empty' })
    .max(4096, { message: 'Local path cannot exceed 4096 characters' })
    .regex(/^[\w\-./\\]+$/, {
      message: 'Local path contains invalid characters'
    })
    .optional()
    .describe('Local storage path (required for local storage provider)')
})
  // Cross-field validation: must have either source or database
  .refine(
    (data) => !!(data.source || data.database),
    { message: 'Either source (file path) or database configuration is required' }
  )
  // Cross-field validation: storage provider must have corresponding config
  .refine(
    (data) => {
      if (data.storageProvider === 's3') return !!data.s3Config;
      if (data.storageProvider === 'azure') return !!data.azureConfig;
      if (data.storageProvider === 'gcs') return !!data.gcsConfig;
      if (data.storageProvider === 'local') return !!data.localPath;
      return true;
    },
    { message: 'Storage provider requires corresponding configuration' }
  )
  // Cross-field validation: retention days must be reasonable
  .refine(
    (data) => {
      if (data.retentionDays && data.retentionDays < 1) return false;
      if (data.retentionDays && data.retentionDays > 36500) return false;
      return true;
    },
    { message: 'Retention days must be between 1 and 36500 (~100 years)' }
  );

// ============================================================================
// VALIDATION HELPER METHOD
// ============================================================================

/**
 * Validates input parameters against the schema
 * Returns validation result with detailed error messages
 */
function validateBackupRestoreInput(input: any): {
  valid: boolean;
  error?: string;
  details?: z.ZodError;
} {
  try {
    BackupRestoreParamsSchema.parse(input);
    return { valid: true };
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Format error messages for better readability
      const formattedErrors = error.errors.map((err) => {
        const path = err.path.join('.') || 'root';
        const message = err.message;
        return `${path}: ${message}`;
      }).join('; ');

      return {
        valid: false,
        error: `Validation failed: ${formattedErrors}`,
        details: error
      };
    }
    return {
      valid: false,
      error: 'Unknown validation error'
    };
  }
}

// ============================================================================
// USAGE IN CLASS
// ============================================================================

/**
 * In the BackupRestoreWorkflow class, add this validation call
 * at the beginning of the execute() method:
 *
 * async execute(input: any): Promise<BackupRestoreResult> {
 *   // Validate input first
 *   const validation = validateBackupRestoreInput(input);
 *   if (!validation.valid) {
 *     return {
 *       success: false,
 *       error: validation.error || 'Input validation failed',
 *       steps: []
 *     };
 *   }
 *
 *   // Continue with existing logic...
 *   const steps = [];
 *   try {
 *     // ... rest of execute method
 *   } catch (error: any) {
 *     return { success: false, error: error.message, steps };
 *   }
 * }
 */

export {
  DatabaseConfigSchema,
  S3ConfigSchema,
  AzureConfigSchema,
  GCSConfigSchema,
  BackupRestoreParamsSchema,
  validateBackupRestoreInput
};
