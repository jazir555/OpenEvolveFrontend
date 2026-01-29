<?php

namespace LHA;

/**
 * Class Retry (Ludicrous Speed Edition - Self-Contained WP - Final Review)
 * Extreme performance, resilience, and observability using WPDB and Object Cache.
 * 
 * REFACTORED: Now uses dependency injection
 */
class Retry implements \LHA\Interfaces\RetryInterface {

	const RETRY_TABLE_BASENAME = 'lha_retry_queue';
	const RETRY_HISTORY_TABLE_BASENAME = 'lha_retry_history';
	const RETRY_DLQ_TABLE_BASENAME = 'lha_retry_dlq';
	const TEXT_DOMAIN = 'self-host-assets'; // Text domain for translations
	const STATUS_PENDING = 'pending';
	const STRATEGY_EXPONENTIAL = 'exponential';
	const PRIORITY_NORMAL = 50; // Example value
	const DLQ_REASON_FAILED = 'failed';

	/**
	 * Logger instance
	 * @var \LHA\Interfaces\LoggerInterface
	 */
	private \LHA\Interfaces\LoggerInterface $logger;

	/**
	 * Database instance
	 * @var \LHA\Interfaces\DatabaseInterface
	 * @phpstan-ignore-next-line Property is injected for future use
	 */
	private \LHA\Interfaces\DatabaseInterface $database;

	/**
	 * WordPress database object
	 * @var \wpdb
	 */
	private \wpdb $wpdb;

	/**
	 * Task queue instance
	 * @var \LHA\Interfaces\TaskQueueInterface
	 */
	private \LHA\Interfaces\TaskQueueInterface $tasks;

	/**
	 * File lock instance
	 * @var \LHA\Interfaces\LockInterface
	 * @phpstan-ignore-next-line Property is injected for future use
	 */
	private \LHA\Interfaces\LockInterface $lock;

	/**
	 * Cleanup instance
	 * @var \LHA\Interfaces\CleanupInterface|null
	 */
	private ?\LHA\Interfaces\CleanupInterface $cleanup = null;

	/**
	 * Normalize instance for URL normalization
	 * @var \LHA\Interfaces\NormalizeInterface|null
	 */
	private ?\LHA\Interfaces\NormalizeInterface $normalize = null;

	/**
	 * UrlProcessor instance for URL processing operations
	 * @var \LHA\Interfaces\UrlProcessorInterface|null
	 */
	private ?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null;

	/**
	 * Constructor
	 * 
	 * @param \LHA\Interfaces\LoggerInterface $logger
	 * @param \LHA\Interfaces\DatabaseInterface $database
	 * @param \wpdb $wpdb
	 * @param \LHA\Interfaces\TaskQueueInterface $tasks
	 * @param \LHA\Interfaces\LockInterface $lock
	 * @param \LHA\Interfaces\CleanupInterface|null $cleanup Optional cleanup service
	 * @param \LHA\Interfaces\NormalizeInterface|null $normalize Optional normalize service for URL normalization
	 * @param \LHA\Interfaces\UrlProcessorInterface|null $urlProcessor Optional URL processor service
	 */
	public function __construct(
		\LHA\Interfaces\LoggerInterface $logger,
		\LHA\Interfaces\DatabaseInterface $database,
		\wpdb $wpdb,
		\LHA\Interfaces\TaskQueueInterface $tasks,
		\LHA\Interfaces\LockInterface $lock,
		?\LHA\Interfaces\CleanupInterface $cleanup = null,
		?\LHA\Interfaces\NormalizeInterface $normalize = null,
		?\LHA\Interfaces\UrlProcessorInterface $urlProcessor = null
	) {
		$this->logger = $logger;
		$this->database = $database;
		$this->wpdb = $wpdb;
		$this->tasks = $tasks;
		$this->lock = $lock;
		$this->cleanup = $cleanup;
		$this->normalize = $normalize;
		$this->urlProcessor = $urlProcessor;
	}

	/**
	 * Normalize a URL for consistent storage and comparison.
	 * Uses UrlProcessor for consistent URL normalization across the codebase.
	 *
	 * @param string $url The URL to normalize.
	 * @return string The normalized URL.
	 */
	private function normalize_url(string $url): string {
		$trimmed = trim($url);
		if ($trimmed === '') {
			return '';
		}
		
		// Use UrlProcessor if available for consistent URL normalization
		if ($this->urlProcessor !== null) {
			return $this->urlProcessor->normalize_url($trimmed);
		}
		
		// Fallback to Normalize service if UrlProcessor is not available
		if ($this->normalize !== null) {
			$normalized = $this->normalize->normalize_asset_url($trimmed);
			if ($normalized !== '' && is_string($normalized)) {
				return $normalized;
			}
		}
		
		// Final fallback to Sanitize::sanitize_url
		return Sanitize::sanitize_url($trimmed);
	}

	/**
	 * Check if external retry handling should be used.
	 * When Action Scheduler is active, it handles retries natively.
	 * This method checks the Tasks class to determine the active processor.
	 *
	 * @return bool True if this Retry class should handle retries.
	 */
	public function should_handle_retries(): bool {
		// Check if Tasks has a method to determine retry handling
		if ( method_exists( $this->tasks, 'should_use_external_retry' ) ) {
			return $this->tasks->should_use_external_retry();
		}
		// Default to true for backward compatibility
		return true;
	}

	/**
	 * Enqueue a retry operation.
	 * Will skip if Action Scheduler is handling retries natively.
	 * 
	 * @param array $data Retry data including related_id, related_type, retry_reason, etc.
	 * @return int|false Retry ID on success, false on failure (returns 0 if AS handles retries)
	 */
	public function enqueue_retry(array $data) {
		// Skip if Action Scheduler is handling retries natively
		if ( ! $this->should_handle_retries() ) {
			$this->logger->log_debug( '[Retry] Skipping enqueue - Action Scheduler handles retries natively' );
			return 0; // Return 0 to indicate "handled" (by AS) - truthy int value
		}
		// Validate required fields
		if (empty($data['related_id']) || !is_numeric($data['related_id']) || $data['related_id'] <= 0) {
			$this->logger->log_error('[Retry] Invalid related_id for enqueue_retry', ['data' => $data]);
			return false;
		}

		if (empty($data['related_type']) || !is_string($data['related_type'])) {
			$this->logger->log_error('[Retry] Invalid or missing related_type for enqueue_retry', ['data' => $data]);
			return false;
		}

		// Map to store_retry_job parameters
		$category = $data['category'] ?? self::CATEGORY_DATABASE;
		$operation_type = $data['related_type'];
		$operation_data = [
			'related_id' => (int)$data['related_id'],
			'related_type' => $data['related_type'],
			'retry_reason' => $data['retry_reason'] ?? '',
			'last_error_data' => $data['last_error_data'] ?? ''
		];
		
		$options = [
			'priority' => $data['priority'] ?? self::PRIORITY_NORMAL,
			'retry_count' => $data['retry_count'] ?? 0
		];

		try {
			$result = $this->store_retry_job($category, $operation_type, $operation_data, $options);
			return $result;
		} catch (\Exception $e) {
			$this->logger->log_error('[Retry] Exception in enqueue_retry: ' . $e->getMessage(), ['exception' => $e]);
			return false;
		}
	}
	
	/**
	 * Alias for enqueue_retry() for backward compatibility
	 * 
	 * @param int $related_id Related entity ID
	 * @param string $related_type Related entity type
	 * @param string $retry_reason Reason for retry
	 * @return int|false Retry ID on success, false on failure
	 */
	public function add_to_retry_queue(int $related_id, string $related_type, string $retry_reason = '') {
		return $this->enqueue_retry([
			'related_id' => $related_id,
			'related_type' => $related_type,
			'retry_reason' => $retry_reason
		]);
	}

	/**
	 * Get pending retry operations.
	 * Returns empty array if Action Scheduler is handling retries natively.
	 * 
	 * @param int $limit Maximum number of retries to return
	 * @return array Array of pending retry operations
	 */
	public function get_pending_retries(int $limit = 100): array {
		// Return empty if Action Scheduler is handling retries natively
		if ( ! $this->should_handle_retries() ) {
			return [];
		}
		
		global $wpdb;
		$table = $this->get_retry_table_name();
		
		try {
			// Optimized to select only needed columns for performance
			$sql = $wpdb->prepare(
				"SELECT id, asset_id, original_url, type, status, priority, attempts, scheduled_at, last_error, metadata FROM `{$table}` WHERE `status` = %s ORDER BY `priority` ASC, `scheduled_at` ASC LIMIT %d",
				self::STATUS_PENDING,
				$limit
			);
			
			$results = $wpdb->get_results($sql, ARRAY_A);
			
			if ($results === null) {
				$this->logger->log_error('[Retry] Database error in get_pending_retries: ' . $wpdb->last_error, []);
				return [];
			}
			
			return $results ?: [];
		} catch (\Exception $e) {
			$this->logger->log_error('[Retry] Exception in get_pending_retries: ' . $e->getMessage(), ['exception' => $e]);
			return [];
		}
	}

	/**
	 * Process a single retry operation.
	 * Will skip if Action Scheduler is handling retries natively.
	 * 
	 * @param int $retry_id Retry operation ID
	 * @return bool True on success, false on failure
	 */
	public function process_retry(int $retry_id): bool {
		// Skip if Action Scheduler is handling retries natively
		if ( ! $this->should_handle_retries() ) {
			$this->logger->log_debug( '[Retry] Skipping process_retry - Action Scheduler handles retries natively' );
			return true;
		}
		
		global $wpdb;
		$table = $this->get_retry_table_name();
		
		// Acquire lock to prevent concurrent processing of the same retry job
		$lock_key = "retry_process_{$retry_id}";
		$lock_acquired = false;
		
		try {
			$lock_acquired = $this->lock->acquire_with_backoff($lock_key, 60, 3, 50000, 200000);
			
			if (!$lock_acquired) {
				$this->logger->log_warning('[Retry] Failed to acquire lock for processing retry job. Another process may be processing it.', [
					'retry_id' => $retry_id,
					'lock_key' => $lock_key
				]);
				return false;
			}
			
			// Get the retry job - optimized to select only needed columns
			$sql = $wpdb->prepare("SELECT id, asset_id, original_url, type, status, priority, attempts, scheduled_at, last_error, metadata, created_at, updated_at FROM `{$table}` WHERE `id` = %d", $retry_id);
			$job = $wpdb->get_row($sql, ARRAY_A);
			
			if (!$job) {
				$this->logger->log_warning('[Retry] Retry job not found: ' . $retry_id, []);
				return false;
			}
			
			// Decrypt and decode the operation data
			$operation_data = [];
			if (!empty($job['operation_data_encrypted'])) {
				$decrypted = apply_filters('lha_decrypt_data', $job['operation_data_encrypted']);
				if (!is_wp_error($decrypted)) {
					$operation_data = json_decode($decrypted, true) ?: [];
				}
			}
			
			// Execute the retry based on operation type
			$success = false;
			$operation_type = $job['operation_type'] ?? '';
			
			// Check if we have a registered executor
			if (isset(self::$executors[$operation_type])) {
				$executor = self::$executors[$operation_type];
				$success = call_user_func($executor, $operation_data, $job);
			} else {
				// Default handling - mark as processed
				$success = true;
			}
			
			if ($success) {
				// Remove the retry operation
				$this->remove_retry_operation($retry_id);
				return true;
			} else {
				// Increment retry count and reschedule
				$retry_count = ((int)($job['retry_count'] ?? 0)) + 1;
				$config = $this->get_retry_config();
				$max_retries = $config['max_retries'] ?? 5;
				
				if ($retry_count >= $max_retries) {
					// Move to DLQ - pass job array, reason string, and DLQ reason code
					$this->move_to_dlq($job, 'Max retries exceeded', self::DLQ_REASON_MAX_ATTEMPTS);
					return false;
				} else {
					// Reschedule - pass job array, error message, error code, and lock token
					$lock_token = $job['lock_token'] ?? '';
					$this->reschedule_failed_job($job, 'Retry attempt failed', 'RETRY_FAILED', $lock_token);
					return false;
				}
			}
		} catch (\Exception $e) {
			$this->logger->log_error('[Retry] Exception in process_retry: ' . $e->getMessage(), ['exception' => $e, 'retry_id' => $retry_id]);
			return false;
		} finally {
			// Always release the lock
			if ($lock_acquired) {
				$this->lock->release($lock_key);
			}
		}
	}

	/**
	 * Get retry statistics
	 * 
	 * @return array Statistics about retry operations
	 */
	public function get_retry_stats(): array {
		return $this->get_queue_stats();
	}
	
	
	/**
	 * Get retry system status.
	 * Indicates whether this Retry class is active or if Action Scheduler handles retries.
	 *
	 * @return array Status information
	 */
	public function get_retry_system_status(): array {
		$should_handle = $this->should_handle_retries();
		
		$status = [
			'active'              => $should_handle,
			'handler'             => $should_handle ? 'LHA Retry Class' : 'Action Scheduler',
			'reason'              => $should_handle 
				? 'WP-Cron processor does not have native retry support'
				: 'Action Scheduler handles retries natively',
		];
		
		// Add queue stats only if this class is handling retries
		if ( $should_handle ) {
			$status['queue_stats'] = $this->get_queue_stats();
		}
		
		return $status;
	}

	/**
	 * Cleanup old retry operations
	 * 
	 * @param int $days Number of days to keep
	 * @return int Number of records cleaned up
	 */
	public function cleanup_old_retries(int $days = 30): int {
		$result = $this->cleanup_old_records();
		return ($result['hist'] ?? 0) + ($result['dlq'] ?? 0);
	}

	// --- Static properties for caching ---
	private static ?string $datetime_type = null;
	private static ?string $datetime_default = null;
	private static ?string $datetime_default_next_attempt = null;
	const DEFAULT_CRON_INTERVAL = 'one_minute'; // Default WP Cron schedule name
	const CRON_HOOK = 'self_host_assets_process_retries_ludicrous_sc';

	/** @var string Default value for hostname if retrieval fails or is disabled. */
	protected const DEFAULT_HOSTNAME = 'unknown_host';
	/** @var string Default value for PID if retrieval fails or is disabled. */
	protected const DEFAULT_PID = 'unknown_pid';
	/** @var string Value indicating gethostname() returned false. */
	protected const HOSTNAME_FAILED_INDICATOR = 'gethostname_failed';
	/** @var string Value indicating gethostname() returned an empty string. */
	protected const HOSTNAME_EMPTY_INDICATOR = 'empty_host';
	/** @var string Value indicating getmypid() returned false. */
	protected const PID_FAILED_INDICATOR = 'getmypid_failed';
	/** @var string Value indicating sanitization resulted in an empty string. */
	protected const SANITIZATION_FAILED_INDICATOR = 'sanitization_failed';
	/** @var string Prefix for fallback IDs generated during errors. */
	protected const FALLBACK_ID_PREFIX = 'fb_proc_';
	/** @var string Prefix for absolute minimal fallback IDs. */
	protected const ABSOLUTE_MIN_ID_PREFIX = 'min_fb_';
	/** @var string Critical fallback ID constant. */
	protected const CRITICAL_FALLBACK_ID = 'critical_fallback_id';
	/** @var int Maximum length of the generated processor ID. */
	protected const MAX_ID_LENGTH = 100;

	// --- Job Categories (Examples) ---
	const CATEGORY_DATABASE = 'database';
	const CATEGORY_DOWNLOAD = 'download';
	const CATEGORY_API_CALL = 'api_call';
	const CATEGORY_CLEANUP = 'cleanup';



	// --- Job Statuses (Main Queue) ---
	const STATUS_SCHEDULED = 'scheduled';           // Waiting for a specific future time (scheduled_at)
	const STATUS_PROCESSING = 'processing';         // Locked and being worked on
	const STATUS_FAILED = 'failed';                 // Terminal status within the main table (fallback if DLQ fails)
	const STATUS_FAILURE = 'failure';               // Failure status for history table
	const STATUS_WAITING_DEPENDENCY = 'waiting_dependency'; // Waiting for depends_on_job_id to complete/fail
	const STATUS_PAUSED = 'paused';                 // Manually paused, won't be processed

	// --- Job Priorities ---
	const PRIORITY_HIGH = 10;
	const PRIORITY_LOW = 200;

	// --- Retry Strategies ---
	const STRATEGY_DEFAULT = 'exponential'; // Default strategy
	const STRATEGY_FIXED = 'fixed';
	const STRATEGY_LINEAR = 'linear';
	const STRATEGY_NONE = 'none'; // Execute only once


	// --- Dead Letter Queue Reasons ---
	const DLQ_REASON_EXPIRED = 'expired';             // Job expired before completion
	const DLQ_REASON_POISON = 'poison_pill';          // Repeated identical failures
	const DLQ_REASON_CANCELLED = 'cancelled';         // Cancelled by hook, logic, or manually
	const DLQ_REASON_STALLED_RESET = 'stalled_reset'; // Job was processing, lock expired, reset (more for history) - DLQ if consistently stalling? Maybe not needed here.
	const DLQ_REASON_MANUAL = 'manual_move';          // Manually moved to DLQ by admin/tool
	const DLQ_REASON_THROTTLED = 'throttled';         // Terminally failed *due to* persistent throttling denials (less common, usually reschedule)
	const DLQ_REASON_CIRCUIT_OPEN = 'circuit_open';   // Terminally failed *due to* persistent circuit breaker denials (less common, usually reschedule)
	const DLQ_REASON_DATA_INTEGRITY = 'data_integrity'; // Payload decryption/decoding failed
	const DLQ_REASON_DEPENDENCY_FAILED = 'dependency_failed'; // A job this depended on failed permanently (moved to DLQ)
	const DLQ_REASON_CONFIGURATION = 'configuration'; // Configuration error (no executor, etc.)
	const DLQ_REASON_DENIED = 'denied';               // Execution denied by filter/hook
	const DLQ_REASON_MAX_ATTEMPTS = 'max_attempts';   // Maximum retry attempts reached
	const DLQ_REASON_NON_RETRYABLE = 'non_retryable'; // Non-retryable error type

	// --- Internal Constants ---
	private const LOCK_HEARTBEAT_INTERVAL = 45; // Seconds between heartbeats while processing
	private const POISON_PILL_THRESHOLD = 5; // Consecutive failures with same code to trigger DLQ
	/**
	 * Define your current schema version here.
	 * Example: const SCHEMA_VERSION = 2;
	 * Make sure this constant is defined within your Retry class.
	 */
	const SCHEMA_VERSION = 4;

	/**
	 * Option name storing the current database schema version for the retry system.
	 */
	const DB_VERSION_OPTION_NAME = 'sha_retry_schema_version_ludicrous_sc';
	private const CACHE_GROUP = 'sha_retry_queue_lc'; // Unique cache group for WP Object Cache
	private const STATS_CACHE_KEY = 'queue_stats_lc';
	private const STATS_CACHE_TTL = MINUTE_IN_SECONDS; // How long queue stats are cached

	// --- State ---
	private static ?array $configCache = null;
	private static array $executors = [];
	private static ?string $processorId = null;
	private static bool $signalShutdown = false; // Flag for graceful shutdown (primarily for long-running external runners, less impact on basic cron)

	// =========================================================================
	// Schema Definition & Management
	// =========================================================================

	/**
	 * Determines the appropriate DATETIME type and default values based on DB version.
	 * Caches the result for subsequent calls.
	 *
	 * @global wpdb $wpdb WordPress database abstraction object.
	 */
	private function determine_datetime_features(): void {
		if ( null !== self::$datetime_type ) {
			// Already determined and cached
			return;
		}

		global $wpdb;

		// --- Set Defaults ---
		// Default to standard DATETIME without microsecond precision
		self::$datetime_type = 'DATETIME';
		// SQL Keyword `CURRENT_TIMESTAMP` - No quotes needed in the SQL DEFAULT clause
		self::$datetime_default = 'CURRENT_TIMESTAMP';
		// Default for next_attempt_at: SQL *string literal* for a specific date - Needs single quotes for SQL
		self::$datetime_default_next_attempt = "'1970-01-01 00:00:01'";

		// --- Check for Microsecond Support ---
		$db_version_str = $wpdb->db_version();

		if ( empty($db_version_str) ) {
			// Handle case where version couldn't be retrieved (rare, but defensive)
			// Keep the defaults. Optionally log an error/warning.
			// error_log( 'Could not determine DB version for DATETIME precision. Using default DATETIME.' );
			return;
		}

		// Check for MariaDB (case-insensitive check for robustness)
		// Use strpos for PHP < 8 compatibility
		$is_mariadb = ( function_exists('str_contains') )
			? (strpos(strtolower($db_version_str), 'mariadb') !== false)
			: ( strpos( strtolower($db_version_str), 'mariadb' ) !== false );

		// Extract version number reliably (handles X.Y.Z formats, ignores suffixes)
		// Example formats: '5.7.30', '10.4.17-MariaDB-log', '8.0.23'
		if ( ! preg_match( '/^([0-9]+\.[0-9]+\.?[0-9]*)/', $db_version_str, $matches ) ) {
			// Fallback if regex fails (e.g., very unusual version string)
			// Keep the defaults. Optionally log an error/warning.
			// error_log( 'Could not parse DB version string: ' . $db_version_str . '. Using default DATETIME.' );
			return;
		}
		$version_number = $matches[1];

		// MySQL >= 5.6.4 supports DATETIME(6) and CURRENT_TIMESTAMP(6)
		$mysql_supports_precision = version_compare( $version_number, '5.6.4', '>=' );

		// MariaDB >= 10.1.2 supports DATETIME(6) and CURRENT_TIMESTAMP(6)
		$mariadb_supports_precision = version_compare( $version_number, '10.1.2', '>=' );

		if ( ( ! $is_mariadb && $mysql_supports_precision ) || ( $is_mariadb && $mariadb_supports_precision ) ) {
			// Precision is supported, update the cached values
			self::$datetime_type = 'DATETIME(6)';
			// SQL Keyword `CURRENT_TIMESTAMP(6)` - No quotes needed in the SQL DEFAULT clause
			self::$datetime_default = 'CURRENT_TIMESTAMP(6)';
			// Default for next_attempt_at: SQL *string literal* with microseconds - Needs single quotes for SQL
			self::$datetime_default_next_attempt = "'1970-01-01 00:00:01.000000'";
		}
		// If precision is not supported, the initial defaults remain correct.
	}

    /**
     * Schedule a download retry task.
     * Wraps the generic enqueue_task with specific context for download retries.
     *
     * @param string $url           Original URL.
     * @param string $type          Asset type.
     * @param int    $cache_exp     Cache expiration in days.
     * @param bool   $force         Force refresh flag.
     * @param int    $depth         Current depth.
     * @param int    $retry_num     The current retry number (e.g., 1 for first retry).
     * @return bool True if scheduling succeeded, false otherwise.
     */
    public function schedule_download_retry(string $url, string $type, int $cache_exp, bool $force, int $depth, int $retry_num): bool {
        // Implementation based on assumptions about enqueue_task signature and functionality.
        $config = $this->get_retry_config();
        $task_data = [
            'type'             => $type,
            'original_url'     => $url,
            'force_refresh'    => $force,
            'current_depth'    => $depth,
            'retry_count'      => $retry_num, // Use the passed retry number
            'max_retries'      => $config['max_attempts'] ?? 3, // Get max retries from config
            'cache_expiration' => $cache_exp,
            'delay_seconds'    => $this->calculate_retry_delay($retry_num), // Calculate delay based on attempt number
            // Add other necessary fields for enqueue_task
        ];
        // Determine priority, maybe increase for retries?
        $priority = $this->calculate_task_priority($type) - 1; // Example: slightly higher priority for retries

        return $this->enqueue_task($task_data, $priority);
    }

    /**
     * Calculates retry delay (simple exponential backoff example).
     * @param int $retry_num The current retry number (1 for first retry, 2 for second etc.).
     * @return int Delay in seconds.
     */
    private function calculate_retry_delay(int $retry_num): int {
        $base_delay = 60; // 1 minute base
        $max_delay = HOUR_IN_SECONDS; // 1 hour max
        // Exponential backoff: 60 * 2^(retry_num - 1)
        $delay = $base_delay * pow(2, max(0, $retry_num - 1));
        return min($max_delay, (int)$delay);
    }

    /**
     * Calculate task priority by delegating to Tasks
     * @param string $type Task type
     * @return int Priority level
     */
    private function calculate_task_priority(string $type): int {
        return $this->tasks->calculate_task_priority($type);
    }

    /**
     * Enqueue task by delegating to Tasks
     * @param array $task_data Task data
     * @param int $priority Priority level
     * @return bool Success status
     */
    private function enqueue_task(array $task_data, int $priority = 10): bool {
        return $this->tasks->enqueue_task($task_data, $priority);
    }

    /**
     * Get logging enabled flag
     * @return bool Whether logging is enabled
     */
    private function get_logging_enabled_flag(): bool {
        return defined('WP_DEBUG') && WP_DEBUG;
    }

    /**
     * Promote dependent jobs when a job completes or fails
     * @param int $job_id The job ID that completed/failed
     * @param bool $success Whether the job succeeded
     * @return void
     */
    private function promote_dependent_jobs(int $job_id, bool $success): void {
        // Get jobs that depend on this job
        $table_name = $this->wpdb->prefix . self::RETRY_TABLE_BASENAME;
        
        $dependent_jobs = $this->wpdb->get_results(
            $this->wpdb->prepare(
                "SELECT id, status FROM {$table_name} WHERE depends_on_job_id = %d AND status = %s",
                $job_id,
                self::STATUS_WAITING_DEPENDENCY
            ),
            ARRAY_A
        );
        
        if (empty($dependent_jobs)) {
            return;
        }
        
        // Update dependent jobs based on parent success/failure
        foreach ($dependent_jobs as $dependent) {
            if ($success) {
                // Parent succeeded, promote to pending so it can be processed
                $this->wpdb->update(
                    $table_name,
                    ['status' => self::STATUS_PENDING],
                    ['id' => $dependent['id']],
                    ['%s'],
                    ['%d']
                );
            } else {
                // Parent failed, mark dependent as failed too
                $this->wpdb->update(
                    $table_name,
                    ['status' => self::STATUS_FAILED],
                    ['id' => $dependent['id']],
                    ['%s'],
                    ['%d']
                );
            }
        }
        
        // Clear cache if we updated anything
        if (!empty($dependent_jobs)) {
            $this->clear_stats_cache();
        }
    }


	/**
	 * Generates SQL CREATE TABLE statements for the retry infrastructure.
	 * Implements fallback for DATETIME precision based on DB version.
	 * Suitable for use with the dbDelta function.
	 *
	 * @global wpdb $wpdb WordPress database abstraction object.
	 * @return array<string> Array of SQL CREATE TABLE statements.
	 */
	public function get_retry_table_definitions(): array {
		global $wpdb;

		// 1. Determine and cache the correct DATETIME types/defaults first
		$this->determine_datetime_features();

		// 2. Prepare common variables
		$prefix = $wpdb->prefix;
		$retry_table_name = $prefix . self::RETRY_TABLE_BASENAME;
		$history_table_name = $prefix . self::RETRY_HISTORY_TABLE_BASENAME;
		$dlq_table_name = $prefix . self::RETRY_DLQ_TABLE_BASENAME;

		// Use the determined values from determine_datetime_features()
		$dt_type = self::$datetime_type; // e.g., 'DATETIME' or 'DATETIME(6)'
		$default_dt_keyword = self::$datetime_default; // e.g., 'CURRENT_TIMESTAMP' or 'CURRENT_TIMESTAMP(6)' (SQL Keyword - NO quotes needed in SQL DEFAULT)
		$default_next_attempt_literal = self::$datetime_default_next_attempt; // e.g., "'1970-01-01 00:00:01'" (SQL String Literal - quotes ARE included here)

		// 3. Define default values for use in SQL statements
		// NOTE: dbDelta generally prefers standard SQL syntax for DEFAULT clauses.
		//       - String types: Need single quotes around the default value. Use esc_sql for safety if value isn't a fixed constant.
		//       - Numeric types (INT, TINYINT, DECIMAL): Should use the number directly (e.g., DEFAULT 0, DEFAULT 5, DEFAULT 2.00). NO quotes.
		//       - Date/Time types: Use SQL keywords (CURRENT_TIMESTAMP) without quotes, or string literals ('YYYY-MM-DD...') with quotes.

		// String defaults (need quoting)
		$default_status_pending_sql = "'" . esc_sql( self::STATUS_PENDING ) . "'";
		$default_strategy_sql       = "'" . esc_sql( self::STRATEGY_EXPONENTIAL ) . "'";
		$default_dlq_reason_sql     = "'" . esc_sql( self::DLQ_REASON_FAILED ) . "'";
		$default_empty_string_sql   = "''";

		// Numeric defaults (NO quotes needed in SQL DEFAULT)
		$default_priority_sql       = (string) self::PRIORITY_NORMAL; // e.g., 5
		$default_retry_count_sql    = '0';
		$default_max_attempts_sql   = '3';        // Example default
		$default_base_delay_sql     = '60';       // Example default (seconds)
		$default_backoff_factor_sql = '2.00';     // Example default (DECIMAL)

		// Character set and collation - Use WordPress defaults
		$charset_collate = $wpdb->get_charset_collate();

		$tables = [];

		// --- Retry Table ---
		$tables[] = "CREATE TABLE {$retry_table_name} (
			id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			category VARCHAR(50) NOT NULL DEFAULT {$default_empty_string_sql},
			status VARCHAR(20) NOT NULL DEFAULT {$default_status_pending_sql},
			operation_type VARCHAR(100) NOT NULL DEFAULT {$default_empty_string_sql},
			unique_key VARCHAR(191) DEFAULT NULL,
			priority TINYINT UNSIGNED NOT NULL DEFAULT {$default_priority_sql},
			job_group VARCHAR(100) DEFAULT NULL,
			depends_on_job_id BIGINT UNSIGNED DEFAULT NULL,
			correlation_id VARCHAR(100) DEFAULT NULL,
			operation_data LONGTEXT DEFAULT NULL,
			metadata LONGTEXT DEFAULT NULL,
			retry_strategy VARCHAR(50) NOT NULL DEFAULT {$default_strategy_sql},
			retry_count INT UNSIGNED NOT NULL DEFAULT {$default_retry_count_sql},
			max_attempts INT UNSIGNED NOT NULL DEFAULT {$default_max_attempts_sql},
			base_delay_sec INT UNSIGNED NOT NULL DEFAULT {$default_base_delay_sql},
			backoff_factor DECIMAL(5,2) NOT NULL DEFAULT {$default_backoff_factor_sql},
			max_delay_sec INT UNSIGNED DEFAULT NULL,
			state_metadata TEXT DEFAULT NULL,
			lock_token VARCHAR(64) DEFAULT NULL,
			lock_expires_at {$dt_type} DEFAULT NULL,
			processor_id VARCHAR(100) DEFAULT NULL,
			last_error TEXT DEFAULT NULL,
			last_error_code VARCHAR(100) DEFAULT NULL,
			last_attempt_at {$dt_type} DEFAULT NULL,
			next_attempt_at {$dt_type} NOT NULL DEFAULT {$default_next_attempt_literal},
			scheduled_at {$dt_type} DEFAULT NULL,
			expires_at {$dt_type} DEFAULT NULL,
			created_at {$dt_type} NOT NULL DEFAULT {$default_dt_keyword},
			updated_at {$dt_type} NOT NULL DEFAULT {$default_dt_keyword} ON UPDATE {$default_dt_keyword},
			PRIMARY KEY  (id),
			KEY idx_query_ready (status,next_attempt_at,priority,scheduled_at),
			UNIQUE KEY idx_unique_active (unique_key(191),status),
			KEY idx_stalled_check (status,lock_expires_at),
			KEY idx_lock_token (lock_token),
			KEY idx_dependency_check (depends_on_job_id,status),
			KEY idx_status_category_prio (status,category,priority),
			KEY idx_job_group_status (job_group,status),
			KEY idx_op_type_status (operation_type,status),
			KEY idx_correlation_id (correlation_id),
			KEY idx_created_at (created_at),
			KEY idx_expires_at_status (expires_at,status)
		) {$charset_collate};";

		// --- Retry History Table ---
		$tables[] = "CREATE TABLE {$history_table_name} (
			history_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			job_id BIGINT UNSIGNED NOT NULL,
			attempt_number INT UNSIGNED NOT NULL,
			status VARCHAR(20) NOT NULL DEFAULT {$default_empty_string_sql},
			started_at {$dt_type} NOT NULL DEFAULT {$default_dt_keyword},
			finished_at {$dt_type} DEFAULT NULL,
			duration_ms INT UNSIGNED DEFAULT NULL,
			error_message TEXT DEFAULT NULL,
			error_code VARCHAR(100) DEFAULT NULL,
			stack_trace LONGTEXT DEFAULT NULL,
			processor_id VARCHAR(100) DEFAULT NULL,
			log_context LONGTEXT DEFAULT NULL,
			state_metadata TEXT DEFAULT NULL,
			PRIMARY KEY  (history_id),
			UNIQUE KEY idx_job_attempt (job_id,attempt_number),
			KEY idx_job_status (job_id,status),
			KEY idx_started_at (started_at),
			KEY idx_finished_at (finished_at),
			KEY idx_status_code (status,error_code),
			KEY idx_processor (processor_id,started_at)
		) {$charset_collate};";

		// --- Retry DLQ (Dead Letter Queue) Table ---
		$tables[] = "CREATE TABLE {$dlq_table_name} (
			dlq_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
			original_job_id BIGINT UNSIGNED NOT NULL,
			category VARCHAR(50) NOT NULL DEFAULT {$default_empty_string_sql},
			operation_type VARCHAR(100) NOT NULL DEFAULT {$default_empty_string_sql},
			unique_key VARCHAR(191) DEFAULT NULL,
			correlation_id VARCHAR(100) DEFAULT NULL,
			job_group VARCHAR(100) DEFAULT NULL,
			operation_data LONGTEXT DEFAULT NULL,
			metadata LONGTEXT DEFAULT NULL,
			final_status VARCHAR(20) NOT NULL DEFAULT {$default_dlq_reason_sql},
			retry_count INT UNSIGNED NOT NULL DEFAULT {$default_retry_count_sql},
			max_attempts INT UNSIGNED NOT NULL DEFAULT {$default_max_attempts_sql},
			last_error TEXT DEFAULT NULL,
			last_error_code VARCHAR(100) DEFAULT NULL,
			last_attempt_at {$dt_type} DEFAULT NULL,
			created_at {$dt_type} NOT NULL,
			failed_at {$dt_type} NOT NULL DEFAULT {$default_dt_keyword},
			last_history_ids TEXT DEFAULT NULL,
			PRIMARY KEY  (dlq_id),
			UNIQUE KEY idx_original_job (original_job_id),
			KEY idx_failed_at (failed_at),
			KEY idx_category_op_dlq (category,operation_type),
			KEY idx_final_status_dlq (final_status),
			KEY idx_correlation_dlq (correlation_id),
			KEY idx_group_dlq (job_group)
		) {$charset_collate};";

		return $tables;
	}

    /** Placeholder for get_current_utc_time */
    protected function get_current_utc_time(): \DateTimeImmutable
    {
        return new \DateTimeImmutable('now', new \DateTimeZone('UTC'));
    }

    /** Placeholder for format_datetime_for_sql */
    protected function format_datetime_for_sql(?\DateTimeImmutable $datetime): ?string
    {
        if ($datetime === null) return null;
    // Ensure correct format based on DB type (DATETIME vs DATETIME(6))
        return $datetime->format(self::$datetime_type === 'DATETIME(6)' ? 'Y-m-d H:i:s.u' : 'Y-m-d H:i:s');
    }

    /** Placeholder for clear_stats_cache */
    protected function clear_stats_cache(): void
    {
        if ($this->get_retry_config()['cache_enabled'] && wp_using_ext_object_cache()) {
            wp_cache_delete(self::STATS_CACHE_KEY, self::CACHE_GROUP);
        }
    }

    /** Placeholder for get_recently_locked_cache_key */
    protected function get_recently_locked_cache_key(): string
    {
        return 'lha_recently_locked';
    }

    /** Placeholder for get_job_lock_cache_key */
    protected function get_job_lock_cache_key(int $job_id): string
    {
        return 'lha_job_lock_' . $job_id;
    }

    /**
     * Move a job to the Dead Letter Queue (DLQ)
     * 
     * Moves a failed job that has exceeded retry limits to a separate DLQ table
     * for manual review and potential reprocessing.
     *
     * @param array|int $job_data_or_id Job data array or job ID
     * @param string $reason Human-readable reason for moving to DLQ
     * @param string $dlq_reason_code Machine-readable reason code
     * @param string|null $lock_token Optional lock token for verification
     * @param array|null $original_job_for_log Optional original job data for logging
     * @return bool True on success, false on failure
     */
    protected function move_to_dlq($job_data_or_id, string $reason, string $dlq_reason_code, ?string $lock_token = null, ?array $original_job_for_log = null): bool
    {
        global $wpdb;
        
        // Get full job data if only ID was passed
        if (is_int($job_data_or_id)) {
            $job_id = $job_data_or_id;
            $retry_table = $wpdb->prefix . self::RETRY_TABLE_BASENAME;
            
            // Optimized to select only needed columns
            $job_data = $wpdb->get_row(
                $wpdb->prepare("SELECT id, asset_id, original_url, type, status, priority, attempts, scheduled_at, last_error, metadata FROM {$retry_table} WHERE id = %d", $job_id),
                ARRAY_A
            );
            
            if (!$job_data) {
                \LHA\Logging::log_error("Cannot move job to DLQ: Job ID {$job_id} not found");
                return false;
            }
        } else {
            $job_data = $job_data_or_id;
            $job_id = $job_data['id'] ?? null;
            
            if (!$job_id) {
                \LHA\Logging::log_error("Cannot move job to DLQ: No job ID in job data");
                return false;
            }
        }
        
        // Log the DLQ move
        \LHA\Logging::log_warning(
            "Moving job to DLQ. ID: {$job_id}, Reason: {$reason}, Code: {$dlq_reason_code}",
            [
                'job_id' => $job_id,
                'category' => $job_data['category'] ?? null,
                'operation_type' => $job_data['operation_type'] ?? null,
                'retry_count' => $job_data['retry_count'] ?? 0,
                'reason' => $reason,
                'reason_code' => $dlq_reason_code
            ]
        );
        
        // Get table names
        $retry_table = $wpdb->prefix . self::RETRY_TABLE_BASENAME;
        $dlq_table = $wpdb->prefix . self::RETRY_DLQ_TABLE_BASENAME;
        
        // Insert into DLQ table
        $dlq_data = [
            'original_job_id' => $job_id,
            'category' => $job_data['category'] ?? '',
            'operation_type' => $job_data['operation_type'] ?? '',
            'unique_key' => $job_data['unique_key'] ?? null,
            'correlation_id' => $job_data['correlation_id'] ?? null,
            'job_group' => $job_data['job_group'] ?? null,
            'operation_data' => $job_data['operation_data'] ?? null,
            'metadata' => $job_data['metadata'] ?? null,
            'final_status' => $dlq_reason_code,
            'retry_count' => $job_data['retry_count'] ?? 0,
            'max_attempts' => $job_data['max_attempts'] ?? 0,
            'last_error' => $job_data['last_error'] ?? null,
            'last_error_code' => $job_data['last_error_code'] ?? null,
            'last_attempt_at' => $job_data['last_attempt_at'] ?? null,
            'created_at' => $job_data['created_at'] ?? current_time('mysql', true),
        ];
        
        $inserted = $wpdb->insert($dlq_table, $dlq_data);
        
        if ($inserted === false) {
            \LHA\Logging::log_error("Failed to insert job {$job_id} into DLQ table");
            return false;
        }
        
        // Delete from retry queue
        $deleted = $wpdb->delete(
            $retry_table,
            ['id' => $job_id],
            ['%d']
        );
        
        if ($deleted === false) {
            \LHA\Logging::log_error("Failed to delete job {$job_id} from retry queue when moving to DLQ");
            return false;
        }
        
        // Apply filter to allow custom DLQ handling
        do_action('lha_job_moved_to_dlq', $job_data, $reason, $dlq_reason_code);
        
        return true;
    }

	/**
	 * Generates the CREATE TABLE SQL statement for a given table definition.
	 *
	 * @param array $table_def Table definition array from get_table_definitions().
	 * @param string $charset_collate Database charset and collation.
	 * @return string The CREATE TABLE SQL statement.
	 * @throws \InvalidArgumentException If definition is invalid.
	 */
	public function generate_create_table_sql(array $table_def, string $charset_collate): string {
		if (empty($table_def['name']) || empty($table_def['columns']) || !is_array($table_def['columns'])) {
			throw new \InvalidArgumentException('Table definition must include a non-empty name and a columns array.');
		}
		$table_name = $table_def['name'];
		$column_sqls = [];
		$index_sqls = [];

		// Columns
		foreach ($table_def['columns'] as $col_name => $properties) {
			if (empty($properties['Type'])) {
				throw new \InvalidArgumentException("Column '{$col_name}' in table '{$table_name}' is missing the 'Type' property.");
			}
			$sql = "`" . esc_sql($col_name) . "` " . $properties['Type'];
			$sql .= (isset($properties['Null']) && $properties['Null'] === 'NO') ? ' NOT NULL' : ' NULL';

			// Default values
			if (array_key_exists('Default', $properties)) {
				$default = $properties['Default'];
				if ($default === null) {
					// Only add DEFAULT NULL if the column is actually nullable
					if (!isset($properties['Null']) || $properties['Null'] !== 'NO') {
						$sql .= ' DEFAULT NULL';
					}
				} elseif (is_string($default)) {
					// Check for SQL functions/keywords vs literal strings
					$upper_default = strtoupper($default);
					if ($upper_default === 'CURRENT_TIMESTAMP' || $upper_default === 'CURRENT_TIMESTAMP(6)' || $upper_default === 'NOW()' || $default === "'1970-01-01 00:00:01.000000'") {
						$sql .= " DEFAULT " . $default; // Keep case if specified, like CURRENT_TIMESTAMP(6)
					} else {
						// Treat as a literal string, requires quoting
						$sql .= " DEFAULT '" . esc_sql($default) . "'";
					}
				} elseif (is_numeric($default)) {
					$sql .= " DEFAULT " . $default; // Numeric defaults don't need quotes
				} elseif (is_bool($default)) {
					$sql .= " DEFAULT " . ($default ? '1' : '0'); // Boolean defaults
				}
				// Note: No explicit handling for other types like arrays - assumed covered by above
			}

			// Extra attributes (e.g., AUTO_INCREMENT)
			if (!empty($properties['Extra'])) {
				$sql .= " " . $properties['Extra'];
			}
			$column_sqls[] = $sql;
		}

		// Indexes
		if (!empty($table_def['indexes']) && is_array($table_def['indexes'])) {
			foreach ($table_def['indexes'] as $index_name => $index_def) {
				if (empty($index_def['columns']) || !is_array($index_def['columns'])) {
					Logging::log_warning("[Retry Schema] Skipping invalid index definition for '{$index_name}' in table '{$table_name}'.", 'warning');
					continue;
				}

				$index_cols = array_map(function ($col) {
					// Handle column lengths like `column_name(191)`
					if (preg_match('/^([\w_]+)\s*\((\d+)\)$/', $col, $matches)) {
						return "`" . esc_sql($matches[1]) . "`(" . intval($matches[2]) . ")";
					}
					// Standard column name
					return "`" . esc_sql($col) . "`";
				}, $index_def['columns']);

				$index_cols_sql = implode(', ', $index_cols);
				$index_type = !empty($index_def['unique']) ? 'UNIQUE KEY' : 'KEY';

				if (strtoupper($index_name) === 'PRIMARY') {
					// Primary keys might already be defined via AUTO_INCREMENT, but defining explicitly is robust.
					$index_sqls[] = "PRIMARY KEY ({$index_cols_sql})";
				} else {
					$index_sqls[] = "{$index_type} `" . esc_sql($index_name) . "` ({$index_cols_sql})";
				}
			}
		}

		// Combine parts
		$all_parts = array_merge($column_sqls, $index_sqls);
		$table_parts_sql = implode(",\n    ", $all_parts);
		// Escape table name in case it contains special characters (though prefix usually prevents this)
		$escaped_table_name = "`" . str_replace('`', '``', $table_name) . "`";

		// Final SQL statement
		return "CREATE TABLE {$escaped_table_name} (\n    {$table_parts_sql}\n) {$charset_collate};";
	}

	/**
	 * Creates or updates ALL retry infrastructure tables using dbDelta.
	 * Handles schema versioning. Relies on get_retry_table_definitions() to provide SQL.
	 *
	 * @return bool True on success (all tables checked/created/updated without error), false otherwise.
	 */
	public function create_retry_infrastructure_tables(): bool {
		global $wpdb;

		// 1. Dependency Check: dbDelta function
		if ( ! function_exists( 'dbDelta' ) ) {
			// Check if ABSPATH is defined before using it
			if ( ! defined( 'ABSPATH' ) ) {
				// Use your logging class if available, otherwise fallback
				if ( class_exists( 'Logging' ) ) {
					Logging::log_error( "[Retry Schema] CRITICAL: ABSPATH constant not defined. Cannot include dbDelta.", 'critical' );
				} else {
					error_log("[Retry Schema] CRITICAL: ABSPATH constant not defined. Cannot include dbDelta.");
				}
				return false;
			}
			require_once ABSPATH . 'wp-admin/includes/upgrade.php';
		}

		// Check again after include attempt
		if ( ! function_exists( 'dbDelta' ) ) {
			if ( class_exists( 'Logging' ) ) {
				Logging::log_error( "[Retry Schema] CRITICAL: dbDelta function not available after include attempt. Cannot create/update tables.", 'critical' );
			} else {
				error_log("[Retry Schema] CRITICAL: dbDelta function not available after include attempt. Cannot create/update tables.");
			}
			return false;
		}

		// 2. Get SQL Definitions
		$sql_statements = $this->get_retry_table_definitions();
		if ( empty( $sql_statements ) ) {
			if ( class_exists( 'Logging' ) ) {
				Logging::log_error( "[Retry Schema] CRITICAL: No SQL table definitions were generated by get_retry_table_definitions(). Cannot proceed.", 'critical' );
			} else {
				error_log("[Retry Schema] CRITICAL: No SQL table definitions were generated by get_retry_table_definitions(). Cannot proceed.");
			}
			return false;
		}

		$all_ok = true; // Assume success until an error occurs
		$schema_changed = false; // Track if dbDelta reported any changes

		if ( class_exists( 'Logging' ) ) {
			Logging::log_info( "[Retry Schema] Starting schema check/update for " . count( $sql_statements ) . " tables.", 'info' );
		}

		// 3. Process Each Table Schema
		foreach ( $sql_statements as $sql ) {
			// Attempt to parse the table name from the SQL for logging purposes
			$table_name = '(unknown table)'; // Default if parsing fails
			if ( preg_match( '/CREATE TABLE\s+`?([^`\s\(]+)`?/', $sql, $matches ) ) {
				$table_name = $matches[1]; // Get the captured table name (strip prefix if desired later)
			}

			if ( class_exists( 'Logging' ) ) {
				Logging::log_debug( "[Retry Schema] Processing schema for table '{$table_name}'.", 'debug' ); // SQL can be large, keep in debug or context
			}

			// Run dbDelta for this specific table's SQL
			try {
				// Clear previous WPDB errors *before* calling dbDelta
				$wpdb->last_error = '';
				$wpdb->suppress_errors(false); // Ensure errors are not suppressed if previously set

				// Call dbDelta. It returns an array of messages describing actions/errors.
				$result_messages = dbDelta( $sql );

				// Capture potential DB error immediately after dbDelta
				$db_error = $wpdb->last_error;

				// Check for Errors (Both $wpdb->last_error and dbDelta's output heuristics)
				$delta_has_errors = false;
				if ( ! empty( $result_messages ) ) {
					foreach ( $result_messages as $message ) {
						// Check for common dbDelta error/warning patterns (case-insensitive)
						// Note: "Changed type" messages are normal and not errors
						$is_type_change = stripos($message, 'Changed type') !== false;
						
						if (!$is_type_change && (stripos( $message, 'error' ) !== false || stripos( $message, 'fail' ) !== false || stripos( $message, 'invalid' ) !== false)) {
							// Only treat as error if it's not a type change and contains actual error keywords
							$delta_has_errors = true;
							// Log the specific problematic message if needed
							// Logging::log_debug("[Retry Schema] dbDelta potential issue message for '{$table_name}': {$message}", 'debug');
							break; // One error is enough to flag the table
						}
						// Check if any actual change was reported (for schema_changed flag)
						if (stripos($message, 'created') !== false || stripos($message, 'altered') !== false || stripos($message, 'Changed type') !== false) {
							$schema_changed = true;
						}
					}
				}

				// Determine if this table processing failed
				if ( $db_error !== '' || $delta_has_errors ) {
					$all_ok = false; // Mark failure if *any* table has an issue
					if ( class_exists( 'Logging' ) ) {
						Logging::log_error(
							"[Retry Schema] Error processing schema for table '{$table_name}'.",
							'error',
							[
								'wpdb_last_error'  => $db_error !== '' ? $db_error : '(empty)',
								'dbdelta_messages' => $result_messages,
								// Optionally include SQL in error logs if safe/needed for debugging AND not too large
								// 'sql_statement' => substr($sql, 0, 1000) // Example: Log truncated SQL
							]
						);
					} else {
						error_log("[Retry Schema] Error processing schema for table '{$table_name}'. WPDB Error: " . ($db_error !== '' ? $db_error : '(empty)') . " dbDelta Messages: " . implode('; ', $result_messages));
					}
				} else {
					if ( class_exists( 'Logging' ) ) {
						// Log success, potentially with messages if needed for info
						Logging::log_info( "[Retry Schema] Table '{$table_name}' checked/updated successfully.", 'info', ['dbdelta_messages' => $result_messages] );
					}
				}

			} catch ( \Throwable $e ) { // Catch any unexpected PHP errors during dbDelta execution
				$all_ok = false;
				if ( class_exists( 'Logging' ) ) {
					Logging::log_error( "[Retry Schema] Unexpected PHP error during dbDelta execution for table '{$table_name}': " . $e->getMessage(), 'critical', ['exception' => $e] );
				} else {
					error_log("[Retry Schema] Unexpected PHP error during dbDelta execution for table '{$table_name}': " . $e->getMessage());
				}
			}
		} // End foreach SQL statement

		// 4. Update Schema Version Option (Only if ALL tables were processed without error)
		if ( $all_ok ) {
			$option_key      = self::DB_VERSION_OPTION_NAME; // Use the constant
			$current_version = get_option( $option_key, null ); // Default to null if not set
			$new_version     = self::SCHEMA_VERSION;

			// Update only if the version is different
			if ( $current_version !== $new_version ) {
				if ( class_exists( 'Logging' ) ) {
					Logging::log_info( "[Retry Schema] Attempting to update schema version from '{$current_version}' to '{$new_version}'. Schema changed: " . ($schema_changed ? 'Yes' : 'No'), 'info' );
				}

				// Use 'no' (false) for autoload parameter as this isn't needed on every page load
				$updated = update_option( $option_key, $new_version, false );

				// Check if the update_option call itself failed
				// Note: update_option returns false if the value hasn't changed OR if there was an error
				// We already checked that the value is different above, so false means an actual error
				if ( $updated ) {
					if ( class_exists( 'Logging' ) ) {
						Logging::log_info( "[Retry Schema] Update process completed successfully. Schema version updated to '{$new_version}'.", 'info' );
					}
				} else {
					// Verify the value was actually set despite update_option returning false
					$verify_version = get_option( $option_key, null );
					if ( $verify_version == $new_version ) {
						// Value is correct, update_option just returned false (common with some caching plugins)
						if ( class_exists( 'Logging' ) ) {
							Logging::log_info( "[Retry Schema] Schema version verified as '{$new_version}' (update_option returned false but value is correct).", 'info' );
						}
					} else {
						// Actual failure - value wasn't set
						$option_update_db_error = $wpdb->last_error;
						if ( class_exists( 'Logging' ) ) {
							Logging::log_error( "[Retry Schema] Update process completed for tables, but FAILED to update schema version option '{$option_key}' in database. Check permissions/DB.", 'critical', ['error' => $option_update_db_error !== '' ? $option_update_db_error : 'update_option returned false', 'expected' => $new_version, 'actual' => $verify_version] );
						} else {
							error_log("[Retry Schema] Update process completed for tables, but FAILED to update schema version option '{$option_key}'. Error: " . ($option_update_db_error !== '' ? $option_update_db_error : 'update_option returned false'));
						}
						// NOTE: $all_ok remains true because tables are structurally OK.
						// This failure indicates a problem saving the version marker, not the table structure itself.
					}
				}
			} else {
				if ( class_exists( 'Logging' ) ) {
					Logging::log_info( "[Retry Schema] Update process completed. Schema version '{$new_version}' already up-to-date.", 'info' );
				}
			}
		} else {
			if ( class_exists( 'Logging' ) ) {
				Logging::log_error( "[Retry Schema] Schema update process FAILED for one or more tables. Schema version NOT updated.", 'critical' );
			} else {
				error_log("[Retry Schema] Schema update process FAILED for one or more tables. Schema version NOT updated.");
			}
		}

		return $all_ok;
	}
	/**
	 * Checks if the database schema version matches the code version.
	 * Logs discrepancies and registers WP Admin notices for administrators.
	 * Should typically run during 'admin_init' or a similar admin-only hook.
	 *
	 * @since x.y.z
	 */
	protected function check_schema_version(): void
	{
		// Check if the required constant is defined.
		if (!defined(__CLASS__ . '::SCHEMA_VERSION')) {
			// Log a critical error if the constant isn't defined - fundamental issue.
			$error_message = sprintf(
			// translators: %s: The fully qualified name of the missing constant (e.g., MyPlugin\SchemaManager::SCHEMA_VERSION).
				esc_html__('[LHA Retry] CRITICAL: The required constant %s is not defined. Cannot perform schema check.', self::TEXT_DOMAIN),
				'<code>' . esc_html(__CLASS__ . '::SCHEMA_VERSION') . '</code>' // Use esc_html here for safety if displayed raw
			);

			// Use is_callable for a more robust check
			if (is_callable(['Logging', 'log_error'])) {
				// Strip potential HTML added for the admin notice before logging
				Logging::log_error(strip_tags($error_message), 'critical');
			} elseif (function_exists('error_log')) {
				// Fallback to PHP's error log
				error_log(strip_tags($error_message)); // strip_tags for cleaner log output
			}

			// Display an admin notice for this critical configuration error.
			// Ensure the notice callback function exists and hook is not already added.
			if (method_exists($this, 'display_missing_constant_notice') && !has_action('admin_notices', [$this, 'display_missing_constant_notice'])) {
				add_action('admin_notices', [$this, 'display_missing_constant_notice']);
			}
			return; // Stop execution since the constant is missing
		}

		// Fetch versions
		$db_version   = (int) get_option(self::DB_VERSION_OPTION_NAME, 0);
		$code_version = (int) self::SCHEMA_VERSION; // Constant exists, safe to access

		// --- Database Outdated ---
		if ($db_version < $code_version) {
			$message = sprintf(
			// translators: %1$d: Current DB schema version, %2$d: Required code schema version.
				__('[LHA Retry] Database schema is outdated (DB version: %1$d, Code version: %2$d). Retry system may not function correctly. Please run the upgrade process (e.g., re-activate the plugin or trigger the schema update routine).', self::TEXT_DOMAIN),
				$db_version,
				$code_version
			);

			// Log the critical error
			if (is_callable(['Logging', 'log_error'])) {
				Logging::log_error($message, 'critical');
			} elseif (function_exists('error_log')) {
				error_log('ERROR: ' . $message); // Prefix with ERROR for clarity
			}

			// Register an admin notice hook if the method exists and hook isn't already added.
			if (method_exists($this, 'display_db_outdated_notice') && !has_action('admin_notices', [$this, 'display_db_outdated_notice'])) {
				add_action('admin_notices', [$this, 'display_db_outdated_notice']);
			}

			// --- Code Outdated ---
		} elseif ($db_version > $code_version) {
			$message = sprintf(
			// translators: %1$d: Current code schema version, %2$d: Current DB schema version.
				__('[LHA Retry] Code version (%1$d) is older than the database schema version (%2$d). Please update the Locally Host Assets plugin code. Functionality may be limited or incorrect.', self::TEXT_DOMAIN),
				$code_version,
				$db_version
			);

			// Log the warning
			if (is_callable(['Logging', 'log_warning'])) {
				Logging::log_warning($message, 'warning');
			} elseif (function_exists('error_log')) {
				error_log('WARNING: ' . $message); // Prefix with WARNING
			}

			// Register an admin notice hook if the method exists and hook isn't already added.
			if (method_exists($this, 'display_code_outdated_notice') && !has_action('admin_notices', [$this, 'display_code_outdated_notice'])) {
				add_action('admin_notices', [$this, 'display_code_outdated_notice']);
			}

			// --- Versions Match ---
		} else {
			// Log success only if debugging is enabled.
			if (defined('WP_DEBUG') && WP_DEBUG && defined('WP_DEBUG_LOG') && WP_DEBUG_LOG) {
				// Format message only when needed
				$message = sprintf(
				// translators: %d: Current matching schema version.
					__('[LHA Retry] Database schema version (%d) matches code version.', self::TEXT_DOMAIN),
					$db_version // or $code_version, they are equal
				);

				// Log the debug message
				if (is_callable(['Logging', 'log_debug'])) {
					Logging::log_debug($message, 'debug');
				}
				// No error_log fallback for debug messages by default
			}
		}
	}
	// =========================================================================
	// Initialization & Configuration
	// =========================================================================


	/**
	 * Displays an admin notice for an outdated database schema.
	 * Hooked into 'admin_notices'. Must be public static.
	 *
	 * @since x.y.z
	 */
	public function display_db_outdated_notice(): void
	{
		// --- Permission Check ---
		if (!current_user_can('manage_options')) { // Or 'update_plugins' or a custom capability
			return;
		}

		// --- Re-check Condition & Constant Existence ---
		// Crucial check: Ensure constant exists *before* accessing it.
		// Also re-validates the condition, as state might have changed between
		// 'admin_init' and 'admin_notices' (though unlikely for this case).
		if (!defined(__CLASS__ . '::SCHEMA_VERSION')) {
			// The 'missing constant' notice might already be queued by check_schema_version.
			// Avoid potential duplicate notices; simply return or log if needed.
			return;
		}
		$code_version = (int) self::SCHEMA_VERSION;
		$db_version   = (int) get_option(self::DB_VERSION_OPTION_NAME, 0);

		// Only display if the DB is *still* outdated
		if ($db_version < $code_version) {
			?>
            <div class="notice notice-error is-dismissible lha-notice lha-notice-db-outdated">
                <p>
                    <strong><?php esc_html_e('[Locally Host Assets - Retry System] Database Update Required:', self::TEXT_DOMAIN); ?></strong>
					<?php
					printf(
					/* translators: 1: Database version number (e.g., 1), 2: Code version number (e.g., 2). */
						esc_html__('The database schema (version %1$s) is outdated and needs updating to match the plugin code (version %2$s). The retry system may not function correctly. Please run the upgrade process (e.g., by re-activating the plugin or visiting the plugin settings page if it triggers updates).', self::TEXT_DOMAIN),
						'<strong>' . esc_html((string) $db_version) . '</strong>',
						'<strong>' . esc_html((string) $code_version) . '</strong>'
					// Consider adding a link here if applicable:
					// , '<a href="' . esc_url( admin_url( 'plugins.php' ) ) . '">' . esc_html__( 're-activating the plugin', self::TEXT_DOMAIN ) . '</a>'
					);
					?>
                </p>
            </div>
			<?php
		}
	}

	/**
	 * Displays an admin notice for outdated plugin code.
	 * Hooked into 'admin_notices'. Must be public static.
	 *
	 * @since x.y.z
	 */
	public function display_code_outdated_notice(): void
	{
		// --- Permission Check ---
		if (!current_user_can('manage_options')) { // Or 'update_plugins'
			return;
		}

		// --- Re-check Condition & Constant Existence ---
		if (!defined(__CLASS__ . '::SCHEMA_VERSION')) {
			return; // Constant missing, can't compare. The other notice handles this.
		}
		$code_version = (int) self::SCHEMA_VERSION;
		$db_version   = (int) get_option(self::DB_VERSION_OPTION_NAME, 0);

		// Only display if the code is *still* outdated compared to the DB
		if ($db_version > $code_version) {
			?>
            <div class="notice notice-warning is-dismissible lha-notice lha-notice-code-outdated">
                <p>
                    <strong><?php esc_html_e('[Locally Host Assets - Retry System] Plugin Update Recommended:', self::TEXT_DOMAIN); ?></strong>
					<?php
					printf(
					/* translators: 1: Code version number (e.g., 1), 2: Database version number (e.g., 2). */
						esc_html__('The plugin code (version %1$s) is older than the database schema (version %2$s). Please update the Locally Host Assets plugin to the latest version. Functionality may be limited or incorrect.', self::TEXT_DOMAIN),
						'<strong>' . esc_html((string) $code_version) . '</strong>',
						'<strong>' . esc_html((string) $db_version) . '</strong>'
					// Consider adding a link here if applicable:
					// , '<a href="' . esc_url( network_admin_url( 'update-core.php' ) ) . '">' . esc_html__( 'update the Locally Host Assets plugin', self::TEXT_DOMAIN ) . '</a>'
					);
					?>
                </p>
            </div>
			<?php
		}
	}

	/**
	 * Displays an admin notice if the SCHEMA_VERSION constant is missing.
	 * Hooked into 'admin_notices'. Must be public static.
	 *
	 * @since x.y.z
	 */
	public function display_missing_constant_notice(): void
	{
		if (!current_user_can('manage_options')) {
			return;
		}
		// Optional: Re-check if the constant is *still* missing.
		// If it somehow got defined between admin_init and admin_notices, hide the notice.
		if (defined(__CLASS__ . '::SCHEMA_VERSION')) {
			return;
		}
		?>
        <div class="notice notice-error lha-notice lha-notice-config-error"> <!-- Consider adding is-dismissible -->
            <p>
                <strong><?php esc_html_e('[Locally Host Assets - Retry System] Configuration Error:', self::TEXT_DOMAIN); ?></strong>
				<?php
				printf(
				/* translators: %s: The name of the missing constant (e.g., YourClassName::SCHEMA_VERSION) */
					esc_html__('The required constant %s is missing or not defined within the plugin code. The plugin cannot function correctly. Please contact support or try reinstalling the plugin.', self::TEXT_DOMAIN),
					'<code>' . esc_html(__CLASS__ . '::SCHEMA_VERSION') . '</code>'
				);
				?>
            </p>
        </div>
		<?php
	}

	/**
	 * Hook the schema check into WordPress admin initialization.
	 *
	 * @since x.y.z
	 * @return void
	 */
	public function init_schema_check(): void
	{
		// Only hook actions if in the admin area
		if (is_admin()) {
			add_action('admin_init', [$this, 'check_schema_version']);
		}
	}

	/**
	 * Retrieves the retry system configuration.
	 *
	 * Uses a static cache for performance within a request. Applies filters to allow
	 * overrides, performs validation and sanitization on all values, and optionally
	 * logs the final configuration on the first load if logging is enabled.
	 *
	 * Assumes necessary class constants (PRIORITY_*, STRATEGY_*, LOCK_HEARTBEAT_INTERVAL,
	 * DEFAULT_CRON_INTERVAL) are defined within this class (e.g., lha\Retry).
	 * Assumes a filter mechanism like WordPress's apply_filters is available.
	 * Assumes a logging mechanism might be available (e.g., class Logging).
	 *
	 * @return array<string, mixed> Configuration options array. Keys match the defaults.
	 */
	public function get_retry_config(): array {
		// Return from static cache if already loaded
		if (self::$configCache !== null) {
			return self::$configCache;
		}

		$is_first_run = (self::$configCache === null); // Track if this is the first run in the request

		// --- Define Defaults ---
		// Using literals directly as MINUTE_IN_SECONDS/HOUR_IN_SECONDS are standard
		// and defining them globally can sometimes be problematic.
		$one_hour_in_seconds = 3600;
		$ten_minutes_in_seconds = 600;

		/** @var array<string, mixed> $defaults */
		$defaults = [
			'max_attempts'           => 7,        // Max times a task is attempted before DLQ. Min: 1.
			'initial_delay'          => 15,       // Seconds before the first retry. Min: 1.
			'backoff_factor'         => 1.6,      // Multiplier for delay (exponential). Min: 1.0.
			'max_delay'              => $one_hour_in_seconds, // Max seconds between retries. Min: 0.
			'batch_size'             => 50,       // Max tasks processed per run. Min: 1.
			'lock_timeout'           => $ten_minutes_in_seconds, // Seconds before lock expires. Min enforced.
			'jitter_factor'          => 0.4,      // % of delay to add/subtract randomly (0.0 to 1.0).
			'default_priority'       => self::PRIORITY_NORMAL, // Default task priority.
			'default_strategy'       => self::STRATEGY_EXPONENTIAL, // Default retry strategy.
			'dlq_retention_days'     => 90,       // Days to keep tasks in Dead Letter Queue. Min: 0 (forever).
			'history_retention_days' => 30,       // Days to keep task history. Min: 0 (forever).
			'concurrency_limit'      => 10,       // Max tasks running concurrently. Min: 1.
			'cache_enabled'          => true,     // Enable/disable internal caching where applicable.
			'cron_interval'          => self::DEFAULT_CRON_INTERVAL, // Identifier for the task runner schedule.
		];

		// --- Calculate Minimum Lock Timeout ---
		// Lock must last longer than the heartbeat interval to prevent premature expiry.
		$min_lock_timeout_buffer = 15; // Minimum seconds buffer above heartbeat.
		$min_required_lock_timeout = self::LOCK_HEARTBEAT_INTERVAL + $min_lock_timeout_buffer;

		// --- Apply Filters ---
		/**
		 * Filters the LHA retry system configuration BEFORE validation.
		 *
		 * @param array $defaults Default configuration values.
		 * @return array Filtered configuration values. Must return an array.
		 */
		$filter_tag = 'lha_retry_config'; // Use a specific prefix
		$filtered_config = apply_filters($filter_tag, $defaults);

		// --- Validate Filter Output ---
		if (!is_array($filtered_config)) {
			$log_message = sprintf(
				'[%s] Filter "%s" did not return an array. Falling back to default configuration.',
				__METHOD__,
				$filter_tag
			);
			// Log the warning consistently
			if (class_exists(Logging::class) && method_exists(Logging::class, 'log_warning')) {
				Logging::log_warning($log_message, ['filter_tag' => $filter_tag, 'returned_type' => gettype($filtered_config)]);
			} else {
				error_log("Warning: " . $log_message); // Fallback logging
			}
			$filtered_config = $defaults; // Fallback to defaults
		}

		// --- Validation & Sanitization ---
		/** @var array<string, mixed> $sanitized_config */
		$sanitized_config = [];
		$valid_priorities = [self::PRIORITY_LOW, self::PRIORITY_NORMAL, self::PRIORITY_HIGH];
		$valid_strategies = [self::STRATEGY_EXPONENTIAL, self::STRATEGY_LINEAR, self::STRATEGY_FIXED];

		foreach ($defaults as $key => $default_value) {
			// Use the value from the filtered config if it exists, otherwise use the default.
			$value = $filtered_config[$key] ?? $default_value;

			switch ($key) {
				// Integers >= 1
				case 'max_attempts':
				case 'initial_delay':
				case 'batch_size':
				case 'concurrency_limit':
					$sanitized_config[$key] = max(1, (int) $value);
					break;

				// Float >= 1.0
				case 'backoff_factor':
					$sanitized_config[$key] = max(1.0, (float) $value);
					break;

				// Integers >= 0
				case 'max_delay':
				case 'dlq_retention_days':
				case 'history_retention_days':
					$sanitized_config[$key] = max(0, (int) $value);
					break;

				// Integer >= minimum required timeout
				case 'lock_timeout':
					$sanitized_config[$key] = max($min_required_lock_timeout, (int) $value);
					break;

				// Float between 0.0 and 1.0 inclusive
				case 'jitter_factor':
					$sanitized_config[$key] = max(0.0, min(1.0, (float) $value));
					break;

				// Boolean
				case 'cache_enabled':
					// FILTER_VALIDATE_BOOLEAN handles '1', 'true', 'on', 'yes', '0', 'false', 'off', 'no', ''
					// FILTER_NULL_ON_FAILURE makes it return null for non-boolean values.
					$sanitized_config[$key] = filter_var($value, FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE) ?? (bool) $default_value;
					break;

				// Non-empty string
				case 'cron_interval':
					$sanitized_value = trim((string) $value);
					// Ensure it's not an empty string after trimming
					$sanitized_config[$key] = $sanitized_value !== '' ? $sanitized_value : (string) $default_value;
					break;

				// Integer matching defined priorities
				case 'default_priority':
					$input_priority = (int) $value;
					$sanitized_config[$key] = in_array($input_priority, $valid_priorities, true)
						? $input_priority
						: (int) $default_value; // Fallback to default
					break;

				// String matching defined strategies
				case 'default_strategy':
					$input_strategy = (string) $value;
					$sanitized_config[$key] = in_array($input_strategy, $valid_strategies, true)
						? $input_strategy
						: (string) $default_value; // Fallback to default
					break;

				// Should not happen if iterating over $defaults keys, but good for safety.
				default:
					// This indicates a developer error: a key exists in $defaults but isn't handled in the switch.
					$error_message = sprintf(
						'[%s] Unhandled default configuration key "%s" during sanitization. Assigning default value.',
						__METHOD__,
						$key
					);
					if (class_exists(Logging::class) && method_exists(Logging::class, 'log_warning')) {
						Logging::log_warning($error_message, ['key' => $key]);
					} else {
						trigger_error($error_message, E_USER_WARNING);
					}
					$sanitized_config[$key] = $default_value; // Assign default as a safe fallback
					break;
			}
		}

		// --- Optional Logging (Only on first calculation) ---
		if ($is_first_run) {
			// Check if logging is desired (e.g., based on another config or WP_DEBUG)
			$should_log_config = defined('WP_DEBUG') && WP_DEBUG; // Example condition
			// Or: $should_log_config = $this->get_logging_enabled_flag();

			if ($should_log_config) {
				$log_context = ['final_config' => $sanitized_config];
				// Optionally include original filtered values for comparison if debugging filters
				// $log_context['filtered_input'] = $filtered_config;
				$log_message = sprintf('[%s] Retry system configuration loaded and sanitized.', __METHOD__);

				if (class_exists(Logging::class) && method_exists(Logging::class, 'log_debug')) {
					Logging::log_debug($log_message, $log_context);
				} else {
					// Fallback using error_log, pretty print for readability
					error_log($log_message . ' Config: ' . json_encode($sanitized_config, JSON_PRETTY_PRINT));
				}
			}
		}

		// --- Cache Result ---
		self::$configCache = $sanitized_config;

		return self::$configCache;
	}

	/**
	 * Generates a unique ID for this processor instance (PHP process).
	 * Useful for tracking which worker processed which job/attempt.
	 * Ensures the ID is consistent within the same PHP process execution using static caching.
	 *
	 * The ID incorporates hostname, process ID, and a strong random component for uniqueness.
	 * It includes fallbacks for component retrieval failures and sanitizes the output.
	 * The resulting ID is guaranteed to be lowercase [a-z0-9_-] and adhere to MAX_ID_LENGTH.
	 *
	 * @return string Processor ID (max $this->MAX_ID_LENGTH chars, sanitized [a-z0-9_-]). Always lowercase.
	 *                Returns an empty string only if MAX_ID_LENGTH is 0 or negative.
	 * @throws \RuntimeException If secure random generation fails fundamentally
	 *                          (random_bytes throws an Exception). Fallback mechanisms handle
	 *                          most other internal exceptions/failures gracefully.
	 */
	public function generate_processor_id(): string
	{
		// 1. Return cached ID if already generated in this process
		if (self::$processorId !== null) {
			return self::$processorId;
		}

		$finalId = null; // Initialize
		$primaryGenerationFailed = false;
		$primaryFailureReason = '';
		$primaryException = null;

		try {
			// --- Gather Components ---

			// 2. Get Hostname (Best effort)
			$hostname = self::getHostnameComponent();

			// 3. Get Process ID (Best effort)
			$pid = self::getPidComponent();

			// 4. Generate strong random component (Cryptographically Secure)
			// This is the most critical part; failure here is escalated.
			try {
				// 12 bytes = 96 bits entropy => 24 hex characters. Sufficient.
				$random = bin2hex(random_bytes(12));
			} catch (\Exception $e) {
				// Re-throw a specific exception. This is the one failure mode that bubbles up
				// if fallbacks related to it also fail catastrophically.
				throw new \RuntimeException('Failed to generate secure random bytes for Processor ID.', 0, $e);
			}

			// --- Combine, Sanitize, Finalize ---

			// 5. Combine components into a raw ID string
			// Using a short prefix "p" to save space if MAX_ID_LENGTH is small
			$rawId = "p_{$hostname}_{$pid}_{$random}";

			// 6. Sanitize the raw ID using the helper function
			$sanitizedId = self::sanitize_id_component($rawId);

			// 7. Check if sanitization succeeded
			if ($sanitizedId === self::SANITIZATION_FAILED_INDICATOR) {
				$primaryGenerationFailed = true;
				$primaryFailureReason = 'Primary Processor ID sanitization failed or resulted in empty string. Raw: "' . $rawId . '"';
			} else {
				// 8. Limit length (applied *after* successful sanitization)
				$limitedId = substr($sanitizedId, 0, self::MAX_ID_LENGTH);

				// 9. Final check: Ensure trimming didn't result in an empty string (only if MAX > 0)
				// @phpstan-ignore-next-line Defensive check for future-proofing
				if ($limitedId === '' && self::MAX_ID_LENGTH > 0) {
					$primaryGenerationFailed = true;
					$primaryFailureReason = 'Processor ID became empty after length limiting (max=' . self::MAX_ID_LENGTH . '). Sanitized: "' . $sanitizedId . '"';
				} else {
					// Success! Assign the generated ID.
					$finalId = $limitedId;
				}
			}

		} catch (\RuntimeException $e) {
			// Catch the specific RuntimeException from random_bytes failure
			$primaryGenerationFailed = true;
			$primaryFailureReason = 'Critical failure during primary generation: ' . $e->getMessage();
			$primaryException = $e; // Keep original exception for context
		} catch (\Throwable $e) {
			// Catch any other unexpected error during primary generation
			$primaryGenerationFailed = true;
			$primaryFailureReason = sprintf(
				'Unexpected error during primary generation (%s): %s in %s:%d',
				get_class($e),
				$e->getMessage(),
				basename($e->getFile()), // Use basename for brevity
				$e->getLine()
			);
			$primaryException = $e;
		}

		// --- Fallback Logic ---
		// Triggered if primary generation failed for any reason caught above.
		if ($primaryGenerationFailed) {
			// Log the reason for entering the fallback path. Use a PSR-3 logger if available.
			$logMessage = '[Warning] Processor ID generation failed. Reason: ' . $primaryFailureReason . '. Using fallback ID.';
			if ($primaryException) {
				// Append exception details if available (especially for unexpected errors)
				$logMessage .= sprintf(
					' | Exception: %s in %s:%d',
					$primaryException->getMessage(),
					basename($primaryException->getFile()),
					$primaryException->getLine()
				);
				// Optionally log the full stack trace if needed for debugging
				// error_log("Stack Trace:\n" . $primaryException->getTraceAsString());
			}
			error_log($logMessage);


			// --- Fallback Stage 1: Standard Fallback using uniqid ---
			$fallbackRawId = null;
			try {
				// uniqid() is NOT secure/unique. Only a LAST RESORT. time() adds minor sorting potential.
				// Using more entropy flag increases uniqueness slightly. Prefix added here.
				$uniqidResult = uniqid(self::FALLBACK_ID_PREFIX, true);
				if (is_string($uniqidResult) && $uniqidResult !== '') {
					$fallbackRawId = $uniqidResult . '_' . time(); // Append time for a bit more variance
				} else {
					// Handle rare case where uniqid returns non-string or empty
					throw new \RuntimeException('uniqid() returned invalid value: ' . var_export($uniqidResult, true));
				}
			} catch (\Throwable $uniqidException) {
				// Extremely unlikely uniqid fails, but handle defensively
				error_log('[Critical Error] uniqid() failed during fallback ID generation: ' . $uniqidException->getMessage() . '. Using emergency random.');
				$fallbackRawId = self::FALLBACK_ID_PREFIX . 'emergency_rand_' . time() . '_' . mt_rand(10000, 99999);
			}

			// Sanitize the standard fallback ID
			$sanitizedFallback = self::sanitize_id_component($fallbackRawId);

			// --- Fallback Stage 2: Minimal Fallback if Standard Fallback Sanitization Fails ---
			if ($sanitizedFallback === self::SANITIZATION_FAILED_INDICATOR) {
				error_log('[Warning] Standard fallback ID sanitization failed (raw: "' . $fallbackRawId . '"). Using absolute minimal fallback.');
				// Use a simple prefix and time - should always sanitize reasonably
				$minimalFallbackRaw = self::ABSOLUTE_MIN_ID_PREFIX . time();

				// Use the robust critical constant sanitizer for this minimal fallback
				// This avoids repeating similar regex logic and leverages the most resilient sanitizer.
				$sanitizedFallback = self::sanitize_critical_constant($minimalFallbackRaw);
				error_log('[Info] Using minimal fallback base: ' . $sanitizedFallback);
			}
			// At this point, $sanitizedFallback holds the result of Stage 1 OR Stage 2 sanitization

			// Limit length for the chosen fallback ID (standard or minimal)
			$finalId = substr($sanitizedFallback, 0, self::MAX_ID_LENGTH);

			// --- Fallback Stage 3: Critical Fallback Constant if Fallback is Empty After Length Limit ---
			// @phpstan-ignore-next-line Defensive check for future-proofing
			if ($finalId === '' && self::MAX_ID_LENGTH > 0) {
				error_log('[Critical Error] Fallback ID resulted in empty string after length limit (' . self::MAX_ID_LENGTH . '). Using critical fallback constant.');
				// Use a predefined constant, ensuring it adheres to length limit and allowed chars
				$criticalBase = self::sanitize_critical_constant(self::CRITICAL_FALLBACK_ID);
				$finalId = substr($criticalBase, 0, self::MAX_ID_LENGTH);

				// Final check: If MAX_ID_LENGTH is so small even *this* is empty, use a single char.
				// @phpstan-ignore-next-line Defensive check for future-proofing
				if ($finalId === '' && self::MAX_ID_LENGTH > 0) {
					$finalId = 'e'; // Single character error indicator
					error_log('[Critical Error] Critical fallback ID was also empty after length limit. Using single char "e".');
				}
			}
			// @phpstan-ignore-next-line Defensive check for future-proofing
			elseif ($finalId === '' && self::MAX_ID_LENGTH <= 0) {
				// Respect MAX_ID_LENGTH being zero or negative, allow empty string result.
				$finalId = ''; // Explicitly set to empty string
				error_log('[Info] Using empty processor ID due to MAX_ID_LENGTH <= 0 in fallback path.');
			}

			// Log the final fallback ID chosen (unless empty due to limit)
			if ($finalId !== '') {
				error_log('[Info] Using fallback processor ID: ' . $finalId);
			}
			// else: Either empty due to limit (logged above) or 'e' (logged above).
		}

		// --- Final Assertion & Caching ---

		// Final type assertion and emergency fallback (should ideally never be needed)
		// Guards against unforeseen internal logic flaws leading to non-string $finalId.
		if (!is_string($finalId)) {
			error_log('[Critical Error] Failed to assign a string value to final Processor ID. Internal logic error suspected. Using emergency hardcoded string.');
			$emergencyBase = self::sanitize_critical_constant(self::CRITICAL_FALLBACK_ID . '_logic_err');
			$finalId = substr($emergencyBase, 0, self::MAX_ID_LENGTH);

			// Final guarantee of non-empty if MAX_ID_LENGTH > 0
			// @phpstan-ignore-next-line Defensive check for future-proofing
			if ($finalId === '' && self::MAX_ID_LENGTH > 0) $finalId = 'x'; // Ultimate fallback character
			// @phpstan-ignore-next-line Defensive check for future-proofing
            elseif ($finalId === '' && self::MAX_ID_LENGTH <= 0) $finalId = ''; // Respect limit

			error_log('[Critical Error] Emergency fallback ID used: ' . $finalId);
		}

		// Cache the final ID (whether successfully generated or fallback) for this process
		self::$processorId = $finalId;
		return self::$processorId;
	}

	/**
	 * Retrieves and prepares the hostname component for the ID.
	 * @return string Hostname component (sanitized later).
	 * @internal
	 */
	private static function getHostnameComponent(): string
	{
		$hostname = self::DEFAULT_HOSTNAME; // Start with default
		$rawHostname = false;

		// Prefer gethostname if available
		if (function_exists('gethostname')) {
			$rawHostname = gethostname(); // Returns string on success, false on failure
		}

		// Fallback if gethostname unavailable or failed
		if ($rawHostname === false && function_exists('php_uname')) {
			$rawHostname = php_uname('n'); // Might return false or string on failure/success.
		}

		// Process the retrieved hostname
		if ($rawHostname !== false && is_string($rawHostname)) {
			$trimmedHostname = trim($rawHostname);
			// Use indicator only if the system *explicitly* returns an empty string after trim
			$hostname = ($trimmedHostname !== '') ? $trimmedHostname : self::HOSTNAME_EMPTY_INDICATOR;
			// Log if the indicator is used (might be useful for diagnosing network/system config)
			if ($hostname === self::HOSTNAME_EMPTY_INDICATOR) {
				error_log('[Info] System returned an empty hostname; using indicator "' . self::HOSTNAME_EMPTY_INDICATOR . '".');
			}
		} elseif ($rawHostname !== false) {
			// Log if a non-string, non-false value was returned (unexpected)
			error_log('[Warning] Hostname retrieval function returned unexpected type: ' . gettype($rawHostname) . '. Using default "' . self::DEFAULT_HOSTNAME . '".');
			// $hostname remains self::DEFAULT_HOSTNAME
		} else {
			// Both attempts failed (returned false) or functions unavailable.
			// Log the failure to obtain hostname
			error_log('[Info] Could not retrieve hostname using gethostname() or php_uname(\'n\'). Using default "' . self::DEFAULT_HOSTNAME . '".');
			// $hostname remains self::DEFAULT_HOSTNAME
		}

		return $hostname;
	}

	/**
	 * Retrieves and prepares the process ID component for the ID.
	 * @return string PID component (sanitized later).
	 * @internal
	 */
	private static function getPidComponent(): string
	{
		$pid = self::DEFAULT_PID; // Start with default (unavailable/disabled)

		if (function_exists('getmypid')) {
			$pidResult = getmypid(); // Returns int on success, false on failure.
			if ($pidResult !== false) {
				$pid = (string) $pidResult;
			} else {
				// Use specific indicator if PID retrieval *failed*
				$pid = self::PID_FAILED_INDICATOR;
				error_log('[Warning] getmypid() function failed (returned false). Using indicator "' . self::PID_FAILED_INDICATOR . '".');
			}
		} else {
			// Log if function doesn't exist (might indicate a restricted environment)
			error_log('[Info] getmypid() function does not exist or is disabled. Using default "' . self::DEFAULT_PID . '".');
			// $pid remains self::DEFAULT_PID
		}
		return $pid;
	}


	/**
	 * Helper function to sanitize ID components or the final raw ID.
	 * Ensures the result contains only lowercase alphanumeric chars, underscore, hyphen [a-z0-9_-].
	 * Replaces sequences of invalid characters with a single underscore.
	 * Collapses multiple consecutive underscores into one underscore.
	 * Trims leading/trailing underscores.
	 *
	 * @param string $input The raw string to sanitize.
	 * @return string The sanitized string (lowercase [a-z0-9_-]),
	 *                or SANITIZATION_FAILED_INDICATOR if the input is effectively empty
	 *                after processing, if a preg_replace operation fails, or if input is null/empty.
	 *                Guaranteed non-empty on success.
	 */
	protected static function sanitize_id_component(string $input): string
	{
		// 1. Handle effectively empty input early
		$trimmedInput = trim($input);
		if ($trimmedInput === '') {
			// Input was empty or only whitespace. Return indicator.
			return self::SANITIZATION_FAILED_INDICATOR;
		}

		// 2. Convert to lowercase
		$lowercaseInput = strtolower($trimmedInput);

		// 3. Replace any sequence of one or more characters NOT in [a-z0-9_-] with a single underscore.
		//    PCRE modifier 'u' might be needed if input could be non-ASCII UTF-8, but target charset is simple.
		$sanitized = preg_replace('/[^a-z0-9_-]+/', '_', $lowercaseInput);

		// Handle potential preg_replace error (e.g., regex issues, memory limits, invalid UTF-8 if 'u' used)
		if ($sanitized === null) {
			$pcreError = function_exists('preg_last_error_msg') ? preg_last_error_msg() : 'code ' . preg_last_error();
			error_log('[Error] preg_replace (initial sanitization) failed in sanitize_id_component. Input (trimmed): "' . $trimmedInput . '". PCRE Error: ' . $pcreError);
			return self::SANITIZATION_FAILED_INDICATOR; // Indicate failure
		}

		// 4. Collapse multiple consecutive underscores (e.g., "a___b" -> "a_b")
		$collapsed = preg_replace('/_+/', '_', $sanitized);
		if ($collapsed === null) {
			$pcreError = function_exists('preg_last_error_msg') ? preg_last_error_msg() : 'code ' . preg_last_error();
			error_log('[Error] preg_replace (collapse underscores) failed in sanitize_id_component. Input after initial sanitize: "' . $sanitized . '". PCRE Error: ' . $pcreError);
			return self::SANITIZATION_FAILED_INDICATOR; // Indicate failure
		}

		// 5. Trim leading/trailing underscores that might result from replacements at ends
		$final = trim($collapsed, '_');

		// 6. Ensure it's not empty *after* all processing
		// (e.g., input was only invalid chars/underscores like "-_-" or "!!!")
		if ($final === '') {
			// Input consisted only of characters that were replaced or trimmed away.
			// Log this condition for potential debugging, but it's a valid sanitization outcome leading to failure indicator.
			error_log('[Info] Sanitization resulted in an empty string. Original input (trimmed): "' . $trimmedInput . '"');
			return self::SANITIZATION_FAILED_INDICATOR; // Indicate effective emptiness
		}

		return $final;
	}

	/**
	 * Sanitizes the CRITICAL_FALLBACK_ID constant or similar hardcoded/simple strings.
	 * Uses a simplified, robust sanitization. Assumes input is simple ASCII/UTF-8.
	 * Guarantees a non-empty result (uses a hardcoded value if sanitization itself fails badly).
	 *
	 * @param string $constantValue The value to sanitize.
	 * @return string Sanitized string [a-z0-9_-], guaranteed non-empty.
	 * @internal
	 */
	private static function sanitize_critical_constant(string $constantValue): string
	{
		$lower = strtolower($constantValue);
		// Basic replacement of anything not allowed
		$sanitized = preg_replace('/[^a-z0-9_-]+/', '_', $lower);

		if ($sanitized === null) { // preg_replace failed
			error_log('[Critical Error] preg_replace failed during critical constant sanitization (step 1). Input: ' . $constantValue);
			return 'crit_preg_fail'; // Should be extremely rare, but must return *something* valid
		}

		// Collapse underscores
		$collapsed = preg_replace('/_+/', '_', $sanitized);
		if ($collapsed === null) { // preg_replace failed again
			error_log('[Critical Error] preg_replace failed during critical constant sanitization (step 2). Input: ' . $sanitized);
			return 'crit_preg_fail2';
		}

		// Trim underscores
		$trimmed = trim($collapsed, '_');

		// Ensure non-empty output
		if ($trimmed === '') {
			error_log('[Critical Error] Critical constant sanitization resulted in empty string. Original: ' . $constantValue);
			return 'critical_empty'; // Must return something valid
		}

		return $trimmed;
	}

	/**
	 * Get the current processor ID for this process instance.
	 *
	 * @return string The processor ID.
	 */
	public function get_processor_id(): string {
		return self::$processorId ?: $this->generate_processor_id();
	}

	// =========================================================================
	// Job Management - Storing, Retrieving, Locking
	// =========================================================================

	/**
	 * Stores a job in the retry queue with advanced options.
	 * Handles uniqueness checks, dependencies, scheduling, and payload encryption hooks.
	 *
	 * @param string $category        Job category (e.g., 'download', 'api_call'). Use constants like $this->CATEGORY_DOWNLOAD.
	 * @param string $operation_type  Identifier for the executor function (must be registered).
	 * @param array  $operation_data  Payload for the executor (will be JSON encoded).
	 * @param array  $options {
	 *     Optional. An array of job options.
	 *
	 *     @type string          $unique_key        If set, prevents queuing if another *active* job with the same key exists. Max 191 chars. Active means not completed or in DLQ.
	 *     @type int             $priority          Job priority (lower value = higher priority). Default: PRIORITY_NORMAL. Use constants like $this->PRIORITY_HIGH.
	 *     @type array|null      $metadata          Additional non-essential context data (JSON encoded).
	 *     @type string          $strategy          Retry strategy ('exponential', 'fixed', 'linear', 'none'). Default: from config. Use constants like self::STRATEGY_EXPONENTIAL.
	 *     @type int             $max_attempts      Max attempts for this specific job. Default: from config.
	 *     @type int             $base_delay_sec    Base delay for this job (seconds). Default: from config.
	 *     @type float           $backoff_factor    Backoff factor for this job. Default: from config.
	 *     @type int|null        $max_delay_sec     Max delay cap for this job (seconds). Null/0 means no limit. Default: from config.
	 *     @type \DateTimeImmutable|null $scheduled_at If set, job won't run before this UTC time. Use `new \DateTimeImmutable('...', new \DateTimeZone('UTC'))`.
	 *     @type \DateTimeImmutable|null $expires_at   If set, job will be moved to DLQ if not completed by this UTC time. Use `new \DateTimeImmutable('...', new \DateTimeZone('UTC'))`.
	 *     @type int|null        $depends_on_job_id Job ID this job depends on. Status becomes 'waiting_dependency'. Job won't run until dependency succeeds or fails permanently.
	 *     @type string|null     $correlation_id    ID to correlate related jobs across systems/requests. Max 100 chars.
	 *     @type string|null     $job_group         Group identifier for concurrency control etc. Max 100 chars.
	 *     @type array|null      $state_metadata    Initial state for stateful executors (JSON encoded).
	 *     @type bool            $encrypt_payload   If true, attempts to encrypt $operation_data via 'sha_retry_encrypt_payload' filter. Default false.
	 * }
	 * @return int|false The inserted job ID on success, false on failure. Returns the existing *active* job ID if a unique key constraint is hit.
	 * @throws \InvalidArgumentException If required options are invalid (e.g., non-\DateTimeImmutable for dates, non-array metadata).
	 * @throws \RuntimeException If JSON encoding fails or the encryption filter returns an error.
	 */
	public function store_retry_job(string $category, string $operation_type, array $operation_data, array $options = []) /* : int|false */ {
		// Skip if Action Scheduler handles retries natively - zero performance impact
		if ( ! $this->should_handle_retries() ) {
			$this->logger->log_debug( '[Retry] Skipping store_retry_job - Action Scheduler handles retries natively' );
			return 0; // Return 0 to indicate "handled" (by AS) - truthy int value
		}
		
		global $wpdb;
		$config = $this->get_retry_config();
		$retry_table = $this->get_retry_table_name();
		
		// Acquire lock to prevent concurrent retry job creation for the same operation
		$lock_key = 'retry_store_' . md5($category . '|' . $operation_type . '|' . wp_json_encode($operation_data));
		$lock_acquired = false;
		
		try {
			$lock_acquired = $this->lock->acquire_with_backoff($lock_key, 30, 3, 50000, 200000);
			
			if (!$lock_acquired) {
				$this->logger->log_warning('[Retry Store] Failed to acquire lock for storing retry job. Another process may be creating it.', [
					'category' => $category,
					'operation_type' => $operation_type,
					'lock_key' => $lock_key
				]);
				return false;
			}
			
			$now_utc = $this->get_current_utc_time();

		// Truncate category and operation_type to match schema limits
		$category = substr($category, 0, 50); // VARCHAR(50)
		$operation_type = substr($operation_type, 0, 100); // VARCHAR(100)

		// --- 1. Extract & Validate Options ---
		$unique_key = isset($options['unique_key']) ? substr((string)$options['unique_key'], 0, 191) : null;
		$priority = $options['priority'] ?? $config['default_priority'];
		$priority = max(0, min(255, (int)$priority)); // Ensure TINYINT UNSIGNED range

		$metadata = $options['metadata'] ?? null;
		if ($metadata !== null && !is_array($metadata)) {
			throw new \InvalidArgumentException('$options[\'metadata\'] must be null or an array.');
		}

		$strategy = $options['strategy'] ?? $config['default_strategy'];
		if (!in_array($strategy, [self::STRATEGY_EXPONENTIAL, self::STRATEGY_FIXED, self::STRATEGY_LINEAR, self::STRATEGY_NONE])) {
			Logging::log_warning("[Retry Store] Invalid strategy '{$strategy}' provided for job type '{$operation_type}', using default '{$config['default_strategy']}'.", 'warning', ['provided' => $strategy]);
			$strategy = $config['default_strategy'];
		}

		$max_attempts = $options['max_attempts'] ?? $config['max_attempts'];
		$max_attempts = max(1, (int)$max_attempts); // Must allow at least one attempt

		$base_delay = $options['base_delay_sec'] ?? $config['initial_delay'];
		$base_delay = max(1, (int)$base_delay); // Minimum 1 second delay

		$backoff_factor = $options['backoff_factor'] ?? $config['backoff_factor'];
		$backoff_factor = max(1.0, (float)$backoff_factor); // Factor must be >= 1.0

		$max_delay = $options['max_delay_sec'] ?? $config['max_delay'];
		$max_delay = ($max_delay !== null) ? max(0, (int)$max_delay) : null; // Allow null or 0+

		$scheduled_at_input = $options['scheduled_at'] ?? null;
		if ($scheduled_at_input !== null && !($scheduled_at_input instanceof \DateTimeImmutable)) {
			throw new \InvalidArgumentException('$options[\'scheduled_at\'] must be null or an instance of \DateTimeImmutable.');
		}
		// Ensure timezone is UTC if provided
		if ($scheduled_at_input instanceof \DateTimeImmutable && $scheduled_at_input->getTimezone()->getName() !== 'UTC') {
			Logging::log_warning("[Retry Store] scheduled_at provided with non-UTC timezone. Converting to UTC.", 'warning');
			$scheduled_at_input = $scheduled_at_input->setTimezone(new \DateTimeZone('UTC'));
		}

		$expires_at = $options['expires_at'] ?? null;
		if ($expires_at !== null && !($expires_at instanceof \DateTimeImmutable)) {
			throw new \InvalidArgumentException('$options[\'expires_at\'] must be null or an instance of \DateTimeImmutable.');
		}
		if ($expires_at instanceof \DateTimeImmutable && $expires_at->getTimezone()->getName() !== 'UTC') {
			Logging::log_warning("[Retry Store] expires_at provided with non-UTC timezone. Converting to UTC.", 'warning');
			$expires_at = $expires_at->setTimezone(new \DateTimeZone('UTC'));
		}

		$depends_on = isset($options['depends_on_job_id']) ? (int)$options['depends_on_job_id'] : null;
		if ($depends_on !== null && $depends_on <= 0) $depends_on = null; // Treat 0 or negative as null

		$correlation_id = isset($options['correlation_id']) ? substr((string)$options['correlation_id'], 0, 100) : null;
		$job_group = isset($options['job_group']) ? substr((string)$options['job_group'], 0, 100) : null;

		$state_metadata = $options['state_metadata'] ?? null;
		if ($state_metadata !== null && !is_array($state_metadata)) {
			throw new \InvalidArgumentException('$options[\'state_metadata\'] must be null or an array.');
		}

		$encrypt_payload = !empty($options['encrypt_payload']);

		// --- 2. Check Uniqueness (if key provided) ---
		if ($unique_key !== null) {
			// Define statuses considered "active" (i.e., not permanently failed or successfully completed)
			// Excludes STATUS_FAILED (terminal in main table)
			// Excludes jobs successfully completed (deleted) or moved to DLQ
			$active_statuses = [
				self::STATUS_PENDING,
				self::STATUS_SCHEDULED,
				self::STATUS_PROCESSING,
				self::STATUS_WAITING_DEPENDENCY,
				self::STATUS_PAUSED
			];
			$status_placeholders = implode(',', array_fill(0, count($active_statuses), '%s'));
			$existing_sql = $wpdb->prepare(
				"SELECT id FROM `{$retry_table}` WHERE `unique_key` = %s AND `status` IN ({$status_placeholders}) LIMIT 1",
				array_merge([$unique_key], $active_statuses)
			);
			$existing_id = $wpdb->get_var($existing_sql);
			if ($existing_id) {
				Logging::log_info("[Retry Store] Active job with unique key '{$unique_key}' already exists (ID: {$existing_id}). Skipping creation.", 'info', ['unique_key' => $unique_key, 'existing_id' => $existing_id]);
				// Return the ID of the existing active job
				return (int)$existing_id;
			}
		}

		// --- 3. Dependency Check & Initial Status ---
		$status = self::STATUS_PENDING; // Default assumption

		if ($depends_on !== null) {
			// Check status of the dependency in the main table
			$dep_status_sql = $wpdb->prepare("SELECT status FROM `{$retry_table}` WHERE id = %d", $depends_on);
			$dep_status = $wpdb->get_var($dep_status_sql);

			// A. Dependency is explicitly marked as 'failed' in the main table (rare, usually moved to DLQ)
			if ($dep_status === self::STATUS_FAILED) {
				Logging::log_error("[Retry Store] Cannot queue job: Dependency job ID {$depends_on} has terminal status 'failed' in the main queue.", 'error', ['dependency_id' => $depends_on]);
				// Optionally: Move this new job directly to DLQ with reason DEPENDENCY_FAILED? Or just prevent creation?
				// Prevent creation is simpler and less prone to cluttering DLQ unnecessarily.
				return false;
			}

			// B. Check if dependency has permanently failed (is in the DLQ)
			$dlq_table = $this->get_dlq_table_name();
			$dep_failed_check_sql = $wpdb->prepare("SELECT 1 FROM `{$dlq_table}` WHERE original_job_id = %d LIMIT 1", $depends_on);
			$dep_in_dlq = $wpdb->get_var($dep_failed_check_sql);

			if ($dep_in_dlq) {
				Logging::log_error("[Retry Store] Cannot queue job: Dependency job ID {$depends_on} has failed permanently (found in DLQ).", 'error', ['dependency_id' => $depends_on]);
				// Optionally: Move this new job directly to DLQ?
				return false; // Prevent creation
			}

			// C. If the dependency exists in the main table and isn't failed/DLQ'd, it's still active or waiting. Set status to 'waiting_dependency'.
			if ($dep_status !== null) {
				$status = self::STATUS_WAITING_DEPENDENCY;
				Logging::log_info("[Retry Store] Job depends on active job ID {$depends_on} (Status: {$dep_status}). Setting status to 'waiting_dependency'.", 'info', ['dependency_id' => $depends_on]);
			} else {
				// D. Dependency ID not found in main table AND not in DLQ.
				// Assumption: It completed successfully and was deleted.
				// Alternative: It never existed (bad ID provided).
				// We proceed assuming success/completion for simplicity. Add warning.
				Logging::log_warning("[Retry Store] Dependency job ID {$depends_on} not found in active queue or DLQ. Assuming completed or ID invalid. Proceeding with status '{$status}'.", 'warning', ['dependency_id' => $depends_on]);
				// Status remains whatever it was before this check (e.g., PENDING or will become SCHEDULED below)
			}
		}

		// --- 4. Scheduling Logic & Next Attempt Time ---
		$next_attempt_time = null; // Calculated below
		$scheduled_at_for_db = null; // DB column value for original schedule request

		if ($scheduled_at_input instanceof \DateTimeImmutable) {
			$scheduled_at_for_db = $scheduled_at_input; // Store the original request time

			// Set next_attempt_at based on the schedule, but ensure it's not in the past relative to NOW
			// If scheduled for the past, it should run ASAP (like a pending job).
			$next_attempt_time = ($scheduled_at_for_db > $now_utc) ? $scheduled_at_for_db : $now_utc;

			// Only set status to SCHEDULED if it's genuinely waiting for a *future* time
			// AND it's not already waiting on a dependency (dependency takes precedence).
			if ($status !== self::STATUS_WAITING_DEPENDENCY && $scheduled_at_for_db > $now_utc) {
				$status = self::STATUS_SCHEDULED;
			}
			// If status is waiting_dependency, it stays that way, even if scheduled_at is set.
			// next_attempt_at will be updated when dependency completes.
		}

		// If not scheduled, or schedule was in the past, calculate initial next_attempt_at based on NOW + initial delay
		if ($next_attempt_time === null) {
			// Create a temporary job structure to calculate the delay for the *first* attempt (index 0)
			// We use retry_count = -1 so that calculate_delay computes the delay *before* the first attempt (attempt 0).
			$job_for_delay_calc = [
				'retry_strategy' => $strategy,
				'retry_count'    => -1, // Key: calculates delay for the upcoming 0th attempt
				'base_delay_sec' => $base_delay,
				'backoff_factor' => $backoff_factor,
				'max_delay_sec'  => $max_delay
			];
			$initial_delay_seconds = $this->calculate_delay($job_for_delay_calc);
			$next_attempt_time = $now_utc->modify("+{$initial_delay_seconds} seconds");

			// If status somehow ended up as SCHEDULED (e.g., schedule was in the past), revert to PENDING
			if ($status === self::STATUS_SCHEDULED) {
				$status = self::STATUS_PENDING;
			}
		}
		// If status is WAITING_DEPENDENCY, next_attempt_at is set here but won't be used until promotion.
		// The promotion logic will reset next_attempt_at to NOW when dependency clears.

		// --- 5. Payload Handling (JSON Encoding & Optional Encryption) ---
		try {
			$operation_data_json = wp_json_encode($operation_data);
		} catch (\Throwable $e) { // Catch potential JsonException in PHP 7.3+ if flags used
			Logging::log_error("[Retry Store] Failed to JSON encode operation_data.", 'error', ['error' => $e->getMessage()]);
			throw new \RuntimeException("Failed to JSON encode operation_data: " . $e->getMessage(), 0, $e);
		}
		if ($operation_data_json === false) { // Fallback check for older PHP or no exception
			throw new \RuntimeException("Failed to JSON encode operation_data (returned false).");
		}

		$final_operation_data_str = $operation_data_json; // Start with JSON string

		if ($encrypt_payload) {
			/**
			 * Filters the operation data payload before saving, allowing for encryption.
			 * The filter should return the encrypted string on success, or WP_Error on failure.
			 *
			 * @param string $payload_json    The JSON-encoded payload string.
			 * @param array  $original_data   The raw operation data array (before JSON).
			 * @param string $category        The job category.
			 * @param string $operation_type  The job operation type.
			 * @return string|WP_Error The potentially encrypted string, or WP_Error on encryption failure.
			 */
			$encrypted_result = apply_filters('sha_retry_encrypt_payload', $final_operation_data_str, $operation_data, $category, $operation_type);

			if (is_wp_error($encrypted_result)) {
				Logging::log_error("[Retry Store] Payload encryption filter failed.", 'error', ['error_code' => $encrypted_result->get_error_code(), 'error_message' => $encrypted_result->get_error_message()]);
				throw new \RuntimeException("Payload encryption filter failed: " . $encrypted_result->get_error_message());
			}
			if (is_string($encrypted_result) && $encrypted_result !== $final_operation_data_str) {
				// Assume successful encryption if it's a non-empty string and different from input
				$final_operation_data_str = $encrypted_result;
				Logging::log_debug("[Retry Store] Payload encrypted via filter.", 'debug', ['category' => $category, 'type' => $operation_type]);
			} elseif ($encrypted_result !== $final_operation_data_str) {
				// Filter returned something unexpected (not a string, or same string but didn't error)
				Logging::log_warning("[Retry Store] Encrypt filter returned an unexpected or unchanged value. Payload stored unencrypted.", 'warning', ['category' => $category, 'type' => $operation_type]);
			}
			// If filter returns the exact same string, assume no encryption was performed.
		}

		// --- 6. Prepare Final Data & Format for wpdb->insert ---
		$metadata_json = null;
		if ($metadata !== null) {
			try {
				$metadata_json = wp_json_encode($metadata);
			} catch (\Throwable $e) {
				Logging::log_error("[Retry Store] Failed to JSON encode metadata.", 'error', ['error' => $e->getMessage()]);
				throw new \RuntimeException("Failed to JSON encode metadata: " . $e->getMessage(), 0, $e);
			}
			if ($metadata_json === false) {
				throw new \RuntimeException("Failed to JSON encode metadata (returned false).");
			}
		}

		$state_metadata_json = null;
		if ($state_metadata !== null) {
			try {
				$state_metadata_json = wp_json_encode($state_metadata);
			} catch (\Throwable $e) {
				Logging::log_error("[Retry Store] Failed to JSON encode state_metadata.", 'error', ['error' => $e->getMessage()]);
				throw new \RuntimeException("Failed to JSON encode state_metadata: " . $e->getMessage(), 0, $e);
			}
			if ($state_metadata_json === false) {
				throw new \RuntimeException("Failed to JSON encode state_metadata (returned false).");
			}
		}


		$data = [
			'category'          => $category,
			'status'            => $status,
			'operation_type'    => $operation_type,
			'unique_key'        => $unique_key, // Will be NULL if not provided
			'priority'          => $priority,
			'job_group'         => $job_group, // NULL if not provided
			'depends_on_job_id' => $depends_on, // NULL if not provided or invalid
			'correlation_id'    => $correlation_id, // NULL if not provided
			'operation_data'    => $final_operation_data_str, // JSON string (potentially encrypted)
			'metadata'          => $metadata_json, // JSON string or NULL
			'retry_strategy'    => $strategy,
			'retry_count'       => 0, // Initial state
			'max_attempts'      => $max_attempts,
			'base_delay_sec'    => $base_delay,
			'backoff_factor'    => sprintf('%.2f', $backoff_factor), // Format for DECIMAL(5,2)
			'max_delay_sec'     => $max_delay, // NULL if not applicable
			'state_metadata'    => $state_metadata_json, // JSON string or NULL
			'lock_token'        => null, // Initial state
			'lock_expires_at'   => null, // Initial state
			'processor_id'      => null, // Initial state
			'last_error'        => null, // Initial state
			'last_error_code'   => null, // Initial state
			'last_attempt_at'   => null, // Initial state
			'next_attempt_at'   => $this->format_datetime_for_sql($next_attempt_time), // Calculated above (UTC)
			'scheduled_at'      => $this->format_datetime_for_sql($scheduled_at_for_db), // Original request (UTC) or NULL
			'expires_at'        => $this->format_datetime_for_sql($expires_at), // Expiry time (UTC) or NULL
			// 'created_at' is handled by DB default CURRENT_TIMESTAMP(6)
		];

		// Format array MUST EXACTLY match the order and types in the $data array above
		$format = [
			'%s', // category
			'%s', // status
			'%s', // operation_type
			($unique_key === null) ? '%s' : '%s', // unique_key (use %s for potential NULL)
			'%d', // priority
			($job_group === null) ? '%s' : '%s', // job_group (%s for NULL)
			($depends_on === null) ? '%s' : '%d', // depends_on_job_id (%s for NULL)
			($correlation_id === null) ? '%s' : '%s', // correlation_id (%s for NULL)
			'%s', // operation_data
			($metadata_json === null) ? '%s' : '%s', // metadata (%s for NULL)
			'%s', // retry_strategy
			'%d', // retry_count
			'%d', // max_attempts
			'%d', // base_delay_sec
			'%s', // backoff_factor (string format for decimal)
			($max_delay === null) ? '%s' : '%d', // max_delay_sec (%s for NULL)
			($state_metadata_json === null) ? '%s' : '%s', // state_metadata (%s for NULL)
			'%s', // lock_token (null)
			'%s', // lock_expires_at (null)
			'%s', // processor_id (null)
			'%s', // last_error (null)
			'%s', // last_error_code (null)
			'%s', // last_attempt_at (null)
			'%s', // next_attempt_at
			($scheduled_at_for_db === null) ? '%s' : '%s', // scheduled_at (%s for NULL)
			($expires_at === null) ? '%s' : '%s', // expires_at (%s for NULL)
		];

		// --- Sanity Check: Ensure data and format arrays align ---
		if (count($data) !== count($format)) {
			$msg = '[Retry Store] CRITICAL: Data and Format array count mismatch. Aborting insert.';
			Logging::log_error($msg, 'critical', ['data_count' => count($data), 'format_count' => count($format), 'data_keys' => array_keys($data)]);
			// Trigger error to halt execution in development/testing
			trigger_error($msg, E_USER_ERROR);
			return false; // Return false on mismatch
		}

		// --- 7. Insert into Database ---
		$inserted = $wpdb->insert($retry_table, $data, $format);

		if ($inserted === false) {
			// Log the error, avoid logging full sensitive data in production error logs
			Logging::log_error('[Retry Store] CRITICAL: Failed to insert job into database. Error: ' . $wpdb->last_error, 'critical', ['category' => $category, 'type' => $operation_type]);
			return false;
		}

		$job_id = $wpdb->insert_id;
		if ($job_id === 0) {
			// This might happen on some DB configs if insert succeeded but ID wasn't returned, rare with AUTO_INCREMENT
			// Try to retrieve based on unique key if possible? Very complex fallback.
			Logging::log_warning('[Retry Store] Job inserted but insert_id was 0. Check DB configuration (esp. AUTO_INCREMENT). Cannot reliably return ID.', 'warning', ['category' => $category, 'type' => $operation_type]);
			// We don't have the ID, so can't return it reliably. Treat as failure for consistency.
			return false;
		}

		/**
		 * Action hook fired after a retry job has been successfully stored in the database.
		 *
		 * @param int   $job_id The ID of the newly stored job.
		 * @param array $data   The data that was inserted (Note: sensitive data might be present if not encrypted).
		 * @param array $options The original options passed to store_retry_job.
		 * @ignore
		 */
		do_action('sha_retry_job_stored_lc', $job_id, $data, $options);

		Logging::log_notice(
			sprintf(
				'[Retry Store] Stored job ID %d (Category: %s, Type: %s, Priority: %d, Status: %s, Next: %s).',
				$job_id, $category, $operation_type, $priority, $status, $data['next_attempt_at'] ?? 'N/A'
			),
			['job_id' => $job_id, 'category' => $category, 'operation_type' => $operation_type]
		);

		// Ensure the processor is scheduled to run if it isn't already (idempotent)
		$this->schedule_retry_processor_event();
		// Invalidate cached stats as queue content has changed
		$this->clear_stats_cache();

		return $job_id;
		
		} finally {
			// Always release the lock
			if ($lock_acquired) {
				$this->lock->release($lock_key);
			}
		}
	}

	protected function get_active_job_count_for_group( string $group ): int {
		global $wpdb;
		$retry_table = $this->get_retry_table_name();
		
		// Normalize group value for DB query
		$group_value = ( $group === 'default' ? '' : $group );
		
		$sql = $wpdb->prepare(
			"SELECT COUNT(*) FROM `{$retry_table}` WHERE `status` = %s AND `job_group` = %s",
			self::STATUS_PROCESSING,
			$group_value
		);

		return (int) $wpdb->get_var( $sql );
	}

	/**
	 * Retrieves and locks a batch of ready jobs using DB atomic UPDATE + cache hints.
	 * Handles stalled job recovery and promotes waiting jobs before fetching.
	 *
	 * @param int    $batch_size   Maximum number of jobs to retrieve. Must be > 0.
	 * @param string $processor_id The ID of the current processor instance. Must be non-empty and trimmed.
	 * @return array<int, array<string, mixed>> An array of locked job data (associative arrays keyed by job ID), ready for processing.
	 *                                          Keys are job IDs (int), values are job data arrays (payload/metadata decoded).
	 *                                          Empty array if no jobs found, locking failed, data corruption, or other non-critical issues.
	 * @throws \InvalidArgumentException If $batch_size or $processor_id are invalid.
	 * @throws \RuntimeException If essential operations like secure random byte generation fail or critical DB errors occur that prevent safe continuation.
	 */
	protected function retrieve_and_lock_ready_jobs(int $batch_size, string $processor_id): array
	{
		global $wpdb;
		// Ensure WPDB is available
		if (null === $this->wpdb) {
			if (!isset($wpdb) || !$wpdb instanceof \wpdb) {
				throw new \RuntimeException('WPDB instance is not available or invalid for job locking.');
			}
			$this->wpdb = $wpdb; // Initialize $this->wpdb for this static context if needed
		}
		$db = $this->wpdb; // Use the initialized reference

		// --- Input Validation ---
		if ($batch_size <= 0) {
			throw new \InvalidArgumentException(sprintf(__('Batch size must be a positive integer. Received: %d', self::TEXT_DOMAIN), $batch_size));
		}
		$trimmed_processor_id = trim($processor_id);
		if (empty($trimmed_processor_id)) {
			throw new \InvalidArgumentException(__('Processor ID cannot be empty or whitespace.', self::TEXT_DOMAIN));
		}
		$processor_id = $trimmed_processor_id; // Use the validated & trimmed version consistently.

		// --- Configuration and Setup ---
		$lock_token = 'unknown'; // Initialize for logging in case of early failure
		try {
			$retry_table = $this->get_retry_table_name();
			$config = $this->get_retry_config();
			$now_utc = $this->get_current_utc_time();
			$now_sql = $this->format_datetime_for_sql($now_utc);
			$lock_timeout_seconds = max(30, (int) ($config['lock_timeout'] ?? 300));
			$use_cache_lock_hint = !empty($config['cache_enabled']) && function_exists('wp_using_ext_object_cache') && wp_using_ext_object_cache();
		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s] Initialization error: %s', $processor_id, $e->getMessage()));
			throw new \RuntimeException('Failed during initialization: ' . $e->getMessage(), 0, $e);
		}

		// --- 1. Reset Stalled Jobs ---
		try {
			$reset_error_details = 'Reset: Lock expired before ' . $now_sql;
			// Use prepare() correctly. Parameter Order: PENDING, %LIKE%, CONCAT%, PROCESSING, NOW
			$reset_sql = $db->prepare(
				"UPDATE `" . esc_sql($retry_table) . "`
                 SET `status` = %s,                     -- 1: Back to pending
                     `processor_id` = NULL,
                     `lock_token` = NULL,
                     `lock_expires_at` = NULL,
                     `last_error` = CASE
                                     WHEN `last_error` LIKE %s THEN `last_error` -- 2: Avoid duplicate reset messages
                                     ELSE CONCAT(LEFT(COALESCE(`last_error`, ''), 65000), %s) -- 3: Append reset reason
                                   END,
                     `retry_count` = `retry_count` + 1
                 WHERE `status` = %s                   -- 4: Was processing
                  AND `lock_expires_at` IS NOT NULL
                  AND `lock_expires_at` < %s",
    // Arguments for prepare:
    self::STATUS_PENDING,                      // Param 1: SET status = %s
    '%' . $db->esc_like($reset_error_details) . '%', // Param 2: LIKE %s (escaped value)
    "\n---\n" . $reset_error_details,          // Param 3: CONCAT(..., %s)
    self::STATUS_PROCESSING,                   // Param 4: WHERE status = %s
    $now_sql                                   // Param 5: WHERE lock_expires_at < %s
   ); // End of $db->prepare() call

			$reset_result = $db->query($reset_sql);

			if ($reset_result === false) {
				error_log(sprintf('[Retry Lock][%s] DB error resetting stalled jobs: %s. SQL used (approx): %s', $processor_id, $db->last_error, $reset_sql));
			} elseif ($reset_result > 0) {
				error_log(sprintf("[Retry Lock][%s] Reset %d stalled jobs whose locks expired before %s.", $processor_id, $reset_result, $now_sql));
				$this->clear_stats_cache();
			}
		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s] Exception during stalled job reset: %s', $processor_id, $e->getMessage()));
		}

		// --- 2. Promote Waiting Jobs (if applicable) ---
		try {
			$this->promote_waiting_jobs(); // Assumes this helper exists, is safe, and logs internally
		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s] Exception during promotion of waiting jobs: %s', $processor_id, $e->getMessage()));
		}

		// --- 3. Find Eligible Job IDs ---
		$potential_jobs_info = [];
		try {
			$find_limit = max($batch_size * 5, 20);
			$exclude_ids_sql_part = '';
			$exclude_ids_params = []; // Parameters for the exclusion clause

			if ($use_cache_lock_hint) {
				$recently_locked_cache_key = $this->get_recently_locked_cache_key();
				$cached_ids = wp_cache_get($recently_locked_cache_key, self::CACHE_GROUP);
				if (!empty($cached_ids) && is_array($cached_ids)) {
					$valid_ids = array_filter(array_map('intval', $cached_ids), fn($id) => $id > 0);
					if (!empty($valid_ids)) {
						$placeholders = implode(',', array_fill(0, count($valid_ids), '%d')); // Placeholders for prepare
						$exclude_ids_sql_part = " AND `id` NOT IN ({$placeholders})";
						$exclude_ids_params = $valid_ids; // Actual IDs for prepare() parameters array
					}
				}
			}

			// Build the parameter list IN ORDER for prepare().
			// Order: PENDING(1), SCHEDULED(2), NOW(sched)(3), NOW(next_att)(4), NOW(expires)(5), ...exclude_ids (if any), limit
			$find_sql_params = [
				self::STATUS_PENDING,       // Param 1: WHERE status = %s
				self::STATUS_SCHEDULED,     // Param 2: OR (status = %s ...)
				$now_sql,                   // Param 3: AND scheduled_at <= %s
				$now_sql,                   // Param 4: AND next_attempt_at <= %s
				$now_sql,                   // Param 5: AND (expires_at IS NULL OR expires_at > %s)
				// Params 6...N: Parameters for potential exclude clause (spread them here)
				...$exclude_ids_params,
				// Param N+1: Parameter for LIMIT
				$find_limit                 // LIMIT %d
			];

			// Build the SQL structure string by directly injecting the exclude clause
			// Use %s and %d for wpdb->prepare placeholders (no doubling needed)
			// Table name and exclude clause are injected via string concatenation
			$escaped_table = esc_sql($retry_table);
			$find_sql_structure = "SELECT `id`, `job_group`
                 FROM `{$escaped_table}`
                 WHERE (`status` = %s OR (`status` = %s AND `scheduled_at` <= %s)) -- Params 1, 2, 3
                   AND `next_attempt_at` <= %s          -- Param 4
                   AND (`expires_at` IS NULL OR `expires_at` > %s) -- Param 5
                   AND `retry_count` < `max_attempts`
                   {$exclude_ids_sql_part}              -- Optional exclusion clause (Params 6..N if IDs excluded)
                 ORDER BY `priority` ASC, `next_attempt_at` ASC, `id` ASC
                 LIMIT %d";

			$prepared_find_sql = $db->prepare($find_sql_structure, ...$find_sql_params);

			if (!$prepared_find_sql) {
				error_log(sprintf('[Retry Lock][%s] DB error preparing find eligible jobs SQL: %s', $processor_id, $db->last_error));
				return []; // Return empty on prepare error
			}

			$potential_jobs_info = $db->get_results($prepared_find_sql, ARRAY_A);

			if ($db->last_error) {
				error_log(sprintf('[Retry Lock][%s] DB error executing find eligible job IDs: %s', $processor_id, $db->last_error));
				return [];
			}
			if (empty($potential_jobs_info)) {
				// Optional: error_log(sprintf('[Retry Lock][%s] No eligible jobs found matching criteria.', $processor_id));
				return [];
			}
		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s] Exception finding eligible job IDs: %s', $processor_id, $e->getMessage()));
			return [];
		}

		// --- 4. Filter Potential IDs by Concurrency Limits ---
		$ids_to_attempt_lock = [];
		try {
			$concurrency_limits = apply_filters('sha_retry_concurrency_limits', ['default' => (int)($config['concurrency_limit'] ?? 1)]);
			$concurrency_limits = is_array($concurrency_limits) ? $concurrency_limits : ['default' => 1];
			$concurrency_limits['default'] = isset($concurrency_limits['default']) ? max(1, (int)$concurrency_limits['default']) : 1;

			$positive_limits = array_filter($concurrency_limits, fn($limit) => is_int($limit) && $limit > 0);
			$has_positive_limit = !empty($positive_limits);

			if ($has_positive_limit) {
				// error_log(sprintf("[Retry Lock][%s] Applying concurrency limits: %s", $processor_id, json_encode($concurrency_limits)));
				$active_counts_this_run = []; // Cache DB counts fetched within this run + increments for jobs added this run

				foreach ($potential_jobs_info as $job_info) {
					if (count($ids_to_attempt_lock) >= $batch_size) break;

					$job_id = (int) $job_info['id'];
					$group_raw = trim($job_info['job_group'] ?? '');
					$group_for_limit_check = $group_raw ?: 'default';
					$group_for_db_query = $group_raw ?: ''; // Assumes helper expects '' for default

					$limit_for_group = isset($concurrency_limits[$group_for_limit_check])
						? (int) $concurrency_limits[$group_for_limit_check]
						: $concurrency_limits['default'];

					if ($limit_for_group <= 0) { // Unlimited
						$ids_to_attempt_lock[] = $job_id;
						continue;
					}

					// Fetch active job count from DB only once per group *within this function execution*.
					if (!isset($active_counts_this_run[$group_for_limit_check])) {
						$current_db_active_count = max(0, $this->get_active_job_count_for_group($group_for_db_query));
						$active_counts_this_run[$group_for_limit_check] = $current_db_active_count;
						// error_log(sprintf("[Retry Lock][%s] Fetched active count for group '%s' (DB group '%s'): %d. Limit: %d.", $processor_id, $group_for_limit_check, $group_for_db_query, $current_db_active_count, $limit_for_group));
					}

					// Check if adding this job would exceed the limit, considering jobs already added *in this run*.
					if ($active_counts_this_run[$group_for_limit_check] < $limit_for_group) {
						$ids_to_attempt_lock[] = $job_id;
						// Increment the *local count* for this run immediately after adding.
						$active_counts_this_run[$group_for_limit_check]++;
						// error_log(sprintf("[Retry Lock][%s] Job %d (Group '%s') added. Current count for run now: %d. Limit: %d.", $processor_id, $job_id, $group_for_limit_check, $active_counts_this_run[$group_for_limit_check], $limit_for_group));
					} else {
						// error_log(sprintf("[Retry Lock][%s] Job %d (Group '%s') skipped due to concurrency limit (%d). Current run count: %d.", $processor_id, $job_id, $group_for_limit_check, $limit_for_group, $active_counts_this_run[$group_for_limit_check]));
					}
				}
				// error_log(sprintf("[Retry Lock][%s] Filtered potential jobs by concurrency. Attempting lock on %d IDs: [%s]", $processor_id, count($ids_to_attempt_lock), implode(',', $ids_to_attempt_lock)));
			} else {
				// No positive limits, just take the first $batch_size IDs found.
				$potential_ids = array_column($potential_jobs_info, 'id');
				$ids_to_attempt_lock = array_map('intval', array_slice($potential_ids, 0, $batch_size));
				// error_log(sprintf("[Retry Lock][%s] No positive concurrency limits found. Attempting lock on first %d potential jobs: [%s]", $processor_id, count($ids_to_attempt_lock), implode(',', $ids_to_attempt_lock)));
			}

			$ids_to_attempt_lock = array_filter($ids_to_attempt_lock, fn($id) => $id > 0);

			if (empty($ids_to_attempt_lock)) {
				// error_log(sprintf('[Retry Lock][%s] No jobs eligible to attempt lock after filtering (cache/concurrency).', $processor_id));
				return [];
			}
		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s] Exception during concurrency filtering: %s', $processor_id, $e->getMessage()));
			return [];
		}

		// --- 5. Attempt Atomic Lock using UPDATE ---
		$locked_count = 0;
		$successfully_locked_ids = []; // Initialize here
		try {
			$lock_token = bin2hex(random_bytes(16)); // Regenerate token for this specific attempt
		} catch (\Exception $e) {
			error_log(sprintf('[Retry Lock][%s] CRITICAL: Failed to generate secure lock token: %s', $processor_id, $e->getMessage()));
			throw new \RuntimeException('Failed to generate secure lock token for job locking.', 0, $e);
		}

		try {
			$lock_expires_at = $now_utc->modify("+{$lock_timeout_seconds} seconds");
			$lock_expires_at_sql = $this->format_datetime_for_sql($lock_expires_at);

			$ids_placeholders = implode(',', array_fill(0, count($ids_to_attempt_lock), '%d')); // For IN(id...)
			$status_placeholders = implode(',', array_fill(0, 2, '%s')); // For IN(status...)

			// Parameter Order for prepare() MUST match placeholders in the final SQL structure below:
			// SET values: PROCESSING(1), NOW(last_attempt)(2), LOCK_EXPIRES(3), PROC_ID(4), TOKEN(5)
			// WHERE values: ...IDs (for IN)(6..N), PENDING (for IN)(N+1), SCHEDULED (for IN)(N+2), NOW(next_attempt)(N+3)
			$lock_sql_params = [
				self::STATUS_PROCESSING,     // Param 1: SET `status` = %s
				$now_sql,                    // Param 2: SET `last_attempt_at` = %s
				$lock_expires_at_sql,        // Param 3: SET `lock_expires_at` = %s
				$processor_id,               // Param 4: SET `processor_id` = %s
				$lock_token,                 // Param 5: SET `lock_token` = %s
				// --- WHERE clause parameters start here ---
				...$ids_to_attempt_lock,      // Params 6..N: WHERE id IN (%d, %d...) - Spread the IDs
				self::STATUS_PENDING,        // Param N+1: WHERE status IN (%s, %s) - Param 1 for IN
				self::STATUS_SCHEDULED,      // Param N+2: WHERE status IN (%s, %s) - Param 2 for IN
				$now_sql,                    // Param N+3: WHERE next_attempt_at <= %s
			];

			// Build the SQL structure by directly injecting the dynamic placeholders
			// Table name, ID placeholders, and status placeholders are injected via string concatenation
			$escaped_table = esc_sql($retry_table);
			$lock_sql_structure = "UPDATE `{$escaped_table}`
                 SET `status` = %s,            -- Param 1
                     `last_attempt_at` = %s,   -- Param 2
                     `lock_expires_at` = %s,   -- Param 3
                     `processor_id` = %s,      -- Param 4
                     `lock_token` = %s         -- Param 5: Assign our unique token
                 WHERE `id` IN ({$ids_placeholders})             -- Target eligible IDs (Params 6..N)
                   AND `status` IN ({$status_placeholders})      -- Re-verify: Must still be ready (Params N+1, N+2)
                   AND `lock_token` IS NULL     -- *** ATOMIC LOCK CONDITION ***
                   AND `next_attempt_at` <= %s";

			$prepared_lock_sql = $db->prepare($lock_sql_structure, ...$lock_sql_params);

			if (!$prepared_lock_sql) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error preparing atomic lock SQL: %s', $processor_id, $lock_token, $db->last_error));
				return []; // Return empty on prepare error
			}

			$locked_count_result = $db->query($prepared_lock_sql);

			if ($locked_count_result === false) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error executing atomic lock UPDATE: %s', $processor_id, $lock_token, $db->last_error));
				return []; // Return empty on DB error during locking
			}

			$locked_count = (int) $locked_count_result;

			if ($locked_count === 0) {
				// This is expected under contention or if job states changed. Not an error.
				// error_log(sprintf("[Retry Lock][%s][T:%s] Atomic lock UPDATE affected 0 rows (contention/conditions changed). Attempted IDs: [%s]", $processor_id, $lock_token, implode(',', $ids_to_attempt_lock)));
				return [];
			}

			// Log the raw count from UPDATE before confirmation
			// error_log(sprintf("[Retry Lock][%s][T:%s] Atomic lock UPDATE reported %d rows affected. Attempting confirmation.", $processor_id, $lock_token, $locked_count));


			// --- 6. Confirm Which IDs Were *Actually* Locked by Us ---
			// Use the unique lock_token AND processor_id for confirmation.
			$confirm_lock_sql = $db->prepare(
				"SELECT `id` FROM `" . esc_sql($retry_table) . "` WHERE `processor_id` = %s AND `lock_token` = %s AND `status` = %s ORDER BY `id` ASC",
				$processor_id,           // Param 1
				$lock_token,             // Param 2
				self::STATUS_PROCESSING // Param 3
			);

			if (!$confirm_lock_sql) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error preparing lock confirmation SELECT: %s', $processor_id, $lock_token, $db->last_error));
				return []; // State is uncertain.
			}

			$confirmed_ids_result = $db->get_col($confirm_lock_sql);

			if ($db->last_error) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error executing lock confirmation SELECT: %s', $processor_id, $lock_token, $db->last_error));
				return []; // Safest to assume failure.
			}

			$successfully_locked_ids = array_map('intval', $confirmed_ids_result ?: []);
			$confirmed_count = count($successfully_locked_ids);

			// --- Lock Confirmation Mismatch Handling ---
			if ($confirmed_count === 0 && $locked_count > 0) {
				error_log(sprintf("[Retry Lock][%s][T:%s] CRITICAL Lock Mismatch: UPDATE reported %d locked, but confirmation SELECT found 0. Aborting.", $processor_id, $lock_token, $locked_count));
				return [];
			} elseif ($confirmed_count < $locked_count) {
				error_log(sprintf("[Retry Lock][%s][T:%s] WARNING: Lock Confirmation Count Mismatch: UPDATE reported %d locked, SELECT confirmed only %d. Proceeding ONLY with confirmed IDs [%s].", $processor_id, $lock_token, $locked_count, $confirmed_count, implode(',', $successfully_locked_ids)));
				// Proceed ONLY with the confirmed IDs ($successfully_locked_ids) - this is already the case.
			} elseif ($confirmed_count > $locked_count) {
				error_log(sprintf("[Retry Lock][%s][T:%s] CRITICAL Lock Anomaly: Confirmation SELECT found %d jobs, but UPDATE reported only %d. Aborting due to inconsistent state. Confirmed IDs: [%s].", $processor_id, $lock_token, $confirmed_count, $locked_count, implode(',', $successfully_locked_ids)));
				return [];
			}

			if (empty($successfully_locked_ids)) {
				// error_log(sprintf("[Retry Lock][%s][T:%s] No jobs confirmed locked via SELECT.", $processor_id, $lock_token));
				return [];
			}

			// Log success after confirmation
			// error_log(sprintf("[Retry Lock][%s][T:%s] Successfully confirmed lock via SELECT on %d jobs. IDs: [%s].", $processor_id, $lock_token, $confirmed_count, implode(',', $successfully_locked_ids)));

		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s][T:%s] Exception during lock attempt/confirmation: %s', $processor_id, $lock_token, $e->getMessage()));
			return [];
		}

		// --- 7. Update Cache Hints (Optional) ---
		if ($use_cache_lock_hint && !empty($successfully_locked_ids)) {
			try {
				$cache_key_recent = $this->get_recently_locked_cache_key();
				$current_recent = wp_cache_get($cache_key_recent, self::CACHE_GROUP) ?: [];
				$current_recent = is_array($current_recent) ? $current_recent : [];
				$merged_ids = array_filter(array_map('intval', array_unique(array_merge($current_recent, $successfully_locked_ids))), fn($id) => $id > 0);
				$max_cached_recent_ids = max(100, $batch_size * 5);
				$updated_recent = array_slice($merged_ids, -$max_cached_recent_ids);
				wp_cache_set($cache_key_recent, $updated_recent, self::CACHE_GROUP, 15); // Short TTL hint

				$cache_ttl_job = $lock_timeout_seconds + 15; // Buffer
				$expiry_timestamp = $lock_expires_at->getTimestamp();
				foreach ($successfully_locked_ids as $locked_id) {
					$cache_key_job = $this->get_job_lock_cache_key($locked_id);
					$cache_data_job = ['token' => $lock_token, 'proc' => $processor_id, 'expires' => $expiry_timestamp];
					wp_cache_set($cache_key_job, $cache_data_job, self::CACHE_GROUP, $cache_ttl_job);
				}
				// error_log(sprintf("[Retry Lock][%s][T:%s] Updated cache hints for %d locked jobs.", $processor_id, $lock_token, count($successfully_locked_ids)));
			} catch (\Throwable $e) {
				error_log(sprintf('[Retry Lock][%s][T:%s] Non-critical error updating cache hints: %s', $processor_id, $lock_token, $e->getMessage()));
			}
		}

		// --- 8. Retrieve Full Data for Successfully Locked Jobs ---
		$jobs_raw = [];
		// Use the list confirmed by SELECT ($successfully_locked_ids) as the source of truth for fetching.
		$final_locked_ids = $successfully_locked_ids;
		$baseline_confirmed_count = count($final_locked_ids); // Count *before* fetching data

		if (empty($final_locked_ids)) {
			// Should have been caught earlier, but defensive check.
			// error_log(sprintf("[Retry Lock][%s][T:%s] Logic Error Check: No confirmed IDs remain before fetching data. Lock Count: %d.", $processor_id, $lock_token, $baseline_confirmed_count));
			return [];
		}

		try {
			$fetch_ids_placeholders = implode(',', array_fill(0, $baseline_confirmed_count, '%d'));

			// Parameter order MUST match placeholders: ...IDs first(1..N), then processor_id(N+1), then lock_token(N+2).
			$fetch_sql_params = [
				...$final_locked_ids, // Params 1..N: Spread the confirmed IDs for IN clause
				$processor_id,       // Param N+1: For WHERE processor_id = %s
				$lock_token          // Param N+2: For WHERE lock_token = %s
			];

			// Build the fetch SQL structure by directly injecting the ID placeholders
			// Table name and ID placeholders are injected via string concatenation
			$escaped_table = esc_sql($retry_table);
			// Optimized: Select only needed columns instead of SELECT *
			$fetch_sql_structure = "SELECT id, processor_id, status, retry_count, max_retries, next_retry_at, last_error, payload, created_at, updated_at, expires_at FROM `{$escaped_table}`
                 WHERE `id` IN ({$fetch_ids_placeholders})  -- Params 1..N
                   AND `processor_id` = %s      -- Param N+1
                   AND `lock_token` = %s        -- Param N+2
                 ORDER BY `priority` ASC, `next_attempt_at` ASC, `id` ASC";

			$prepared_fetch_sql = $db->prepare($fetch_sql_structure, ...$fetch_sql_params);

			if (!$prepared_fetch_sql) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error preparing final data fetch SQL: %s', $processor_id, $lock_token, $db->last_error));
				return []; // Can't fetch data, jobs will be reset later.
			}

			$jobs_raw = $db->get_results($prepared_fetch_sql, ARRAY_A);

			if ($db->last_error) {
				error_log(sprintf('[Retry Lock][%s][T:%s] CRITICAL DB error executing fetch for %d confirmed locked jobs: %s. IDs: [%s]', $processor_id, $lock_token, $baseline_confirmed_count, $db->last_error, implode(',', $final_locked_ids)));
				return []; // Failed to get data for locked jobs.
			}

			// --- Final Verification and Mismatch Handling (Fetch vs Confirmed IDs) ---
			$fetched_count = count($jobs_raw);

			if ($fetched_count === 0 && $baseline_confirmed_count > 0) {
				error_log(sprintf("[Retry Lock][%s][T:%s] CRITICAL Fetch Mismatch: Failed to fetch details for %d confirmed jobs [%s]. Aborting.", $processor_id, $lock_token, $baseline_confirmed_count, implode(',', $final_locked_ids)));
				return []; // Jobs are locked but inaccessible.
			}

			if ($fetched_count !== $baseline_confirmed_count) {
				$fetched_ids = array_map('intval', array_column($jobs_raw, 'id'));
				$expected_ids_list = implode(',', $final_locked_ids);
				$fetched_ids_list = implode(',', $fetched_ids);
				$missing_ids = array_diff($final_locked_ids, $fetched_ids);
				$unexpected_ids = array_diff($fetched_ids, $final_locked_ids);

				error_log(sprintf("[Retry Lock][%s][T:%s] WARNING: Fetch Count Mismatch: Expected %d confirmed IDs [%s], but fetched %d rows [%s]. Missing: [%s]. Unexpected: [%s]. Proceeding ONLY with fetched data.", $processor_id, $lock_token, $baseline_confirmed_count, $expected_ids_list, $fetched_count, $fetched_ids_list, implode(',', $missing_ids), implode(',', $unexpected_ids)));

				// *** CRITICAL Adjustment: Trust only the data we actually received ***
				$final_locked_ids = $fetched_ids; // Update the list of IDs we will process

				if (empty($final_locked_ids)) {
					error_log(sprintf("[Retry Lock][%s][T:%s] Fetch mismatch resulted in empty job list. Aborting fetch.", $processor_id, $lock_token));
					return [];
				}
				// error_log(sprintf("[Retry Lock][%s][T:%s] Proceeding with %d jobs after handling fetch mismatch. Final IDs: [%s].", $processor_id, $lock_token, count($final_locked_ids), implode(',', $final_locked_ids)));
			}
			// If counts matched, $final_locked_ids remains unchanged.

		} catch (\Throwable $e) {
			error_log(sprintf('[Retry Lock][%s][T:%s] Exception during final data fetch or verification: %s', $processor_id, $lock_token, $e->getMessage()));
			return []; // Abort on fetch/verification errors.
		}

		// If we reach here and $final_locked_ids is empty (e.g., due to mismatch handling), return empty.
		if (empty($final_locked_ids)) {
			// This state should have been logged previously.
			// error_log(sprintf("[Retry Lock][%s][T:%s] No jobs available after fetch and final verification steps.", $processor_id, $lock_token));
			return [];
		}

		// --- 9. Decrypt (via hook) & Decode JSON Payloads ---
		$valid_jobs = [];
		$final_locked_ids_set = array_flip($final_locked_ids); // Flipped array for efficient check
		$baseline_decode_count = count($final_locked_ids); // How many jobs entering the decode loop
		// error_log(sprintf("[Retry Lock][%s][T:%s] Starting decode loop for %d fetched jobs. IDs: [%s].", $processor_id, $lock_token, $baseline_decode_count, implode(',', $final_locked_ids)));

		foreach ($jobs_raw as $job_raw) {
			$job_id = (int) $job_raw['id'];

			// Ensure this job ID is one we decided to process after fetch verification.
			if (!isset($final_locked_ids_set[$job_id])) {
				error_log(sprintf("[Retry Lock][%s][T:%s] Internal Logic WARNING: Job ID %d was fetched but is not in the final verified lock list (started decode with %d). Skipping decode. Possible earlier fetch mismatch.", $processor_id, $lock_token, $job_id, $baseline_decode_count));
				continue;
			}

			$job = $job_raw; // Start with raw data
			$is_valid = true;
			$decode_error_reason = '';

			try {
				// Payload Decryption & Decoding
				$payload_from_db = $job['operation_data'] ?? null;
				$decoded_payload = [];
				if (is_string($payload_from_db) && $payload_from_db !== '') {
					// Assumes filter exists, handles errors, returns string/null, or throws.
					$decrypted_payload_str = apply_filters('sha_retry_decrypt_payload', $payload_from_db, $job_raw);

					if (is_wp_error($decrypted_payload_str)) {
						throw new \RuntimeException("Payload decryption filter failed: " . $decrypted_payload_str->get_error_message());
					}
					if (!is_string($decrypted_payload_str) && $decrypted_payload_str !== null) {
						throw new \RuntimeException("Decrypt filter returned unexpected non-string/non-null type: " . gettype($decrypted_payload_str));
					}

					if (is_string($decrypted_payload_str) && $decrypted_payload_str !== '') {
						$decoded_payload = json_decode($decrypted_payload_str, true, 512, JSON_THROW_ON_ERROR);
						if (!is_array($decoded_payload)) {
							throw new \JsonException("Decoded operation_data is not a JSON object or array.");
						}
					}
				}
				$job['operation_data'] = $decoded_payload;

				// Metadata Decoding
				$metadata_from_db = $job['metadata'] ?? null;
				$decoded_metadata = [];
				if (is_string($metadata_from_db) && $metadata_from_db !== '') {
					$decoded_metadata = json_decode($metadata_from_db, true, 512, JSON_THROW_ON_ERROR);
					if (!is_array($decoded_metadata)) {
						throw new \JsonException("Decoded metadata is not a JSON object or array.");
					}
				}
				$job['metadata'] = $decoded_metadata;

				// State Metadata Decoding
				$state_from_db = $job['state_metadata'] ?? null;
				$decoded_state = [];
				if (is_string($state_from_db) && $state_from_db !== '') {
					$decoded_state = json_decode($state_from_db, true, 512, JSON_THROW_ON_ERROR);
					if (!is_array($decoded_state)) {
						throw new \JsonException("Decoded state_metadata is not a JSON object or array.");
					}
				}
				$job['state_metadata'] = $decoded_state;

			} catch (\JsonException $e) {
				$error_message = sprintf("JSON decode error for job ID %d: %s", $job_id, $e->getMessage());
				error_log(sprintf("[Retry Lock][%s][T:%s] Data Corruption: %s", $processor_id, $lock_token, $error_message));
				$decode_error_reason = "Data integrity error (JSON decode): " . substr($e->getMessage(), 0, 250);
				$is_valid = false;
			} catch (\Throwable $e) { // Catch broader errors (RuntimeException, Error, Exception etc.)
				$error_type = get_class($e);
				$error_message = sprintf("Decrypt/Decode processing error for job ID %d (%s): %s", $job_id, $error_type, $e->getMessage());
				error_log(sprintf("[Retry Lock][%s][T:%s] Data Processing Error: %s", $processor_id, $lock_token, $error_message));
				$decode_error_reason = "Data processing error ({$error_type}): " . substr($e->getMessage(), 0, 250);
				$is_valid = false;
			}

			// --- Post-processing action ---
			if ($is_valid) {
				$valid_jobs[(int)$job_id] = $job; // Add to results, keyed by integer Job ID
			} else {
				// Attempt to move to DLQ
				try {
					$this->move_to_dlq($job_raw, $decode_error_reason, self::DLQ_REASON_DATA_INTEGRITY, $lock_token);
				} catch (\Throwable $dlqError) {
					error_log(sprintf("[Retry Lock][%s][T:%s] CRITICAL: Failed to move job %d to DLQ after decode error: %s. Decode reason: %s", $processor_id, $lock_token, $job_id, $dlqError->getMessage(), $decode_error_reason));
				}
			}
		} // End foreach job loop

		// --- Final Logging ---
		$final_returned_count = count($valid_jobs);
		$moved_to_dlq_count = $baseline_decode_count - $final_returned_count; // Jobs started decode - jobs returned = jobs moved to DLQ

		if ($final_returned_count > 0) {
			$returned_ids = implode(',', array_keys($valid_jobs));
			if ($moved_to_dlq_count > 0) {
				error_log(sprintf("[Retry Lock][%s][T:%s] Completed: Returning %d valid jobs [%s]. Started decode with %d, moved %d to DLQ.", $processor_id, $lock_token, $final_returned_count, $returned_ids, $baseline_decode_count, $moved_to_dlq_count));
			} else {
				error_log(sprintf("[Retry Lock][%s][T:%s] Completed: Returning %d successfully locked and processed jobs. IDs: [%s].", $processor_id, $lock_token, $final_returned_count, $returned_ids));
			}
		} else {
			if ($baseline_decode_count > 0) { // We started decoding but none were valid
				error_log(sprintf("[Retry Lock][%s][T:%s] Completed: No valid jobs to return. Started decode with %d, all failed processing or moved to DLQ.", $processor_id, $lock_token, $baseline_decode_count));
			} else { // No jobs even made it to the decode loop
				error_log(sprintf("[Retry Lock][%s][T:%s] Completed: No jobs were available or passed verification steps before decode.", $processor_id, $lock_token));
			}
		}

		// Return the array of valid, locked, and processed jobs, keyed by their integer ID.
		return $valid_jobs;
	}
	// =========================================================================
	// Job Execution & Lifecycle
	// =========================================================================

    protected function execute_retry_operation(array $job): bool {
        $job_id = (int) $job['id'];
        $proc_id = $this->get_processor_id();
        $lock_token = $job['lock_token'] ?? null; // Should always be set if locked correctly

        // Critical check: Must have a lock token to proceed safely.
        if (empty($lock_token)) {
            Logging::log_error("[Retry] Attempting to execute job ID {$job_id} without a lock token. Aborting.", 'critical', ['job_id' => $job_id, 'job_type' => $job['operation_type'] ?? 'unknown']);
            // Cannot safely modify job state without a lock. Avoid rescheduling or DLQ move.
            // This indicates a potential issue in the locking mechanism upstream.
            return false; // Indicate failure to execute this attempt.
        }

        $start_time = microtime(true);
        $original_job = $job; // Keep a copy for logging/hooks if $job is modified by filters
        $execution_result = null; // Result returned by the executor on success
        $next_state = $job['state_metadata'] ?? []; // Start with current state, executor modifies this by reference
        $error = null; // Stores Throwable if execution fails

        Logging::log_debug("[Retry] Starting execution attempt for job ID {$job_id}", 'debug', ['job_id' => $job_id, 'type' => $job['operation_type'], 'attempt' => (int)$job['retry_count'] + 1]);

        // --- Pre-Execution Filter ---
        $job_ref = $job; // Create a variable to pass by reference
        try {
            $pre_exec_result = apply_filters_ref_array('sha_retry_before_job_execution_lc', [&$job_ref]);
        } catch (\Throwable $filter_error) {
            // Treat errors in the filter itself as a cancellation/failure
            $pre_exec_result = new \WP_Error('FILTER_EXCEPTION', "Exception in sha_retry_before_job_execution_lc filter: " . $filter_error->getMessage(), ['exception' => $filter_error]);
            Logging::log_error("[Retry] Exception in pre-execution filter for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $filter_error->getMessage()]);
        }
        $job = $job_ref; // Update job with any modifications from filter

        if ($pre_exec_result === false || is_wp_error($pre_exec_result)) {
            $reason = is_wp_error($pre_exec_result) ? $pre_exec_result->get_error_message() : 'Cancelled by sha_retry_before_job_execution_lc filter.';
            $error_code = is_wp_error($pre_exec_result) ? $pre_exec_result->get_error_code() ?: 'FILTER_CANCEL' : 'FILTER_CANCEL';
            Logging::log_info("[Retry] Job ID {$job_id} execution cancelled by pre-execution filter: {$reason}", 'info', ['job_id' => $job_id, 'error_code' => $error_code]);
            // Treat cancellation like a permanent failure for this job instance
            $this->log_history($original_job, 'cancelled', 0, $reason, $error_code, $proc_id, null, $next_state); // Log cancellation
            $this->move_to_dlq($job_id, "Cancelled before execution: {$reason}", self::DLQ_REASON_CANCELLED, $lock_token, $original_job);
            $this->promote_dependent_jobs($job_id, false); // Dependencies fail if cancelled
            return false; // Job did not complete successfully
        }

        // --- Get Executor ---
        $executor = $this->get_executor($job['operation_type']);
        if ($executor === null) {
            $error_message = "No valid executor registered for operation type '{$job['operation_type']}'.";
            Logging::log_error("[Retry] {$error_message} Job ID {$job_id}", 'error', ['job_id' => $job_id, 'operation_type' => $job['operation_type']]);
            $this->log_history($original_job, 'failure', 0, $error_message, 'NO_EXECUTOR', $proc_id, null, $next_state);
            $this->move_to_dlq($job_id, $error_message, self::DLQ_REASON_CONFIGURATION, $lock_token, $original_job); // Non-retryable config error
            $this->promote_dependent_jobs($job_id, false); // Dependencies fail
            return false; // Job did not complete successfully
        }

        // --- Rate Limiting / Circuit Breaker Hook ---
        try {
            $allow_execution = apply_filters('sha_retry_allow_execution_lc', true, $job);
        } catch (\Throwable $filter_error) {
            // Treat errors in the filter itself as a denial/failure
            $allow_execution = new \WP_Error('FILTER_EXCEPTION', "Exception in sha_retry_allow_execution_lc filter: " . $filter_error->getMessage(), ['exception' => $filter_error]);
            Logging::log_error("[Retry] Exception in allow execution filter for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $filter_error->getMessage()]);
        }

        if ($allow_execution !== true) { // Checks for false or WP_Error
            $denial_reason = 'Execution denied by sha_retry_allow_execution_lc hook.';
            $denial_code = 'HOOK_DENY';
            $log_status = 'denied'; // More specific than 'cancelled'
            $dlq_reason_on_fail = self::DLQ_REASON_DENIED; // Specific DLQ reason if reschedule fails

            if (is_wp_error($allow_execution)) {
                $denial_reason = $allow_execution->get_error_message();
                $denial_code = $allow_execution->get_error_code() ?: $denial_code;
            }

            try {
                $classification = apply_filters('sha_retry_classify_denial_lc', [
                    'dlq_reason' => $dlq_reason_on_fail, // Used if reschedule fails
                    'history_status' => $log_status,
                ], $job, $allow_execution);
            } catch (\Throwable $filter_error) {
                Logging::log_error("[Retry] Exception in denial classification filter for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $filter_error->getMessage()]);
                // Use defaults if classification filter fails
                $classification = ['dlq_reason' => $dlq_reason_on_fail, 'history_status' => $log_status];
            }

            $log_status = $classification['history_status'] ?? $log_status;
            $dlq_reason_on_fail = $classification['dlq_reason'] ?? $dlq_reason_on_fail; // Allow override

            Logging::log_info("[Retry] Job ID {$job_id} execution denied: {$denial_reason}. Status: {$log_status}", 'info', ['job_id' => $job_id, 'denial_code' => $denial_code]);

            // Log the denial attempt in history
            $this->log_history($original_job, $log_status, 0, $denial_reason, $denial_code, $proc_id, null, $next_state);

            // Reschedule, assuming denial is temporary (rate limit, circuit breaker)
            // Force reschedule even if lock might seem lost, as we still hold it conceptually here.
            $rescheduled = $this->reschedule_failed_job($job, $denial_reason, $denial_code, $lock_token, $next_state, true /* force reschedule */);

            if (!$rescheduled) {
                // Reschedule failed (DB error, lock really was lost?). Move to DLQ as fallback.
                Logging::log_error("[Retry] Failed to reschedule job ID {$job_id} after execution denial. Moving to DLQ.", 'error', ['job_id' => $job_id, 'reason' => $denial_reason]);
                $this->move_to_dlq($job_id, "Execution denied ({$log_status}) and reschedule failed: {$denial_reason}", $dlq_reason_on_fail, $lock_token, $original_job);
                $this->promote_dependent_jobs($job_id, false); // Dependencies fail
            }
            // Even if rescheduled, return false as it didn't complete successfully *this attempt*.
            return false;
        }

        // --- Execute the Job ---
        $attempt_successful = false; // Track success within the try block
        try {
            // Define the heartbeat function
            $heartbeat_func = function() use ($job_id, $lock_token): bool {
                // Add a try-catch here to prevent heartbeat errors from crashing the job
                try {
                    return $this->update_heartbeat($job_id, $lock_token);
                } catch (\Throwable $hb_error) {
                    Logging::log_warning("[Retry] Heartbeat failed for job ID {$job_id}. Error: " . $hb_error->getMessage(), 'warning', ['job_id' => $job_id]);
                    return false; // Indicate heartbeat failure, though job continues
                }
            };

            // Call the registered executor function
            // Pass original job for context, but executor modifies $next_state
            $execution_result = call_user_func_array($executor, [
                $job['operation_data'], // Current operation data
                $job['metadata'],       // Current metadata
                &$next_state,           // State to be modified
                $heartbeat_func,        // Heartbeat callback
                $original_job           // Provide original job for context if needed
            ]);

            $attempt_successful = true; // No exception thrown

            try {
                do_action('sha_retry_report_success_lc', $original_job, $execution_result);
            } catch (\Throwable $hook_error) {
                Logging::log_error("[Retry] Exception in success reporting hook for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $hook_error->getMessage()]);
                // Continue processing even if hook fails
            }

        } catch (\Throwable $e) {
            // Execution failed, store the error
            $error = $e;
            $attempt_successful = false;
            // Use max length that fits reasonably in most DB text fields, slightly less than 64k
            $err_msg = mb_substr($e->getMessage(), 0, 65500);
            $err_code = (string) $e->getCode();

            Logging::log_warning(
                sprintf("[Retry] Job ID %d execution failed (Attempt %d). Error: %s (%s)",
                    $job_id, (int)$job['retry_count'] + 1, $err_msg, $err_code
                ),
                'warning',
                ['job_id' => $job_id, 'error_code' => $err_code, 'error_class' => get_class($e)]
            // Avoid logging full trace here, it goes into history
            );

            try {
                do_action('sha_retry_report_failure_lc', $original_job, $error);
            } catch (\Throwable $hook_error) {
                Logging::log_error("[Retry] Exception in failure reporting hook for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $hook_error->getMessage()]);
                // Continue processing even if hook fails
            }
        }

        // --- Post-Execution: Finalize State, Log History, Handle Outcome ---
        $duration_ms = (int) round((microtime(true) - $start_time) * 1000);
        $attempt_number = (int) $job['retry_count'] + 1;
        $max_attempts = (int) $job['max_attempts'];

        $history_log_status = $attempt_successful ? 'success' : 'failure';
        $final_error_message = $error ? mb_substr($error->getMessage(), 0, 65500) : null;
        $final_error_code = $error ? (string) $error->getCode() : '';
        // Limit stack trace size for history log to prevent excessive storage use
        $stack_trace = $error ? mb_substr($error->getTraceAsString(), 0, 10000) : null;

        try {
            do_action('sha_retry_after_job_execution_lc', $job_id, $original_job, $attempt_successful, $error, $duration_ms / 1000.0, $next_state);
        } catch (\Throwable $hook_error) {
            Logging::log_error("[Retry] Exception in post-execution hook for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $hook_error->getMessage()]);
            // Continue processing even if hook fails
        }

        // Log the result of this attempt to the history table
        $this->log_history($original_job, $history_log_status, $duration_ms, $final_error_message, $final_error_code, $proc_id, $stack_trace, $next_state);

        // --- Handle Outcome ---
        $final_job_completed = false; // Overall completion status for return value

        if ($attempt_successful) {
            // --- SUCCESS ---
            Logging::log_info(
                sprintf('[Retry] Job ID %d completed successfully (Attempt %d/%d, Duration: %dms). Removing from queue.',
                    $job_id, $attempt_number, max(1, $max_attempts), $duration_ms // Ensure max_attempts is at least 1 for display
                ),
                'info',
                ['job_id' => $job_id, 'duration_ms' => $duration_ms]
            );

            // Remove the job from the main retry table
            $removed = $this->remove_retry_operation($job_id, $lock_token);
            if (!$removed) {
                // This is problematic - job succeeded but couldn't be removed. Lock might have expired or DB error.
                Logging::log_error("[Retry] CRITICAL: Job ID {$job_id} succeeded but failed to remove from queue (lock lost or DB error?). Potential for duplicate execution.", 'critical', ['job_id' => $job_id]);
                // Don't promote dependents if removal failed, as the job might still run again.
                // The job state is inconsistent. Return false because the *overall* operation (execute AND remove) failed.
                $final_job_completed = false;
            } else {
                try {
                    do_action('sha_retry_job_success_lc', $job_id, $original_job, $execution_result);
                } catch (\Throwable $hook_error) {
                    Logging::log_error("[Retry] Exception in final job success hook for job ID {$job_id}.", 'error', ['job_id' => $job_id, 'error' => $hook_error->getMessage()]);
                    // Continue cleanup even if hook fails
                }

                // Promote jobs waiting on this one *only if* successfully removed
                $this->promote_dependent_jobs($job_id, true); // True indicates success
                $this->clear_stats_cache(); // Stats changed
                $final_job_completed = true; // Mark as completed successfully
            }

        } else {
            // --- FAILURE ---
            $final_job_completed = false; // Failed this attempt

            // Decide whether to reschedule or move to DLQ
            $should_retry_this_error = $error && $this->should_retry_exception($error); // Is the specific error retryable?
            $attempts_remain = $max_attempts <= 0 || $attempt_number < $max_attempts; // Max attempts 0 or less means infinite retries
            $strategy_allows_retry = ($job['retry_strategy'] ?? self::STRATEGY_DEFAULT) !== self::STRATEGY_NONE;
            $is_poison_pill = $this->check_poison_pill($job_id, $final_error_code); // Check for repeated identical failures

            $permanent_failure_reason = null;
            $dlq_reason_code = self::DLQ_REASON_FAILED; // Default DLQ reason if moved

            if ($is_poison_pill) {
                $permanent_failure_reason = "Poison pill detected after {$attempt_number} attempts with error code '{$final_error_code}'.";
                $dlq_reason_code = self::DLQ_REASON_POISON;
                Logging::log_error("[Retry] Job ID {$job_id} identified as poison pill. Moving to DLQ.", 'error', ['job_id' => $job_id, 'error_code' => $final_error_code]);

            } elseif (!$attempts_remain && $max_attempts > 0) { // Only check max attempts if it's > 0
                $permanent_failure_reason = "Maximum attempts ({$max_attempts}) reached.";
                $dlq_reason_code = self::DLQ_REASON_MAX_ATTEMPTS;

            } elseif (!$strategy_allows_retry) {
                $permanent_failure_reason = "Retry strategy is 'none'.";
                $dlq_reason_code = self::DLQ_REASON_FAILED; // Keep default

            } elseif (!$should_retry_this_error) {
                $permanent_failure_reason = "Non-retryable error encountered: " . get_class($error);
                $dlq_reason_code = self::DLQ_REASON_NON_RETRYABLE; // More specific reason
            }

            // --- Handle Reschedule vs. DLQ ---
            if ($permanent_failure_reason === null) {
                // Conditions met for rescheduling
                Logging::log_info(
                    sprintf('[Retry] Job ID %d failed (Attempt %d/%s), rescheduling. Error: %s',
                        $job_id, $attempt_number, ($max_attempts <= 0 ? 'unlimited' : $max_attempts), $final_error_message
                    ),
                    'info',
                    ['job_id' => $job_id, 'error' => $final_error_message]
                );
                // Pass the updated state ($next_state) to be saved for the next attempt
                $rescheduled = $this->reschedule_failed_job($job, $final_error_message, $final_error_code, $lock_token, $next_state);
                if (!$rescheduled) {
                    // Reschedule failed (DB error, lock lost?). Move to DLQ as fallback.
                    $dlq_message = "Failed to reschedule after attempt {$attempt_number}. Last Error: " . $final_error_message;
                    Logging::log_error("[Retry] Failed to reschedule job ID {$job_id} after failure. Moving to DLQ.", 'error', ['job_id' => $job_id, 'error' => $final_error_message]);
                    $this->move_to_dlq($job_id, $dlq_message, self::DLQ_REASON_FAILED, $lock_token, $original_job);
                    $this->promote_dependent_jobs($job_id, false); // Dependencies fail
                }
                // Reschedule successful, but the job didn't complete this attempt. Still return false.

            } else {
                // Permanent failure condition met. Move to DLQ.
                $error_class_name = $error ? get_class($error) : 'N/A';
                $dlq_message = sprintf('%s Last Error (%s): %s', $permanent_failure_reason, $error_class_name, $final_error_message);
                // Ensure DLQ message doesn't exceed reasonable limits
                $dlq_message = mb_substr($dlq_message, 0, 1000);

                Logging::log_error("[Retry] Job ID {$job_id} failed permanently. {$permanent_failure_reason} Moving to DLQ.", 'error', ['job_id' => $job_id, 'error' => $final_error_message, 'dlq_reason_code' => $dlq_reason_code]);
                $this->move_to_dlq($job_id, $dlq_message, $dlq_reason_code, $lock_token, $original_job);
                $this->promote_dependent_jobs($job_id, false); // Dependencies fail
            }
        }

        return $final_job_completed; // Return true ONLY if the job executed AND was removed successfully
    }
	/**
	 * Updates the lock expiration time (heartbeat) for a job currently being processed.
	 * Also updates the cache entry for the lock.
	 *
	 * @param int    $job_id     The ID of the job being processed.
	 * @param string $lock_token The lock token held by the current processor.
	 * @return bool True if the heartbeat update was successful, false otherwise (e.g., lock lost, DB error).
	 */
	protected function update_heartbeat(int $job_id, string $lock_token): bool {
		global $wpdb;
		$table = $this->get_retry_table_name();
		$config = $this->get_retry_config();
		$now = $this->get_current_utc_time();
		$lock_timeout_seconds = max(30, (int) $config['lock_timeout']);
		$new_expires_dt = $now->modify("+{$lock_timeout_seconds} seconds");
		$new_expires_sql = $this->format_datetime_for_sql($new_expires_dt);

		$updated = $wpdb->update(
			$table,
			['lock_expires_at' => $new_expires_sql], // Data to update
			[                                         // WHERE clause
				'id' => $job_id,
				'lock_token' => $lock_token,
				'status' => self::STATUS_PROCESSING // Ensure it's still processing
			],
			['%s'],                                   // Format for data
			['%d', '%s', '%s']                        // Format for WHERE
		);

		if ($updated === 1) {
			// Update successful, refresh cache entry
			if ($config['cache_enabled'] && wp_using_ext_object_cache()) {
				$cache_key = $this->get_job_lock_cache_key($job_id);
				wp_cache_set($cache_key, ['token' => $lock_token, 'proc' => $this->get_processor_id(), 'expires' => $new_expires_dt->getTimestamp()], self::CACHE_GROUP, $lock_timeout_seconds + 10);
			}
			// Logging::log_debug("[Retry] Heartbeat updated for job ID {$job_id}.", 'debug');
			return true;
		}

		// Update failed (0 rows affected or DB error)
		if ($wpdb->last_error) {
			Logging::log_error("[Retry] DB error updating heartbeat for job ID {$job_id}: " . $wpdb->last_error, 'error');
		} else {
			Logging::log_warning("[Retry] Heartbeat update failed for job ID {$job_id} (rows affected != 1). Lock may have been lost or job status changed.", 'warning');
		}

		// Clear cache entry as the lock seems invalid now
		if ($config['cache_enabled'] && wp_using_ext_object_cache()) {
			wp_cache_delete($this->get_job_lock_cache_key($job_id), self::CACHE_GROUP);
		}

		return false;
	}

	protected function log_history( array $job, string $status, ?int $dur_ms, ?string $err_msg, ?string $err_code, string $proc_id, ?string $stack = null, ?array $state = null ): void {
		global $wpdb;
		$history_table = $this->get_history_table_name();
// Use a consistent UTC time for 'finished_at'
		$finished_at_dt = $this->get_current_utc_time();

// Determine the start time. Prioritize 'last_attempt_at', otherwise estimate.
		$start_at_dt    = null;
		$start_time_str = $job['last_attempt_at'] ?? null;

		if ( $start_time_str ) {
			try {
// Assume 'last_attempt_at' is already in UTC or parse it as such
				$start_at_dt = new \DateTimeImmutable( $start_time_str, new \DateTimeZone( 'UTC' ) );
			} catch ( \Exception $e ) {
				Logging::log_warning(
					sprintf(
						"[Retry::log_history] Could not parse last_attempt_at '%s' for job %s. Estimating start time.",
						$start_time_str,
						$job['id'] ?? 'unknown'
					),
					[ 'exception' => $e->getMessage(), 'job_id' => $job['id'] ?? 'unknown' ]
				);
// Fall through to estimation logic below
			}
		}

// If start time couldn't be determined from 'last_attempt_at', estimate it
		if ( ! $start_at_dt ) {
			if ( $dur_ms !== null && $dur_ms >= 0 ) {
// Estimate start time by subtracting duration from finish time
				$interval_spec = sprintf( 'PT%fS', $dur_ms / 1000.0 );
				try {
					$interval    = new \DateInterval( $interval_spec );
					$start_at_dt = $finished_at_dt->sub( $interval );
				} catch ( \Exception $e ) {
// Fallback if DateInterval creation or subtraction fails
					Logging::log_warning(
						sprintf(
							"[Retry::log_history] Could not subtract duration interval '%s' for job %s. Using finish time as start time.",
							$interval_spec,
							$job['id'] ?? 'unknown'
						),
						[ 'exception' => $e->getMessage(), 'job_id' => $job['id'] ?? 'unknown' ]
					);
					$start_at_dt = $finished_at_dt; // Fallback: use finish time
				}
			} else {
// If duration is unknown or invalid, use finish time as the best guess
				$start_at_dt = $finished_at_dt;
				Logging::log_info(
					sprintf(
						"[Retry::log_history] Missing 'last_attempt_at' and valid duration for job %s. Using finish time as start time.",
						$job['id'] ?? 'unknown'
					),
					[ 'job_id' => $job['id'] ?? 'unknown' ]
				);
			}
		}

// Prepare state metadata for storage as JSON
		$state_json = null;
		if ( $state !== null ) {
			$state_json = wp_json_encode( $state, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION );
			if ( $state_json === false ) {
				$json_error = json_last_error_msg();
				Logging::log_error(
					sprintf(
						"[Retry::log_history] Failed to JSON encode state metadata for job %s. Error: %s",
						$job['id'] ?? 'unknown',
						$json_error
					),
					[ 'job_id' => $job['id'] ?? 'unknown', 'json_error' => $json_error ]
				);
// Provide a fallback JSON string indicating the encoding error
				$state_json_fallback = wp_json_encode( [ 'error'           => 'Failed to encode state.',
				                                         'json_last_error' => $json_error
				] );
				$state_json          = $state_json_fallback ?: '{"error":"Failed to encode state and fallback encoding also failed."}';
			}
		}

// Ensure job ID and attempt number are integers
		$job_id         = isset( $job['id'] ) ? (int) $job['id'] : 0;
		$attempt_number = isset( $job['retry_count'] ) ? (int) $job['retry_count'] + 1 : 1;

		if ( $job_id <= 0 ) {
			Logging::log_error( "[Retry::log_history] Invalid or missing job ID encountered.", [ 'job_data' => $job ] );
// Consider whether to return early if job_id is essential for history usefulness
// return; // Optional: uncomment to prevent logging history without a valid job ID
		}

// Prepare data array for database insertion, ensuring types and lengths fit schema.
		$data = [
			'job_id'         => $job_id,
			'attempt_number' => $attempt_number,
			'status'         => substr( $status, 0, 20 ),
			// Ensure fits VARCHAR(20)
			'started_at'     => $this->format_datetime_for_sql( $start_at_dt ),
			'finished_at'    => $this->format_datetime_for_sql( $finished_at_dt ),
			'duration_ms'    => ( $dur_ms !== null && $dur_ms >= 0 ) ? (int) $dur_ms : null,
			'error_message'  => $err_msg ? mb_substr( $err_msg, 0, 65530, 'UTF-8' ) : null,
			// Limit for TEXT, multibyte safe
			'error_code'     => $err_code ? substr( $err_code, 0, 100 ) : null,
			// Ensure fits VARCHAR(100)
			'stack_trace'    => $stack,
			// Assumes LONGTEXT or similar
			'processor_id'   => substr( $proc_id, 0, 100 ),
			// Ensure fits VARCHAR(100)
			'log_context'    => null,
			// Field exists in schema, explicitly set to null
			'state_metadata' => $state_json,
			// Store JSON string
		];

// Define corresponding formats for $wpdb->insert placeholders.
		$format = [
			'job_id'         => '%d',
			'attempt_number' => '%d',
			'status'         => '%s',
			'started_at'     => '%s',
			'finished_at'    => '%s',
			'duration_ms'    => '%d', // wpdb handles null correctly for nullable INT columns with %d
			'error_message'  => '%s',
			'error_code'     => '%s',
			'stack_trace'    => '%s',
			'processor_id'   => '%s',
			'log_context'    => '%s', // Use %s even for null for consistency with column type
			'state_metadata' => '%s',
		];

// Sanity check: Ensure data and format arrays align
		if ( count( $data ) !== count( $format ) || array_keys( $data ) !== array_keys( $format ) ) {
			Logging::log_error(
				'[Retry::log_history] CRITICAL: Data and Format array mismatch before DB insert. Aborting history log.',
				[ 'data_keys' => array_keys( $data ), 'format_keys' => array_keys( $format ), 'job_id' => $job_id ]
			);

			return; // Prevent insert with mismatched arrays
		}

// Insert the history record
		$inserted = $wpdb->insert( $history_table, $data, $format );

// Check insertion result and log errors/warnings
		if ( $inserted === false ) {
			Logging::log_error(
				sprintf(
					"[Retry::log_history] Failed to insert history record for job ID %d, attempt %d. DB Error: %s",
					$data['job_id'],
					$data['attempt_number'],
					$wpdb->last_error ?: 'Unknown error (insert returned false)'
				),
				[
					'job_id'       => $data['job_id'],
					'db_error'     => $wpdb->last_error,
					'db_data_keys' => array_keys( $data ), // Avoid logging potentially large values
					'db_format'    => $format
				]
			);
		} elseif ( $inserted === 0 ) {
			Logging::log_warning(
				sprintf(
					"[Retry::log_history] History record insert for job ID %d, attempt %d returned 0 rows affected. DB Error: %s",
					$data['job_id'],
					$data['attempt_number'],
					$wpdb->last_error ?: 'None reported (insert returned 0)'
				),
				[ 'job_id' => $data['job_id'], 'db_error' => $wpdb->last_error ]
			);
		}
	}

	/**
	 * Reschedules a failed job for a future attempt.
	 * Calculates the next delay, updates the job record, and releases the lock.
	 *
	 * @param array      $job         The job data array.
	 * @param string     $err_msg     The error message from the failed attempt.
	 * @param string     $err_code    The error code from the failed attempt.
	 * @param string     $lock_token  The lock token currently held for the job.
	 * @param array|null $next_state  The state metadata to save for the next attempt.
	 * @param bool       $force       If true, attempts update even if lock seems lost (e.g., after hook denial). Default false.
	 * @return bool True if rescheduling was successful, false otherwise (DB error, lock mismatch unless forced).
	 */
	protected function reschedule_failed_job(array $job, string $err_msg, string $err_code, string $lock_token, ?array $next_state = null, bool $force = false): bool {
		global $wpdb;
		$retry_table = $this->get_retry_table_name();
		$job_id = (int) $job['id'];
		$current_attempt = (int) $job['retry_count'] + 1; // The attempt that just failed

		// Calculate delay for the *next* attempt
		$delay_seconds = $this->calculate_delay($job); // calculate_delay uses retry_count, so it calculates delay *after* this attempt
		$next_attempt_time = $this->get_current_utc_time()->modify("+{$delay_seconds} seconds");
		$current_time = $this->get_current_utc_time(); // Time this reschedule is happening

		// Prepare state metadata
		$state_json = null;
		if ($next_state !== null) {
			$state_json = wp_json_encode($next_state);
			if ($state_json === false) {
				Logging::log_error("[Retry] Failed to JSON encode state metadata for reschedule.", 'error', ['job_id' => $job_id]);
				$state_json = '{"error": "Failed to encode state on reschedule."}';
			}
		} else {
			// If no new state provided, keep the existing state (already decoded in $job)
			$existing_state = $job['state_metadata'] ?? null;
			if ($existing_state !== null) {
				$state_json = wp_json_encode($existing_state); // Re-encode
				if ($state_json === false) { // Should ideally not fail if it decoded ok
					Logging::log_error("[Retry] Failed to re-encode existing state metadata for reschedule.", 'error', ['job_id' => $job_id]);
					$state_json = '{"error": "Failed to re-encode state."}';
				}
			}
		}


		$update_data = [
			'status'           => self::STATUS_PENDING, // Set back to pending
			'retry_count'      => $current_attempt,      // Increment attempt count (reflects completed attempt)
			'next_attempt_at'  => $this->format_datetime_for_sql($next_attempt_time),
			'last_attempt_at'  => $this->format_datetime_for_sql($current_time), // When this failed attempt *finished*
			'last_error'       => mb_substr($err_msg, 0, 65530),
			'last_error_code'  => substr($err_code, 0, 100),
			'state_metadata'   => $state_json, // Save state for next time
			'processor_id'     => null, // Release processor
			'lock_token'       => null, // Release lock
			'lock_expires_at'  => null, // Clear lock expiry
		];

		$update_format = [
			'%s', // status
			'%d', // retry_count
			'%s', // next_attempt_at
			'%s', // last_attempt_at
			'%s', // last_error
			'%s', // last_error_code
			'%s', // state_metadata
			'%s', // processor_id
			'%s', // lock_token
			'%s', // lock_expires_at
		];

		$where = ['id' => $job_id];
		$where_format = ['%d'];

		// Only include lock token in WHERE if not forcing
		if (!$force) {
			$where['lock_token'] = $lock_token;
			$where_format[] = '%s';
		}

		$updated = $wpdb->update($retry_table, $update_data, $where, $update_format, $where_format);

		if ($updated === false) {
			Logging::log_error("[Retry] CRITICAL DB error rescheduling job ID {$job_id}: " . $wpdb->last_error, 'critical');
			return false;
		}

		if ($updated === 0 && !$force) {
			// We expected to update 1 row based on ID and lock token, but updated 0. Lock was likely lost.
			Logging::log_warning("[Retry] Reschedule failed for job ID {$job_id} - lock token mismatch or job modified/deleted?", 'warning', ['lock_token_used' => $lock_token]);
			return false;
		}
		if ($updated === 0 && $force) {
			Logging::log_warning("[Retry] Forced reschedule update affected 0 rows for job ID {$job_id}. Job might have been deleted.", 'warning');
			// Still return true as the 'intent' was handled, even if job was gone. Or return false? Let's return false.
			return false;
		}


		Logging::log_info(
			sprintf('[Retry] Rescheduled job ID %d for attempt %d after %ds delay.',
				$job_id, $current_attempt + 1, $delay_seconds
			),
			'info'
		);

		/**
		 * Action hook fired after a job has been rescheduled due to failure.
		 *
		 * @param int   $job_id         The job ID.
		 * @param array $job            The job data before reschedule.
		 * @param int   $next_attempt   The number of the upcoming attempt (e.g., 2 if the 1st failed).
		 * @param int   $delay_seconds  The calculated delay before the next attempt.
		 * @param array|null $saved_state State metadata saved for the next attempt.
		 */
		do_action('sha_retry_job_rescheduled_lc', $job_id, $job, $current_attempt + 1, $delay_seconds, $next_state);

		// Clear cache for this job's lock and the overall stats
		if ($this->get_retry_config()['cache_enabled'] && wp_using_ext_object_cache()) {
			wp_cache_delete($this->get_job_lock_cache_key($job_id), self::CACHE_GROUP);
		}
		$this->clear_stats_cache();

		return true;
	}

    protected function check_poison_pill(int $job_id, string $error_code): bool {
        // Use a class constant or defined value for the threshold
        // Example: $threshold = JobRetryConstants::POISON_PILL_THRESHOLD;
        // For this example, let's assume it's available via $this->POISON_PILL_THRESHOLD
        $threshold = self::POISON_PILL_THRESHOLD;

        // Cannot check without a specific error code or if the threshold is disabled/invalid.
        // @phpstan-ignore-next-line Defensive check for future-proofing
        if (empty($error_code) || $threshold <= 0) {
            return false;
        }

        global $wpdb;
        // Assume $this->get_history_table_name() returns the correct table name
        $history_table = $this->get_history_table_name();
        // Use the class constant for failure status
        $failure_status = self::STATUS_FAILURE;

        // Get the status and error code of the last $threshold attempts for this job.
        // Order by attempt_number or a reliable timestamp (assuming attempt_number is sequential and reliable).
        $sql = $wpdb->prepare(
            "SELECT status, error_code
             FROM `{$history_table}`
             WHERE job_id = %d
             ORDER BY attempt_number DESC
             LIMIT %d",
            $job_id,
            $threshold
        );

        // Fetch results as an array of associative arrays.
        $last_attempts = $wpdb->get_results($sql, ARRAY_A);

        // Handle potential database errors during the query.
        if ($wpdb->last_error) {
            // Assume Logging::log_error exists for logging.
            Logging::log_error(
                sprintf(
                    "[Retry] DB error checking poison pill for job ID %d: %s",
                    $job_id,
                    $wpdb->last_error
                ),
                'error'
            );
            // Fail safe: don't assume a poison pill if the database check failed.
            return false;
        }

        // Check if we retrieved exactly $threshold records. If not, the condition isn't met.
        if (count($last_attempts) !== $threshold) {
            return false;
        }

        // Verify that ALL of these last attempts were failures AND had the SAME specific error code.
        foreach ($last_attempts as $attempt) {
            if ($attempt['status'] !== $failure_status || $attempt['error_code'] !== $error_code) {
                // If any attempt in the sequence was not a failure or had a different error code,
                // it breaks the consecutive pattern for this specific error.
                return false;
            }
        }

        // If the loop completed without returning false, all conditions are met.
        // Assume Logging::log_warning exists for logging.
        Logging::log_warning(
            sprintf(
                "[Retry] Poison pill condition detected for job ID %d with error code '%s' across %d attempts.",
                $job_id,
                $error_code,
                $threshold
            ),
            'warning'
        );
        return true;
    }
	/**
	 * Removes a successfully completed job from the main retry table.
	 * Optionally checks the lock token to ensure the correct processor is removing it.
	 *
	 * @param int         $job_id     The ID of the job to remove.
	 * @param string|null $lock_token If provided, only deletes if the job has this lock token.
	 * @return bool True if the job was deleted, false otherwise (not found, lock mismatch, or DB error).
	 */
	public function remove_retry_operation(int $job_id, ?string $lock_token = null): bool {
		global $wpdb;
		$table = $this->get_retry_table_name();

		$where = ['id' => $job_id];
		$where_format = ['%d'];

		if ($lock_token !== null) {
			$where['lock_token'] = $lock_token;
			$where_format[] = '%s';
		}

		$deleted = $wpdb->delete($table, $where, $where_format);

		if ($deleted === false) {
			Logging::log_error("[Retry] DB error deleting job ID {$job_id}: " . $wpdb->last_error, 'error');
			return false;
		}

		if ($deleted > 0) {
			Logging::log_debug("[Retry] Deleted job ID {$job_id}.", 'debug');
			// Clear cache entry for this job if caching is enabled
			if ($this->get_retry_config()['cache_enabled'] && wp_using_ext_object_cache()) {
				wp_cache_delete($this->get_job_lock_cache_key($job_id), self::CACHE_GROUP);
			}
			$this->clear_stats_cache(); // Queue composition changed
			return true;
		} else {
			// Deleted 0 rows
			if ($lock_token !== null) {
				// If lock token was specified, this likely means the lock was lost or the job was already deleted/modified.
				Logging::log_warning("[Retry] Delete job ID {$job_id} failed - row not found or lock token mismatch?", 'warning', ['lock_token_used' => $lock_token]);
			} else {
				// If no lock token was specified, it simply means the job ID didn't exist.
				Logging::log_debug("[Retry] Delete job ID {$job_id} failed - row not found.", 'debug');
			}
			// No need to clear stats cache if nothing was deleted
			return false;
		}
	}

	/**
	 * Checks if there are any active or scheduled jobs in the queue.
	 * "Active" includes pending, processing, scheduled, waiting, and paused jobs.
	 *
	 * @param string|null $category Optional. If provided, checks only within this category.
	 * @return bool True if pending/active operations exist, false otherwise or on DB error.
	 */
	public function has_pending_retry_operations(?string $category = null): bool {
		global $wpdb;
		$table = $this->get_retry_table_name();

		$statuses_to_check = [
			self::STATUS_PENDING,
			self::STATUS_PROCESSING,
			self::STATUS_SCHEDULED,
			self::STATUS_WAITING_DEPENDENCY,
			self::STATUS_PAUSED
		];
		$status_placeholders = implode(',', array_fill(0, count($statuses_to_check), '%s'));

		$sql = "SELECT 1 FROM `{$table}` WHERE `status` IN ({$status_placeholders})";
		$params = $statuses_to_check;

		if ($category !== null) {
			$sql .= " AND `category` = %s";
			$params[] = $category;
		}

		$sql .= " LIMIT 1";

		$query = $wpdb->prepare($sql, $params);
		$result = $wpdb->get_var($query);

		if ($wpdb->last_error) {
			Logging::log_error("[Retry] DB error checking for pending operations: " . $wpdb->last_error, 'error');
			return false; // Fail safe
		}

		return !empty($result); // Returns 1 if found, null/0 otherwise
	}

	/**
	 * Finds and moves expired jobs (past their `expires_at` time) to the Dead Letter Queue.
	 * Processes in batches to avoid excessive load.
	 *
	 * @return int Number of jobs successfully moved to DLQ due to expiration.
	 */
	protected function process_expired_jobs(): int {
		global $wpdb;
		$retry_table = $this->get_retry_table_name();
		$now_utc = $this->get_current_utc_time();
		$now_sql = $this->format_datetime_for_sql($now_utc);
		$expired_count = 0;
		$batch_limit = 100; // Process up to 100 expired jobs per run

		// Find jobs in states where expiration matters, and whose expires_at is in the past.
		// Excludes 'processing' because expiry during processing is handled differently (stall detection).
		// Excludes 'failed' as that's terminal in the main table already.
		$relevant_statuses = [
			self::STATUS_PENDING,
			self::STATUS_SCHEDULED,
			self::STATUS_WAITING_DEPENDENCY,
			self::STATUS_PAUSED
		];
		$status_placeholders = implode(',', array_fill(0, count($relevant_statuses), '%s'));

		// Optimized: Select only needed columns instead of SELECT *
		$sql = $wpdb->prepare(
			"SELECT id, processor_id, status, retry_count, max_retries, next_retry_at, last_error, expires_at FROM `{$retry_table}`
             WHERE `status` IN ({$status_placeholders})
               AND `expires_at` IS NOT NULL
               AND `expires_at` <= %s
             LIMIT %d",
			array_merge($relevant_statuses, [$now_sql, $batch_limit])
		);

		$expired_jobs = $wpdb->get_results($sql, ARRAY_A);

		if ($wpdb->last_error) {
			Logging::log_error("[Retry] DB error finding expired jobs: " . $wpdb->last_error, 'error');
			return 0;
		}

		if (empty($expired_jobs)) {
			return 0; // No expired jobs found
		}

		foreach ($expired_jobs as $job) {
			$job_id = (int)$job['id'];
			$expiry_time = $job['expires_at'] ?? 'N/A';
			$reason = sprintf("Job expired at %s before completion.", $expiry_time);

			// Attempt to move to DLQ
			$moved = $this->move_to_dlq(
				$job,
				$reason,
				self::DLQ_REASON_EXPIRED
			// No lock token needed/available here, move_to_dlq handles deleting by ID
			);

			if ($moved) {
				$expired_count++;
				Logging::log_info("[Retry] Moved expired job ID {$job_id} (Expired at: {$expiry_time}) to DLQ.", 'info');
				// Promote dependents as the primary job failed (expired)
				$this->promote_dependent_jobs($job_id, false);
			} else {
				// Log failure to move, might need manual intervention or retry later
				Logging::log_error("[Retry] Failed to move expired job ID {$job_id} to DLQ.", 'error');
				// Consider marking as failed in main table as fallback?
				// $this->mark_as_failed($job_id, $reason, self::DLQ_REASON_EXPIRED);
			}
		}

		if ($expired_count > 0) {
			Logging::log_info("[Retry] Processed expiration: Moved {$expired_count} jobs to DLQ.", 'info');
			$this->clear_stats_cache(); // Queue composition changed
		}

		return $expired_count;
	}

	/**
	 * Checks for jobs waiting on dependencies ('waiting_dependency' status)
	 * and promotes them to 'pending' if the dependency has completed or failed.
	 * If dependency failed permanently (in DLQ), moves the waiting job to DLQ.
	 */
	protected function promote_waiting_jobs(): void {
		global $wpdb;
		$retry_table = $this->get_retry_table_name();
		$dlq_table = $this->get_dlq_table_name();
		$batch_limit = 100; // Check up to 100 waiting jobs per run
		$promoted_count = 0;
		$failed_dependency_count = 0;
		$now_utc = $this->get_current_utc_time();

		// Find jobs currently waiting on dependencies
		$waiting_sql = $wpdb->prepare(
			"SELECT id, depends_on_job_id FROM `{$retry_table}` WHERE `status` = %s LIMIT %d",
			self::STATUS_WAITING_DEPENDENCY,
			$batch_limit
		);
		$waiting_jobs = $wpdb->get_results($waiting_sql, ARRAY_A);

		if ($wpdb->last_error) {
			Logging::log_error("[Retry Promote] DB error finding waiting jobs: " . $wpdb->last_error, 'error');
			return;
		}
		if (empty($waiting_jobs)) {
			return; // Nothing to promote
		}

		$dependency_ids = array_unique(array_filter(array_column($waiting_jobs, 'depends_on_job_id'), 'is_numeric'));
		if (empty($dependency_ids)) {
			Logging::log_warning("[Retry Promote] Found waiting jobs with invalid depends_on_job_id.", 'warning');
			// Maybe handle these invalid dependencies (e.g., mark as failed)? For now, we just ignore them.
			return;
		}

		// Check status of dependencies in main table and DLQ
		$dep_ids_placeholder = implode(',', array_fill(0, count($dependency_ids), '%d'));

		// Check main table (find IDs that *still* exist)
		$active_deps_sql = $wpdb->prepare("SELECT id FROM `{$retry_table}` WHERE id IN ({$dep_ids_placeholder})", $dependency_ids);
		$active_dependency_ids = $wpdb->get_col($active_deps_sql);
		if ($wpdb->last_error) {
			Logging::log_error("[Retry Promote] DB error checking active dependencies: " . $wpdb->last_error, 'error');
			return; // Cannot proceed reliably
		}
		$active_dependency_ids = array_map('intval', $active_dependency_ids ?: []); // Ensure array of ints

		// Check DLQ table (find IDs that failed permanently)
		$failed_deps_sql = $wpdb->prepare("SELECT original_job_id FROM `{$dlq_table}` WHERE original_job_id IN ({$dep_ids_placeholder})", $dependency_ids);
		$failed_dependency_ids = $wpdb->get_col($failed_deps_sql);
		if ($wpdb->last_error) {
			Logging::log_error("[Retry Promote] DB error checking failed dependencies (DLQ): " . $wpdb->last_error, 'error');
			return; // Cannot proceed reliably
		}
		$failed_dependency_ids = array_map('intval', $failed_dependency_ids ?: []); // Ensure array of ints

		$ids_to_promote = [];
		$jobs_to_fail = []; // [job_id => dependency_id]

		// Determine fate of each waiting job
		foreach ($waiting_jobs as $waiting_job) {
			$waiting_job_id = (int)$waiting_job['id'];
			$dependency_id = (int)$waiting_job['depends_on_job_id'];

			if ($dependency_id <= 0) continue; // Skip invalid IDs

			if (in_array($dependency_id, $failed_dependency_ids, true)) {
				// Dependency failed permanently (is in DLQ)
				$jobs_to_fail[$waiting_job_id] = $dependency_id;
			} elseif (!in_array($dependency_id, $active_dependency_ids, true)) {
				// Dependency is NOT active AND NOT in DLQ -> Assume completed successfully
				$ids_to_promote[] = $waiting_job_id;
			}
			// If dependency is still active, do nothing - job continues waiting.
		}

		// --- Process Promotions ---
		if (!empty($ids_to_promote)) {
			// Calculate initial next_attempt_at for promoted jobs (now + initial delay)
			// Note: We use a generic delay here. Ideally, we'd use the job's specific config,
			// but that requires fetching full job data. Using config default is a reasonable compromise.
			$config = $this->get_retry_config();
			$initial_delay = max(1, (int)($config['initial_delay'] ?? 15));
			$next_attempt_time = $now_utc->modify("+{$initial_delay} seconds");
			$next_attempt_sql = $this->format_datetime_for_sql($next_attempt_time);

			$promote_ids_placeholder = implode(',', array_fill(0, count($ids_to_promote), '%d'));
			$promote_sql = $wpdb->prepare(
				"UPDATE `{$retry_table}`
                 SET `status` = %s,
                     `next_attempt_at` = %s
                 WHERE `id` IN ({$promote_ids_placeholder})
                   AND `status` = %s", // Ensure it's still waiting
				array_merge(
					[self::STATUS_PENDING, $next_attempt_sql],
					$ids_to_promote,
					[self::STATUS_WAITING_DEPENDENCY]
				)
			);
			$promoted_rows = $wpdb->query($promote_sql);

			if ($promoted_rows === false) {
				Logging::log_error("[Retry Promote] DB error promoting jobs: " . $wpdb->last_error, 'error');
			} elseif ($promoted_rows > 0) {
				$promoted_count = $promoted_rows;
				Logging::log_info("[Retry Promote] Promoted {$promoted_count} jobs from waiting_dependency to pending.", 'info');
				// Schedule the processor if it wasn't already, as new jobs are ready
				$this->schedule_retry_processor_event();
			}
			if (count($ids_to_promote) !== $promoted_rows) {
				Logging::log_warning("[Retry Promote] Mismatch promoting jobs. Tried: " . count($ids_to_promote) . ", Succeeded: " . $promoted_rows, 'warning');
			}
		}

		// --- Process Dependency Failures ---
		if (!empty($jobs_to_fail)) {
			$jobs_to_fail_ids = array_keys($jobs_to_fail);
			$fail_ids_placeholder = implode(',', array_fill(0, count($jobs_to_fail_ids), '%d'));

			// Fetch full data for jobs that need moving to DLQ
			// Optimized: Select only needed columns instead of SELECT *
			$fetch_fail_sql = $wpdb->prepare(
				"SELECT id, processor_id, status, retry_count, max_retries, next_retry_at, last_error, payload, created_at, updated_at, expires_at FROM `{$retry_table}` WHERE id IN ({$fail_ids_placeholder}) AND status = %s",
				array_merge($jobs_to_fail_ids, [self::STATUS_WAITING_DEPENDENCY])
			);
			$fail_job_data = $wpdb->get_results($fetch_fail_sql, ARRAY_A);

			if ($wpdb->last_error) {
				Logging::log_error("[Retry Promote] DB error fetching jobs whose dependencies failed: " . $wpdb->last_error, 'error');
			} elseif (!empty($fail_job_data)) {
				foreach ($fail_job_data as $job_data) {
					$job_id = (int)$job_data['id'];
					$failed_dep_id = $jobs_to_fail[$job_id] ?? 'unknown';
					$reason = "Dependency job ID {$failed_dep_id} failed permanently (moved to DLQ).";

					$moved = $this->move_to_dlq($job_data, $reason, self::DLQ_REASON_DEPENDENCY_FAILED);
					if ($moved) {
						$failed_dependency_count++;
						Logging::log_info("[Retry Promote] Moved job ID {$job_id} to DLQ due to failed dependency {$failed_dep_id}.", 'info');
					} else {
						Logging::log_error("[Retry Promote] Failed to move job ID {$job_id} to DLQ after dependency failure.", 'error');
						// Consider marking as failed in main table?
					}
				}
			}
		}

		if ($promoted_count > 0 || $failed_dependency_count > 0) {
			$this->clear_stats_cache(); // Queue composition changed
		}
	}


	/**
	 * Marks a job as permanently failed in the main retry table.
	 * This is a fallback action, typically used if moving to DLQ fails.
	 * It clears lock information to prevent further processing attempts.
	 *
	 * @param int         $job_id      The ID of the job to mark as failed.
	 * @param string      $err_msg     The final error message.
	 * @param string      $err_code    The final error code.
	 * @param string|null $lock_token  Optional. If provided, only updates if the job has this lock token.
	 * @return bool True if the job was successfully marked as failed, false otherwise.
	 */
	protected function mark_as_failed(int $job_id, string $err_msg, string $err_code, ?string $lock_token = null): bool {
		global $wpdb;
		$table = $this->get_retry_table_name();

		$update_data = [
			'status'           => self::STATUS_FAILED, // Terminal status in main table
			'last_error'       => mb_substr($err_msg, 0, 65530), // Limit TEXT size
			'last_error_code'  => mb_substr($err_code, 0, 100), // Limit VARCHAR size
			'last_attempt_at'  => $this->format_datetime_for_sql($this->get_current_utc_time()),
			'processor_id'     => null, // Clear processor
			'lock_token'       => null, // Clear lock
			'lock_expires_at'  => null, // Clear lock expiry
		];
		$update_format = ['%s', '%s', '%s', '%s', '%s', '%s', '%s'];

		$where = ['id' => $job_id];
		$where_format = ['%d'];

		// Add lock token check if provided
		if ($lock_token !== null) {
			$where['lock_token'] = $lock_token;
			$where_format[] = '%s';
		}

		$updated = $wpdb->update($table, $update_data, $where, $update_format, $where_format);

		if ($updated === false) {
			Logging::log_error("[Retry] DB error marking job ID {$job_id} as failed: " . $wpdb->last_error, 'error');
			return false;
		}

		if ($updated > 0) {
			Logging::log_warning("[Retry] Marked job ID {$job_id} as failed in main table (DLQ fallback?).", 'warning');
			$this->clear_stats_cache(); // Queue composition changed
			// Clear any lingering cache lock entry
			if ($this->get_retry_config()['cache_enabled'] && wp_using_ext_object_cache()) {
				wp_cache_delete($this->get_job_lock_cache_key($job_id), self::CACHE_GROUP);
			}
			/**
			 * Action hook fired when a job is marked as failed directly in the main table (usually a fallback).
			 *
			 * @param int    $job_id   The job ID.
			 * @param string $err_msg  The error message.
			 * @param string $err_code The error code.
			 */
			do_action('sha_retry_job_marked_failed_lc', $job_id, $err_msg, $err_code);
			return true;
		} else {
			Logging::log_warning("[Retry] Mark as failed update affected 0 rows for job ID {$job_id}. Job not found or lock mismatch?", 'warning', ['lock_token_used' => $lock_token]);
			return false;
		}
	}

	/**
	 * Cleans up old records from the history and DLQ tables based on configured retention days.
	 * Runs DELETE queries in batches.
	 *
	 * @return array Associative array with counts of deleted records: ['hist' => int, 'dlq' => int].
	 */
	public function cleanup_old_records(): array {
		global $wpdb;
		// Assume $this->get_retry_config(), $this->get_history_table_name(),
		// $this->get_dlq_table_name(), $this->get_current_utc_time(),
		// $this->format_datetime_for_sql(), Logging::log_error(), Logging::log_info()
		// exist and function as expected.

		$config = $this->get_retry_config();
		$history_table = $this->get_history_table_name();
		$dlq_table = $this->get_dlq_table_name();
		$deleted_counts = ['hist' => 0, 'dlq' => 0];

		$now_utc = $this->get_current_utc_time();

		// Use constants or fetch from config if preferred
		$history_delete_limit = 5000;
		$dlq_delete_limit = 1000;

		// --- Cleanup History Table ---
		$history_retention_days = (int) ($config['history_retention_days'] ?? 0);
		if ($history_retention_days > 0) {
			try {
				// Clone $now_utc to avoid modifying the original object for subsequent calculations
				$cutoff_dt = (clone $now_utc)->modify("-{$history_retention_days} days");
				if (!$cutoff_dt) {
					// Handle potential DateTime::modify failure
					throw new \Exception("Failed to calculate history cutoff date.");
				}
				$cutoff_sql = $this->format_datetime_for_sql($cutoff_dt);

				$sql = $wpdb->prepare(
					"DELETE FROM `{$history_table}` WHERE `finished_at` < %s LIMIT %d",
					$cutoff_sql,
					$history_delete_limit
				);
				// $wpdb->query returns number of rows affected or false on error
				$deleted = $wpdb->query($sql);

				if ($deleted !== false) {
					$deleted_counts['hist'] = (int) $deleted;
				} else {
					// Log error if query failed
					Logging::log_error("[Retry Cleanup] Error cleaning history table: " . $wpdb->last_error, 'error');
				}
			} catch (\Throwable $e) {
				// Log any exception during the history cleanup process
				Logging::log_error("[Retry Cleanup] Exception cleaning history table: " . $e->getMessage(), 'error');
			}
		}

		// --- Cleanup DLQ Table ---
		$dlq_retention_days = (int) ($config['dlq_retention_days'] ?? 0);
		if ($dlq_retention_days > 0) {
			try {
				// Clone $now_utc again to ensure calculation is based on the original, unmodified time
				$cutoff_dt = (clone $now_utc)->modify("-{$dlq_retention_days} days");
				if (!$cutoff_dt) {
					// Handle potential DateTime::modify failure
					throw new \Exception("Failed to calculate DLQ cutoff date.");
				}
				$cutoff_sql = $this->format_datetime_for_sql($cutoff_dt);

				$sql = $wpdb->prepare(
					"DELETE FROM `{$dlq_table}` WHERE `failed_at` < %s LIMIT %d",
					$cutoff_sql,
					$dlq_delete_limit
				);
				// $wpdb->query returns number of rows affected or false on error
				$deleted = $wpdb->query($sql);

				if ($deleted !== false) {
					$deleted_counts['dlq'] = (int) $deleted;
				} else {
					// Log error if query failed
					Logging::log_error("[Retry Cleanup] Error cleaning DLQ table: " . $wpdb->last_error, 'error');
				}
			} catch (\Throwable $e) {
				// Log any exception during the DLQ cleanup process
				Logging::log_error("[Retry Cleanup] Exception cleaning DLQ table: " . $e->getMessage(), 'error');
			}
		}

		// Log summary if any records were deleted
		if ($deleted_counts['hist'] > 0 || $deleted_counts['dlq'] > 0) {
			Logging::log_info(
				sprintf("[Retry Cleanup] Completed: Deleted %d history records, %d DLQ records.",
					$deleted_counts['hist'], $deleted_counts['dlq']
				),
				'info'
			);
		}

		return $deleted_counts;
	}

	/**
	 * Retrieves statistics about the retry queue (counts per status).
	 * Uses WP Object Cache if enabled and available.
	 *
	 * @return array Associative array of status counts (e.g., ['pending' => 10, 'dlq' => 5]).
	 */
	public function get_queue_stats(): array {
		$config = $this->get_retry_config();

		if (!$config['cache_enabled'] || !wp_using_ext_object_cache()) {
			// Caching disabled or external cache not available, query directly
			return $this->query_queue_stats();
		}

		$stats = wp_cache_get(self::STATS_CACHE_KEY, self::CACHE_GROUP);

		if ($stats === false) {
			// Cache miss, query the database
			$stats = $this->query_queue_stats();
			// Store in cache
			wp_cache_set(self::STATS_CACHE_KEY, $stats, self::CACHE_GROUP, self::STATS_CACHE_TTL);
			Logging::log_debug("[Retry Stats] Cache miss. Queried and cached stats.", 'debug');
		} else {
			Logging::log_debug("[Retry Stats] Cache hit.", 'debug');
		}

		// Ensure result is always an array
		return is_array($stats) ? $stats : $this->get_default_stats_array();
	}

	/** Helper to get the default stats array structure */
	private function get_default_stats_array(): array {
		return [
			'pending' => 0, 'scheduled' => 0, 'processing' => 0,
			'waiting_dependency' => 0, 'paused' => 0, 'failed' => 0,
			'dlq' => 0, 'total' => 0 // 'total' is sum of main table statuses
		];
	}

	/**
	 * Queries the database directly to get current queue statistics.
	 *
	 * @return array Associative array of status counts.
	 */
	protected function query_queue_stats(): array {
		global $wpdb;
		$retry_table = $this->get_retry_table_name();
		$dlq_table = $this->get_dlq_table_name();
		$stats = $this->get_default_stats_array();

		// Get counts from the main retry table grouped by status
		$main_results = $wpdb->get_results(
			"SELECT status, COUNT(*) as count FROM `{$retry_table}` GROUP BY status",
			ARRAY_A
		);

		if ($wpdb->last_error) {
			Logging::log_error("[Retry Stats] DB error querying main table stats: " . $wpdb->last_error, 'error');
			// Return default empty stats on error
			return $stats;
		}

		$total_main = 0;
		if ($main_results) {
			foreach ($main_results as $row) {
				if (isset($stats[$row['status']])) {
					$count = (int) $row['count'];
					$stats[$row['status']] = $count;
					$total_main += $count;
				}
			}
		}
		$stats['total'] = $total_main;

		// Get count from the DLQ table
		$dlq_count = $wpdb->get_var("SELECT COUNT(*) FROM `{$dlq_table}`");
		if ($wpdb->last_error) {
			Logging::log_error("[Retry Stats] DB error querying DLQ table count: " . $wpdb->last_error, 'error');
			// Keep DLQ count as 0 on error
		} else {
			$stats['dlq'] = (int) $dlq_count;
		}

		/**
		 * Filters the calculated queue statistics before returning or caching.
		 *
		 * @param array $stats The calculated statistics array.
		 * @return array The filtered statistics array.
		 */
		return apply_filters('sha_retry_queue_stats_lc', $stats);
	}

	/**
	 * Main processing function called by the scheduler (e.g., WP-Cron).
	 * Handles expired jobs, retrieves and locks a batch of ready jobs,
	 * and executes them.
	 */
	public function process_all_retries(): void {
		$config = $this->get_retry_config();
		$batch_size = max(1, (int) ($config['batch_size'] ?? 50));
		$start_time = microtime(true);
		$processor_id = $this->get_processor_id(); // Ensure processor ID is generated/retrieved

		if (self::$signalShutdown) {
			Logging::log_info("[Retry Run] Shutdown signal detected. Aborting processing run.", 'info');
			return;
		}

		Logging::log_info(
			sprintf('[Retry Run] Starting processing run (Batch Size: %d, Processor ID: %s).', $batch_size, $processor_id),
			'info'
		);
		/**
		 * Action hook fired at the beginning of a retry processing run.
		 * @ignore Internal use.
		 * @param string $processor_id The ID of the processor instance running.
		 */
		do_action('sha_retry_run_start_lc', $processor_id);

		$expired_count = 0;
		$processed_count = 0;
		$jobs_processed_ids = []; // Track IDs processed in this run

		// 1. Process Expired Jobs
		try {
			$expired_count = $this->process_expired_jobs();
		} catch (\Throwable $e) {
			Logging::log_error("[Retry Run] Error processing expired jobs: " . $e->getMessage(), 'error', ['exception' => $e]);
		}

		// Check for shutdown signal again before heavy lifting
		if (self::$signalShutdown) {
			Logging::log_info("[Retry Run] Shutdown signal detected after expiration check.", 'info');
			return;
		}

		// 2. Retrieve and Lock Ready Jobs
		$jobs_to_process = [];
		try {
			$jobs_to_process = $this->retrieve_and_lock_ready_jobs($batch_size, $processor_id);
		} catch (\InvalidArgumentException $e) {
			Logging::log_error("[Retry Run] Invalid argument during retrieve/lock: " . $e->getMessage(), 'critical');
			// Stop run if fundamental args are wrong
			return;
		} catch (\RuntimeException $e) {
			Logging::log_error("[Retry Run] Runtime exception during retrieve/lock (e.g., random bytes): " . $e->getMessage(), 'critical');
			// Stop run on critical internal error
			return;
		} catch (\Throwable $e) {
			// Catch any other unexpected errors during locking phase
			Logging::log_error("[Retry Run] CRITICAL error during job retrieval/locking: " . $e->getMessage(), 'critical', ['exception' => $e]);
			// Stop run on major failure
			return;
		}

		// 3. Execute Locked Jobs
		if (!empty($jobs_to_process)) {
			Logging::log_info(sprintf('[Retry Run] Locked %d jobs for processing.', count($jobs_to_process)), 'info');

			foreach ($jobs_to_process as $job_id => $job) {
				if (self::$signalShutdown) {
					Logging::log_info("[Retry Run] Shutdown signal detected during batch execution. Breaking loop.", 'info');
					// Release lock for the remaining jobs? No, let heartbeat timeout handle it for safety.
					break;
				}

				$job_start_time = microtime(true);
				$job_success = false;
				try {
					// Execute the operation
					$job_success = $this->execute_retry_operation($job);
					$processed_count++;
					$jobs_processed_ids[] = $job_id;

				} catch (\Throwable $ex_err) {
					// This catch block is a safeguard against FATAL errors within execute_retry_operation itself.
					// Normal execution errors are handled *inside* execute_retry_operation.
					Logging::log_error(
						"[Retry Run] UNEXPECTED FATAL error during execute_retry_operation for job ID {$job_id}: " . $ex_err->getMessage(),
						'critical',
						['exception' => $ex_err, 'job_id' => $job_id]
					);
					// Try to move the job to DLQ as it caused a critical failure in the runner
					// Pass the raw job data (might be slightly stale if filter modified it)
					$this->move_to_dlq(
						$job, // Use the locked job data we have
						"Critical runner error during execution: " . mb_substr($ex_err->getMessage(), 0, 500),
						self::DLQ_REASON_FAILED, // Or a specific 'runner_error' code
						$job['lock_token'] ?? null // Pass lock token if available
					);
					// Mark as processed even though it critically failed, to avoid infinite loops if the error persists
					$processed_count++;
					$jobs_processed_ids[] = $job_id;
				} finally {
					$job_duration = microtime(true) - $job_start_time;
					Logging::log_debug(sprintf("[Retry Run] Job ID %d finished processing (Success: %s, Duration: %.4fs).",
						$job_id, $job_success ? 'Yes' : 'No', $job_duration
					), 'debug');
				}
			} // End foreach job
		} else {
			Logging::log_debug('[Retry Run] No jobs found or locked in this run.', 'debug');
		}

		// 4. Finalize Run
		$run_duration = microtime(true) - $start_time;
		Logging::log_info(
			sprintf('[Retry Run] Finished processing run. Processed: %d jobs (%d expired). Duration: %.4fs.',
				$processed_count, $expired_count, $run_duration
			),
			'info'
		);

		/**
		 * Action hook fired at the end of a retry processing run.
		 * @ignore Internal use.
		 * @param string $processor_id        The ID of the processor instance.
		 * @param int    $processed_count     Number of jobs attempted (success or failure).
		 * @param float  $run_duration        Total duration of the run in seconds.
		 * @param int[]  $jobs_processed_ids  Array of job IDs processed in this run.
		 */
		do_action('sha_retry_run_end_lc', $processor_id, $processed_count, $run_duration, $jobs_processed_ids);

		// 5. Reschedule or Unschedule based on pending work
		if (self::$signalShutdown) {
			Logging::log_info("[Retry Run] Shutdown signalled. Not rescheduling cron.", 'info');
			// Optionally unschedule here, or let shutdown handler do it.
		} else {
			try {
				if (!$this->has_pending_retry_operations()) {
					// No more work left, unschedule the cron event
					$this->unschedule_retry_processor_event();
				} else {
					// Still work to do, ensure the cron event is scheduled
					$this->schedule_retry_processor_event(); // Idempotent check
				}
			} catch (\Throwable $e) {
				Logging::log_error("[Retry Run] Error during post-run scheduling check: " . $e->getMessage(), 'error');
			}
		}
	}
	public function cron_callback(): void {
		// Set higher limits for potentially long-running task
		// Use @ to suppress errors on restricted hosting environments
		@set_time_limit(MINUTE_IN_SECONDS * 10); // 10 minutes
		
		// Attempt to increase memory limit - use WP function for compatibility
		// WordPress handles this gracefully on restricted hosts
		if (function_exists('wp_raise_memory_limit')) {
			wp_raise_memory_limit('cron'); // Let WP decide appropriate level
		} else {
			// Fallback - will silently fail on restricted hosts, which is acceptable
			@ini_set('memory_limit', '512M');
		}

		$lock_key_base = 'sha_retry_cron_lock_ludicrous_sc';
		$lock_key_cache = $lock_key_base . '_cache'; // Key for WP Object Cache lock
		$lock_key_option = $lock_key_base . '_opt';  // Key for DB option lock (fallback)
		$lock_value = time(); // Use timestamp as lock value
		$config = $this->get_retry_config();
		// Cache lock TTL: Job lock timeout + buffer (e.g., 1 minute)
		// Use max() to ensure TTL is at least 60 seconds.
		$cache_lock_ttl = max(60, (int)($config['lock_timeout'] ?? 600)) + 60;
		// Option lock expiry check: Longer timeout (e.g., 15 minutes) to be more lenient with DB checks
		$option_lock_timeout = MINUTE_IN_SECONDS * 15;

		$acquired_lock = false;
		$lock_acquired_method = 'none'; // Track how the lock was acquired ('cache', 'db', 'db_stale')

		// 1. Try acquiring lock via WP Object Cache (if enabled and available)
		if (($config['cache_enabled'] ?? false) && wp_using_ext_object_cache()) {
			// wp_cache_add is atomic - returns true only if key didn't exist and was added.
			if (wp_cache_add($lock_key_cache, $lock_value, '', $cache_lock_ttl)) {
				$acquired_lock = true;
				$lock_acquired_method = 'cache';
				Logging::log_debug("[Retry Cron] Acquired lock via WP Cache ('{$lock_key_cache}').", 'debug');
			} else {
				// Cache lock exists or add failed. Log for awareness.
				$existing_lock_time = wp_cache_get($lock_key_cache);
				if ($existing_lock_time !== false) {
					Logging::log_info(
						sprintf("[Retry Cron] Cache lock ('%s') is held (value: %s). Skipping run.", $lock_key_cache, $existing_lock_time),
						'info'
					);
				} else {
					// wp_cache_add failed but get returns false - potentially transient issue. Log and proceed to DB lock.
					Logging::log_info("[Retry Cron] Failed to acquire cache lock ('{$lock_key_cache}'), key not found immediately after. Will attempt DB lock.", 'info');
				}
			}
		}

		// 2. If cache lock failed or wasn't used, try acquiring lock via DB option (atomic add_option)
		if (!$acquired_lock) {
			// add_option is atomic - returns true if option was added, false if it already exists.
			// 'no' prevents autoloading.
			if (add_option($lock_key_option, $lock_value, '', false)) {
				// Successfully added the DB option lock
				$acquired_lock = true;
				$lock_acquired_method = 'db';
				Logging::log_debug("[Retry Cron] Acquired lock via DB Option ('{$lock_key_option}').", 'debug');
			} else {
				// Option already exists, check if it's expired
				$current_lock_time = (int) get_option($lock_key_option, 0);
				$lock_age = ($current_lock_time > 0) ? ($lock_value - $current_lock_time) : 0;

				if ($current_lock_time > 0 && $lock_age > $option_lock_timeout) {
					// Lock is older than the timeout, assume stale. Attempt to override it.
					Logging::log_warning(
						sprintf("[Retry Cron] Stale DB lock detected ('%s', age: %d s). Attempting override.", $lock_key_option, $lock_age),
						'warning',
						['lock_key' => $lock_key_option, 'current_lock_time' => $current_lock_time, 'option_lock_timeout' => $option_lock_timeout]
					);

					// update_option is NOT atomic, but we accept the small race condition risk for stale lock recovery.
					// Update the option with the new lock value.
					$updated = update_option($lock_key_option, $lock_value, false);

					// Verify the update took hold with our value (basic check against immediate overwrite)
					// Only acquire lock if update reported success AND the current value is the one we set.
					if ($updated && (int) get_option($lock_key_option, 0) === $lock_value) {
						$acquired_lock = true;
						$lock_acquired_method = 'db_stale';
						Logging::log_info("[Retry Cron] Successfully overrode stale DB lock ('{$lock_key_option}').", 'info');
					} else {
						Logging::log_warning("[Retry Cron] Failed to definitively override stale DB lock ('{$lock_key_option}'). Update status: " . ($updated ? 'true' : 'false') . ". Value may have changed. Skipping run.", 'warning');
						// Do not set $acquired_lock = true
					}
				} else {
					// Lock exists and is not expired. Another process is running.
					Logging::log_info(
						sprintf("[Retry Cron] DB lock ('%s') is held (age: %d s). Skipping run.", $lock_key_option, $lock_age),
						'info'
					);
				}
			}
		}

		// 3. If we still haven't acquired a lock after all attempts
		if (!$acquired_lock) {
			Logging::log_info("[Retry Cron] Failed to acquire lock via Cache or DB. Skipping run.", 'info');
			return; // Exit gracefully
		}

		// 4. Register Shutdown Function to Release Lock
		// This ensures the lock is released even if the script terminates unexpectedly.
		register_shutdown_function(function() use ($lock_key_cache, $lock_key_option, $config, $lock_acquired_method, $lock_value) {
			$released_cache = false;
			$released_db = false;
			$current_process_owned_db_lock = false; // Check if we owned the DB lock we might delete

			// Check if this process owned the DB lock before deleting
			// This helps prevent accidental deletion if another process acquired the lock after this one started
			// (especially relevant if stale lock override occurred). Check only needed for DB lock.
			$current_db_lock_val = (int) get_option($lock_key_option, 0);
			if ($current_db_lock_val === $lock_value) {
				$current_process_owned_db_lock = true;
			} else if ($current_db_lock_val !== 0) {
				// Log if the lock value changed during execution - indicates potential issue or race condition
				Logging::log_warning(
					sprintf("[Retry Cron Shutdown] DB lock ('%s') value changed during execution (expected: %d, found: %d). Lock not released by this process.", $lock_key_option, $lock_value, $current_db_lock_val),
					'warning'
				);
			}

			// Release cache lock if cache was enabled (safer to attempt release regardless of which lock type was acquired).
			if (($config['cache_enabled'] ?? false) && wp_using_ext_object_cache()) {
				wp_cache_delete($lock_key_cache);
				$released_cache = true;
			}

			// Only release the DB option lock if this process successfully set it and it hasn't been overwritten.
			if ($current_process_owned_db_lock) {
				delete_option($lock_key_option);
				$released_db = true;
			}

			Logging::log_debug(
				sprintf("[Retry Cron Shutdown] Acquired via: %s. Attempted Release - Cache: %s, DB: %s (Owned: %s).",
					$lock_acquired_method,
					($config['cache_enabled'] ?? false) ? ($released_cache ? 'yes' : 'no') : 'disabled',
					$released_db ? 'yes' : 'no',
					$current_process_owned_db_lock ? 'yes' : 'no'
				),
				'debug'
			);
		});

		// 5. Check for graceful shutdown signal *before* starting intensive work
		// Assumes self::$signalShutdown is a static property managed elsewhere
		if (self::$signalShutdown) {
			Logging::log_info("[Retry Cron] Shutdown signal detected before processing started. Releasing lock (via shutdown function) and exiting.", 'info');
			// The registered shutdown function will handle the lock release.
			return;
		}

		// 6. Execute the main processing logic
		Logging::log_debug("[Retry Cron] Lock acquired (Method: {$lock_acquired_method}). Starting processing.", 'debug');
		try {
			$this->process_all_retries();
			Logging::log_debug("[Retry Cron] Processing finished successfully.", 'debug');
		} catch (\Throwable $e) {
			// Catch critical errors/exceptions during the main processing
			Logging::log_error("[Retry Cron] Uncaught exception during process_all_retries: " . $e->getMessage(), 'critical', [
				'exception_type' => get_class($e),
				'exception_file' => $e->getFile(),
				'exception_line' => $e->getLine(),
				'exception_trace' => $e->getTraceAsString(), // Consider limiting trace length
			]);
		}

		// Note: The shutdown function registered above will automatically release the lock.
		Logging::log_debug("[Retry Cron] cron_callback finished. Lock release handled by shutdown handler.", 'debug');
	}
	/**
	 * Sets up the WP-Cron schedule and action hook for the retry processor.
	 * Ensures custom intervals are available.
	 * 
	 * NOTE: Skipped when Action Scheduler handles retries natively.
	 */
	public function setup_cron(): void {
		// Skip if Action Scheduler handles retries natively - zero performance impact
		if ( ! $this->should_handle_retries() ) {
			$this->logger->log_debug( '[Retry] Skipping setup_cron - Action Scheduler handles retries natively' );
			// Also remove any existing cron to clean up
			$this->remove_cron();
			return;
		}
		
		// Add custom cron schedules if they don't exist
		add_filter('cron_schedules', function($schedules) {
			if (!isset($schedules['one_minute'])) {
				$schedules['one_minute'] = ['interval' => 60, 'display' => __('Every Minute', 'your-text-domain')];
			}
			if (!isset($schedules['five_minutes'])) {
				$schedules['five_minutes'] = ['interval' => 300, 'display' => __('Every 5 Minutes', 'your-text-domain')];
			}
			// Add other intervals if needed
			return $schedules;
		});

		// Ensure the cron action hook is added
		if (!has_action(self::CRON_HOOK, [$this, 'cron_callback'])) {
			add_action(self::CRON_HOOK, [$this, 'cron_callback']);
			Logging::log_debug("[Retry Cron] Added WP Cron action hook: " . self::CRON_HOOK, 'debug');
		}

		// Ensure the event is scheduled (idempotent)
		$this->schedule_retry_processor_event();
	}


	/**
	 * Removes the WP-Cron schedule and action hook for the retry processor.
	 */
	public function remove_cron(): void {
		// Clear any scheduled instances of the cron hook.
		// wp_clear_scheduled_hook() returns the number of events cleared, false on error, or 0 if none were scheduled.
		$num_unscheduled = wp_clear_scheduled_hook( self::CRON_HOOK );

		if ( false === $num_unscheduled ) {
			// An error occurred trying to clear the schedule.
			Logging::log_error( "[Retry Cron] Error attempting to clear scheduled WP Cron hook: " . self::CRON_HOOK );
		} elseif ( 0 === $num_unscheduled ) {
			// No scheduled hooks were found to clear.
			Logging::log_debug( "[Retry Cron] No scheduled WP Cron hook found to clear for: " . self::CRON_HOOK, 'debug' );
		} else {
			// Successfully cleared one or more scheduled hooks.
			Logging::log_debug( "[Retry Cron] Cleared {$num_unscheduled} scheduled WP Cron hook(s): " . self::CRON_HOOK, 'debug' );
		}
	}
	/**
	 * Schedules the retry processor WP-Cron event if it's not already scheduled.
	 * Uses the interval defined in the configuration or a default.
	 */
	public function schedule_retry_processor_event(): void {
		if (!wp_next_scheduled(self::CRON_HOOK)) {
			$config = $this->get_retry_config();
			$default_interval = self::DEFAULT_CRON_INTERVAL; // e.g., 'one_minute'

			/**
			 * Filters the cron interval used for the retry processor.
			 *
			 * @param string $interval Default interval name (from config or class default).
			 * @return string Valid WP Cron schedule name (e.g., 'one_minute', 'hourly').
			 */
			$interval_name = apply_filters('sha_retry_cron_interval_ludicrous_sc', $config['cron_interval'] ?? $default_interval);

			// Validate the interval name against available schedules
			$schedules = wp_get_schedules();
			if (!isset($schedules[$interval_name])) {
				Logging::log_warning(
					sprintf("[Retry Cron] Invalid interval '%s' configured or filtered. Falling back to '%s'.", $interval_name, $default_interval),
					'warning'
				);
				$interval_name = $default_interval;
				// Ensure the fallback interval exists
				if (!isset($schedules[$interval_name])) {
					Logging::log_error("[Retry Cron] CRITICAL: Default interval '{$interval_name}' not found. Falling back to 'hourly'.", 'critical');
					$interval_name = 'hourly'; // Absolute fallback
				}
			}

			// Schedule the event to run shortly (e.g., 5 seconds from now)
			wp_schedule_event(time() + 5, $interval_name, self::CRON_HOOK);

			Logging::log_info(
				sprintf('[Retry Cron] Scheduled WP Cron event "%s" with interval "%s".', self::CRON_HOOK, $interval_name),
				'info'
			);
		} else {
			Logging::log_debug("[Retry Cron] Event '" . self::CRON_HOOK . "' already scheduled.", 'debug');
		}
	}

	/**
	 * Unschedules the retry processor WP-Cron event if it is scheduled.
	 */
	public function unschedule_retry_processor_event(): void {
		$timestamp = wp_next_scheduled(self::CRON_HOOK);
		if ($timestamp) {
			wp_unschedule_event($timestamp, self::CRON_HOOK);
			Logging::log_info(
				sprintf('[Retry Cron] Unscheduled WP Cron event "%s" (was scheduled for %s).', self::CRON_HOOK, date('Y-m-d H:i:s', $timestamp)),
				'info'
			);
		} else {
			Logging::log_debug("[Retry Cron] Event '" . self::CRON_HOOK . "' was not scheduled. No action taken.", 'debug');
		}
	}

	// =========================================================================
	// Helper & Utility Functions (Existing or Assumed Needed)
	// =========================================================================

	/**
	 * Get the full name of the main retry table.
	 * @return string Table name with prefix.
	 */
	protected function get_retry_table_name(): string {
		return $this->wpdb->prefix . self::RETRY_TABLE_BASENAME;
	}

	/**
	 * Get the full name of the history table.
	 * @return string Table name with prefix.
	 */
	protected function get_history_table_name(): string {
		global $wpdb;
		return $wpdb->prefix . self::RETRY_HISTORY_TABLE_BASENAME;
	}

	/**
	 * Get the full name of the Dead Letter Queue table.
	 * @return string Table name with prefix.
	 */
	protected function get_dlq_table_name(): string {
		global $wpdb;
		return $wpdb->prefix . self::RETRY_DLQ_TABLE_BASENAME;
	}

	/**
	 * Calculate the delay in seconds before the next attempt based on strategy and counts.
	 * Applies jitter based on configuration.
	 *
	 * @param array $job The job data array (must contain strategy, counts, delays).
	 * @return int Delay in seconds.
	 */
	protected function calculate_delay(array $job): int {
		$config = $this->get_retry_config();
		$strategy = $job['retry_strategy'] ?? $config['default_strategy'];
		// Note: retry_count reflects *completed* attempts. We calculate delay for the *next* one.
		$attempt_number = (int) ($job['retry_count'] ?? 0); // Next attempt is count + 1
		$base_delay = (int) ($job['base_delay_sec'] ?? $config['initial_delay']);
		$max_delay = ($job['max_delay_sec'] !== null) ? (int)$job['max_delay_sec'] : null;
		$jitter_factor = (float) ($config['jitter_factor'] ?? 0.2); // Default 20% jitter

		$delay = $base_delay; // Start with base delay

		switch ($strategy) {
			case self::STRATEGY_EXPONENTIAL:
				$factor = (float) ($job['backoff_factor'] ?? $config['backoff_factor']);
				$factor = max(1.0, $factor); // Ensure factor is at least 1.0
				// Exponential calculation: base * factor ^ attempt_number
				$delay = $base_delay * pow($factor, $attempt_number);
				break;

			case self::STRATEGY_LINEAR:
				// Linear calculation: base + (base * attempt_number)
				// Or simply: base * (1 + attempt_number)
				$delay = $base_delay * (1 + $attempt_number);
				break;

			case self::STRATEGY_FIXED:
				// Fixed delay, always use base_delay
				$delay = $base_delay;
				break;

			case self::STRATEGY_NONE:
			default:
				// No retry planned, but calculate initial delay anyway if needed.
				// Effectively same as fixed for the first (and only) potential delay.
				$delay = $base_delay;
				break;
		}

		// Apply Max Delay Cap (if set and positive)
		if ($max_delay !== null && $max_delay > 0) {
			$delay = min($delay, $max_delay);
		}

		// Ensure delay is at least 1 second (or 0 if specifically calculated that way)
		$delay = max(1, (int) round($delay));

		// Apply Jitter: +/- jitter_factor % of the calculated delay
		if ($jitter_factor > 0 && $delay > 1) { // No jitter on 1s delay
			$jitter_amount = $delay * $jitter_factor;
			// Random float between -jitter_amount and +jitter_amount
			$random_jitter = (mt_rand() / mt_getrandmax()) * (2 * $jitter_amount) - $jitter_amount;
			$delay += $random_jitter;
			// Ensure delay doesn't go below 1 second after jitter
			$delay = max(1, (int) round($delay));
		}

		return (int) $delay;
	}

	/**
	 * Determines if an exception suggests a retry might be successful.
	 * Default implementation retries most generic Exceptions but not specific fatal errors.
	 * Can be extended via filters.
	 *
	 * @param \Throwable $exception The exception caught during execution.
	 * @return bool True if the exception is considered retryable, false otherwise.
	 */
	protected function should_retry_exception(\Throwable $exception): bool {
		$is_retryable = true; // Default assumption

		// --- Add specific non-retryable exception types here ---
		// Example: Don't retry ArgumentCountError, TypeError, ParseError etc.
		if ($exception instanceof \ArgumentCountError ||
		    $exception instanceof \TypeError ||
		    $exception instanceof \ParseError ||
		    $exception instanceof \DomainException || // Often indicates invalid input
		    $exception instanceof \InvalidArgumentException || // Often indicates invalid input
		    $exception instanceof \LengthException) { // Often data-related
			$is_retryable = false;
		}

		// --- Example: Non-retryable based on error code (if applicable) ---
		// $code = $exception->getCode();
		// if (is_numeric($code) && $code >= 400 && $code < 500 && $code !== 429) {
		//     // Treat client errors (4xx, except 429 Too Many Requests) as non-retryable
		//     $is_retryable = false;
		// }

		/**
		 * Filters whether an exception should be considered retryable.
		 *
		 * @param bool       $is_retryable The default determination based on exception type/code.
		 * @param \Throwable $exception    The exception object caught.
		 * @return bool Return true to retry, false to mark as non-retryable (will lead to DLQ if other conditions met).
		 */
		return apply_filters('sha_retry_should_retry_exception_lc', $is_retryable, $exception);
	}

	/**
	 * Registers an executor function for a specific operation type.
	 *
	 * @param string   $operation_type The identifier for the operation.
	 * @param callable $executor       The callable function/method to execute the job.
	 *                                 Signature: function(array $payload, array $metadata, array &$state, callable $heartbeat)
	 *                                 Should throw exceptions on failure. Can modify $state by reference.
	 * @param bool     $overwrite      Allow overwriting an existing executor for the type.
	 * @return bool True on success, false if type already registered and overwrite is false.
	 */
	public function register_executor(string $operation_type, callable $executor, bool $overwrite = false): bool {
		if (!$overwrite && isset(self::$executors[$operation_type])) {
			Logging::log_warning("[Retry] Executor for type '{$operation_type}' already registered. Not overwriting.", 'warning');
			return false;
		}
		self::$executors[$operation_type] = $executor;
		Logging::log_debug("[Retry] Registered executor for type '{$operation_type}'.", 'debug');
		return true;
	}

	/**
	 * Retrieves the registered executor for a given operation type.
	 *
	 * @param string $operation_type The operation type identifier.
	 * @return callable|null The registered callable executor, or null if not found.
	 */
	protected function get_executor(string $operation_type): ?callable {
		return self::$executors[$operation_type] ?? null;
	}
	/**
	 * Initializes the Retry system.
	 * Checks schema, sets up processor ID.
	 * Should be called once, e.g., during plugin initialization (`plugins_loaded` or similar).
	 */
	public function init(): void {
		$this->check_schema_version(); // Check if DB schema matches code
		self::$processorId = $this->generate_processor_id(); // Generate unique ID for this PHP process instance

		// Signal handling (removed for brevity, see original for context if needed for CLI)

		Logging::log_debug("[Retry] Initialized. Processor ID: " . self::$processorId, 'debug');
	}

	// ========================================================================
	// RetryInterface Implementation - Production Methods
	// ========================================================================

	/**
	 * Retry a failed job
	 * 
	 * @param mixed $asset_id
	 * @return mixed
	 */
	public function retry_failed_job($asset_id) {
		$asset_id = absint( $asset_id );
		
		if ( $asset_id <= 0 ) {
			$this->logger->log_warning( 'Invalid asset ID provided to retry_failed_job' );
			return false;
		}

		// Get the retry table name
		$retry_table = $this->wpdb->prefix . self::RETRY_TABLE_BASENAME;
		
		// Find the failed job
		// Optimized to select only needed columns
		$job = $this->wpdb->get_row(
			$this->wpdb->prepare(
				"SELECT id, asset_id, original_url, type, status, priority, attempts, scheduled_at, last_error, metadata FROM {$retry_table} WHERE id = %d AND status = %s LIMIT 1",
				$asset_id,
				self::STATUS_FAILED
			),
			ARRAY_A
		);

		if ( ! $job ) {
			$this->logger->log_warning( sprintf( 'No failed job found for asset ID: %d', $asset_id ) );
			return false;
		}

		// Reset the job to pending status
		$result = $this->wpdb->update(
			$retry_table,
			[
				'status'          => self::STATUS_PENDING,
				'retry_count'     => 0,
				'next_attempt_at' => current_time( 'mysql', 1 ),
			],
			[ 'id' => $asset_id ],
			[ '%s', '%d', '%s' ],
			[ '%d' ]
		);

		if ( $result !== false ) {
			$this->logger->log_info( sprintf( 'Successfully reset job %d for retry', $asset_id ) );
			return true;
		}

		$this->logger->log_error( sprintf( 'Failed to reset job %d for retry', $asset_id ) );
		return false;
	}

	/**
	 * Get jobs based on criteria
	 * 
	 * @param array $args
	 * @return array
	 */
	public function get_jobs(array $args): array {
		$retry_table = $this->wpdb->prefix . self::RETRY_TABLE_BASENAME;
		
		$defaults = [
			'status'   => null,
			'limit'    => 100,
			'offset'   => 0,
			'order_by' => 'next_attempt_at',
			'order'    => 'ASC',
		];
		
		$args = wp_parse_args( $args, $defaults );
		
		$where = '1=1';
		$where_values = [];
		
		if ( ! empty( $args['status'] ) ) {
			$where .= ' AND status = %s';
			$where_values[] = $args['status'];
		}
		
		$order_by = Sanitize::sanitize_key( $args['order_by'] );
		$order = strtoupper( $args['order'] ) === 'DESC' ? 'DESC' : 'ASC';
		$limit = absint( $args['limit'] );
		$offset = absint( $args['offset'] );
		
		// Optimized: Select only needed columns instead of SELECT *
		$query = "SELECT id, processor_id, status, retry_count, max_retries, next_retry_at, last_error, payload, created_at, updated_at, expires_at FROM {$retry_table} WHERE {$where} ORDER BY {$order_by} {$order} LIMIT {$limit} OFFSET {$offset}";
		
		if ( ! empty( $where_values ) ) {
			$query = $this->wpdb->prepare( $query, ...$where_values );
		}
		
		// phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared
		$results = $this->wpdb->get_results( $query, ARRAY_A );
		
		return is_array( $results ) ? $results : [];
	}

	/**
	 * Cancel a job
	 * 
	 * @param mixed $job_id
	 * @return mixed
	 */
	public function cancel_job($job_id) {
		$job_id = absint( $job_id );
		
		if ( $job_id <= 0 ) {
			$this->logger->log_warning( 'Invalid job ID provided to cancel_job' );
			return false;
		}

		$retry_table = $this->wpdb->prefix . self::RETRY_TABLE_BASENAME;
		
		// Update job status to cancelled (or move to DLQ)
		$result = $this->wpdb->update(
			$retry_table,
			[
				'status' => self::STATUS_FAILED,
			],
			[ 'id' => $job_id ],
			[ '%s' ],
			[ '%d' ]
		);

		if ( $result !== false ) {
			$this->logger->log_info( sprintf( 'Successfully cancelled job %d', $job_id ) );
			return true;
		}

		$this->logger->log_error( sprintf( 'Failed to cancel job %d', $job_id ) );
		return false;
	}

    // ========================================
    // STATIC PROXY METHODS FOR BACKWARD COMPATIBILITY
    // ========================================
    
    private static ?\LHA\Interfaces\RetryInterface $staticInstance = null;
    
    private static function getStaticInstance(): \LHA\Interfaces\RetryInterface {
        if (self::$staticInstance === null) {
            global $lha_container;
            if ($lha_container) {
                self::$staticInstance = $lha_container->get(\LHA\Interfaces\RetryInterface::class);
            } else {
                throw new \RuntimeException('LHA Container not initialized. Cannot use static methods.');
            }
        }
        return self::$staticInstance;
    }
    
    // Explicit static wrappers for commonly used methods
    public static function static_enqueue_retry(array $data) {
        return self::getStaticInstance()->enqueue_retry($data);
    }
    
    public static function static_get_pending_retries(int $limit = 100): array {
        return self::getStaticInstance()->get_pending_retries($limit);
    }
    
    public static function static_process_retry(int $retry_id): bool {
        return self::getStaticInstance()->process_retry($retry_id);
    }
    
    public static function static_remove_retry_operation(int $job_id, ?string $lock_token = null): bool {
        return self::getStaticInstance()->remove_retry_operation($job_id, $lock_token);
    }
    
    public static function static_get_retry_stats(): array {
        return self::getStaticInstance()->get_retry_stats();
    }
    
    public static function static_cleanup_old_retries(int $days = 30): int {
        return self::getStaticInstance()->cleanup_old_retries($days);
    }
    
    // Static proxy using __callStatic magic method
    // NOTE: This only works for methods that don't exist as instance methods.
    // For methods that exist as instance methods, use the explicit static_ prefixed methods above
    // or get the instance from the container.
    public static function __callStatic(string $method, array $arguments) {
        return self::getStaticInstance()->$method(...$arguments);
    }
    
    /**
     * Cleanup orphaned files when a retry job fails permanently
     * 
     * This method is called when a retry job fails permanently (moved to DLQ)
     * to ensure any partially downloaded or orphaned files are removed.
     * 
     * @param int $job_id The retry job ID that failed
     * @param array $job_data The job data containing operation details
     * @return bool True if cleanup was attempted, false if cleanup service unavailable
     */
    protected function cleanup_failed_retry_files(int $job_id, array $job_data): bool {
        if (!$this->cleanup) {
            $this->logger->log_debug('Cleanup service not available for failed retry cleanup', [
                'job_id' => $job_id
            ]);
            return false;
        }
        
        // Check if this is an asset-related retry
        if (empty($job_data['operation_data'])) {
            return false;
        }
        
        $operation_data = $job_data['operation_data'];
        
        // If retry has an associated asset_id or related_id that refers to an asset
        $asset_id = null;
        
        if (!empty($operation_data['asset_id'])) {
            $asset_id = absint($operation_data['asset_id']);
        } elseif (!empty($operation_data['related_id']) && 
                  !empty($operation_data['related_type']) && 
                  $operation_data['related_type'] === 'asset') {
            $asset_id = absint($operation_data['related_id']);
        }
        
        if ($asset_id && $asset_id > 0) {
            $this->logger->log_info('Cleaning up file for failed retry job', [
                'job_id' => $job_id,
                'asset_id' => $asset_id
            ]);
            
            $this->cleanup->delete_asset_file($asset_id);
            return true;
        }
        
        $this->logger->log_debug('No asset file to cleanup for failed retry job', [
            'job_id' => $job_id,
            'operation_data' => $operation_data
        ]);
        
        return false;
    }

    /**
     * Retry multiple failed jobs in a single batch operation
     * OPTIMIZATION: Reduces N queries to 2-3 queries for bulk operations
     * 
     * @param array $asset_ids Array of asset IDs to retry
     * @return array ['success' => int, 'failed' => int, 'errors' => array]
     */
    public function retry_failed_jobs_bulk(array $asset_ids): array {
        $result = [
            'success' => 0,
            'failed' => 0,
            'errors' => []
        ];
        
        if (empty($asset_ids)) {
            return $result;
        }
        
        try {
            // Sanitize and validate asset IDs
            $asset_ids = array_unique(array_filter(array_map('absint', $asset_ids), function($id) {
                return $id > 0;
            }));
            
            if (empty($asset_ids)) {
                return $result;
            }
            
            // Get table names directly
            $tasks_table = $this->wpdb->prefix . 'lha_tasks';
            $mappings_table = $this->wpdb->prefix . 'lha_mappings';
            
            // Fetch all failed tasks in a single query
            $placeholders = implode(',', array_fill(0, count($asset_ids), '%d'));
            $query = $this->wpdb->prepare(
                "SELECT task_id, asset_id, original_url, type, attempts 
                 FROM `{$tasks_table}` 
                 WHERE asset_id IN ({$placeholders}) 
                 AND status = 'failed'",
                $asset_ids
            );
            
            $failed_tasks = $this->wpdb->get_results($query, ARRAY_A);
            
            if (empty($failed_tasks)) {
                $result['failed'] = count($asset_ids);
                $result['errors'][] = 'No failed tasks found for the provided asset IDs';
                return $result;
            }
            
            // Prepare batch update for tasks
            $task_ids = array_column($failed_tasks, 'task_id');
            $task_ids_list = implode(',', array_map('absint', $task_ids));
            $current_time = current_time('mysql', true);
            $next_attempt = date('Y-m-d H:i:s', strtotime('+5 minutes'));
            
            // Update all failed tasks to retry status (batch update)
            $update_query = $this->wpdb->prepare(
                "UPDATE `{$tasks_table}` 
                 SET status = 'retry',
                     next_attempt_at = %s,
                     updated_at = %s
                 WHERE task_id IN ({$task_ids_list})",
                $next_attempt,
                $current_time
            );
            
            $updated = $this->wpdb->query($update_query);
            
            if ($updated !== false) {
                $result['success'] = $updated;
                
                // Update corresponding assets to retry status (batch update)
                $asset_ids_from_tasks = array_column($failed_tasks, 'asset_id');
                if (!empty($asset_ids_from_tasks)) {
                    $asset_ids_list = implode(',', array_map('absint', $asset_ids_from_tasks));
                    $this->wpdb->query(
                        $this->wpdb->prepare(
                            "UPDATE `{$mappings_table}` 
                             SET status = 'retry', updated_at = %s 
                             WHERE id IN ({$asset_ids_list})",
                            $current_time
                        )
                    );
                }
                
                // Clear relevant caches
                foreach ($failed_tasks as $task) {
                    wp_cache_delete('lha_asset_' . $task['asset_id'], 'lha_assets');
                    wp_cache_delete('lha_task_' . $task['task_id'], 'lha_tasks');
                }
                wp_cache_delete('lha_all_assets_api_cache_v2', 'lha_assets');
                
            } else {
                $result['failed'] = count($failed_tasks);
                $result['errors'][] = 'Database error during batch update: ' . $this->wpdb->last_error;
                $this->logger->log_error('Bulk retry batch update failed: ' . $this->wpdb->last_error);
            }
            
            // Calculate failed count
            $result['failed'] += (count($asset_ids) - count($failed_tasks));
            
        } catch (\Throwable $e) {
            $result['failed'] = count($asset_ids);
            $result['errors'][] = 'Exception during bulk retry: ' . $e->getMessage();
            $this->logger->log_error('Bulk retry error: ' . $e->getMessage());
        }
        
        return $result;
    }
}

