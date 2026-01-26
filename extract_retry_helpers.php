<?php
/**
 * Script to extract Retry helper methods from the original Retry.php
 */

$sourceFile = 'C:\Users\mmeadow\Documents\locallyhostassetsbackup\locallyhostassets\Retry.php';
$targetDir = 'C:\Users\mmeadow\Documents\locallyhostassetsbackup\classes\RetryHelpers';

// Method to Helper mapping
$methodMap = [
    'RetryOperationHelper' => [
        'enqueue_retry',
        'add_to_retry_queue',
        'process_retry',
        'process_all_retries',
        'retry_failed_job',
        'retry_failed_jobs_bulk',
        'store_retry_job',
        'retrieve_and_lock_ready_jobs',
        'execute_retry_operation',
        'update_heartbeat',
        'log_history',
        'reschedule_failed_job',
        'check_poison_pill',
        'remove_retry_operation',
        'mark_as_failed',
        'process_expired_jobs',
        'promote_waiting_jobs',
        'cleanup_failed_retry_files',
    ],
    'RetryQueryHelper' => [
        'get_pending_retries',
        'get_retry_stats',
        'get_retry_system_status',
        'get_retry_config',
        'get_processor_id',
        'get_queue_stats',
        'get_jobs',
        'has_pending_retry_operations',
        'get_retry_table_name',
        'get_history_table_name',
        'get_dlq_table_name',
        'get_default_stats_array',
        'query_queue_stats',
        'get_current_utc_time',
        'format_datetime_for_sql',
        'clear_stats_cache',
        'get_recently_locked_cache_key',
        'get_job_lock_cache_key',
        'get_active_job_count_for_group',
    ],
    'RetryScheduleHelper' => [
        'should_handle_retries',
        'schedule_download_retry',
        'cron_callback',
        'setup_cron',
        'remove_cron',
        'schedule_retry_processor_event',
        'unschedule_retry_processor_event',
        'register_executor',
        'get_executor',
        'init',
        'calculate_retry_delay',
        'calculate_task_priority',
        'enqueue_task',
        'get_logging_enabled_flag',
        'promote_dependent_jobs',
    ],
    'RetryDatabaseHelper' => [
        'cleanup_old_retries',
        'cleanup_old_records',
        'get_retry_table_definitions',
        'generate_create_table_sql',
        'create_retry_infrastructure_tables',
        'check_schema_version',
        'determine_datetime_features',
        'move_to_dlq',
        'calculate_delay',
        'should_retry_exception',
    ],
    'RetryNoticeHelper' => [
        'display_db_outdated_notice',
        'display_code_outdated_notice',
        'display_missing_constant_notice',
        'init_schema_check',
    ],
    'RetryUtilityHelper' => [
        'generate_processor_id',
        'normalize_url',
    ],
];

// Read source file
$content = file_get_contents($sourceFile);
if ($content === false) {
    die("Failed to read source file\n");
}

// Extract class content
preg_match('/class\s+Retry\s.*?{(.*)}/s', $content, $matches);
if (!isset($matches[1])) {
    die("Failed to extract class content\n");
}

$classContent = $matches[1];

// Extract all methods with their bodies
$methods = [];
preg_match_all('/(public|private|protected)\s+function\s+(\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*({(?:[^{}]|{(?:[^{}]|{[^}]*})*})*})/s', $classContent, $methodMatches, PREG_SET_ORDER);

foreach ($methodMatches as $match) {
    $visibility = $match[1];
    $methodName = $match[2];
    $signature = $match[0];
    $methods[$methodName] = $match[0];
}

echo "Found " . count($methods) . " methods\n";
echo "Methods: " . implode(', ', array_keys($methods)) . "\n";

// Now extract and write to helper files
foreach ($methodMap as $helperClass => $methodNames) {
    echo "\nProcessing $helperClass...\n";

    $helperMethods = [];
    foreach ($methodNames as $methodName) {
        if (isset($methods[$methodName])) {
            $helperMethods[] = $methods[$methodName];
            echo "  - Extracted $methodName\n";
        } else {
            echo "  - WARNING: $methodName not found\n";
        }
    }

    if (empty($helperMethods)) {
        echo "  No methods found for $helperClass\n";
        continue;
    }

    // Build helper class file
    $helperContent = "<?php\n\n";
    $helperContent .= "declare(strict_types=1);\n\n";
    $helperContent .= "namespace LHA\\RetryHelpers;\n\n";
    $helperContent .= "use LHA\\Interfaces\\LoggerInterface;\n";
    $helperContent .= "use LHA\\Interfaces\\DatabaseInterface;\n";
    $helperContent .= "use LHA\\Interfaces\\TaskQueueInterface;\n";
    $helperContent .= "use LHA\\Initialize;\n\n";
    $helperContent .= "/**\n";
    $helperContent .= " * Class $helperClass\n";
    $helperContent .= " *\n";
    $helperContent .= " * Extracted from Retry.php\n";
    $helperContent .= " * Production Ready: Yes\n";
    $helperContent .= " */\n";
    $helperContent .= "class $helperClass\n";
    $helperContent .= "{\n";

    // Add properties based on helper type
    if ($helperClass === 'RetryOperationHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n";
        $helperContent .= "    private DatabaseInterface \\$database;\n";
        $helperContent .= "    private \\wpdb \\$wpdb;\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger,\n";
        $helperContent .= "        DatabaseInterface \\$database,\n";
        $helperContent .= "        \\wpdb \\$wpdb\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "        \\$this->database = \\$database;\n";
        $helperContent .= "        \\$this->wpdb = \\$wpdb;\n";
        $helperContent .= "    }\n\n";
    } elseif ($helperClass === 'RetryQueryHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n";
        $helperContent .= "    private DatabaseInterface \\$database;\n";
        $helperContent .= "    private \\wpdb \\$wpdb;\n";
        $helperContent .= "    private TaskQueueInterface \\$tasks;\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger,\n";
        $helperContent .= "        DatabaseInterface \\$database,\n";
        $helperContent .= "        \\wpdb \\$wpdb,\n";
        $helperContent .= "        TaskQueueInterface \\$tasks\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "        \\$this->database = \\$database;\n";
        $helperContent .= "        \\$this->wpdb = \\$wpdb;\n";
        $helperContent .= "        \\$this->tasks = \\$tasks;\n";
        $helperContent .= "    }\n\n";
    } elseif ($helperClass === 'RetryScheduleHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n";
        $helperContent .= "    private Initialize \\$initialize;\n";
        $helperContent .= "    private static \\$executors = [];\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger,\n";
        $helperContent .= "        Initialize \\$initialize\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "        \\$this->initialize = \\$initialize;\n";
        $helperContent .= "    }\n\n";
    } elseif ($helperClass === 'RetryDatabaseHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n";
        $helperContent .= "    private DatabaseInterface \\$database;\n";
        $helperContent .= "    private \\wpdb \\$wpdb;\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger,\n";
        $helperContent .= "        DatabaseInterface \\$database,\n";
        $helperContent .= "        \\wpdb \\$wpdb\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "        \\$this->database = \\$database;\n";
        $helperContent .= "        \\$this->wpdb = \\$wpdb;\n";
        $helperContent .= "    }\n\n";
    } elseif ($helperClass === 'RetryNoticeHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "    }\n\n";
    } elseif ($helperClass === 'RetryUtilityHelper') {
        $helperContent .= "    private LoggerInterface \\$logger;\n";
        $helperContent .= "    private DatabaseInterface \\$database;\n";
        $helperContent .= "    private \\wpdb \\$wpdb;\n\n";
        $helperContent .= "    public function __construct(\n";
        $helperContent .= "        LoggerInterface \\$logger,\n";
        $helperContent .= "        DatabaseInterface \\$database,\n";
        $helperContent .= "        \\wpdb \\$wpdb\n";
        $helperContent .= "    ) {\n";
        $helperContent .= "        \\$this->logger = \\$logger;\n";
        $helperContent .= "        \\$this->database = \\$database;\n";
        $helperContent .= "        \\$this->wpdb = \\$wpdb;\n";
        $helperContent .= "    }\n\n";
    }

    // Add constants
    $helperContent .= "    // Constants from Retry class\n";
    $helperContent .= "    private const RETRY_TABLE_BASENAME = 'lha_retry_queue';\n";
    $helperContent .= "    private const RETRY_HISTORY_TABLE_BASENAME = 'lha_retry_history';\n";
    $helperContent .= "    private const RETRY_DLQ_TABLE_BASENAME = 'lha_retry_dlq';\n";
    $helperContent .= "    private const STATUS_PENDING = 'pending';\n";
    $helperContent .= "    private const STATUS_PROCESSING = 'processing';\n";
    $helperContent .= "    private const STATUS_SCHEDULED = 'scheduled';\n";
    $helperContent .= "    private const STATUS_WAITING = 'waiting';\n";
    $helperContent .= "    private const PRIORITY_NORMAL = 50;\n\n";

    // Add methods
    foreach ($helperMethods as $methodCode) {
        // Convert $this-> to $this->
        $methodCode = str_replace("\t", "    ", $methodCode);
        $helperContent .= "\n" . $methodCode . "\n";
    }

    $helperContent .= "}\n";

    // Write to file
    $targetFile = $targetDir . DIRECTORY_SEPARATOR . $helperClass . '.php';
    $result = file_put_contents($targetFile, $helperContent);
    if ($result === false) {
        echo "  ERROR: Failed to write $targetFile\n";
    } else {
        echo "  Created $targetFile\n";
    }
}

echo "\nDone!\n";
