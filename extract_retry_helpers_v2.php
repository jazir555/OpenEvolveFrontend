<?php
/**
 * Script to extract Retry helper methods using line numbers
 */

$sourceFile = 'C:/Users/mmeadow/Documents/locallyhostassetsbackup/locallyhostassets/Retry.php';

// Read file line by line
$lines = file($sourceFile, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
if ($lines === false) {
    die("Failed to read source file\n");
}

echo "Total lines: " . count($lines) . "\n";

// Method extraction with line ranges (from grep output)
$methods = [
    // RetryOperationHelper
    'enqueue_retry' => [90, 133],
    'add_to_retry_queue' => [134, 147],
    'process_retry' => [180, 246],
    'process_all_retries' => [4221, 4364],
    'retry_failed_job' => [4832, 4886],
    'retry_failed_jobs_bulk' => [5074, 5178],
    'store_retry_job' => [1960, 2319],
    'retrieve_and_lock_ready_jobs' => [2348, 2914],
    'execute_retry_operation' => [2915, 3210],
    'update_heartbeat' => [3211, 3256],
    'log_history' => [3257, 3441],
    'reschedule_failed_job' => [3442, 3554],
    'check_poison_pill' => [3555, 3636],
    'remove_retry_operation' => [3637, 3684],
    'mark_as_failed' => [3965, 4023],
    'process_expired_jobs' => [3725, 3803],
    'promote_waiting_jobs' => [3804, 3964],
    'cleanup_failed_retry_files' => [5023, 5073],

    // RetryQueryHelper
    'get_pending_retries' => [148, 179],
    'get_retry_stats' => [247, 256],
    'get_retry_system_status' => [null, null], // Not found, skip
    'get_retry_config' => [1376, 1569],
    'get_processor_id' => [1922, 1959],
    'get_queue_stats' => [4130, 4154],
    'get_jobs' => [4887, 4931],
    'has_pending_retry_operations' => [3685, 3724],
    'get_retry_table_name' => [4652, 4659],
    'get_history_table_name' => [4660, 4668],
    'get_dlq_table_name' => [4669, 4680],
    'get_default_stats_array' => [4155, 4167],
    'query_queue_stats' => [4168, 4220],
    'get_current_utc_time' => [702, 707],
    'format_datetime_for_sql' => [708, 715],
    'clear_stats_cache' => [716, 723],
    'get_recently_locked_cache_key' => [724, 729],
    'get_job_lock_cache_key' => [730, 747],
    'get_active_job_count_for_group' => [2320, 2347],

    // RetryScheduleHelper
    'should_handle_retries' => [null, null], // Not found, skip
    'schedule_download_retry' => [438, 462],
    'cron_callback' => [4365, 4540],
    'setup_cron' => [4541, 4567],
    'remove_cron' => [4568, 4587],
    'schedule_retry_processor_event' => [4588, 4630],
    'unschedule_retry_processor_event' => [4631, 4651],
    'register_executor' => [4789, 4804],
    'get_executor' => [4805, 4812],
    'init' => [4813, 4831],
    'calculate_retry_delay' => [463, 475],
    'calculate_task_priority' => [476, 485],
    'enqueue_task' => [486, 493],
    'get_logging_enabled_flag' => [494, 503],
    'promote_dependent_jobs' => [504, 558],

    // RetryDatabaseHelper
    'cleanup_old_retries' => [257, 260],
    'cleanup_old_records' => [4024, 4129],
    'get_retry_table_definitions' => [559, 701],
    'generate_create_table_sql' => [846, 938],
    'create_retry_infrastructure_tables' => [939, 1128],
    'check_schema_version' => [1129, 1232],
    'determine_datetime_features' => [367, 437],
    'move_to_dlq' => [748, 845],
    'calculate_delay' => [4681, 4747],
    'should_retry_exception' => [4748, 4788],

    // RetryNoticeHelper
    'display_db_outdated_notice' => [1233, 1279],
    'display_code_outdated_notice' => [1280, 1321],
    'display_missing_constant_notice' => [1322, 1353],
    'init_schema_check' => [1354, 1375],

    // RetryUtilityHelper
    'generate_processor_id' => [1570, 1921],
    'normalize_url' => [null, null], // Not found, skip
];

// Extract methods
$extractedMethods = [];
foreach ($methods as $methodName => $range) {
    if ($range[0] === null) {
        echo "Skipping $methodName - not found\n";
        continue;
    }

    $startLine = $range[0] - 1; // Convert to 0-indexed
    $endLine = $range[1];

    $methodLines = [];
    for ($i = $startLine; $i < $endLine && $i < count($lines); $i++) {
        $methodLines[] = $lines[$i];
    }

    $extractedMethods[$methodName] = implode("\n", $methodLines);
    echo "Extracted $methodName (lines {$range[0]}-{$range[1]})\n";
}

echo "\nExtracted " . count($extractedMethods) . " methods\n";

// Helper class assignments
$helpers = [
    'RetryOperationHelper' => [
        'enqueue_retry', 'add_to_retry_queue', 'process_retry', 'process_all_retries',
        'retry_failed_job', 'retry_failed_jobs_bulk', 'store_retry_job',
        'retrieve_and_lock_ready_jobs', 'execute_retry_operation', 'update_heartbeat',
        'log_history', 'reschedule_failed_job', 'check_poison_pill',
        'remove_retry_operation', 'mark_as_failed', 'process_expired_jobs',
        'promote_waiting_jobs', 'cleanup_failed_retry_files'
    ],
    'RetryQueryHelper' => [
        'get_pending_retries', 'get_retry_stats', 'get_retry_config',
        'get_processor_id', 'get_queue_stats', 'get_jobs',
        'has_pending_retry_operations', 'get_retry_table_name',
        'get_history_table_name', 'get_dlq_table_name',
        'get_default_stats_array', 'query_queue_stats',
        'get_current_utc_time', 'format_datetime_for_sql',
        'clear_stats_cache', 'get_recently_locked_cache_key',
        'get_job_lock_cache_key', 'get_active_job_count_for_group'
    ],
    'RetryScheduleHelper' => [
        'schedule_download_retry', 'cron_callback', 'setup_cron',
        'remove_cron', 'schedule_retry_processor_event',
        'unschedule_retry_processor_event', 'register_executor',
        'get_executor', 'init', 'calculate_retry_delay',
        'calculate_task_priority', 'enqueue_task',
        'get_logging_enabled_flag', 'promote_dependent_jobs'
    ],
    'RetryDatabaseHelper' => [
        'cleanup_old_retries', 'cleanup_old_records',
        'get_retry_table_definitions', 'generate_create_table_sql',
        'create_retry_infrastructure_tables', 'check_schema_version',
        'determine_datetime_features', 'move_to_dlq',
        'calculate_delay', 'should_retry_exception'
    ],
    'RetryNoticeHelper' => [
        'display_db_outdated_notice', 'display_code_outdated_notice',
        'display_missing_constant_notice', 'init_schema_check'
    ],
    'RetryUtilityHelper' => [
        'generate_processor_id'
    ],
];

// Build helper files
$targetDir = 'C:/Users/mmeadow/Documents/locallyhostassetsbackup/classes/RetryHelpers';

foreach ($helpers as $helperClass => $methodNames) {
    echo "\n=== Creating $helperClass ===\n";

    $helperMethods = [];
    foreach ($methodNames as $methodName) {
        if (isset($extractedMethods[$methodName])) {
            $helperMethods[] = $extractedMethods[$methodName];
            echo "  - Added $methodName\n";
        } else {
            echo "  - WARNING: $methodName not found\n";
        }
    }

    if (empty($helperMethods)) {
        echo "  No methods to write!\n";
        continue;
    }

    // Build the helper file
    ob_start();
    echo "<?php\n\n";
    echo "declare(strict_types=1);\n\n";
    echo "namespace LHA\\RetryHelpers;\n\n";
    echo "use LHA\\Interfaces\\LoggerInterface;\n";
    echo "use LHA\\Interfaces\\DatabaseInterface;\n";
    echo "use LHA\\Interfaces\\TaskQueueInterface;\n";
    echo "use LHA\\Initialize;\n\n";
    echo "/**\n";
    echo " * Class $helperClass\n";
    echo " *\n";
    echo " * Extracted from Retry.php\n";
    echo " * Production Ready: Yes\n";
    echo " */\n";
    echo "class $helperClass\n";
    echo "{\n";

    // Add properties based on helper type
    if ($helperClass === 'RetryOperationHelper') {
        echo "    private LoggerInterface \$logger;\n";
        echo "    private DatabaseInterface \$database;\n";
        echo "    private \wpdb \$wpdb;\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger,\n";
        echo "        DatabaseInterface \$database,\n";
        echo "        \wpdb \$wpdb\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "        \$this->database = \$database;\n";
        echo "        \$this->wpdb = \$wpdb;\n";
        echo "    }\n\n";
    } elseif ($helperClass === 'RetryQueryHelper') {
        echo "    private LoggerInterface \$logger;\n";
        echo "    private DatabaseInterface \$database;\n";
        echo "    private \wpdb \$wpdb;\n";
        echo "    private TaskQueueInterface \$tasks;\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger,\n";
        echo "        DatabaseInterface \$database,\n";
        echo "        \wpdb \$wpdb,\n";
        echo "        TaskQueueInterface \$tasks\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "        \$this->database = \$database;\n";
        echo "        \$this->wpdb = \$wpdb;\n";
        echo "        \$this->tasks = \$tasks;\n";
        echo "    }\n\n";
    } elseif ($helperClass === 'RetryScheduleHelper') {
        echo "    private LoggerInterface \$logger;\n";
        echo "    private Initialize \$initialize;\n";
        echo "    private static \$executors = [];\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger,\n";
        echo "        Initialize \$initialize\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "        \$this->initialize = \$initialize;\n";
        echo "    }\n\n";
    } elseif ($helperClass === 'RetryDatabaseHelper') {
        echo "    private LoggerInterface \$logger;\n";
        echo "    private DatabaseInterface \$database;\n";
        echo "    private \wpdb \$wpdb;\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger,\n";
        echo "        DatabaseInterface \$database,\n";
        echo "        \wpdb \$wpdb\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "        \$this->database = \$database;\n";
        echo "        \$this->wpdb = \$wpdb;\n";
        echo "    }\n\n";
    } elseif ($helperClass === 'RetryNoticeHelper') {
        echo "    private LoggerInterface \$logger;\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "    }\n\n";
    } elseif ($helperClass === 'RetryUtilityHelper') {
        echo "    private LoggerInterface \$logger;\n";
        echo "    private DatabaseInterface \$database;\n";
        echo "    private \wpdb \$wpdb;\n\n";
        echo "    public function __construct(\n";
        echo "        LoggerInterface \$logger,\n";
        echo "        DatabaseInterface \$database,\n";
        echo "        \wpdb \$wpdb\n";
        echo "    ) {\n";
        echo "        \$this->logger = \$logger;\n";
        echo "        \$this->database = \$database;\n";
        echo "        \$this->wpdb = \$wpdb;\n";
        echo "    }\n\n";
    }

    // Add constants
    echo "    // Constants from Retry class\n";
    echo "    private const RETRY_TABLE_BASENAME = 'lha_retry_queue';\n";
    echo "    private const RETRY_HISTORY_TABLE_BASENAME = 'lha_retry_history';\n";
    echo "    private const RETRY_DLQ_TABLE_BASENAME = 'lha_retry_dlq';\n";
    echo "    private const STATUS_PENDING = 'pending';\n";
    echo "    private const STATUS_PROCESSING = 'processing';\n";
    echo "    private const STATUS_SCHEDULED = 'scheduled';\n";
    echo "    private const STATUS_WAITING = 'waiting';\n";
    echo "    private const PRIORITY_NORMAL = 50;\n\n";

    // Add methods
    foreach ($helperMethods as $methodCode) {
        echo "\n";
        echo $methodCode;
        echo "\n";
    }

    echo "}\n";

    $content = ob_get_clean();

    // Write to file
    $targetFile = $targetDir . '/' . $helperClass . '.php';
    $result = file_put_contents($targetFile, $content);
    if ($result === false) {
        echo "  ERROR: Failed to write $targetFile\n";
    } else {
        echo "  Created $targetFile (" . strlen($content) . " bytes)\n";
    }
}

echo "\n=== Extraction complete ===\n";
