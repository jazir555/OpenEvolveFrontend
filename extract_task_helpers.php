<?php
/**
 * Task Helper Extraction Script
 * Parses Tasks.php.backup and extracts methods into helper classes
 */

$inputFile = __DIR__ . '/Tasks.php.backup';
$helpersDir = __DIR__ . '/TaskHelpers/';

// Method to helper mapping based on functionality
$methodHelpers = [
    // Cache operations
    'get_transient_via_cache' => 'TaskCacheHelper',
    'set_transient_via_cache' => 'TaskCacheHelper',
    'delete_transient_via_cache' => 'TaskCacheHelper',
    'warm_caches' => 'TaskCacheHelper',
    'invalidate_task_count_cache' => 'TaskCacheHelper',
    'track_query_performance' => 'TaskCacheHelper',
    'track_batch_metrics' => 'TaskCacheHelper',

    // Cron operations
    'schedule_database_retry' => 'TaskCronHelper',
    'manage_cron_events' => 'TaskCronHelper',
    'schedule_cron_event' => 'TaskCronHelper',
    'execute_cron_tasks' => 'TaskCronHelper',
    'delete_cron_lock' => 'TaskCronHelper',
    'handle_schedule_change' => 'TaskCronHelper',
    'reschedule_cron_event' => 'TaskCronHelper',
    'clear_scheduled_cron_events' => 'TaskCronHelper',
    'unschedule_cron_event' => 'TaskCronHelper',
    'add_five_minute_cron_schedule' => 'TaskCronHelper',
    'get_cron_hook' => 'TaskCronHelper',

    // Enqueue operations
    'enqueue_task' => 'TaskEnqueueHelper',
    'enqueue_svg_processing_task' => 'TaskEnqueueHelper',
    'enqueue_asset_task' => 'TaskEnqueueHelper',
    'enqueue_asset_task_by_id' => 'TaskEnqueueHelper',
    'enqueue_reprocess_task' => 'TaskEnqueueHelper',
    'enqueue_reprocess_tasks_bulk' => 'TaskEnqueueHelper',
    'enqueue_asset_tasks_bulk' => 'TaskEnqueueHelper',
    'batch_enqueue_tasks' => 'TaskEnqueueHelper',
    'enqueue_task_immediately' => 'TaskEnqueueHelper',
    'schedule_task_processing' => 'TaskEnqueueHelper',
    'schedule_task_processing_via_cron' => 'TaskEnqueueHelper',
    'ensure_batch_processor_scheduled' => 'TaskEnqueueHelper',
    'ensure_batch_processor_scheduled_public' => 'TaskEnqueueHelper',

    // Maintenance operations
    'schedule_daily_maintenance' => 'TaskMaintenanceHelper',
    'daily_maintenance_callback' => 'TaskMaintenanceHelper',
    'batch_delete_old_tasks' => 'TaskMaintenanceHelper',
    'optimize_database_tables' => 'TaskMaintenanceHelper',
    'verify_task_indexes' => 'TaskMaintenanceHelper',
    'cleanup_individual_task_crons' => 'TaskMaintenanceHelper',
    'reset_stuck_tasks' => 'TaskMaintenanceHelper',
    'refresh_asset_caches' => 'TaskMaintenanceHelper',

    // Processing operations
    'process_task' => 'TaskProcessingHelper',
    'process_task_batch' => 'TaskProcessingHelper',
    'process_scheduled_task' => 'TaskProcessingHelper',
    'execute_delayed_task' => 'TaskProcessingHelper',
    'handle_delayed_js_task' => 'TaskProcessingHelper',
    'get_pending_tasks_batch' => 'TaskProcessingHelper',
    'get_pending_tasks_optimized' => 'TaskProcessingHelper',
    'get_stuck_tasks_optimized' => 'TaskProcessingHelper',

    // Query operations
    'get_task_table_name' => 'TaskQueryHelper',
    'get_pending_tasks' => 'TaskQueryHelper',
    'get_pending_asset_tasks' => 'TaskQueryHelper',
    'get_task_by_id' => 'TaskQueryHelper',
    'get_tasks_by_ids' => 'TaskQueryHelper',
    'get_last_task_id' => 'TaskQueryHelper',
    'get_pending_tasks_count' => 'TaskQueryHelper',
    'has_pending_tasks' => 'TaskQueryHelper',

    // Schedule operations (high-level scheduling)
    'increment_completed_tasks' => 'TaskScheduleHelper',
    'calculate_task_priority' => 'TaskScheduleHelper',
    'store_task_metadata' => 'TaskScheduleHelper',
    'topological_sort_tasks' => 'TaskScheduleHelper',

    // Scheduler operations (Action Scheduler integration)
    'get_processor_manager' => 'TaskSchedulerHelper',
    'is_using_action_scheduler' => 'TaskSchedulerHelper',
    'get_processor_status' => 'TaskSchedulerHelper',
    'should_use_external_retry' => 'TaskSchedulerHelper',
    'has_native_retry' => 'TaskSchedulerHelper',
    'are_tasks_in_progress' => 'TaskSchedulerHelper',

    // Status operations
    'update_task_status' => 'TaskStatusHelper',
    'update_task_fields' => 'TaskStatusHelper',
    'batch_update_task_status' => 'TaskStatusHelper',
    'check_task_timeout' => 'TaskStatusHelper',
    'map_task_status_to_human_readable' => 'TaskStatusHelper',

    // Utility operations
    'get_process' => 'TaskUtilityHelper',
    'get_config_value' => 'TaskUtilityHelper',
    'safely_unserialize_task' => 'TaskUtilityHelper',
    'is_js_task_with_delay' => 'TaskUtilityHelper',
    'is_valid_http_url' => 'TaskUtilityHelper',

    // Validation operations
    'validate_task_structure' => 'TaskValidationHelper',
    'is_task_enqueued' => 'TaskValidationHelper',
];

// Read the backup file
$content = file_get_contents($inputFile);
$lines = explode("\n", $content);

// Extract methods into their respective helpers
$extractedMethods = [];
$currentMethod = null;
$currentMethodContent = [];
$inMethod = false;
$braceCount = 0;
$methodStartLine = 0;

for ($i = 0; $i < count($lines); $i++) {
    $line = $lines[$i];

    // Check if this line starts a method
    if (preg_match('/^\s+(public|private|protected)\s+(static\s+)?function\s+([a-z_][a-z0-9_]*)\s*\(/i', $line, $matches)) {
        $methodName = $matches[3];

        // Save previous method if any
        if ($currentMethod !== null && isset($methodHelpers[$currentMethod])) {
            $helperClass = $methodHelpers[$currentMethod];
            if (!isset($extractedMethods[$helperClass])) {
                $extractedMethods[$helperClass] = [];
            }
            $extractedMethods[$helperClass][$currentMethod] = [
                'start_line' => $methodStartLine,
                'content' => implode("\n", $currentMethodContent)
            ];
        }

        // Start new method
        $currentMethod = $methodName;
        $methodStartLine = $i + 1;
        $currentMethodContent = [$line];
        $inMethod = true;
        $braceCount = 0;

        // Count opening braces in this line
        $braceCount += substr_count($line, '{');
        $braceCount -= substr_count($line, '}');
    } elseif ($inMethod && $currentMethod !== null) {
        // Continue collecting method content
        $currentMethodContent[] = $line;
        $braceCount += substr_count($line, '{');
        $braceCount -= substr_count($line, '}');

        // Method ends when brace count returns to 0 or negative
        if ($braceCount <= 0 && !empty(trim($line))) {
            // Save method
            if (isset($methodHelpers[$currentMethod])) {
                $helperClass = $methodHelpers[$currentMethod];
                if (!isset($extractedMethods[$helperClass])) {
                    $extractedMethods[$helperClass] = [];
                }
                $extractedMethods[$helperClass][$currentMethod] = [
                    'start_line' => $methodStartLine,
                    'content' => implode("\n", $currentMethodContent)
                ];
            }

            // Reset
            $currentMethod = null;
            $currentMethodContent = [];
            $inMethod = false;
            $braceCount = 0;
        }
    }
}

// Save last method if any
if ($currentMethod !== null && isset($methodHelpers[$currentMethod])) {
    $helperClass = $methodHelpers[$currentMethod];
    if (!isset($extractedMethods[$helperClass])) {
        $extractedMethods[$helperClass] = [];
    }
    $extractedMethods[$helperClass][$currentMethod] = [
        'start_line' => $methodStartLine,
        'content' => implode("\n", $currentMethodContent)
    ];
}

// Create helper files
foreach ($extractedMethods as $helperClass => $methods) {
    $fileName = $helpersDir . $helperClass . '.php';

    $fileContent = "<?php\n\n";
    $fileContent .= "declare(strict_types=1);\n\n";
    $fileContent .= "namespace LHA\\TaskHelpers;\n\n";
    $fileContent .= "/**\n";
    $fileContent .= " * Helper class $helperClass\n";
    $fileContent .= " * Extracted from Tasks.php.backup\n";
    $fileContent .= " * " . count($methods) . " methods\n";
    $fileContent .= " */\n";
    $fileContent .= "class $helperClass {\n\n";

    foreach ($methods as $methodName => $methodInfo) {
        $fileContent .= "    /**\n";
        $fileContent .= "     * Method: $methodName\n";
        $fileContent .= "     * Extracted from line {$methodInfo['start_line']}\n";
        $fileContent .= "     */\n";
        $fileContent .= $methodInfo['content'] . "\n\n";
    }

    $fileContent .= "}\n";

    file_put_contents($fileName, $fileContent);
    echo "Created: $fileName (" . count($methods) . " methods)\n";
}

echo "\nExtraction complete!\n";
echo "Total helper classes created: " . count($extractedMethods) . "\n";
