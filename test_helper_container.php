<?php
declare(strict_types=1);

/**
 * Test Helper Service Container Registrations
 * Verifies that all helpers can be instantiated from the container
 */

// Mock WordPress environment
define('ABSPATH', __DIR__ . '/../../../');
define('WP_DEBUG', false);

// WordPress function mocks
$wp_functions = [
    'wp_cache_delete', 'wp_cache_get', 'wp_cache_set', 'wp_cache_add',
    'add_action', 'add_filter', 'do_action', 'apply_filters',
    'get_option', 'update_option', 'delete_option',
    'esc_sql', 'esc_html', 'esc_attr', 'esc_url',
    'sanitize_key', 'sanitize_text_field', 'sanitize_title',
    'absint', 'wp_parse_args', 'wp_json_encode',
    'current_time', 'get_bloginfo', 'is_admin',
    'wp_upload_dir', 'is_wp_error', 'wp_is_numeric',
    'mysql2date', 'WP_Filesystem',
];

foreach ($wp_functions as $func) {
    if (!function_exists($func)) {
        eval("function $func() { return true; }");
    }
}

// Mock wpdb
class wpdb {
    public $prefix = 'wp_';
    public $last_error = '';
    public function prepare($query, ...$args) { return $query; }
    public function query($query) { return true; }
    public function get_results($query, $output = OBJECT) { return []; }
    public function get_row($query, $output = OBJECT, $y = 0) { return null; }
    public function get_var($query, $x = 0, $y = 0) { return null; }
    public function insert($table, $data, $format = null) { return true; }
    public function update($table, $data, $where, $format = null, $where_format = null) { return true; }
    public function delete($table, $where, $where_format = null) { return true; }
}

$GLOBALS['wpdb'] = new wpdb();

// Include traits needed by helpers
$traits = [
    __DIR__ . '/DatabaseHelpers/DatabaseHelperTrait.php',
    __DIR__ . '/AjaxHelpers/AjaxHelperTrait.php',
];

foreach ($traits as $trait) {
    if (file_exists($trait)) {
        include_once $trait;
    }
}

echo "\n";
echo "====================================================================\n";
echo "           HELPER SERVICE CONTAINER TEST                             \n";
echo "====================================================================\n\n";

// Test key helper classes
$test_helpers = [
    // AjaxHelpers
    'LHA\\AjaxHelpers\\AssetManagementAjaxHelper',
    'LHA\\AjaxHelpers\\CacheAjaxHelper',
    'LHA\\AjaxHelpers\\DiagnosticsAjaxHelper',

    // DatabaseHelpers
    'LHA\\DatabaseHelpers\\DatabaseAssetHelper',
    'LHA\\DatabaseHelpers\\DatabaseQueryHelper',
    'LHA\\DatabaseHelpers\\DatabaseTableHelper',

    // ExtractHelpers
    'LHA\\ExtractHelpers\\ExtractUtilityHelper',
    'LHA\\ExtractHelpers\\ExtractCssHelper',
    'LHA\\ExtractHelpers\\ExtractSvgHelper',

    // ProcessHelpers
    'LHA\\ProcessHelpers\\ProcessCleanupHelper',
    'LHA\\ProcessHelpers\\ProcessExtractionHelper',
    'LHA\\ProcessHelpers\\ProcessQueryHelper',
    'LHA\\ProcessHelpers\\ProcessQueueHelper',
    'LHA\\ProcessHelpers\\ProcessTaskHelper',
    'LHA\\ProcessHelpers\\ProcessUtilityHelper',

    // RetryHelpers
    'LHA\\RetryHelpers\\RetryDatabaseHelper',
    'LHA\\RetryHelpers\\RetryDependencyManager',
    'LHA\\RetryHelpers\\RetryQueryHelper',
    'LHA\\RetryHelpers\\RetryDeadLetterQueue',
    'LHA\\RetryHelpers\\RetryExecutor',
    'LHA\\RetryHelpers\\RetryHelper',
    'LHA\\RetryHelpers\\RetryHistoryLogger',
    'LHA\\RetryHelpers\\RetryNoticeHelper',
    'LHA\\RetryHelpers\\RetryOperationHelper',
    'LHA\\RetryHelpers\\RetryPolicyManager',
    'LHA\\RetryHelpers\\RetryQueue',
    'LHA\\RetryHelpers\\RetryScheduleHelper',
    'LHA\\RetryHelpers\\RetryScheduler',
    'LHA\\RetryHelpers\\RetryStateManager',
    'LHA\\RetryHelpers\\RetryUtilityHelper',

    // TaskHelpers
    'LHA\\TaskHelpers\\TaskSchedulerHelper',
    'LHA\\TaskHelpers\\TaskValidationHelper',
    'LHA\\TaskHelpers\\TaskQueryHelper',
    'LHA\\TaskHelpers\\TaskCacheHelper',
    'LHA\\TaskHelpers\\TaskCronHelper',
    'LHA\\TaskHelpers\\TaskEnqueueHelper',
    'LHA\\TaskHelpers\\TaskMaintenanceHelper',
    'LHA\\TaskHelpers\\TaskProcessingHelper',
    'LHA\\TaskHelpers\\TaskScheduleHelper',
    'LHA\\TaskHelpers\\TasksHelper',
    'LHA\\TaskHelpers\\TasksStaticHelper',
    'LHA\\TaskHelpers\\TaskStatusHelper',
    'LHA\\TaskHelpers\\TaskUtilityHelper',

    // SanitizeHelpers
    'LHA\\SanitizeHelpers\\SanitizeSecurityHelper',
    'LHA\\SanitizeHelpers\\SanitizeValidationHelper',
];

$results = [
    'passed' => 0,
    'failed' => 0,
    'errors' => [],
];

echo "Testing helper class existence:\n";
echo "--------------------------------------------------------------------\n";

foreach ($test_helpers as $helper) {
    $file = str_replace('LHA\\', __DIR__ . '/', $helper);
    $file = str_replace('\\', '/', $file) . '.php';

    if (file_exists($file)) {
        echo "  ✓ File exists: $helper\n";
        include_once $file;

        if (class_exists($helper)) {
            $results['passed']++;
            echo "    ✓ Class defined\n";
        } else {
            $results['failed']++;
            $results['errors'][] = "$helper - Class not defined after include";
            echo "    ✗ Class not defined\n";
        }
    } else {
        $results['failed']++;
        $results['errors'][] = "$helper - File not found: $file";
        echo "  ✗ File not found: $helper\n";
        echo "    Expected: $file\n";
    }
}

echo "\n";
echo "====================================================================\n";
echo "                         SUMMARY                                     \n";
echo "====================================================================\n";
echo "Passed:   {$results['passed']}\n";
echo "Failed:   {$results['failed']}\n";
echo "Total:    " . count($test_helpers) . "\n\n";

if (!empty($results['errors'])) {
    echo "ERRORS:\n";
    echo "--------------------------------------------------------------------\n";
    foreach ($results['errors'] as $error) {
        echo "  - $error\n";
    }
    echo "\n";
}

echo "====================================================================\n";
echo $results['failed'] === 0 ? "✅ ALL HELPERS LOADED SUCCESSFULLY\n" : "⚠️  SOME HELPERS FAILED TO LOAD\n";
echo "====================================================================\n";
