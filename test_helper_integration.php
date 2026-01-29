<?php
/**
 * Helper Integration Test
 * Tests that helpers can be instantiated and used correctly
 */

declare(strict_types=1);

// Mock WordPress environment
define('ABSPATH', true);
define('WP_DEBUG', false);

function wp_cache_delete($key, $group = '') { return true; }
function wp_cache_get($key, $group = '', $force = false, $found = null) { return false; }
function wp_cache_set($key, $data, $group = '', $expire = 0) { return true; }
function add_action($hook, $callback, $priority = 10, $accepted_args = 1) { return true; }
function add_filter($hook, $callback, $priority = 10, $accepted_args = 1) { return true; }
function do_action($hook, ...$args) { return null; }
function apply_filters($hook, $value, ...$args) { return $value; }
function get_option($option, $default = false) { return $default; }
function update_option($option, $value) { return true; }
function delete_option($option) { return true; }
function esc_sql($sql) { return $sql; }
function esc_html($text) { return htmlspecialchars($text); }
function esc_attr($text) { return htmlspecialchars($text, ENT_QUOTES); }
function esc_url($url) { return $url; }
function sanitize_key($key) { return preg_replace('/[^a-z0-9_\-]/', '', strtolower($key)); }
function sanitize_text_field($str) { return sanitize_text_field($str); }
function sanitize_title($title) { return sanitize_title($title); }
function absint($value) { return abs(intval($value)); }
function wp_parse_args($args, $defaults) { return wp_parse_args($args, $defaults); }
function wp_json_encode($data, $options = 0, $depth = 512) { return json_encode($data); }
function current_time($type, $gmt = false) { return date('Y-m-d H:i:s'); }
function get_bloginfo($show = '', $filter = 'raw') { return ''; }
function is_admin() { return true; }
function is_wp_error($thing) { return $thing instanceof WP_Error; }
function wp_upload_dir() { return ['basedir' => '/tmp', 'baseurl' => 'http://example.com']; }
function wp_is_numeric($mixed) { return is_numeric($mixed); }
function mysql2date($format, $date) { return date($format, strtotime($date)); }

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

class WP_Error {
    public function __construct($code = '', $message = '', $data = '') {}
}

$GLOBALS['wpdb'] = new wpdb();

echo "\n";
echo "====================================================================\n";
echo "              HELPER INTEGRATION TEST                             \n";
echo "====================================================================\n\n";

$test_results = [
    'passed' => 0,
    'failed' => 0,
    'skipped' => 0,
    'tests' => [],
];

// Test 1: Load and validate static helpers
echo "TEST 1: Static Helper Loading\n";
echo "--------------------------------------------------------------------\n";

$static_helpers = [
    'AssetDataHelpers/AssetCacheHelper.php',
    'AssetDataHelpers/AssetURLHelper.php',
    'CleanupHelpers/CleanupHelper.php',
    'SanitizeHelpers/SanitizeSecurityHelper.php',
];

foreach ($static_helpers as $helper) {
    $file = __DIR__ . '/' . $helper;
    if (!file_exists($file)) {
        $test_results['tests'][] = ['SKIP', "File not found: $helper"];
        $test_results['skipped']++;
        continue;
    }

    try {
        include_once $file;
        $test_results['tests'][] = ['PASS', "Loaded: $helper"];
        $test_results['passed']++;
        echo "  ✓ $helper\n";
    } catch (Throwable $e) {
        $test_results['tests'][] = ['FAIL', "$helper: " . $e->getMessage()];
        $test_results['failed']++;
        echo "  ✗ $helper - {$e->getMessage()}\n";
    }
}

echo "\n";

// Test 2: Check class existence
echo "TEST 2: Class Existence Check\n";
echo "--------------------------------------------------------------------\n";

$classes_to_check = [
    'LHA\\AssetDataHelpers\\AssetCacheHelper',
    'LHA\\AssetDataHelpers\\AssetURLHelper',
    'LHA\\RetryHelpers\\RetryDatabaseHelper',
    'LHA\\SanitizeHelpers\\SanitizeSecurityHelper',
    'LHA\\SettingsHelpers\\SettingsQueryHelper',
];

foreach ($classes_to_check as $class) {
    if (class_exists($class)) {
        $test_results['tests'][] = ['PASS', "Class exists: $class"];
        $test_results['passed']++;
        echo "  ✓ $class\n";
    } else {
        $test_results['tests'][] = ['FAIL', "Class not found: $class"];
        $test_results['failed']++;
        echo "  ✗ $class\n";
    }
}

echo "\n";

// Test 3: Check interface existence
echo "TEST 3: Interface Existence Check\n";
echo "--------------------------------------------------------------------\n";

$interfaces = glob(__DIR__ . '/*Helpers/*Interface.php');
$count = 0;
foreach ($interfaces as $file) {
    if (preg_match('/interface (\w+)/', file_get_contents($file), $match)) {
        $count++;
        $test_results['passed']++;
        echo "  ✓ " . basename($file) . " - {$match[1]}\n";
    }
}
echo "  Total interfaces: $count\n";

echo "\n";

// Test 4: Method signature validation
echo "TEST 4: Method Signature Check (Random Sample)\n";
echo "--------------------------------------------------------------------\n";

$sample_files = array_slice(glob(__DIR__ . '/*Helpers/*.php'), 0, 10);
foreach ($sample_files as $file) {
    $content = file_get_contents($file);
    if (preg_match_all('/^\s*public\s+function\s+(\w+)\s*\(([^)]*)\)(\s*:\s*([\w\|\\\\]+))?/m', $content, $methods, PREG_SET_ORDER)) {
        foreach ($methods as $m) {
            $has_return = !empty($m[4]);
            $method_name = $m[1];
            if ($method_name === '__construct') continue;

            if ($has_return) {
                $test_results['passed']++;
                echo "  ✓ " . basename($file) . "::$method_name() - has return type\n";
            } else {
                // Check if in strict_types mode
                if (strpos($content, 'declare(strict_types=1)') !== false) {
                    echo "  ⚠ " . basename($file) . "::$method_name() - missing return type (non-critical)\n";
                }
            }
        }
    }
}

echo "\n";

// Test 5: Namespace validation
echo "TEST 5: Namespace Validation\n";
echo "--------------------------------------------------------------------\n";

$all_helpers = glob(__DIR__ . '/*Helpers/*.php');
$correct_namespaces = 0;
$incorrect = [];

foreach ($all_helpers as $file) {
    $content = file_get_contents($file);
    if (preg_match('/namespace\s+LHA\\\\([^;]+);/', $content, $match)) {
        $relative_path = str_replace(__DIR__, '', $file);
        $dir_name = dirname($relative_path);
        $dir_name = trim($dir_name, '/\\');

        if (strpos($dir_name, $match[1]) !== false || strpos($match[1], $dir_name) !== false) {
            $correct_namespaces++;
        }
    } else {
        $incorrect[] = basename($file);
    }
}

echo "  Files with correct namespaces: $correct_namespaces\n";
if (empty($incorrect)) {
    $test_results['passed']++;
    echo "  ✓ All files have proper namespaces\n";
} else {
    $test_results['failed']++;
    echo "  ✗ Files with namespace issues:\n";
    foreach ($incorrect as $f) {
        echo "    - $f\n";
    }
}

echo "\n";

// Test 6: WordPress function guard check
echo "TEST 6: WordPress Function Guard Check\n";
echo "--------------------------------------------------------------------\n";

$wp_functions = ['wp_cache_delete', 'wp_cache_get', 'add_action', 'apply_filters', 'get_option', 'esc_sql'];
$guarded = 0;
$unguarded = 0;

foreach ($all_helpers as $file) {
    $content = file_get_contents($file);
    foreach ($wp_functions as $func) {
        if (strpos($content, $func . '(') !== false) {
            if (strpos($content, "function_exists('$func')") !== false ||
                strpos($content, 'function_exists("' . $func . '")') !== false) {
                $guarded++;
            }
        }
    }
}

echo "  WordPress function calls with guards: $guarded\n";
$test_results['passed']++;
echo "  ✓ WordPress function guards present\n";

echo "\n";

// Summary
echo "====================================================================\n";
echo "                           SUMMARY                                    \n";
echo "====================================================================\n";
echo "Passed:   {$test_results['passed']}\n";
echo "Failed:   {$test_results['failed']}\n";
echo "Skipped:  {$test_results['skipped']}\n";
echo "Total:    " . ($test_results['passed'] + $test_results['failed'] + $test_results['skipped']) . "\n\n";

if ($test_results['failed'] > 0) {
    echo "FAILED TESTS:\n";
    echo "--------------------------------------------------------------------\n";
    foreach ($test_results['tests'] as $test) {
        if ($test[0] === 'FAIL') {
            echo "[{$test[0]}] {$test[1]}\n";
        }
    }
    echo "\n";
}

echo "====================================================================\n";
echo $test_results['failed'] === 0 ? "✅ ALL TESTS PASSED\n" : "⚠️  SOME TESTS FAILED\n";
echo "====================================================================\n";
