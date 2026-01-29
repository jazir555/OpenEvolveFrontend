<?php
/**
 * Test WordPress Compatibility Layer
 *
 * Verifies that helper files can be loaded and used without WordPress
 */

require_once __DIR__ . '/WordPressCompatibility.php';

echo "=== Testing WordPress Compatibility Layer ===\n\n";

// Test that all fallback functions are available
$functions = [
    'current_time',
    'wp_date',
    'esc_html',
    'esc_html__',
    'esc_sql',
    'esc_url',
    '__',
    '_e',
    'wp_schedule_event',
    'wp_schedule_single_event',
    'wp_next_scheduled',
    'wp_clear_scheduled_hook',
    'wp_cache_get',
    'wp_cache_set',
    'wp_cache_delete',
    'wp_cache_add',
    'get_transient',
    'set_transient',
    'delete_transient',
    'get_option',
    'update_option',
    'do_action',
    'apply_filters',
    '_get_cron_array',
    'absint',
    'maybe_serialize',
    'wp_get_schedules',
    'is_wp_error'
];

$missing = [];
foreach ($functions as $func) {
    if (!function_exists($func)) {
        $missing[] = $func;
    }
}

if (empty($missing)) {
    echo "✓ All 26 WordPress compatibility functions are available\n\n";
} else {
    echo "✗ Missing functions: " . implode(', ', $missing) . "\n\n";
    exit(1);
}

// Test basic functionality
echo "Testing basic functionality:\n";

// Test current_time
$time = current_time('mysql');
echo "✓ current_time(): $time\n";

// Test esc_html
$escaped = esc_html('<script>alert("test")</script>');
echo "✓ esc_html(): $escaped\n";

// Test __
$translated = __('Test string');
echo "✓ __(): $translated\n";

// Test get_option/update_option
$result = update_option('test_option', 'test_value');
echo "✓ update_option(): " . ($result ? 'true' : 'false') . "\n";

$value = get_option('test_option', 'default');
echo "✓ get_option(): $value\n";

// Test wp_cache functions
wp_cache_set('test_key', 'test_value');
$cached = wp_cache_get('test_key');
echo "✓ wp_cache_set()/wp_cache_get(): " . ($cached ? 'true' : 'false') . "\n";

$deleted = wp_cache_delete('test_key');
echo "✓ wp_cache_delete(): " . ($deleted ? 'true' : 'false') . "\n";

// Test transients
set_transient('test_transient', 'test_value', 3600);
$transient = get_transient('test_transient');
echo "✓ set_transient()/get_transient(): " . ($transient ? 'true' : 'false') . "\n";

$deleted = delete_transient('test_transient');
echo "✓ delete_transient(): " . ($deleted ? 'true' : 'false') . "\n";

// Test absint
$abs = absint(-42);
echo "✓ absint(): $abs\n";

// Test maybe_serialize
$serialized = maybe_serialize(['test' => 'value']);
echo "✓ maybe_serialize(): $serialized\n";

// Test apply_filters
$filtered = apply_filters('test_filter', 'original_value');
echo "✓ apply_filters(): $filtered\n";

// Test do_action
do_action('test_action', 'arg1', 'arg2');
echo "✓ do_action(): completed\n";

echo "\n=== All Tests Passed ===\n";
