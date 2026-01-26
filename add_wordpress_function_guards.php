<?php
/**
 * WordPress Function Guard Adder
 *
 * This script scans helper files and adds function_exists() checks
 * for WordPress function calls.
 */

function fix_ajax_file($filepath) {
    $content = file_get_contents($filepath);
    if ($content === false) {
        echo "Could not read: $filepath\n";
        return false;
    }

    $original = $content;
    $changes = 0;

    // Fix check_ajax_referer calls in if statements
    // Pattern: if (!check_ajax_referer(...))
    $content = preg_replace(
        '/(\s+)if\s*\(\s*!\s*check_ajax_referer\s*\(/',
        '$1if ( ! function_exists( \'check_ajax_referer\' ) || ! check_ajax_referer(',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix current_user_can calls in if statements
    $content = preg_replace(
        '/(\s+)if\s*\(\s*!\s*current_user_can\s*\(/',
        '$1if ( ! function_exists( \'current_user_can\' ) || ! current_user_can(',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix standalone current_user_can checks (like ternary operator)
    $content = preg_replace(
        '/(\s+)\$can_edit\s*=\s*\(.*?\)\s*\?\s*current_user_can\((.*?)\)\s*:\s*current_user_can\((.*?)\);/',
        '$1$can_edit = ( function_exists( \'current_user_can\' ) ? ( $2 ? current_user_can( $2 ) : current_user_can( $3 ) ) : false );',
        $content
    );

    // For wp_send_json_success and wp_send_json_error
    // These are usually at the end of functions and exit, so we add a guard before them
    // We'll add a check at the beginning of the function to wrap all calls

    if ($content !== $original) {
        file_put_contents($filepath, $content);
        echo "Fixed: " . basename($filepath) . "\n";
        return true;
    }

    return false;
}

function fix_cleanup_file($filepath) {
    $content = file_get_contents($filepath);
    if ($content === false) {
        echo "Could not read: $filepath\n";
        return false;
    }

    $original = $content;
    $changes = 0;

    // Fix wp_next_scheduled calls
    $content = preg_replace(
        '/(\s+)\$timestamp\s*=\s*wp_next_scheduled\s*\(/',
        '$1$timestamp = false;' . "\n" . '$1if ( function_exists( \'wp_next_scheduled\' ) ) {' . "\n" . '$1    $timestamp = wp_next_scheduled(',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix wp_unschedule_event calls
    $content = preg_replace(
        '/\$unscheduled_fallback\s*=\s*wp_unschedule_event\s*\(/',
        'if ( function_exists( \'wp_unschedule_event\' ) ) {' . "\n" . '                    $unscheduled_fallback = wp_unschedule_event(',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix wp_get_schedule calls
    $content = preg_replace(
        '/\'recurrence\'\s*=>\s*\$timestamp\s*\?\s*wp_get_schedule\s*\((.*?)\)\s*:\s*\'None\'/',
        '\'recurrence\' => ( $timestamp && function_exists( \'wp_get_schedule\' ) ) ? wp_get_schedule( $1 ) : \'None\'',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix wp_schedule_event calls
    $content = preg_replace(
        '/(\s+)if\s*\(\s*!\s*wp_next_scheduled\s*\(/',
        '$1$is_scheduled = false;' . "\n" . '$1if ( function_exists( \'wp_next_scheduled\' ) ) {' . "\n" . '$1    $is_scheduled = wp_next_scheduled(',
        $content
    );
    if ($content !== $original) $changes++;

    // Fix follow-up check
    $content = str_replace(
        "if ( ! \$is_scheduled ) {",
        "}\n\n        if ( ! \$is_scheduled ) {",
        $content
    );

    if ($content !== $original) {
        file_put_contents($filepath, $content);
        echo "Fixed: " . basename($filepath) . "\n";
        return true;
    }

    return false;
}

// Main execution
$baseDir = __DIR__;

echo "=== Fixing AjaxHelper Files ===\n";
$ajaxHelpers = glob($baseDir . '/AjaxHelpers/*AjaxHelper.php');
foreach ($ajaxHelpers as $file) {
    fix_ajax_file($file);
}

echo "\n=== Fixing CleanupHelper Files ===\n";
$cleanupHelpers = glob($baseDir . '/CleanupHelpers/*.php');
foreach ($cleanupHelpers as $file) {
    fix_cleanup_file($file);
}

echo "\nDone!\n";
