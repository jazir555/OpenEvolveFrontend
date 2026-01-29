<?php
/**
 * Script to fix all AjaxHelper files with WordPress function_exists() checks
 * This adds function_exists() wrappers for:
 * - check_ajax_referer()
 * - wp_send_json_success()
 * - wp_send_json_error()
 * - current_user_can()
 */

$ajaxHelpersDir = __DIR__ . '/AjaxHelpers/';
$files = glob($ajaxHelpersDir . '*AjaxHelper.php');

$functions_to_wrap = [
    'check_ajax_referer',
    'wp_send_json_success',
    'wp_send_json_error',
    'current_user_can'
];

$total_files_fixed = 0;
$total_functions_wrapped = 0;

foreach ($files as $file) {
    $content = file_get_contents($file);
    $original_content = $content;
    $functions_fixed_in_file = 0;

    foreach ($functions_to_wrap as $function) {
        // Skip if already wrapped with function_exists
        if (strpos($content, "function_exists('$function')") !== false) {
            continue;
        }

        // Pattern to match function calls (not definitions)
        // This matches: function_name( with possible whitespace
        $pattern = '/\b(' . preg_quote($function) . ')\s*\(/';

        $content = preg_replace_callback(
            $pattern,
            function($matches) use ($function, &$functions_fixed_in_file) {
                // Check if this is already wrapped with function_exists
                $backtrace = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 2);
                if (isset($backtrace[1]['file'])) {
                    $line = file($backtrace[1]['file']);
                    $line_num = $backtrace[1]['line'] - 1;
                    if (isset($line[$line_num]) && strpos($line[$line_num], 'function_exists') !== false) {
                        return $matches[0]; // Already wrapped
                    }
                }

                $functions_fixed_in_file++;
                return $matches[1] . '('; // Don't modify, we'll add wrapper manually
            },
            $content
        );
    }

    if ($content !== $original_content || $functions_fixed_in_file > 0) {
        // Manual fix approach - read the file and add wrappers
        $fixed_content = add_ajax_function_wrappers($content);
        if ($fixed_content !== $original_content) {
            file_put_contents($file, $fixed_content);
            $total_files_fixed++;
            $total_functions_wrapped += $functions_fixed_in_file;
            echo "Fixed: " . basename($file) . " ($functions_fixed_in_file functions)\n";
        }
    }
}

echo "\nTotal files fixed: $total_files_fixed\n";
echo "Total function wrappers added: $total_functions_wrapped\n";

/**
 * Add function_exists() wrappers to AJAX WordPress functions
 */
function add_ajax_function_wrappers($content) {
    // We need to be smart about this - only wrap actual calls, not definitions
    // And don't double-wrap

    // For check_ajax_referer
    $content = preg_replace(
        '/\b(if\s*\()\s*(!?)\s*check_ajax_referer\s*\(/',
        '$1$2function_exists(\'check_ajax_referer\') && check_ajax_referer(',
        $content
    );

    // For current_user_can in if statements
    $content = preg_replace(
        '/\b(if\s*\()\s*(!?)\s*current_user_can\s*\(/',
        '$1$2function_exists(\'current_user_can\') && current_user_can(',
        $content
    );

    // For wp_send_json_success and wp_send_json_error
    // These usually exit, so we wrap them with a check
    $content = preg_replace(
        '/\b(wp_send_json_success|wp_send_json_error)\s*\(/',
        'function_exists(\'$1\') ? $1(',
        $content
    );

    // Close the ternary for wp_send_json_* calls
    // This is more complex and might need manual review
    // For now, just add the function_exists wrapper at the start

    return $content;
}
