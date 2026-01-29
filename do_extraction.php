<?php
/**
 * Extract helper methods from AssetData.php.backup
 */

$backupFile = __DIR__ . '/AssetData.php.backup';
$helpersDir = __DIR__ . '/AssetDataHelpers/';

// Read the entire backup file
$content = file_get_contents($backupFile);

// Helper file mapping
$helperMapping = [
    'AssetValidationHelper.php' => [
        'is_valid_url',
        'is_valid_asset_url',
        'is_external_url',
        'is_url_safe_to_fetch',
        'is_asset_fully_processed',
        'determine_asset_status',
        'determine_asset_type',
        'determine_primary_asset_type',
        'determine_asset_type_from_extension',
    ],
    'AssetStatisticsHelper.php' => [
        'get_asset_statistics',
        'get_processed_assets_count_by_pattern',
        'has_processed_clarity_assets',
        'get_progress',
    ],
    'AssetCacheHelper.php' => [
        'invalidate_asset_cache',
        'invalidate_paginated_cache',
        'clear_asset_cache',
        'clear_all_asset_caches',
    ],
    'AssetDatabaseHelper.php' => [
        'get_mapping_table_name',
        'get_tasks_table_name',
        'get_order_table_name',
        'get_task_table_name',
        'get_asset_exists',
        'get_asset_status_by_id',
        'get_mapping_entry_by_id',
    ],
    'AssetMemoryHelper.php' => [
        'get_memory_usage',
        'get_peak_memory_usage',
        'get_memory_threshold',
        'get_dynamic_memory_threshold',
    ],
    'AssetMetadataHelper.php' => [
        'get_asset_local_directory',
        'get_asset_image_dimensions',
        'get_asset_file_size',
        'get_asset_checksum',
        'get_asset_version',
        'get_asset_embed_code',
        'get_asset_exif_data',
        'get_asset_aspect_ratio',
        'get_asset_metadata',
        'get_asset_last_modified',
        'get_asset_sanitized_filename',
        'get_asset_download_link',
    ],
    'AssetOrderHelper.php' => [
        'get_order_settings',
        'get_current_order_for_post',
    ],
    'AssetQueryHelper.php' => [
        'get_paginated_assets',
        'get_asset_ids_by_handles',
        'get_asset_status',
        'batch_get_asset_statuses',
        'get_custom_registry_url',
        'get_local_url_if_processed',
        'get_asset_data',
        'get_all_processed_assets_for_reversal',
    ],
    'AssetTaskHelper.php' => [
        'is_task_enqueued',
        'get_task_by_id',
    ],
    'AssetURLHelper.php' => [
        'normalize_url',
        'extract_urls_from_src',
        'extract_url_from_parentheses',
        'get_local_url_from_path',
        'get_original_url_from_handle',
        'find_all_urls_in_css',
        'find_script_handle_by_url',
        'find_media_handle',
        'find_style_handle_by_url',
        'get_font_urls',
        'get_image_urls',
    ],
    'AssetUtilityHelper.php' => [
        'is_rate_limited',
        'get_validated_cache_expiration',
        'get_pending_option_key',
        'get_transient_prefix',
        'get_clean_dom_content',
        'fetch_asset_content',
        'get_dynamic_asset_urls',
        'get_uploaded_media_handles',
        'get_max_visited_urls',
        'get_svg_allowed_html',
        'get_external_css_from_enqueued_styles',
    ],
    'AssetDataRegistryHelper.php' => [
        'get_data_registry',
    ],
    'AssetIntegrationHelper.php' => [
        'get_data',
    ],
];

// Function to extract a method from source code
function extractMethod($content, $methodName) {
    // Find the method start
    $pattern = '/(public|private|protected)\s+(static\s+)?function\s+' . preg_quote($methodName, '/') . '\s*\(/';
    preg_match($pattern, $content, $matches, PREG_OFFSET_CAPTURE);

    if (empty($matches)) {
        echo "Warning: Method $methodName not found\n";
        return null;
    }

    $offset = $matches[0][1];
    $remaining = substr($content, $offset);

    // Extract the full method including body
    $braceCount = 0;
    $inMethod = false;
    $methodEnd = 0;
    $parenCount = 0;
    $inSignature = true;

    for ($i = 0; $i < strlen($remaining); $i++) {
        $char = $remaining[$i];

        if ($inSignature) {
            if ($char === '(') {
                $parenCount++;
            } elseif ($char === ')') {
                $parenCount--;
            } elseif ($char === '{' && $parenCount === 0) {
                $inSignature = false;
                $inMethod = true;
                $braceCount = 1;
                continue;
            }
        } else {
            if ($char === '{') {
                $braceCount++;
            } elseif ($char === '}') {
                $braceCount--;
                if ($braceCount === 0) {
                    $methodEnd = $i + 1;
                    break;
                }
            }
        }
    }

    if ($methodEnd === 0) {
        echo "Warning: Could not find end of method $methodName\n";
        return null;
    }

    $methodCode = substr($remaining, 0, $methodEnd);

    // Find leading whitespace and comments before the method
    $beforeMethod = substr($content, 0, $offset);
    $lastNewline = strrpos($beforeMethod, "\n");
    $beforeMethod = substr($beforeMethod, $lastNewline + 1);

    // Check if there's a docblock
    if (preg_match('/\/\*\*[\s\S]*?\*\//', $beforeMethod, $docblockMatches)) {
        // Include the docblock
        $docblockStart = strpos($beforeMethod, $docblockMatches[0]);
        $methodCode = substr($beforeMethod, $docblockStart) . "\n" . $methodCode;
    }

    return $methodCode;
}

// Process each helper file
foreach ($helperMapping as $filename => $methods) {
    echo "Processing $filename...\n";

    $helperContent = "<?php\n\n";
    $helperContent .= "declare(strict_types=1);\n\n";
    $helperContent .= "namespace LHA\\AssetDataHelpers;\n\n";

    foreach ($methods as $method) {
        echo "  Extracting $method...\n";
        $methodCode = extractMethod($content, $method);

        if ($methodCode) {
            $helperContent .= $methodCode . "\n\n";
        }
    }

    // Write to file
    $outputPath = $helpersDir . $filename;
    file_put_contents($outputPath, $helperContent);
    echo "  Created $filename\n\n";
}

echo "Extraction complete!\n";
